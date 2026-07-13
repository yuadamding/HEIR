"""Finite, auditable training loop for generic or personalized HEIR stages."""

import hashlib
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ..config import LossWeightConfig, OptimizationConfig
from ..expression import EXPRESSION_MAX, EXPRESSION_SPACE_ID
from ..losses import (
    HEIRCompositeLoss,
    HEIRLossConfig,
    UnbalancedSinkhornResult,
    unbalanced_sinkhorn,
)
from ..models.heir import HEIRModel, HEIROutput
from ..models.rna import RNAVAE
from ..utils import resolve_device, set_seed
from .batch import HEIRTrainingBatch
from .contracts import MolecularEStepArtifact
from .stages import TrainingStage


@dataclass(frozen=True)
class HEIRTrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: Tuple[Dict[str, float], ...]


def aggregate_to_spots(
    expression: Tensor,
    assignment: Tensor,
    eps: float = 1e-8,
    expression_space_id: str = EXPRESSION_SPACE_ID,
    cell_rna_mass: Optional[Tensor] = None,
) -> Tensor:
    """Aggregate cell log1p-CPM in linear space and return spot log1p-CPM.

    ``cell_rna_mass`` may combine known-state probability, fractional cell
    overlap, and an externally calibrated type/state RNA-mass estimate. Equal
    cell mass remains the backward-compatible default.
    """

    if expression.ndim != 2 or assignment.ndim != 2 or assignment.shape[1] != expression.shape[0]:
        raise ValueError("expression or spot assignment has the wrong shape")
    if bool((assignment < 0).any()) or not torch.isfinite(assignment).all():
        raise ValueError("spot assignment must be finite and non-negative")
    if expression_space_id != EXPRESSION_SPACE_ID:
        raise ValueError("unsupported expression space for spot aggregation")
    if not torch.isfinite(expression).all() or bool((expression < 0).any()):
        raise ValueError("log1p-CPM expression must be finite and non-negative")
    weighted_assignment = assignment
    if cell_rna_mass is not None:
        if cell_rna_mass.shape != (expression.shape[0],):
            raise ValueError("cell_rna_mass must contain one value per cell")
        if (
            not torch.is_floating_point(cell_rna_mass)
            or not torch.isfinite(cell_rna_mass).all()
            or bool((cell_rna_mass < 0).any())
        ):
            raise ValueError("cell_rna_mass must be finite and non-negative")
        weighted_assignment = assignment * cell_rna_mass.to(assignment.dtype).unsqueeze(0)
    mass = weighted_assignment.sum(dim=1, keepdim=True)
    linear = torch.expm1(expression.clamp_max(EXPRESSION_MAX))
    spot_linear = weighted_assignment.matmul(linear) / mass.clamp_min(eps)
    return torch.log1p(spot_linear)


class HEIRTrainer:
    """Train one coherent graph bag at a time to preserve neighborhoods."""

    @staticmethod
    def _source_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _bernoulli_uot_costs(
        prototype_cost: Tensor,
        unknown_probability: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply the two-outcome unknown gate to real and dustbin UOT costs."""

        if prototype_cost.ndim != 2:
            raise ValueError("prototype_cost must have shape (cells, prototypes)")
        if unknown_probability.shape != (prototype_cost.shape[0],):
            raise ValueError("unknown_probability must have one value per cell")
        probability = unknown_probability.to(dtype=torch.float32)
        epsilon = torch.finfo(torch.float32).eps
        probability = probability.clamp(min=epsilon, max=1.0 - epsilon)
        real_gate = -torch.log1p(-probability)
        dustbin_gate = -torch.log(probability)
        return prototype_cost + real_gate.unsqueeze(-1), dustbin_gate

    def _anchor_constraints(self, batch: HEIRTrainingBatch) -> Optional[Tensor]:
        """Translate accepted fine/parent anchors into prototype routing masks."""

        # Pseudo-anchors supervise the type heads through confidence-weighted
        # losses, but must not exclude contradictory molecular prototypes.
        # Hard routing is reserved for independently reviewed labels and is an
        # explicit trainer option rather than the refinement default.
        if not self.hard_anchor_routing:
            return None

        cells = len(batch.morphology)
        types = self.model.config.num_cell_types
        constraints = batch.morphology.new_ones((cells, types))
        constrained = torch.zeros(cells, dtype=torch.bool, device=batch.morphology.device)
        parent_labels = batch.parent_anchor_labels
        if parent_labels is not None:
            mapping = self.model.config.fine_to_parent
            if mapping is None:
                raise ValueError("parent anchors require a hierarchical HEIR model")
            valid = parent_labels >= 0
            if batch.parent_anchor_weights is not None:
                valid = valid & (batch.parent_anchor_weights > 0)
            if bool(valid.any()):
                fine_to_parent = torch.tensor(mapping, device=batch.morphology.device)
                constraints[valid] = (
                    fine_to_parent.unsqueeze(0) == parent_labels[valid].unsqueeze(1)
                ).to(constraints.dtype)
                constrained = constrained | valid
        fine_labels = batch.anchor_labels
        if fine_labels is not None:
            valid = fine_labels >= 0
            if batch.anchor_weights is not None:
                valid = valid & (batch.anchor_weights > 0)
            if bool(valid.any()):
                constraints[valid] = F.one_hot(
                    fine_labels[valid],
                    num_classes=types,
                ).to(constraints.dtype)
                constrained = constrained | valid
        return constraints if bool(constrained.any()) else None

    def transport_responsibilities(
        self,
        batch: HEIRTrainingBatch,
        output: Optional[HEIROutput] = None,
    ) -> Tuple[Tensor, UnbalancedSinkhornResult]:
        """Return detached known-state subprobabilities from the UOT plan.

        Each row is normalized by its complete transported mass, including the
        dustbin. Its sum therefore preserves the real-state responsibility and
        can be below one for an unknown cell.
        """

        resolved = self._forward_output(batch) if output is None else output
        real_cost, unknown_cost = self._bernoulli_uot_costs(
            resolved.prototype_cost,
            resolved.unknown_probability,
        )
        result = unbalanced_sinkhorn(
            real_cost,
            batch.cell_weights,
            batch.prototype_weights,
            target_mask=batch.prototype_mask,
            pair_mask=resolved.prototype_mask,
            epsilon=self.uot_epsilon,
            marginal_relaxation=self.uot_relaxation,
            iterations=self.uot_iterations,
            convergence_tolerance=self.uot_convergence_tolerance,
            unknown_mass=self._estimated_uot_unknown_mass(batch, resolved),
            unknown_cost=unknown_cost,
        )
        if (
            self.uot_convergence_tolerance is not None
            and result.converged is not None
            and not bool(result.converged.all())
        ):
            raise FloatingPointError("molecular E-step UOT did not converge")
        real_plan = result.plan[..., : real_cost.shape[-1]]
        row_mass = result.plan.sum(dim=-1, keepdim=True)
        responsibilities = torch.where(
            row_mass > torch.finfo(real_plan.dtype).eps,
            real_plan / row_mass.clamp_min(torch.finfo(real_plan.dtype).eps),
            torch.zeros_like(real_plan),
        )
        return responsibilities.detach(), result

    def _estimated_uot_unknown_mass(
        self,
        batch: HEIRTrainingBatch,
        output: HEIROutput,
    ) -> Tensor:
        """Resolve dustbin mass without defaulting to a self-reinforcing estimate.

        The primary mode is fixed unless independent ``unknown_targets`` are
        available.  Model-estimated mass is retained only as an explicit
        sensitivity mode because the same unknown head also changes real and
        dustbin transport costs.
        """

        fixed = output.unknown_probability.new_tensor(self.uot_unknown_mass)
        if self.uot_unknown_mass_mode == "fixed":
            return fixed
        if batch.unknown_targets is None and self.uot_unknown_mass_mode == "targets_or_fixed":
            return fixed

        observations = (
            output.unknown_probability.detach()
            if batch.unknown_targets is None
            else batch.unknown_targets.detach().to(output.unknown_probability.dtype)
        )
        weights = (
            torch.ones_like(observations)
            if batch.cell_weights is None
            else batch.cell_weights.to(observations.dtype)
        )
        observed_mass = (observations * weights).sum()
        effective_cells = weights.sum()
        numerator = self.uot_unknown_prior_strength * self.uot_unknown_mass + observed_mass
        denominator = self.uot_unknown_prior_strength + effective_cells
        estimate = numerator / denominator.clamp_min(torch.finfo(observations.dtype).eps)
        return estimate.clamp(
            min=0.0,
            max=1.0 - torch.finfo(observations.dtype).eps,
        )

    @staticmethod
    def _uot_known_cell_weights(base_cell_weights: Tensor, responsibilities: Tensor) -> Tensor:
        """Weight biological objectives by detached transported known-state mass."""

        if base_cell_weights.ndim != 1 or responsibilities.ndim != 2:
            raise ValueError("cell weights and molecular responsibilities have wrong dimensions")
        if responsibilities.shape[0] != len(base_cell_weights):
            raise ValueError("molecular responsibilities must align to cell weights")
        known_mass = responsibilities.detach().sum(dim=1).clamp(min=0.0, max=1.0)
        return base_cell_weights * known_mass.to(base_cell_weights.dtype)

    def __init__(
        self,
        model: HEIRModel,
        stage: TrainingStage,
        optimization: OptimizationConfig,
        loss_weights: LossWeightConfig,
        rna_encoder: Optional[RNAVAE] = None,
        uot_epsilon: float = 0.1,
        uot_relaxation: float = 1.0,
        uot_iterations: int = 160,
        uot_convergence_tolerance: Optional[float] = 1.0e-5,
        uot_unknown_mass: float = 0.05,
        uot_unknown_prior_strength: float = 2.0,
        uot_unknown_mass_mode: str = "targets_or_fixed",
        seed: int = 17,
        device: str = "auto",
        allow_split_overlap: bool = False,
        molecular_e_step_mode: str = "strict_artifact",
        hard_anchor_routing: bool = False,
    ) -> None:
        if stage not in {
            TrainingStage.GENERIC_SPATIAL_PRETRAINING,
            TrainingStage.PERSONALIZED,
            TrainingStage.REFINEMENT,
        }:
            raise ValueError("HEIRTrainer supports pretraining, personalized, or refinement stages")
        optimization.validate()
        loss_weights.validate()
        if not 0.0 <= uot_unknown_mass < 1.0:
            raise ValueError("uot_unknown_mass must lie in [0, 1)")
        if uot_iterations <= 0:
            raise ValueError("uot_iterations must be positive")
        if uot_convergence_tolerance is not None and (
            not math.isfinite(uot_convergence_tolerance) or uot_convergence_tolerance <= 0
        ):
            raise ValueError("uot_convergence_tolerance must be finite and positive")
        if not math.isfinite(uot_unknown_prior_strength) or uot_unknown_prior_strength <= 0:
            raise ValueError("uot_unknown_prior_strength must be finite and positive")
        if uot_unknown_mass_mode not in {"fixed", "targets_or_fixed", "model_estimate"}:
            raise ValueError(
                "uot_unknown_mass_mode must be fixed, targets_or_fixed, or model_estimate"
            )
        if molecular_e_step_mode not in {
            "strict_artifact",
            "live_student_negative_control",
        }:
            raise ValueError(
                "molecular_e_step_mode must be strict_artifact or live_student_negative_control"
            )
        self.model = model
        self.stage = stage
        self.optimization = optimization
        self.weights = loss_weights
        self.rna_encoder = rna_encoder
        self.uot_epsilon = uot_epsilon
        self.uot_relaxation = uot_relaxation
        self.uot_iterations = uot_iterations
        self.uot_convergence_tolerance = uot_convergence_tolerance
        self.uot_unknown_mass = uot_unknown_mass
        self.uot_unknown_prior_strength = uot_unknown_prior_strength
        self.uot_unknown_mass_mode = uot_unknown_mass_mode
        self.criterion = HEIRCompositeLoss(
            HEIRLossConfig.from_experiment_weights(
                loss_weights,
                marker_weight=loss_weights.marker,
                uot_epsilon=uot_epsilon,
                uot_marginal_relaxation=uot_relaxation,
                uot_iterations=uot_iterations,
                uot_convergence_tolerance=uot_convergence_tolerance,
                uot_unknown_mass=uot_unknown_mass,
                pseudobulk_metric="smooth_l1",
                program_metric="smooth_l1",
                marker_metric="smooth_l1",
                pseudobulk_log1p_expression=self.model.config.nonnegative_expression,
            )
        )
        self.seed = seed
        self.device = resolve_device(device)
        self.allow_split_overlap = allow_split_overlap
        self.molecular_e_step_mode = molecular_e_step_mode
        self.hard_anchor_routing = hard_anchor_routing

    def _forward_output(self, batch: HEIRTrainingBatch) -> HEIROutput:
        return self.model(
            batch.morphology,
            batch.edge_index,
            batch.edge_weight,
            prototype_means=batch.prototype_means,
            prototype_variances=batch.prototype_variances,
            prototype_types=batch.prototype_types,
            prototype_weights=batch.prototype_weights,
            prototype_mask=batch.prototype_mask,
            cell_type_constraints=self._anchor_constraints(batch),
        )

    def _validate_frozen_e_step_batch(self, batch: HEIRTrainingBatch) -> None:
        """Revalidate the immutable E-step and its copied batch payload."""

        if batch.molecular_responsibilities is None:
            raise ValueError(
                "strict molecular M-step requires responsibilities for every train/validation bag"
            )
        indices = [
            index for index, role in enumerate(batch.source_roles) if role == "frozen_e_step"
        ]
        if len(indices) != 1:
            raise ValueError(
                "strict molecular M-step requires exactly one hash-bound frozen_e_step source"
            )
        index = indices[0]
        path = Path(batch.source_artifacts[index]).expanduser().resolve()
        if not path.is_file():
            raise ValueError("frozen molecular E-step artifact is unavailable: %s" % path)
        digest = self._source_sha256(path)
        if digest != batch.source_sha256[index]:
            raise ValueError("frozen molecular E-step artifact hash no longer matches the batch")
        artifact = MolecularEStepArtifact.load_npz(path)
        upstream_hashes = {}
        for source, recorded, role in zip(
            artifact.source_artifacts,
            artifact.source_sha256,
            artifact.source_roles,
        ):
            upstream = Path(source).expanduser().resolve()
            if not upstream.is_file():
                raise ValueError("frozen E-step upstream source is unavailable: %s" % upstream)
            actual = self._source_sha256(upstream)
            if actual != recorded:
                raise ValueError("frozen E-step upstream source hash no longer matches")
            upstream_hashes[role] = actual
        artifact.validate_binding(
            nucleus_ids=batch.nucleus_ids,
            prototype_ids=batch.prototype_ids,
            source_sha256_by_role=upstream_hashes,
            target_donor=batch.donor_id,
            feature_space_id=batch.feature_space_id,
            latent_space_id=batch.latent_space_id,
            type_names=batch.type_names,
            morphology=batch.morphology.detach().cpu().numpy(),
            edge_index=batch.edge_index.detach().cpu().numpy(),
            edge_weight=(
                None if batch.edge_weight is None else batch.edge_weight.detach().cpu().numpy()
            ),
            prototype_means=batch.prototype_means.detach().cpu().numpy(),
            prototype_variances=batch.prototype_variances.detach().cpu().numpy(),
            prototype_types=batch.prototype_types.detach().cpu().numpy(),
            prototype_weights=batch.prototype_weights.detach().cpu().numpy(),
            cell_weights=(
                np.ones(len(batch.morphology), dtype=np.float32)
                if batch.cell_weights is None
                else batch.cell_weights.detach().cpu().numpy()
            ),
            artifact_threshold=artifact.artifact_threshold,
        )
        expected_weak_scope = "sha256:%s" % upstream_hashes["rna_reference"]
        if batch.weak_target_scope_id != expected_weak_scope:
            raise ValueError("weak_target_scope_id is not bound to the frozen E-step RNA reference")
        if not math.isclose(
            artifact.fixed_unknown_mass,
            self.uot_unknown_mass,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("frozen E-step unknown mass differs from the trainer contract")
        expected = torch.from_numpy(artifact.resolved_conditional_known_prototype_distribution).to(
            batch.molecular_responsibilities.device
        )
        if not torch.equal(batch.molecular_responsibilities, expected):
            raise ValueError("batch molecular responsibilities differ from the frozen E-step")
        if (
            batch.molecular_raw_real_row_mass is None
            or batch.molecular_raw_dustbin_row_mass is None
        ):
            raise ValueError("strict molecular M-step batch omits raw E-step row masses")
        expected_real = torch.from_numpy(artifact.resolved_raw_real_row_mass).to(
            batch.molecular_raw_real_row_mass.device
        )
        expected_dustbin = torch.from_numpy(artifact.resolved_raw_dustbin_row_mass).to(
            batch.molecular_raw_dustbin_row_mass.device
        )
        if not torch.equal(batch.molecular_raw_real_row_mass, expected_real):
            raise ValueError("batch raw real row mass differs from the frozen E-step")
        if not torch.equal(batch.molecular_raw_dustbin_row_mass, expected_dustbin):
            raise ValueError("batch raw dustbin row mass differs from the frozen E-step")
        if not set(artifact.teacher_training_donors).issubset(set(batch.molecular_training_donors)):
            raise ValueError("batch omits frozen E-step teacher training-donor provenance")

    @staticmethod
    def _validate_weak_target_split(
        training_batches: Sequence[HEIRTrainingBatch],
        validation_batches: Sequence[HEIRTrainingBatch],
    ) -> None:
        """Prevent complete-specimen RNA targets from selecting their own model."""

        all_batches = tuple(training_batches) + tuple(validation_batches)
        if any(batch.weak_target_scope_id == "unspecified" for batch in all_batches):
            raise ValueError("strict molecular training requires weak_target_scope_id")
        training_scopes = {batch.weak_target_scope_id for batch in training_batches}
        validation_scopes = {batch.weak_target_scope_id for batch in validation_batches}
        overlap = sorted(training_scopes & validation_scopes)
        if overlap:
            raise ValueError(
                "train/validation reuse complete-specimen molecular targets: %s"
                % ", ".join(overlap[:3])
            )

    @staticmethod
    def _molecular_posterior_loss(
        output: HEIROutput,
        batch: HEIRTrainingBatch,
        responsibilities: Tensor,
        cell_weights: Tensor,
        raw_real_row_mass: Optional[Tensor] = None,
        raw_dustbin_row_mass: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Optimize the complete-data objective against a frozen molecular E-step.

        The known prototype sub-plan supervises routing, type, and latent state.
        The dustbin fraction separately supervises *transport unassignment*; it
        is deliberately not interpreted as a biological unknown/OOD label.
        """

        if responsibilities.shape != output.conditional_prototype_probabilities.shape:
            raise ValueError("molecular responsibilities and prototype routing must align")
        responsibilities = responsibilities.detach()
        row_mass = responsibilities.sum(dim=1)
        if (raw_real_row_mass is None) != (raw_dustbin_row_mass is None):
            raise ValueError("raw real and dustbin row masses must be supplied together")
        if raw_real_row_mass is None:
            # Backward-compatible v3/in-memory path: responsibilities are
            # complete-row known subprobabilities.
            valid = row_mass > 1.0e-8
            conditional = responsibilities / row_mass.unsqueeze(1).clamp_min(1.0e-8)
            transport_unassigned_target = (1.0 - row_mass).clamp(0.0, 1.0)
            effective = cell_weights * row_mass
            unassignment_weights = cell_weights
        else:
            assert raw_dustbin_row_mass is not None
            if (
                raw_real_row_mass.shape != row_mass.shape
                or raw_dustbin_row_mass.shape != row_mass.shape
            ):
                raise ValueError("raw molecular row masses must align to cells")
            raw_real = raw_real_row_mass.detach().to(dtype=cell_weights.dtype)
            raw_dustbin = raw_dustbin_row_mass.detach().to(dtype=cell_weights.dtype)
            if bool((raw_real < 0).any()) or bool((raw_dustbin < 0).any()):
                raise ValueError("raw molecular row masses must be non-negative")
            valid = raw_real > 1.0e-8
            conditional = responsibilities
            complete_mass = raw_real + raw_dustbin
            transport_unassigned_target = torch.where(
                complete_mass > 0,
                raw_dustbin / complete_mass.clamp_min(1.0e-8),
                torch.zeros_like(complete_mass),
            ).clamp(0.0, 1.0)
            # Raw UOT mass already incorporates the E-step source mass. Do not
            # multiply by the input cell weights a second time.
            effective = raw_real
            unassignment_weights = complete_mass
        transport_unassigned_per_cell = F.binary_cross_entropy(
            output.transport_unassigned_probability.clamp(1.0e-8, 1.0 - 1.0e-8),
            transport_unassigned_target,
            reduction="none",
        )
        cell_mass = unassignment_weights.sum().clamp_min(1.0e-8)
        transport_unassigned = (
            transport_unassigned_per_cell * unassignment_weights
        ).sum() / cell_mass
        mass = effective.sum().clamp_min(1.0e-8)
        if not bool(valid.any()):
            zero = output.latent_mu.sum() * 0.0
            return transport_unassigned, {
                "molecular_posterior/routing": zero,
                "molecular_posterior/type": zero,
                "molecular_posterior/latent": zero,
                "molecular_posterior/transport_unassigned": transport_unassigned,
            }

        routing_per_cell = -(
            conditional * output.conditional_prototype_probabilities.clamp_min(1.0e-8).log()
        ).sum(dim=1)
        routing = (routing_per_cell * effective).sum() / mass

        type_targets = responsibilities.new_zeros(
            (len(responsibilities), output.type_probabilities.shape[1])
        )
        type_targets = type_targets.index_add(
            1,
            batch.prototype_types,
            conditional,
        )
        type_targets = type_targets / type_targets.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        type_per_cell = -(type_targets * output.type_probabilities.clamp_min(1.0e-8).log()).sum(
            dim=1
        )
        type_loss = (type_per_cell * effective).sum() / mass

        total_variance = output.residual_logvar.exp().unsqueeze(1) + (
            batch.prototype_variances.unsqueeze(0)
        )
        latent_nll = 0.5 * (
            (output.latent_mu.unsqueeze(1) - batch.prototype_means.unsqueeze(0)).square()
            / total_variance
            + total_variance.log()
        ).mean(dim=2)
        latent_per_cell = (conditional * latent_nll).sum(dim=1)
        latent_loss = (latent_per_cell * effective).sum() / mass
        total = routing + type_loss + latent_loss + transport_unassigned
        return total, {
            "molecular_posterior/routing": routing,
            "molecular_posterior/type": type_loss,
            "molecular_posterior/latent": latent_loss,
            "molecular_posterior/transport_unassigned": transport_unassigned,
        }

    @staticmethod
    def _type_responsibilities(
        responsibilities: Tensor,
        prototype_types: Tensor,
        num_types: int,
    ) -> Tensor:
        """Aggregate detached prototype responsibilities onto the type simplex."""

        if responsibilities.ndim != 2:
            raise ValueError("molecular responsibilities must be a matrix")
        if prototype_types.shape != (responsibilities.shape[1],):
            raise ValueError("prototype types must align to molecular responsibilities")
        if prototype_types.dtype != torch.long or bool((prototype_types < 0).any()):
            raise ValueError("molecular responsibilities require typed prototypes")
        if prototype_types.numel() and int(prototype_types.max()) >= num_types:
            raise ValueError("prototype types exceed the model ontology")
        result = responsibilities.new_zeros((responsibilities.shape[0], num_types))
        result = result.index_add(1, prototype_types, responsibilities.detach())
        return result / result.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

    def _output_loss(
        self,
        output: HEIROutput,
        batch: HEIRTrainingBatch,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        cycle_latent = None
        if self.rna_encoder is not None:
            cycle_latent, _ = self.rna_encoder.encode(output.expression)
        base_cell_weights = (
            output.unknown_probability.new_ones(len(output.unknown_probability))
            if batch.cell_weights is None
            else batch.cell_weights
        )
        precomputed_uot: Optional[UnbalancedSinkhornResult] = None
        if batch.molecular_responsibilities is None:
            if (
                self.stage in {TrainingStage.PERSONALIZED, TrainingStage.REFINEMENT}
                and self.molecular_e_step_mode != "live_student_negative_control"
            ):
                raise ValueError(
                    "strict molecular M-step requires a frozen E-step artifact; "
                    "live student transport is available only as an explicit negative control"
                )
            uot_cost, uot_unknown_cost = self._bernoulli_uot_costs(
                output.prototype_cost,
                output.unknown_probability,
            )
            estimated_unknown_mass: Optional[Tensor] = self._estimated_uot_unknown_mass(
                batch, output
            )
            responsibilities, precomputed_uot = self.transport_responsibilities(batch, output)
        else:
            # A strict M-step consumes the detached artifact and does not
            # evaluate a second live-student UOT objective.  This is the actual
            # E/M boundary: neither the current type/latent output nor the live
            # unknown head can modify the target being optimized.
            responsibilities = batch.molecular_responsibilities.detach()
            uot_cost = None
            uot_unknown_cost = None
            estimated_unknown_mass = None
        if batch.molecular_raw_real_row_mass is None:
            biological_cell_weights = self._uot_known_cell_weights(
                base_cell_weights,
                responsibilities,
            )
        else:
            biological_cell_weights = batch.molecular_raw_real_row_mass.detach().to(
                dtype=base_cell_weights.dtype
            )
        molecular_type_responsibilities = self._type_responsibilities(
            responsibilities,
            batch.prototype_types,
            output.type_probabilities.shape[1],
        )
        total, terms = self.criterion(
            output,
            cell_weights=batch.cell_weights,
            biological_cell_weights=biological_cell_weights,
            molecular_type_responsibilities=molecular_type_responsibilities,
            target_composition=batch.target_composition,
            uot_cost=uot_cost,
            uot_source_mass=batch.cell_weights,
            uot_target_mass=batch.prototype_weights,
            uot_target_mask=batch.prototype_mask,
            uot_pair_mask=output.prototype_mask,
            uot_unknown_cost=uot_unknown_cost,
            uot_unknown_mass=estimated_unknown_mass,
            precomputed_uot=precomputed_uot,
            compute_uot=batch.molecular_responsibilities is None,
            target_pseudobulk=batch.target_pseudobulk,
            program_matrix=batch.program_matrix,
            target_program_scores=batch.target_program_scores,
            marker_centroids=batch.marker_centroids,
            marker_mask=batch.marker_mask,
            residual_precision=batch.prototype_variances.reciprocal(),
            residual_assignments=responsibilities,
            cycle_latent=cycle_latent,
            edge_index=(None if self.model.config.graph_mode == "off" else batch.edge_index),
            edge_weight=(None if self.model.config.graph_mode == "off" else batch.edge_weight),
            anchor_labels=batch.anchor_labels,
            anchor_weights=batch.anchor_weights,
            parent_anchor_labels=getattr(batch, "parent_anchor_labels", None),
            parent_anchor_weights=getattr(batch, "parent_anchor_weights", None),
            fine_to_parent=self.model.config.fine_to_parent,
            unknown_targets=batch.unknown_targets,
            scgpt_type_prototypes=batch.scgpt_type_prototypes,
            scgpt_type_variances=batch.scgpt_type_variances,
        )
        posterior_arguments = (
            output,
            batch,
            responsibilities,
            # The posterior applies transported mass itself. Passing
            # biological_cell_weights here would square that mass.
            base_cell_weights,
        )
        if batch.molecular_raw_real_row_mass is None:
            posterior, posterior_terms = self._molecular_posterior_loss(*posterior_arguments)
        else:
            posterior, posterior_terms = self._molecular_posterior_loss(
                *posterior_arguments,
                raw_real_row_mass=batch.molecular_raw_real_row_mass,
                raw_dustbin_row_mass=batch.molecular_raw_dustbin_row_mass,
            )
        terms.update(posterior_terms)
        if all(
            name in posterior_terms
            for name in (
                "molecular_posterior/routing",
                "molecular_posterior/type",
                "molecular_posterior/latent",
                "molecular_posterior/transport_unassigned",
            )
        ):
            terms["molecular_posterior/raw_total"] = posterior
            component_weights = {
                "routing": self.weights.molecular_routing,
                "type": self.weights.molecular_type,
                "latent": self.weights.molecular_latent,
                "transport_unassigned": self.weights.transport_unassigned,
            }
            weighted_components = []
            for name, weight in component_weights.items():
                weighted = weight * posterior_terms["molecular_posterior/%s" % name]
                terms["weighted/molecular_posterior/%s" % name] = weighted
                weighted_components.append(weighted)
            posterior = sum(weighted_components[1:], weighted_components[0])
        terms["molecular_posterior"] = posterior
        terms["weighted/molecular_posterior"] = self.weights.molecular_posterior * posterior
        total = total + terms["weighted/molecular_posterior"]
        domain = output.expression.sum() * 0.0
        if batch.domain_labels is not None:
            if output.domain_logits is None:
                raise ValueError("domain_labels require HEIRConfig.num_domains >= 2")
            valid_domain = batch.domain_labels >= 0
            if bool(valid_domain.any()):
                domain = F.cross_entropy(
                    output.domain_logits[valid_domain],
                    batch.domain_labels[valid_domain],
                )
        terms["domain"] = domain
        terms["weighted/domain"] = self.weights.domain * domain
        total = total + terms["weighted/domain"]
        if batch.target_spatial_expression is not None and batch.spot_assignment is not None:
            spot_prediction = aggregate_to_spots(
                output.expression,
                batch.spot_assignment,
                expression_space_id=batch.expression_space_id,
                cell_rna_mass=biological_cell_weights,
            )
            finite = torch.isfinite(batch.target_spatial_expression)
            terms["spatial"] = (
                F.huber_loss(
                    spot_prediction[finite],
                    batch.target_spatial_expression[finite],
                )
                if bool(finite.any())
                else spot_prediction.sum() * 0.0
            )
        else:
            terms["spatial"] = output.expression.sum() * 0.0
        total = total + terms["spatial"]
        # Domain CE trains the adversarial classifier, but lower validation CE
        # means domains are *more* predictable.  It must not select the encoder
        # checkpoint used for early stopping.
        terms["selection_total"] = total - terms["weighted/domain"]
        terms["total"] = total
        return total, terms

    def _forward_loss(self, batch: HEIRTrainingBatch) -> Tuple[Tensor, Dict[str, Tensor]]:
        return self._output_loss(self._forward_output(batch), batch)

    @staticmethod
    def _concatenate_outputs(outputs: Sequence[HEIROutput]) -> HEIROutput:
        values: Dict[str, Any] = {}
        for item in fields(HEIROutput):
            tensors = [getattr(output, item.name) for output in outputs]
            if all(value is None for value in tensors):
                values[item.name] = None
            elif any(value is None for value in tensors):
                raise ValueError("model output %s is inconsistent across sample bags" % item.name)
            else:
                values[item.name] = torch.cat(tensors, dim=0)
        return HEIROutput(**values)

    @staticmethod
    def _merge_sample_batches(batches: Sequence[HEIRTrainingBatch]) -> HEIRTrainingBatch:
        """Merge graph patches for one sample before sample-level objectives."""

        if len(batches) == 1:
            return batches[0]
        first = batches[0]
        if any(not batch.nucleus_ids for batch in batches):
            raise ValueError("multi-patch samples require nucleus_ids for overlap accounting")
        for batch in batches[1:]:
            if (
                batch.sample_id != first.sample_id
                or batch.donor_id != first.donor_id
                or batch.block_id != first.block_id
                or batch.analysis_role != first.analysis_role
                or batch.latent_space_id != first.latent_space_id
                or batch.feature_space_id != first.feature_space_id
                or batch.expression_space_id != first.expression_space_id
                or batch.scgpt_space_id != first.scgpt_space_id
                or batch.weak_target_scope_id != first.weak_target_scope_id
                or batch.weak_target_granularity != first.weak_target_granularity
                or batch.type_names != first.type_names
                or batch.gene_names != first.gene_names
                or batch.prototype_ids != first.prototype_ids
                or batch.molecular_training_donors != first.molecular_training_donors
            ):
                raise ValueError("sample bags have inconsistent provenance or ontology")
            for name in (
                "prototype_means",
                "prototype_variances",
                "prototype_types",
                "prototype_weights",
                "target_composition",
                "target_pseudobulk",
                "prototype_mask",
                "marker_centroids",
                "marker_mask",
                "program_matrix",
                "target_program_scores",
                "scgpt_type_prototypes",
                "scgpt_type_variances",
            ):
                if name == "target_pseudobulk" and first.spot_assignment is not None:
                    continue
                left = getattr(first, name)
                right = getattr(batch, name)
                if (left is None) != (right is None):
                    raise ValueError("sample bags disagree on %s" % name)
                if left is not None and not torch.equal(left, right):
                    raise ValueError("sample bags must share identical %s" % name)

        offsets = []
        offset = 0
        for batch in batches:
            offsets.append(batch.edge_index + offset)
            offset += len(batch.morphology)
        edge_index = torch.cat(offsets, dim=1)
        edge_weights = [
            batch.edge_weight
            if batch.edge_weight is not None
            else batch.morphology.new_ones(batch.edge_index.shape[1])
            for batch in batches
        ]

        def concatenate_optional(
            name: str,
            fill_value: float,
            dtype: Optional[torch.dtype] = None,
        ) -> Optional[Tensor]:
            raw = [getattr(batch, name) for batch in batches]
            if all(value is None for value in raw):
                return None
            result = []
            for batch, value in zip(batches, raw):
                if value is None:
                    result.append(
                        torch.full(
                            (len(batch.morphology),),
                            fill_value,
                            dtype=dtype or batch.morphology.dtype,
                            device=batch.morphology.device,
                        )
                    )
                else:
                    result.append(value)
            return torch.cat(result)

        unknown_values = [batch.unknown_targets for batch in batches]
        if any(value is None for value in unknown_values) and not all(
            value is None for value in unknown_values
        ):
            raise ValueError("unknown calibration targets must cover every sample bag")
        unknown_targets = (
            None
            if all(value is None for value in unknown_values)
            else torch.cat([value for value in unknown_values if value is not None])
        )

        responsibility_values = [batch.molecular_responsibilities for batch in batches]
        if any(value is None for value in responsibility_values) and not all(
            value is None for value in responsibility_values
        ):
            raise ValueError("molecular responsibilities must cover every sample bag")
        molecular_responsibilities = (
            None
            if all(value is None for value in responsibility_values)
            else torch.cat([value for value in responsibility_values if value is not None], dim=0)
        )
        molecular_raw_real_row_mass = concatenate_optional("molecular_raw_real_row_mass", 0.0)
        molecular_raw_dustbin_row_mass = concatenate_optional("molecular_raw_dustbin_row_mass", 0.0)

        spot_values = [batch.spot_assignment for batch in batches]
        if any(value is None for value in spot_values) and not all(
            value is None for value in spot_values
        ):
            raise ValueError("spatial targets must cover every sample bag")
        spot_assignment = None
        target_spatial_expression = None
        target_pseudobulk = first.target_pseudobulk
        if all(value is not None for value in spot_values):
            ordered_spots: List[str] = []
            targets_by_spot: Dict[str, Tensor] = {}
            for batch in batches:
                assert batch.target_spatial_expression is not None
                for spot_id, target in zip(batch.spot_ids, batch.target_spatial_expression):
                    if spot_id in targets_by_spot:
                        if not torch.allclose(
                            targets_by_spot[spot_id], target, rtol=1.0e-5, atol=1.0e-6
                        ):
                            raise ValueError(
                                "sample patches disagree on spatial target for %s" % spot_id
                            )
                    else:
                        ordered_spots.append(spot_id)
                        targets_by_spot[spot_id] = target
            spot_lookup = {value: index for index, value in enumerate(ordered_spots)}
            rows = len(ordered_spots)
            columns = sum(len(batch.morphology) for batch in batches)
            spot_assignment = first.morphology.new_zeros((rows, columns))
            column_offset = 0
            for batch in batches:
                assert batch.spot_assignment is not None
                for local_row, spot_id in enumerate(batch.spot_ids):
                    spot_assignment[
                        spot_lookup[spot_id],
                        column_offset : column_offset + len(batch.morphology),
                    ] = batch.spot_assignment[local_row]
                column_offset += len(batch.morphology)
            target_spatial_expression = torch.stack(
                [targets_by_spot[value] for value in ordered_spots]
            )
            spot_mass = spot_assignment.sum(dim=1)
            target_pseudobulk = torch.log1p(
                (
                    torch.expm1(target_spatial_expression.clamp_max(EXPRESSION_MAX))
                    * spot_mass.unsqueeze(1)
                ).sum(dim=0)
                / spot_mass.sum().clamp_min(1.0e-8)
            )

        source_triples = {
            (path, digest, role)
            for batch in batches
            for path, digest, role in zip(
                batch.source_artifacts,
                batch.source_sha256,
                batch.source_roles,
            )
        }
        nucleus_ids = tuple(value for batch in batches for value in batch.nucleus_ids)
        if spot_assignment is not None and len(set(nucleus_ids)) != len(nucleus_ids):
            raise ValueError("spatial-pretraining patches must partition nuclei without overlap")
        if len(set(nucleus_ids)) != len(nucleus_ids):
            mass_by_id: Dict[str, float] = {}
            for batch in batches:
                weights = (
                    torch.ones(len(batch.morphology), device=batch.morphology.device)
                    if batch.cell_weights is None
                    else batch.cell_weights
                )
                for nucleus_id, weight in zip(batch.nucleus_ids, weights.detach().cpu().tolist()):
                    mass_by_id[nucleus_id] = mass_by_id.get(nucleus_id, 0.0) + float(weight)
            excessive = sorted(key for key, value in mass_by_id.items() if value > 1.0 + 1.0e-6)
            if excessive:
                raise ValueError(
                    "overlapping patch nuclei need central/margin weights summing to <=1: %s"
                    % ", ".join(excessive[:5])
                )
        return replace(
            first,
            morphology=torch.cat([batch.morphology for batch in batches]),
            edge_index=edge_index,
            edge_weight=torch.cat(edge_weights),
            cell_weights=concatenate_optional("cell_weights", 1.0),
            anchor_labels=concatenate_optional("anchor_labels", -100, dtype=torch.long),
            anchor_weights=concatenate_optional("anchor_weights", 0.0),
            parent_anchor_labels=concatenate_optional(
                "parent_anchor_labels", -100, dtype=torch.long
            ),
            parent_anchor_weights=concatenate_optional("parent_anchor_weights", 0.0),
            unknown_targets=unknown_targets,
            molecular_responsibilities=molecular_responsibilities,
            molecular_raw_real_row_mass=molecular_raw_real_row_mass,
            molecular_raw_dustbin_row_mass=molecular_raw_dustbin_row_mass,
            domain_labels=concatenate_optional("domain_labels", -100, dtype=torch.long),
            segmentation_confidence=concatenate_optional("segmentation_confidence", 1.0),
            ood_mask=concatenate_optional("ood_mask", 0, dtype=torch.bool),
            spot_assignment=spot_assignment,
            target_spatial_expression=target_spatial_expression,
            target_pseudobulk=target_pseudobulk,
            spot_ids=(() if spot_assignment is None else tuple(ordered_spots)),
            bag_id="__sample_aggregate__",
            nucleus_ids=nucleus_ids,
            source_artifacts=tuple(path for path, _, _ in sorted(source_triples)),
            source_sha256=tuple(digest for _, digest, _ in sorted(source_triples)),
            source_roles=tuple(role for _, _, role in sorted(source_triples)),
        )

    def _epoch(
        self,
        batches: Sequence[HEIRTrainingBatch],
        optimizer: Optional[torch.optim.Optimizer],
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        training = optimizer is not None
        self.model.train(training)
        totals: Dict[str, float] = {}
        grouped: Dict[Tuple[str, str], List[HEIRTrainingBatch]] = {}
        for original in batches:
            grouped.setdefault((original.donor_id, original.sample_id), []).append(original)
        for originals in grouped.values():
            sample_cells = sum(len(batch.morphology) for batch in originals)
            if sample_cells > self.optimization.maximum_sample_cells:
                raise ValueError(
                    "sample exceeds optimization.maximum_sample_cells; use a prespecified "
                    "donor-balanced tissue-region sample instead of retaining an entire WSI graph"
                )
            device_batches = []
            for original in originals:
                original.validate(self.stage)
                if original.morphology.shape[0] > self.optimization.bag_size:
                    raise ValueError(
                        "graph bag exceeds optimization.bag_size; create a coherent graph patch"
                    )
                if original.prototype_means.shape[0] > self.optimization.reference_batch_size:
                    raise ValueError("prototype bank exceeds optimization.reference_batch_size")
                device_batches.append(original.to(self.device))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            amp_enabled = bool(self.optimization.mixed_precision and self.device.type == "cuda")
            with (
                torch.set_grad_enabled(training),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
                    enabled=amp_enabled,
                ),
            ):
                outputs = [self._forward_output(batch) for batch in device_batches]
                merged_batch = self._merge_sample_batches(device_batches)
                merged_output = self._concatenate_outputs(outputs)
                loss, terms = self._output_loss(merged_output, merged_batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite HEIR loss for %s" % merged_batch.sample_id)
                if optimizer is not None:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.optimization.gradient_clip_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
            for name, value in terms.items():
                if value.ndim == 0:
                    totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        return {name: value / len(grouped) for name, value in totals.items()}

    def fit(
        self,
        training_batches: Sequence[HEIRTrainingBatch],
        validation_batches: Sequence[HEIRTrainingBatch],
    ) -> HEIRTrainingResult:
        if not training_batches or not validation_batches:
            raise ValueError("training and weak-validation batches cannot be empty")
        if self.stage in {TrainingStage.PERSONALIZED, TrainingStage.REFINEMENT}:
            if self.molecular_e_step_mode == "strict_artifact":
                if any(
                    batch.unknown_targets is not None
                    for batch in tuple(training_batches) + tuple(validation_batches)
                ):
                    raise ValueError(
                        "strict molecular training cannot map biological unknown targets "
                        "onto the transport-unassigned head"
                    )
                self._validate_weak_target_split(training_batches, validation_batches)
                for batch in tuple(training_batches) + tuple(validation_batches):
                    self._validate_frozen_e_step_batch(batch)
            elif any(
                "frozen_e_step" in batch.source_roles
                for batch in tuple(training_batches) + tuple(validation_batches)
            ):
                raise ValueError(
                    "live-student E-step negative control cannot consume frozen E-step artifacts"
                )
        if not self.allow_split_overlap:
            missing = [
                "%s/%s/%s" % (batch.donor_id, batch.sample_id, batch.bag_id)
                for batch in tuple(training_batches) + tuple(validation_batches)
                if not batch.donor_id.strip() or not batch.block_id.strip()
            ]
            if missing:
                raise ValueError(
                    "non-synthetic training requires explicit donor_id and block_id: %s"
                    % ", ".join(missing[:5])
                )
        keys = [
            (partition, batch.donor_id, batch.sample_id, batch.bag_id)
            for partition, batches in (
                ("train", training_batches),
                ("validation", validation_batches),
            )
            for batch in batches
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("bag_id must be unique within each sample and split")
        training_donors = {batch.donor_id for batch in training_batches}
        validation_donors = {batch.donor_id for batch in validation_batches}
        training_blocks = {batch.block_id for batch in training_batches if batch.block_id}
        validation_blocks = {batch.block_id for batch in validation_batches if batch.block_id}
        if not self.allow_split_overlap:
            donor_overlap = sorted(training_donors & validation_donors)
            block_overlap = sorted(training_blocks & validation_blocks)
            if donor_overlap or block_overlap:
                raise ValueError(
                    "training/validation provenance overlaps donors=%s blocks=%s"
                    % (donor_overlap, block_overlap)
                )
        set_seed(self.seed)
        self.model.to(self.device)
        if self.rna_encoder is not None:
            self.rna_encoder.to(self.device).eval()
            for parameter in self.rna_encoder.parameters():
                parameter.requires_grad_(False)
        adapter_parameters = []
        head_parameters = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("expression_decoder"):
                adapter_parameters.append(parameter)
            else:
                head_parameters.append(parameter)
        parameter_groups = [{"params": head_parameters, "lr": self.optimization.learning_rate}]
        if adapter_parameters:
            parameter_groups.append(
                {"params": adapter_parameters, "lr": self.optimization.adapter_learning_rate}
            )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=self.optimization.weight_decay,
        )
        warmup_epochs = int(round(self.optimization.epochs * self.optimization.warmup_fraction))

        def learning_rate_factor(epoch: int) -> float:
            if warmup_epochs and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, self.optimization.epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=self.optimization.mixed_precision and self.device.type == "cuda"
        )
        best_loss = float("inf")
        best_epoch = -1
        best_state = None
        stale = 0
        history: List[Dict[str, float]] = []
        for epoch in range(self.optimization.epochs):
            scale_before = float(scaler.get_scale())
            train_metrics = self._epoch(training_batches, optimizer, scaler)
            optimizer_step_skipped = bool(
                scaler.is_enabled() and float(scaler.get_scale()) < scale_before
            )
            with torch.no_grad():
                validation_metrics = self._epoch(validation_batches, None)
            row = {"epoch": float(epoch)}
            row.update({"train/%s" % key: value for key, value in train_metrics.items()})
            row.update({"validation/%s" % key: value for key, value in validation_metrics.items()})
            row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            row["optimizer_step_skipped"] = float(optimizer_step_skipped)
            history.append(row)
            if not optimizer_step_skipped:
                scheduler.step()
            validation_loss = validation_metrics["selection_total"]
            if validation_loss < best_loss - 1.0e-8:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= self.optimization.early_stopping_patience:
                break
        assert best_state is not None
        self.model.load_state_dict(best_state)
        return HEIRTrainingResult(best_epoch, best_loss, tuple(history))
