"""Configurable composite objective and structured diagnostics for HEIR."""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .biological import (
    anchor_classification_loss,
    boundary_graph_loss,
    cycle_consistency_loss,
    hierarchy_consistency_loss,
    marker_centroid_loss,
    marker_ranking_loss,
    program_score_loss,
    pseudobulk_loss,
    residual_gaussian_kl_loss,
    residual_mahalanobis_loss,
    scgpt_representation_loss,
    type_conditioned_program_score_loss,
    unknown_calibration_loss,
)
from .distribution import (
    UnbalancedSinkhornResult,
    dirichlet_composition_prior_loss,
    jensen_shannon_composition_loss,
    soft_composition_bounds_loss,
    unbalanced_sinkhorn,
)


@dataclass(frozen=True)
class HEIRLossConfig:
    """Weights and stable numerical settings for :class:`HEIRCompositeLoss`."""

    cell_type_weight: float = 1.0
    uot_weight: float = 1.0
    program_weight: float = 1.0
    marker_weight: float = 1.0
    pseudobulk_weight: float = 0.5
    composition_weight: float = 0.5
    composition_bounds_weight: float = 0.0
    dirichlet_weight: float = 0.0
    cycle_weight: float = 0.25
    residual_weight: float = 0.1
    latent_kl_weight: float = 0.1
    graph_weight: float = 0.05
    calibration_weight: float = 0.05
    hierarchy_weight: float = 0.1
    scgpt_weight: float = 0.5
    composition_dirichlet_strength: float = 10.0
    uot_epsilon: float = 0.1
    uot_marginal_relaxation: float = 1.0
    uot_iterations: int = 160
    uot_convergence_tolerance: Optional[float] = None
    uot_unknown_mass: float = 0.05
    uot_unknown_cost: float = 1.0
    pseudobulk_metric: str = "mse"
    program_metric: str = "mse"
    marker_metric: str = "mse"
    pseudobulk_log1p_expression: bool = False
    graph_power: float = 1.0
    graph_boundary_margin: float = 0.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        names = (
            "cell_type_weight",
            "uot_weight",
            "program_weight",
            "marker_weight",
            "pseudobulk_weight",
            "composition_weight",
            "composition_bounds_weight",
            "dirichlet_weight",
            "cycle_weight",
            "residual_weight",
            "latent_kl_weight",
            "graph_weight",
            "calibration_weight",
            "hierarchy_weight",
            "scgpt_weight",
        )
        if any(getattr(self, name) < 0 for name in names):
            raise ValueError("loss weights cannot be negative")
        if self.uot_epsilon <= 0 or self.uot_marginal_relaxation <= 0:
            raise ValueError("UOT epsilon and marginal relaxation must be positive")
        if self.composition_dirichlet_strength <= 0:
            raise ValueError("composition_dirichlet_strength must be positive")
        if self.uot_iterations <= 0 or self.uot_unknown_cost < 0:
            raise ValueError("UOT iterations must be positive and unknown cost nonnegative")
        if self.uot_convergence_tolerance is not None and (
            not math.isfinite(self.uot_convergence_tolerance) or self.uot_convergence_tolerance <= 0
        ):
            raise ValueError("UOT convergence tolerance must be finite and positive")
        if not 0.0 <= self.uot_unknown_mass < 1.0:
            raise ValueError("uot_unknown_mass must be in [0, 1)")
        if self.graph_power < 1 or self.graph_boundary_margin < 0 or self.eps <= 0:
            raise ValueError("invalid graph or epsilon setting")
        metrics = {"mse", "smooth_l1", "cosine"}
        if any(
            value not in metrics
            for value in (self.pseudobulk_metric, self.program_metric, self.marker_metric)
        ):
            raise ValueError("regression metrics must be mse, smooth_l1, or cosine")
        if not isinstance(self.pseudobulk_log1p_expression, bool):
            raise TypeError("pseudobulk_log1p_expression must be boolean")

    def to_dict(self) -> Dict[str, Any]:
        """Return standard-type checkpoint metadata."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "HEIRLossConfig":
        """Reconstruct a config from metadata."""

        return cls(**dict(values))

    @classmethod
    def from_experiment_weights(cls, values: Any, **overrides: Any) -> "HEIRLossConfig":
        """Map the public ``LossWeightConfig`` names onto this objective."""

        source = asdict(values) if hasattr(values, "__dataclass_fields__") else dict(values)
        mapped = {
            "cell_type_weight": source.get("cell_type", 1.0),
            "marker_weight": source.get("marker", 0.2),
            "uot_weight": source.get("uot", 1.0),
            "program_weight": source.get("program", 1.0),
            "pseudobulk_weight": source.get("pseudobulk", 0.5),
            "composition_weight": 0.0,
            "dirichlet_weight": source.get("composition", 0.5),
            "cycle_weight": source.get("cycle", 0.25),
            "residual_weight": source.get("residual", 0.1),
            "latent_kl_weight": source.get("latent_kl", 0.1),
            "graph_weight": source.get("graph", 0.05),
            "calibration_weight": source.get("calibration", 0.05),
            "hierarchy_weight": source.get("hierarchy", 0.1),
            "scgpt_weight": source.get("scgpt", 0.5),
        }
        mapped.update(overrides)
        return cls(**mapped)


def _get(output: Any, name: str) -> Optional[Tensor]:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


class HEIRCompositeLoss(nn.Module):
    """Combine available weak, biological, graph, and anchor objectives."""

    _TERMS = (
        "cell_type",
        "uot",
        "program",
        "marker",
        "pseudobulk",
        "composition",
        "composition_bounds",
        "dirichlet",
        "cycle",
        "residual",
        "latent_kl",
        "graph",
        "calibration",
        "hierarchy",
        "scgpt",
    )

    def __init__(self, config: HEIRLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        output: Any,
        *,
        sample_index: Optional[Tensor] = None,
        cell_weights: Optional[Tensor] = None,
        biological_cell_weights: Optional[Tensor] = None,
        molecular_type_responsibilities: Optional[Tensor] = None,
        sample_weights: Optional[Tensor] = None,
        target_composition: Optional[Tensor] = None,
        composition_lower: Optional[Tensor] = None,
        composition_upper: Optional[Tensor] = None,
        dirichlet_concentration: Optional[Tensor] = None,
        uot_cost: Optional[Tensor] = None,
        uot_source_mass: Optional[Tensor] = None,
        uot_target_mass: Optional[Tensor] = None,
        uot_source_mask: Optional[Tensor] = None,
        uot_target_mask: Optional[Tensor] = None,
        uot_pair_mask: Optional[Tensor] = None,
        uot_unknown_cost: Optional[Tensor] = None,
        uot_unknown_mass: Optional[Tensor] = None,
        precomputed_uot: Optional[UnbalancedSinkhornResult] = None,
        target_pseudobulk: Optional[Tensor] = None,
        gene_weights: Optional[Tensor] = None,
        program_matrix: Optional[Tensor] = None,
        target_program_scores: Optional[Tensor] = None,
        program_weights: Optional[Tensor] = None,
        marker_centroids: Optional[Tensor] = None,
        marker_mask: Optional[Tensor] = None,
        marker_type_weights: Optional[Tensor] = None,
        residual_precision: Optional[Tensor] = None,
        residual_assignments: Optional[Tensor] = None,
        cycle_latent: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        anchor_labels: Optional[Tensor] = None,
        anchor_weights: Optional[Tensor] = None,
        anchor_class_weights: Optional[Tensor] = None,
        parent_anchor_labels: Optional[Tensor] = None,
        parent_anchor_weights: Optional[Tensor] = None,
        fine_to_parent: Optional[Sequence[int]] = None,
        unknown_targets: Optional[Tensor] = None,
        scgpt_type_prototypes: Optional[Tensor] = None,
        scgpt_type_variances: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Evaluate objectives with supplied targets and return structured logs."""

        probabilities = _get(output, "type_probabilities")
        expression = _get(output, "expression")
        latent = _get(output, "latent")
        if probabilities is None or expression is None or latent is None:
            raise ValueError("output must contain type_probabilities, expression, and latent")
        base_zero = probabilities.sum() * 0.0 + expression.sum() * 0.0 + latent.sum() * 0.0
        terms: Dict[str, Tensor] = {name: base_zero for name in self._TERMS}
        diagnostics: Dict[str, Tensor] = {}
        eps = self.config.eps
        biological_weights = (
            cell_weights if biological_cell_weights is None else biological_cell_weights
        )
        conditioning_probabilities = (
            probabilities
            if molecular_type_responsibilities is None
            else molecular_type_responsibilities.detach()
        )
        if conditioning_probabilities.shape != probabilities.shape:
            raise ValueError("molecular_type_responsibilities must align to type probabilities")

        if target_composition is not None and self.config.composition_weight:
            terms["composition"] = jensen_shannon_composition_loss(
                probabilities,
                target_composition,
                sample_index,
                biological_weights,
                sample_weights,
                eps,
            )
        bounds_count = int(composition_lower is not None) + int(composition_upper is not None)
        if bounds_count == 1:
            raise ValueError("composition_lower and composition_upper must be supplied together")
        if bounds_count == 2 and self.config.composition_bounds_weight:
            assert composition_lower is not None and composition_upper is not None
            terms["composition_bounds"] = soft_composition_bounds_loss(
                probabilities,
                composition_lower,
                composition_upper,
                sample_index,
                biological_weights,
                sample_weights,
                eps,
            )
        resolved_concentration = dirichlet_concentration
        if (
            resolved_concentration is None
            and target_composition is not None
            and self.config.dirichlet_weight
        ):
            normalized_target = target_composition / target_composition.sum(
                dim=-1, keepdim=True
            ).clamp_min(eps)
            resolved_concentration = (
                1.0 + self.config.composition_dirichlet_strength * normalized_target
            )
        if resolved_concentration is not None and self.config.dirichlet_weight:
            terms["dirichlet"] = dirichlet_composition_prior_loss(
                probabilities,
                resolved_concentration,
                sample_index,
                biological_weights,
                sample_weights,
                eps,
            )

        cost = _get(output, "prototype_cost") if uot_cost is None else uot_cost
        pair_mask = _get(output, "prototype_mask") if uot_pair_mask is None else uot_pair_mask
        target_mass = uot_target_mass
        if target_mass is None:
            repeated_weights = _get(output, "prototype_weights")
            if repeated_weights is not None and repeated_weights.ndim == 2:
                if repeated_weights.shape[0]:
                    target_mass = repeated_weights[0]
                else:
                    target_mass = repeated_weights.new_empty(repeated_weights.shape[1])
        if precomputed_uot is not None and self.config.uot_weight:
            terms["uot"] = precomputed_uot.loss
            diagnostics.update(precomputed_uot.diagnostics())
        elif (
            cost is not None
            and target_mass is not None
            and cost.shape[-1] > 0
            and self.config.uot_weight
        ):
            source_mass = cell_weights if uot_source_mass is None else uot_source_mass
            result = unbalanced_sinkhorn(
                cost,
                source_mass,
                target_mass,
                uot_source_mask,
                uot_target_mask,
                pair_mask,
                epsilon=self.config.uot_epsilon,
                marginal_relaxation=self.config.uot_marginal_relaxation,
                iterations=self.config.uot_iterations,
                convergence_tolerance=self.config.uot_convergence_tolerance,
                unknown_mass=(
                    self.config.uot_unknown_mass if uot_unknown_mass is None else uot_unknown_mass
                ),
                unknown_cost=(
                    self.config.uot_unknown_cost if uot_unknown_cost is None else uot_unknown_cost
                ),
                eps=eps,
            )
            if (
                self.config.uot_convergence_tolerance is not None
                and result.converged is not None
                and not bool(result.converged.all())
            ):
                raise FloatingPointError("UOT failed to meet its convergence tolerance")
            terms["uot"] = result.loss
            diagnostics.update(result.diagnostics())

        if target_pseudobulk is not None and self.config.pseudobulk_weight:
            terms["pseudobulk"] = pseudobulk_loss(
                expression,
                target_pseudobulk,
                sample_index,
                biological_weights,
                gene_weights,
                sample_weights,
                self.config.pseudobulk_metric,
                eps,
                log1p_expression=self.config.pseudobulk_log1p_expression,
            )
        program_count = int(program_matrix is not None) + int(target_program_scores is not None)
        if program_count == 1:
            raise ValueError("program_matrix and target_program_scores must be supplied together")
        if program_count == 2 and self.config.program_weight:
            assert program_matrix is not None and target_program_scores is not None
            if target_program_scores.ndim == 2:
                terms["program"] = type_conditioned_program_score_loss(
                    expression,
                    conditioning_probabilities,
                    program_matrix,
                    target_program_scores,
                    biological_weights,
                    program_weights,
                    metric=self.config.program_metric,
                    eps=eps,
                )
            else:
                terms["program"] = program_score_loss(
                    expression,
                    program_matrix,
                    target_program_scores,
                    sample_index,
                    biological_weights,
                    program_weights,
                    sample_weights,
                    self.config.program_metric,
                    eps,
                )
        if marker_centroids is not None and self.config.marker_weight:
            if marker_mask is not None:
                terms["marker"] = marker_ranking_loss(
                    expression,
                    conditioning_probabilities,
                    marker_mask,
                    biological_weights,
                    eps=eps,
                )
            else:
                terms["marker"] = marker_centroid_loss(
                    expression,
                    conditioning_probabilities,
                    marker_centroids,
                    marker_mask,
                    biological_weights,
                    marker_type_weights,
                    self.config.marker_metric,
                    eps=eps,
                )

        residual = _get(output, "residual")
        residual_logvar = _get(output, "residual_logvar")
        prototype_probabilities = _get(output, "prototype_probabilities")
        if residual is not None and self.config.residual_weight:
            resolved_assignments = residual_assignments
            if (
                resolved_assignments is None
                and residual_precision is not None
                and prototype_probabilities is not None
            ):
                if residual_precision.ndim == 3:
                    resolved_assignments = prototype_probabilities
                elif (
                    residual_precision.ndim == 2
                    and residual_precision.shape[0] == prototype_probabilities.shape[1]
                    and residual_precision.shape[0] != residual_precision.shape[1]
                ):
                    resolved_assignments = prototype_probabilities
            terms["residual"] = residual_mahalanobis_loss(
                residual,
                residual_precision,
                resolved_assignments,
                residual_logvar if residual_precision is None else None,
                biological_weights,
                eps,
            )
        residual_mu = _get(output, "residual_mu")
        if residual_mu is not None and residual_logvar is not None and self.config.latent_kl_weight:
            terms["latent_kl"] = residual_gaussian_kl_loss(
                residual_mu,
                residual_logvar,
                biological_weights,
                eps,
            )
        if cycle_latent is not None and self.config.cycle_weight:
            terms["cycle"] = cycle_consistency_loss(
                latent,
                cycle_latent,
                residual_logvar,
                biological_weights,
                eps,
            )
        if edge_index is not None and self.config.graph_weight:
            terms["graph"] = boundary_graph_loss(
                latent,
                edge_index,
                conditioning_probabilities,
                edge_weight,
                self.config.graph_boundary_margin,
                self.config.graph_power,
                eps,
            )

        logits = _get(output, "type_logits")
        unknown_probability = _get(output, "unknown_probability")
        if anchor_labels is not None and self.config.cell_type_weight:
            if logits is None:
                raise ValueError("type logits are required for anchors")
            terms["cell_type"] = anchor_classification_loss(
                logits,
                anchor_labels,
                anchor_weights,
                anchor_class_weights,
                unknown_probability,
                eps=eps,
            )
        if parent_anchor_labels is not None and self.config.cell_type_weight:
            parent_logits = _get(output, "parent_type_logits")
            if parent_logits is None:
                raise ValueError("parent anchors require a hierarchical HEIR model")
            terms["cell_type"] = terms["cell_type"] + anchor_classification_loss(
                parent_logits,
                parent_anchor_labels,
                parent_anchor_weights,
                unknown_probability=unknown_probability,
                eps=eps,
            )
        if unknown_targets is not None and self.config.calibration_weight:
            if unknown_probability is None:
                raise ValueError("unknown probability is required for calibration")
            terms["calibration"] = unknown_calibration_loss(
                unknown_probability,
                unknown_targets,
                cell_weights,
                eps,
            )
        scgpt_embedding = _get(output, "scgpt_embedding")
        if scgpt_type_prototypes is not None and self.config.scgpt_weight:
            if scgpt_embedding is None:
                raise ValueError("scGPT targets require HEIRConfig.scgpt_embedding_dim > 0")
            terms["scgpt"] = scgpt_representation_loss(
                scgpt_embedding,
                conditioning_probabilities,
                scgpt_type_prototypes,
                scgpt_type_variances,
                biological_weights,
                eps=eps,
            )
        parent_probabilities = _get(output, "parent_type_probabilities")
        if (
            fine_to_parent is not None
            and parent_probabilities is not None
            and self.config.hierarchy_weight
        ):
            terms["hierarchy"] = hierarchy_consistency_loss(
                probabilities,
                parent_probabilities,
                fine_to_parent,
                eps,
            )

        weights = {
            "cell_type": self.config.cell_type_weight,
            "uot": self.config.uot_weight,
            "program": self.config.program_weight,
            "marker": self.config.marker_weight,
            "pseudobulk": self.config.pseudobulk_weight,
            "composition": self.config.composition_weight,
            "composition_bounds": self.config.composition_bounds_weight,
            "dirichlet": self.config.dirichlet_weight,
            "cycle": self.config.cycle_weight,
            "residual": self.config.residual_weight,
            "latent_kl": self.config.latent_kl_weight,
            "graph": self.config.graph_weight,
            "calibration": self.config.calibration_weight,
            "hierarchy": self.config.hierarchy_weight,
            "scgpt": self.config.scgpt_weight,
        }
        total = base_zero
        logs: Dict[str, Tensor] = {}
        for name in self._TERMS:
            weighted = terms[name] * weights[name]
            total = total + weighted
            logs[name] = terms[name]
            logs["weighted/" + name] = weighted
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("HEIR composite loss became non-finite")
        logs.update(diagnostics)
        logs["total"] = total
        return total, logs


CompositeLoss = HEIRCompositeLoss
LossConfig = HEIRLossConfig


__all__ = ["HEIRLossConfig", "HEIRCompositeLoss", "LossConfig", "CompositeLoss"]
