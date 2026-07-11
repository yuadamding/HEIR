"""Controlled broad-to-fine refinement coordinator with auditable stopping."""

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import RefinementConfig
from ..training.batch import HEIRTrainingBatch
from ..training.trainer import HEIRTrainer, HEIRTrainingResult
from ..utils import resolve_device
from .anchors import (
    AnchorLifecycle,
    AnchorStatus,
    select_anchors,
    update_anchor_lifecycle,
)
from .ema import EMATeacher
from .priors import update_measured_prior


@dataclass(frozen=True)
class RefinementRound:
    round_id: int
    accepted: int
    acceptance_fraction: float
    changed_fraction: float
    validation_loss: float
    best_epoch: int
    validation_uot: float
    validation_pseudobulk: float
    validation_composition: float
    validation_calibration: float
    objective_relative_change: float
    provisional: int = 0
    trusted: int = 0
    challenged: int = 0
    revoked: int = 0
    assignment_entropy: float = 0.0
    uot_marginal_residual: float = 0.0
    prior_total_variation: float = 0.0
    committed: bool = True


@dataclass(frozen=True)
class RefinementResult:
    rounds: Tuple[RefinementRound, ...]
    stopped_reason: str
    sample_prototype_weights: Mapping[str, np.ndarray]
    round_zero_validation_loss: Optional[float] = None
    selected_round: int = 0


class IterativeRefiner:
    """Generalized-EM wrapper; the RNA decoder and prototypes remain fixed."""

    def __init__(
        self,
        trainer_factory: Callable[[], HEIRTrainer],
        config: RefinementConfig,
        device: str = "cpu",
    ) -> None:
        config.validate()
        self.trainer_factory = trainer_factory
        self.config = config
        self.device = resolve_device(device)

    @staticmethod
    def _model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    @classmethod
    def _teacher_state(cls, teacher: EMATeacher) -> Dict[str, Any]:
        return {"decay": teacher.decay, "model": cls._model_state(teacher.model)}

    @staticmethod
    def _anchor_states(
        values: Mapping[Tuple[str, str, str, str], AnchorLifecycle],
    ) -> Dict[Tuple[str, str, str, str], AnchorLifecycle]:
        return {key: value.copy() for key, value in values.items()}

    @staticmethod
    def _set_round_trainability(
        model: torch.nn.Module,
        original: Mapping[str, bool],
        *,
        parent_only: bool,
    ) -> None:
        """Restrict broad rounds to the parent head and restore fine rounds."""

        resolved: Dict[str, bool] = {}
        for name, parameter in model.named_parameters():
            enabled = original[name]
            if parent_only:
                enabled = enabled and name.startswith("parent_type_head.")
            resolved[name] = enabled
        if parent_only and not any(resolved.values()):
            raise ValueError("broad refinement requires a trainable parent type head")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(resolved[name])

    @staticmethod
    def _objective_values(metrics: Mapping[str, float]) -> np.ndarray:
        """Extract the trust-region objectives from one validation evaluation."""

        return np.asarray(
            [
                metrics.get("uot", 0.0),
                metrics.get("pseudobulk", 0.0),
                metrics.get("dirichlet", metrics.get("composition", 0.0)),
                metrics.get("calibration", 0.0),
            ],
            dtype=np.float64,
        )

    @classmethod
    @torch.no_grad()
    def _round_zero_validation(
        cls,
        trainer: HEIRTrainer,
        validation_batches: Sequence[HEIRTrainingBatch],
    ) -> Tuple[float, np.ndarray]:
        """Evaluate the unrefined student before any candidate can be committed."""

        if not validation_batches:
            raise ValueError("refinement validation batches cannot be empty")
        metrics = trainer._epoch(validation_batches, None)
        if "selection_total" not in metrics:
            raise ValueError("round-0 validation did not report selection_total")
        validation_loss = float(metrics["selection_total"])
        objectives = cls._objective_values(metrics)
        if not np.isfinite(validation_loss) or not np.isfinite(objectives).all():
            raise FloatingPointError("non-finite round-0 refinement validation objective")
        return validation_loss, objectives

    @torch.no_grad()
    def _teacher_probabilities(
        self,
        teacher: EMATeacher,
        batch: HEIRTrainingBatch,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        values = batch.to(self.device)
        output = teacher.model(
            values.morphology,
            values.edge_index,
            values.edge_weight,
            prototype_means=values.prototype_means,
            prototype_variances=values.prototype_variances,
            prototype_types=values.prototype_types,
            prototype_weights=values.prototype_weights,
            prototype_mask=values.prototype_mask,
            sample_latent=False,
        )
        return (
            output.type_probabilities.cpu().numpy(),
            output.prototype_probabilities.cpu().numpy(),
            output.abstain.cpu().numpy(),
            output.unknown_probability.cpu().numpy(),
            (
                None
                if output.parent_type_probabilities is None
                else output.parent_type_probabilities.cpu().numpy()
            ),
        )

    @torch.no_grad()
    def _teacher_transport(
        self,
        trainer: HEIRTrainer,
        teacher: EMATeacher,
        batch: HEIRTrainingBatch,
        parent_probabilities: Optional[np.ndarray] = None,
        broad_level: bool = False,
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """Run the fine-prototype E-step and return detached UOT responsibilities.

        In a broad round, parent probabilities gate compatible fine types. This
        is deliberately named a gate rather than parent transport: the current
        training-batch and loss contracts still require fine-prototype
        responsibilities. A true parent-Gaussian E-step needs a versioned
        parent-level molecular target contract.
        """

        values = batch.to(self.device)
        constraints = trainer._anchor_constraints(values)
        if broad_level:
            mapping = teacher.model.config.fine_to_parent
            if mapping is None or parent_probabilities is None:
                raise ValueError("parent-gated transport requires parent probabilities")
            parent_tensor = torch.from_numpy(np.asarray(parent_probabilities)).to(
                device=self.device,
                dtype=values.morphology.dtype,
            )
            parent_constraints = parent_tensor.index_select(
                1,
                torch.tensor(mapping, dtype=torch.long, device=self.device),
            )
            constraints = (
                parent_constraints if constraints is None else constraints * parent_constraints
            )
        output = teacher.model(
            values.morphology,
            values.edge_index,
            values.edge_weight,
            prototype_means=values.prototype_means,
            prototype_variances=values.prototype_variances,
            prototype_types=values.prototype_types,
            prototype_weights=values.prototype_weights,
            prototype_mask=values.prototype_mask,
            cell_type_constraints=constraints,
            sample_latent=False,
        )
        responsibilities, result = trainer.transport_responsibilities(values, output)
        responsibility_array = responsibilities.cpu().numpy()
        if responsibility_array.shape[1] > 1:
            row_mass = responsibility_array.sum(axis=1, keepdims=True)
            valid = row_mass[:, 0] > 0
            conditional = responsibility_array / np.maximum(row_mass, 1.0e-12)
            entropy = -(conditional * np.log(np.maximum(conditional, 1.0e-12))).sum(
                axis=1
            ) / np.log(responsibility_array.shape[1])
            assignment_entropy = float(entropy[valid].mean()) if valid.any() else 0.0
        else:
            assignment_entropy = 0.0
        source_error = float(result.source_marginal_error.detach().cpu().mean())
        target_error = float(result.target_marginal_error.detach().cpu().mean())
        transported_target_mass = (
            result.target_marginal.detach().cpu().numpy()[: responsibility_array.shape[1]]
        )
        return (
            responsibility_array,
            assignment_entropy,
            source_error + target_error,
            transported_target_mass,
        )

    def fit(
        self,
        training_batches: Sequence[HEIRTrainingBatch],
        validation_batches: Sequence[HEIRTrainingBatch],
        view_probabilities: Optional[Mapping[str, np.ndarray]] = None,
    ) -> RefinementResult:
        if not self.config.enabled or self.config.maximum_rounds == 0:
            return RefinementResult(tuple(), "disabled", {}, selected_round=0)
        if any(batch.target_spatial_expression is not None for batch in training_batches):
            raise ValueError("target spatial expression cannot enter refinement")
        trainer = self.trainer_factory()
        # HEIR v0.1 never lets image pseudo-labels move the molecular manifold.
        # Later low-rank decoder adapters must be an explicit, separately
        # tested model version rather than an accidental optimizer side effect.
        trainer.model.freeze_expression_decoder(True)
        trainer.model.to(self.device)
        if self.config.broad_refinement_rounds > 0 and trainer.model.config.fine_to_parent is None:
            raise ValueError(
                "broad refinement requires a hierarchical model; use "
                "broad_refinement_rounds=0 for a fine-only ablation"
            )
        original_trainability = {
            name: parameter.requires_grad for name, parameter in trainer.model.named_parameters()
        }
        teacher = EMATeacher(trainer.model, self.config.teacher_ema)
        teacher.model.to(self.device)
        if self.config.require_view_agreement and view_probabilities is None:
            raise ValueError(
                "refinement requires independent view predictions when view agreement is enabled"
            )
        anchor_states: Dict[Tuple[str, str, str, str], AnchorLifecycle] = {}
        audit: List[RefinementRound] = []
        stable_rounds = 0
        reason = "maximum_rounds"
        working_batches = list(training_batches)
        measured_priors: Dict[Tuple[str, str], np.ndarray] = {}
        for batch in training_batches:
            sample_key = (batch.donor_id, batch.sample_id)
            values = batch.prototype_weights.detach().cpu().numpy()
            if sample_key in measured_priors and not np.array_equal(
                measured_priors[sample_key], values
            ):
                raise ValueError("all bags of a sample must share the measured RNA prior")
            measured_priors[sample_key] = values.copy()
        committed_priors = {"%s::%s" % key: value.copy() for key, value in measured_priors.items()}
        # Round 0 is the complete rollback target: unrefined student, matching
        # EMA teacher, measured priors, original batches, and no anchor state.
        # Without its validation score, any finite first refinement candidate
        # would be committed unconditionally against +infinity.
        round_zero_validation_loss, previous_objectives = self._round_zero_validation(
            trainer,
            validation_batches,
        )
        global_best_loss = round_zero_validation_loss
        global_best_round = 0
        global_best_state = self._model_state(trainer.model)
        global_best_teacher_state = self._teacher_state(teacher)
        global_best_batches = list(working_batches)
        global_best_anchors: Dict[Tuple[str, str, str, str], AnchorLifecycle] = {}
        global_best_priors = {key: value.copy() for key, value in committed_priors.items()}
        for round_id in range(1, self.config.maximum_rounds + 1):
            parent_only_round = round_id <= self.config.broad_refinement_rounds
            refined_batches: List[HEIRTrainingBatch] = []
            candidate_anchor_states = dict(anchor_states)
            accepted_total = 0
            cell_total = 0
            changed_total = 0
            previous_total = 0
            provisional_total = 0
            trusted_total = 0
            challenged_total = 0
            revoked_total = 0
            predictions = []
            for batch in working_batches:
                teacher_values = self._teacher_probabilities(teacher, batch)
                parent_probabilities = teacher_values[-1]
                broad_transport = bool(
                    parent_probabilities is not None
                    and parent_probabilities.shape[1] >= 2
                    and round_id <= self.config.broad_refinement_rounds
                )
                (
                    responsibilities,
                    assignment_entropy,
                    marginal_residual,
                    transported_target_mass,
                ) = self._teacher_transport(
                    trainer,
                    teacher,
                    batch,
                    parent_probabilities=parent_probabilities,
                    broad_level=broad_transport,
                )
                predictions.append(
                    (
                        batch,
                        *teacher_values,
                        responsibilities,
                        assignment_entropy,
                        marginal_residual,
                        transported_target_mass,
                    )
                )
            sample_prior_sums: Dict[Tuple[str, str], np.ndarray] = {}
            sample_reference_priors: Dict[Tuple[str, str], np.ndarray] = {}
            sample_reference_ids: Dict[Tuple[str, str], Tuple[str, ...]] = {}
            sample_reference_types: Dict[Tuple[str, str], np.ndarray] = {}
            sample_reference_means: Dict[Tuple[str, str], np.ndarray] = {}
            sample_bag_counts: Dict[Tuple[str, str], int] = {}
            entropy_sum = 0.0
            marginal_residual_sum = 0.0
            for (
                batch,
                _,
                _,
                _,
                _,
                _,
                responsibilities,
                entropy,
                marginal_residual,
                transported_target_mass,
            ) in predictions:
                sample_key = (batch.donor_id, batch.sample_id)
                cell_weights = (
                    np.ones(len(responsibilities), dtype=np.float64)
                    if batch.cell_weights is None
                    else batch.cell_weights.detach().cpu().numpy().astype(np.float64)
                )
                # Update only from the actual transported target marginal, not
                # from the image model's local softmax or row-normalized plan.
                # Sinkhorn normalizes each bag's source mass, so restore its
                # effective cell mass before combining graph bags.
                current = transported_target_mass * float(cell_weights.sum())
                current_types = batch.prototype_types.detach().cpu().numpy()
                current_means = batch.prototype_means.detach().cpu().numpy()
                if sample_key in sample_prior_sums:
                    if (
                        sample_prior_sums[sample_key].shape != current.shape
                        or sample_reference_ids[sample_key] != batch.prototype_ids
                        or not np.array_equal(sample_reference_types[sample_key], current_types)
                        or not np.array_equal(sample_reference_means[sample_key], current_means)
                    ):
                        raise ValueError("all bags of a sample must share one prototype bank")
                    sample_prior_sums[sample_key] += current
                else:
                    sample_prior_sums[sample_key] = current.copy()
                    sample_reference_priors[sample_key] = measured_priors[sample_key]
                    sample_reference_ids[sample_key] = batch.prototype_ids
                    sample_reference_types[sample_key] = current_types.copy()
                    sample_reference_means[sample_key] = current_means.copy()
                sample_bag_counts[sample_key] = sample_bag_counts.get(sample_key, 0) + 1
                entropy_sum += float(entropy) * len(responsibilities)
                marginal_residual_sum += float(marginal_residual)
            updated_priors: Dict[Tuple[str, str], np.ndarray] = {}
            for sample_key, summed in sample_prior_sums.items():
                predicted_prior = summed
                if parent_only_round or self.config.prior_old_weight == 1.0:
                    # The primary path is an exact fixed-prior analysis; do not
                    # even renormalize an immutable measured artifact here.
                    # Broad rounds also cannot update fine-prototype priors,
                    # including in an opt-in prior-update sensitivity.
                    updated_priors[sample_key] = sample_reference_priors[sample_key].copy()
                elif predicted_prior.sum() <= 0:
                    updated_priors[sample_key] = sample_reference_priors[sample_key]
                else:
                    updated_priors[sample_key] = update_measured_prior(
                        sample_reference_priors[sample_key],
                        predicted_prior,
                        old_weight=self.config.prior_old_weight,
                        maximum_total_variation=self.config.maximum_prior_total_variation,
                    )
            candidate_priors = {
                "%s::%s" % key: value.copy() for key, value in updated_priors.items()
            }
            prior_total_variation = max(
                (
                    0.5 * np.abs(updated_priors[key] - sample_reference_priors[key]).sum()
                    for key in updated_priors
                ),
                default=0.0,
            )
            mean_assignment_entropy = entropy_sum / max(
                sum(len(item[0].morphology) for item in predictions), 1
            )
            mean_marginal_residual = marginal_residual_sum / max(len(predictions), 1)
            for (
                batch,
                probabilities,
                _,
                model_abstain,
                unknown_probability,
                parent_probabilities,
                responsibilities,
                _,
                _,
                _,
            ) in predictions:
                sample_key = (batch.donor_id, batch.sample_id)
                broad_level = bool(
                    parent_probabilities is not None
                    and parent_probabilities.shape[1] >= 2
                    and round_id <= self.config.broad_refinement_rounds
                )
                anchor_probabilities = parent_probabilities if broad_level else probabilities
                assert anchor_probabilities is not None
                level_name = "parent" if broad_level else "fine"
                batch_key = (
                    batch.donor_id,
                    batch.sample_id,
                    batch.bag_id,
                    level_name,
                )
                views = None
                if view_probabilities is not None:
                    stable_key = "%s::%s::%s" % (
                        batch.donor_id,
                        batch.sample_id,
                        batch.bag_id,
                    )
                    views = view_probabilities.get(stable_key)
                    if (
                        views is None
                        and not self.config.require_view_agreement
                        and sample_bag_counts[sample_key] == 1
                    ):
                        views = view_probabilities.get(batch.sample_id)
                if self.config.require_view_agreement:
                    if views is None or np.asarray(views).ndim not in (2, 3):
                        raise ValueError("missing independent views for %s" % stable_key)
                    if np.asarray(views).shape[0] < 2:
                        raise ValueError(
                            "view agreement requires at least two independent predictions"
                        )
                if broad_level and views is not None:
                    views = np.asarray(views)
                    if views.ndim == 3:
                        if views.shape[1:] != probabilities.shape:
                            raise ValueError("view probability predictions are misaligned")
                        views = views.argmax(axis=2)
                    mapping = teacher.model.fine_to_parent_index.detach().cpu().numpy()
                    converted = np.full_like(views, -1)
                    valid_views = (views >= 0) & (views < len(mapping))
                    converted[valid_views] = mapping[views[valid_views]]
                    views = converted
                ood_mask = None if batch.ood_mask is None else batch.ood_mask.detach().cpu().numpy()
                segmentation_confidence = (
                    None
                    if batch.segmentation_confidence is None
                    else batch.segmentation_confidence.detach().cpu().numpy()
                )
                supported_types = np.zeros(anchor_probabilities.shape[1], dtype=bool)
                prototype_types = batch.prototype_types.detach().cpu().numpy()
                supported_indices = prototype_types[prototype_types >= 0]
                if broad_level:
                    mapping = teacher.model.fine_to_parent_index.detach().cpu().numpy()
                    supported_indices = mapping[supported_indices]
                supported_types[supported_indices] = True
                if self.config.require_view_agreement and views is None:
                    # Two identical teacher predictions do not count as
                    # independent evidence; retain soft constraints only.
                    views = np.empty((0, len(probabilities)), dtype=np.int64)
                if views is not None and views.shape[0] == 0:
                    selection = select_anchors(
                        anchor_probabilities,
                        self.config.min_probability,
                        self.config.max_normalized_entropy,
                        ood_mask=ood_mask,
                        segmentation_confidence=segmentation_confidence,
                        min_segmentation_confidence=self.config.minimum_segmentation_confidence,
                        view_predictions=None,
                        supported_types=supported_types,
                        max_per_class=self.config.max_anchors_per_class,
                    )
                    if self.config.require_view_agreement:
                        accepted = np.zeros_like(selection.accepted)
                        selection = replace(selection, accepted=accepted)
                else:
                    selection = select_anchors(
                        anchor_probabilities,
                        self.config.min_probability,
                        self.config.max_normalized_entropy,
                        ood_mask=ood_mask,
                        segmentation_confidence=segmentation_confidence,
                        min_segmentation_confidence=self.config.minimum_segmentation_confidence,
                        view_predictions=views,
                        supported_types=supported_types,
                        max_per_class=self.config.max_anchors_per_class,
                    )
                # The model's own composite abstention remains an additional
                # gate, distinct from the calibrated pathology-feature OOD flag.
                if broad_level:
                    model_rejection = unknown_probability >= (
                        teacher.model.config.abstain_threshold
                    )
                else:
                    model_rejection = model_abstain
                selection = replace(
                    selection,
                    accepted=selection.accepted & ~model_rejection,
                )
                previous_state = anchor_states.get(batch_key)
                lifecycle = update_anchor_lifecycle(
                    selection,
                    anchor_probabilities,
                    previous_state,
                    min_probability=self.config.min_probability,
                    additional_rejection=model_rejection,
                )
                candidate_anchor_states[batch_key] = lifecycle
                if previous_state is not None:
                    previous_accepted = previous_state.accepted
                    current_accepted = lifecycle.accepted
                    shared = previous_accepted & current_accepted
                    changed_total += int((previous_accepted ^ current_accepted).sum())
                    changed_total += int(
                        (previous_state.labels[shared] != lifecycle.labels[shared]).sum()
                    )
                    previous_total += int((previous_accepted | current_accepted).sum())
                # Only twice-confirmed anchors enter the hard label/routing
                # interface. Provisional anchors remain auditable state until
                # a second independent teacher round confirms them.
                training_anchors = lifecycle.status == AnchorStatus.TRUSTED
                labels = np.full(len(probabilities), -100, dtype=np.int64)
                labels[training_anchors] = lifecycle.labels[training_anchors]
                weights = np.zeros(len(probabilities), dtype=np.float32)
                weights[training_anchors] = lifecycle.confidence[training_anchors]
                fine_labels = labels if not broad_level else np.full_like(labels, -100)
                fine_weights = weights if not broad_level else np.zeros_like(weights)
                parent_labels = (
                    labels
                    if broad_level
                    else (
                        None
                        if batch.parent_anchor_labels is None
                        else batch.parent_anchor_labels.detach().cpu().numpy()
                    )
                )
                parent_weights = (
                    weights
                    if broad_level
                    else (
                        None
                        if batch.parent_anchor_weights is None
                        else batch.parent_anchor_weights.detach().cpu().numpy()
                    )
                )
                refined_batches.append(
                    replace(
                        batch,
                        anchor_labels=torch.from_numpy(fine_labels),
                        anchor_weights=torch.from_numpy(fine_weights),
                        parent_anchor_labels=(
                            None if parent_labels is None else torch.from_numpy(parent_labels)
                        ),
                        parent_anchor_weights=(
                            None if parent_weights is None else torch.from_numpy(parent_weights)
                        ),
                        molecular_responsibilities=torch.from_numpy(responsibilities).to(
                            dtype=batch.morphology.dtype
                        ),
                        prototype_weights=torch.from_numpy(updated_priors[sample_key]),
                    )
                )
                accepted_total += int(training_anchors.sum())
                cell_total += len(probabilities)
                provisional_total += int((lifecycle.status == AnchorStatus.PROVISIONAL).sum())
                trusted_total += int((lifecycle.status == AnchorStatus.TRUSTED).sum())
                challenged_total += int((lifecycle.status == AnchorStatus.CHALLENGED).sum())
                revoked_total += int((lifecycle.status == AnchorStatus.REVOKED).sum())
            self._set_round_trainability(
                trainer.model,
                original_trainability,
                parent_only=parent_only_round,
            )
            try:
                result: HEIRTrainingResult = trainer.fit(refined_batches, validation_batches)
            finally:
                self._set_round_trainability(
                    trainer.model,
                    original_trainability,
                    parent_only=False,
                )
            changed_fraction = changed_total / max(previous_total, 1)
            history_row = result.history[result.best_epoch] if result.history else {}
            objective_values = self._objective_values(
                {
                    key.removeprefix("validation/"): value
                    for key, value in history_row.items()
                    if key.startswith("validation/")
                }
            )
            relative = np.abs(objective_values - previous_objectives) / np.maximum(
                np.abs(previous_objectives), 1.0e-8
            )
            objective_relative_change = float(relative.max(initial=0.0))
            validation_loss = float(result.best_validation_loss)
            # Reuse the configured absolute objective tolerance as the
            # trust-region delta to preserve the public configuration API.
            round_committed = bool(
                np.isfinite(validation_loss)
                and validation_loss <= global_best_loss + self.config.objective_stability_tolerance
            )
            audit.append(
                RefinementRound(
                    round_id=round_id,
                    accepted=accepted_total,
                    acceptance_fraction=accepted_total / max(cell_total, 1),
                    changed_fraction=changed_fraction,
                    validation_loss=result.best_validation_loss,
                    best_epoch=result.best_epoch,
                    validation_uot=float(objective_values[0]),
                    validation_pseudobulk=float(objective_values[1]),
                    validation_composition=float(objective_values[2]),
                    validation_calibration=float(objective_values[3]),
                    objective_relative_change=objective_relative_change,
                    provisional=provisional_total,
                    trusted=trusted_total,
                    challenged=challenged_total,
                    revoked=revoked_total,
                    assignment_entropy=mean_assignment_entropy,
                    uot_marginal_residual=mean_marginal_residual,
                    prior_total_variation=float(prior_total_variation),
                    committed=round_committed,
                )
            )
            if not round_committed:
                # The candidate batches, priors, and lifecycle state were kept
                # local until validation.  Restore the complete best snapshot
                # immediately and never expose the failed student to the EMA.
                trainer.model.load_state_dict(global_best_state)
                teacher.load_state_dict(global_best_teacher_state)
                working_batches = list(global_best_batches)
                anchor_states = self._anchor_states(global_best_anchors)
                committed_priors = {key: value.copy() for key, value in global_best_priors.items()}
                reason = "validation_degraded_rollback"
                break

            teacher.update(trainer.model)
            working_batches = refined_batches
            anchor_states = candidate_anchor_states
            committed_priors = {key: value.copy() for key, value in candidate_priors.items()}
            if validation_loss <= global_best_loss:
                global_best_loss = validation_loss
                global_best_round = round_id
                global_best_state = self._model_state(trainer.model)
                global_best_teacher_state = self._teacher_state(teacher)
                global_best_batches = list(working_batches)
                global_best_anchors = self._anchor_states(anchor_states)
                global_best_priors = {key: value.copy() for key, value in committed_priors.items()}
            previous_objectives = objective_values
            objectives_stable = (
                objective_relative_change <= self.config.objective_stability_tolerance
            )
            # A stable parent phase is not terminal: at least one fine round
            # must run after the configured broad schedule.
            if (
                not parent_only_round
                and previous_total
                and changed_fraction < 0.01
                and objectives_stable
            ):
                stable_rounds += 1
            else:
                stable_rounds = 0
            if stable_rounds >= self.config.stable_rounds_required:
                reason = "accepted_labels_stable"
                break
        trainer.model.load_state_dict(global_best_state)
        return RefinementResult(
            tuple(audit),
            reason,
            global_best_priors,
            round_zero_validation_loss,
            global_best_round,
        )
