"""Controlled broad-to-fine refinement coordinator with auditable stopping."""

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import RefinementConfig
from ..training.batch import HEIRTrainingBatch
from ..training.trainer import HEIRTrainer, HEIRTrainingResult
from ..utils import resolve_device
from .anchors import select_anchors
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


@dataclass(frozen=True)
class RefinementResult:
    rounds: Tuple[RefinementRound, ...]
    stopped_reason: str
    sample_prototype_weights: Mapping[str, np.ndarray]


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

    def fit(
        self,
        training_batches: Sequence[HEIRTrainingBatch],
        validation_batches: Sequence[HEIRTrainingBatch],
        view_probabilities: Optional[Mapping[str, np.ndarray]] = None,
    ) -> RefinementResult:
        if not self.config.enabled or self.config.maximum_rounds == 0:
            return RefinementResult(tuple(), "disabled", {})
        if any(batch.target_spatial_expression is not None for batch in training_batches):
            raise ValueError("target spatial expression cannot enter refinement")
        trainer = self.trainer_factory()
        # HEIR v0.1 never lets image pseudo-labels move the molecular manifold.
        # Later low-rank decoder adapters must be an explicit, separately
        # tested model version rather than an accidental optimizer side effect.
        trainer.model.freeze_expression_decoder(True)
        teacher = EMATeacher(trainer.model, self.config.teacher_ema)
        teacher.model.to(self.device)
        if self.config.require_view_agreement and view_probabilities is None:
            raise ValueError(
                "refinement requires independent view predictions when view agreement is enabled"
            )
        previous_labels: Dict[Tuple[str, str, str, str], np.ndarray] = {}
        previous_confidence: Dict[Tuple[str, str, str, str], np.ndarray] = {}
        audit: List[RefinementRound] = []
        stable_rounds = 0
        reason = "maximum_rounds"
        working_batches = list(training_batches)
        global_best_loss = float("inf")
        global_best_state = None
        previous_objectives: Optional[np.ndarray] = None
        final_priors: Dict[str, np.ndarray] = {}
        measured_priors: Dict[Tuple[str, str], np.ndarray] = {}
        for batch in training_batches:
            sample_key = (batch.donor_id, batch.sample_id)
            values = batch.prototype_weights.detach().cpu().numpy()
            if sample_key in measured_priors and not np.array_equal(
                measured_priors[sample_key], values
            ):
                raise ValueError("all bags of a sample must share the measured RNA prior")
            measured_priors[sample_key] = values.copy()
        global_best_priors: Dict[str, np.ndarray] = {}
        for round_id in range(self.config.maximum_rounds):
            refined_batches: List[HEIRTrainingBatch] = []
            accepted_total = 0
            cell_total = 0
            changed_total = 0
            previous_total = 0
            predictions = [
                (batch, *self._teacher_probabilities(teacher, batch)) for batch in working_batches
            ]
            sample_prior_sums: Dict[Tuple[str, str], np.ndarray] = {}
            sample_prior_counts: Dict[Tuple[str, str], float] = {}
            sample_reference_priors: Dict[Tuple[str, str], np.ndarray] = {}
            sample_reference_ids: Dict[Tuple[str, str], Tuple[str, ...]] = {}
            sample_reference_types: Dict[Tuple[str, str], np.ndarray] = {}
            sample_reference_means: Dict[Tuple[str, str], np.ndarray] = {}
            sample_bag_counts: Dict[Tuple[str, str], int] = {}
            for batch, _, prototype_probabilities, _, _, _ in predictions:
                sample_key = (batch.donor_id, batch.sample_id)
                cell_weights = (
                    np.ones(len(prototype_probabilities), dtype=np.float64)
                    if batch.cell_weights is None
                    else batch.cell_weights.detach().cpu().numpy().astype(np.float64)
                )
                current = (prototype_probabilities * cell_weights[:, None]).sum(axis=0)
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
                sample_prior_counts[sample_key] = sample_prior_counts.get(sample_key, 0.0) + float(
                    cell_weights.sum()
                )
                sample_bag_counts[sample_key] = sample_bag_counts.get(sample_key, 0) + 1
            updated_priors: Dict[Tuple[str, str], np.ndarray] = {}
            for sample_key, summed in sample_prior_sums.items():
                predicted_prior = summed / max(sample_prior_counts[sample_key], 1)
                if predicted_prior.sum() <= 0:
                    updated_priors[sample_key] = sample_reference_priors[sample_key]
                else:
                    updated_priors[sample_key] = update_measured_prior(
                        sample_reference_priors[sample_key],
                        predicted_prior,
                        old_weight=self.config.prior_old_weight,
                        maximum_total_variation=self.config.maximum_prior_total_variation,
                    )
            final_priors = {"%s::%s" % key: value.copy() for key, value in updated_priors.items()}
            for (
                batch,
                probabilities,
                _,
                model_abstain,
                unknown_probability,
                parent_probabilities,
            ) in predictions:
                sample_key = (batch.donor_id, batch.sample_id)
                broad_level = bool(
                    parent_probabilities is not None
                    and parent_probabilities.shape[1] >= 2
                    and round_id < self.config.broad_refinement_rounds
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
                    accepted = selection.accepted & (
                        unknown_probability < teacher.model.config.abstain_threshold
                    )
                else:
                    accepted = selection.accepted & ~model_abstain
                selection = replace(selection, accepted=accepted)
                if batch_key in previous_labels:
                    retained = previous_labels[batch_key] >= 0
                    shared_new = selection.accepted & retained
                    changed_labels = (
                        selection.labels[shared_new] != previous_labels[batch_key][shared_new]
                    )
                    changed_total += int((selection.accepted ^ retained).sum())
                    changed_total += int(changed_labels.sum())
                    previous_total += int((selection.accepted | retained).sum())
                    retained_labels = previous_labels[batch_key]
                    labels_for_selection = selection.labels.copy()
                    labels_for_selection[retained] = retained_labels[retained]
                    accepted = selection.accepted | retained
                    confidence_for_selection = selection.confidence.copy()
                    confidence_for_selection[retained] = previous_confidence[batch_key][retained]
                    selection = replace(
                        selection,
                        accepted=accepted,
                        labels=labels_for_selection,
                        confidence=confidence_for_selection,
                    )
                labels = np.full(len(probabilities), -100, dtype=np.int64)
                labels[selection.accepted] = selection.labels[selection.accepted]
                weights = np.zeros(len(probabilities), dtype=np.float32)
                weights[selection.accepted] = selection.confidence[selection.accepted]
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
                        prototype_weights=torch.from_numpy(updated_priors[sample_key]),
                    )
                )
                accepted_total += int(selection.accepted.sum())
                cell_total += len(probabilities)
                stored = np.full(len(probabilities), -1, dtype=np.int64)
                stored[selection.accepted] = selection.labels[selection.accepted]
                previous_labels[batch_key] = stored
                previous_confidence[batch_key] = selection.confidence.copy()
            result: HEIRTrainingResult = trainer.fit(refined_batches, validation_batches)
            teacher.update(trainer.model)
            working_batches = refined_batches
            if result.best_validation_loss < global_best_loss:
                global_best_loss = result.best_validation_loss
                global_best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in trainer.model.state_dict().items()
                }
                global_best_priors = {key: value.copy() for key, value in final_priors.items()}
            changed_fraction = changed_total / max(previous_total, 1)
            history_row = result.history[result.best_epoch] if result.history else {}
            objective_values = np.asarray(
                [
                    history_row.get("validation/uot", 0.0),
                    history_row.get("validation/pseudobulk", 0.0),
                    history_row.get(
                        "validation/dirichlet",
                        history_row.get("validation/composition", 0.0),
                    ),
                    history_row.get("validation/calibration", 0.0),
                ],
                dtype=np.float64,
            )
            if previous_objectives is None:
                objective_relative_change = float("inf")
            else:
                relative = np.abs(objective_values - previous_objectives) / np.maximum(
                    np.abs(previous_objectives), 1.0e-8
                )
                objective_relative_change = float(relative.max(initial=0.0))
            previous_objectives = objective_values
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
                )
            )
            objectives_stable = (
                objective_relative_change <= self.config.objective_stability_tolerance
            )
            if previous_total and changed_fraction < 0.01 and objectives_stable:
                stable_rounds += 1
            else:
                stable_rounds = 0
            if stable_rounds >= self.config.stable_rounds_required:
                reason = "accepted_labels_stable"
                break
        if global_best_state is not None:
            trainer.model.load_state_dict(global_best_state)
        return RefinementResult(tuple(audit), reason, global_best_priors)
