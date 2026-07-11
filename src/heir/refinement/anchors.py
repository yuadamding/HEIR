"""Confidence-gated, revocable pseudo-label anchors.

Anchor selection is deliberately stateless.  :func:`update_anchor_lifecycle`
adds the state transition policy used by iterative refinement: a new anchor is
provisional, becomes trusted only after a second agreeing round, and can be
challenged, relabelled, or revoked as the evidence changes.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AnchorSelection:
    accepted: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray
    rejected_probability: np.ndarray
    rejected_entropy: np.ndarray
    rejected_ood: np.ndarray
    rejected_segmentation: np.ndarray
    rejected_disagreement: np.ndarray
    rejected_unsupported: np.ndarray

    @property
    def indices(self) -> np.ndarray:
        return np.flatnonzero(self.accepted)


class AnchorStatus(IntEnum):
    """Lifecycle state for one pseudo-label anchor."""

    UNASSIGNED = 0
    PROVISIONAL = 1
    TRUSTED = 2
    CHALLENGED = 3
    REVOKED = 4


@dataclass(frozen=True)
class AnchorLifecycle:
    """Round-to-round anchor state with no permanently sticky confidence."""

    status: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray
    agreement_rounds: np.ndarray
    contradiction_labels: np.ndarray
    contradiction_rounds: np.ndarray

    @property
    def accepted(self) -> np.ndarray:
        return (self.status == AnchorStatus.PROVISIONAL) | (self.status == AnchorStatus.TRUSTED)

    def copy(self) -> "AnchorLifecycle":
        return AnchorLifecycle(
            status=self.status.copy(),
            labels=self.labels.copy(),
            confidence=self.confidence.copy(),
            agreement_rounds=self.agreement_rounds.copy(),
            contradiction_labels=self.contradiction_labels.copy(),
            contradiction_rounds=self.contradiction_rounds.copy(),
        )


def _empty_lifecycle(count: int) -> AnchorLifecycle:
    return AnchorLifecycle(
        status=np.full(count, AnchorStatus.UNASSIGNED, dtype=np.uint8),
        labels=np.full(count, -1, dtype=np.int64),
        confidence=np.zeros(count, dtype=np.float32),
        agreement_rounds=np.zeros(count, dtype=np.uint8),
        contradiction_labels=np.full(count, -1, dtype=np.int64),
        contradiction_rounds=np.zeros(count, dtype=np.uint8),
    )


def _validate_lifecycle(state: AnchorLifecycle, count: int, num_types: int) -> None:
    for name in (
        "status",
        "labels",
        "confidence",
        "agreement_rounds",
        "contradiction_labels",
        "contradiction_rounds",
    ):
        if getattr(state, name).shape != (count,):
            raise ValueError("previous anchor lifecycle must align to cells")
    if np.any(state.status > AnchorStatus.REVOKED):
        raise ValueError("previous anchor lifecycle has an invalid status")
    assigned = state.labels >= 0
    if np.any(state.labels[assigned] >= num_types):
        raise ValueError("previous anchor labels are outside the cell-type ontology")
    if np.any(~np.isfinite(state.confidence)) or np.any(state.confidence < 0):
        raise ValueError("previous anchor confidence must be finite and non-negative")


def update_anchor_lifecycle(
    selection: AnchorSelection,
    probabilities: np.ndarray,
    previous: Optional[AnchorLifecycle] = None,
    *,
    min_probability: float = 0.90,
    hysteresis_probability: Optional[float] = None,
    additional_rejection: Optional[np.ndarray] = None,
) -> AnchorLifecycle:
    """Update revocable anchors from the current round's evidence.

    A high-confidence label is provisional for its first round and trusted
    after two consecutive agreeing rounds.  Existing anchors survive a soft
    confidence dip only while their *current* label posterior stays above a
    lower hysteresis threshold.  A strong contradictory selection challenges
    an anchor and relabels it after two consecutive contradictory rounds.
    Technical rejection (OOD, segmentation, view disagreement, unsupported
    type, or model abstention) revokes an existing anchor immediately.

    Confidence is always read from ``probabilities`` in this call; no previous
    confidence value is carried forward.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape (cells, types>=2)")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("probabilities must be finite and non-negative")
    mass = values.sum(axis=1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("each probability row needs positive mass")
    values = values / mass
    count, num_types = values.shape
    if selection.accepted.shape != (count,) or selection.labels.shape != (count,):
        raise ValueError("anchor selection must align to probabilities")
    if np.any(selection.labels < 0) or np.any(selection.labels >= num_types):
        raise ValueError("selected labels are outside the cell-type ontology")
    if not 0.0 <= min_probability <= 1.0:
        raise ValueError("min_probability must lie in [0, 1]")
    if hysteresis_probability is None:
        hysteresis_probability = max(0.0, min_probability - 0.10)
    if not 0.0 <= hysteresis_probability <= min_probability:
        raise ValueError("hysteresis_probability must lie in [0, min_probability]")
    if additional_rejection is None:
        extra_rejection = np.zeros(count, dtype=bool)
    else:
        extra_rejection = np.asarray(additional_rejection, dtype=bool)
        if extra_rejection.shape != (count,):
            raise ValueError("additional_rejection must align to cells")

    hard_rejection = (
        selection.rejected_ood
        | selection.rejected_segmentation
        | selection.rejected_disagreement
        | selection.rejected_unsupported
        | extra_rejection
    )
    state = _empty_lifecycle(count) if previous is None else previous.copy()
    _validate_lifecycle(state, count, num_types)
    status = state.status
    labels = state.labels
    confidence = state.confidence
    agreement_rounds = state.agreement_rounds
    contradiction_labels = state.contradiction_labels
    contradiction_rounds = state.contradiction_rounds

    for index in range(count):
        old_status = AnchorStatus(int(status[index]))
        old_label = int(labels[index])
        current_label = int(selection.labels[index])
        current_probability = float(values[index, current_label])
        old_probability = 0.0 if old_label < 0 else float(values[index, old_label])

        # Confidence is a property of this round, including for challenged or
        # revoked anchors retained for audit purposes.
        confidence[index] = old_probability if old_label >= 0 else current_probability
        if hard_rejection[index]:
            if old_status != AnchorStatus.UNASSIGNED:
                status[index] = AnchorStatus.REVOKED
                agreement_rounds[index] = 0
                contradiction_labels[index] = -1
                contradiction_rounds[index] = 0
            continue

        if old_status in {AnchorStatus.UNASSIGNED, AnchorStatus.REVOKED}:
            if selection.accepted[index]:
                status[index] = AnchorStatus.PROVISIONAL
                labels[index] = current_label
                confidence[index] = current_probability
                agreement_rounds[index] = 1
                contradiction_labels[index] = -1
                contradiction_rounds[index] = 0
            continue

        if selection.accepted[index] and current_label == old_label:
            agreeing = min(int(agreement_rounds[index]) + 1, np.iinfo(np.uint8).max)
            agreement_rounds[index] = agreeing
            status[index] = AnchorStatus.TRUSTED if agreeing >= 2 else AnchorStatus.PROVISIONAL
            confidence[index] = old_probability
            contradiction_labels[index] = -1
            contradiction_rounds[index] = 0
            continue

        if selection.accepted[index] and current_label != old_label:
            if (
                old_status == AnchorStatus.CHALLENGED
                and int(contradiction_labels[index]) == current_label
            ):
                contradictions = min(
                    int(contradiction_rounds[index]) + 1,
                    np.iinfo(np.uint8).max,
                )
            else:
                contradictions = 1
            if contradictions >= 2:
                status[index] = AnchorStatus.PROVISIONAL
                labels[index] = current_label
                confidence[index] = current_probability
                agreement_rounds[index] = 1
                contradiction_labels[index] = -1
                contradiction_rounds[index] = 0
            else:
                status[index] = AnchorStatus.CHALLENGED
                confidence[index] = old_probability
                contradiction_labels[index] = current_label
                contradiction_rounds[index] = contradictions
            continue

        if old_status != AnchorStatus.CHALLENGED and old_probability >= hysteresis_probability:
            # Retain the label through a modest confidence dip, but use the
            # newly recomputed posterior as its training weight.
            confidence[index] = old_probability
            contradiction_labels[index] = -1
            contradiction_rounds[index] = 0
            continue

        status[index] = AnchorStatus.REVOKED
        agreement_rounds[index] = 0
        contradiction_labels[index] = -1
        contradiction_rounds[index] = 0

    return AnchorLifecycle(
        status=status,
        labels=labels,
        confidence=confidence.astype(np.float32, copy=False),
        agreement_rounds=agreement_rounds,
        contradiction_labels=contradiction_labels,
        contradiction_rounds=contradiction_rounds,
    )


def select_anchors(
    probabilities: np.ndarray,
    min_probability: float = 0.90,
    max_normalized_entropy: float = 0.20,
    ood_mask: Optional[np.ndarray] = None,
    segmentation_confidence: Optional[np.ndarray] = None,
    min_segmentation_confidence: float = 0.50,
    view_predictions: Optional[np.ndarray] = None,
    supported_types: Optional[np.ndarray] = None,
    max_per_class: Optional[int] = None,
    seed: int = 17,
) -> AnchorSelection:
    """Select only anchors satisfying every blueprint acceptance condition."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape (cells, types>=2)")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("probabilities must be finite and non-negative")
    values = values / np.maximum(values.sum(axis=1, keepdims=True), 1.0e-12)
    count, num_types = values.shape
    labels = values.argmax(axis=1).astype(np.int64)
    confidence = values.max(axis=1)
    entropy = -(values * np.log(np.maximum(values, 1.0e-12))).sum(axis=1) / np.log(num_types)
    rejected_probability = confidence < min_probability
    rejected_entropy = entropy > max_normalized_entropy
    rejected_ood = np.zeros(count, dtype=bool) if ood_mask is None else np.asarray(ood_mask, bool)
    if rejected_ood.shape != (count,):
        raise ValueError("ood_mask must align to cells")
    if segmentation_confidence is None:
        rejected_segmentation = np.zeros(count, dtype=bool)
    else:
        segmentation = np.asarray(segmentation_confidence, dtype=np.float64)
        if segmentation.shape != (count,):
            raise ValueError("segmentation_confidence must align to cells")
        if (
            not np.isfinite(segmentation).all()
            or np.any(segmentation < 0)
            or np.any(segmentation > 1)
        ):
            raise ValueError("segmentation_confidence must be finite and lie in [0, 1]")
        rejected_segmentation = segmentation < min_segmentation_confidence
    if view_predictions is None:
        rejected_disagreement = np.zeros(count, dtype=bool)
    else:
        views = np.asarray(view_predictions)
        if views.ndim == 3:
            if views.shape[1:] != values.shape:
                raise ValueError("view probability predictions are misaligned")
            if not np.isfinite(views).all() or np.any(views < 0):
                raise ValueError("view probabilities must be finite and non-negative")
            if np.any(views.sum(axis=2) <= 0):
                raise ValueError("each view probability row needs positive mass")
            views = views.argmax(axis=2)
        if views.ndim != 2 or views.shape[1] != count:
            raise ValueError("view_predictions must have shape (views, cells[, types])")
        if not np.issubdtype(views.dtype, np.integer):
            raise TypeError("view prediction labels must be integers")
        if np.any(views < -1) or np.any(views >= num_types):
            raise ValueError("view prediction labels are outside the cell-type ontology")
        required = 2 if views.shape[0] >= 3 else views.shape[0]
        agreement = (views == labels[None, :]).sum(axis=0)
        rejected_disagreement = agreement < required
    if supported_types is None:
        rejected_unsupported = np.zeros(count, dtype=bool)
    else:
        supported = np.asarray(supported_types, dtype=bool)
        if supported.shape != (num_types,):
            raise ValueError("supported_types must have one flag per type")
        rejected_unsupported = ~supported[labels]
    accepted = ~(
        rejected_probability
        | rejected_entropy
        | rejected_ood
        | rejected_segmentation
        | rejected_disagreement
        | rejected_unsupported
    )
    if max_per_class is not None:
        if max_per_class <= 0:
            raise ValueError("max_per_class must be positive")
        rng = np.random.default_rng(seed)
        for type_index in range(num_types):
            candidates = np.flatnonzero(accepted & (labels == type_index))
            if len(candidates) > max_per_class:
                # Prefer confidence; seeded jitter resolves exact ties reproducibly.
                order = np.argsort(
                    -(confidence[candidates] + rng.uniform(0, 1.0e-12, len(candidates)))
                )
                accepted[candidates[order[max_per_class:]]] = False
    return AnchorSelection(
        accepted=accepted,
        labels=labels,
        confidence=confidence.astype(np.float32),
        entropy=entropy.astype(np.float32),
        rejected_probability=rejected_probability,
        rejected_entropy=rejected_entropy,
        rejected_ood=rejected_ood,
        rejected_segmentation=rejected_segmentation,
        rejected_disagreement=rejected_disagreement,
        rejected_unsupported=rejected_unsupported,
    )
