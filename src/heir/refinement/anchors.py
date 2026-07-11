"""Confidence-gated pseudo-label selection with explicit rejection reasons."""

from dataclasses import dataclass
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
