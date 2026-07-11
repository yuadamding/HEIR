"""Transparent HEIR unknown/abstention decision policy."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AbstentionDecision:
    abstain: np.ndarray
    low_probability: np.ndarray
    ood: np.ndarray
    low_segmentation_confidence: np.ndarray
    high_disagreement: np.ndarray
    high_unknown_probability: np.ndarray
    model_abstain: np.ndarray
    label_indices: np.ndarray


def apply_abstention_policy(
    type_probabilities: np.ndarray,
    probability_threshold: float = 0.60,
    ood_mask: Optional[np.ndarray] = None,
    segmentation_confidence: Optional[np.ndarray] = None,
    segmentation_threshold: float = 0.50,
    disagreement: Optional[np.ndarray] = None,
    disagreement_threshold: float = 0.20,
    unknown_probability: Optional[np.ndarray] = None,
    unknown_threshold: float = 0.50,
    model_abstain: Optional[np.ndarray] = None,
    unknown_index: int = -1,
) -> AbstentionDecision:
    probabilities = np.asarray(type_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("type_probabilities must have shape (items, classes>=2)")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    row_sum = probabilities.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1.0e-5):
        raise ValueError("probability rows must sum to one")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be in [0, 1]")
    count = probabilities.shape[0]
    low_probability = probabilities.max(axis=1) < probability_threshold
    ood = np.zeros(count, dtype=bool) if ood_mask is None else np.asarray(ood_mask, dtype=bool)
    if ood.shape != (count,):
        raise ValueError("ood_mask must align to cells")
    if segmentation_confidence is None:
        low_segmentation = np.zeros(count, dtype=bool)
    else:
        confidence = np.asarray(segmentation_confidence, dtype=np.float64)
        if confidence.shape != (count,):
            raise ValueError("segmentation_confidence must align to cells")
        low_segmentation = confidence < segmentation_threshold
    if disagreement is None:
        high_disagreement = np.zeros(count, dtype=bool)
    else:
        values = np.asarray(disagreement, dtype=np.float64)
        if values.shape != (count,):
            raise ValueError("disagreement must align to cells")
        high_disagreement = values > disagreement_threshold
    if unknown_probability is None:
        high_unknown = np.zeros(count, dtype=bool)
    else:
        unknown = np.asarray(unknown_probability, dtype=np.float64)
        if unknown.shape != (count,) or np.any((unknown < 0) | (unknown > 1)):
            raise ValueError("unknown_probability must align to cells and lie in [0, 1]")
        high_unknown = unknown >= unknown_threshold
    model_decision = (
        np.zeros(count, dtype=bool)
        if model_abstain is None
        else np.asarray(model_abstain, dtype=bool)
    )
    if model_decision.shape != (count,):
        raise ValueError("model_abstain must align to cells")
    abstain = (
        low_probability | ood | low_segmentation | high_disagreement | high_unknown | model_decision
    )
    labels = probabilities.argmax(axis=1).astype(np.int64)
    labels[abstain] = unknown_index
    return AbstentionDecision(
        abstain=abstain,
        low_probability=low_probability,
        ood=ood,
        low_segmentation_confidence=low_segmentation,
        high_disagreement=high_disagreement,
        high_unknown_probability=high_unknown,
        model_abstain=model_decision,
        label_indices=labels,
    )
