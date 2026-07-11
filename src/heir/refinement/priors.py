"""Cautious updates that never replace the measured snRNA prior."""

import numpy as np


def update_measured_prior(
    old_prior: np.ndarray,
    predicted_prior: np.ndarray,
    old_weight: float = 0.80,
    maximum_total_variation: float = 0.10,
) -> np.ndarray:
    """Blend and clip prior drift in total-variation distance."""

    old = np.asarray(old_prior, dtype=np.float64)
    predicted = np.asarray(predicted_prior, dtype=np.float64)
    if old.shape != predicted.shape or old.ndim != 1 or old.size < 2:
        raise ValueError("priors must be aligned one-dimensional vectors")
    if (
        np.any(old < 0)
        or np.any(predicted < 0)
        or not np.isfinite(old).all()
        or not np.isfinite(predicted).all()
        or old.sum() <= 0
        or predicted.sum() <= 0
    ):
        raise ValueError("priors must be finite, non-negative, and have positive mass")
    if not 0.0 <= old_weight <= 1.0 or maximum_total_variation < 0:
        raise ValueError("invalid update weights")
    old = old / old.sum()
    predicted = predicted / predicted.sum()
    candidate = old_weight * old + (1.0 - old_weight) * predicted
    total_variation = 0.5 * np.abs(candidate - old).sum()
    if total_variation > maximum_total_variation and total_variation > 0:
        scale = maximum_total_variation / total_variation
        candidate = old + scale * (candidate - old)
    candidate = np.maximum(candidate, 0.0)
    return (candidate / candidate.sum()).astype(np.float32)
