"""Donor-level resampling; cells are never treated as independent donors."""

from typing import Dict, Sequence

import numpy as np


def paired_donor_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    iterations: int = 10000,
    confidence: float = 0.95,
    seed: int = 17,
) -> Dict[str, float]:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or first.size < 2:
        raise ValueError("paired donor vectors must be aligned and contain at least two donors")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("donor metrics must be finite")
    if iterations <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    differences = first - second
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(iterations, len(differences)))
    bootstrap = differences[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(differences.mean()),
        "ci_lower": float(np.quantile(bootstrap, alpha)),
        "ci_upper": float(np.quantile(bootstrap, 1.0 - alpha)),
        "probability_positive": float((bootstrap > 0).mean()),
        "num_donors": float(len(differences)),
    }
