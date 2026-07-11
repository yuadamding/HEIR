"""Canonical expression scale shared by RNA teachers, weak targets, and outputs."""

import numpy as np

EXPRESSION_TARGET_SUM = 10_000.0
EXPRESSION_SPACE_ID = "log1p-cpm-10000-v1"
EXPRESSION_MAX = float(np.log1p(EXPRESSION_TARGET_SUM))


def log1p_cpm(values: np.ndarray, axis: int = 1) -> np.ndarray:
    """Normalize non-negative counts to 10k total and apply ``log1p``."""

    counts = np.asarray(values, dtype=np.float32)
    if counts.ndim != 2 or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("expression counts must be a finite non-negative matrix")
    library = counts.sum(axis=axis, keepdims=True)
    normalized = counts * (EXPRESSION_TARGET_SUM / np.maximum(library, 1.0))
    return np.log1p(normalized).astype(np.float32)


__all__ = [
    "EXPRESSION_MAX",
    "EXPRESSION_SPACE_ID",
    "EXPRESSION_TARGET_SUM",
    "log1p_cpm",
]
