"""Composition and reference-mean expression baselines."""

import numpy as np


def _reference(expression: np.ndarray) -> np.ndarray:
    values = np.asarray(expression, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("reference expression must be a non-empty matrix")
    if not np.isfinite(values).all():
        raise ValueError("reference expression must be finite")
    return values


def global_mean_prediction(reference_expression: np.ndarray, num_cells: int) -> np.ndarray:
    values = _reference(reference_expression)
    if num_cells <= 0:
        raise ValueError("num_cells must be positive")
    return np.repeat(values.mean(axis=0, keepdims=True), num_cells, axis=0).astype(np.float32)


def sample_pseudobulk_prediction(reference_expression: np.ndarray, num_cells: int) -> np.ndarray:
    """Alias kept explicit because this baseline is reported by that name."""

    return global_mean_prediction(reference_expression, num_cells)


def type_mean_prediction(
    reference_expression: np.ndarray,
    reference_type_indices: np.ndarray,
    predicted_type_probabilities: np.ndarray,
) -> np.ndarray:
    values = _reference(reference_expression)
    labels = np.asarray(reference_type_indices, dtype=np.int64)
    probabilities = np.asarray(predicted_type_probabilities, dtype=np.float64)
    if labels.shape != (values.shape[0],) or probabilities.ndim != 2:
        raise ValueError("reference labels or predicted probabilities are misaligned")
    if labels.size and (labels.min() < 0 or labels.max() >= probabilities.shape[1]):
        raise ValueError("reference label is outside the predicted ontology")
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1.0e-12)
    global_mean = values.mean(axis=0)
    type_means = np.stack(
        [
            values[labels == index].mean(axis=0) if np.any(labels == index) else global_mean
            for index in range(probabilities.shape[1])
        ]
    )
    return (probabilities @ type_means).astype(np.float32)


def prototype_mean_prediction(
    prototype_probabilities: np.ndarray,
    prototype_expression_means: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(prototype_probabilities, dtype=np.float64)
    means = np.asarray(prototype_expression_means, dtype=np.float64)
    if probabilities.ndim != 2 or means.ndim != 2 or probabilities.shape[1] != means.shape[0]:
        raise ValueError("prototype probabilities and expression means are incompatible")
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1.0e-12)
    return (probabilities @ means).astype(np.float32)
