"""Spatial autocorrelation metrics over an explicit sparse edge graph."""

from typing import Dict

import numpy as np


def morans_i(values: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray = None) -> float:
    observations = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edge_index, dtype=np.int64)
    if observations.ndim != 1 or observations.size < 2:
        raise ValueError("values must contain at least two locations")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    if edges.size and (edges.min() < 0 or edges.max() >= len(observations)):
        raise ValueError("edge index is out of range")
    weights = (
        np.ones(edges.shape[1], dtype=np.float64)
        if edge_weight is None
        else np.asarray(edge_weight, dtype=np.float64)
    )
    if weights.shape != (edges.shape[1],) or np.any(weights < 0):
        raise ValueError("edge weights are invalid")
    centered = observations - observations.mean()
    denominator = np.square(centered).sum()
    total_weight = weights.sum()
    if denominator <= 0 or total_weight <= 0:
        return float("nan")
    numerator = (weights * centered[edges[0]] * centered[edges[1]]).sum()
    return float(len(observations) / total_weight * numerator / denominator)


def spatial_autocorrelation_agreement(
    predicted: np.ndarray,
    observed: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray = None,
) -> Dict[str, object]:
    prediction = np.asarray(predicted)
    truth = np.asarray(observed)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("matrices must have identical locations-by-features shape")
    pred_i = [
        morans_i(prediction[:, index], edge_index, edge_weight)
        for index in range(prediction.shape[1])
    ]
    true_i = [morans_i(truth[:, index], edge_index, edge_weight) for index in range(truth.shape[1])]
    difference = np.abs(np.asarray(pred_i) - np.asarray(true_i))
    return {
        "predicted_morans_i": pred_i,
        "observed_morans_i": true_i,
        "median_absolute_difference": float(np.nanmedian(difference)),
    }
