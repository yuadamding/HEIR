"""Donor-balanced feature scaling and molecular bases."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from heir.utils import resolve_device


def donor_type_weights(donors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Give every donor and every supported type within donor equal total mass."""

    donor_values = np.asarray(donors).astype(str)
    label_values = np.asarray(labels, dtype=np.int64)
    weights = np.zeros(len(label_values), dtype=np.float64)
    unique_donors = sorted(set(donor_values.tolist()))
    for donor in unique_donors:
        donor_mask = donor_values == donor
        occupied = sorted(set(label_values[donor_mask].tolist()))
        for type_index in occupied:
            selected = donor_mask & (label_values == type_index)
            weights[selected] = 1.0 / (
                len(unique_donors) * len(occupied) * int(selected.sum())
            )
    return weights / weights.mean(dtype=np.float64)


def donor_weights(donors: np.ndarray) -> np.ndarray:
    """Give each represented donor equal total mass."""

    values = np.asarray(donors).astype(str)
    weights = np.zeros(len(values), dtype=np.float64)
    unique = sorted(set(values.tolist()))
    for donor in unique:
        selected = values == donor
        weights[selected] = 1.0 / (len(unique) * int(selected.sum()))
    return weights / weights.mean(dtype=np.float64)


def weighted_standardization(
    values: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit feature location and scale under frozen biological weights."""

    normalized = np.asarray(weights, dtype=np.float64)
    normalized = normalized / normalized.sum(dtype=np.float64)
    mean = np.sum(values * normalized[:, None], axis=0)
    variance = np.sum(np.square(values - mean) * normalized[:, None], axis=0)
    return mean, np.maximum(np.sqrt(variance), 1.0e-8)


def stable_weighted_basis(
    values: np.ndarray, rank: int, weights: np.ndarray, *, device: str = "auto"
) -> np.ndarray:
    """Fit a sign-stable weighted basis from a CUDA-capable gene covariance."""

    local_weights = np.asarray(weights, dtype=np.float64)
    if local_weights.shape != (len(values),) or np.any(local_weights <= 0):
        raise ValueError("molecular basis weights must be positive and row aligned")
    normalized_weights = local_weights / local_weights.mean(dtype=np.float64)
    target_device = resolve_device(device)
    matrix = torch.as_tensor(values, dtype=torch.float64, device=target_device)
    root = torch.as_tensor(
        np.sqrt(normalized_weights), dtype=torch.float64, device=target_device
    )[:, None]
    weighted = matrix * root
    covariance = weighted.T @ weighted
    _, eigenvectors = torch.linalg.eigh(covariance)
    right = torch.flip(eigenvectors, dims=(1,)).T.cpu().numpy()
    available = min(rank, len(right))
    basis = np.zeros((values.shape[1], rank), dtype=np.float64)
    for component in range(available):
        vector = right[component].copy()
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            vector *= -1.0
        basis[:, component] = vector
    return basis


__all__ = [
    "donor_type_weights",
    "donor_weights",
    "stable_weighted_basis",
    "weighted_standardization",
]
