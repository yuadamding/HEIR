"""Development-only nuisance correction for molecular residual targets."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from heir.data import MorphologyRidgeDatasetArtifact

from .weighted_basis import donor_weights


def _weighted_effects(
    covariates: np.ndarray, residual: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if covariates.shape[1] == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(
            (0, residual.shape[1]), dtype=np.float64
        )
    normalized = np.asarray(weights, dtype=np.float64)
    normalized = normalized / normalized.sum(dtype=np.float64)
    mean = np.sum(covariates * normalized[:, None], axis=0)
    centered = covariates - mean
    root = np.sqrt(normalized)[:, None]
    coefficients = np.linalg.pinv(centered * root, rcond=1.0e-10) @ (residual * root)
    return mean, coefficients


def fit_type_technical_effects(
    covariates: np.ndarray,
    residual: np.ndarray,
    donors: np.ndarray,
    labels: np.ndarray,
    *,
    num_types: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit nuisance effects separately by fine type with donors equally weighted."""

    means = np.zeros((num_types, covariates.shape[1]), dtype=np.float64)
    coefficients = np.zeros(
        (num_types, covariates.shape[1], residual.shape[1]), dtype=np.float64
    )
    for type_index in range(num_types):
        selected = labels == type_index
        if not np.any(selected) or len(set(donors[selected].tolist())) < 2:
            raise ValueError("technical correction requires two development donors per type")
        means[type_index], coefficients[type_index] = _weighted_effects(
            covariates[selected], residual[selected], donor_weights(donors[selected])
        )
    return means, coefficients


def correct_residuals(
    targets: np.ndarray,
    reference_means: np.ndarray,
    covariates: np.ndarray,
    labels: np.ndarray,
    technical_means: np.ndarray,
    technical_coefficients: np.ndarray,
) -> np.ndarray:
    """Apply development-fitted type-specific nuisance effects."""

    residual = np.asarray(targets, dtype=np.float64) - reference_means
    if covariates.shape[1] == 0:
        return residual
    corrected = residual.copy()
    for type_index in sorted(set(np.asarray(labels, dtype=np.int64).tolist())):
        selected = labels == type_index
        corrected[selected] -= (
            covariates[selected] - technical_means[type_index]
        ) @ technical_coefficients[type_index]
    return corrected


def endpoint_covariates(
    artifact: MorphologyRidgeDatasetArtifact, include_composition: bool
) -> np.ndarray:
    """Return depth-only or depth-plus-composition covariates."""

    if not include_composition:
        return artifact.technical_covariates
    return np.concatenate(
        (artifact.technical_covariates, artifact.composition_features), axis=1
    )


__all__ = [
    "correct_residuals",
    "endpoint_covariates",
    "fit_type_technical_effects",
]
