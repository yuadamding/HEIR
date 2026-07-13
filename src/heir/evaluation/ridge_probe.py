"""Donor-balanced molecular target transform and per-type ridge probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from heir.data import MorphologyRidgeDatasetArtifact
from heir.utils import resolve_device

from .hierarchical_metrics import macro_r2
from .residual_targets import correct_residuals, endpoint_covariates, fit_type_technical_effects
from .weighted_basis import (
    donor_type_weights,
    donor_weights,
    stable_weighted_basis,
    weighted_standardization,
)


@dataclass(frozen=True)
class MolecularTargetFit:
    technical_means: np.ndarray
    technical_coefficients: np.ndarray
    residual_means: np.ndarray
    bases: np.ndarray
    rank: int


@dataclass(frozen=True)
class OracleRidgeFit:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target: MolecularTargetFit
    coefficients: np.ndarray
    intercepts: np.ndarray
    alpha: float

    @property
    def technical_mean(self) -> np.ndarray:
        return self.target.technical_means

    @property
    def technical_coefficients(self) -> np.ndarray:
        return self.target.technical_coefficients

    @property
    def residual_means(self) -> np.ndarray:
        return self.target.residual_means

    @property
    def bases(self) -> np.ndarray:
        return self.target.bases

    @property
    def rank(self) -> int:
        return self.target.rank


def _ridge(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    if alpha <= 0:
        raise ValueError("ridge alpha must be positive")
    weight_sum = float(weights.sum())
    feature_mean = np.sum(features * weights[:, None], axis=0) / weight_sum
    target_mean = np.sum(targets * weights[:, None], axis=0) / weight_sum
    root = np.sqrt(weights)[:, None]
    x = torch.as_tensor((features - feature_mean) * root, dtype=torch.float64, device=device)
    y = torch.as_tensor((targets - target_mean) * root, dtype=torch.float64, device=device)
    if x.shape[1] <= x.shape[0]:
        gram = x.T @ x
        gram.diagonal().add_(alpha)
        coefficients = torch.linalg.solve(gram, x.T @ y)
    else:
        gram = x @ x.T
        gram.diagonal().add_(alpha)
        coefficients = x.T @ torch.linalg.solve(gram, y)
    coefficient_array = coefficients.cpu().numpy()
    intercept = target_mean - feature_mean @ coefficient_array
    return coefficient_array, intercept


def fit_molecular_target(
    targets: np.ndarray,
    reference_means: np.ndarray,
    labels: np.ndarray,
    donors: np.ndarray,
    technical_covariates: np.ndarray,
    *,
    num_types: int,
    rank: int,
    device: str = "auto",
) -> MolecularTargetFit:
    if rank <= 0 or rank > targets.shape[1]:
        raise ValueError("ridge rank must be within the molecular target width")
    raw_residual = targets - reference_means
    technical_means, technical_coefficients = fit_type_technical_effects(
        technical_covariates,
        raw_residual,
        donors,
        labels,
        num_types=num_types,
    )
    residual = correct_residuals(
        targets,
        reference_means,
        technical_covariates,
        labels,
        technical_means,
        technical_coefficients,
    )
    residual_means = np.zeros((num_types, targets.shape[1]), dtype=np.float64)
    bases = np.zeros((num_types, targets.shape[1], rank), dtype=np.float64)
    for type_index in range(num_types):
        selected = labels == type_index
        if int(selected.sum()) < max(rank + 1, 3):
            raise ValueError("every type needs enough development cells for the selected rank")
        if len(set(donors[selected].tolist())) < 2:
            raise ValueError("every type needs at least two development donors")
        weights = donor_weights(donors[selected])
        normalized = weights / weights.sum(dtype=np.float64)
        residual_mean = np.sum(residual[selected] * normalized[:, None], axis=0)
        centered = residual[selected] - residual_mean
        residual_means[type_index] = residual_mean
        bases[type_index] = stable_weighted_basis(
            centered, rank, weights, device=device
        )
    return MolecularTargetFit(
        technical_means=technical_means,
        technical_coefficients=technical_coefficients,
        residual_means=residual_means,
        bases=bases,
        rank=rank,
    )


def target_coordinates(
    target_fit: MolecularTargetFit,
    targets: np.ndarray,
    reference_means: np.ndarray,
    covariates: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    residual = correct_residuals(
        targets,
        reference_means,
        covariates,
        labels,
        target_fit.technical_means,
        target_fit.technical_coefficients,
    )
    coordinates = np.zeros((len(targets), target_fit.rank), dtype=np.float64)
    projected = reference_means.copy()
    for type_index in sorted(set(labels.tolist())):
        selected = labels == type_index
        centered = residual[selected] - target_fit.residual_means[type_index]
        local = centered @ target_fit.bases[type_index]
        coordinates[selected] = local
        projected[selected] += target_fit.residual_means[type_index]
        projected[selected] += local @ target_fit.bases[type_index].T
    return coordinates, projected


def fit_oracle_ridge_probe(
    features: np.ndarray,
    targets: np.ndarray,
    reference_means: np.ndarray,
    labels: np.ndarray,
    donors: np.ndarray,
    technical_covariates: np.ndarray,
    *,
    num_types: int,
    rank: int,
    alpha: float,
    device: str = "auto",
    target_fit: Optional[MolecularTargetFit] = None,
) -> OracleRidgeFit:
    if len(set(donors.tolist())) < 2:
        raise ValueError("oracle ridge fitting requires at least two development donors")
    fitted_target = target_fit or fit_molecular_target(
        targets,
        reference_means,
        labels,
        donors,
        technical_covariates,
        num_types=num_types,
        rank=rank,
        device=device,
    )
    if fitted_target.rank != rank:
        raise ValueError("shared molecular target rank differs from probe rank")
    biological_weights = donor_type_weights(donors, labels)
    feature_mean, feature_scale = weighted_standardization(features, biological_weights)
    normalized_features = (features - feature_mean) / feature_scale
    truth, _ = target_coordinates(
        fitted_target, targets, reference_means, technical_covariates, labels
    )
    coefficients = np.zeros((num_types, features.shape[1], rank), dtype=np.float64)
    intercepts = np.zeros((num_types, rank), dtype=np.float64)
    target_device = resolve_device(device)
    for type_index in range(num_types):
        selected = labels == type_index
        local_weights = donor_weights(donors[selected])
        coefficients[type_index], intercepts[type_index] = _ridge(
            normalized_features[selected],
            truth[selected],
            local_weights,
            alpha,
            target_device,
        )
    return OracleRidgeFit(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target=fitted_target,
        coefficients=coefficients,
        intercepts=intercepts,
        alpha=alpha,
    )


def predict_oracle_ridge(
    fit: OracleRidgeFit,
    features: np.ndarray,
    reference_means: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    normalized = (features - fit.feature_mean) / fit.feature_scale
    coordinates = np.zeros((len(features), fit.rank), dtype=np.float64)
    prediction = np.asarray(reference_means, dtype=np.float64).copy()
    for type_index in sorted(set(labels.tolist())):
        selected = labels == type_index
        local = normalized[selected] @ fit.coefficients[type_index] + fit.intercepts[type_index]
        coordinates[selected] = local
        prediction[selected] += fit.residual_means[type_index]
        prediction[selected] += local @ fit.bases[type_index].T
    return coordinates, prediction


def fit_and_score(
    development: MorphologyRidgeDatasetArtifact,
    evaluation: MorphologyRidgeDatasetArtifact,
    features: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    rank: int,
    alpha: float,
    minimum_support: int,
    device: str,
    include_composition: bool = False,
    target_fit: Optional[MolecularTargetFit] = None,
) -> Tuple[
    float,
    OracleRidgeFit,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Sequence[Mapping[str, object]],
    Mapping[str, float],
]:
    development_covariates = endpoint_covariates(development, include_composition)
    evaluation_covariates = endpoint_covariates(evaluation, include_composition)
    fit = fit_oracle_ridge_probe(
        features,
        development.molecular_targets,
        development.reference_means,
        development.type_labels,
        development.donor_ids,
        development_covariates,
        num_types=len(development.type_names),
        rank=rank,
        alpha=alpha,
        device=device,
        target_fit=target_fit,
    )
    predicted_coordinates, prediction = predict_oracle_ridge(
        fit, evaluation_features, evaluation.reference_means, evaluation.type_labels
    )
    truth_coordinates, _ = target_coordinates(
        fit.target,
        evaluation.molecular_targets,
        evaluation.reference_means,
        evaluation_covariates,
        evaluation.type_labels,
    )
    macro, rows, donors = macro_r2(
        truth_coordinates,
        predicted_coordinates,
        evaluation.donor_ids,
        evaluation.type_labels,
        minimum_support,
    )
    return macro, fit, prediction, predicted_coordinates, truth_coordinates, rows, donors


__all__ = [
    "MolecularTargetFit",
    "OracleRidgeFit",
    "fit_and_score",
    "fit_molecular_target",
    "fit_oracle_ridge_probe",
    "predict_oracle_ridge",
    "target_coordinates",
]
