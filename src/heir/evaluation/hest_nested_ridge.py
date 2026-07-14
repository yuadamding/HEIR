"""Leakage-resistant utilities for donor-nested HEST ridge analyses.

The functions in this module deliberately operate on explicit training arrays.
They do not inspect held-out rows when fitting weights, standardizers, PCA
projections, or ridge coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, Sequence[float]]


def _matrix(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not len(array):
        raise ValueError(f"{name} must be a non-empty one- or two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _weights(sample_weight: Optional[ArrayLike], rows: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(rows, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (rows,):
        raise ValueError("sample weights must be one-dimensional and row aligned")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("sample weights must be finite and strictly positive")
    return weights


def donor_type_row_weights(donor_ids: ArrayLike, type_ids: ArrayLike) -> np.ndarray:
    """Equalize donors, then supported types, then cells within type.

    Every donor has the same total mass.  Within a donor, every observed type
    has the same mass even when donors support different numbers of types.
    """

    donors = np.asarray(donor_ids).astype(str)
    types = np.asarray(type_ids).astype(str)
    if donors.ndim != 1 or types.ndim != 1 or not len(donors) or len(donors) != len(types):
        raise ValueError("donor and type IDs must be non-empty, one-dimensional, and aligned")
    unique_donors = sorted(set(donors.tolist()))
    weights = np.zeros(len(donors), dtype=np.float64)
    for donor in unique_donors:
        donor_rows = donors == donor
        supported_types = sorted(set(types[donor_rows].tolist()))
        for type_id in supported_types:
            stratum = donor_rows & (types == type_id)
            weights[stratum] = 1.0 / (
                len(unique_donors) * len(supported_types) * int(stratum.sum())
            )
    return weights / weights.mean(dtype=np.float64)


def donor_section_type_row_weights(
    donor_ids: ArrayLike,
    section_ids: ArrayLike,
    type_ids: ArrayLike,
) -> np.ndarray:
    """Equalize donors, sections within donor, types within section, then cells."""

    donors = np.asarray(donor_ids).astype(str)
    sections = np.asarray(section_ids).astype(str)
    types = np.asarray(type_ids).astype(str)
    if (
        donors.ndim != 1
        or sections.ndim != 1
        or types.ndim != 1
        or not len(donors)
        or len(donors) != len(sections)
        or len(donors) != len(types)
    ):
        raise ValueError(
            "donor, section, and type IDs must be non-empty, one-dimensional, and aligned"
        )
    unique_donors = sorted(set(donors.tolist()))
    weights = np.zeros(len(donors), dtype=np.float64)
    for donor in unique_donors:
        donor_rows = donors == donor
        supported_sections = sorted(set(sections[donor_rows].tolist()))
        for section in supported_sections:
            section_rows = donor_rows & (sections == section)
            supported_types = sorted(set(types[section_rows].tolist()))
            for type_id in supported_types:
                stratum = section_rows & (types == type_id)
                weights[stratum] = 1.0 / (
                    len(unique_donors)
                    * len(supported_sections)
                    * len(supported_types)
                    * int(stratum.sum())
                )
    return weights / weights.mean(dtype=np.float64)


def grouped_donor_folds(
    donor_ids: ArrayLike,
    *,
    n_splits: int,
    seed: int = 0,
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """Create deterministic, row-balanced folds without splitting a donor.

    Donors are ordered by decreasing row count.  Seeded random ranks break
    count ties, after which each donor is assigned to the currently smallest
    fold.  This remains deterministic across Python hash seeds.
    """

    donors = np.asarray(donor_ids).astype(str)
    if donors.ndim != 1 or not len(donors):
        raise ValueError("donor IDs must be a non-empty one-dimensional array")
    unique, counts = np.unique(donors, return_counts=True)
    if n_splits < 2 or n_splits > len(unique):
        raise ValueError("n_splits must be between two and the number of donors")

    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(len(unique))
    tie_rank = np.empty(len(unique), dtype=np.int64)
    tie_rank[tie_order] = np.arange(len(unique), dtype=np.int64)
    order = sorted(range(len(unique)), key=lambda index: (-int(counts[index]), tie_rank[index]))

    fold_donors: list[list[str]] = [[] for _ in range(n_splits)]
    fold_rows = np.zeros(n_splits, dtype=np.int64)
    for index in order:
        fold = min(
            range(n_splits),
            key=lambda candidate: (
                int(fold_rows[candidate]),
                len(fold_donors[candidate]),
                candidate,
            ),
        )
        fold_donors[fold].append(str(unique[index]))
        fold_rows[fold] += int(counts[index])

    all_rows = np.arange(len(donors), dtype=np.int64)
    folds = []
    for validation_donors in fold_donors:
        validation = np.isin(donors, validation_donors)
        folds.append((all_rows[~validation], all_rows[validation]))
    return tuple(folds)


@dataclass(frozen=True)
class WeightedStandardizer:
    """Location and scale fitted under training-row weights."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: ArrayLike) -> np.ndarray:
        matrix = _matrix(values, "values")
        if matrix.shape[1] != len(self.mean):
            raise ValueError("values do not match the fitted standardizer width")
        return (matrix - self.mean) / self.scale

    def inverse_transform(self, values: ArrayLike) -> np.ndarray:
        matrix = _matrix(values, "values")
        if matrix.shape[1] != len(self.mean):
            raise ValueError("values do not match the fitted standardizer width")
        return matrix * self.scale + self.mean


def fit_weighted_standardizer(
    train_values: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    *,
    minimum_scale: float = 1.0e-8,
) -> WeightedStandardizer:
    """Fit a population-variance standardizer using training rows only."""

    values = _matrix(train_values, "training values")
    weights = _weights(sample_weight, len(values))
    if minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be positive")
    normalized = weights / weights.sum(dtype=np.float64)
    mean = np.sum(values * normalized[:, None], axis=0)
    variance = np.sum(np.square(values - mean) * normalized[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < minimum_scale] = 1.0
    return WeightedStandardizer(mean=mean, scale=scale)


def _torch_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    requested = "auto" if device is None else str(device)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.device(requested)
    if target.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("only CPU and CUDA devices are supported")
    return target


def _torch_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "cuda" else torch.float64


@dataclass(frozen=True)
class WeightedPCA:
    """A centered, weighted PCA projection fitted on training rows."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    fit_device: str

    def transform(self, values: ArrayLike) -> np.ndarray:
        matrix = _matrix(values, "values")
        if matrix.shape[1] != len(self.mean):
            raise ValueError("values do not match the fitted PCA width")
        return (matrix - self.mean) @ self.components.T


def _fit_weighted_pca_on_device(
    values: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    device: torch.device,
) -> WeightedPCA:
    dtype = _torch_dtype(device)
    matrix = torch.as_tensor(values, dtype=dtype, device=device)
    weight_tensor = torch.as_tensor(weights, dtype=dtype, device=device)
    weight_tensor = weight_tensor / weight_tensor.sum()
    mean = torch.sum(matrix * weight_tensor[:, None], dim=0)
    centered = matrix - mean
    weighted = centered * torch.sqrt(weight_tensor)[:, None]
    covariance = weighted.T @ weighted
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    selected = torch.arange(
        eigenvalues.numel() - 1,
        eigenvalues.numel() - n_components - 1,
        -1,
        device=device,
    )
    eigenvalues = torch.clamp(eigenvalues[selected], min=0.0)
    components = eigenvectors[:, selected].T
    mean_array = mean.detach().cpu().numpy().astype(np.float64, copy=False)
    component_array = components.detach().cpu().numpy().astype(np.float64, copy=False)
    eigenvalue_array = eigenvalues.detach().cpu().numpy().astype(np.float64, copy=False)

    # Eigenvector signs are arbitrary.  Orient the largest absolute loading
    # positively so serialized projections are stable and directly comparable.
    for component in component_array:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0.0:
            component *= -1.0
    return WeightedPCA(
        mean=mean_array,
        components=component_array,
        explained_variance=eigenvalue_array,
        fit_device=str(device),
    )


def fit_weighted_pca(
    train_values: ArrayLike,
    n_components: int,
    sample_weight: Optional[ArrayLike] = None,
    *,
    device: Optional[Union[str, torch.device]] = "auto",
) -> WeightedPCA:
    """Fit centered weighted PCA on training rows, with safe CUDA fallback."""

    values = _matrix(train_values, "training values")
    weights = _weights(sample_weight, len(values))
    if n_components <= 0 or n_components > min(values.shape):
        raise ValueError("n_components must be positive and no larger than min(rows, columns)")
    target = _torch_device(device)
    try:
        return _fit_weighted_pca_on_device(values, weights, n_components, target)
    except RuntimeError:
        if target.type != "cuda":
            raise
        torch.cuda.empty_cache()
        return _fit_weighted_pca_on_device(values, weights, n_components, torch.device("cpu"))


@dataclass(frozen=True)
class WeightedRidgeGrid:
    """Ridge coefficients for a complete alpha grid in standardized space."""

    alphas: np.ndarray
    coefficients: np.ndarray
    feature_standardizer: WeightedStandardizer
    target_standardizer: WeightedStandardizer
    fit_device: str

    def predict(self, values: ArrayLike) -> np.ndarray:
        """Return predictions with shape ``(alpha, row, target)``."""

        standardized = self.feature_standardizer.transform(values)
        prediction = np.einsum("np,apt->ant", standardized, self.coefficients)
        return (
            prediction * self.target_standardizer.scale[None, None, :]
            + self.target_standardizer.mean[None, None, :]
        )


def _ridge_coefficients_on_device(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    alphas: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    dtype = _torch_dtype(device)
    x = torch.as_tensor(features, dtype=dtype, device=device)
    y = torch.as_tensor(targets, dtype=dtype, device=device)
    weight_tensor = torch.as_tensor(weights, dtype=dtype, device=device)
    weight_tensor = weight_tensor / weight_tensor.sum()
    root_weight = torch.sqrt(weight_tensor)[:, None]
    weighted_x = x * root_weight
    weighted_y = y * root_weight
    gram = weighted_x.T @ weighted_x
    cross = weighted_x.T @ weighted_y

    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    projected = eigenvectors.T @ cross
    alpha_tensor = torch.as_tensor(alphas, dtype=dtype, device=device)
    scaled = projected[None, :, :] / (eigenvalues[None, :, None] + alpha_tensor[:, None, None])
    coefficients = torch.einsum("pq,aqt->apt", eigenvectors, scaled)
    return coefficients.detach().cpu().numpy().astype(np.float64, copy=False)


def fit_weighted_ridge_grid(
    train_features: ArrayLike,
    train_targets: ArrayLike,
    alphas: Sequence[float],
    sample_weight: Optional[ArrayLike] = None,
    *,
    device: Optional[Union[str, torch.device]] = "auto",
) -> WeightedRidgeGrid:
    """Fit weighted ridge for every alpha after train-only standardization.

    The optimized loss is weighted mean squared error plus ``alpha * ||B||²``.
    Normalizing the data term makes fits invariant to exact duplication of all
    rows and gives alpha the same meaning across folds with different sizes.
    """

    features = _matrix(train_features, "training features")
    targets = _matrix(train_targets, "training targets")
    if len(features) != len(targets):
        raise ValueError("training features and targets must be row aligned")
    alpha_array = np.asarray(alphas, dtype=np.float64)
    if alpha_array.ndim != 1 or not len(alpha_array):
        raise ValueError("alphas must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(alpha_array)) or np.any(alpha_array <= 0.0):
        raise ValueError("ridge alphas must be finite and strictly positive")
    weights = _weights(sample_weight, len(features))
    feature_standardizer = fit_weighted_standardizer(features, weights)
    target_standardizer = fit_weighted_standardizer(targets, weights)
    standardized_features = feature_standardizer.transform(features)
    standardized_targets = target_standardizer.transform(targets)

    target = _torch_device(device)
    try:
        coefficients = _ridge_coefficients_on_device(
            standardized_features,
            standardized_targets,
            weights,
            alpha_array,
            target,
        )
        fit_device = str(target)
    except RuntimeError:
        if target.type != "cuda":
            raise
        torch.cuda.empty_cache()
        target = torch.device("cpu")
        coefficients = _ridge_coefficients_on_device(
            standardized_features,
            standardized_targets,
            weights,
            alpha_array,
            target,
        )
        fit_device = str(target)
    return WeightedRidgeGrid(
        alphas=alpha_array.copy(),
        coefficients=coefficients,
        feature_standardizer=feature_standardizer,
        target_standardizer=target_standardizer,
        fit_device=fit_device,
    )


def weighted_ridge_predict_grid(
    train_features: ArrayLike,
    train_targets: ArrayLike,
    test_features: ArrayLike,
    alphas: Sequence[float],
    sample_weight: Optional[ArrayLike] = None,
    *,
    device: Optional[Union[str, torch.device]] = "auto",
) -> np.ndarray:
    """Fit on training rows and predict all alphas for explicit test rows."""

    fit = fit_weighted_ridge_grid(
        train_features,
        train_targets,
        alphas,
        sample_weight,
        device=device,
    )
    return fit.predict(test_features)


__all__ = [
    "WeightedPCA",
    "WeightedRidgeGrid",
    "WeightedStandardizer",
    "donor_section_type_row_weights",
    "donor_type_row_weights",
    "fit_weighted_pca",
    "fit_weighted_ridge_grid",
    "fit_weighted_standardizer",
    "grouped_donor_folds",
    "weighted_ridge_predict_grid",
]
