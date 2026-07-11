"""Landmark affine fitting with explicit target-registration error."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..image.coordinates import AffineTransform2D


@dataclass(frozen=True)
class RegistrationResult:
    transform: AffineTransform2D
    residuals: np.ndarray
    mean_target_error: float
    max_target_error: float


def fit_affine_landmarks(source: np.ndarray, target: np.ndarray) -> RegistrationResult:
    source_points = np.asarray(source, dtype=np.float64)
    target_points = np.asarray(target, dtype=np.float64)
    if (
        source_points.shape != target_points.shape
        or source_points.ndim != 2
        or source_points.shape[1] != 2
    ):
        raise ValueError("source and target landmarks must have identical (points, 2) shapes")
    if (
        len(source_points) < 3
        or not np.isfinite(source_points).all()
        or not np.isfinite(target_points).all()
    ):
        raise ValueError("at least three finite landmark pairs are required")
    design = np.column_stack((source_points, np.ones(len(source_points))))
    coefficients, _, rank, _ = np.linalg.lstsq(design, target_points, rcond=None)
    if rank < 3:
        raise ValueError("source landmarks are degenerate")
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :] = coefficients.T
    transform = AffineTransform2D(matrix)
    transformed = transform.transform(source_points)
    residuals = np.linalg.norm(transformed - target_points, axis=1).astype(np.float32)
    return RegistrationResult(
        transform=transform,
        residuals=residuals,
        mean_target_error=float(residuals.mean()),
        max_target_error=float(residuals.max()),
    )


def match_registered_cells(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    transform: AffineTransform2D,
    maximum_distance: float,
    require_mutual: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-centroid matching with optional mutual-nearest rejection."""

    source = np.asarray(source_centroids, dtype=np.float64)
    target = np.asarray(target_centroids, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1:] != (2,) or target.shape[1:] != (2,):
        raise ValueError("centroids must have shape (cells, 2)")
    if maximum_distance <= 0:
        raise ValueError("maximum_distance must be positive")
    transformed = transform.transform(source)
    target_tree = cKDTree(target)
    distances, indices = target_tree.query(transformed, distance_upper_bound=maximum_distance)
    valid = np.isfinite(distances) & (indices < len(target))
    if require_mutual and np.any(valid):
        source_tree = cKDTree(transformed)
        _, reverse = source_tree.query(target[indices[valid]], k=1)
        source_indices = np.flatnonzero(valid)
        valid[source_indices[reverse != source_indices]] = False
    matches = np.full(len(source), -1, dtype=np.int64)
    matches[valid] = indices[valid].astype(np.int64)
    output_distances = np.full(len(source), np.inf, dtype=np.float32)
    output_distances[valid] = distances[valid].astype(np.float32)
    return matches, output_distances
