"""Architecture-matched feature registries for nonlinear qualification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

REQUIRED_NEURAL_CONTROL_FAMILIES = (
    "neural_reference_mean_only",
    "neural_combined_nuisance_only",
    "neural_image_only",
    "neural_combined_nuisance_plus_image",
    "neural_blank_patch",
    "neural_target_removed",
)


@dataclass(frozen=True)
class NeuralFeatureArm:
    name: str
    features: Optional[np.ndarray]
    view_dims: tuple[int, ...]
    available: bool
    unavailable_reason: Optional[str]
    uses_image: bool
    uses_nuisance: bool


def deduplicate_named_feature_parts(
    parts: Sequence[tuple[np.ndarray, Sequence[str]]],
) -> tuple[np.ndarray, tuple[str, ...], Mapping[str, object]]:
    """Concatenate named controls while retaining the first duplicate only."""

    if not parts:
        raise ValueError("at least one named control part is required")
    row_count = len(np.asarray(parts[0][0]))
    columns = []
    names = []
    seen = set()
    duplicate_names = []
    for values, local_names in parts:
        matrix = np.asarray(values, dtype=np.float32)
        local = tuple(str(name) for name in local_names)
        if matrix.ndim != 2 or len(matrix) != row_count or matrix.shape[1] != len(local):
            raise ValueError("named control part is malformed or not row aligned")
        if not np.all(np.isfinite(matrix)) or any(not name.strip() for name in local):
            raise ValueError("named control part contains non-finite values or empty names")
        for index, name in enumerate(local):
            if name in seen:
                duplicate_names.append(name)
                continue
            seen.add(name)
            names.append(name)
            columns.append(matrix[:, index])
    if not columns:
        raise ValueError("deduplication removed every control feature")
    return (
        np.column_stack(columns).astype(np.float32, copy=False),
        tuple(names),
        {
            "deduplicated": True,
            "input_columns": int(sum(np.asarray(values).shape[1] for values, _ in parts)),
            "retained_columns": len(columns),
            "duplicate_names": sorted(set(duplicate_names)),
        },
    )


def _matrix(values: object, name: str, rows: Optional[int] = None) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix) or (rows is not None and len(matrix) != rows):
        raise ValueError(f"{name} must be a non-empty row-aligned feature matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def build_neural_control_arms(
    primary_image: np.ndarray,
    target_removed_image: np.ndarray,
    combined_nuisance: np.ndarray,
    *,
    blank_patch: Optional[np.ndarray] = None,
) -> Mapping[str, NeuralFeatureArm]:
    """Build the mandatory paired neural families without fabricating controls."""

    image = _matrix(primary_image, "primary image")
    removed = _matrix(target_removed_image, "target-removed image", len(image))
    nuisance = _matrix(combined_nuisance, "combined nuisance", len(image))
    blank = None if blank_patch is None else _matrix(blank_patch, "blank patch", len(image))
    arms = {
        "neural_reference_mean_only": NeuralFeatureArm(
            "neural_reference_mean_only", None, (), True, None, False, False
        ),
        "neural_combined_nuisance_only": NeuralFeatureArm(
            "neural_combined_nuisance_only",
            nuisance,
            (),
            True,
            None,
            False,
            True,
        ),
        "neural_image_only": NeuralFeatureArm(
            "neural_image_only", image, (), True, None, True, False
        ),
        "neural_combined_nuisance_plus_image": NeuralFeatureArm(
            "neural_combined_nuisance_plus_image",
            np.column_stack((nuisance, image)).astype(np.float32, copy=False),
            (),
            True,
            None,
            True,
            True,
        ),
        "neural_blank_patch": NeuralFeatureArm(
            "neural_blank_patch",
            blank,
            (),
            blank is not None,
            None if blank is not None else "blocked_missing_receipt_bound_blank_feature",
            True,
            False,
        ),
        "neural_target_removed": NeuralFeatureArm(
            "neural_target_removed", removed, (), True, None, True, False
        ),
    }
    if tuple(arms) != REQUIRED_NEURAL_CONTROL_FAMILIES:
        raise RuntimeError("neural control registry differs from the frozen contract")
    return arms


def build_multiview_arms(
    full_image: np.ndarray,
    nucleus_image: np.ndarray,
    cell_image: np.ndarray,
    combined_nuisance: np.ndarray,
) -> Mapping[str, NeuralFeatureArm]:
    """Build N6/N7 with an explicit, width-aware nuisance branch."""

    full = _matrix(full_image, "full image")
    nucleus = _matrix(nucleus_image, "nucleus image", len(full))
    cell = _matrix(cell_image, "cell image", len(full))
    nuisance = _matrix(combined_nuisance, "combined nuisance", len(full))
    n6 = np.column_stack((full, nucleus, cell)).astype(np.float32, copy=False)
    n7 = np.column_stack((nuisance, full, nucleus, cell)).astype(np.float32, copy=False)
    return {
        "N6": NeuralFeatureArm(
            "N6",
            n6,
            (full.shape[1], nucleus.shape[1], cell.shape[1]),
            True,
            None,
            True,
            False,
        ),
        "N7": NeuralFeatureArm(
            "N7",
            n7,
            (nuisance.shape[1], full.shape[1], nucleus.shape[1], cell.shape[1]),
            True,
            None,
            True,
            True,
        ),
    }


__all__ = [
    "NeuralFeatureArm",
    "REQUIRED_NEURAL_CONTROL_FAMILIES",
    "build_multiview_arms",
    "build_neural_control_arms",
    "deduplicate_named_feature_parts",
]
