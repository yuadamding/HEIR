"""Feature-family registry for independently tuned morphology controls."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from heir.data import MorphologyRidgeDatasetArtifact

REQUIRED_MODEL_FAMILIES = (
    "reference_mean_only",
    "technical_only",
    "coordinate_only",
    "stain_only",
    "nuclear_morphometrics_only",
    "cell_morphometrics_only",
    "context_only",
    "nucleus_mask_image",
    "cell_mask_image",
    "full_context_image",
    "image_plus_morphometrics",
)


def _optional_matrix(
    artifact: MorphologyRidgeDatasetArtifact, names: Sequence[str]
) -> Optional[np.ndarray]:
    for name in names:
        value = getattr(artifact, name, None)
        if isinstance(value, np.ndarray) and value.ndim == 2 and len(value) == len(
            artifact.observation_ids
        ):
            if value.shape[1]:
                return np.asarray(value, dtype=np.float64)
    return None


def feature_family_registry(
    artifact: MorphologyRidgeDatasetArtifact,
) -> Mapping[str, Optional[np.ndarray]]:
    """Resolve every required family without inventing unavailable measurements."""

    rows = len(artifact.observation_ids)
    nuclear = _optional_matrix(
        artifact, ("nuclear_morphometrics", "nucleus_morphometrics")
    )
    cell = _optional_matrix(artifact, ("cell_morphometrics",))
    context = _optional_matrix(artifact, ("context_features", "context_only_features"))
    nucleus_mask = _optional_matrix(
        artifact, ("nucleus_mask_features", "nuclear_mask_features")
    )
    cell_mask = _optional_matrix(artifact, ("cell_mask_features",))
    full_context = _optional_matrix(artifact, ("full_context_features",))
    crop_identity = artifact.crop_scale.lower()
    if context is None and ("context_only" in crop_identity or "annulus" in crop_identity):
        context = artifact.frozen_features
    if nucleus_mask is None and "nucleus_mask" in crop_identity:
        nucleus_mask = artifact.frozen_features
    if cell_mask is None and "cell_mask" in crop_identity:
        cell_mask = artifact.frozen_features
    if full_context is None and artifact.crop_scale == "full_context":
        full_context = artifact.frozen_features
    morphometrics = [value for value in (nuclear, cell) if value is not None]
    return {
        "reference_mean_only": np.ones((rows, 1), dtype=np.float64),
        "technical_only": (
            artifact.technical_covariates
            if artifact.technical_covariates.shape[1]
            else np.ones((rows, 1), dtype=np.float64)
        ),
        "coordinate_only": artifact.coordinate_features,
        "stain_only": artifact.stain_features if artifact.stain_features.shape[1] else None,
        "nuclear_morphometrics_only": nuclear,
        "cell_morphometrics_only": cell,
        "context_only": context,
        "nucleus_mask_image": nucleus_mask,
        "cell_mask_image": cell_mask,
        "full_context_image": full_context,
        "image_plus_morphometrics": (
            np.concatenate((artifact.frozen_features, *morphometrics), axis=1)
            if morphometrics
            else None
        ),
    }


def paired_feature_families(
    development: MorphologyRidgeDatasetArtifact,
    locked_test: MorphologyRidgeDatasetArtifact,
) -> Mapping[str, Optional[Tuple[np.ndarray, np.ndarray]]]:
    development_registry = feature_family_registry(development)
    locked_registry = feature_family_registry(locked_test)
    result: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    for family in REQUIRED_MODEL_FAMILIES:
        development_values = development_registry[family]
        locked_values = locked_registry[family]
        if development_values is None or locked_values is None:
            result[family] = None
        elif development_values.shape[1] != locked_values.shape[1]:
            raise ValueError("development and locked-test %s widths differ" % family)
        else:
            result[family] = (development_values, locked_values)
    return result


__all__ = [
    "REQUIRED_MODEL_FAMILIES",
    "feature_family_registry",
    "paired_feature_families",
]
