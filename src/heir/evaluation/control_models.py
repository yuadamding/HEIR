"""Feature-family registry for independently tuned morphology controls."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from heir.data import MorphologyRidgeDatasetArtifact

REQUIRED_MODEL_FAMILIES = (
    "reference_mean_only",
    "technical_only",
    "coordinate_only",
    "spatial_only",
    "local_density_only",
    "boundary_only",
    "stain_only",
    "nuclear_morphometrics_only",
    "cell_morphometrics_only",
    "cellvit_context_only",
    "disease_site_batch_only",
    "disease_site_batch_section_only",
    "combined_nuisance_only",
    "context_only",
    "nucleus_mask_image",
    "cell_mask_image",
    "target_cell_removed_context_image",
    "blank_patch_image",
    "full_context_image",
    "primary_local_context_image",
    "image_plus_morphometrics",
)

HEST_CROP_CONTRACT = {
    "crop_112um": ("registered_cell_local_context_112um", "g2_primary"),
    "nucleus_mask_only": ("nucleus_intrinsic_white_fill", "intrinsic_common_canvas"),
    "nucleus_mask_mean_fill_112um": ("nucleus_intrinsic_mean_fill", "mask_artifact_control"),
    "nucleus_mask_blurred_112um": (
        "nucleus_intrinsic_blurred_context",
        "mask_artifact_control",
    ),
    "nucleus_shape_random_location_mean_fill_112um": (
        "random_location_nucleus_shape",
        "mask_artifact_control",
    ),
    "cell_mask_only": ("cell_intrinsic_white_fill", "intrinsic_common_canvas"),
    "cell_mask_mean_fill_112um": ("cell_intrinsic_mean_fill", "mask_artifact_control"),
    "cell_mask_blurred_112um": (
        "cell_intrinsic_blurred_context",
        "mask_artifact_control",
    ),
    "cell_shape_random_location_mean_fill_112um": (
        "random_location_cell_shape",
        "mask_artifact_control",
    ),
    "context_ring_32_to_112um": ("context_outside_32um", "context_control"),
    "context_ring_64_to_112um": ("context_outside_64um", "context_control"),
    "target_cell_removed_112um": (
        "target_cell_removed_white_fill",
        "context_control",
    ),
    "target_cell_removed_mean_fill_112um": (
        "target_cell_removed_mean_fill",
        "mask_artifact_control",
    ),
    "target_cell_removed_blurred_112um": (
        "target_cell_removed_blurred",
        "mask_artifact_control",
    ),
    "random_location_cell_removed_mean_fill_112um": (
        "random_location_cell_removed",
        "mask_artifact_control",
    ),
    "crop_32um": ("field_of_view_32um_resolution_sensitivity", "resolution_sensitivity"),
    "crop_64um": ("field_of_view_64um_resolution_sensitivity", "resolution_sensitivity"),
    "blank_patch": ("blank_white_negative_control", "negative_control"),
}
REQUIRED_HEST_CROP_IDS = tuple(HEST_CROP_CONTRACT)

REQUIRED_HEST_CONTROL_DECLARATIONS = (
    "smooth_coordinates_density_boundary",
    "local_and_section_stain_quality",
    "comprehensive_nuclear_morphometrics_texture",
    "comprehensive_cell_morphometrics_texture",
    "cellvit_context_sensitivity",
    "disease_site_batch_section_adjustment",
    "within_roi_derangement",
    "spatial_block_reassignment",
    "blank_patch",
    "target_cell_removed_112um",
    "mean_fill_mask_controls",
    "blurred_replacement_mask_controls",
    "random_location_shape_matched_mask_controls",
    "common_112um_0.5mpp_intrinsic_context_canvas",
)


def _optional_matrix(
    artifact: MorphologyRidgeDatasetArtifact, names: Sequence[str]
) -> Optional[np.ndarray]:
    for name in names:
        value = getattr(artifact, name, None)
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 2
            and len(value) == len(artifact.observation_ids)
        ):
            if value.shape[1]:
                return np.asarray(value, dtype=np.float64)
    return None


def _categorical_design(arrays: Sequence[np.ndarray]) -> np.ndarray:
    rows = len(arrays[0]) if arrays else 0
    columns = []
    for values in arrays:
        identities = np.asarray(values).astype(str)
        for category in sorted(set(identities.tolist()))[1:]:
            columns.append((identities == category).astype(np.float64))
    return np.column_stack(columns) if columns else np.ones((rows, 1), dtype=np.float64)


def _crop_registry(
    artifact: MorphologyRidgeDatasetArtifact,
) -> tuple[Dict[str, np.ndarray], Mapping[str, np.ndarray]]:
    dynamic: Dict[str, np.ndarray] = {}
    by_family: Dict[str, list[np.ndarray]] = {}
    by_role: Dict[str, np.ndarray] = {}
    crop_ids = tuple(getattr(artifact, "crop_ids", ()))
    crop_roles = tuple(getattr(artifact, "crop_roles", ()))
    crop_families = tuple(getattr(artifact, "crop_comparison_families", ()))
    tensor = getattr(artifact, "image_feature_tensor", None)
    if (
        not isinstance(tensor, np.ndarray)
        or tensor.ndim != 3
        or len(crop_ids) != tensor.shape[1]
        or len(crop_roles) != len(crop_ids)
        or len(crop_families) != len(crop_ids)
    ):
        return dynamic, {}
    for index, crop_id in enumerate(crop_ids):
        # Keep each crop as a view of the source tensor.  Materializing every
        # 1,536-wide arm as float64 can multiply peak memory by the full crop
        # ladder before the independently tuned probes even start.
        values = np.asarray(tensor[:, index, :])
        dynamic["crop_image::%s" % crop_id] = values
        by_family.setdefault(crop_families[index], []).append(values)
        by_role[crop_roles[index]] = values
    aliases: Dict[str, np.ndarray] = {}
    primary = dynamic.get("crop_image::%s" % getattr(artifact, "primary_crop_id", ""))
    if primary is not None:
        aliases["full_context_image"] = primary
        aliases["primary_local_context_image"] = primary

    preferred = {
        "nucleus_mask_image": (
            "nucleus_mask_only",
            "nucleus_intrinsic_white_fill",
        ),
        "cell_mask_image": ("cell_mask_only", "cell_intrinsic_white_fill"),
        "target_cell_removed_context_image": (
            "target_cell_removed_112um",
            "target_cell_removed_white_fill",
        ),
        "blank_patch_image": ("blank_patch", "blank_white_negative_control"),
    }
    for alias, (crop_id, role) in preferred.items():
        value = dynamic.get("crop_image::%s" % crop_id)
        if value is None:
            value = by_role.get(role)
        if value is not None:
            aliases[alias] = value
    context_id = next(
        (
            crop_id
            for crop_id in (
                "target_cell_removed_112um",
                "context_ring_32_to_112um",
                "context_ring_64_to_112um",
            )
            if "crop_image::%s" % crop_id in dynamic
        ),
        None,
    )
    if context_id is not None:
        aliases["context_only"] = dynamic["crop_image::%s" % context_id]
    elif by_family.get("context_control"):
        aliases["context_only"] = by_family["context_control"][0]
    return dynamic, aliases


def feature_family_registry(
    artifact: MorphologyRidgeDatasetArtifact,
) -> Mapping[str, Optional[np.ndarray]]:
    """Resolve every carried family without inferring crop source from row level."""

    rows = len(artifact.observation_ids)
    nuclear = _optional_matrix(artifact, ("nuclear_morphometrics",))
    cell = _optional_matrix(artifact, ("cell_morphometrics",))
    cellvit = _optional_matrix(artifact, ("cellvit_context_features",))
    density = _optional_matrix(artifact, ("local_density_features",))
    boundary = _optional_matrix(artifact, ("boundary_features",))
    spatial = _optional_matrix(artifact, ("spatial_control_features",))
    crop_dynamic, crop_aliases = _crop_registry(artifact)
    legacy_aliases = {
        "context_only": _optional_matrix(artifact, ("context_features", "context_only_features")),
        "nucleus_mask_image": _optional_matrix(
            artifact, ("nucleus_mask_features", "nuclear_mask_features")
        ),
        "cell_mask_image": _optional_matrix(artifact, ("cell_mask_features",)),
        "full_context_image": _optional_matrix(artifact, ("full_context_features",)),
    }
    if (
        legacy_aliases["full_context_image"] is None
        and getattr(artifact, "crop_scale", "") == "full_context"
    ):
        legacy_aliases["full_context_image"] = artifact.frozen_features
    for name, value in legacy_aliases.items():
        if name not in crop_aliases and value is not None:
            crop_aliases[name] = value
    morphometrics = [value for value in (nuclear, cell) if value is not None]
    spatial_parts = [
        value
        for value in (artifact.coordinate_features, spatial, density, boundary)
        if value is not None and value.shape[1]
    ]
    disease_site_batch_design = _categorical_design(
        tuple(
            np.asarray(getattr(artifact, name, np.repeat("unknown", rows))).astype(str)
            for name in ("disease_states", "site_ids", "batch_ids")
        )
    )
    section_design = _categorical_design((np.asarray(artifact.section_ids).astype(str),))
    full_metadata_design = np.concatenate((disease_site_batch_design, section_design), axis=1)
    combined_nuisance_parts = [
        value
        for value in (
            artifact.technical_covariates,
            artifact.coordinate_features,
            spatial,
            density,
            boundary,
            artifact.stain_features,
            nuclear,
            cell,
            cellvit,
            full_metadata_design,
        )
        if value is not None and value.shape[1]
    ]
    result: Dict[str, Optional[np.ndarray]] = {
        "reference_mean_only": np.ones((rows, 1), dtype=np.float64),
        "technical_only": (
            artifact.technical_covariates
            if artifact.technical_covariates.shape[1]
            else np.ones((rows, 1), dtype=np.float64)
        ),
        "coordinate_only": artifact.coordinate_features,
        "spatial_only": np.concatenate(spatial_parts, axis=1),
        "local_density_only": density,
        "boundary_only": boundary,
        "stain_only": artifact.stain_features if artifact.stain_features.shape[1] else None,
        "nuclear_morphometrics_only": nuclear,
        "cell_morphometrics_only": cell,
        "cellvit_context_only": cellvit,
        "disease_site_batch_only": disease_site_batch_design,
        "disease_site_batch_section_only": full_metadata_design,
        "combined_nuisance_only": np.concatenate(combined_nuisance_parts, axis=1),
        "context_only": None,
        "nucleus_mask_image": None,
        "cell_mask_image": None,
        "target_cell_removed_context_image": None,
        "blank_patch_image": None,
        "full_context_image": None,
        "primary_local_context_image": None,
        "image_plus_morphometrics": (
            np.concatenate((artifact.frozen_features, *morphometrics), axis=1)
            if morphometrics
            else None
        ),
    }
    result.update(crop_aliases)
    result.update(crop_dynamic)
    return result


def _paired_categorical(
    development: MorphologyRidgeDatasetArtifact,
    locked_test: MorphologyRidgeDatasetArtifact,
    names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    development_columns = []
    locked_columns = []
    for name in names:
        development_values = np.asarray(getattr(development, name)).astype(str)
        locked_values = np.asarray(getattr(locked_test, name)).astype(str)
        categories = sorted(set(development_values.tolist()) | set(locked_values.tolist()))
        for category in categories[1:]:
            development_columns.append((development_values == category).astype(np.float64))
            locked_columns.append((locked_values == category).astype(np.float64))
    if not development_columns:
        return (
            np.ones((len(development.observation_ids), 1), dtype=np.float64),
            np.ones((len(locked_test.observation_ids), 1), dtype=np.float64),
        )
    return np.column_stack(development_columns), np.column_stack(locked_columns)


def paired_feature_families(
    development: MorphologyRidgeDatasetArtifact,
    locked_test: MorphologyRidgeDatasetArtifact,
) -> Mapping[str, Optional[Tuple[np.ndarray, np.ndarray]]]:
    development_registry = dict(feature_family_registry(development))
    locked_registry = dict(feature_family_registry(locked_test))
    disease_site_batch = _paired_categorical(
        development, locked_test, ("disease_states", "site_ids", "batch_ids")
    )
    with_section = _paired_categorical(
        development,
        locked_test,
        ("disease_states", "site_ids", "batch_ids", "section_ids"),
    )
    development_registry["disease_site_batch_only"] = disease_site_batch[0]
    locked_registry["disease_site_batch_only"] = disease_site_batch[1]
    development_registry["disease_site_batch_section_only"] = with_section[0]
    locked_registry["disease_site_batch_section_only"] = with_section[1]
    combined_families = (
        "technical_only",
        "coordinate_only",
        "spatial_only",
        "local_density_only",
        "boundary_only",
        "stain_only",
        "nuclear_morphometrics_only",
        "cell_morphometrics_only",
        "cellvit_context_only",
    )
    development_registry["combined_nuisance_only"] = np.concatenate(
        tuple(
            development_registry[family]
            for family in combined_families
            if development_registry.get(family) is not None
        )
        + (with_section[0],),
        axis=1,
    )
    locked_registry["combined_nuisance_only"] = np.concatenate(
        tuple(
            locked_registry[family]
            for family in combined_families
            if locked_registry.get(family) is not None
        )
        + (with_section[1],),
        axis=1,
    )
    result: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    families = sorted(set(development_registry) | set(locked_registry))
    for family in families:
        development_values = development_registry.get(family)
        locked_values = locked_registry.get(family)
        if development_values is None or locked_values is None:
            result[family] = None
        elif development_values.shape[1] != locked_values.shape[1]:
            raise ValueError("development and locked-test %s widths differ" % family)
        else:
            result[family] = (development_values, locked_values)
    return result


__all__ = [
    "HEST_CROP_CONTRACT",
    "REQUIRED_HEST_CONTROL_DECLARATIONS",
    "REQUIRED_HEST_CROP_IDS",
    "REQUIRED_MODEL_FAMILIES",
    "feature_family_registry",
    "paired_feature_families",
]
