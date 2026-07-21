"""Receipt-gated biological inputs for the exposed HEST v2 qualification."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from heir.evaluation.blank_patch_receipt import (
    array_sha256,
    load_supplement,
    ordered_observation_ids_sha256,
)
from heir.evaluation.neural_control_models import deduplicate_named_feature_parts
from heir.evaluation.nonlinear_qualification_contract import file_sha256
from heir.evaluation.nonlinear_qualification_v2_contract import (
    BLANK_FEATURE_SHA256,
    BLANK_SUPPLEMENT_SHA256,
    SOURCE_SHA256,
)
from heir.evaluation.nonlinear_qualification_v2_execution import validate_biological_authorization

SOURCE_ROWS = 36_121
REFERENCE_ROWS = 18_059
EVALUATION_ROWS = 18_062
TARGET_WIDTH = 260
IMAGE_WIDTH = 1_536
NUISANCE_WIDTH = 158
TYPE_COUNT = 38
PRIMARY_SUPPORT = 20
CROP_IDS = (
    "crop_112um",
    "cell_mask_only",
    "nucleus_mask_only",
    "target_cell_removed_112um",
)
BIOLOGICAL_SOURCE_MEMBERS = (
    "observation_ids",
    "donor_ids",
    "section_ids",
    "type_labels",
    "type_names",
    "pool_roles",
    "block_ids",
    "registration_quality_strata",
    "crop_ids",
    "image_features_by_crop_and_encoder",
    "technical_covariates",
    "technical_covariate_names",
    "coordinate_features",
    "coordinate_feature_names",
    "stain_features",
    "stain_feature_names",
    "nuclear_morphometric_features",
    "nuclear_morphometric_feature_names",
    "cell_morphometric_features",
    "cell_morphometric_feature_names",
    "local_density_features",
    "local_density_feature_names",
    "nucleus_molecular_targets",
)


@dataclass(frozen=True)
class HestBiologicalData:
    source_path: Path
    observation_ids: np.ndarray
    donor_ids: np.ndarray
    section_ids: np.ndarray
    type_labels: np.ndarray
    type_names: tuple[str, ...]
    pool_roles: np.ndarray
    block_ids: np.ndarray
    registration_quality_strata: np.ndarray
    evaluation_indices: np.ndarray
    image_features: np.ndarray
    molecular_targets: np.ndarray
    reference_means: np.ndarray
    technical_covariates: np.ndarray
    control_families: Mapping[str, np.ndarray]
    control_feature_names: Mapping[str, tuple[str, ...]]
    blank_feature: np.ndarray
    source_access: Mapping[str, object]

    @property
    def evaluation_donors(self) -> np.ndarray:
        return self.donor_ids[self.evaluation_indices]

    @property
    def evaluation_sections(self) -> np.ndarray:
        return self.section_ids[self.evaluation_indices]

    @property
    def evaluation_types(self) -> np.ndarray:
        return self.type_labels[self.evaluation_indices]

    @property
    def evaluation_observation_ids(self) -> np.ndarray:
        return self.observation_ids[self.evaluation_indices]

    @property
    def evaluation_registration_strata(self) -> np.ndarray:
        return self.registration_quality_strata[self.evaluation_indices]


class _BiologicalReader:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.members_opened: list[str] = []
        self.molecular_member_open_count = 0

    def read(self, name: str) -> np.ndarray:
        if name not in BIOLOGICAL_SOURCE_MEMBERS:
            raise ValueError(f"biological source member {name!r} is outside the frozen allowlist")
        if name == "nucleus_molecular_targets":
            if self.molecular_member_open_count:
                raise RuntimeError("HEST molecular target member may be opened only once")
            self.molecular_member_open_count += 1
        member = f"{name}.npy"
        try:
            with zipfile.ZipFile(self.path) as archive:
                if archive.namelist().count(member) != 1:
                    raise ValueError(f"registered source must contain exactly one {member}")
                with archive.open(member) as handle:
                    value = np.lib.format.read_array(handle, allow_pickle=False)
        except (OSError, zipfile.BadZipFile) as error:
            raise ValueError("registered HEST source is not a readable NPZ archive") from error
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"biological source member {name} cannot use object dtype")
        self.members_opened.append(name)
        return array


def _strings(value: object, name: str, rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype.kind not in {"S", "U"}:
        raise ValueError(f"{name} must be a row-aligned string vector")
    result = array.astype(str)
    if np.any(np.char.str_len(result) == 0):
        raise ValueError(f"{name} cannot contain empty identifiers")
    return result


def _matrix(value: object, name: str, rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or len(array) != rows or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty row-aligned matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _named_matrix(
    reader: _BiologicalReader,
    value_name: str,
    names_name: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = _matrix(reader.read(value_name), value_name, SOURCE_ROWS).astype(
        np.float32, copy=False
    )
    names = tuple(_strings(reader.read(names_name), names_name, values.shape[1]).tolist())
    if len(set(names)) != len(names):
        raise ValueError(f"{value_name} feature names must be unique")
    return values, names


def _control_families(
    reader: _BiologicalReader,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, tuple[str, ...]]]:
    technical = _named_matrix(reader, "technical_covariates", "technical_covariate_names")
    coordinate = _named_matrix(reader, "coordinate_features", "coordinate_feature_names")
    stain = _named_matrix(reader, "stain_features", "stain_feature_names")
    nucleus = _named_matrix(
        reader,
        "nuclear_morphometric_features",
        "nuclear_morphometric_feature_names",
    )
    cell = _named_matrix(
        reader,
        "cell_morphometric_features",
        "cell_morphometric_feature_names",
    )
    density = _named_matrix(reader, "local_density_features", "local_density_feature_names")
    density_names = set(density[1])
    spatial_indices = tuple(
        index
        for index, name in enumerate(coordinate[1])
        if name not in density_names and "background" not in name
    )
    if not spatial_indices:
        raise ValueError("frozen spatial control family is empty")
    spatial = (
        coordinate[0][:, spatial_indices],
        tuple(coordinate[1][index] for index in spatial_indices),
    )
    parts = {
        "technical": (technical,),
        "spatial": (spatial,),
        "stain_qc": (technical, stain),
        "morphometry_density": (nucleus, cell, density),
        "combined_nonimage": (spatial, technical, stain, nucleus, cell, density),
    }
    matrices: dict[str, np.ndarray] = {}
    names: dict[str, tuple[str, ...]] = {}
    for family, local_parts in parts.items():
        matrices[family], names[family], _ = deduplicate_named_feature_parts(local_parts)
    if matrices["combined_nonimage"].shape[1] != NUISANCE_WIDTH:
        raise ValueError("combined nonimage controls differ from the frozen 158 columns")
    return matrices, names


def _compact_types(raw: np.ndarray, type_names: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(raw)
    names = np.asarray(type_names).astype(str)
    if values.shape != (SOURCE_ROWS,) or values.dtype.kind not in {"i", "u"}:
        raise ValueError("HEST fine-type labels must be a row-aligned integer vector")
    observed = tuple(sorted(set(values.tolist())))
    if len(observed) != TYPE_COUNT or min(observed) < 0 or max(observed) >= len(names):
        raise ValueError("HEST observed fine-type ontology differs from the frozen 38 types")
    lookup = {raw_label: index for index, raw_label in enumerate(observed)}
    compact = np.asarray([lookup[int(value)] for value in values], dtype=np.int64)
    return compact, tuple(str(names[index]) for index in observed)


def _reference_means(
    targets: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    labels: np.ndarray,
    roles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    evaluation = roles == "evaluation"
    reference = roles == "reference"
    evaluation_indices = np.flatnonzero(evaluation)
    means = np.empty((len(evaluation_indices), targets.shape[1]), dtype=np.float64)
    destination = np.full(len(roles), -1, dtype=np.int64)
    destination[evaluation_indices] = np.arange(len(evaluation_indices))
    keys = sorted(set(zip(donors[evaluation], sections[evaluation], labels[evaluation])))
    for donor, section, label in keys:
        stratum = (donors == donor) & (sections == section) & (labels == label)
        local_reference = stratum & reference
        local_evaluation = stratum & evaluation
        if (
            int(local_reference.sum()) < PRIMARY_SUPPORT
            or int(local_evaluation.sum()) < PRIMARY_SUPPORT
        ):
            raise ValueError(
                "frozen donor/section/type reference or evaluation support fell below 20"
            )
        means[destination[local_evaluation]] = targets[local_reference].mean(axis=0)
    if not np.isfinite(means).all():
        raise ValueError("same-section reference means are incomplete or non-finite")
    return evaluation_indices, means


def load_hest_biological_data(
    source: Path,
    blank_supplement: Path,
    *,
    authorization: Mapping[str, object],
) -> HestBiologicalData:
    """Open each frozen biological input once after the production gate passes."""

    validate_biological_authorization(authorization)
    resolved_source = source.expanduser().resolve()
    if file_sha256(resolved_source) != SOURCE_SHA256:
        raise ValueError("HEST biological source bytes differ from the production authorization")
    reader = _BiologicalReader(resolved_source)
    observation_ids = _strings(reader.read("observation_ids"), "observation_ids", SOURCE_ROWS)
    if len(set(observation_ids.tolist())) != SOURCE_ROWS:
        raise ValueError("HEST observation IDs must remain unique")
    supplement = load_supplement(blank_supplement)
    if (
        supplement.sha256 != BLANK_SUPPLEMENT_SHA256
        or supplement.registered_source_sha256 != SOURCE_SHA256
        or supplement.ordered_observation_ids_sha256
        != ordered_observation_ids_sha256(observation_ids)
        or supplement.source_rows != SOURCE_ROWS
        or supplement.broadcast_rows != SOURCE_ROWS
        or supplement.feature.shape != (1, IMAGE_WIDTH)
        or supplement.feature_sha256 != BLANK_FEATURE_SHA256
        or array_sha256(supplement.feature) != BLANK_FEATURE_SHA256
    ):
        raise ValueError("blank supplement differs from the authorized v2 input")
    donor_ids = _strings(reader.read("donor_ids"), "donor_ids", SOURCE_ROWS)
    section_ids = _strings(reader.read("section_ids"), "section_ids", SOURCE_ROWS)
    raw_labels = reader.read("type_labels")
    raw_type_names = reader.read("type_names")
    type_labels, type_names = _compact_types(raw_labels, raw_type_names)
    pool_roles = _strings(reader.read("pool_roles"), "pool_roles", SOURCE_ROWS)
    role_names, role_counts = np.unique(pool_roles, return_counts=True)
    if dict(zip(role_names.tolist(), role_counts.tolist())) != {
        "evaluation": EVALUATION_ROWS,
        "reference": REFERENCE_ROWS,
    }:
        raise ValueError("HEST reference/evaluation pool counts changed")
    block_ids = _strings(reader.read("block_ids"), "block_ids", SOURCE_ROWS)
    registration = _strings(
        reader.read("registration_quality_strata"),
        "registration_quality_strata",
        SOURCE_ROWS,
    )
    if not set(registration.tolist()) <= {"best", "intermediate", "near_threshold", "failed"}:
        raise ValueError("HEST registration strata changed")
    crop_ids = tuple(np.asarray(reader.read("crop_ids")).astype(str).tolist())
    if crop_ids != CROP_IDS:
        raise ValueError("HEST image crop order differs from the frozen four views")
    image_features = np.asarray(reader.read("image_features_by_crop_and_encoder"))
    if image_features.shape != (SOURCE_ROWS, len(CROP_IDS), IMAGE_WIDTH):
        raise ValueError("HEST H-optimus image feature tensor has the wrong shape")
    image_features = image_features.astype(np.float32, copy=False)
    if not np.isfinite(image_features).all():
        raise ValueError("HEST H-optimus image features contain non-finite values")
    controls, control_names = _control_families(reader)
    targets = _matrix(
        reader.read("nucleus_molecular_targets"),
        "nucleus_molecular_targets",
        SOURCE_ROWS,
    ).astype(np.float64, copy=False)
    if targets.shape[1] != TARGET_WIDTH or reader.molecular_member_open_count != 1:
        raise ValueError("HEST molecular target shape or open count differs from v2")
    evaluation_indices, reference_means = _reference_means(
        targets,
        donor_ids,
        section_ids,
        type_labels,
        pool_roles,
    )
    evaluation_targets = targets[evaluation_indices].copy()
    technical = controls["technical"][evaluation_indices].astype(np.float64, copy=False)
    evaluation_controls = {
        name: values[evaluation_indices].astype(np.float32, copy=False)
        for name, values in controls.items()
    }
    if reader.members_opened != list(BIOLOGICAL_SOURCE_MEMBERS):
        raise RuntimeError("biological source reader did not follow the exact access order")
    return HestBiologicalData(
        source_path=reader.path,
        observation_ids=observation_ids,
        donor_ids=donor_ids,
        section_ids=section_ids,
        type_labels=type_labels,
        type_names=type_names,
        pool_roles=pool_roles,
        block_ids=block_ids,
        registration_quality_strata=registration,
        evaluation_indices=evaluation_indices,
        image_features=image_features,
        molecular_targets=evaluation_targets,
        reference_means=reference_means,
        technical_covariates=technical,
        control_families=evaluation_controls,
        control_feature_names=control_names,
        blank_feature=supplement.feature.astype(np.float32, copy=False),
        source_access={
            "registered_source_sha256": SOURCE_SHA256,
            "blank_supplement_sha256": supplement.sha256,
            "blank_feature_sha256": supplement.feature_sha256,
            "source_members_opened": list(reader.members_opened),
            "molecular_member": "nucleus_molecular_targets",
            "molecular_member_open_count": reader.molecular_member_open_count,
            "molecular_outcomes_read": True,
            "rows": SOURCE_ROWS,
            "evaluation_rows": EVALUATION_ROWS,
            "reference_rows": REFERENCE_ROWS,
        },
    )


__all__ = [
    "BIOLOGICAL_SOURCE_MEMBERS",
    "CROP_IDS",
    "EVALUATION_ROWS",
    "HestBiologicalData",
    "IMAGE_WIDTH",
    "NUISANCE_WIDTH",
    "PRIMARY_SUPPORT",
    "REFERENCE_ROWS",
    "SOURCE_ROWS",
    "TARGET_WIDTH",
    "TYPE_COUNT",
    "load_hest_biological_data",
]
