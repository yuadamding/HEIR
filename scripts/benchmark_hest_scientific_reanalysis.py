#!/usr/bin/env python3
"""Run the program-first, donor-nested HEST scientific reanalysis.

This is a retrospective, outcome-exposed analysis.  It can compare a frozen
H-optimus-1 qualification against the prespecified UNI2-h ridge probe, but it can never authorize
H-CELL, H-INTRINSIC, reference refinement, or full HEIR development.

The runner deliberately stages work: measurement reliability is calculated
before molecular models; observed donor-held-out effects are calculated before
expensive pairing nulls; and intrinsic claims fail closed when matched mask
artifact crops are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from heir.evaluation.hest_measurement import (
    feature_reliability_report,
    normalize_halves,
    ordered_program_scores,
    reference_residualize_halves,
    support_threshold_audit,
)
from heir.evaluation.hest_nested_ridge import (
    donor_section_type_row_weights,
    donor_type_row_weights,
    fit_weighted_pca,
    grouped_donor_folds,
    weighted_ridge_predict_grid,
)
from heir.evaluation.hest_scoring import (
    holm_adjust,
    multiclass_metrics,
    score_continuous_targets,
    summarize_paired_donor_effects,
)

SCHEMA = "heir.hest_scientific_reanalysis.v2"
SOURCE_SCHEMA = "heir.registered_observations_retrospective.v1"
REGISTERED_SOURCE_SHA256 = "57b77c7be2e30026a2da9ba0f9d5b205cf630f5d138942db6366e15cae2ef7a3"
UNI2H_REPOSITORY = "MahmoodLab/UNI2-h"
HOPTIMUS1_REPOSITORY = "bioptimus/H-optimus-1"
HOPTIMUS1_REVISION = "3592cb220dec7a150c5d7813fb56e68bd57473b9"
HOPTIMUS1_MANIFEST_SHA256 = "f6852288e1ae146a4865bf19e38ce994c0be9ce1c2bfa09bdf77747043ac8fd9"
HOPTIMUS1_CONFIG_SHA256 = "b10c4f37ce804ff58bec7f2ffd35cc29ecbfdb2b96ac81f3c1b3e37e2b27616e"
HOPTIMUS1_CHECKPOINT_SHA256 = "c4f1e5b457ddf00679626053b0bf2899be6a19c3a04ad191c87ad1cdfd1abfe1"
HOPTIMUS1_PARITY_RECEIPT_SHA256 = (
    "a67ca37feae12a3ca444399f12dc983de01283b05f14ffe16adfcdae80a4d761"
)
HOPTIMUS1_PARITY_IMPLEMENTATION_SHA256 = (
    "856a3521fa8388787c43bb8cdd8a8faa202c3d3fd980aeac661f134b8e0711d1"
)
HOPTIMUS1_PRODUCTION_RUNTIME_SHA256 = {
    "src/heir/features/__init__.py": (
        "d363e99327d1b77abd996dd02e943a4089d2584c93f93cded7f25e86f9b66d24"
    ),
    "src/heir/features/base.py": (
        "5b6d26bb4cb69fcd6454a8868f65699a3a287db1985fddd002ae854108014a86"
    ),
    "src/heir/features/hoptimus1.py": (
        "ffc3cf81bddc77e041cf5554fcd04424af627a8649d0d6752b04281ecfc6f20c"
    ),
}
ENCODER_DEPENDENT_SOURCE_FIELDS = frozenset(
    {
        "encoder_comparison_non_encoder_identity_sha256",
        "encoder_comparison_source_path",
        "encoder_comparison_source_sha256",
        "encoder_manifest_sha256",
        "encoder_name",
        "encoder_parity_receipt_path",
        "encoder_parity_receipt_sha256",
        "encoder_revision",
        "feature_checkpoint_sha256",
        "feature_config_sha256",
        "feature_space_id",
        "frozen_feature_names",
        "frozen_features",
        "image_features",
        "image_features_by_crop_and_encoder",
        "cellvit_nearest_frozen_features",
        "provenance_json",
    }
)
EXPECTED_DONORS = (
    "THD0008",
    "THD0011",
    "TILD117",
    "TILD175",
    "VUHD069",
    "VUHD116",
    "VUILD102",
    "VUILD105",
    "VUILD106",
    "VUILD107",
    "VUILD110",
    "VUILD115",
    "VUILD78",
    "VUILD91",
    "VUILD96",
)
EXPECTED_SECTIONS = (
    "NCBI856",
    "NCBI857",
    "NCBI858",
    "NCBI859",
    "NCBI860",
    "NCBI861",
    "NCBI864",
    "NCBI865",
    "NCBI866",
    "NCBI867",
    "NCBI870",
    "NCBI873",
    "NCBI875",
    "NCBI876",
    "NCBI879",
    "NCBI880",
    "NCBI881",
    "NCBI882",
    "NCBI883",
    "NCBI884",
)
REQUIRED_BASE_CROPS = (
    "crop_112um",
    "cell_mask_only",
    "nucleus_mask_only",
    "target_cell_removed_112um",
)
ARTIFACT_CONTROL_CROPS = (
    "nucleus_mask_mean_fill_112um",
    "nucleus_mask_blurred_112um",
    "nucleus_shape_random_location_mean_fill_112um",
    "cell_mask_mean_fill_112um",
    "cell_mask_blurred_112um",
    "cell_shape_random_location_mean_fill_112um",
)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SUPPORT_THRESHOLDS = (5, 10, 20, 30)
PRIMARY_SUPPORT = 20
PROGRAM_RELIABILITY_FLOOR = 0.20
GENE_RELIABILITY_FLOOR = 0.50
FROZEN_PROGRAM_NAMES = (
    "fibrotic_mesenchymal",
    "macrophage_inflammation",
    "epithelial_injury",
    "stress_hypoxia",
    "interferon_chemokine",
    "proliferation",
)
UNI2_BASELINE_GEOMETRY_TARGETS = (
    "nucleus_area_um2",
    "nucleus_perimeter_um",
    "nucleus_circularity",
    "nucleus_solidity",
)
UNI2_BASELINE_APPEARANCE_TARGETS = (
    "nucleus_gray_mean",
    "nucleus_hematoxylin_od_mean",
    "nucleus_glcm_contrast",
)
UNI2_BASELINE_AMENDMENT_TIMING = (
    "after_UNI2_visible_controls_before_any_Hoptimus_molecular_outcomes"
)
REQUIRED_PAIRED_ENCODER_REPRESENTATIONS = (
    "full_1536_broad_lineage_heads",
    "pca_256_broad_lineage_heads",
    "pca_512_broad_lineage_heads",
)
PAIRED_ENCODER_CONTRACT_FIELDS = (
    "schema",
    "analysis_status",
    "study_stage",
    "requested_phase",
    "donors",
    "sections",
    "encoder_feature_width",
    "crop_contract",
    "folding",
    "inner_folding",
    "inner_folds",
    "alpha_grid",
    "target_standardization",
    "pca_fit",
    "training_weighting",
    "primary_support",
    "support_sensitivities",
    "seed",
)
PAIRED_ENCODER_NUMERIC_FIELDS = (
    "requested_device",
    "torch_threads",
    "cuda_available",
    "ridge_cuda_dtype",
    "cpu_thread_environment",
    "gpu_name",
    "gpu_capability",
    "gpu_total_memory_bytes",
    "deterministic_algorithms_enabled",
    "cublas_workspace_config",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
)
PAIRED_IMPLEMENTATION_RUNTIME_FIELDS = (
    "python",
    "platform",
    "numpy",
    "torch",
    "torch_cuda_runtime",
)
PAIRED_OBSERVED_CONTRACT_FIELDS = (
    "target_scope",
    "endpoint",
    "evaluation_rows",
    "evaluation_mask_sha256",
    "program_outcome_sha256",
    "minimum_reference_and_scoring_support",
    "primary_crop",
    "program_names",
    "donors",
)


def _log(message: str) -> None:
    print(message, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _source_array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("identity-bound source arrays cannot contain Python objects")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_frozen_hoptimus1_parity_sidecar(
    path: Path,
    *,
    declared_sha256: str,
    declared_manifest_sha256: str,
) -> Path:
    resolved = path.expanduser().resolve()
    if (
        declared_sha256 != HOPTIMUS1_PARITY_RECEIPT_SHA256
        or not resolved.is_file()
        or _sha256(resolved) != HOPTIMUS1_PARITY_RECEIPT_SHA256
    ):
        raise ValueError("H-optimus-1 source does not bind the frozen real parity sidecar")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("H-optimus-1 parity sidecar is unreadable") from error
    model = value.get("model") if isinstance(value, Mapping) else None
    comparisons = value.get("comparisons") if isinstance(value, Mapping) else None
    runtime = value.get("production_runtime_contract") if isinstance(value, Mapping) else None
    probe = runtime.get("resampling_probe") if isinstance(runtime, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "heir.hoptimus1_official_local_parity.v1"
        or value.get("status") != "passed"
        or value.get("passed") is not True
        or value.get("repository") != HOPTIMUS1_REPOSITORY
        or value.get("revision") != HOPTIMUS1_REVISION
        or value.get("encoder_manifest_sha256") != HOPTIMUS1_MANIFEST_SHA256
        or value.get("implementation_sha256") != HOPTIMUS1_PARITY_IMPLEMENTATION_SHA256
        or declared_manifest_sha256 != HOPTIMUS1_MANIFEST_SHA256
        or not isinstance(model, Mapping)
        or model.get("checkpoint_sha256") != HOPTIMUS1_CHECKPOINT_SHA256
        or model.get("config_sha256") != HOPTIMUS1_CONFIG_SHA256
        or model.get("local_fp16_path") != "HOptimus1Encoder.encode_exact_biological_path"
        or not isinstance(comparisons, Mapping)
        or not isinstance(runtime, Mapping)
        or runtime.get("code_sha256") != HOPTIMUS1_PRODUCTION_RUNTIME_SHA256
        or not isinstance(probe, Mapping)
        or probe.get("input_shape") != [527, 527, 3]
        or probe.get("output_shape") != [224, 224, 3]
        or probe.get("implementation") != "Pillow.Image.Resampling.BICUBIC"
        or probe.get("resampling_count") != 1
    ):
        raise ValueError("H-optimus-1 parity sidecar content differs from the frozen contract")
    for name, threshold in (
        ("official_fp32_vs_local_fp32", 0.999999),
        ("local_fp32_vs_local_fp16", 0.9999),
    ):
        row = comparisons.get(name)
        if (
            not isinstance(row, Mapping)
            or row.get("passed") is not True
            or float(row.get("minimum_required_cosine", float("nan"))) != threshold
            or float(row.get("minimum_cosine", float("nan"))) < threshold
        ):
            raise ValueError("H-optimus-1 parity sidecar metrics failed the frozen thresholds")
    return resolved


def _validate_only_encoder_changed_sources(
    qualification_source: Path,
    comparison_source: Path,
    *,
    declared_comparison_sha256: str,
    declared_identity_sha256: str,
) -> Mapping[str, object]:
    comparison_source = comparison_source.expanduser().resolve()
    if (
        declared_comparison_sha256 != REGISTERED_SOURCE_SHA256
        or not comparison_source.is_file()
        or _sha256(comparison_source) != REGISTERED_SOURCE_SHA256
    ):
        raise ValueError("H-optimus-1 qualification comparison is not registered UNI2-h HEST")
    with (
        np.load(qualification_source, allow_pickle=False) as candidate,
        np.load(comparison_source, allow_pickle=False) as comparison,
    ):
        candidate_fields = set(candidate.files) - ENCODER_DEPENDENT_SOURCE_FIELDS
        comparison_fields = set(comparison.files) - ENCODER_DEPENDENT_SOURCE_FIELDS
        if candidate_fields != comparison_fields:
            raise ValueError("H-optimus-1 and UNI2-h non-encoder source field sets differ")
        field_hashes = {}
        for field in sorted(candidate_fields):
            candidate_sha256 = _source_array_sha256(candidate[field])
            comparison_sha256 = _source_array_sha256(comparison[field])
            if candidate_sha256 != comparison_sha256:
                raise ValueError("H-optimus-1 qualification changed non-encoder field: " + field)
            field_hashes[field] = candidate_sha256
    identity_sha256 = _canonical_sha256(field_hashes)
    if declared_identity_sha256 != identity_sha256:
        raise ValueError("H-optimus-1 source non-encoder identity receipt is inconsistent")
    return {
        "comparison_source_path": str(comparison_source),
        "comparison_source_sha256": REGISTERED_SOURCE_SHA256,
        "compared_field_count": len(field_hashes),
        "field_hash_manifest_sha256": identity_sha256,
        "only_encoder_changed": True,
    }


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _array(archive: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in archive.files:
            return np.asarray(archive[name])
    raise ValueError("source is missing one of: " + ", ".join(names))


def _optional_array(archive: np.lib.npyio.NpzFile, *names: str) -> Optional[np.ndarray]:
    for name in names:
        if name in archive.files:
            return np.asarray(archive[name])
    return None


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    return np.asarray(_array(archive, name)).reshape(()).item()


def _merge_unique(
    parts: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    seen: set[str] = set()
    rows = len(parts[0][0])
    for values, local_names in parts:
        matrix = np.asarray(values, dtype=np.float32)
        local = tuple(np.asarray(local_names).astype(str).tolist())
        if matrix.ndim != 2 or len(matrix) != rows or matrix.shape[1] != len(local):
            raise ValueError("control values and names are not aligned")
        for index, name in enumerate(local):
            if name not in seen:
                seen.add(name)
                names.append(name)
                columns.append(matrix[:, index])
    if not columns:
        raise ValueError("control family is empty")
    return np.column_stack(columns).astype(np.float32), tuple(names)


def _load_controls(
    archive: np.lib.npyio.NpzFile,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    technical = _array(archive, "technical_covariates")
    technical_names = _array(archive, "technical_covariate_names")
    coordinates = _array(archive, "coordinate_features")
    coordinate_names = _array(archive, "coordinate_feature_names").astype(str)
    density = _array(archive, "local_density_features")
    density_names = _array(archive, "local_density_feature_names").astype(str)
    density_set = set(density_names.tolist())
    spatial_indices = [
        index
        for index, name in enumerate(coordinate_names.tolist())
        if name not in density_set and "background" not in name
    ]
    spatial = coordinates[:, spatial_indices]
    spatial_names = coordinate_names[spatial_indices]
    stain = _array(archive, "stain_quality_features", "stain_features")
    stain_names = _array(archive, "stain_quality_feature_names", "stain_feature_names")
    nucleus = _array(archive, "nuclear_morphometric_features", "nucleus_geometry_features")
    nucleus_names = _array(
        archive,
        "nuclear_morphometric_feature_names",
        "nucleus_geometry_feature_names",
    )
    cell = _array(archive, "cell_morphometric_features", "cell_geometry_features")
    cell_names = _array(archive, "cell_morphometric_feature_names", "cell_geometry_feature_names")

    specifications = {
        "metadata_technical": ((technical, technical_names),),
        "spatial": ((spatial, spatial_names),),
        "technical_spatial": (
            (technical, technical_names),
            (spatial, spatial_names),
        ),
        "handcrafted_image": (
            (stain, stain_names),
            (nucleus, nucleus_names),
            (cell, cell_names),
            (density, density_names),
        ),
        "all_controls": (
            (technical, technical_names),
            (spatial, spatial_names),
            (stain, stain_names),
            (nucleus, nucleus_names),
            (cell, cell_names),
            (density, density_names),
        ),
    }
    matrices: dict[str, np.ndarray] = {}
    names: dict[str, tuple[str, ...]] = {}
    for family, parts in specifications.items():
        matrices[family], names[family] = _merge_unique(parts)
    return matrices, names


@dataclass
class HestSource:
    path: Path
    sha256: str
    encoder_name: str
    encoder_revision: str
    encoder_manifest_sha256: str
    encoder_parity_receipt_sha256: str
    encoder_parity_receipt_path: str
    encoder_comparison_source_sha256: str
    encoder_comparison_non_encoder_identity_sha256: str
    encoder_comparison_receipt: Mapping[str, object] | None
    donors: np.ndarray
    sections: np.ndarray
    fine_types: np.ndarray
    broad_types: np.ndarray
    roles: np.ndarray
    roles_by_split: np.ndarray
    split_ids: np.ndarray
    crop_ids: tuple[str, ...]
    crop_roles: tuple[str, ...]
    crop_mask_modes: tuple[str, ...]
    crop_fill_modes: tuple[str, ...]
    images: np.ndarray
    controls: dict[str, np.ndarray]
    control_names: dict[str, tuple[str, ...]]
    gene_ids: tuple[str, ...]
    program_names: tuple[str, ...]
    program_membership: np.ndarray
    nucleus_targets: np.ndarray
    whole_cell_targets: np.ndarray
    nucleus_counts_half_a: np.ndarray
    nucleus_counts_half_b: np.ndarray
    nucleus_library_half_a: np.ndarray
    nucleus_library_half_b: np.ndarray
    whole_cell_counts_half_a: np.ndarray
    whole_cell_counts_half_b: np.ndarray
    whole_cell_library_half_a: np.ndarray
    whole_cell_library_half_b: np.ndarray
    nuclear_morphometrics: np.ndarray
    nuclear_morphometric_names: tuple[str, ...]


def load_source(
    path: Path,
    *,
    enforce_registered_hash: bool = True,
    expected_source_sha256: str | None = None,
    expected_encoder: str = UNI2H_REPOSITORY,
    comparison_source_path: Path | None = None,
) -> HestSource:
    path = path.expanduser().resolve()
    source_hash = _sha256(path)
    registered_hash = expected_source_sha256 or REGISTERED_SOURCE_SHA256
    if enforce_registered_hash and source_hash != registered_hash:
        raise ValueError("source does not match the registered retrospective HEST receipt")
    _log(f"HEST reanalysis: loading receipt-bound source {path}")
    with np.load(path, allow_pickle=False) as archive:
        identity = {
            "schema": str(_scalar(archive, "schema_version")),
            "stage": str(_scalar(archive, "study_stage")),
            "status": str(_scalar(archive, "analysis_status")),
            "authorizes_h_cell": bool(_scalar(archive, "authorizes_h_cell")),
            "authorizes_h_intrinsic": bool(_scalar(archive, "authorizes_h_intrinsic")),
            "authorizes_full_heir": bool(_scalar(archive, "authorizes_full_heir")),
            "encoder": str(_scalar(archive, "encoder_name")),
        }
        expected = {
            "schema": SOURCE_SCHEMA,
            "stage": "retrospective_exposed",
            "status": "retrospective_exposed_non_authorizing",
            "authorizes_h_cell": False,
            "authorizes_h_intrinsic": False,
            "authorizes_full_heir": False,
            "encoder": expected_encoder,
        }
        if identity != expected:
            raise ValueError("source identity is not the requested exposed HEST encoder artifact")
        encoder_revision = str(_scalar(archive, "encoder_revision"))
        encoder_manifest_sha256 = str(_scalar(archive, "encoder_manifest_sha256"))
        encoder_parity_receipt_sha256 = (
            str(_scalar(archive, "encoder_parity_receipt_sha256"))
            if "encoder_parity_receipt_sha256" in archive.files
            else ""
        )
        encoder_parity_receipt_path = (
            str(_scalar(archive, "encoder_parity_receipt_path"))
            if "encoder_parity_receipt_path" in archive.files
            else ""
        )
        encoder_comparison_source_sha256 = (
            str(_scalar(archive, "encoder_comparison_source_sha256"))
            if "encoder_comparison_source_sha256" in archive.files
            else ""
        )
        encoder_comparison_non_encoder_identity_sha256 = (
            str(_scalar(archive, "encoder_comparison_non_encoder_identity_sha256"))
            if "encoder_comparison_non_encoder_identity_sha256" in archive.files
            else ""
        )
        feature_checkpoint_sha256 = str(_scalar(archive, "feature_checkpoint_sha256"))
        feature_config_sha256 = str(_scalar(archive, "feature_config_sha256"))
        if expected_encoder not in {UNI2H_REPOSITORY, HOPTIMUS1_REPOSITORY}:
            raise ValueError("requested HEST qualification encoder is unsupported")
        if expected_encoder == HOPTIMUS1_REPOSITORY:
            if (
                encoder_revision != HOPTIMUS1_REVISION
                or encoder_manifest_sha256 != HOPTIMUS1_MANIFEST_SHA256
                or feature_checkpoint_sha256 != HOPTIMUS1_CHECKPOINT_SHA256
                or feature_config_sha256 != HOPTIMUS1_CONFIG_SHA256
            ):
                raise ValueError("H-optimus-1 qualification source identity is not pinned")
            _validate_frozen_hoptimus1_parity_sidecar(
                Path(encoder_parity_receipt_path),
                declared_sha256=encoder_parity_receipt_sha256,
                declared_manifest_sha256=encoder_manifest_sha256,
            )
            if comparison_source_path is None:
                raise ValueError("H-optimus-1 qualification requires --comparison-source")
        elif comparison_source_path is not None:
            raise ValueError("comparison source is accepted only for H-optimus-1 qualification")
        donors = _array(archive, "donor_ids", "donor_id").astype(str)
        sections = _array(archive, "section_ids", "section_id").astype(str)
        fine_types = _array(archive, "fine_type_ids", "fine_type").astype(str)
        broad_names = _array(archive, "broad_type_names").astype(str)
        broad_labels = _array(archive, "broad_type_labels", "broad_type_label").astype(int)
        broad_types = broad_names[broad_labels]
        roles = _array(archive, "pool_roles", "pool_role").astype(str)
        roles_by_split = _array(archive, "pool_roles_by_split").astype(str)
        split_ids = _array(archive, "reference_split_ids").astype(str)
        crop_ids = tuple(_array(archive, "crop_ids").astype(str).tolist())
        crop_roles = tuple(_array(archive, "crop_roles", "crop_role").astype(str).tolist())
        crop_mask_modes = tuple(_array(archive, "crop_mask_modes").astype(str).tolist())
        crop_fill_modes = tuple(_array(archive, "crop_fill_modes").astype(str).tolist())
        images = _array(archive, "image_features_by_crop_and_encoder", "image_features").astype(
            np.float32, copy=False
        )
        controls, control_names = _load_controls(archive)
        gene_ids = tuple(_array(archive, "gene_ids").astype(str).tolist())
        program_names = tuple(_array(archive, "program_names").astype(str).tolist())
        program_membership = _array(archive, "program_gene_membership").astype(bool)
        nucleus_targets = _array(archive, "nucleus_molecular_targets").astype(np.float32)
        whole_cell_targets = _array(archive, "whole_cell_molecular_targets").astype(np.float32)
        nucleus_counts_half_a = _array(archive, "nucleus_target_counts_half_a").astype(np.uint32)
        nucleus_counts_half_b = _array(archive, "nucleus_target_counts_half_b").astype(np.uint32)
        nucleus_library_half_a = _array(archive, "nucleus_library_size_half_a").astype(np.float64)
        nucleus_library_half_b = _array(archive, "nucleus_library_size_half_b").astype(np.float64)
        whole_cell_counts_half_a = _array(archive, "whole_cell_target_counts_half_a").astype(
            np.uint32
        )
        whole_cell_counts_half_b = _array(archive, "whole_cell_target_counts_half_b").astype(
            np.uint32
        )
        whole_cell_library_half_a = _array(archive, "whole_cell_library_size_half_a").astype(
            np.float64
        )
        whole_cell_library_half_b = _array(archive, "whole_cell_library_size_half_b").astype(
            np.float64
        )
        nuclear_morphometrics = _array(
            archive, "nuclear_morphometric_features", "nucleus_geometry_features"
        ).astype(np.float32)
        nuclear_morphometric_names = tuple(
            _array(
                archive,
                "nuclear_morphometric_feature_names",
                "nucleus_geometry_feature_names",
            )
            .astype(str)
            .tolist()
        )

    rows = len(donors)
    row_vectors = (sections, fine_types, broad_types, roles)
    if any(values.shape != (rows,) for values in row_vectors):
        raise ValueError("source row identifiers disagree")
    if set(donors.tolist()) != set(EXPECTED_DONORS):
        raise ValueError("reanalysis requires the exact 15 biological donors")
    if set(sections.tolist()) != set(EXPECTED_SECTIONS):
        raise ValueError("reanalysis requires the exact 20 HEST sections")
    if tuple(crop_ids[:4]) != REQUIRED_BASE_CROPS:
        raise ValueError("source base crop order differs from its registered contract")
    if images.shape != (rows, len(crop_ids), 1536):
        raise ValueError("source frozen-encoder tensor is malformed")
    if roles_by_split.shape != (rows, len(split_ids)):
        raise ValueError("reference-split matrix is malformed")
    if len(gene_ids) != 260 or program_membership.shape != (len(program_names), 260):
        raise ValueError("frozen molecular panel/program definitions are malformed")
    if any(not np.isfinite(values).all() for values in controls.values()):
        raise ValueError("control features contain non-finite values")
    comparison_receipt = None
    if expected_encoder == HOPTIMUS1_REPOSITORY:
        assert comparison_source_path is not None
        comparison_receipt = _validate_only_encoder_changed_sources(
            path,
            comparison_source_path,
            declared_comparison_sha256=encoder_comparison_source_sha256,
            declared_identity_sha256=encoder_comparison_non_encoder_identity_sha256,
        )
    return HestSource(
        path,
        source_hash,
        expected_encoder,
        encoder_revision,
        encoder_manifest_sha256,
        encoder_parity_receipt_sha256,
        encoder_parity_receipt_path,
        encoder_comparison_source_sha256,
        encoder_comparison_non_encoder_identity_sha256,
        comparison_receipt,
        donors,
        sections,
        fine_types,
        broad_types,
        roles,
        roles_by_split,
        split_ids,
        crop_ids,
        crop_roles,
        crop_mask_modes,
        crop_fill_modes,
        images,
        controls,
        control_names,
        gene_ids,
        program_names,
        program_membership,
        nucleus_targets,
        whole_cell_targets,
        nucleus_counts_half_a,
        nucleus_counts_half_b,
        nucleus_library_half_a,
        nucleus_library_half_b,
        whole_cell_counts_half_a,
        whole_cell_counts_half_b,
        whole_cell_library_half_a,
        whole_cell_library_half_b,
        nuclear_morphometrics,
        nuclear_morphometric_names,
    )


def _programs(values: np.ndarray, source: HestSource) -> np.ndarray:
    return ordered_program_scores(values, source.program_names, source.program_membership)


def _residualize_values(
    values: np.ndarray,
    source: HestSource,
    roles: np.ndarray,
    *,
    minimum_support: int,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = reference_residualize_halves(
        values,
        values,
        source.donors,
        source.sections,
        source.fine_types,
        roles,
        minimum_support=minimum_support,
    )
    return residuals.half_a[residuals.evaluation_mask], residuals.evaluation_mask


def _measurement_scope(
    source: HestSource,
    *,
    target_scope: str,
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    library_a: np.ndarray,
    library_b: np.ndarray,
) -> Mapping[str, object]:
    _log(f"HEST measurement: scope={target_scope} normalize split halves")
    half_a, half_b = normalize_halves(
        counts_a,
        counts_b,
        library_sizes_half_a=library_a,
        library_sizes_half_b=library_b,
    )
    residuals = reference_residualize_halves(
        half_a,
        half_b,
        source.donors,
        source.sections,
        source.fine_types,
        source.roles,
        minimum_support=PRIMARY_SUPPORT,
    )
    program_half_a = _programs(residuals.half_a, source)
    program_half_b = _programs(residuals.half_b, source)
    gene_report = feature_reliability_report(
        residuals.half_a,
        residuals.half_b,
        source.gene_ids,
        source.donors,
        source.fine_types,
        evaluation_mask=residuals.evaluation_mask,
        minimum_rows=PRIMARY_SUPPORT,
    )
    program_report = feature_reliability_report(
        program_half_a,
        program_half_b,
        source.program_names,
        source.donors,
        source.fine_types,
        evaluation_mask=residuals.evaluation_mask,
        minimum_rows=PRIMARY_SUPPORT,
    )
    gene_macro = gene_report["donor_type_macro"]["features"]
    reliable_genes = [
        gene
        for gene in source.gene_ids
        if gene_macro[gene]["median_spearman_brown_reliability"] is not None
        and gene_macro[gene]["median_spearman_brown_reliability"] >= GENE_RELIABILITY_FLOOR
    ]
    program_macro = program_report["donor_type_macro"]["features"]
    measurable_programs = [
        name
        for name in source.program_names
        if program_macro[name]["median_spearman_brown_reliability"] is not None
        and program_macro[name]["median_spearman_brown_reliability"] >= PROGRAM_RELIABILITY_FLOOR
    ]
    outer_training_reliability: dict[str, Mapping[str, object]] = {}
    eligible_outer_fold_counts = {name: 0 for name in source.program_names}
    heldout_donors = sorted(set(source.donors.tolist()))
    for heldout in heldout_donors:
        training = source.donors != heldout
        training_residuals = reference_residualize_halves(
            half_a[training],
            half_b[training],
            source.donors[training],
            source.sections[training],
            source.fine_types[training],
            source.roles[training],
            minimum_support=PRIMARY_SUPPORT,
        )
        training_program_a = _programs(training_residuals.half_a, source)
        training_program_b = _programs(training_residuals.half_b, source)
        training_report = feature_reliability_report(
            training_program_a,
            training_program_b,
            source.program_names,
            source.donors[training],
            source.fine_types[training],
            evaluation_mask=training_residuals.evaluation_mask,
            minimum_rows=PRIMARY_SUPPORT,
        )
        training_features = training_report["donor_type_macro"]["features"]
        fold_programs = {}
        for name in source.program_names:
            reliability = training_features[name]["median_spearman_brown_reliability"]
            eligible = bool(reliability is not None and reliability >= PROGRAM_RELIABILITY_FLOOR)
            eligible_outer_fold_counts[name] += int(eligible)
            fold_programs[name] = {
                "median_spearman_brown_reliability": reliability,
                "meets_floor": eligible,
            }
        outer_training_reliability[heldout] = {
            "heldout_donor": heldout,
            "training_donors": sorted(set(source.donors[training].tolist())),
            "eligible_evaluation_rows": int(training_residuals.evaluation_mask.sum()),
            "programs": fold_programs,
            "heldout_donor_measurements_used": False,
        }
    primary_reliable_programs = [
        name
        for name in source.program_names
        if eligible_outer_fold_counts[name] == len(heldout_donors)
    ]
    return {
        "target_scope": target_scope,
        "normalization": "independent_half_library_log1p_cpm_10000",
        "endpoint": "same_donor_section_fine_type_reference_residual",
        "minimum_reference_and_evaluation_support": PRIMARY_SUPPORT,
        "eligible_evaluation_rows": int(residuals.evaluation_mask.sum()),
        "genes": gene_report,
        "programs": program_report,
        "selection_diagnostic_only": {
            "gene_reliability_floor": GENE_RELIABILITY_FLOOR,
            "genes_meeting_global_macro_floor": reliable_genes,
            "gene_count": len(reliable_genes),
            "program_reliability_floor": PROGRAM_RELIABILITY_FLOOR,
            "programs_meeting_global_macro_floor": measurable_programs,
            "program_count": len(measurable_programs),
            "heldout_outcomes_used_for_model_target_selection": False,
            "note": (
                "Global ranks are descriptive only; program definitions stay frozen and "
                "gene-model selection, if run, must be repeated inside each outer fold."
            ),
        },
        "outer_training_program_reliability_gate": {
            "program_reliability_floor": PROGRAM_RELIABILITY_FLOOR,
            "rule": "meets_floor_in_every_outer_training_donor_partition",
            "programs_eligible_for_primary_inference": primary_reliable_programs,
            "eligible_outer_fold_counts": eligible_outer_fold_counts,
            "outer_fold_count": len(heldout_donors),
            "per_heldout_donor": outer_training_reliability,
            "fixed_six_program_models_still_reported_as_sensitivity": True,
        },
    }


def measurement_analysis(source: HestSource) -> Mapping[str, object]:
    return {
        "support": support_threshold_audit(
            source.donors,
            source.sections,
            source.fine_types,
            source.roles,
            thresholds=SUPPORT_THRESHOLDS,
        ),
        "nucleus_overlap": _measurement_scope(
            source,
            target_scope="nucleus_overlap",
            counts_a=source.nucleus_counts_half_a,
            counts_b=source.nucleus_counts_half_b,
            library_a=source.nucleus_library_half_a,
            library_b=source.nucleus_library_half_b,
        ),
        "whole_cell_assignment_sensitivity": _measurement_scope(
            source,
            target_scope="whole_cell_assignment",
            counts_a=source.whole_cell_counts_half_a,
            counts_b=source.whole_cell_counts_half_b,
            library_a=source.whole_cell_library_half_a,
            library_b=source.whole_cell_library_half_b,
        ),
    }


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    kind: str
    dimension: int
    seed: Optional[int] = None


@dataclass
class ControlPlan:
    predictions: dict[str, np.ndarray]
    outer_folds: dict[str, Mapping[str, object]]
    scores: dict[str, Mapping[str, object]]


def _row_weights(
    donors: np.ndarray,
    sections: np.ndarray,
    fine_types: np.ndarray,
    weighting: str,
) -> np.ndarray:
    if weighting == "donor_type":
        return donor_type_row_weights(donors, fine_types)
    if weighting == "donor_section_type":
        return donor_section_type_row_weights(donors, sections, fine_types)
    raise ValueError("unknown training weighting")


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        _log("HEST reanalysis: CUDA unavailable; using CPU")
        return "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return requested


def _fixed_random_projection(
    values: np.ndarray,
    dimension: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    projection = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(values.shape[1], dimension),
    ) / np.sqrt(float(dimension))
    receipt = hashlib.sha256(np.ascontiguousarray(projection).view(np.uint8)).hexdigest()
    if device == "cuda":
        with torch.inference_mode():
            x = torch.as_tensor(values, dtype=torch.float32, device="cuda")
            p = torch.as_tensor(projection, dtype=torch.float32, device="cuda")
            result = (x @ p).cpu().numpy()
            del x, p
            torch.cuda.empty_cache()
    else:
        result = np.asarray(values, dtype=np.float32) @ projection
    return np.asarray(result, dtype=np.float32), receipt


def _prepare_representation(
    raw_or_projected: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    spec: RepresentationSpec,
    donors: np.ndarray,
    sections: np.ndarray,
    fine_types: np.ndarray,
    weighting: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    if spec.kind in {"full", "random_projection"}:
        return (
            raw_or_projected[train],
            raw_or_projected[test],
            {
                "fit_partition": "fixed_outcome_free"
                if spec.kind == "random_projection"
                else "none",
                "heldout_rows_used_in_fit": False,
            },
        )
    if spec.kind != "pca":
        raise ValueError("unknown image representation")
    weights = _row_weights(donors[train], sections[train], fine_types[train], weighting)
    fitted = fit_weighted_pca(
        raw_or_projected[train],
        spec.dimension,
        weights,
        device=device,
    )
    explained = fitted.explained_variance
    return (
        fitted.transform(raw_or_projected[train]).astype(np.float32),
        fitted.transform(raw_or_projected[test]).astype(np.float32),
        {
            "fit_partition": "current_training_donors_only",
            "heldout_rows_used_in_fit": False,
            "fit_device": fitted.fit_device,
            "component_count": spec.dimension,
            "retained_eigenvalue_sum": float(explained.sum()),
        },
    )


def _architecture_predict_grid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    fine_types: np.ndarray,
    broad_types: np.ndarray,
    *,
    architecture: str,
    weighting: str,
    alphas: Sequence[float],
    device: str,
) -> np.ndarray:
    if architecture == "pooled":
        weights = _row_weights(
            donors[train_indices],
            sections[train_indices],
            fine_types[train_indices],
            weighting,
        )
        return weighted_ridge_predict_grid(
            x_train,
            y_train,
            x_test,
            alphas,
            weights,
            device=device,
        )
    if architecture != "broad_lineage_heads":
        raise ValueError("unknown model architecture")
    predictions = np.full(
        (len(alphas), len(test_indices), y_train.shape[1]),
        np.nan,
        dtype=np.float64,
    )
    train_broad = broad_types[train_indices]
    test_broad = broad_types[test_indices]
    for lineage in sorted(set(test_broad.tolist())):
        local_train = train_broad == lineage
        local_test = test_broad == lineage
        if not local_train.any() or len(set(donors[train_indices][local_train])) < 2:
            raise ValueError(f"broad-lineage head lacks training donors: {lineage}")
        weights = _row_weights(
            donors[train_indices][local_train],
            sections[train_indices][local_train],
            fine_types[train_indices][local_train],
            weighting,
        )
        predictions[:, local_test] = weighted_ridge_predict_grid(
            x_train[local_train],
            y_train[local_train],
            x_test[local_test],
            alphas,
            weights,
            device=device,
        )
    if not np.isfinite(predictions).all():
        raise ValueError("a broad-lineage head failed to predict held-out rows")
    return predictions


def _grid_scores(
    truth: np.ndarray,
    predictions: np.ndarray,
    indices: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    fine_types: np.ndarray,
    target_names: Sequence[str],
    minimum_support: int,
) -> np.ndarray:
    result = np.full((len(predictions), truth.shape[1]), -np.inf, dtype=np.float64)
    reference = np.zeros_like(truth)
    for alpha_index, prediction in enumerate(predictions):
        score = score_continuous_targets(
            truth,
            prediction,
            reference,
            donors[indices],
            sections[indices],
            fine_types[indices],
            target_names=target_names,
            minimum_support=minimum_support,
        )
        for target_index, target in enumerate(target_names):
            value = score["targets"][target]["donor_type_macro_r2"]
            if value is not None:
                result[alpha_index, target_index] = float(value)
    return result


def _selected_alpha_indices(fold_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(fold_scores)
        counts = finite.sum(axis=0)
        sums = np.where(finite, fold_scores, 0.0).sum(axis=0)
        means = np.divide(
            sums,
            counts,
            out=np.full(fold_scores.shape[1:], -np.inf, dtype=np.float64),
            where=counts > 0,
        )
    indices = np.argmax(means, axis=0)
    selected = means[indices, np.arange(means.shape[1])]
    if not np.isfinite(selected).all():
        raise ValueError("nested donor CV could not score every target")
    return indices, selected


def _take_target_alphas(prediction_grid: np.ndarray, alpha_indices: np.ndarray) -> np.ndarray:
    result = np.empty(prediction_grid.shape[1:], dtype=np.float64)
    for target_index, alpha_index in enumerate(alpha_indices.tolist()):
        result[:, target_index] = prediction_grid[alpha_index, :, target_index]
    return result


def _fit_control_plan(
    y: np.ndarray,
    controls: Mapping[str, np.ndarray],
    donors: np.ndarray,
    sections: np.ndarray,
    fine_types: np.ndarray,
    broad_types: np.ndarray,
    target_names: Sequence[str],
    *,
    architecture: str,
    weighting: str,
    inner_folds: int,
    seed: int,
    device: str,
    minimum_support: int = PRIMARY_SUPPORT,
) -> ControlPlan:
    families = tuple(sorted(controls))
    predictions = {family: np.full_like(y, np.nan, dtype=np.float64) for family in families}
    predictions["selected_best_control"] = np.full_like(y, np.nan, dtype=np.float64)
    predictions["reference_mean"] = np.zeros_like(y, dtype=np.float64)
    outer_reports: dict[str, Mapping[str, object]] = {}
    all_rows = np.arange(len(y), dtype=np.int64)
    for outer_index, heldout in enumerate(sorted(set(donors.tolist()))):
        outer_train = all_rows[donors != heldout]
        outer_test = all_rows[donors == heldout]
        local_folds = grouped_donor_folds(
            donors[outer_train], n_splits=inner_folds, seed=seed + outer_index
        )
        family_fold_scores = {
            family: np.full(
                (len(local_folds), len(ALPHAS), y.shape[1]),
                np.nan,
                dtype=np.float64,
            )
            for family in families
        }
        for fold_index, (local_train, local_validation) in enumerate(local_folds):
            train = outer_train[local_train]
            validation = outer_train[local_validation]
            for family in families:
                grid = _architecture_predict_grid(
                    controls[family][train],
                    y[train],
                    controls[family][validation],
                    train,
                    validation,
                    donors,
                    sections,
                    fine_types,
                    broad_types,
                    architecture=architecture,
                    weighting=weighting,
                    alphas=ALPHAS,
                    device=device,
                )
                family_fold_scores[family][fold_index] = _grid_scores(
                    y[validation],
                    grid,
                    validation,
                    donors,
                    sections,
                    fine_types,
                    target_names,
                    minimum_support=minimum_support,
                )
        selected_indices: dict[str, np.ndarray] = {}
        selected_scores: dict[str, np.ndarray] = {}
        for family in families:
            indices, scores = _selected_alpha_indices(family_fold_scores[family])
            selected_indices[family] = indices
            selected_scores[family] = scores
        selected_family_by_target = {
            target: sorted(
                families,
                key=lambda family: (
                    -float(selected_scores[family][target_index]),
                    family,
                ),
            )[0]
            for target_index, target in enumerate(target_names)
        }
        for family in families:
            grid = _architecture_predict_grid(
                controls[family][outer_train],
                y[outer_train],
                controls[family][outer_test],
                outer_train,
                outer_test,
                donors,
                sections,
                fine_types,
                broad_types,
                architecture=architecture,
                weighting=weighting,
                alphas=ALPHAS,
                device=device,
            )
            predictions[family][outer_test] = _take_target_alphas(grid, selected_indices[family])
        for target_index, target in enumerate(target_names):
            selected_family = selected_family_by_target[target]
            predictions["selected_best_control"][outer_test, target_index] = predictions[
                selected_family
            ][outer_test, target_index]
        outer_reports[heldout] = {
            "training_donors": sorted(set(donors[outer_train].tolist())),
            "heldout_donor": heldout,
            "selected_control_family_by_target": selected_family_by_target,
            "control_family_inner_scores": {
                family: {
                    target: float(selected_scores[family][target_index])
                    for target_index, target in enumerate(target_names)
                }
                for family in families
            },
            "selected_alphas": {
                family: {
                    target: float(ALPHAS[selected_indices[family][target_index]])
                    for target_index, target in enumerate(target_names)
                }
                for family in families
            },
            "selection_partition": "inner_training_donors_only",
            "heldout_outcomes_used_for_selection": False,
            "inner_fold_seed": seed + outer_index,
            "inner_validation_donors": [
                sorted(set(donors[outer_train][validation].tolist()))
                for _training, validation in local_folds
            ],
        }
        _log(
            "HEST controls: architecture=%s heldout=%s selected=%s"
            % (
                architecture,
                heldout,
                ",".join(sorted(set(selected_family_by_target.values()))),
            )
        )
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("control plan left non-finite held-out predictions")
    scores = _score_models(
        y,
        predictions,
        donors,
        sections,
        fine_types,
        target_names,
        minimum_support=minimum_support,
    )
    return ControlPlan(predictions, outer_reports, scores)


def _score_models(
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    donors: np.ndarray,
    sections: np.ndarray,
    fine_types: np.ndarray,
    target_names: Sequence[str],
    *,
    minimum_support: int,
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    reference = np.zeros_like(truth)
    for model, prediction in predictions.items():
        result[model] = {
            str(minimum_support): score_continuous_targets(
                truth,
                prediction,
                reference,
                donors,
                sections,
                fine_types,
                target_names=target_names,
                minimum_support=minimum_support,
            )
        }
    return result


def _combined_features(
    image_values: np.ndarray,
    control_values: Optional[np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    if control_values is None:
        return image_values
    return np.column_stack((control_values[indices], image_values)).astype(np.float32, copy=False)


def _fit_image_plan(
    raw_image: np.ndarray,
    spec: RepresentationSpec,
    y: np.ndarray,
    controls: Mapping[str, np.ndarray],
    control_plan: ControlPlan,
    donors: np.ndarray,
    sections: np.ndarray,
    fine_types: np.ndarray,
    broad_types: np.ndarray,
    target_names: Sequence[str],
    *,
    architecture: str,
    weighting: str,
    inner_folds: int,
    seed: int,
    device: str,
    minimum_support: int = PRIMARY_SUPPORT,
) -> Mapping[str, object]:
    image_source = raw_image
    projection_receipt = None
    if spec.kind == "random_projection":
        if spec.seed is None:
            raise ValueError("random projection requires a frozen seed")
        image_source, projection_receipt = _fixed_random_projection(
            raw_image, spec.dimension, spec.seed, device
        )
    predictions = {
        model: np.full_like(y, np.nan, dtype=np.float64)
        for model in (
            "foundation_image_only",
            "metadata_technical_plus_foundation_image",
            "selected_best_control_plus_foundation_image",
            "all_controls_plus_foundation_image",
        )
    }
    outer_reports: dict[str, Mapping[str, object]] = {}
    all_rows = np.arange(len(y), dtype=np.int64)
    for outer_index, heldout in enumerate(sorted(set(donors.tolist()))):
        outer_train = all_rows[donors != heldout]
        outer_test = all_rows[donors == heldout]
        selected_control_by_target = {
            str(target): str(family)
            for target, family in control_plan.outer_folds[heldout][
                "selected_control_family_by_target"
            ].items()
        }
        control_by_model: dict[str, Optional[np.ndarray]] = {
            "foundation_image_only": None,
            "metadata_technical_plus_foundation_image": controls["metadata_technical"],
            "all_controls_plus_foundation_image": controls["all_controls"],
        }
        local_folds = grouped_donor_folds(
            donors[outer_train], n_splits=inner_folds, seed=seed + outer_index
        )
        fold_scores = {
            model: np.full(
                (len(local_folds), len(ALPHAS), y.shape[1]),
                np.nan,
                dtype=np.float64,
            )
            for model in predictions
        }
        pca_receipts = []
        for fold_index, (local_train, local_validation) in enumerate(local_folds):
            train = outer_train[local_train]
            validation = outer_train[local_validation]
            image_train, image_validation, receipt = _prepare_representation(
                image_source,
                train,
                validation,
                spec,
                donors,
                sections,
                fine_types,
                weighting,
                device,
            )
            pca_receipts.append(receipt)
            for model, control in control_by_model.items():
                x_train = _combined_features(image_train, control, train)
                x_validation = _combined_features(image_validation, control, validation)
                grid = _architecture_predict_grid(
                    x_train,
                    y[train],
                    x_validation,
                    train,
                    validation,
                    donors,
                    sections,
                    fine_types,
                    broad_types,
                    architecture=architecture,
                    weighting=weighting,
                    alphas=ALPHAS,
                    device=device,
                )
                fold_scores[model][fold_index] = _grid_scores(
                    y[validation],
                    grid,
                    validation,
                    donors,
                    sections,
                    fine_types,
                    target_names,
                    minimum_support=minimum_support,
                )
            selected_model = "selected_best_control_plus_foundation_image"
            for target_index, target in enumerate(target_names):
                control = controls[selected_control_by_target[target]]
                grid = _architecture_predict_grid(
                    _combined_features(image_train, control, train),
                    y[train, target_index : target_index + 1],
                    _combined_features(image_validation, control, validation),
                    train,
                    validation,
                    donors,
                    sections,
                    fine_types,
                    broad_types,
                    architecture=architecture,
                    weighting=weighting,
                    alphas=ALPHAS,
                    device=device,
                )
                selected_score = _grid_scores(
                    y[validation, target_index : target_index + 1],
                    grid,
                    validation,
                    donors,
                    sections,
                    fine_types,
                    (target,),
                    minimum_support=minimum_support,
                )
                fold_scores[selected_model][fold_index, :, target_index] = selected_score[:, 0]
        selected_indices: dict[str, np.ndarray] = {}
        selected_scores: dict[str, np.ndarray] = {}
        for model in predictions:
            selected_indices[model], selected_scores[model] = _selected_alpha_indices(
                fold_scores[model]
            )
        image_train, image_test, outer_receipt = _prepare_representation(
            image_source,
            outer_train,
            outer_test,
            spec,
            donors,
            sections,
            fine_types,
            weighting,
            device,
        )
        for model, control in control_by_model.items():
            grid = _architecture_predict_grid(
                _combined_features(image_train, control, outer_train),
                y[outer_train],
                _combined_features(image_test, control, outer_test),
                outer_train,
                outer_test,
                donors,
                sections,
                fine_types,
                broad_types,
                architecture=architecture,
                weighting=weighting,
                alphas=ALPHAS,
                device=device,
            )
            predictions[model][outer_test] = _take_target_alphas(grid, selected_indices[model])
        selected_model = "selected_best_control_plus_foundation_image"
        for target_index, target in enumerate(target_names):
            control = controls[selected_control_by_target[target]]
            grid = _architecture_predict_grid(
                _combined_features(image_train, control, outer_train),
                y[outer_train, target_index : target_index + 1],
                _combined_features(image_test, control, outer_test),
                outer_train,
                outer_test,
                donors,
                sections,
                fine_types,
                broad_types,
                architecture=architecture,
                weighting=weighting,
                alphas=ALPHAS,
                device=device,
            )
            alpha_index = int(selected_indices[selected_model][target_index])
            predictions[selected_model][outer_test, target_index] = grid[alpha_index, :, 0]
        outer_reports[heldout] = {
            "heldout_donor": heldout,
            "training_donors": sorted(set(donors[outer_train].tolist())),
            "selected_control_family_by_target": selected_control_by_target,
            "selected_alphas": {
                model: {
                    target: float(ALPHAS[selected_indices[model][target_index]])
                    for target_index, target in enumerate(target_names)
                }
                for model in predictions
            },
            "inner_selected_scores": {
                model: {
                    target: float(selected_scores[model][target_index])
                    for target_index, target in enumerate(target_names)
                }
                for model in predictions
            },
            "representation_fit": outer_receipt,
            "inner_representation_fits": pca_receipts,
            "selection_partition": "inner_training_donors_only",
            "heldout_outcomes_used_for_selection": False,
            "inner_fold_seed": seed + outer_index,
            "inner_validation_donors": [
                sorted(set(donors[outer_train][validation].tolist()))
                for _training, validation in local_folds
            ],
        }
        _log(
            "HEST image: representation=%s architecture=%s heldout=%s"
            % (spec.name, architecture, heldout)
        )
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("image plan left non-finite held-out predictions")
    image_scores = _score_models(
        y,
        predictions,
        donors,
        sections,
        fine_types,
        target_names,
        minimum_support=minimum_support,
    )
    central_model = image_scores["selected_best_control_plus_foundation_image"][
        str(minimum_support)
    ]
    central_control = control_plan.scores["selected_best_control"][str(minimum_support)]
    effects: dict[str, Mapping[str, object]] = {}
    p_values: dict[str, float] = {}
    for target in target_names:
        model_per_donor = {
            donor: values["donor_type_r2"]
            for donor, values in central_model["targets"][target]["per_donor"].items()
            if values["donor_type_r2"] is not None
        }
        control_per_donor = {
            donor: values["donor_type_r2"]
            for donor, values in central_control["targets"][target]["per_donor"].items()
            if values["donor_type_r2"] is not None
        }
        paired = summarize_paired_donor_effects(
            model_per_donor,
            control_per_donor,
            bootstrap_iterations=5000,
            bootstrap_seed=seed + 50_000,
        )
        effects[target] = paired
        p_values[target] = float(paired["exact_sign_flip_p"])
    adjusted = holm_adjust(p_values)
    effects = {
        target: {**effect, "holm_adjusted_exact_sign_flip_p": adjusted[target]}
        for target, effect in effects.items()
    }
    directional = []
    for target in target_names:
        effect = effects[target]
        model_target = central_model["targets"][target]
        if (
            effect["mean_effect"] > 0
            and effect["positive_fraction"] > 0.5
            and model_target["donor_type_macro_reference_error_reduction"] is not None
            and model_target["donor_type_macro_reference_error_reduction"] > 0
        ):
            directional.append(target)
    return {
        "representation": {
            "name": spec.name,
            "kind": spec.kind,
            "dimension": spec.dimension,
            "random_projection_seed": spec.seed,
            "random_projection_sha256": projection_receipt,
        },
        "architecture": architecture,
        "training_weighting": weighting,
        "alpha_grid": list(ALPHAS),
        "inner_donor_folds": inner_folds,
        "inner_fold_seed": seed,
        "minimum_reference_and_scoring_support": minimum_support,
        "outer_folds": outer_reports,
        "controls": control_plan.scores,
        "image_models": image_scores,
        "central_increment": effects,
        "programs_passing_directional_prefilter": directional,
        "pairing_null_policy": {
            "status": (
                "required_for_directional_programs" if directional else "not_run_observed_futility"
            ),
            "programs_requiring_pairing_nulls": directional,
            "reason": (
                "Both registered pairing nulls are required before support. "
                "They are skipped only when the observed direction/error gates already fail."
            ),
        },
    }


def _program_outcome(
    source: HestSource,
    targets: np.ndarray,
    roles: np.ndarray,
    *,
    minimum_support: int = PRIMARY_SUPPORT,
) -> tuple[np.ndarray, np.ndarray]:
    program_targets = _programs(targets, source)
    return _residualize_values(
        program_targets,
        source,
        roles,
        minimum_support=minimum_support,
    )


def _subset_eval(
    source: HestSource, evaluation: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    return (
        source.donors[evaluation],
        source.sections[evaluation],
        source.fine_types[evaluation],
        source.broad_types[evaluation],
        {name: values[evaluation] for name, values in source.controls.items()},
    )


def _representation_specs() -> tuple[tuple[RepresentationSpec, str], ...]:
    return (
        (RepresentationSpec("full_1536_pooled", "full", 1536), "pooled"),
        (
            RepresentationSpec("full_1536_broad_lineage_heads", "full", 1536),
            "broad_lineage_heads",
        ),
        (
            RepresentationSpec("pca_256_broad_lineage_heads", "pca", 256),
            "broad_lineage_heads",
        ),
        (
            RepresentationSpec("pca_512_broad_lineage_heads", "pca", 512),
            "broad_lineage_heads",
        ),
        (
            RepresentationSpec("random_96_seed_20260711", "random_projection", 96, 20260711),
            "broad_lineage_heads",
        ),
        (
            RepresentationSpec("random_96_seed_20260713", "random_projection", 96, 20260713),
            "broad_lineage_heads",
        ),
        (
            RepresentationSpec("random_96_seed_20260717", "random_projection", 96, 20260717),
            "broad_lineage_heads",
        ),
    )


def observed_program_analysis(
    source: HestSource,
    *,
    target_scope: str,
    targets: np.ndarray,
    roles: np.ndarray,
    device: str,
    inner_folds: int,
    seed: int,
    full_representation_sensitivity: bool,
    minimum_support: int = PRIMARY_SUPPORT,
) -> Mapping[str, object]:
    y, evaluation = _program_outcome(
        source,
        targets,
        roles,
        minimum_support=minimum_support,
    )
    donors, sections, fine_types, broad_types, controls = _subset_eval(source, evaluation)
    crop_index = source.crop_ids.index("crop_112um")
    raw_image = source.images[evaluation, crop_index]
    specifications = (
        _representation_specs()
        if full_representation_sensitivity
        else (
            (
                RepresentationSpec("pca_512_broad_lineage_heads", "pca", 512),
                "broad_lineage_heads",
            ),
        )
    )
    architectures = {architecture for _spec, architecture in specifications}
    control_plans = {
        architecture: _fit_control_plan(
            y,
            controls,
            donors,
            sections,
            fine_types,
            broad_types,
            source.program_names,
            architecture=architecture,
            weighting="donor_type",
            inner_folds=inner_folds,
            seed=seed + index * 1000,
            device=device,
            minimum_support=minimum_support,
        )
        for index, architecture in enumerate(sorted(architectures))
    }
    results = {}
    for spec_index, (spec, architecture) in enumerate(specifications):
        results[spec.name] = _fit_image_plan(
            raw_image,
            spec,
            y,
            controls,
            control_plans[architecture],
            donors,
            sections,
            fine_types,
            broad_types,
            source.program_names,
            architecture=architecture,
            weighting="donor_type",
            inner_folds=inner_folds,
            seed=seed + 10_000 + spec_index * 1000,
            device=device,
            minimum_support=minimum_support,
        )
    weighting_sensitivity: Mapping[str, object] = {"status": "not_run_for_secondary_target_scope"}
    if full_representation_sensitivity:
        sensitivity_spec = RepresentationSpec("pca_512_broad_lineage_heads", "pca", 512)
        section_control = _fit_control_plan(
            y,
            controls,
            donors,
            sections,
            fine_types,
            broad_types,
            source.program_names,
            architecture="broad_lineage_heads",
            weighting="donor_section_type",
            inner_folds=inner_folds,
            seed=seed + 80_000,
            device=device,
            minimum_support=minimum_support,
        )
        weighting_sensitivity = _fit_image_plan(
            raw_image,
            sensitivity_spec,
            y,
            controls,
            section_control,
            donors,
            sections,
            fine_types,
            broad_types,
            source.program_names,
            architecture="broad_lineage_heads",
            weighting="donor_section_type",
            inner_folds=inner_folds,
            seed=seed + 90_000,
            device=device,
            minimum_support=minimum_support,
        )
    return {
        "target_scope": target_scope,
        "endpoint": "same_donor_section_fine_type_reference_residual_program_score",
        "evaluation_rows": int(len(y)),
        "evaluation_mask_sha256": _array_sha256(evaluation),
        "program_outcome_sha256": _array_sha256(y),
        "minimum_reference_and_scoring_support": minimum_support,
        "donors": sorted(set(donors.tolist())),
        "program_names": list(source.program_names),
        "primary_crop": "crop_112um",
        "representations": results,
        "donor_section_type_training_weight_sensitivity": weighting_sensitivity,
    }


def reference_support_sensitivity_analysis(
    source: HestSource,
    primary_observed: Mapping[str, object],
    *,
    device: str,
    inner_folds: int,
    seed: int,
) -> Mapping[str, object]:
    """Refit the primary program probe whenever a support cutoff changes outcomes."""

    primary_mask_sha = str(primary_observed["evaluation_mask_sha256"])
    primary_outcome_sha = str(primary_observed["program_outcome_sha256"])
    thresholds: dict[str, Mapping[str, object]] = {}
    for offset, threshold in enumerate(SUPPORT_THRESHOLDS):
        y, evaluation = _program_outcome(
            source,
            source.nucleus_targets,
            source.roles,
            minimum_support=threshold,
        )
        mask_sha = _array_sha256(evaluation)
        outcome_sha = _array_sha256(y)
        identical = bool(mask_sha == primary_mask_sha and outcome_sha == primary_outcome_sha)
        receipt: dict[str, object] = {
            "minimum_reference_and_scoring_support": threshold,
            "evaluation_rows": int(len(y)),
            "evaluation_mask_sha256": mask_sha,
            "program_outcome_sha256": outcome_sha,
            "identical_to_primary_support_20": identical,
        }
        if identical:
            receipt.update(
                {
                    "execution": "exact_primary_analysis_reuse",
                    "analysis_pointer": "observed_nucleus_programs",
                    "reason": (
                        "The eligible evaluation mask and reference-residual program "
                        "matrix are byte-identical to the primary support-20 outcome."
                    ),
                }
            )
        else:
            receipt.update(
                {
                    "execution": "complete_reference_residual_refit_and_nested_retuning",
                    "analysis": observed_program_analysis(
                        source,
                        target_scope=(
                            f"nucleus_overlap_reference_support_sensitivity::{threshold}"
                        ),
                        targets=source.nucleus_targets,
                        roles=source.roles,
                        device=device,
                        inner_folds=inner_folds,
                        seed=seed + offset * 100_000,
                        full_representation_sensitivity=False,
                        minimum_support=threshold,
                    ),
                }
            )
        thresholds[str(threshold)] = receipt
    return {
        "status": "complete_threshold_specific_reference_and_model_sensitivity",
        "thresholds": thresholds,
        "primary_threshold": PRIMARY_SUPPORT,
        "note": (
            "A threshold is reused only when both its eligibility mask and its "
            "reference-residual program matrix are byte-identical to support 20; "
            "otherwise reference means, nested tuning, and held-out models are refit."
        ),
    }


def _classification_macro_by_donor(
    truth: np.ndarray, prediction: np.ndarray, donors: np.ndarray
) -> Mapping[str, object]:
    per_donor = {
        donor: multiclass_metrics(truth[donors == donor], prediction[donors == donor])
        for donor in sorted(set(donors.tolist()))
    }
    return {
        "per_donor": per_donor,
        "donor_macro_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in per_donor.values()])
        ),
        "donor_macro_f1": float(np.mean([row["macro_f1"] for row in per_donor.values()])),
    }


def _positive_classification(
    raw_image: np.ndarray,
    truth_labels: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    fine_types: np.ndarray,
    *,
    target_name: str,
    inner_folds: int,
    seed: int,
    device: str,
) -> Mapping[str, object]:
    classes = tuple(sorted(set(truth_labels.tolist())))
    lookup = {label: index for index, label in enumerate(classes)}
    targets = np.zeros((len(truth_labels), len(classes)), dtype=np.float64)
    targets[np.arange(len(targets)), [lookup[value] for value in truth_labels]] = 1.0
    predictions = np.empty(len(truth_labels), dtype=object)
    baseline = np.empty(len(truth_labels), dtype=object)
    outer_reports = {}
    all_rows = np.arange(len(truth_labels), dtype=np.int64)
    spec = RepresentationSpec("pca_256", "pca", 256)
    for outer_index, heldout in enumerate(sorted(set(donors.tolist()))):
        outer_train = all_rows[donors != heldout]
        outer_test = all_rows[donors == heldout]
        folds = grouped_donor_folds(
            donors[outer_train], n_splits=inner_folds, seed=seed + outer_index
        )
        scores = np.zeros((len(folds), len(ALPHAS)), dtype=np.float64)
        for fold_index, (local_train, local_validation) in enumerate(folds):
            train = outer_train[local_train]
            validation = outer_train[local_validation]
            x_train, x_validation, _receipt = _prepare_representation(
                raw_image,
                train,
                validation,
                spec,
                donors,
                sections,
                fine_types,
                "donor_type",
                device,
            )
            grid = _architecture_predict_grid(
                x_train,
                targets[train],
                x_validation,
                train,
                validation,
                donors,
                sections,
                truth_labels,
                truth_labels,
                architecture="pooled",
                weighting="donor_type",
                alphas=ALPHAS,
                device=device,
            )
            for alpha_index, values in enumerate(grid):
                predicted = np.asarray(classes)[np.argmax(values, axis=1)]
                fold_metrics = _classification_macro_by_donor(
                    truth_labels[validation], predicted, donors[validation]
                )
                scores[fold_index, alpha_index] = fold_metrics["donor_macro_balanced_accuracy"]
        mean_scores = scores.mean(axis=0)
        selected = int(np.argmax(mean_scores))
        x_train, x_test, receipt = _prepare_representation(
            raw_image,
            outer_train,
            outer_test,
            spec,
            donors,
            sections,
            fine_types,
            "donor_type",
            device,
        )
        grid = _architecture_predict_grid(
            x_train,
            targets[outer_train],
            x_test,
            outer_train,
            outer_test,
            donors,
            sections,
            truth_labels,
            truth_labels,
            architecture="pooled",
            weighting="donor_type",
            alphas=ALPHAS,
            device=device,
        )
        predictions[outer_test] = np.asarray(classes)[np.argmax(grid[selected], axis=1)]
        training_counts = {
            label: int(np.count_nonzero(truth_labels[outer_train] == label)) for label in classes
        }
        majority = sorted(classes, key=lambda label: (-training_counts[label], label))[0]
        baseline[outer_test] = majority
        outer_reports[heldout] = {
            "selected_alpha": float(ALPHAS[selected]),
            "inner_donor_macro_balanced_accuracy": float(mean_scores[selected]),
            "representation_fit": receipt,
            "heldout_outcomes_used_for_selection": False,
        }
    prediction_values = predictions.astype(str)
    baseline_values = baseline.astype(str)
    return {
        "target": target_name,
        "classes": list(classes),
        "representation": "crop_112um_train_only_pca_256",
        "model": {
            "global": multiclass_metrics(truth_labels, prediction_values, class_labels=classes),
            "donor_balanced": _classification_macro_by_donor(
                truth_labels, prediction_values, donors
            ),
        },
        "training_majority_baseline": {
            "global": multiclass_metrics(truth_labels, baseline_values, class_labels=classes),
            "donor_balanced": _classification_macro_by_donor(truth_labels, baseline_values, donors),
        },
        "outer_folds": outer_reports,
    }


def _positive_morphology_regression(
    raw_image: np.ndarray,
    targets: np.ndarray,
    target_names: Sequence[str],
    donors: np.ndarray,
    sections: np.ndarray,
    fine_types: np.ndarray,
    *,
    inner_folds: int,
    seed: int,
    device: str,
) -> Mapping[str, object]:
    predictions = np.full_like(targets, np.nan, dtype=np.float64)
    reference_predictions = np.full_like(targets, np.nan, dtype=np.float64)
    all_rows = np.arange(len(targets), dtype=np.int64)
    outer_reports = {}
    spec = RepresentationSpec("full_1536", "full", 1536)
    for outer_index, heldout in enumerate(sorted(set(donors.tolist()))):
        outer_train = all_rows[donors != heldout]
        outer_test = all_rows[donors == heldout]
        training_donor_means = [
            targets[outer_train][donors[outer_train] == donor].mean(axis=0, dtype=np.float64)
            for donor in sorted(set(donors[outer_train].tolist()))
        ]
        global_training_reference = np.mean(training_donor_means, axis=0)
        for fine_type in sorted(set(fine_types[outer_test].tolist())):
            local_test = outer_test[fine_types[outer_test] == fine_type]
            type_donor_means = [
                targets[outer_train][
                    (donors[outer_train] == donor) & (fine_types[outer_train] == fine_type)
                ].mean(axis=0, dtype=np.float64)
                for donor in sorted(set(donors[outer_train].tolist()))
                if np.any((donors[outer_train] == donor) & (fine_types[outer_train] == fine_type))
            ]
            reference_predictions[local_test] = (
                np.mean(type_donor_means, axis=0) if type_donor_means else global_training_reference
            )
        folds = grouped_donor_folds(
            donors[outer_train], n_splits=inner_folds, seed=seed + outer_index
        )
        scores = np.full(
            (len(folds), len(ALPHAS), targets.shape[1]),
            np.nan,
            dtype=np.float64,
        )
        for fold_index, (local_train, local_validation) in enumerate(folds):
            train = outer_train[local_train]
            validation = outer_train[local_validation]
            x_train, x_validation, _receipt = _prepare_representation(
                raw_image,
                train,
                validation,
                spec,
                donors,
                sections,
                fine_types,
                "donor_type",
                device,
            )
            grid = _architecture_predict_grid(
                x_train,
                targets[train],
                x_validation,
                train,
                validation,
                donors,
                sections,
                fine_types,
                fine_types,
                architecture="pooled",
                weighting="donor_type",
                alphas=ALPHAS,
                device=device,
            )
            scores[fold_index] = _grid_scores(
                targets[validation],
                grid,
                validation,
                donors,
                sections,
                fine_types,
                target_names,
                minimum_support=PRIMARY_SUPPORT,
            )
        selected, selected_scores = _selected_alpha_indices(scores)
        x_train, x_test, receipt = _prepare_representation(
            raw_image,
            outer_train,
            outer_test,
            spec,
            donors,
            sections,
            fine_types,
            "donor_type",
            device,
        )
        grid = _architecture_predict_grid(
            x_train,
            targets[outer_train],
            x_test,
            outer_train,
            outer_test,
            donors,
            sections,
            fine_types,
            fine_types,
            architecture="pooled",
            weighting="donor_type",
            alphas=ALPHAS,
            device=device,
        )
        predictions[outer_test] = _take_target_alphas(grid, selected)
        outer_reports[heldout] = {
            "selected_alphas": {
                name: float(ALPHAS[selected[index]]) for index, name in enumerate(target_names)
            },
            "inner_selected_r2": {
                name: float(selected_scores[index]) for index, name in enumerate(target_names)
            },
            "representation_fit": receipt,
            "heldout_outcomes_used_for_selection": False,
        }
    if not np.isfinite(predictions).all() or not np.isfinite(reference_predictions).all():
        raise ValueError("morphology positive control left non-finite predictions")
    score = score_continuous_targets(
        targets,
        predictions,
        reference_predictions,
        donors,
        sections,
        fine_types,
        target_names=target_names,
        minimum_support=PRIMARY_SUPPORT,
    )
    return {
        "target": "basic_nuclear_morphology",
        "target_names": list(target_names),
        "representation": "full_1536_foundation_features",
        "reference": (
            "outer-training-donor-balanced fine-type mean; falls back to the "
            "outer-training-donor-balanced global mean"
        ),
        "scores": score,
        "outer_folds": outer_reports,
    }


def positive_control_analysis(
    source: HestSource,
    *,
    device: str,
    inner_folds: int,
    seed: int,
) -> Mapping[str, object]:
    _y, evaluation = _program_outcome(source, source.nucleus_targets, source.roles)
    donors = source.donors[evaluation]
    sections = source.sections[evaluation]
    fine_types = source.fine_types[evaluation]
    broad_types = source.broad_types[evaluation]
    crop_index = source.crop_ids.index("crop_112um")
    raw_image = source.images[evaluation, crop_index]
    nucleus_crop_index = source.crop_ids.index("nucleus_mask_only")
    nucleus_image = source.images[evaluation, nucleus_crop_index]
    desired = (
        "nucleus_area_um2",
        "nucleus_perimeter_um",
        "nucleus_eccentricity",
        "nucleus_circularity",
        "nucleus_solidity",
        "nucleus_gray_mean",
        "nucleus_hematoxylin_od_mean",
        "nucleus_glcm_contrast",
    )
    name_lookup = {name: index for index, name in enumerate(source.nuclear_morphometric_names)}
    if any(name not in name_lookup for name in desired):
        raise ValueError("source lacks a prespecified nuclear morphology positive target")
    morphology = source.nuclear_morphometrics[evaluation][
        :, [name_lookup[name] for name in desired]
    ].astype(np.float64)
    return {
        "broad_lineage": _positive_classification(
            raw_image,
            broad_types,
            donors,
            sections,
            fine_types,
            target_name="broad_lineage",
            inner_folds=inner_folds,
            seed=seed,
            device=device,
        ),
        "fine_type": _positive_classification(
            raw_image,
            fine_types,
            donors,
            sections,
            fine_types,
            target_name="fine_type",
            inner_folds=inner_folds,
            seed=seed + 10_000,
            device=device,
        ),
        "nuclear_morphology": {
            "full_context": _positive_morphology_regression(
                raw_image,
                morphology,
                desired,
                donors,
                sections,
                fine_types,
                inner_folds=inner_folds,
                seed=seed + 20_000,
                device=device,
            ),
            "nucleus_mask_only": _positive_morphology_regression(
                nucleus_image,
                morphology,
                desired,
                donors,
                sections,
                fine_types,
                inner_folds=inner_folds,
                seed=seed + 30_000,
                device=device,
            ),
        },
    }


def positive_control_gate(controls: Mapping[str, object]) -> Mapping[str, object]:
    """Prespecified visible-signal gate evaluated before molecular outcomes."""

    classification_rows: dict[str, Mapping[str, object]] = {}
    classification_passes = []
    for target in ("broad_lineage", "fine_type"):
        row = controls[target]
        model = float(row["model"]["donor_balanced"]["donor_macro_balanced_accuracy"])
        baseline = float(
            row["training_majority_baseline"]["donor_balanced"][
                "donor_macro_balanced_accuracy"
            ]
        )
        passed = bool(np.isfinite(model) and np.isfinite(baseline) and model > baseline)
        classification_rows[target] = {
            "metric": "donor_macro_balanced_accuracy",
            "model": model,
            "outer_training_majority_baseline": baseline,
            "minimum_required_increment": 0.0,
            "observed_increment": model - baseline,
            "passed": passed,
        }
        classification_passes.append(passed)

    required_morphology = (
        "nucleus_area_um2",
        "nucleus_perimeter_um",
        "nucleus_circularity",
        "nucleus_solidity",
        "nucleus_gray_mean",
        "nucleus_hematoxylin_od_mean",
        "nucleus_glcm_contrast",
    )
    morphology_scores = controls["nuclear_morphology"]["full_context"]["scores"]["targets"]
    morphology_rows = {}
    morphology_passes = []
    for target in required_morphology:
        value = float(morphology_scores[target]["donor_type_macro_reference_error_reduction"])
        passed = bool(np.isfinite(value) and value > 0.0)
        morphology_rows[target] = {
            "metric": "donor_type_macro_reference_error_reduction",
            "minimum_required": 0.0,
            "observed": value,
            "passed": passed,
        }
        morphology_passes.append(passed)

    passed = bool(all(classification_passes) and all(morphology_passes))
    return {
        "schema": "heir.hest_visible_positive_control_gate.v1",
        "evaluated_before_molecular_models": True,
        "thresholds_frozen_without_hoptimus_molecular_outcomes": True,
        "natural_unmasked_112um_is_primary": True,
        "classification": classification_rows,
        "full_context_morphology": morphology_rows,
        "nucleus_mask_only_role": "secondary_attribution_not_used_for_gate",
        "passed": passed,
        "molecular_interpretation_allowed": passed,
        "failure_action": (
            None
            if passed
            else "stop_before_molecular_models_and_audit_loading_preprocessing_crop_scale"
        ),
    }


def uni2_baseline_only_eligibility(
    encoder: str, gate: object
) -> Mapping[str, object]:
    """Apply the pre-H-optimus descriptive-baseline amendment without changing the gate."""

    reasons: list[str] = []
    expected_classification = {"broad_lineage", "fine_type"}
    expected_morphology = set(UNI2_BASELINE_GEOMETRY_TARGETS) | set(
        UNI2_BASELINE_APPEARANCE_TARGETS
    )
    if encoder != UNI2H_REPOSITORY:
        reasons.append("encoder_is_not_UNI2h")
    if not isinstance(gate, Mapping):
        reasons.append("positive_control_gate_missing_or_malformed")
        return {
            "schema": "heir.hest_uni2_baseline_only_eligibility.v1",
            "eligible": False,
            "role": None,
            "amendment_timing": UNI2_BASELINE_AMENDMENT_TIMING,
            "original_gate_preserved": True,
            "original_gate_passed": None,
            "original_molecular_interpretation_allowed": None,
            "classification_targets_passed": [],
            "appearance_targets_passed": [],
            "failed_geometry_targets": [],
            "comparison_inference_allowed": False,
            "descriptive_only": True,
            "reasons": reasons,
        }
    if gate.get("schema") != "heir.hest_visible_positive_control_gate.v1":
        reasons.append("unexpected_gate_schema")
    if gate.get("evaluated_before_molecular_models") is not True:
        reasons.append("gate_timing_contract_failed")
    if gate.get("thresholds_frozen_without_hoptimus_molecular_outcomes") is not True:
        reasons.append("gate_threshold_freeze_contract_failed")
    if gate.get("natural_unmasked_112um_is_primary") is not True:
        reasons.append("gate_primary_crop_contract_failed")
    if gate.get("passed") is not False:
        reasons.append("original_gate_is_not_failed")
    if gate.get("molecular_interpretation_allowed") is not False:
        reasons.append("original_gate_does_not_block_molecular_interpretation")
    if gate.get("failure_action") != (
        "stop_before_molecular_models_and_audit_loading_preprocessing_crop_scale"
    ):
        reasons.append("original_gate_failure_action_changed")

    classification = gate.get("classification")
    classification_passed: list[str] = []
    if not isinstance(classification, Mapping) or set(classification) != expected_classification:
        reasons.append("classification_target_contract_failed")
    else:
        for target in sorted(expected_classification):
            row = classification[target]
            row_valid = isinstance(row, Mapping)
            try:
                model = float(row["model"]) if isinstance(row, Mapping) else float("nan")
                baseline = (
                    float(row["outer_training_majority_baseline"])
                    if isinstance(row, Mapping)
                    else float("nan")
                )
                increment = (
                    float(row["observed_increment"])
                    if isinstance(row, Mapping)
                    else float("nan")
                )
            except (KeyError, TypeError, ValueError):
                model = baseline = increment = float("nan")
                row_valid = False
            expected_pass = bool(
                np.isfinite(model)
                and np.isfinite(baseline)
                and model > baseline
                and np.isclose(increment, model - baseline, rtol=0.0, atol=1e-12)
            )
            row_valid = bool(
                row_valid
                and row.get("metric") == "donor_macro_balanced_accuracy"
                and row.get("minimum_required_increment") == 0.0
                and row.get("passed") is expected_pass
            )
            if not row_valid:
                reasons.append(f"classification_row_contract_failed::{target}")
            elif expected_pass:
                classification_passed.append(target)
            else:
                reasons.append(f"classification_failed::{target}")

    morphology = gate.get("full_context_morphology")
    appearance_passed: list[str] = []
    failed_geometry: list[str] = []
    if not isinstance(morphology, Mapping) or set(morphology) != expected_morphology:
        reasons.append("morphology_target_contract_failed")
    else:
        for target in sorted(expected_morphology):
            row = morphology[target]
            row_valid = isinstance(row, Mapping)
            try:
                observed = (
                    float(row["observed"]) if isinstance(row, Mapping) else float("nan")
                )
            except (KeyError, TypeError, ValueError):
                observed = float("nan")
                row_valid = False
            expected_pass = bool(np.isfinite(observed) and observed > 0.0)
            row_valid = bool(
                row_valid
                and row.get("metric")
                == "donor_type_macro_reference_error_reduction"
                and row.get("minimum_required") == 0.0
                and row.get("passed") is expected_pass
            )
            if not row_valid:
                reasons.append(f"morphology_row_contract_failed::{target}")
            elif target in UNI2_BASELINE_APPEARANCE_TARGETS:
                if expected_pass:
                    appearance_passed.append(target)
                else:
                    reasons.append(f"appearance_failed::{target}")
            elif not expected_pass:
                failed_geometry.append(target)
    if set(failed_geometry) != set(UNI2_BASELINE_GEOMETRY_TARGETS):
        reasons.append("all_four_geometry_targets_must_fail_exactly")
    eligible = not reasons
    return {
        "schema": "heir.hest_uni2_baseline_only_eligibility.v1",
        "eligible": eligible,
        "role": (
            "retrospective_exposed_non_authorizing_baseline" if eligible else None
        ),
        "amendment_timing": UNI2_BASELINE_AMENDMENT_TIMING,
        "source_and_technical_identity_precondition": (
            "validated_by_load_source_before_eligibility"
        ),
        "original_gate_preserved": True,
        "original_gate_sha256": _canonical_sha256(gate),
        "original_gate_passed": gate.get("passed"),
        "original_molecular_interpretation_allowed": gate.get(
            "molecular_interpretation_allowed"
        ),
        "classification_targets_passed": classification_passed,
        "appearance_targets_passed": appearance_passed,
        "failed_geometry_targets": failed_geometry,
        "allowed_failure_family": list(UNI2_BASELINE_GEOMETRY_TARGETS),
        "comparison_inference_allowed": False,
        "descriptive_only": True,
        "reasons": reasons,
    }


def _central_model_score(arm: Mapping[str, object]) -> Mapping[str, object]:
    return arm["image_models"]["selected_best_control_plus_foundation_image"][str(PRIMARY_SUPPORT)]


def crop_sensitivity_analysis(
    source: HestSource,
    *,
    device: str,
    inner_folds: int,
    seed: int,
) -> Mapping[str, object]:
    y, evaluation = _program_outcome(source, source.nucleus_targets, source.roles)
    donors, sections, fine_types, broad_types, controls = _subset_eval(source, evaluation)
    spec = RepresentationSpec("pca_512_broad_lineage_heads", "pca", 512)
    control_plan = _fit_control_plan(
        y,
        controls,
        donors,
        sections,
        fine_types,
        broad_types,
        source.program_names,
        architecture="broad_lineage_heads",
        weighting="donor_type",
        inner_folds=inner_folds,
        seed=seed,
        device=device,
    )
    shared_image_seed = seed + 10_000
    arms: dict[str, Mapping[str, object]] = {}
    for crop_id in REQUIRED_BASE_CROPS:
        crop_index = source.crop_ids.index(crop_id)
        arms[crop_id] = _fit_image_plan(
            source.images[evaluation, crop_index],
            spec,
            y,
            controls,
            control_plan,
            donors,
            sections,
            fine_types,
            broad_types,
            source.program_names,
            architecture="broad_lineage_heads",
            weighting="donor_type",
            inner_folds=inner_folds,
            seed=shared_image_seed,
            device=device,
        )
    contrast_specifications = {
        "full_context_minus_target_removed": (
            "crop_112um",
            "target_cell_removed_112um",
        ),
        "cell_mask_minus_target_removed": (
            "cell_mask_only",
            "target_cell_removed_112um",
        ),
        "nucleus_mask_minus_target_removed": (
            "nucleus_mask_only",
            "target_cell_removed_112um",
        ),
    }
    contrasts: dict[str, dict[str, Mapping[str, object]]] = {}
    p_values = {}
    for contrast, (left, right) in contrast_specifications.items():
        left_score = _central_model_score(arms[left])
        right_score = _central_model_score(arms[right])
        contrasts[contrast] = {}
        for program in source.program_names:
            left_donor = {
                donor: values["donor_type_r2"]
                for donor, values in left_score["targets"][program]["per_donor"].items()
                if values["donor_type_r2"] is not None
            }
            right_donor = {
                donor: values["donor_type_r2"]
                for donor, values in right_score["targets"][program]["per_donor"].items()
                if values["donor_type_r2"] is not None
            }
            summary = summarize_paired_donor_effects(
                left_donor,
                right_donor,
                bootstrap_iterations=5000,
                bootstrap_seed=seed + 70_000,
            )
            contrasts[contrast][program] = summary
            p_values[f"{contrast}::{program}"] = summary["exact_sign_flip_p"]
    adjusted = holm_adjust(p_values)
    for contrast in contrasts:
        for program in contrasts[contrast]:
            key = f"{contrast}::{program}"
            contrasts[contrast][program] = {
                **contrasts[contrast][program],
                "holm_adjusted_across_all_crop_program_contrasts": adjusted[key],
                "biological_intrinsic_interpretation_allowed": False,
            }
    return {
        "status": "exploratory_artifact_confounded",
        "representation": spec.name,
        "matched_tuning_contract": {
            "all_four_arms_refit_in_this_analysis": True,
            "shared_control_plan_object": True,
            "shared_control_plan_seed": seed,
            "shared_image_inner_fold_seed": shared_image_seed,
            "identical_inner_validation_donor_folds": True,
            "arm_specific_alpha_selection_on_identical_folds": True,
            "primary_observed_arm_reused": False,
        },
        "arms": arms,
        "direct_paired_contrasts": contrasts,
        "multiplicity_family": "three_crop_contrasts_by_six_programs",
        "h_intrinsic_authorized": False,
        "blocking_reason": (
            "White-fill arms change retained tissue fraction, edge geometry, and fill. "
            "Matched mean-fill, blurred, and random-location embeddings are absent."
        ),
    }


def reference_split_sensitivity_analysis(
    source: HestSource,
    *,
    device: str,
    inner_folds: int,
    seed: int,
) -> Mapping[str, object]:
    results = {}
    for split_index, split_id in enumerate(source.split_ids.tolist()):
        if split_id == "primary":
            continue
        results[split_id] = observed_program_analysis(
            source,
            target_scope=f"nucleus_overlap_reference_split::{split_id}",
            targets=source.nucleus_targets,
            roles=source.roles_by_split[:, split_index],
            device=device,
            inner_folds=inner_folds,
            seed=seed + split_index * 100_000,
            full_representation_sensitivity=False,
        )
    return {
        "status": "alternate_frozen_spatial_pool_sensitivity",
        "primary_split_id": "primary",
        "alternate_splits": results,
        "new_deterministic_cell_sample_available": False,
        "note": (
            "These splits test reference-pool sampling stability. A new cell sample or "
            "uncapped reference pool requires rebuilding the source."
        ),
    }


def _primary_probe_summary(
    observed: Mapping[str, object],
    measurement: Mapping[str, object],
    *,
    primary_representation: str = "pca_512_broad_lineage_heads",
) -> Mapping[str, object]:
    primary_name = primary_representation
    primary = observed["representations"][primary_name]
    fixed_six_directional = list(primary["programs_passing_directional_prefilter"])
    reliability_gate = measurement["nucleus_overlap"]["outer_training_program_reliability_gate"]
    primary_programs = list(reliability_gate["programs_eligible_for_primary_inference"])
    directional = [program for program in fixed_six_directional if program in primary_programs]
    reliability = measurement["nucleus_overlap"]["programs"]["donor_type_macro"]["features"]
    program_rows = {}
    model_score = primary["image_models"]["selected_best_control_plus_foundation_image"][
        str(PRIMARY_SUPPORT)
    ]
    control_score = primary["controls"]["selected_best_control"][str(PRIMARY_SUPPORT)]
    for program in observed["program_names"]:
        effect = primary["central_increment"][program]
        program_rows[program] = {
            "measurement_ceiling_spearman_brown_donor_type_median": reliability[program][
                "median_spearman_brown_reliability"
            ],
            "control_donor_type_macro_r2": control_score["targets"][program]["donor_type_macro_r2"],
            "model_donor_type_macro_r2": model_score["targets"][program]["donor_type_macro_r2"],
            "model_reference_error_reduction": model_score["targets"][program][
                "donor_type_macro_reference_error_reduction"
            ],
            "image_increment": effect,
            "primary_reliability_eligible": program in primary_programs,
            "analysis_role": (
                "primary_reliability_qualified_inference"
                if program in primary_programs
                else "fixed_program_descriptive_sensitivity"
            ),
            "directional_prefilter_pass_fixed_six": (program in fixed_six_directional),
            "directional_prefilter_pass_primary": program in directional,
            "pairing_null_pass": None,
            "supports_h_cell": False,
        }
    return {
        "primary_revised_probe": primary_name,
        "primary_reliability_rule": reliability_gate["rule"],
        "primary_inference_programs": primary_programs,
        "fixed_six_programs_reported": list(observed["program_names"]),
        "programs": program_rows,
        "directional_programs": directional,
        "fixed_six_directional_programs": fixed_six_directional,
        "h_cell_retrospective_status": (
            "indeterminate_directional_program_requires_pairing_nulls"
            if directional
            else "negative_evidence_for_revised_program_probe"
        ),
        "authorizes_h_cell": False,
        "authorizes_reference_refinement": False,
        "reason_reference_refinement_blocked": (
            "No donor-held-out positive image increment has passed both pairing nulls."
        ),
    }


def _descriptive_uni2_baseline_summary(
    observed: Mapping[str, object],
) -> Mapping[str, object]:
    representation_name = "pca_512_broad_lineage_heads"
    representations = observed.get("representations")
    if not isinstance(representations, Mapping):
        raise ValueError("UNI2-h descriptive baseline lacks representation effects")
    representation = representations.get(representation_name)
    if not isinstance(representation, Mapping):
        raise ValueError("UNI2-h descriptive baseline lacks its frozen PCA representation")
    effects = representation.get("central_increment")
    if not isinstance(effects, Mapping) or set(effects) != set(FROZEN_PROGRAM_NAMES):
        raise ValueError("UNI2-h descriptive baseline lacks all frozen program effects")
    program_effects: dict[str, object] = {}
    for program in FROZEN_PROGRAM_NAMES:
        effect = effects[program]
        if not isinstance(effect, Mapping):
            raise ValueError(f"UNI2-h descriptive effect is malformed: {program}")
        numeric_fields = {
            "donor_count": effect.get("donor_count"),
            "mean_delta_r2": effect.get("mean_effect"),
            "positive_donor_fraction": effect.get("positive_fraction"),
        }
        if (
            isinstance(numeric_fields["donor_count"], bool)
            or not isinstance(numeric_fields["donor_count"], int)
            or numeric_fields["donor_count"] <= 0
        ):
            raise ValueError(f"UNI2-h descriptive donor count is invalid: {program}")
        for name, value in numeric_fields.items():
            if name == "donor_count":
                continue
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"UNI2-h descriptive numeric value is invalid: {program}/{name}")
        program_effects[program] = numeric_fields
    return {
        "schema": "heir.hest_uni2_descriptive_baseline_summary.v1",
        "role": "retrospective_exposed_non_authorizing_baseline",
        "descriptive_only": True,
        "representation": representation_name,
        "program_effects": program_effects,
    }


def _artifact_boundary(source: HestSource) -> Mapping[str, object]:
    available = sorted(set(ARTIFACT_CONTROL_CROPS) & set(source.crop_ids))
    missing = sorted(set(ARTIFACT_CONTROL_CROPS) - set(source.crop_ids))
    return {
        "required_matched_artifact_control_crops": list(ARTIFACT_CONTROL_CROPS),
        "available": available,
        "missing": missing,
        "current_source_sufficient_for_h_intrinsic": not missing,
        "h_intrinsic_cell_status": ("indeterminate_artifact_confounded_directional_crop_effect"),
        "h_intrinsic_nucleus_status": ("indeterminate_artifact_confounded_directional_crop_effect"),
        "direct_biological_contrast_status": (
            "available" if not missing else "blocked_requires_new_crop_embeddings"
        ),
        "white_fill_contrasts_authorize_intrinsic_claim": False,
        "required_next_artifact": (
            None
            if not missing
            else (
                "A feature-only, resumable artifact-control supplement bound to the "
                "registered source SHA and raw native polygons."
            )
        ),
    }


def _implementation_receipt() -> Mapping[str, object]:
    repository = Path(__file__).resolve().parents[1]
    relative_files = (
        "scripts/benchmark_hest_scientific_reanalysis.py",
        "src/heir/evaluation/hest_measurement.py",
        "src/heir/evaluation/hest_nested_ridge.py",
        "src/heir/evaluation/hest_scoring.py",
    )
    file_hashes = {relative: _sha256(repository / relative) for relative in relative_files}
    try:
        git_head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_head = None
        git_dirty = None
    return {
        "git_head": git_head,
        "git_worktree_dirty_at_start": git_dirty,
        "file_sha256": file_hashes,
        "command": list(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
    }


def _configure_cuda_determinism() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError(
            "CUDA execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before process start"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _numeric_backend_receipt(device: str) -> Mapping[str, object]:
    gpu_name: str | None = None
    gpu_capability: list[int] | None = None
    gpu_total_memory_bytes: int | None = None
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        gpu_name = str(properties.name)
        gpu_capability = [int(value) for value in torch.cuda.get_device_capability(0)]
        gpu_total_memory_bytes = int(properties.total_memory)
    return {
        "requested_device": device,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "ridge_cuda_dtype": "float32" if device == "cuda" else "float64",
        "cpu_thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "gpu_name": gpu_name,
        "gpu_capability": gpu_capability,
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _base_report(
    source: HestSource,
    *,
    seed: int,
    device: str,
    inner_folds: int,
    phase: str,
    allow_gate_failed_uni2_baseline_only: bool,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "analysis_status": "retrospective_exposed_non_authorizing",
        "study_stage": "retrospective_exposed",
        "requested_phase": phase,
        "allow_gate_failed_uni2_baseline_only": (
            allow_gate_failed_uni2_baseline_only
        ),
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_reference_refinement": False,
        "authorizes_full_heir": False,
        "source": str(source.path),
        "source_sha256": source.sha256,
        "source_receipt_expected_sha256": source.sha256,
        "implementation_receipt": _implementation_receipt(),
        "donors": sorted(set(source.donors.tolist())),
        "sections": sorted(set(source.sections.tolist())),
        "encoder": source.encoder_name,
        "encoder_revision": source.encoder_revision,
        "encoder_manifest_sha256": source.encoder_manifest_sha256,
        "encoder_parity_receipt_sha256": source.encoder_parity_receipt_sha256 or None,
        "encoder_parity_receipt_path": source.encoder_parity_receipt_path or None,
        "encoder_comparison_source_sha256": (
            source.encoder_comparison_source_sha256 or None
        ),
        "encoder_comparison_non_encoder_identity_sha256": (
            source.encoder_comparison_non_encoder_identity_sha256 or None
        ),
        "encoder_comparison_receipt": source.encoder_comparison_receipt,
        "encoder_role": (
            "primary_Hoptimus1_qualification"
            if source.encoder_name == HOPTIMUS1_REPOSITORY
            else "prespecified_UNI2h_historical_comparator"
        ),
        "encoder_feature_width": 1536,
        "crop_contract": {
            "crop_ids": list(source.crop_ids),
            "crop_roles": list(source.crop_roles),
            "mask_modes": list(source.crop_mask_modes),
            "fill_modes": list(source.crop_fill_modes),
        },
        "artifact_control_boundary": _artifact_boundary(source),
        "folding": "15_outer_leave_one_biological_donor_out",
        "inner_folding": "deterministic_grouped_training_donors",
        "inner_folds": inner_folds,
        "alpha_grid": list(ALPHAS),
        "target_standardization": "training_fold_only",
        "pca_fit": "training_fold_only",
        "training_weighting": "equal_donor_then_type_then_cell",
        "primary_support": PRIMARY_SUPPORT,
        "support_sensitivities": list(SUPPORT_THRESHOLDS),
        "seed": seed,
        "numeric_backend": _numeric_backend_receipt(device),
        "limitations": [
            "All HEST outcomes were previously exposed; this analysis is non-authorizing.",
            "RNA-derived fine types may retain ontology dependence.",
            "The source caps each section/type/pool at 32 cells.",
            "Support thresholds 5, 10, and 20 retain byte-identical outcomes in this artifact.",
            "White-fill intrinsic crops lack matched mean/blur/random-location controls.",
            "Broad-lineage residual analyses can retain fine-subtype composition signal.",
        ],
    }


def _load_registered_same_runner_uni2_report(
    hoptimus_report: Mapping[str, object], comparator_report_path: Path
) -> tuple[Mapping[str, object], Path, str]:
    if hoptimus_report.get("encoder") != HOPTIMUS1_REPOSITORY:
        raise ValueError("paired encoder comparison requires an H-optimus-1 primary report")
    source_receipt = hoptimus_report.get("encoder_comparison_receipt")
    if (
        not isinstance(source_receipt, Mapping)
        or source_receipt.get("only_encoder_changed") is not True
    ):
        raise ValueError("H-optimus-1 report lacks an exact only-encoder-changed source receipt")
    resolved = comparator_report_path.expanduser().resolve()
    try:
        comparator_bytes = resolved.read_bytes()
        comparator = json.loads(comparator_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("UNI2-h comparison report is unreadable") from error
    comparator_sha256 = hashlib.sha256(comparator_bytes).hexdigest()
    if (
        not isinstance(comparator, Mapping)
        or comparator.get("encoder") != UNI2H_REPOSITORY
        or comparator.get("source_sha256") != REGISTERED_SOURCE_SHA256
    ):
        raise ValueError("comparison report is not the registered UNI2-h HEST analysis")
    return comparator, resolved, comparator_sha256


def _recompute_and_validate_stored_gate(
    report: Mapping[str, object], *, report_label: str
) -> Mapping[str, object]:
    controls = report.get("positive_controls")
    stored_gate = report.get("positive_control_gate")
    if not isinstance(controls, Mapping) or not isinstance(stored_gate, Mapping):
        raise ValueError(f"{report_label} lacks positive controls or its stored gate")
    try:
        recomputed_gate = positive_control_gate(controls)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{report_label} positive controls cannot regenerate its gate") from error
    if _canonical_sha256(recomputed_gate) != _canonical_sha256(stored_gate):
        raise ValueError(f"{report_label} stored positive-control gate was relabeled or tampered")
    return recomputed_gate


def _validate_paired_gate_roles(
    hoptimus_report: Mapping[str, object], comparator: Mapping[str, object]
) -> Mapping[str, object]:
    if hoptimus_report.get("allow_gate_failed_uni2_baseline_only") is not False:
        raise ValueError("H-optimus-1 primary has an invalid UNI2-h baseline-only opt-in")
    primary_gate = _recompute_and_validate_stored_gate(
        hoptimus_report, report_label="H-optimus-1 primary"
    )
    if (
        not isinstance(primary_gate, Mapping)
        or primary_gate.get("passed") is not True
        or primary_gate.get("molecular_interpretation_allowed") is not True
    ):
        raise ValueError("H-optimus-1 primary did not pass its positive-control gate")
    comparator_gate = _recompute_and_validate_stored_gate(
        comparator, report_label="UNI2-h comparator"
    )
    comparator_gate_passed = bool(
        comparator_gate.get("passed") is True
        and comparator_gate.get("molecular_interpretation_allowed") is True
    )
    comparator_baseline_only = False
    if comparator_gate_passed:
        stored_amendment = comparator.get("uni2_baseline_only_eligibility")
        if isinstance(stored_amendment, Mapping) and stored_amendment.get("eligible") is True:
            raise ValueError("gate-passing UNI2-h comparator has an invalid baseline-only receipt")
    elif (
        comparator_gate.get("passed") is False
        and comparator_gate.get("molecular_interpretation_allowed") is False
    ):
        if comparator.get("allow_gate_failed_uni2_baseline_only") is not True:
            raise ValueError("UNI2-h baseline-only comparator lacks the explicit opt-in")
        recomputed_amendment = uni2_baseline_only_eligibility(
            UNI2H_REPOSITORY, comparator_gate
        )
        stored_amendment = comparator.get("uni2_baseline_only_eligibility")
        if (
            recomputed_amendment.get("eligible") is not True
            or not isinstance(stored_amendment, Mapping)
            or _canonical_sha256(stored_amendment)
            != _canonical_sha256(recomputed_amendment)
        ):
            raise ValueError("UNI2-h comparator lacks an exact baseline-only eligibility receipt")
        if (
            comparator.get("analysis_status")
            != "retrospective_exposed_non_authorizing_baseline"
            or comparator.get("molecular_analysis_role")
            != "retrospective_exposed_non_authorizing_baseline"
            or comparator.get("comparison_inference_allowed") is not False
            or comparator.get("descriptive_only") is not True
        ):
            raise ValueError("UNI2-h baseline-only comparator role is not fail-closed")
        comparator_baseline_only = True
    else:
        raise ValueError(
            "UNI2-h comparator neither passed its gate nor qualified as baseline-only"
        )
    return {
        "primary_hoptimus_gate_passed": True,
        "comparator_gate_passed": comparator_gate_passed,
        "comparator_baseline_only": comparator_baseline_only,
        "comparison_inference_allowed": False,
        "descriptive_only": True,
        "primary_hoptimus_inference_independent": True,
        "baseline_only_amendment_timing": (
            UNI2_BASELINE_AMENDMENT_TIMING if comparator_baseline_only else None
        ),
    }


def _validate_runner_command(
    implementation: Mapping[str, object],
    *,
    report_label: str,
    expected_encoder: str,
    expected_phase: str,
    baseline_only: bool,
) -> None:
    command = implementation.get("command")
    if (
        not isinstance(command, Sequence)
        or isinstance(command, (str, bytes))
        or not command
        or any(not isinstance(value, str) for value in command)
    ):
        raise ValueError(f"{report_label} implementation command is missing or malformed")
    command_values = list(command)
    for option, expected in (
        ("--expected-encoder", expected_encoder),
        ("--phase", expected_phase),
    ):
        if command_values.count(option) != 1:
            raise ValueError(f"{report_label} command must contain {option} exactly once")
        index = command_values.index(option)
        if index + 1 >= len(command_values) or command_values[index + 1] != expected:
            raise ValueError(f"{report_label} command has an inconsistent {option}")
    opt_in = "--allow-gate-failed-uni2-baseline-only"
    expected_opt_in_count = 1 if baseline_only else 0
    if command_values.count(opt_in) != expected_opt_in_count:
        raise ValueError(
            f"{report_label} command has an invalid UNI2-h baseline-only opt-in count"
        )


def _validate_same_runner_uni2_contract(
    hoptimus_report: Mapping[str, object], comparator: Mapping[str, object]
) -> tuple[Mapping[str, object], str, Mapping[str, object]]:
    missing_contract = [
        field
        for field in PAIRED_ENCODER_CONTRACT_FIELDS
        if field not in hoptimus_report or field not in comparator
    ]
    if missing_contract:
        raise ValueError(
            "encoder reports lack frozen analysis contract fields: "
            + ", ".join(missing_contract)
        )
    for report_label, report in (
        ("H-optimus-1 primary", hoptimus_report),
        ("UNI2-h comparator", comparator),
    ):
        folds = report["inner_folds"]
        if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
            raise ValueError(f"{report_label} report has an invalid numeric inner_folds")
    gate_roles = _validate_paired_gate_roles(hoptimus_report, comparator)
    mismatched_contract = [
        field
        for field in PAIRED_ENCODER_CONTRACT_FIELDS
        if hoptimus_report.get(field) != comparator.get(field)
        and not (
            field == "analysis_status"
            and gate_roles["comparator_baseline_only"] is True
            and hoptimus_report.get(field)
            == "retrospective_exposed_non_authorizing"
            and comparator.get(field)
            == "retrospective_exposed_non_authorizing_baseline"
        )
    ]
    if mismatched_contract:
        raise ValueError(
            "encoder reports use different frozen analysis contracts: "
            + ", ".join(mismatched_contract)
        )
    requested_phase = hoptimus_report["requested_phase"]
    if requested_phase not in {"nucleus", "full"}:
        raise ValueError("paired encoder comparison requires a nucleus or full phase")
    expected_comparator_status = (
        "scientific_reanalysis_complete"
        if requested_phase == "full"
        else "observed_nucleus_program_models_complete"
    )
    if comparator.get("execution_status") != expected_comparator_status:
        raise ValueError(
            "UNI2-h comparator is incomplete for requested phase "
            f"{requested_phase}: expected {expected_comparator_status}"
        )
    primary_implementation = hoptimus_report.get("implementation_receipt")
    comparator_implementation = comparator.get("implementation_receipt")
    if not isinstance(primary_implementation, Mapping) or not isinstance(
        comparator_implementation, Mapping
    ):
        raise ValueError("encoder reports lack implementation receipts")
    _validate_runner_command(
        primary_implementation,
        report_label="H-optimus-1 primary",
        expected_encoder=HOPTIMUS1_REPOSITORY,
        expected_phase=str(requested_phase),
        baseline_only=False,
    )
    _validate_runner_command(
        comparator_implementation,
        report_label="UNI2-h comparator",
        expected_encoder=UNI2H_REPOSITORY,
        expected_phase=str(requested_phase),
        baseline_only=gate_roles["comparator_baseline_only"] is True,
    )
    primary_files = primary_implementation.get("file_sha256")
    comparator_files = comparator_implementation.get("file_sha256")
    if (
        not isinstance(primary_files, Mapping)
        or not isinstance(comparator_files, Mapping)
        or not primary_files
        or not comparator_files
        or primary_files != comparator_files
    ):
        raise ValueError("encoder reports were not generated by the same frozen implementation")
    missing_runtime = [
        field
        for field in PAIRED_IMPLEMENTATION_RUNTIME_FIELDS
        if field not in primary_implementation or field not in comparator_implementation
    ]
    if missing_runtime:
        raise ValueError(
            "encoder reports lack implementation runtime fields: "
            + ", ".join(missing_runtime)
        )
    mismatched_runtime = [
        field
        for field in PAIRED_IMPLEMENTATION_RUNTIME_FIELDS
        if primary_implementation.get(field) != comparator_implementation.get(field)
    ]
    if mismatched_runtime:
        raise ValueError(
            "encoder reports use different implementation runtimes: "
            + ", ".join(mismatched_runtime)
        )
    primary_backend = hoptimus_report.get("numeric_backend")
    comparator_backend = comparator.get("numeric_backend")
    if not isinstance(primary_backend, Mapping) or not isinstance(
        comparator_backend, Mapping
    ):
        raise ValueError("encoder reports lack numeric-backend receipts")
    missing_numeric = [
        field
        for field in PAIRED_ENCODER_NUMERIC_FIELDS
        if field not in primary_backend or field not in comparator_backend
    ]
    if missing_numeric:
        raise ValueError(
            "encoder reports lack numeric-backend contract fields: "
            + ", ".join(missing_numeric)
        )
    for report_label, backend in (
        ("H-optimus-1 primary", primary_backend),
        ("UNI2-h comparator", comparator_backend),
    ):
        if backend.get("requested_device") == "cuda":
            capability = backend.get("gpu_capability")
            total_memory = backend.get("gpu_total_memory_bytes")
            if (
                backend.get("cuda_available") is not True
                or not isinstance(backend.get("gpu_name"), str)
                or not backend.get("gpu_name")
                or not isinstance(capability, Sequence)
                or isinstance(capability, (str, bytes))
                or len(capability) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in capability
                )
                or isinstance(total_memory, bool)
                or not isinstance(total_memory, int)
                or total_memory <= 0
                or backend.get("deterministic_algorithms_enabled") is not True
                or backend.get("cublas_workspace_config") != ":4096:8"
                or backend.get("cudnn_deterministic") is not True
                or backend.get("cudnn_benchmark") is not False
                or backend.get("cuda_matmul_allow_tf32") is not False
                or backend.get("cudnn_allow_tf32") is not False
            ):
                raise ValueError(
                    f"{report_label} lacks the required deterministic CUDA contract"
                )
    if any(
        primary_backend.get(field) != comparator_backend.get(field)
        for field in PAIRED_ENCODER_NUMERIC_FIELDS
    ):
        raise ValueError("encoder reports use different numeric backends")
    primary_measurement = hoptimus_report.get("measurement")
    comparator_measurement = comparator.get("measurement")
    if not isinstance(primary_measurement, Mapping) or not isinstance(
        comparator_measurement, Mapping
    ) or not primary_measurement or not comparator_measurement:
        raise ValueError("encoder reports lack the shared measurement analysis")
    primary_measurement_sha256 = _canonical_sha256(primary_measurement)
    comparator_measurement_sha256 = _canonical_sha256(comparator_measurement)
    if primary_measurement_sha256 != comparator_measurement_sha256:
        raise ValueError("encoder reports use different molecular outcomes or evaluation masks")
    return primary_files, primary_measurement_sha256, gate_roles


def _validate_required_observed_effect_structure(
    observed: object, *, report_label: str
) -> Mapping[str, object]:
    if not isinstance(observed, Mapping):
        raise ValueError(f"{report_label} report lacks observed nucleus program analysis")
    missing_observed_contract = [
        field for field in PAIRED_OBSERVED_CONTRACT_FIELDS if field not in observed
    ]
    if missing_observed_contract:
        raise ValueError(
            f"{report_label} report lacks observed-program contract fields: "
            + ", ".join(missing_observed_contract)
        )
    for field in ("evaluation_mask_sha256", "program_outcome_sha256"):
        value = observed[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            raise ValueError(f"{report_label} report has an invalid {field}")
    support = observed["minimum_reference_and_scoring_support"]
    if isinstance(support, bool) or not isinstance(support, int) or support <= 0:
        raise ValueError(
            f"{report_label} report has invalid minimum_reference_and_scoring_support"
        )
    if tuple(observed.get("program_names", ())) != FROZEN_PROGRAM_NAMES:
        raise ValueError(
            f"{report_label} report does not contain the frozen six-program family"
        )
    observed_donors = observed.get("donors")
    if not isinstance(observed_donors, Sequence) or isinstance(
        observed_donors, (str, bytes)
    ):
        raise ValueError(f"{report_label} report lacks the frozen donor population")
    if len(observed_donors) != len(EXPECTED_DONORS) or set(observed_donors) != set(
        EXPECTED_DONORS
    ):
        raise ValueError(f"{report_label} report must contain all 15 frozen donors")
    representations = observed.get("representations")
    if not isinstance(representations, Mapping):
        raise ValueError(f"{report_label} report lacks representation-level effects")
    missing_representations = sorted(
        set(REQUIRED_PAIRED_ENCODER_REPRESENTATIONS) - set(representations)
    )
    if missing_representations:
        raise ValueError(
            f"{report_label} report lacks required full/PCA paired representations: "
            + ", ".join(missing_representations)
        )
    for representation in REQUIRED_PAIRED_ENCODER_REPRESENTATIONS:
        arm = representations[representation]
        if not isinstance(arm, Mapping):
            raise ValueError(
                f"{report_label} {representation} paired representation is malformed"
            )
        effects = arm.get("central_increment")
        if not isinstance(effects, Mapping) or set(effects) != set(FROZEN_PROGRAM_NAMES):
            raise ValueError(
                f"{report_label} {representation} lacks the complete frozen six-program family"
            )
        for program in FROZEN_PROGRAM_NAMES:
            effect = effects[program]
            if not isinstance(effect, Mapping):
                raise ValueError(
                    f"{report_label} {representation}/{program} effect is malformed"
                )
            per_donor = effect.get("per_donor_effect")
            if not isinstance(per_donor, Mapping) or set(per_donor) != set(EXPECTED_DONORS):
                raise ValueError(
                    f"{report_label} {representation}/{program} must contain all 15 "
                    "frozen donors"
                )
            try:
                values = np.asarray(
                    [per_donor[donor] for donor in EXPECTED_DONORS], dtype=np.float64
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{report_label} {representation}/{program} has non-numeric donor effects"
                ) from error
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{report_label} {representation}/{program} has non-finite donor effects"
                )
    return representations


def _validated_same_runner_uni2_report(
    hoptimus_report: Mapping[str, object], comparator_report_path: Path
) -> tuple[
    Mapping[str, object],
    Path,
    Mapping[str, object],
    str,
    Mapping[str, object],
    str,
]:
    comparator, resolved, comparator_sha256 = _load_registered_same_runner_uni2_report(
        hoptimus_report, comparator_report_path
    )
    implementation_files, measurement_sha256, gate_roles = (
        _validate_same_runner_uni2_contract(hoptimus_report, comparator)
    )
    _validate_required_observed_effect_structure(
        comparator.get("observed_nucleus_programs"), report_label="UNI2-h comparator"
    )
    return (
        comparator,
        resolved,
        implementation_files,
        measurement_sha256,
        gate_roles,
        comparator_sha256,
    )


def same_runner_uni2_comparator_preflight(
    hoptimus_report: Mapping[str, object], comparator_report_path: Path
) -> Mapping[str, object]:
    """Fail before H-optimus fitting unless its fixed UNI2-h comparator is usable."""

    (
        _comparator,
        resolved,
        implementation_files,
        measurement_sha256,
        gate_roles,
        comparator_sha256,
    ) = (
        _validated_same_runner_uni2_report(hoptimus_report, comparator_report_path)
    )
    return {
        "schema": "heir.hest_same_runner_uni2_preflight.v1",
        "passed": True,
        "comparison_report_path": str(resolved),
        "comparison_report_sha256": comparator_sha256,
        "analysis_contract_sha256": _canonical_sha256(
            {
                field: hoptimus_report[field]
                for field in PAIRED_ENCODER_CONTRACT_FIELDS
            }
        ),
        "implementation_file_sha256": dict(implementation_files),
        "shared_measurement_sha256": measurement_sha256,
        "required_representations": list(REQUIRED_PAIRED_ENCODER_REPRESENTATIONS),
        "required_programs": list(FROZEN_PROGRAM_NAMES),
        "required_donors": list(EXPECTED_DONORS),
        **gate_roles,
    }


def paired_encoder_comparison_report(
    hoptimus_report: Mapping[str, object], comparator_report_path: Path
) -> Mapping[str, object]:
    """Pair donor-level image increments across identity-matched encoder reports."""

    (
        comparator,
        resolved,
        primary_files,
        primary_measurement_sha256,
        gate_roles,
        comparator_sha256,
    ) = (
        _validated_same_runner_uni2_report(hoptimus_report, comparator_report_path)
    )
    preflight = hoptimus_report.get("same_runner_uni2_comparator_preflight")
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("passed") is not True
        or preflight.get("comparison_report_path") != str(resolved)
        or preflight.get("comparison_report_sha256") != comparator_sha256
    ):
        raise ValueError("UNI2-h comparator bytes changed after preflight")
    primary_observed = hoptimus_report.get("observed_nucleus_programs")
    primary_representations = _validate_required_observed_effect_structure(
        primary_observed, report_label="H-optimus-1 primary"
    )
    assert isinstance(primary_observed, Mapping)
    comparator_observed = comparator["observed_nucleus_programs"]
    assert isinstance(comparator_observed, Mapping)
    comparator_representations = _validate_required_observed_effect_structure(
        comparator_observed, report_label="UNI2-h comparator"
    )
    if any(
        primary_observed.get(field) != comparator_observed.get(field)
        for field in PAIRED_OBSERVED_CONTRACT_FIELDS
    ):
        raise ValueError("encoder reports use different observed-program analysis populations")
    representation_rows: dict[str, object] = {}
    for representation in REQUIRED_PAIRED_ENCODER_REPRESENTATIONS:
        primary_arm = primary_representations[representation]
        comparator_arm = comparator_representations[representation]
        assert isinstance(primary_arm, Mapping) and isinstance(comparator_arm, Mapping)
        primary_effects = primary_arm.get("central_increment")
        comparator_effects = comparator_arm.get("central_increment")
        assert isinstance(primary_effects, Mapping) and isinstance(
            comparator_effects, Mapping
        )
        program_rows: dict[str, object] = {}
        for program in FROZEN_PROGRAM_NAMES:
            primary_effect = primary_effects[program]
            comparator_effect = comparator_effects[program]
            assert isinstance(primary_effect, Mapping) and isinstance(
                comparator_effect, Mapping
            )
            primary_donors = primary_effect.get("per_donor_effect")
            comparator_donors = comparator_effect.get("per_donor_effect")
            assert isinstance(primary_donors, Mapping) and isinstance(
                comparator_donors, Mapping
            )
            delta = {
                str(donor): float(primary_donors[donor]) - float(comparator_donors[donor])
                for donor in EXPECTED_DONORS
            }
            values = np.asarray(list(delta.values()), dtype=np.float64)
            program_rows[str(program)] = {
                "estimand": "Hoptimus1_minus_UNI2h_donor_image_increment",
                "per_donor_delta": delta,
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "Hoptimus1_better_fraction": float(np.mean(values > 0.0)),
            }
        if tuple(program_rows) != FROZEN_PROGRAM_NAMES:
            raise ValueError("paired encoder output is missing frozen program rows")
        representation_rows[str(representation)] = {"programs": program_rows}
    if tuple(representation_rows) != REQUIRED_PAIRED_ENCODER_REPRESENTATIONS:
        raise ValueError("paired encoder output is missing required representation rows")
    return {
        "schema": "heir.hest_paired_encoder_comparison.v1",
        "scope": "retrospective_exposed_non_authorizing",
        "same_cells_and_non_encoder_design_verified": True,
        "same_analysis_implementation_and_contract_verified": True,
        "analysis_contract_sha256": _canonical_sha256(
            {
                field: hoptimus_report[field]
                for field in PAIRED_ENCODER_CONTRACT_FIELDS
            }
        ),
        "implementation_file_sha256": dict(primary_files),
        "shared_measurement_sha256": primary_measurement_sha256,
        "comparison_source_sha256": REGISTERED_SOURCE_SHA256,
        "comparison_report_path": str(resolved),
        "comparison_report_sha256": comparator_sha256,
        "primary_encoder": HOPTIMUS1_REPOSITORY,
        "comparator_encoder": UNI2H_REPOSITORY,
        **gate_roles,
        "representations": representation_rows,
    }


def _format_number(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _write_uni2_baseline_markdown(
    path: Path, report: Mapping[str, object]
) -> None:
    lines = [
        "# HEST UNI2-h descriptive baseline",
        "",
        "> Retrospective, outcome-exposed, descriptive-only, and permanently "
        "non-authorizing.",
        "",
        "## Frozen role amendment",
        "",
        "The original UNI2-h positive-control gate remains **FAIL**, and its "
        "molecular-interpretation flag remains **blocked**. The explicit default-off "
        "opt-in permits this run only because all four geometry controls failed while "
        "both type controls and all three appearance controls passed.",
        f"Amendment timing: `{UNI2_BASELINE_AMENDMENT_TIMING}`.",
        "Comparator inference is prohibited. Numeric molecular-model results below are "
        "descriptive deltas only.",
        "",
        "## Receipt",
        "",
        f"- Source SHA-256: `{report['source_sha256']}`",
        "- Runner SHA-256: `%s`"
        % report["implementation_receipt"]["file_sha256"][
            "scripts/benchmark_hest_scientific_reanalysis.py"
        ],
        f"- Encoder: {report['encoder']}",
    ]
    summary = report.get("descriptive_baseline_summary")
    if isinstance(summary, Mapping):
        lines.extend(
            [
                "",
                "## Descriptive program deltas",
                "",
                f"Representation: `{summary['representation']}`.",
                "",
                "| Program | Mean delta R2 | Positive donors |",
                "|---|---:|---:|",
            ]
        )
        for program, effect in summary["program_effects"].items():
            lines.append(
                "| %s | %s | %s |"
                % (
                    program,
                    _format_number(effect["mean_delta_r2"]),
                    _format_number(effect["positive_donor_fraction"], 3),
                )
            )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This baseline does not authorize a molecular claim, an encoder comparison "
            "claim, H-CELL, H-INTRINSIC, reference refinement, or full HEIR development.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown(path: Path, report: Mapping[str, object]) -> None:
    baseline_receipt = report.get("uni2_baseline_only_eligibility")
    if report.get("encoder") == UNI2H_REPOSITORY and (
        report.get("analysis_status")
        == "retrospective_exposed_non_authorizing_baseline"
        or (
            isinstance(baseline_receipt, Mapping)
            and baseline_receipt.get("eligible") is True
        )
    ):
        _write_uni2_baseline_markdown(path, report)
        return
    summary = report.get("scientific_summary")
    primary_probe = (
        summary.get("primary_revised_probe")
        if isinstance(summary, Mapping)
        else None
    )
    primary_description = (
        "full 1,536-dimensional frozen H-optimus-1 features with separate broad-lineage "
        "ridge heads"
        if primary_probe == "full_1536_broad_lineage_heads"
        else "512-component training-only PCA with separate broad-lineage ridge heads"
    )
    lines = [
        "# HEST scientific reanalysis",
        "",
        "> Retrospective, outcome-exposed, and permanently non-authorizing.",
        "",
        "## Receipt and scope",
        "",
        f"- Source SHA-256: `{report['source_sha256']}`",
        "- Runner SHA-256: `%s`"
        % report["implementation_receipt"]["file_sha256"][
            "scripts/benchmark_hest_scientific_reanalysis.py"
        ],
        f"- Git HEAD: `{report['implementation_receipt']['git_head']}`; "
        f"dirty at run start: `{report['implementation_receipt']['git_worktree_dirty_at_start']}`",
        f"- Donors: {len(report['donors'])}; sections: {len(report['sections'])}",
        f"- Encoder: full frozen {report['encoder']} (1,536 dimensions), train-only PCA "
        "sensitivities, and frozen 96-dimensional random-projection sensitivities.",
        "- Outer evaluation: leave one biological donor out; alpha/control selection: "
        "inner grouped training-donor CV.",
        "",
        "## Measurement reliability",
        "",
        "Spearman–Brown values below are medians across evaluable donor/type strata.",
        "",
        "| Program | Nucleus overlap | Nucleus primary eligible | Whole-cell sensitivity |",
        "|---|---:|---:|---:|",
    ]
    if "measurement" in report:
        nucleus = report["measurement"]["nucleus_overlap"]["programs"]["donor_type_macro"][
            "features"
        ]
        whole = report["measurement"]["whole_cell_assignment_sensitivity"]["programs"][
            "donor_type_macro"
        ]["features"]
        nucleus_eligible = set(
            report["measurement"]["nucleus_overlap"]["outer_training_program_reliability_gate"][
                "programs_eligible_for_primary_inference"
            ]
        )
        for name in nucleus:
            lines.append(
                "| %s | %s | %s | %s |"
                % (
                    name,
                    _format_number(nucleus[name]["median_spearman_brown_reliability"]),
                    "yes" if name in nucleus_eligible else "no (sensitivity)",
                    _format_number(whole[name]["median_spearman_brown_reliability"]),
                )
            )
    else:
        lines.append("| Pending | NA | NA | NA |")
    lines.extend(
        [
            "",
            "## Revised primary program probe",
            "",
            "Primary revised probe: "
            + primary_description
            + ", donor/type-balanced fitting, and a nested-selected control.",
            "",
            "| Program | Role | Ceiling | Control R2 | Control + image R2 | Delta R2 | "
            "Positive donors | Holm p |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if summary is not None:
        for name, row in summary["programs"].items():
            effect = row["image_increment"]
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    name,
                    ("primary" if row["primary_reliability_eligible"] else "sensitivity"),
                    _format_number(row["measurement_ceiling_spearman_brown_donor_type_median"]),
                    _format_number(row["control_donor_type_macro_r2"]),
                    _format_number(row["model_donor_type_macro_r2"]),
                    _format_number(effect["mean_effect"]),
                    _format_number(effect["positive_fraction"], 3),
                    _format_number(effect["holm_adjusted_exact_sign_flip_p"], 4),
                )
            )
        lines.extend(
            [
                "",
                "Decision: **%s**. This cannot authorize H-CELL or reference refinement."
                % summary["h_cell_retrospective_status"],
            ]
        )
    else:
        lines.append("| Pending | NA | NA | NA | NA | NA | NA | NA |")
    if "reference_support_sensitivity" in report:
        lines.extend(
            [
                "",
                "## Reference-support sensitivity",
                "",
                "Thresholds are reused only for byte-identical residual outcomes; otherwise "
                "reference means, nested tuning, and held-out models are rerun.",
                "",
                "| Minimum support | Evaluation rows | Execution |",
                "|---:|---:|---|",
            ]
        )
        for threshold, row in report["reference_support_sensitivity"]["thresholds"].items():
            lines.append("| %s | %s | %s |" % (threshold, row["evaluation_rows"], row["execution"]))
        support_30 = report["reference_support_sensitivity"]["thresholds"]["30"]
        if "analysis" in support_30:
            effects_30 = support_30["analysis"]["representations"]["pca_512_broad_lineage_heads"][
                "central_increment"
            ]
            primary_programs = report["scientific_summary"]["primary_inference_programs"]
            lines.extend(
                [
                    "",
                    "Support-30 primary-program Delta R2: "
                    + "; ".join(
                        f"{name} {_format_number(effects_30[name]['mean_effect'])}"
                        for name in primary_programs
                    )
                    + ".",
                ]
            )
    if "observed_whole_cell_program_sensitivity" in report:
        whole_analysis = report["observed_whole_cell_program_sensitivity"]
        whole_primary = report.get("scientific_summary", {}).get(
            "primary_revised_probe", "pca_512_broad_lineage_heads"
        )
        whole_effects = whole_analysis["representations"][whole_primary]["central_increment"]
        lines.extend(
            [
                "",
                "## Whole-cell-assignment sensitivity",
                "",
                "No program is treated as supported unless it passes the complete "
                "directional-and-error prefilter.",
                "",
                "| Program | Delta R2 | Positive donors | Holm p |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, effect in whole_effects.items():
            lines.append(
                "| %s | %s | %s | %s |"
                % (
                    name,
                    _format_number(effect["mean_effect"]),
                    _format_number(effect["positive_fraction"], 3),
                    _format_number(effect["holm_adjusted_exact_sign_flip_p"]),
                )
            )
    if "positive_controls" in report:
        broad = report["positive_controls"]["broad_lineage"]
        fine = report["positive_controls"]["fine_type"]
        morphology_full = report["positive_controls"]["nuclear_morphology"]["full_context"][
            "scores"
        ]
        morphology_nucleus = report["positive_controls"]["nuclear_morphology"]["nucleus_mask_only"][
            "scores"
        ]
        lines.extend(
            [
                "",
                "## Image-pipeline positive controls",
                "",
                "| Endpoint | Model | Training-majority baseline |",
                "|---|---:|---:|",
                "| Broad lineage, donor-macro balanced accuracy | %s | %s |"
                % (
                    _format_number(
                        broad["model"]["donor_balanced"]["donor_macro_balanced_accuracy"]
                    ),
                    _format_number(
                        broad["training_majority_baseline"]["donor_balanced"][
                            "donor_macro_balanced_accuracy"
                        ]
                    ),
                ),
                "| Fine type, donor-macro balanced accuracy | %s | %s |"
                % (
                    _format_number(
                        fine["model"]["donor_balanced"]["donor_macro_balanced_accuracy"]
                    ),
                    _format_number(
                        fine["training_majority_baseline"]["donor_balanced"][
                            "donor_macro_balanced_accuracy"
                        ]
                    ),
                ),
                "| Nuclear morphology from full context, target-macro donor/type R2 | %s | NA |"
                % _format_number(morphology_full["target_macro"]["donor_type_macro_r2"]),
                "| Nuclear morphology from nucleus crop, target-macro donor/type R2 | %s | NA |"
                % _format_number(morphology_nucleus["target_macro"]["donor_type_macro_r2"]),
            ]
        )
        gate = report.get("positive_control_gate")
        if isinstance(gate, Mapping):
            lines.extend(
                [
                    "",
                    "Prespecified positive-control gate: **%s**. Molecular interpretation: "
                    "**%s**."
                    % (
                        "PASS" if gate["passed"] else "FAIL",
                        "allowed" if gate["molecular_interpretation_allowed"] else "blocked",
                    ),
                    "The natural unmasked 112-um crop is primary; the nucleus-mask arm is "
                    "secondary and is not used to pass this gate.",
                ]
            )
    baseline_receipt = report.get("uni2_baseline_only_eligibility")
    paired_receipt = report.get("paired_encoder_comparison")
    preflight_receipt = report.get("same_runner_uni2_comparator_preflight")
    baseline_only_disclosed = bool(
        isinstance(baseline_receipt, Mapping)
        and baseline_receipt.get("eligible") is True
    ) or bool(
        isinstance(paired_receipt, Mapping)
        and paired_receipt.get("comparator_baseline_only") is True
    ) or bool(
        isinstance(preflight_receipt, Mapping)
        and preflight_receipt.get("comparator_baseline_only") is True
    )
    if baseline_only_disclosed:
        lines.extend(
            [
                "",
                "## Prespecified UNI2-h baseline-only amendment",
                "",
                "The original UNI2-h positive-control gate remains **FAIL**, and its "
                "molecular-interpretation flag remains **blocked**. The explicit opt-in "
                "permits only the exact geometry-confined failure pattern to continue as a "
                "retrospective, exposed, non-authorizing descriptive baseline.",
                f"Amendment timing: `{UNI2_BASELINE_AMENDMENT_TIMING}`.",
                "Comparator inference is prohibited; H-optimus-1 remains the independent "
                "primary analysis and must pass the complete visible-control gate.",
            ]
        )
    if "exploratory_crop_sensitivity" in report:
        crop_report = report["exploratory_crop_sensitivity"]
        lines.extend(
            [
                "",
                "## Direct white-fill crop contrasts",
                "",
                "All four arms were refit with one control plan and identical inner donor "
                "folds. These paired contrasts remain artifact-confounded because matched "
                "fill/blur/random-location crops are absent.",
                "",
                "| Contrast | Program | Delta R2 | Positive donors | Holm p |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for contrast, programs in crop_report["direct_paired_contrasts"].items():
            for name, effect in programs.items():
                lines.append(
                    "| %s | %s | %s | %s | %s |"
                    % (
                        contrast,
                        name,
                        _format_number(effect["mean_effect"]),
                        _format_number(effect["positive_fraction"], 3),
                        _format_number(effect["holm_adjusted_across_all_crop_program_contrasts"]),
                    )
                )
    boundary = report["artifact_control_boundary"]
    lines.extend(
        [
            "",
            "## Intrinsic morphology boundary",
            "",
            "Cell intrinsic: **%s**; nucleus intrinsic: **%s**."
            % (
                boundary["h_intrinsic_cell_status"],
                boundary["h_intrinsic_nucleus_status"],
            ),
            "The current source lacks the six matched mean-fill, blurred, and random-location "
            "crop embeddings. White-fill contrasts therefore remain artifact-confounded and "
            "cannot validate H-INTRINSIC.",
            "",
            "## Claim boundary",
            "",
            "No result in this report authorizes H-CELL, H-INTRINSIC, iterative reference "
            "refinement, matched-reference testing, or full HEIR development.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def benchmark(
    source_path: Path,
    output: Path,
    markdown_output: Optional[Path],
    *,
    phase: str,
    device: str,
    inner_folds: int,
    seed: int,
    representation_profile: str = "full",
    enforce_registered_hash: bool = True,
    expected_source_sha256: str | None = None,
    expected_encoder: str = UNI2H_REPOSITORY,
    comparison_source_path: Path | None = None,
    comparison_report_path: Path | None = None,
    allow_gate_failed_uni2_baseline_only: bool = False,
) -> None:
    started = time.monotonic()
    if allow_gate_failed_uni2_baseline_only and expected_encoder != UNI2H_REPOSITORY:
        raise ValueError("UNI2-h baseline-only opt-in is invalid for H-optimus-1")
    if allow_gate_failed_uni2_baseline_only and phase not in {"nucleus", "full"}:
        raise ValueError("UNI2-h baseline-only opt-in requires a nucleus or full phase")
    if comparison_report_path is not None and expected_encoder != HOPTIMUS1_REPOSITORY:
        raise ValueError("paired encoder report is accepted only for H-optimus-1 qualification")
    if (
        expected_encoder == HOPTIMUS1_REPOSITORY
        and phase in {"nucleus", "full"}
        and comparison_report_path is None
    ):
        raise ValueError(
            "H-optimus-1 nucleus/full qualification requires a same-runner UNI2-h report"
        )
    source = load_source(
        source_path,
        enforce_registered_hash=enforce_registered_hash,
        expected_source_sha256=expected_source_sha256,
        expected_encoder=expected_encoder,
        comparison_source_path=comparison_source_path,
    )
    hoptimus_qualification = source.encoder_name == HOPTIMUS1_REPOSITORY
    full_representation_set = hoptimus_qualification or representation_profile == "full"
    report = _base_report(
        source,
        seed=seed,
        device=device,
        inner_folds=inner_folds,
        phase=phase,
        allow_gate_failed_uni2_baseline_only=(
            allow_gate_failed_uni2_baseline_only
        ),
    )
    report["execution_status"] = "measurement_in_progress"
    _write_json(output, report)
    report["measurement"] = measurement_analysis(source)
    report["execution_status"] = "measurement_complete"
    report["elapsed_seconds"] = float(time.monotonic() - started)
    _write_json(output, report)
    if markdown_output is not None and not allow_gate_failed_uni2_baseline_only:
        _write_markdown(markdown_output, report)
    if phase == "measurement":
        _log(f"HEST reanalysis measurement complete: {output}")
        return

    report["execution_status"] = "positive_controls_in_progress"
    _write_json(output, report)
    report["positive_controls"] = positive_control_analysis(
        source,
        device=device,
        inner_folds=inner_folds,
        seed=seed + 200_000,
    )
    report["positive_control_gate"] = positive_control_gate(report["positive_controls"])
    gate_passed = report["positive_control_gate"]["passed"] is True
    baseline_only = False
    if (
        not gate_passed
        and source.encoder_name == UNI2H_REPOSITORY
        and allow_gate_failed_uni2_baseline_only
    ):
        baseline_receipt = uni2_baseline_only_eligibility(
            source.encoder_name, report["positive_control_gate"]
        )
        report["uni2_baseline_only_eligibility"] = baseline_receipt
        baseline_only = baseline_receipt["eligible"] is True
        if baseline_only:
            report["analysis_status"] = (
                "retrospective_exposed_non_authorizing_baseline"
            )
            report["encoder_role"] = (
                "prespecified_UNI2h_retrospective_exposed_non_authorizing_baseline"
            )
            report["molecular_analysis_role"] = (
                "retrospective_exposed_non_authorizing_baseline"
            )
            report["comparison_inference_allowed"] = False
            report["descriptive_only"] = True
    molecular_fitting_allowed = gate_passed or baseline_only
    if gate_passed:
        report["execution_status"] = "positive_controls_complete"
    elif baseline_only:
        report["execution_status"] = (
            "positive_controls_complete_uni2_baseline_only_eligible"
        )
    else:
        report["execution_status"] = "blocked_positive_control_gate_failed"
    report["elapsed_seconds"] = float(time.monotonic() - started)
    _write_json(output, report)
    if markdown_output is not None:
        _write_markdown(markdown_output, report)
    if phase == "positive" or not molecular_fitting_allowed:
        _log(f"HEST reanalysis positive-control gate complete: {output}")
        return

    if comparison_report_path is not None:
        report["execution_status"] = "same_runner_uni2_comparator_preflight_in_progress"
        _write_json(output, report)
        try:
            report["same_runner_uni2_comparator_preflight"] = (
                same_runner_uni2_comparator_preflight(report, comparison_report_path)
            )
        except ValueError as error:
            report["same_runner_uni2_comparator_preflight"] = {
                "schema": "heir.hest_same_runner_uni2_preflight.v1",
                "passed": False,
                "error": str(error),
            }
            report["execution_status"] = (
                "blocked_same_runner_uni2_comparator_preflight_failed"
            )
            report["elapsed_seconds"] = float(time.monotonic() - started)
            _write_json(output, report)
            if markdown_output is not None:
                _write_markdown(markdown_output, report)
            raise
        report["execution_status"] = "same_runner_uni2_comparator_preflight_complete"
        report["elapsed_seconds"] = float(time.monotonic() - started)
        _write_json(output, report)

    report["execution_status"] = "observed_nucleus_program_models_in_progress"
    _write_json(output, report)
    report["observed_nucleus_programs"] = observed_program_analysis(
        source,
        target_scope="nucleus_overlap",
        targets=source.nucleus_targets,
        roles=source.roles,
        device=device,
        inner_folds=inner_folds,
        seed=seed,
        full_representation_sensitivity=full_representation_set,
    )
    if baseline_only:
        report["descriptive_baseline_summary"] = _descriptive_uni2_baseline_summary(
            report["observed_nucleus_programs"]
        )
    else:
        report["scientific_summary"] = _primary_probe_summary(
            report["observed_nucleus_programs"],
            report["measurement"],
            primary_representation=(
                "full_1536_broad_lineage_heads"
                if hoptimus_qualification
                else "pca_512_broad_lineage_heads"
            ),
        )
    if comparison_report_path is not None:
        report["paired_encoder_comparison"] = paired_encoder_comparison_report(
            report, comparison_report_path
        )
    report["execution_status"] = "reference_support_sensitivity_in_progress"
    report["elapsed_seconds"] = float(time.monotonic() - started)
    _write_json(output, report)
    report["reference_support_sensitivity"] = reference_support_sensitivity_analysis(
        source,
        report["observed_nucleus_programs"],
        device=device,
        inner_folds=inner_folds,
        seed=seed + 500_000,
    )
    report["execution_status"] = "observed_nucleus_program_models_complete"
    report["elapsed_seconds"] = float(time.monotonic() - started)
    _write_json(output, report)
    if markdown_output is not None:
        _write_markdown(markdown_output, report)
    if phase == "nucleus":
        _log(f"HEST reanalysis nucleus program phase complete: {output}")
        return

    report["execution_status"] = "whole_cell_sensitivity_in_progress"
    _write_json(output, report)
    report["observed_whole_cell_program_sensitivity"] = observed_program_analysis(
        source,
        target_scope="whole_cell_assignment_sensitivity",
        targets=source.whole_cell_targets,
        roles=source.roles,
        device=device,
        inner_folds=inner_folds,
        seed=seed + 100_000,
        full_representation_sensitivity=hoptimus_qualification,
    )
    report["execution_status"] = "crop_sensitivity_in_progress"
    report["elapsed_seconds"] = float(time.monotonic() - started)
    _write_json(output, report)
    report["exploratory_crop_sensitivity"] = crop_sensitivity_analysis(
        source,
        device=device,
        inner_folds=inner_folds,
        seed=seed + 300_000,
    )
    report["execution_status"] = "reference_split_sensitivity_in_progress"
    report["elapsed_seconds"] = float(time.monotonic() - started)
    _write_json(output, report)
    report["reference_split_sensitivity"] = reference_split_sensitivity_analysis(
        source,
        device=device,
        inner_folds=inner_folds,
        seed=seed + 400_000,
    )
    report["execution_status"] = "scientific_reanalysis_complete"
    report["elapsed_seconds"] = float(time.monotonic() - started)
    _write_json(output, report)
    if markdown_output is not None:
        _write_markdown(markdown_output, report)
    _log(f"HEST scientific reanalysis complete: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--phase",
        choices=("measurement", "positive", "nucleus", "full"),
        default="full",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--representation-profile", choices=("primary", "full"), default="full")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--max-gpu-memory-gb", type=float, default=8.0)
    parser.add_argument(
        "--expected-source-sha256",
        default=REGISTERED_SOURCE_SHA256,
        help="immutable SHA-256 of the source archive being qualified",
    )
    parser.add_argument(
        "--expected-encoder",
        choices=(UNI2H_REPOSITORY, HOPTIMUS1_REPOSITORY),
        default=UNI2H_REPOSITORY,
    )
    parser.add_argument(
        "--comparison-source",
        type=Path,
        default=None,
        help="registered UNI2-h source required for H-optimus-1 only-encoder-changed validation",
    )
    parser.add_argument(
        "--comparison-report",
        type=Path,
        default=None,
        help="completed registered UNI2-h report for paired donor-level encoder contrasts",
    )
    parser.add_argument(
        "--allow-gate-failed-uni2-baseline-only",
        action="store_true",
        help=(
            "explicitly allow only the frozen geometry-only UNI2-h gate-failed pattern "
            "to run as a descriptive non-authorizing baseline"
        ),
    )
    args = parser.parse_args()
    if args.inner_folds < 2 or args.torch_threads < 1:
        raise ValueError("inner folds and torch thread count are too small")
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = _resolve_device(args.device)
    if device == "cuda":
        _configure_cuda_determinism()
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        fraction = min(max(args.max_gpu_memory_gb / total_gb, 0.1), 0.9)
        torch.cuda.set_per_process_memory_fraction(fraction)
    benchmark(
        args.source,
        args.output,
        args.markdown_output,
        phase=args.phase,
        device=device,
        inner_folds=args.inner_folds,
        seed=args.seed,
        representation_profile=args.representation_profile,
        expected_source_sha256=args.expected_source_sha256,
        expected_encoder=args.expected_encoder,
        comparison_source_path=args.comparison_source,
        comparison_report_path=args.comparison_report,
        allow_gate_failed_uni2_baseline_only=(
            args.allow_gate_failed_uni2_baseline_only
        ),
    )


if __name__ == "__main__":
    main()
