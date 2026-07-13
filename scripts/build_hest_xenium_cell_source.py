#!/usr/bin/env python3
"""Build a pinned HEST Xenium cell source for the morphology-state gate.

Native Xenium cell IDs are the only registration key.  Native nucleus polygons
provide H&E crop centres; CellViT, when supplied, is exported as a separate
sensitivity matrix and never changes target membership or registration.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from heir.data.study_manifest import (
    LABEL_TARGET_INDEPENDENCE_PROTOCOL_FIELDS,
    StudyManifest,
)
from heir.features import EncoderManifest, FrozenPatchEncoder, create_frozen_encoder
from heir.features import load_encoder_manifest as _load_encoder_manifest

PROTOCOL_SCHEMA = "heir.hest_xenium_cell_protocol.v4"
SOURCE_SCHEMA = "heir.registered_observations.v4"
ANNOTATION_RECEIPT_SCHEMA = "heir.independent_annotation_receipt.v1"
ANNOTATION_CROSS_FIT_SCHEMA = "heir.annotation_cross_fitting_receipt.v1"
ANNOTATION_PREDICTION_COLUMNS = ("hest_id", "cell_id", "broad_lineage", "fine_type")
DATASET_REPO = "MahmoodLab/hest"
DATASET_REVISION = "7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec"
SOURCE_MPP = 0.2125
ANNOTATION_SHA256 = "4c4b0d159569a3ff86753b700f28a14807d00639788cb7aba2d675738e243423"
ANNOTATION_ROWS = 938_345
TYPE_NAMES = ("Endothelial", "Epithelial", "Immune", "Mesenchymal")
CONTROL_PREFIXES = (
    "NegControlProbe_",
    "NegControlCodeword_",
    "UnassignedCodeword_",
    "BLANK_",
)
DEVELOPMENT_DONORS = (
    "VUILD91",
    "TILD175",
    "VUHD069",
    "VUHD116",
    "VUILD102",
    "VUILD105",
    "VUILD106",
    "VUILD107",
    "VUILD110",
    "VUILD115",
)
LOCKED_TEST_DONORS = ("THD0008", "THD0011", "VUILD78", "VUILD96", "TILD117")
SECTION_IDENTITIES = {
    "NCBI856": ("VUILD96", "VUILD96LA"),
    "NCBI857": ("VUILD96", "VUILD96MA"),
    "NCBI858": ("VUILD91", "VUILD91LA"),
    "NCBI859": ("VUILD91", "VUILD91MA"),
    "NCBI860": ("VUILD78", "VUILD78LA"),
    "NCBI861": ("VUILD78", "VUILD78MA"),
    "NCBI864": ("VUILD115", "VUILD115MA"),
    "NCBI865": ("VUILD110", "VUILD110LA"),
    "NCBI866": ("VUILD107", "VUILD107MA"),
    "NCBI867": ("VUILD106", "VUILD106MA"),
    "NCBI870": ("VUILD105", "VUILD105MA2"),
    "NCBI873": ("VUILD102", "VUILD102LA"),
    "NCBI875": ("VUHD116", "VUHD116B"),
    "NCBI876": ("VUHD116", "VUHD116A"),
    "NCBI879": ("VUHD069", "VUHD069"),
    "NCBI880": ("TILD175", "TILD175MA"),
    "NCBI881": ("TILD117", "TILD117LA"),
    "NCBI882": ("TILD117", "TILD117MA1"),
    "NCBI883": ("THD0011", "THD0011"),
    "NCBI884": ("THD0008", "THD0008"),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _section_scoped_cell_id(sample_id: str, raw_cell_id: str) -> str:
    """Return the globally unique identity of a section-local Xenium cell."""

    section = str(sample_id).strip()
    cell = str(raw_cell_id).strip()
    if not section or not cell or ":" in section:
        raise ValueError("HEST section and raw cell identities must be nonempty and unambiguous")
    return section + ":" + cell


def _validate_protocol_independence_contract(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != LABEL_TARGET_INDEPENDENCE_PROTOCOL_FIELDS:
        raise ValueError("HEST label-target independence protocol is incomplete")
    raw_annotation_ids = value["ordered_annotation_feature_ids"]
    raw_training_donor_ids = value["annotation_training_donor_ids"]
    if not isinstance(raw_annotation_ids, list) or not isinstance(raw_training_donor_ids, list):
        raise ValueError("HEST annotation feature/training IDs must be lists")
    annotation_ids = tuple(str(item) for item in raw_annotation_ids)
    training_donor_ids = tuple(str(item) for item in raw_training_donor_ids)
    if (
        len(set(annotation_ids)) != len(annotation_ids)
        or len(set(training_donor_ids)) != len(training_donor_ids)
        or any(not item.strip() for item in annotation_ids)
        or any(not item.strip() for item in training_donor_ids)
        or not isinstance(value["same_cohort_annotation"], bool)
        or not isinstance(value["establishes_full_target_independence"], bool)
        or not str(value["strategy"]).strip()
        or not str(value["limitation"]).strip()
    ):
        raise ValueError("HEST label-target independence protocol is malformed")
    evidence_kind = str(value["evidence_kind"])
    if evidence_kind == "pending":
        if (
            value["annotation_receipt_sha256"] is not None
            or annotation_ids
            or value["ordered_annotation_feature_ids_sha256"] is not None
            or value["annotation_training_scope"] != "unknown_pending_provenance"
            or training_donor_ids
            or value["annotation_training_donor_ids_sha256"] is not None
            or value["locked_donors_used_for_training"] is not None
            or value["cross_fitting_method"] != "pending"
            or value["cross_fitting_receipt_sha256"] is not None
            or value["establishes_full_target_independence"] is not False
        ):
            raise ValueError("pending HEST label-target evidence overstates independence")
        return value
    if evidence_kind not in {
        "external_gene_disjoint_annotation",
        "development_donor_cross_fitted_gene_disjoint_annotation",
        "orthogonal_modality_annotation",
    }:
        raise ValueError("HEST label-target evidence kind is unsupported")
    if (
        not _is_sha256(value["annotation_receipt_sha256"])
        or not annotation_ids
        or value["ordered_annotation_feature_ids_sha256"] != _canonical_sha256(list(annotation_ids))
        or value["locked_donors_used_for_training"] is not False
        or value["establishes_full_target_independence"] is not True
    ):
        raise ValueError("HEST label-target evidence is not receipt- and feature-bound")
    if value["same_cohort_annotation"] is True:
        if (
            evidence_kind != "development_donor_cross_fitted_gene_disjoint_annotation"
            or value["annotation_training_scope"] != "development_donors_only"
            or not training_donor_ids
            or value["annotation_training_donor_ids_sha256"]
            != _canonical_sha256(list(training_donor_ids))
            or value["cross_fitting_method"] != "leave_one_donor_out"
            or not _is_sha256(value["cross_fitting_receipt_sha256"])
        ):
            raise ValueError("same-cohort HEST annotation is not donor-cross-fitted")
    elif evidence_kind == "external_gene_disjoint_annotation":
        if (
            value["annotation_training_scope"] != "external_donors_only"
            or not training_donor_ids
            or value["annotation_training_donor_ids_sha256"]
            != _canonical_sha256(list(training_donor_ids))
            or value["cross_fitting_method"] != "not_applicable"
            or value["cross_fitting_receipt_sha256"] is not None
        ):
            raise ValueError("external HEST annotation training scope is invalid")
    elif evidence_kind == "orthogonal_modality_annotation":
        if (
            value["annotation_training_scope"] != "orthogonal_no_rna_training"
            or training_donor_ids
            or value["annotation_training_donor_ids_sha256"] is not None
            or value["cross_fitting_method"] != "not_applicable"
            or value["cross_fitting_receipt_sha256"] is not None
        ):
            raise ValueError("orthogonal HEST annotation training scope is invalid")
    else:
        raise ValueError(
            "HEST label-target evidence kind conflicts with same-cohort annotation scope"
        )
    return value


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("HEST Xenium protocol is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("HEST Xenium protocol must be a JSON object")
    return value


@dataclass(frozen=True)
class CropVariant:
    crop_id: str
    role: str
    comparison_family: str
    diameter_um: float
    mask_mode: str
    fill_mode: str
    inner_diameter_um: float
    model_input_pixels: int
    effective_mpp: float


@dataclass(frozen=True)
class CropManifest:
    path: Path
    sha256: str
    source_mpp: float
    padding: str
    primary_crop_id: str
    random_mask_salt: str
    blur_sigma_um: float
    variants: Tuple[CropVariant, ...]


def _load_crop_manifest(path: Path) -> CropManifest:
    resolved = path.expanduser().resolve()
    value = _read_json(resolved)
    required = {
        "schema",
        "source_mpp",
        "padding",
        "primary_crop_id",
        "random_mask_salt",
        "blur_sigma_um",
        "variants",
    }
    if value.get("schema") != "heir.crop_manifest.v1" or not required <= set(value):
        raise ValueError("crop manifest is incomplete or unsupported")
    raw_variants = value["variants"]
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("crop manifest variants are missing")
    variants = []
    for raw in raw_variants:
        if not isinstance(raw, Mapping):
            raise ValueError("crop manifest variant is malformed")
        variant = CropVariant(
            crop_id=str(raw.get("crop_id", "")),
            role=str(raw.get("role", "")),
            comparison_family=str(raw.get("comparison_family", "")),
            diameter_um=float(raw.get("diameter_um", 0.0)),
            mask_mode=str(raw.get("mask_mode", "")),
            fill_mode=str(raw.get("fill_mode", "")),
            inner_diameter_um=float(raw.get("inner_diameter_um", 0.0)),
            model_input_pixels=int(raw.get("model_input_pixels", 0)),
            effective_mpp=float(raw.get("effective_mpp", 0.0)),
        )
        if (
            not variant.crop_id
            or not variant.role
            or not variant.comparison_family
            or variant.diameter_um <= 0
            or variant.mask_mode
            not in {
                "none",
                "keep_nucleus",
                "keep_cell",
                "remove_context_circle",
                "remove_cell",
                "random_keep_nucleus",
                "random_keep_cell",
                "random_remove_cell",
                "blank",
            }
            or variant.fill_mode not in {"none", "white", "mean_color", "blurred"}
            or variant.inner_diameter_um < 0
            or variant.inner_diameter_um >= variant.diameter_um
            or variant.model_input_pixels <= 0
            or not math.isclose(
                variant.effective_mpp,
                variant.diameter_um / variant.model_input_pixels,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError("crop manifest variant geometry is invalid")
        if (
            (variant.mask_mode == "none" and variant.fill_mode != "none")
            or (variant.mask_mode != "none" and variant.fill_mode == "none")
            or (variant.mask_mode == "blank" and variant.fill_mode != "white")
            or (variant.mask_mode == "remove_context_circle" and variant.inner_diameter_um <= 0)
            or (variant.mask_mode != "remove_context_circle" and variant.inner_diameter_um != 0)
        ):
            raise ValueError("crop manifest mask/fill pairing is invalid")
        variants.append(variant)
    expected = {
        "crop_112um",
        "nucleus_mask_only",
        "nucleus_mask_mean_fill_112um",
        "nucleus_mask_blurred_112um",
        "nucleus_shape_random_location_mean_fill_112um",
        "cell_mask_only",
        "cell_mask_mean_fill_112um",
        "cell_mask_blurred_112um",
        "cell_shape_random_location_mean_fill_112um",
        "context_ring_32_to_112um",
        "context_ring_64_to_112um",
        "target_cell_removed_112um",
        "target_cell_removed_mean_fill_112um",
        "target_cell_removed_blurred_112um",
        "random_location_cell_removed_mean_fill_112um",
        "crop_32um",
        "crop_64um",
        "blank_patch",
    }
    crop_ids = tuple(variant.crop_id for variant in variants)
    if set(crop_ids) != expected or len(crop_ids) != len(expected):
        raise ValueError("crop manifest must contain the frozen physical crop/mask ladder")
    if str(value["primary_crop_id"]) != "crop_112um":
        raise ValueError("unmasked crop_112um must remain the context-association primary")
    primary = variants[crop_ids.index("crop_112um")]
    if (
        primary.role != "registered_cell_local_context_112um"
        or primary.comparison_family != "g2_primary"
        or primary.diameter_um != 112.0
        or primary.effective_mpp != 0.5
        or primary.mask_mode != "none"
        or primary.fill_mode != "none"
    ):
        raise ValueError("G2 primary must be the unmasked registered-cell 112-um context arm")
    common_canvas = {
        variant.crop_id
        for variant in variants
        if variant.comparison_family
        in {"intrinsic_common_canvas", "mask_artifact_control", "context_control"}
    }
    if not common_canvas or any(
        variant.diameter_um != 112.0
        or variant.effective_mpp != 0.5
        or variant.model_input_pixels != 224
        for variant in variants
        if variant.crop_id in common_canvas
    ):
        raise ValueError("intrinsic/context mask comparisons must share a 112-um 0.5-MPP canvas")
    resolution_ids = {
        variant.crop_id
        for variant in variants
        if variant.comparison_family == "resolution_sensitivity"
    }
    if resolution_ids != {"crop_32um", "crop_64um"}:
        raise ValueError("32/64-um crops must be resolution sensitivities only")
    if float(value["source_mpp"]) != SOURCE_MPP or value["padding"] != "white":
        raise ValueError("crop manifest source MPP or padding differs from HEST")
    random_mask_salt = str(value["random_mask_salt"])
    blur_sigma_um = float(value["blur_sigma_um"])
    if not random_mask_salt.strip():
        raise ValueError("crop manifest random-mask salt is empty")
    if not math.isfinite(blur_sigma_um) or blur_sigma_um <= 0:
        raise ValueError("crop manifest blur sigma must be finite and positive")
    return CropManifest(
        path=resolved,
        sha256=_sha256_file(resolved),
        source_mpp=float(value["source_mpp"]),
        padding=str(value["padding"]),
        primary_crop_id=str(value["primary_crop_id"]),
        random_mask_salt=random_mask_salt,
        blur_sigma_um=blur_sigma_um,
        variants=tuple(variants),
    )


@dataclass(frozen=True)
class InputFile:
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class ScopedAnnotationExport:
    file: InputFile
    row_count: int
    sample_ids: Tuple[str, ...]
    source_annotation_sha256: str


@dataclass(frozen=True)
class Sample:
    sample_id: str
    donor_id: str
    split_id: str
    pixel_size_um: float
    wsi: InputFile
    transcripts: InputFile
    cell_seg: InputFile
    nucleus_seg: InputFile
    cellvit_seg: Optional[InputFile]


def _frozen_ita_stratum_ids(
    samples: Sequence[Sample], supported_fine_types: Sequence[str]
) -> tuple[str, ...]:
    """Freeze donor/section/type coverage without consulting observed RNA labels."""

    roster = tuple((str(sample.donor_id), str(sample.sample_id)) for sample in samples)
    fine_types = tuple(str(value) for value in supported_fine_types)
    if (
        not roster
        or not fine_types
        or len(set(roster)) != len(roster)
        or len(set(fine_types)) != len(fine_types)
        or any(not value.strip() or "|" in value for pair in roster for value in pair)
        or any(not value.strip() or "|" in value for value in fine_types)
    ):
        raise ValueError("frozen ITA donor/section/type population is malformed")
    return tuple(
        sorted(
            "%s|%s|%s" % (donor_id, sample_id, fine_type)
            for donor_id, sample_id in roster
            for fine_type in fine_types
        )
    )


def _parse_file(value: object, label: str) -> InputFile:
    if not isinstance(value, Mapping):
        raise ValueError("HEST %s file declaration is missing" % label)
    relative = str(value.get("path", ""))
    digest = value.get("sha256")
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts or not _is_sha256(digest):
        raise ValueError("HEST %s file identity is unsafe or unpinned" % label)
    return InputFile(relative, str(digest))


def _parse_development_annotation_export(
    value: object,
    expected_sample_ids: Sequence[str],
) -> ScopedAnnotationExport:
    fields = {
        "path",
        "sha256",
        "row_count",
        "sample_ids",
        "sample_ids_sha256",
        "source_annotation_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(
            "HEST measurement development requires a frozen development-only annotation export"
        )
    declaration = _parse_file(value, "development-only annotation export")
    sample_ids_value = value["sample_ids"]
    if not isinstance(sample_ids_value, list):
        raise ValueError("HEST development-only annotation sample IDs must be a list")
    sample_ids = tuple(str(sample_id) for sample_id in sample_ids_value)
    expected = tuple(str(sample_id) for sample_id in expected_sample_ids)
    row_count = value["row_count"]
    if (
        sample_ids != expected
        or value["sample_ids_sha256"] != _canonical_sha256(list(sample_ids))
        or value["source_annotation_sha256"] != ANNOTATION_SHA256
        or not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count < 1
    ):
        raise ValueError("HEST development-only annotation export receipt is malformed")
    return ScopedAnnotationExport(
        file=declaration,
        row_count=row_count,
        sample_ids=sample_ids,
        source_annotation_sha256=str(value["source_annotation_sha256"]),
    )


def _parse_samples(protocol: Mapping[str, object]) -> Tuple[Sample, ...]:
    raw = protocol.get("samples")
    if not isinstance(raw, list) or not raw:
        raise ValueError("HEST protocol needs frozen sample declarations")
    development = {str(value) for value in protocol.get("development_donors", ())}
    locked = {str(value) for value in protocol.get("locked_test_donors", ())}
    result = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise ValueError("HEST sample declarations must be JSON objects")
        sample_id = str(value.get("sample_id", ""))
        donor_id = str(value.get("donor_id", ""))
        if not re.fullmatch(r"NCBI\d+", sample_id) or not donor_id.strip():
            raise ValueError("HEST sample or donor identity is malformed")
        if donor_id in development:
            split_id = "development"
        elif donor_id in locked:
            split_id = "locked_test"
        else:
            raise ValueError("HEST sample donor is outside the frozen partitions")
        pixel_size = float(value.get("pixel_size_um", 0.0))
        if not math.isfinite(pixel_size) or pixel_size != SOURCE_MPP:
            raise ValueError("HEST sample pixel size differs from the frozen source MPP")
        cellvit = value.get("cellvit_seg")
        result.append(
            Sample(
                sample_id=sample_id,
                donor_id=donor_id,
                split_id=split_id,
                pixel_size_um=pixel_size,
                wsi=_parse_file(value.get("wsi"), sample_id + " WSI"),
                transcripts=_parse_file(value.get("transcripts"), sample_id + " transcripts"),
                cell_seg=_parse_file(value.get("cell_seg"), sample_id + " cell segmentation"),
                nucleus_seg=_parse_file(
                    value.get("nucleus_seg"), sample_id + " nucleus segmentation"
                ),
                cellvit_seg=(
                    _parse_file(cellvit, sample_id + " CellViT segmentation")
                    if cellvit is not None
                    else None
                ),
            )
        )
    if len({sample.sample_id for sample in result}) != len(result):
        raise ValueError("HEST sample IDs must be unique")
    if tuple(sample.sample_id for sample in result) != tuple(
        sorted(sample.sample_id for sample in result)
    ):
        raise ValueError("HEST samples must be ordered by section for identity hashing")
    observed_sections = {sample.sample_id for sample in result}
    if observed_sections != set(SECTION_IDENTITIES):
        raise ValueError("HEST protocol must contain the frozen 20-section lung cohort")
    for sample in result:
        if SECTION_IDENTITIES[sample.sample_id][0] != sample.donor_id:
            raise ValueError("HEST section differs from the corrected true-donor identity")
    observed_donors = {sample.donor_id for sample in result}
    if observed_donors != development | locked:
        raise ValueError("every frozen HEST donor must have a declared sample")
    return tuple(result)


def _validate_protocol(
    protocol: Mapping[str, object],
    encoder_manifest: EncoderManifest,
    crop_manifest: CropManifest,
) -> Tuple[Sample, ...]:
    exact = {
        "schema": PROTOCOL_SCHEMA,
        "scientific_scope": "registered_cell_local_context_112um_association",
        "g2_claim_scope": "registered_cell_local_context_112um",
        "authorizes_nucleus_intrinsic_claim": False,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "normalization": "log1p_cpm_10000",
        "assay": "Xenium",
        "observation_level": "cell",
        "target_construction": "nucleus_overlapping_xenium_transcripts",
        "registration_method": "native_xenium_cell_id_join",
        "minimum_transcript_qv": 20.0,
    }
    for name, expected in exact.items():
        if protocol.get(name) != expected:
            raise ValueError("HEST protocol %s differs from the pinned design" % name)
    if tuple(protocol.get("disease_estimands", ())) != (
        "disease_inclusive",
        "disease_adjusted",
    ):
        raise ValueError("HEST disease estimands are not frozen")
    if tuple(protocol.get("nuisance_fields", ())) != (
        "log1p_library_size",
        "section_id",
        "disease_status",
        "site_id",
        "batch_id",
        "stain_quality",
        "nuclear_morphology",
        "cell_morphology",
        "local_density",
        "boundary_position",
        "smooth_spatial_basis",
    ):
        raise ValueError("HEST nuisance feature registry is not frozen")
    if protocol.get("encoder_manifest_sha256") != encoder_manifest.sha256:
        raise ValueError("HEST protocol differs from the supplied encoder manifest")
    if protocol.get("crop_manifest_sha256") != crop_manifest.sha256:
        raise ValueError("HEST protocol differs from the supplied crop manifest")
    development = tuple(str(value) for value in protocol.get("development_donors", ()))
    locked = tuple(str(value) for value in protocol.get("locked_test_donors", ()))
    if development != DEVELOPMENT_DONORS or locked != LOCKED_TEST_DONORS:
        raise ValueError("HEST protocol differs from the frozen 10/5 true-donor split")
    type_names = tuple(str(value) for value in protocol.get("broad_type_names", ()))
    markers = protocol.get("type_markers")
    if type_names != TYPE_NAMES or not isinstance(markers, Mapping):
        raise ValueError("HEST RNA-only broad-type ontology is not frozen")
    if set(markers) != set(type_names):
        raise ValueError("HEST type marker groups differ from type_names")
    flattened = []
    for name in type_names:
        genes = tuple(str(gene) for gene in markers[name])
        if not genes:
            raise ValueError("every HEST type needs RNA-only markers")
        flattened.extend(genes)
    fine_markers = tuple(str(value) for value in protocol.get("fine_type_marker_gene_ids", ()))
    _validate_protocol_independence_contract(protocol.get("label_target_independence"))
    gene_ids = tuple(str(value) for value in protocol.get("gene_ids", ()))
    if (
        len(flattened) != len(set(flattened))
        or not fine_markers
        or len(fine_markers) != len(set(fine_markers))
        or not gene_ids
        or len(gene_ids) != len(set(gene_ids))
        or set(flattened) & set(gene_ids)
        or set(flattened) & set(fine_markers)
        or set(fine_markers) & set(gene_ids)
        or any(
            gene.startswith(prefix)
            for gene in flattened + list(fine_markers) + list(gene_ids)
            for prefix in CONTROL_PREFIXES
        )
    ):
        raise ValueError("HEST broad/fine marker and evaluation genes must be unique and disjoint")
    for name in (
        "minimum_transcripts_per_cell",
        "minimum_transcript_qv",
        "minimum_reference_cells_per_donor_section_type",
        "minimum_evaluation_cells_per_donor_section_type",
        "spatial_block_um",
        "spatial_roi_um",
        "opposite_pool_guard_um",
        "cellvit_sensitivity_radius_um",
        "maximum_affine_registration_residual_p95_um",
        "maximum_annotation_nucleus_distance_p95_um",
        "maximum_registration_outlier_fraction",
        "maximum_crop_padding_fraction",
    ):
        value = float(protocol.get(name, -1.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError("HEST protocol %s must be finite and nonnegative" % name)
    block = float(protocol["spatial_block_um"])
    roi = float(protocol["spatial_roi_um"])
    guard = float(protocol["opposite_pool_guard_um"])
    if block <= 2 * guard or roi <= 0 or block < roi:
        raise ValueError("HEST spatial block/ROI/guard design is invalid")
    if (
        int(protocol["minimum_reference_cells_per_donor_section_type"]) < 1
        or int(protocol["minimum_evaluation_cells_per_donor_section_type"]) < 1
    ):
        raise ValueError("HEST donor/section/type source support minima must be positive")
    for name in (
        "maximum_registration_outlier_fraction",
        "maximum_crop_padding_fraction",
    ):
        if not 0 <= float(protocol[name]) < 1:
            raise ValueError("HEST %s must be in [0, 1)" % name)
    for name in (
        "minimum_development_donors_per_fine_type",
        "minimum_locked_donors_per_fine_type",
    ):
        if int(protocol.get(name, 0)) < 1:
            raise ValueError("HEST fine-type donor coverage minima must be positive")
    prefixes = tuple(str(value) for value in protocol.get("excluded_feature_prefixes", ()))
    if prefixes != CONTROL_PREFIXES:
        raise ValueError("HEST non-control transcript filter differs from the frozen release")
    salt = protocol.get("pool_assignment_salt")
    if not isinstance(salt, str) or not salt.strip():
        raise ValueError("HEST spatial pool assignment salt is not frozen")
    reference_splits = protocol.get("reference_splits")
    if not isinstance(reference_splits, Mapping):
        raise ValueError("HEST alternate reference splits are not frozen")
    alternates = reference_splits.get("alternate_splits")
    if (
        reference_splits.get("primary_split_id") != "primary"
        or reference_splits.get("selection_unit") != "spatial_block"
        or reference_splits.get("primary_evaluation_rows_fixed") is not True
        or not isinstance(alternates, list)
        or [value.get("split_id") for value in alternates if isinstance(value, Mapping)]
        != ["reference_hash_fold_0", "reference_hash_fold_1"]
    ):
        raise ValueError("HEST alternate reference-split contract is malformed")
    alternate_salts = []
    for declaration in alternates:
        if not isinstance(declaration, Mapping):
            raise ValueError("HEST alternate reference split is malformed")
        split_salt = str(declaration.get("salt", ""))
        retention = float(declaration.get("initial_reference_retention_fraction", -1.0))
        if not split_salt.strip() or not 0.5 <= retention < 1.0:
            raise ValueError("HEST alternate reference split settings are invalid")
        alternate_salts.append(split_salt)
    if len(set(alternate_salts)) != len(alternate_salts):
        raise ValueError("HEST alternate reference split salts must be unique")
    transcript_split_salt = protocol.get("transcript_split_salt")
    if not isinstance(transcript_split_salt, str) or not transcript_split_salt.strip():
        raise ValueError("HEST transcript split-half salt is not frozen")
    programs = protocol.get("target_programs")
    if not isinstance(programs, Mapping) or not programs:
        raise ValueError("HEST target programs are not frozen")
    for name, program_genes in programs.items():
        members = tuple(str(gene) for gene in program_genes)
        if not str(name).strip() or not members or not set(members) <= set(gene_ids):
            raise ValueError("HEST target program is empty or outside ordered target genes")
    samples = _parse_samples(protocol)
    annotation = _parse_file(protocol.get("annotation_export"), "GSE250346 annotation export")
    if annotation.sha256 != ANNOTATION_SHA256:
        raise ValueError("HEST GSE250346 annotation export differs from the pinned SHA-256")
    development_annotation = protocol.get("measurement_development_annotation_export")
    if development_annotation is not None:
        _parse_development_annotation_export(
            development_annotation,
            tuple(sample.sample_id for sample in samples if sample.split_id == "development"),
        )
    cellvit_classes = tuple(str(value) for value in protocol.get("cellvit_class_names", ()))
    uses_cellvit = any(sample.cellvit_seg is not None for sample in samples)
    if uses_cellvit and (
        not cellvit_classes
        or len(cellvit_classes) != len(set(cellvit_classes))
        or any(not value.strip() for value in cellvit_classes)
    ):
        raise ValueError("HEST CellViT sensitivity classes are not frozen")
    if uses_cellvit and not all(sample.cellvit_seg is not None for sample in samples):
        raise ValueError("CellViT sensitivity must be present for all samples or none")
    for sample in samples:
        if (
            sample.cell_seg.relative_path == sample.nucleus_seg.relative_path
            or sample.cell_seg.sha256 == sample.nucleus_seg.sha256
        ):
            raise ValueError("native Xenium cell and nucleus sources must be distinct")
    return samples


def _resolve_input(root: Path, declaration: InputFile) -> Path:
    path = (root / declaration.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("HEST input escapes the frozen data root") from error
    if not path.is_file() or _sha256_file(path) != declaration.sha256:
        raise ValueError("HEST input is missing or differs from its frozen SHA-256: %s" % path)
    return path


@dataclass(frozen=True)
class NativeGeometry:
    centroid: Tuple[float, float]
    vertices: np.ndarray
    area_pixels2: float


def _polygon_geometry(wkb: bytes) -> NativeGeometry:
    """Read the exterior ring of a finite little/big-endian WKB Polygon."""

    if not isinstance(wkb, (bytes, bytearray, memoryview)) or len(wkb) < 13:
        raise ValueError("native Xenium segmentation contains invalid WKB")
    data = memoryview(wkb)
    endian = data[0]
    if endian not in (0, 1):
        raise ValueError("native Xenium WKB has an invalid byte order")
    order = "<" if endian == 1 else ">"
    geometry_type = struct.unpack_from(order + "I", data, 1)[0]
    if geometry_type != 3:
        raise ValueError("native Xenium segmentation must contain WKB Polygons")
    rings = struct.unpack_from(order + "I", data, 5)[0]
    if rings < 1:
        raise ValueError("native Xenium Polygon has no exterior ring")
    points = struct.unpack_from(order + "I", data, 9)[0]
    offset = 13
    if points < 4 or offset + points * 16 > len(data):
        raise ValueError("native Xenium exterior ring is malformed")
    coordinates = np.asarray(
        [struct.unpack_from(order + "dd", data, offset + index * 16) for index in range(points)],
        dtype=np.float64,
    )
    if not np.isfinite(coordinates).all():
        raise ValueError("native Xenium Polygon has nonfinite coordinates")
    left = coordinates[:-1]
    right = coordinates[1:]
    cross = left[:, 0] * right[:, 1] - right[:, 0] * left[:, 1]
    area_twice = cross.sum()
    if abs(area_twice) < 1.0e-12:
        centre = coordinates[:-1].mean(axis=0)
    else:
        centre = np.asarray(
            [
                ((left[:, 0] + right[:, 0]) * cross).sum(),
                ((left[:, 1] + right[:, 1]) * cross).sum(),
            ]
        ) / (3.0 * area_twice)
    if not np.isfinite(centre).all():
        raise ValueError("native Xenium Polygon centroid is invalid")
    area = abs(area_twice) / 2.0
    if area <= 0:
        raise ValueError("native Xenium Polygon has zero area")
    return NativeGeometry(
        centroid=(float(centre[0]), float(centre[1])),
        vertices=coordinates[:-1].copy(),
        area_pixels2=float(area),
    )


def _polygon_centroid(wkb: bytes) -> Tuple[float, float]:
    return _polygon_geometry(wkb).centroid


def _point_inside_polygon(point: Tuple[float, float], vertices: np.ndarray) -> bool:
    x, y = point
    inside = False
    previous = vertices[-1]
    for current in vertices:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside


def _read_native_segmentation(path: Path) -> Dict[str, NativeGeometry]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install HEIR with the hest optional dependencies") from error
    schema = parquet.read_schema(path)
    required = {"__index_level_0__", "geometry"}
    if not required <= set(schema.names):
        raise ValueError("native Xenium segmentation schema is incomplete")
    table = parquet.read_table(path, columns=["__index_level_0__", "geometry"])
    identities = table["__index_level_0__"].to_pylist()
    geometries = table["geometry"].to_pylist()
    result: Dict[str, NativeGeometry] = {}
    for identity, geometry in zip(identities, geometries):
        cell_id = str(identity)
        if not cell_id or cell_id == "UNASSIGNED" or cell_id in result:
            raise ValueError("native Xenium segmentation IDs are empty or duplicated")
        result[cell_id] = _polygon_geometry(geometry)
    if not result:
        raise ValueError("native Xenium segmentation is empty")
    return result


@dataclass(frozen=True)
class AnnotationCell:
    broad_label: int
    fine_type: str
    disease_status: str
    site_id: str
    batch_id: str
    source_centroid: Tuple[float, float]
    ncount_rna: float
    nfeature_rna: float
    percent_negative_or_unassigned: float


@dataclass(frozen=True)
class IndependentAnnotationArtifacts:
    """Verified, row-ordered labels produced by the frozen independent annotator."""

    receipt_path: Path
    receipt_sha256: str
    predictions_path: Path
    predictions_sha256: str
    prediction_row_count: int


_ANNOTATION_RECEIPT_FIELDS = {
    "schema",
    "evidence_kind",
    "prediction_export_sha256",
    "prediction_row_count",
    "prediction_columns",
    "row_order",
    "ordered_annotation_feature_ids",
    "ordered_annotation_feature_ids_sha256",
    "annotation_training_scope",
    "annotation_training_donor_ids",
    "annotation_training_donor_ids_sha256",
    "locked_donors_used_for_training",
    "same_cohort_annotation",
    "cross_fitting_method",
    "cross_fitting_receipt",
}


def _read_independent_annotation_receipt(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("independent annotation receipt is not valid JSON") from error
    if not isinstance(value, Mapping) or set(value) != _ANNOTATION_RECEIPT_FIELDS:
        raise ValueError("independent annotation receipt has an incomplete or extra field set")
    return value


def _verify_cross_fitting_receipt(
    receipt: object,
    contract: Mapping[str, object],
    prediction_donor_ids: Sequence[str],
) -> None:
    if contract["same_cohort_annotation"] is not True:
        if receipt is not None or contract["cross_fitting_receipt_sha256"] is not None:
            raise ValueError("non-cross-fitted independent annotation cannot carry a fold receipt")
        return
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != {"schema", "folds"}
        or receipt.get("schema") != ANNOTATION_CROSS_FIT_SCHEMA
        or not isinstance(receipt.get("folds"), list)
        or _canonical_sha256(receipt) != contract["cross_fitting_receipt_sha256"]
    ):
        raise ValueError("independent annotation cross-fitting receipt differs from the lock")
    training_donors = tuple(str(value) for value in contract["annotation_training_donor_ids"])
    expected_prediction_donors = tuple(str(value) for value in prediction_donor_ids)
    folds: dict[str, tuple[str, ...]] = {}
    for raw_fold in receipt["folds"]:
        if not isinstance(raw_fold, Mapping) or set(raw_fold) != {
            "prediction_donor_id",
            "training_donor_ids",
        }:
            raise ValueError("independent annotation cross-fitting fold is malformed")
        prediction_donor = str(raw_fold["prediction_donor_id"])
        raw_training = raw_fold["training_donor_ids"]
        if not isinstance(raw_training, list):
            raise ValueError("independent annotation cross-fitting donor IDs must be a list")
        fold_training = tuple(str(value) for value in raw_training)
        if (
            not prediction_donor
            or prediction_donor in folds
            or not fold_training
            or len(set(fold_training)) != len(fold_training)
        ):
            raise ValueError("independent annotation cross-fitting fold is malformed")
        folds[prediction_donor] = fold_training
    if set(folds) != set(expected_prediction_donors):
        raise ValueError(
            "independent annotation cross-fitting folds do not cover prediction donors"
        )
    for prediction_donor in expected_prediction_donors:
        expected_training = tuple(donor for donor in training_donors if donor != prediction_donor)
        if folds[prediction_donor] != expected_training:
            raise ValueError(
                "independent annotation fold is not donor-held-out from frozen development training"
            )


def _verify_independent_annotation_artifacts(
    receipt_path: Path,
    predictions_path: Path,
    contract: Mapping[str, object],
    *,
    prediction_donor_ids: Sequence[str],
) -> IndependentAnnotationArtifacts:
    """Bind the actual label export to the frozen non-pending evidence contract."""

    receipt_path = receipt_path.expanduser().resolve()
    predictions_path = predictions_path.expanduser().resolve()
    if (
        receipt_path == predictions_path
        or not receipt_path.is_file()
        or not predictions_path.is_file()
    ):
        raise ValueError(
            "independent annotation receipt and prediction export must be distinct files"
        )
    if contract.get("evidence_kind") == "pending":
        raise ValueError("pending label-target evidence cannot materialize independent labels")
    receipt_sha256 = _sha256_file(receipt_path)
    if receipt_sha256 != contract.get("annotation_receipt_sha256"):
        raise ValueError("independent annotation receipt differs from the frozen SHA-256")
    receipt = _read_independent_annotation_receipt(receipt_path)
    contract_fields = {
        "evidence_kind",
        "ordered_annotation_feature_ids",
        "ordered_annotation_feature_ids_sha256",
        "annotation_training_scope",
        "annotation_training_donor_ids",
        "annotation_training_donor_ids_sha256",
        "locked_donors_used_for_training",
        "same_cohort_annotation",
        "cross_fitting_method",
    }
    if receipt["schema"] != ANNOTATION_RECEIPT_SCHEMA or any(
        receipt[name] != contract.get(name) for name in contract_fields
    ):
        raise ValueError("independent annotation receipt differs from the frozen evidence contract")
    if (
        tuple(str(value) for value in receipt["prediction_columns"])
        != (ANNOTATION_PREDICTION_COLUMNS)
        or receipt["row_order"] != "filtered_annotation_export_order"
    ):
        raise ValueError("independent annotation prediction layout is not frozen")
    prediction_count = receipt["prediction_row_count"]
    if (
        not isinstance(prediction_count, int)
        or isinstance(prediction_count, bool)
        or prediction_count < 1
    ):
        raise ValueError("independent annotation prediction row count must be positive")
    predictions_sha256 = _sha256_file(predictions_path)
    if predictions_sha256 != receipt["prediction_export_sha256"]:
        raise ValueError("independent annotation predictions differ from their receipt")
    _verify_cross_fitting_receipt(
        receipt["cross_fitting_receipt"],
        contract,
        prediction_donor_ids,
    )
    return IndependentAnnotationArtifacts(
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        predictions_path=predictions_path,
        predictions_sha256=predictions_sha256,
        prediction_row_count=prediction_count,
    )


def _open_annotation_predictions(path: Path):
    with path.open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    if compressed:
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _read_annotations(
    path: Path,
    *,
    allowed_sample_ids: Optional[Sequence[str]] = None,
    independent_artifacts: Optional[IndependentAnnotationArtifacts] = None,
    expected_row_count: int = ANNOTATION_ROWS,
    strict_sample_scope: bool = False,
) -> Dict[str, Dict[str, AnnotationCell]]:
    required = {
        "hest_id",
        "sample",
        "patient",
        "cell_id",
        "sample_type",
        "sample_affect",
        "disease_status",
        "tma",
        "run",
        "x_centroid",
        "y_centroid",
        "nCount_RNA",
        "nFeature_RNA",
        "perc_negcontrolorunassigned",
    }
    if independent_artifacts is None:
        required.update(("final_CT", "final_lineage"))
    allowed = (
        set(SECTION_IDENTITIES)
        if allowed_sample_ids is None
        else {str(value) for value in allowed_sample_ids}
    )
    if not allowed or not allowed <= set(SECTION_IDENTITIES):
        raise ValueError("HEST annotation sample scope is empty or unknown")
    result: Dict[str, Dict[str, AnnotationCell]] = {sample_id: {} for sample_id in allowed}
    rows = 0
    prediction_context = (
        nullcontext(None)
        if independent_artifacts is None
        else _open_annotation_predictions(independent_artifacts.predictions_path)
    )
    prediction_rows = 0
    try:
        with (
            gzip.open(path, "rt", encoding="utf-8", newline="") as handle,
            prediction_context as prediction_handle,
        ):
            reader = csv.DictReader(handle, delimiter="\t")
            if not required <= set(reader.fieldnames or ()):
                raise ValueError("HEST GSE250346 annotation schema is incomplete")
            prediction_reader = None
            if prediction_handle is not None:
                prediction_reader = csv.DictReader(prediction_handle, delimiter="\t")
                if tuple(prediction_reader.fieldnames or ()) != ANNOTATION_PREDICTION_COLUMNS:
                    raise ValueError(
                        "independent annotation prediction columns differ from receipt"
                    )
            for row in reader:
                sample_id = row["hest_id"]
                if sample_id not in SECTION_IDENTITIES:
                    raise ValueError("HEST annotation contains an unknown section")
                if strict_sample_scope and sample_id not in allowed:
                    raise ValueError("development-only annotation export contains a locked section")
                expected_donor, expected_source_sample = SECTION_IDENTITIES[sample_id]
                if row["patient"] != expected_donor or row["sample"] != expected_source_sample:
                    raise ValueError("HEST annotation differs from corrected donor identities")
                rows += 1
                if sample_id not in allowed:
                    continue
                if prediction_reader is None:
                    lineage = row["final_lineage"]
                    fine_type = row["final_CT"].strip()
                else:
                    prediction = next(prediction_reader, None)
                    if prediction is None:
                        raise ValueError("independent annotation predictions end before metadata")
                    prediction_rows += 1
                    if (
                        prediction["hest_id"] != sample_id
                        or prediction["cell_id"] != row["cell_id"]
                    ):
                        raise ValueError(
                            "independent annotation predictions differ from frozen metadata order"
                        )
                    lineage = prediction["broad_lineage"]
                    fine_type = prediction["fine_type"].strip()
                disease = row["disease_status"].strip()
                site_id = row["sample_type"].strip()
                tma = row["tma"].strip()
                run = row["run"].strip()
                if (
                    lineage not in TYPE_NAMES
                    or not fine_type
                    or disease not in {"Control", "Disease"}
                    or not site_id
                    or not tma
                    or not run
                ):
                    raise ValueError("HEST annotation has an unknown or empty RNA-derived label")
                cell_id = row["cell_id"]
                if not cell_id or cell_id == "UNASSIGNED" or cell_id in result[sample_id]:
                    raise ValueError("HEST annotation cell IDs are empty or duplicated")
                centroid = (float(row["x_centroid"]), float(row["y_centroid"]))
                ncount = float(row["nCount_RNA"])
                nfeature = float(row["nFeature_RNA"])
                negative = float(row["perc_negcontrolorunassigned"])
                if (
                    not np.isfinite(centroid).all()
                    or ncount <= 0
                    or nfeature <= 0
                    or not 0 <= negative <= 100
                ):
                    raise ValueError("HEST annotation contains a malformed high-QC cell")
                result[sample_id][cell_id] = AnnotationCell(
                    broad_label=TYPE_NAMES.index(lineage),
                    fine_type=fine_type,
                    disease_status=disease,
                    site_id=site_id,
                    batch_id=tma + ":" + run,
                    source_centroid=centroid,
                    ncount_rna=ncount,
                    nfeature_rna=nfeature,
                    percent_negative_or_unassigned=negative,
                )
            if prediction_reader is not None:
                if next(prediction_reader, None) is not None:
                    raise ValueError("independent annotation predictions contain extra rows")
                if prediction_rows != independent_artifacts.prediction_row_count:
                    raise ValueError(
                        "independent annotation prediction count differs from its receipt"
                    )
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError("HEST GSE250346 annotation export cannot be read") from error
    if rows != expected_row_count or any(not values for values in result.values()):
        raise ValueError("HEST annotation must contain the pinned cohort and every allowed section")
    return result


@dataclass(frozen=True)
class CoordinateRegistration:
    coefficients: np.ndarray
    residual_p95_um: float
    residual_max_um: float

    def transform(self, coordinates: Sequence[Tuple[float, float]]) -> np.ndarray:
        values = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
        design = np.column_stack((values, np.ones(len(values))))
        return design @ self.coefficients


def _fit_coordinate_registration(path: Path, source_mpp: float) -> CoordinateRegistration:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install HEIR with the hest optional dependencies") from error
    required = ("x_location", "y_location", "he_x", "he_y")
    schema = parquet.read_schema(path)
    if not set(required) <= set(schema.names):
        raise ValueError("HEST transcripts lack Xenium-to-H&E registration coordinates")
    batches = parquet.ParquetFile(path).iter_batches(batch_size=100_000, columns=list(required))
    try:
        batch = next(batches)
    except StopIteration as error:
        raise ValueError("HEST transcripts cannot fit a coordinate registration") from error
    values = np.column_stack(
        [
            np.asarray(
                batch.column(batch.schema.get_field_index(name)).to_numpy(), dtype=np.float64
            )
            for name in required
        ]
    )
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 100:
        raise ValueError("HEST coordinate registration has insufficient finite anchors")
    design = np.column_stack((values[:, :2], np.ones(len(values))))
    coefficients = np.asarray(
        [[1.0 / source_mpp, 0.0], [0.0, 1.0 / source_mpp], [0.0, 0.0]],
        dtype=np.float64,
    )
    residual_um = np.linalg.norm(design @ coefficients - values[:, 2:], axis=1) * source_mpp
    return CoordinateRegistration(
        coefficients=coefficients,
        residual_p95_um=float(np.quantile(residual_um, 0.95)),
        residual_max_um=float(residual_um.max()),
    )


@dataclass(frozen=True)
class ExpressionTargets:
    nucleus_counts: np.ndarray
    whole_cell_counts: np.ndarray
    nucleus_half_a_counts: np.ndarray
    nucleus_half_b_counts: np.ndarray
    whole_cell_half_a_counts: np.ndarray
    whole_cell_half_b_counts: np.ndarray
    nucleus_library_size_half_a: np.ndarray
    nucleus_library_size_half_b: np.ndarray
    whole_cell_library_size_half_a: np.ndarray
    whole_cell_library_size_half_b: np.ndarray
    nucleus_library_sizes: np.ndarray
    whole_cell_library_sizes: np.ndarray
    nucleus_eligible_transcripts: int
    whole_cell_eligible_transcripts: int
    whole_cell_qv_summary: np.ndarray


def _aggregate_expression(
    path: Path,
    cell_ids: Sequence[str],
    segmented_cell_ids: Sequence[str],
    genes: Sequence[str],
    known_genes: Sequence[str],
    *,
    minimum_qv: float,
    excluded_prefixes: Sequence[str],
    split_salt: str,
) -> ExpressionTargets:
    try:
        import duckdb
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install HEIR with the hest optional dependencies") from error
    schema = parquet.read_schema(path)
    if not {"cell_id", "feature_name", "qv", "transcript_id", "overlaps_nucleus"} <= set(
        schema.names
    ):
        raise ValueError("HEST transcript Parquet schema is incomplete")
    ordered_ids = tuple(str(value) for value in cell_ids)
    segmented_ids = tuple(str(value) for value in segmented_cell_ids)
    row_for_id = {value: index for index, value in enumerate(ordered_ids)}
    gene_for_name = {str(value): index for index, value in enumerate(genes)}
    known = tuple(str(value) for value in known_genes)
    if (
        len(row_for_id) != len(ordered_ids)
        or len(set(segmented_ids)) != len(segmented_ids)
        or len(gene_for_name) != len(genes)
        or len(set(known)) != len(known)
        or not set(genes) <= set(known)
    ):
        raise ValueError("HEST registered cell or gene identities are duplicated")
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("registered_cells", pa.table({"cell_id": ordered_ids}))
        connection.register("segmented_cells", pa.table({"cell_id": segmented_ids}))
        exclusion = "".join(" AND NOT starts_with(t.feature_name, ?)" for _ in excluded_prefixes)
        invalid_qv = connection.execute(
            """
            SELECT COUNT(*)::BIGINT
            FROM read_parquet(?)
            WHERE qv IS NULL OR NOT isfinite(qv) OR qv < 0
            """,
            [str(path)],
        ).fetchone()[0]
        if int(invalid_qv):
            raise ValueError("HEST transcripts contain invalid QV values")
        unknown_query = (
            """
            SELECT COUNT(*)::BIGINT
            FROM read_parquet(?) AS t
            WHERE t.qv >= ? AND (
                t.feature_name IS NULL OR (
                    NOT (t.feature_name = ANY(?))
        """
            + exclusion
            + "))"
        )
        unknown_parameters: list[object] = [str(path), float(minimum_qv), list(known)]
        unknown_parameters.extend(excluded_prefixes)
        unknown_features = connection.execute(unknown_query, unknown_parameters).fetchone()[0]
        if int(unknown_features):
            raise ValueError("HEST transcripts contain unknown non-control feature names")
        unsegmented = connection.execute(
            """
            SELECT COUNT(*)::BIGINT
            FROM read_parquet(?) AS t
            LEFT JOIN segmented_cells AS s USING (cell_id)
            WHERE t.qv >= ? AND t.cell_id IS NOT NULL AND t.cell_id != 'UNASSIGNED'
                  AND s.cell_id IS NULL
            """,
            [str(path), float(minimum_qv)],
        ).fetchone()[0]
        if int(unsegmented):
            raise ValueError("HEST high-QV transcript is assigned to an unsegmented cell")
        duplicate_query = (
            """
            SELECT COUNT(*)::BIGINT
            FROM (
                SELECT t.transcript_id
                FROM read_parquet(?) AS t
                WHERE t.qv >= ? AND t.cell_id IS NOT NULL AND t.cell_id != 'UNASSIGNED'
        """
            + exclusion
            + """
                GROUP BY t.transcript_id
                HAVING t.transcript_id IS NULL OR COUNT(*) != 1 OR COUNT(DISTINCT t.cell_id) != 1
            ) AS duplicated
        """
        )
        duplicate_parameters: list[object] = [str(path), float(minimum_qv)]
        duplicate_parameters.extend(excluded_prefixes)
        duplicated = connection.execute(duplicate_query, duplicate_parameters).fetchone()[0]
        if int(duplicated):
            raise ValueError("HEST eligible transcripts contain duplicate transcript_id values")

        def aggregate(
            overlap_clause: str,
        ) -> Tuple[
            np.ndarray,
            np.ndarray,
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]:
            totals_query = (
                """
                SELECT t.cell_id, COUNT(DISTINCT t.transcript_id)::BIGINT AS library_size
                FROM read_parquet(?) AS t
                INNER JOIN registered_cells AS r USING (cell_id)
                WHERE t.qv >= ?
            """
                + overlap_clause
                + exclusion
                + " GROUP BY t.cell_id"
            )
            base_parameters: list[object] = [str(path), float(minimum_qv)]
            base_parameters.extend(excluded_prefixes)
            totals = np.zeros(len(ordered_ids), dtype=np.int64)
            total_result = connection.execute(totals_query, base_parameters)
            total_batches = (
                total_result.to_arrow_reader(65_536)
                if hasattr(total_result, "to_arrow_reader")
                else total_result.fetch_record_batch(65_536)
            )
            for batch in total_batches:
                for cell_id, count in zip(
                    batch.column("cell_id").to_pylist(),
                    batch.column("library_size").to_pylist(),
                ):
                    totals[row_for_id[str(cell_id)]] = int(count)
            library_half_a = np.zeros(len(ordered_ids), dtype=np.int64)
            library_half_b = np.zeros(len(ordered_ids), dtype=np.int64)
            library_half_query = (
                """
                SELECT t.cell_id,
                       CASE WHEN RIGHT(SHA256(? || CAST(t.transcript_id AS VARCHAR)), 1)
                                      IN ('1','3','5','7','9','b','d','f')
                            THEN 1 ELSE 0 END AS split_half,
                       COUNT(DISTINCT t.transcript_id)::BIGINT AS library_size
                FROM read_parquet(?) AS t
                INNER JOIN registered_cells AS r USING (cell_id)
                WHERE t.qv >= ?
            """
                + overlap_clause
                + exclusion
                + " GROUP BY t.cell_id, split_half"
            )
            library_half_parameters: list[object] = [
                split_salt + "\0",
                str(path),
                float(minimum_qv),
            ]
            library_half_parameters.extend(excluded_prefixes)
            for cell_id, split_half, count in connection.execute(
                library_half_query, library_half_parameters
            ).fetchall():
                destination = library_half_a if int(split_half) == 0 else library_half_b
                destination[row_for_id[str(cell_id)]] = int(count)
            if not np.array_equal(totals, library_half_a + library_half_b):
                raise ValueError(
                    "HEST transcript split-half library sizes do not reconstruct totals"
                )
            selected_query = (
                """
                SELECT t.cell_id, t.feature_name,
                       CASE WHEN RIGHT(SHA256(? || CAST(t.transcript_id AS VARCHAR)), 1)
                                      IN ('1','3','5','7','9','b','d','f')
                            THEN 1 ELSE 0 END AS split_half,
                       COUNT(DISTINCT t.transcript_id)::BIGINT AS gene_count
                FROM read_parquet(?) AS t
                INNER JOIN registered_cells AS r USING (cell_id)
                WHERE t.qv >= ?
            """
                + overlap_clause
                + """
                AND t.feature_name = ANY(?)
                GROUP BY t.cell_id, t.feature_name, split_half
            """
            )
            counts = np.zeros((len(ordered_ids), len(genes)), dtype=np.float32)
            half_a = np.zeros_like(counts)
            half_b = np.zeros_like(counts)
            parameters = [split_salt + "\0", str(path), float(minimum_qv), list(genes)]
            selected_result = connection.execute(selected_query, parameters)
            selected_batches = (
                selected_result.to_arrow_reader(65_536)
                if hasattr(selected_result, "to_arrow_reader")
                else selected_result.fetch_record_batch(65_536)
            )
            for batch in selected_batches:
                for cell_id, gene, split_half, count in zip(
                    batch.column("cell_id").to_pylist(),
                    batch.column("feature_name").to_pylist(),
                    batch.column("split_half").to_pylist(),
                    batch.column("gene_count").to_pylist(),
                ):
                    row = row_for_id[str(cell_id)]
                    column = gene_for_name[str(gene)]
                    value = float(count)
                    counts[row, column] += value
                    (half_a if int(split_half) == 0 else half_b)[row, column] = value
            if not np.array_equal(counts, half_a + half_b):
                raise ValueError("HEST transcript split-half counts do not reconstruct targets")
            return (
                counts,
                totals,
                int(totals.sum()),
                half_a,
                half_b,
                library_half_a,
                library_half_b,
            )

        (
            nucleus_counts,
            nucleus_library,
            nucleus_eligible,
            nucleus_half_a,
            nucleus_half_b,
            nucleus_library_half_a,
            nucleus_library_half_b,
        ) = aggregate(" AND t.overlaps_nucleus = 1")
        (
            whole_counts,
            whole_library,
            whole_eligible,
            whole_half_a,
            whole_half_b,
            whole_library_half_a,
            whole_library_half_b,
        ) = aggregate("")
        qv_summary = np.zeros((len(ordered_ids), 3), dtype=np.float32)
        qv_query = (
            """
            SELECT t.cell_id, MIN(t.qv) AS minimum_qv, MEDIAN(t.qv) AS median_qv,
                   AVG(t.qv) AS mean_qv
            FROM read_parquet(?) AS t
            INNER JOIN registered_cells AS r USING (cell_id)
            WHERE t.qv >= ?
        """
            + exclusion
            + " GROUP BY t.cell_id"
        )
        qv_parameters: list[object] = [str(path), float(minimum_qv)]
        qv_parameters.extend(excluded_prefixes)
        for cell_id, qv_minimum, qv_median, qv_mean in connection.execute(
            qv_query, qv_parameters
        ).fetchall():
            qv_summary[row_for_id[str(cell_id)]] = (
                float(qv_minimum),
                float(qv_median),
                float(qv_mean),
            )
    finally:
        connection.close()
    return ExpressionTargets(
        nucleus_counts=nucleus_counts,
        whole_cell_counts=whole_counts,
        nucleus_half_a_counts=nucleus_half_a,
        nucleus_half_b_counts=nucleus_half_b,
        whole_cell_half_a_counts=whole_half_a,
        whole_cell_half_b_counts=whole_half_b,
        nucleus_library_size_half_a=nucleus_library_half_a,
        nucleus_library_size_half_b=nucleus_library_half_b,
        whole_cell_library_size_half_a=whole_library_half_a,
        whole_cell_library_size_half_b=whole_library_half_b,
        nucleus_library_sizes=nucleus_library,
        whole_cell_library_sizes=whole_library,
        nucleus_eligible_transcripts=nucleus_eligible,
        whole_cell_eligible_transcripts=whole_eligible,
        whole_cell_qv_summary=qv_summary,
    )


def _update_transcript_identity_manifest(
    path: Path,
    cell_ids: Sequence[str],
    genes: Sequence[str],
    *,
    minimum_qv: float,
    section_id: str,
    identity_hasher: "hashlib._Hash",
) -> int:
    """Hash section-qualified eligible target IDs without materializing them in memory."""

    prefix = str(section_id) + ":"
    scoped_cell_ids = tuple(str(value) for value in cell_ids)
    if any(not value.startswith(prefix) or len(value) == len(prefix) for value in scoped_cell_ids):
        raise ValueError("HEST transcript receipt cell identities are not section-scoped")
    raw_cell_ids = tuple(value[len(prefix) :] for value in scoped_cell_ids)

    try:
        import duckdb
        import pyarrow as pa
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install HEIR with the hest optional dependencies") from error
    connection = duckdb.connect(database=":memory:")
    count = 0
    try:
        connection.register("retained_cells", pa.table({"cell_id": raw_cell_ids}))
        result = connection.execute(
            """
            SELECT DISTINCT CAST(t.transcript_id AS VARCHAR) AS transcript_id
            FROM read_parquet(?) AS t
            INNER JOIN retained_cells AS r USING (cell_id)
            WHERE t.qv >= ? AND t.feature_name = ANY(?)
            ORDER BY transcript_id
            """,
            [str(path), float(minimum_qv), list(genes)],
        )
        batches = (
            result.to_arrow_reader(65_536)
            if hasattr(result, "to_arrow_reader")
            else result.fetch_record_batch(65_536)
        )
        for batch in batches:
            for transcript_id in batch.column("transcript_id").to_pylist():
                identity_hasher.update(
                    (section_id + "\0" + str(transcript_id) + "\n").encode("utf-8")
                )
                count += 1
    finally:
        connection.close()
    return count


def _log_cpm(counts: np.ndarray, library_sizes: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    library = np.asarray(library_sizes, dtype=np.float64)
    if (
        values.ndim != 2
        or library.shape != (len(values),)
        or np.any(values < 0)
        or np.any(library < 0)
        or not np.isfinite(values).all()
    ):
        raise ValueError("HEST registered expression is malformed")
    result = np.zeros_like(values)
    valid = library > 0
    result[valid] = np.log1p(values[valid] * (10_000.0 / library[valid, None]))
    return result


def _block_role(sample_id: str, block_x: int, block_y: int, salt: str) -> str:
    key = "%s:%d:%d:%s" % (sample_id, block_x, block_y, salt)
    return "reference" if hashlib.sha256(key.encode("utf-8")).digest()[0] & 1 else "evaluation"


def _reference_split_matrix(
    donors: np.ndarray,
    section_ids: np.ndarray,
    fine_type_labels: np.ndarray,
    block_ids: np.ndarray,
    primary_roles: np.ndarray,
    split_protocol: Mapping[str, object],
    *,
    minimum_reference: int,
    full_support_donors: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Freeze alternate reference-block subsets while keeping evaluation rows fixed."""

    primary_id = str(split_protocol["primary_split_id"])
    alternates = split_protocol["alternate_splits"]
    if primary_id != "primary" or not isinstance(alternates, list) or len(alternates) != 2:
        raise ValueError("HEST reference-split registry is malformed")
    split_ids = (primary_id,) + tuple(str(value["split_id"]) for value in alternates)
    roles = np.full((len(donors), len(split_ids)), "excluded", dtype="<U10")
    roles[:, 0] = primary_roles.astype(str)
    primary_evaluation = primary_roles.astype(str) == "evaluation"
    primary_reference = primary_roles.astype(str) == "reference"
    full_support = set(str(value) for value in full_support_donors)
    for split_index, declaration in enumerate(alternates, start=1):
        salt = str(declaration["salt"])
        retention = float(declaration["initial_reference_retention_fraction"])
        roles[primary_evaluation, split_index] = "evaluation"
        donor_values = donors.astype(str)
        section_values = section_ids.astype(str)
        block_values = block_ids.astype(str)
        for donor in sorted(set(donor_values.tolist())):
            for section_id in sorted(set(section_values[donor_values == donor].tolist())):
                stratum_reference = (
                    primary_reference & (donor_values == donor) & (section_values == section_id)
                )
                candidate_blocks = sorted(set(block_values[stratum_reference].tolist()))
                scores = {
                    block: int.from_bytes(
                        hashlib.sha256(
                            (donor + "\0" + section_id + "\0" + block + "\0" + salt).encode("utf-8")
                        ).digest()[:8],
                        "big",
                    )
                    / 2**64
                    for block in candidate_blocks
                }
                selected = {block for block in candidate_blocks if scores[block] < retention}
                section_types = sorted(set(fine_type_labels[stratum_reference].tolist()))
                # Development donor/section/type strata always retain full
                # support. A locked stratum that already fails the primary
                # minimum remains visible for ITA accounting, but alternates
                # never make it spuriously evaluable.
                required_by_type = {
                    type_index: (
                        minimum_reference
                        if donor in full_support
                        or np.count_nonzero(stratum_reference & (fine_type_labels == type_index))
                        >= minimum_reference
                        else 0
                    )
                    for type_index in section_types
                }

                def supported(block_selection: set[str]) -> bool:
                    selected_rows = stratum_reference & np.isin(
                        block_values, np.asarray(sorted(block_selection), dtype=str)
                    )
                    return all(
                        np.count_nonzero(selected_rows & (fine_type_labels == type_index))
                        >= required_by_type[type_index]
                        for type_index in section_types
                    )

                for block in sorted(candidate_blocks, key=lambda value: (scores[value], value)):
                    if supported(selected):
                        break
                    selected.add(block)
                if not supported(selected):
                    raise ValueError(
                        "an alternate HEST reference split lacks donor/section/type support"
                    )
                selected_rows = stratum_reference & np.isin(
                    block_values, np.asarray(sorted(selected), dtype=str)
                )
                roles[selected_rows, split_index] = "reference"
    if not np.all(roles[primary_evaluation, :] == "evaluation"):
        raise ValueError("alternate reference splits changed primary evaluation rows")
    return split_ids, roles


@dataclass(frozen=True)
class SpatialIdentity:
    block_id: str
    roi_id: str
    pool_role: str
    guard_pass: bool


def _spatial_identity(
    sample_id: str,
    centre: Tuple[float, float],
    pixel_size_um: float,
    *,
    block_um: float,
    roi_um: float,
    guard_um: float,
    salt: str,
) -> SpatialIdentity:
    x_um, y_um = centre[0] * pixel_size_um, centre[1] * pixel_size_um
    block_x, block_y = int(math.floor(x_um / block_um)), int(math.floor(y_um / block_um))
    roi_x, roi_y = int(math.floor(x_um / roi_um)), int(math.floor(y_um / roi_um))
    local_x, local_y = x_um % block_um, y_um % block_um
    distance = min(local_x, block_um - local_x, local_y, block_um - local_y)
    return SpatialIdentity(
        block_id="%s:block:%d:%d" % (sample_id, block_x, block_y),
        roi_id="%s:roi:%d:%d" % (sample_id, roi_x, roi_y),
        pool_role=_block_role(sample_id, block_x, block_y, salt),
        guard_pass=distance >= guard_um,
    )


def _cellvit_sensitivity(
    native_centres: np.ndarray,
    cellvit_path: Path,
    *,
    pixel_size_um: float,
    radius_um: float,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, Tuple[str, ...], np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install HEIR with the hest optional dependencies") from error
    schema = parquet.read_schema(cellvit_path)
    if not {"geometry", "class"} <= set(schema.names):
        raise ValueError("CellViT sensitivity Parquet schema is incomplete")
    table = parquet.read_table(cellvit_path, columns=["geometry", "class"])
    centres = np.asarray([_polygon_centroid(value) for value in table["geometry"].to_pylist()])
    classes = np.asarray(table["class"].to_pylist()).astype(str)
    class_names = tuple(str(value) for value in class_names)
    if not class_names or not set(classes.tolist()) <= set(class_names):
        raise ValueError("CellViT sensitivity segmentation is empty")
    radius_pixels = radius_um / pixel_size_um
    bin_size = max(radius_pixels, 1.0)
    bins: Dict[Tuple[int, int], list[int]] = {}
    for index, (x, y) in enumerate(centres):
        bins.setdefault((int(x // bin_size), int(y // bin_size)), []).append(index)
    class_index = {name: index for index, name in enumerate(class_names)}
    result = np.zeros((len(native_centres), len(class_names)), dtype=np.float32)
    nearest_centres = np.empty_like(native_centres, dtype=np.float64)
    nearest_distances = np.full(len(native_centres), np.inf, dtype=np.float64)
    radius_squared = radius_pixels * radius_pixels
    for row, (x, y) in enumerate(native_centres):
        centre_bin = (int(x // bin_size), int(y // bin_size))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for index in bins.get((centre_bin[0] + dx, centre_bin[1] + dy), ()):
                    distance = (centres[index, 0] - x) ** 2 + (centres[index, 1] - y) ** 2
                    if distance <= radius_squared:
                        result[row, class_index[classes[index]]] += 1.0
                        if distance < nearest_distances[row]:
                            nearest_distances[row] = distance
                            nearest_centres[row] = centres[index]
    if not np.isfinite(nearest_distances).all():
        raise ValueError("a native Xenium nucleus has no CellViT sensitivity match")
    names = tuple("cellvit_log1p_count_%s" % name for name in class_names)
    return (
        np.log1p(result),
        names,
        nearest_centres,
        np.sqrt(nearest_distances) * pixel_size_um,
    )


class _TiffPatchReader:
    def __init__(self, path: Path):
        try:
            import tifffile
            import zarr
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("install HEIR with the hest optional dependencies") from error
        self._tiff = tifffile.TiffFile(path)
        self._store = self._tiff.series[0].aszarr(level=0)
        self._array = zarr.open(self._store, mode="r")
        axes = self._tiff.series[0].axes
        if axes not in {"YXS", "YXC"} or len(self._array.shape) != 3 or self._array.shape[2] < 3:
            self.close()
            raise ValueError("HEST WSI level zero must be an RGB YXS image")
        self.height, self.width = int(self._array.shape[0]), int(self._array.shape[1])

    def read(self, centre: Tuple[float, float], size: int) -> np.ndarray:
        patch, _, _, _ = self.read_with_padding(centre, size)
        return patch

    def read_with_padding(
        self, centre: Tuple[float, float], size: int
    ) -> Tuple[np.ndarray, float, int, int]:
        if size <= 0:
            raise ValueError("HEST crop size must be positive")
        if not (0 <= centre[0] < self.width and 0 <= centre[1] < self.height):
            raise ValueError("native Xenium nucleus centre falls outside the registered H&E WSI")
        x0 = int(round(centre[0] - size / 2.0))
        y0 = int(round(centre[1] - size / 2.0))
        x1, y1 = x0 + size, y0 + size
        source_x0, source_y0 = max(x0, 0), max(y0, 0)
        source_x1, source_y1 = min(x1, self.width), min(y1, self.height)
        patch = np.full((size, size, 3), 255, dtype=np.uint8)
        copied_pixels = max(source_x1 - source_x0, 0) * max(source_y1 - source_y0, 0)
        if source_x1 > source_x0 and source_y1 > source_y0:
            value = np.asarray(
                self._array[source_y0:source_y1, source_x0:source_x1, :3], dtype=np.uint8
            )
            patch[
                source_y0 - y0 : source_y1 - y0,
                source_x0 - x0 : source_x1 - x0,
            ] = value
        padding_fraction = 1.0 - copied_pixels / float(size * size)
        return patch, padding_fraction, x0, y0

    def close(self) -> None:
        if getattr(self, "_store", None) is not None and hasattr(self._store, "close"):
            self._store.close()
        if getattr(self, "_tiff", None) is not None:
            self._tiff.close()

    def __enter__(self) -> "_TiffPatchReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _randomly_translated_polygon(
    vertices: np.ndarray,
    *,
    x0: int,
    y0: int,
    size: int,
    key: str,
) -> list[Tuple[float, float]]:
    """Translate an exact native shape to a deterministic non-target canvas location."""

    local = np.asarray(vertices, dtype=np.float64) - np.asarray([x0, y0], dtype=np.float64)
    minimum = local.min(axis=0)
    maximum = local.max(axis=0)
    width, height = maximum - minimum
    if width >= size - 4 or height >= size - 4:
        raise ValueError("native mask cannot be translated within the frozen canvas")
    shape_centre = (minimum + maximum) / 2.0
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    low = np.asarray([width / 2.0 + 2.0, height / 2.0 + 2.0])
    high = np.asarray([size - width / 2.0 - 2.0, size - height / 2.0 - 2.0])
    fractions = np.asarray([digest[0] / 255.0, digest[1] / 255.0])
    destination = low + fractions * (high - low)
    canvas_centre = np.asarray([size / 2.0, size / 2.0])
    minimum_displacement = max(width, height, size * 0.2)
    if np.linalg.norm(destination - canvas_centre) < minimum_displacement:
        corners = np.asarray(
            [
                [low[0], low[1]],
                [low[0], high[1]],
                [high[0], low[1]],
                [high[0], high[1]],
            ]
        )
        destination = corners[digest[2] % len(corners)]
    translated = local + (destination - shape_centre)
    return [(float(x), float(y)) for x, y in translated]


def _mask_replacement(
    patch: np.ndarray,
    fill_mode: str,
    *,
    blur_sigma_pixels: float,
) -> np.ndarray:
    if fill_mode == "white":
        return np.full_like(patch, 255)
    if fill_mode == "mean_color":
        tissue = np.any(patch < 245, axis=2)
        pixels = patch[tissue] if np.any(tissue) else patch.reshape(-1, 3)
        colour = np.rint(pixels.mean(axis=0)).astype(np.uint8)
        return np.broadcast_to(colour, patch.shape).copy()
    if fill_mode == "blurred":
        try:
            from PIL import Image, ImageFilter
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("install HEIR with the hest optional dependencies") from error
        image = Image.fromarray(patch, mode="RGB")
        return np.asarray(
            image.filter(ImageFilter.GaussianBlur(radius=blur_sigma_pixels)), dtype=np.uint8
        ).copy()
    raise ValueError("masked crop needs a supported non-empty fill mode")


def _render_crop_variant(
    slide: _TiffPatchReader,
    centre: Tuple[float, float],
    nucleus_vertices: np.ndarray,
    cell_vertices: np.ndarray,
    variant: CropVariant,
    crop_manifest: CropManifest,
) -> Tuple[np.ndarray, float, float]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install HEIR with the hest optional dependencies") from error
    size = int(round(variant.diameter_um / crop_manifest.source_mpp))
    patch, padding_fraction, x0, y0 = slide.read_with_padding(centre, size)
    if variant.mask_mode == "none":
        return patch, padding_fraction, 0.0
    if variant.mask_mode == "blank":
        return np.full_like(patch, 255), padding_fraction, 1.0
    mask_image = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(mask_image)
    if variant.mask_mode in {
        "keep_nucleus",
        "keep_cell",
        "remove_cell",
        "random_keep_nucleus",
        "random_keep_cell",
        "random_remove_cell",
    }:
        uses_nucleus = variant.mask_mode in {"keep_nucleus", "random_keep_nucleus"}
        vertices = nucleus_vertices if uses_nucleus else cell_vertices
        if variant.mask_mode.startswith("random_"):
            polygon = _randomly_translated_polygon(
                vertices,
                x0=x0,
                y0=y0,
                size=size,
                key=(
                    "%s|%s|%.6f|%.6f"
                    % (
                        crop_manifest.random_mask_salt,
                        variant.crop_id,
                        centre[0],
                        centre[1],
                    )
                ),
            )
        else:
            polygon = [(float(x - x0), float(y - y0)) for x, y in vertices]
        draw.polygon(polygon, fill=1)
    elif variant.mask_mode == "remove_context_circle":
        radius = variant.inner_diameter_um / (2.0 * crop_manifest.source_mpp)
        local_x, local_y = centre[0] - x0, centre[1] - y0
        draw.ellipse(
            (local_x - radius, local_y - radius, local_x + radius, local_y + radius),
            fill=1,
        )
    else:  # pragma: no cover - manifest validation is exhaustive
        raise ValueError("unsupported crop mask mode")
    mask = np.asarray(mask_image, dtype=np.bool_)
    replacement = _mask_replacement(
        patch,
        variant.fill_mode,
        blur_sigma_pixels=crop_manifest.blur_sigma_um / crop_manifest.source_mpp,
    )
    if variant.mask_mode in {
        "keep_nucleus",
        "keep_cell",
        "random_keep_nucleus",
        "random_keep_cell",
    }:
        patch[~mask] = replacement[~mask]
    else:
        patch[mask] = replacement[mask]
    return patch, padding_fraction, float(mask.mean())


GEOMETRY_FEATURE_NAMES = (
    "area_um2",
    "perimeter_um",
    "equivalent_diameter_um",
    "major_axis_um",
    "minor_axis_um",
    "eccentricity",
    "circularity",
    "solidity",
    "convexity",
    "aspect_ratio",
    "extent",
    "orientation_sin_2theta",
    "orientation_cos_2theta",
)
REGION_TEXTURE_FEATURE_NAMES = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "gray_mean",
    "gray_std",
    "hematoxylin_od_mean",
    "hematoxylin_od_std",
    "eosin_od_mean",
    "eosin_od_std",
    "gray_entropy_16bin",
    "gradient_mean",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
)
STAIN_QUALITY_FEATURE_NAMES = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "rgb_q25_r",
    "rgb_q25_g",
    "rgb_q25_b",
    "rgb_q75_r",
    "rgb_q75_g",
    "rgb_q75_b",
    "od_mean_r",
    "od_mean_g",
    "od_mean_b",
    "od_std_r",
    "od_std_g",
    "od_std_b",
    "hematoxylin_od_mean",
    "hematoxylin_od_std",
    "hematoxylin_od_q90",
    "eosin_od_mean",
    "eosin_od_std",
    "eosin_od_q90",
    "gray_entropy_32bin",
    "laplacian_variance",
    "gradient_mean",
    "gradient_q90",
    "edge_fraction",
    "white_background_fraction",
    "tissue_fraction",
    "saturation_mean",
    "saturation_std",
    "center_to_background_um",
    "central_background_fraction",
)
LOCAL_DENSITY_FEATURE_NAMES = (
    "nearest_neighbor_distance_um",
    "mean_3_neighbor_distance_um",
    "mean_5_neighbor_distance_um",
    "neighbor_count_25um",
    "neighbor_count_50um",
    "neighbor_count_100um",
    "density_50um_per_mm2",
    "density_100um_per_mm2",
)
COORDINATE_BASE_FEATURE_NAMES = (
    "he_x_normalized",
    "he_y_normalized",
    "he_x_squared",
    "he_y_squared",
    "he_xy",
    "he_x_sin_2pi",
    "he_x_cos_2pi",
    "he_y_sin_2pi",
    "he_y_cos_2pi",
    "he_x_sin_4pi",
    "he_x_cos_4pi",
    "he_y_sin_4pi",
    "he_y_cos_4pi",
    "distance_to_wsi_edge_um",
)
SPATIAL_FEATURE_NAMES = (
    *COORDINATE_BASE_FEATURE_NAMES,
    *LOCAL_DENSITY_FEATURE_NAMES,
    "center_to_local_background_um",
    "central_background_fraction",
)


def _convex_hull(vertices: np.ndarray) -> np.ndarray:
    points = sorted({(float(x), float(y)) for x, y in np.asarray(vertices)})
    if len(points) <= 2:
        return np.asarray(points, dtype=np.float64)

    def cross(
        origin: Tuple[float, float],
        first: Tuple[float, float],
        second: Tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    lower: list[Tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[Tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _closed_perimeter(vertices: np.ndarray) -> float:
    values = np.asarray(vertices, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    return float(np.linalg.norm(np.roll(values, -1, axis=0) - values, axis=1).sum())


def _shoelace_area(vertices: np.ndarray) -> float:
    values = np.asarray(vertices, dtype=np.float64)
    if len(values) < 3:
        return 0.0
    shifted = np.roll(values, -1, axis=0)
    return float(abs(np.sum(values[:, 0] * shifted[:, 1] - shifted[:, 0] * values[:, 1])) / 2.0)


def _classical_geometry_features(
    vertices: np.ndarray, area_pixels2: float, source_mpp: float
) -> np.ndarray:
    values = np.asarray(vertices, dtype=np.float64)
    perimeter_pixels = _closed_perimeter(values)
    centred = values - values.mean(axis=0)
    covariance = np.cov(centred, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    major_variance, minor_variance = eigenvalues[order]
    major_axis_pixels = 4.0 * math.sqrt(float(major_variance))
    minor_axis_pixels = 4.0 * math.sqrt(float(minor_variance))
    eccentricity = (
        math.sqrt(max(0.0, 1.0 - minor_variance / major_variance)) if major_variance > 0 else 0.0
    )
    major_vector = eigenvectors[:, order[0]]
    orientation = math.atan2(float(major_vector[1]), float(major_vector[0]))
    hull = _convex_hull(values)
    hull_area = _shoelace_area(hull)
    hull_perimeter = _closed_perimeter(hull)
    width, height = np.maximum(values.max(axis=0) - values.min(axis=0), 1.0e-12)
    area_um2 = area_pixels2 * source_mpp**2
    perimeter_um = perimeter_pixels * source_mpp
    return np.asarray(
        [
            area_um2,
            perimeter_um,
            2.0 * math.sqrt(area_um2 / math.pi),
            major_axis_pixels * source_mpp,
            minor_axis_pixels * source_mpp,
            eccentricity,
            4.0 * math.pi * area_pixels2 / max(perimeter_pixels**2, 1.0e-12),
            area_pixels2 / max(hull_area, 1.0e-12),
            hull_perimeter / max(perimeter_pixels, 1.0e-12),
            major_axis_pixels / max(minor_axis_pixels, 1.0e-12),
            area_pixels2 / (width * height),
            math.sin(2.0 * orientation),
            math.cos(2.0 * orientation),
        ],
        dtype=np.float32,
    )


def _stain_concentrations(rgb: np.ndarray) -> np.ndarray:
    pixels = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    optical_density = -np.log(np.clip((pixels + 1.0) / 256.0, 1.0e-6, 1.0))
    stain_matrix = np.asarray([[0.650, 0.072], [0.704, 0.990], [0.286, 0.105]], dtype=np.float64)
    return optical_density @ np.linalg.pinv(stain_matrix).T


def _histogram_entropy(values: np.ndarray, bins: int) -> float:
    quantized = np.asarray(values, dtype=np.uint8).ravel().astype(np.int32)
    counts = np.bincount(quantized * bins // 256, minlength=bins)
    probabilities = counts[counts > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _glcm_features(gray: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float]:
    quantized = np.minimum(np.asarray(gray, dtype=np.uint8) // 32, 7)
    valid = mask[:, :-1] & mask[:, 1:]
    if not np.any(valid):
        return (0.0, 0.0, 0.0, 0.0)
    first = quantized[:, :-1][valid]
    second = quantized[:, 1:][valid]
    matrix = np.zeros((8, 8), dtype=np.float64)
    np.add.at(matrix, (first, second), 1.0)
    matrix += matrix.T
    matrix /= matrix.sum()
    row, column = np.indices(matrix.shape)
    contrast = float(np.sum(matrix * (row - column) ** 2))
    homogeneity = float(np.sum(matrix / (1.0 + (row - column) ** 2)))
    energy = float(np.sum(matrix**2))
    mean_row = float(np.sum(matrix * row))
    mean_column = float(np.sum(matrix * column))
    std_row = math.sqrt(float(np.sum(matrix * (row - mean_row) ** 2)))
    std_column = math.sqrt(float(np.sum(matrix * (column - mean_column) ** 2)))
    correlation = float(
        np.sum(matrix * (row - mean_row) * (column - mean_column))
        / max(std_row * std_column, 1.0e-12)
    )
    return contrast, homogeneity, energy, correlation


def _polygon_region_texture(
    patch: np.ndarray,
    vertices: np.ndarray,
    *,
    x0: int,
    y0: int,
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install HEIR with the hest optional dependencies") from error
    local = np.asarray(vertices, dtype=np.float64) - np.asarray([x0, y0])
    left = max(int(math.floor(local[:, 0].min())) - 1, 0)
    top = max(int(math.floor(local[:, 1].min())) - 1, 0)
    right = min(int(math.ceil(local[:, 0].max())) + 2, patch.shape[1])
    bottom = min(int(math.ceil(local[:, 1].max())) + 2, patch.shape[0])
    if right <= left or bottom <= top:
        raise ValueError("native polygon does not intersect its registered H&E crop")
    region = patch[top:bottom, left:right]
    mask_image = Image.new("L", (right - left, bottom - top), color=0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon([(float(x - left), float(y - top)) for x, y in local], fill=1)
    mask = np.asarray(mask_image, dtype=np.bool_)
    if not np.any(mask):
        raise ValueError("native polygon has no rasterized H&E pixels")
    pixels = region[mask].astype(np.float64)
    gray_image = np.rint(
        region[..., 0] * 0.299 + region[..., 1] * 0.587 + region[..., 2] * 0.114
    ).astype(np.uint8)
    gray = gray_image[mask].astype(np.float64)
    stains = _stain_concentrations(pixels)
    gradient_x = np.abs(np.diff(gray_image.astype(np.float64), axis=1))
    gradient_y = np.abs(np.diff(gray_image.astype(np.float64), axis=0))
    gradient_mean = (
        float(
            (gradient_x.mean() if gradient_x.size else 0.0)
            + (gradient_y.mean() if gradient_y.size else 0.0)
        )
        / 2.0
    )
    return np.asarray(
        [
            *pixels.mean(axis=0),
            *pixels.std(axis=0),
            gray.mean(),
            gray.std(),
            stains[:, 0].mean(),
            stains[:, 0].std(),
            stains[:, 1].mean(),
            stains[:, 1].std(),
            _histogram_entropy(gray.astype(np.uint8), 16),
            gradient_mean,
            *_glcm_features(gray_image, mask),
        ],
        dtype=np.float32,
    )


def _stain_quality_features(patch: np.ndarray, diameter_um: float) -> np.ndarray:
    stride = max(1, int(math.floor(min(patch.shape[:2]) / 64)))
    sample = patch[::stride, ::stride, :]
    pixels = sample.reshape(-1, 3).astype(np.float64)
    gray = np.rint(sample[..., 0] * 0.299 + sample[..., 1] * 0.587 + sample[..., 2] * 0.114).astype(
        np.uint8
    )
    optical_density = -np.log(np.clip((pixels + 1.0) / 256.0, 1.0e-6, 1.0))
    stains = _stain_concentrations(pixels)
    gray_float = gray.astype(np.float64)
    laplacian = (
        -4.0 * gray_float[1:-1, 1:-1]
        + gray_float[:-2, 1:-1]
        + gray_float[2:, 1:-1]
        + gray_float[1:-1, :-2]
        + gray_float[1:-1, 2:]
    )
    gx = np.diff(gray_float, axis=1)
    gy = np.diff(gray_float, axis=0)
    common_height = min(gx.shape[0], gy.shape[0])
    common_width = min(gx.shape[1], gy.shape[1])
    gradient = np.hypot(gx[:common_height, :common_width], gy[:common_height, :common_width])
    maximum = pixels.max(axis=1)
    saturation = np.divide(
        maximum - pixels.min(axis=1),
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    white = (gray > 240) & ((sample.max(axis=2) - sample.min(axis=2)) < 15)
    centre_y, centre_x = np.asarray(gray.shape) // 2
    white_coordinates = np.argwhere(white)
    if len(white_coordinates):
        distance_pixels = float(
            np.linalg.norm(white_coordinates - np.asarray([centre_y, centre_x]), axis=1).min()
        )
        center_to_background_um = distance_pixels * diameter_um / max(gray.shape)
    else:
        center_to_background_um = diameter_um / 2.0
    radius = max(1, int(round(min(gray.shape) * 0.125)))
    central = white[
        max(centre_y - radius, 0) : centre_y + radius + 1,
        max(centre_x - radius, 0) : centre_x + radius + 1,
    ]
    return np.asarray(
        [
            *pixels.mean(axis=0),
            *pixels.std(axis=0),
            *np.quantile(pixels, 0.25, axis=0),
            *np.quantile(pixels, 0.75, axis=0),
            *optical_density.mean(axis=0),
            *optical_density.std(axis=0),
            stains[:, 0].mean(),
            stains[:, 0].std(),
            np.quantile(stains[:, 0], 0.9),
            stains[:, 1].mean(),
            stains[:, 1].std(),
            np.quantile(stains[:, 1], 0.9),
            _histogram_entropy(gray, 32),
            float(laplacian.var()) if laplacian.size else 0.0,
            float(gradient.mean()) if gradient.size else 0.0,
            float(np.quantile(gradient, 0.9)) if gradient.size else 0.0,
            float(np.mean(gradient > 20.0)) if gradient.size else 0.0,
            float(white.mean()),
            float(1.0 - white.mean()),
            float(saturation.mean()),
            float(saturation.std()),
            center_to_background_um,
            float(central.mean()),
        ],
        dtype=np.float32,
    )


def _local_density_features(
    query_centres: Sequence[Tuple[float, float]],
    population_centres: Sequence[Tuple[float, float]],
    source_mpp: float,
) -> np.ndarray:
    maximum_radius_pixels = 100.0 / source_mpp
    bin_size = maximum_radius_pixels
    population = np.asarray(population_centres, dtype=np.float64).reshape(-1, 2)
    bins: Dict[Tuple[int, int], list[int]] = {}
    for index, (x, y) in enumerate(population):
        bins.setdefault((int(x // bin_size), int(y // bin_size)), []).append(index)
    result = np.empty((len(query_centres), len(LOCAL_DENSITY_FEATURE_NAMES)), dtype=np.float32)
    for row, (x, y) in enumerate(query_centres):
        cell_bin = (int(x // bin_size), int(y // bin_size))
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(bins.get((cell_bin[0] + dx, cell_bin[1] + dy), ()))
        distances_um = (
            np.linalg.norm(
                population[np.asarray(candidates, dtype=np.int64)] - np.asarray([x, y]), axis=1
            )
            * source_mpp
        )
        distances_um = np.sort(distances_um[distances_um > 1.0e-6])
        nearest = float(distances_um[0]) if len(distances_um) else 100.0
        mean_three = float(distances_um[:3].mean()) if len(distances_um) else 100.0
        mean_five = float(distances_um[:5].mean()) if len(distances_um) else 100.0
        count_25 = int(np.count_nonzero(distances_um <= 25.0))
        count_50 = int(np.count_nonzero(distances_um <= 50.0))
        count_100 = int(np.count_nonzero(distances_um <= 100.0))
        result[row] = (
            nearest,
            mean_three,
            mean_five,
            count_25,
            count_50,
            count_100,
            count_50 / (math.pi * 0.05**2),
            count_100 / (math.pi * 0.1**2),
        )
    return result


def _categorical_one_hot(values: Sequence[str], prefix: str) -> Tuple[np.ndarray, Tuple[str, ...]]:
    names = tuple(sorted({str(value) for value in values}))
    index = {name: column for column, name in enumerate(names)}
    matrix = np.zeros((len(values), len(names)), dtype=np.float32)
    for row, value in enumerate(values):
        matrix[row, index[str(value)]] = 1.0
    return matrix, tuple("%s=%s" % (prefix, name) for name in names)


def _reference_evaluation_balance(
    donor_ids: np.ndarray,
    pool_roles: np.ndarray,
    continuous: Mapping[str, np.ndarray],
    categorical: Mapping[str, Sequence[str]],
) -> Dict[str, object]:
    """Summarize frozen-pool balance without using molecular outcomes."""

    result: Dict[str, object] = {}
    for donor in sorted(set(donor_ids.astype(str).tolist())):
        donor_mask = donor_ids.astype(str) == donor
        reference = donor_mask & (pool_roles.astype(str) == "reference")
        evaluation = donor_mask & (pool_roles.astype(str) == "evaluation")
        if not np.any(reference) or not np.any(evaluation):
            raise ValueError("reference/evaluation balance needs both roles per donor")
        donor_summary: Dict[str, object] = {}
        for family, raw in continuous.items():
            matrix = np.asarray(raw, dtype=np.float64)
            if matrix.ndim == 1:
                matrix = matrix[:, None]
            if matrix.shape[0] != len(donor_ids) or not np.isfinite(matrix).all():
                raise ValueError("reference/evaluation balance matrix is malformed")
            reference_values = matrix[reference]
            evaluation_values = matrix[evaluation]
            pooled = np.sqrt((reference_values.var(axis=0) + evaluation_values.var(axis=0)) / 2.0)
            global_scale = matrix[donor_mask].std(axis=0)
            scale = np.maximum.reduce((pooled, global_scale, np.full(matrix.shape[1], 1.0e-6)))
            absolute_smd = (
                np.abs(reference_values.mean(axis=0) - evaluation_values.mean(axis=0)) / scale
            )
            donor_summary[family] = {
                "columns": int(matrix.shape[1]),
                "maximum_absolute_standardized_mean_difference": float(
                    absolute_smd.max(initial=0.0)
                ),
                "median_absolute_standardized_mean_difference": float(np.median(absolute_smd)),
            }
        for family, raw in categorical.items():
            values = np.asarray(tuple(str(value) for value in raw))
            if values.shape != (len(donor_ids),):
                raise ValueError("reference/evaluation categorical balance is malformed")
            levels = sorted(set(values[donor_mask].tolist()))
            reference_fraction = np.asarray(
                [np.mean(values[reference] == level) for level in levels]
            )
            evaluation_fraction = np.asarray(
                [np.mean(values[evaluation] == level) for level in levels]
            )
            donor_summary[family] = {
                "levels": levels,
                "total_variation_distance": float(
                    0.5 * np.abs(reference_fraction - evaluation_fraction).sum()
                ),
            }
        result[donor] = donor_summary
    return result


@dataclass
class _Rows:
    observation_ids: list[str]
    cell_ids: list[str]
    donor_ids: list[str]
    sample_ids: list[str]
    split_ids: list[str]
    block_ids: list[str]
    roi_ids: list[str]
    pool_roles: list[str]
    type_labels: list[int]
    broad_type_labels: list[int]
    fine_type_ids: list[str]
    disease_statuses: list[str]
    site_ids: list[str]
    batch_ids: list[str]
    centres: list[Tuple[float, float]]
    cell_centres: list[Tuple[float, float]]
    nucleus_vertices: list[np.ndarray]
    cell_vertices: list[np.ndarray]
    nucleus_areas_um2: list[float]
    cell_areas_um2: list[float]
    nucleus_centroid_inside_cell: list[bool]
    source_centres: list[Tuple[float, float]]
    annotation_he_centres: list[Tuple[float, float]]
    annotation_nucleus_distances_um: list[float]
    annotation_cell_distances_um: list[float]
    affine_residual_p95_um: list[float]
    ncount_rna: list[float]
    nfeature_rna: list[float]
    percent_negative_or_unassigned: list[float]
    sample_rows: list[Tuple[int, int, Sample, Path]]
    targets: list[np.ndarray]
    whole_cell_targets: list[np.ndarray]
    target_counts: list[np.ndarray]
    whole_cell_target_counts: list[np.ndarray]
    nucleus_half_a_counts: list[np.ndarray]
    nucleus_half_b_counts: list[np.ndarray]
    whole_cell_half_a_counts: list[np.ndarray]
    whole_cell_half_b_counts: list[np.ndarray]
    nucleus_library_sizes: list[int]
    whole_cell_library_sizes: list[int]
    nucleus_library_size_half_a: list[int]
    nucleus_library_size_half_b: list[int]
    whole_cell_library_size_half_a: list[int]
    whole_cell_library_size_half_b: list[int]
    nucleus_detected_target_genes: list[int]
    whole_cell_detected_target_genes: list[int]
    transcript_qv_summaries: list[np.ndarray]
    technical: list[float]
    cellvit: list[np.ndarray]
    cellvit_centres: list[Tuple[float, float]]
    cellvit_distances_um: list[float]
    local_density: list[np.ndarray]
    cellvit_names: Optional[Tuple[str, ...]] = None


def _write_npz(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz.tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".json.tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_source(
    protocol_path: Path,
    study_manifest_path: Path,
    encoder_manifest_path: Path,
    crop_manifest_path: Path,
    data_root: Path,
    model_dir: Path,
    output_path: Path,
    plan_output_path: Path,
    qc_output_path: Path,
    *,
    annotation_receipt_path: Optional[Path] = None,
    annotation_predictions_path: Optional[Path] = None,
    device: str = "cuda",
    batch_size: int = 64,
    encoder: Optional[FrozenPatchEncoder] = None,
) -> None:
    protocol_path = protocol_path.expanduser().resolve()
    study_manifest_path = study_manifest_path.expanduser().resolve()
    encoder_manifest_path = encoder_manifest_path.expanduser().resolve()
    crop_manifest_path = crop_manifest_path.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    plan_output_path = plan_output_path.expanduser().resolve()
    qc_output_path = qc_output_path.expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[1]
    manifest_identity = StudyManifest.load(study_manifest_path)
    if manifest_identity.study_stage == "measurement_development":
        study_manifest = StudyManifest.load(
            study_manifest_path,
            require_status="locked",
            verify_runtime=True,
            repository_root=repository_root,
        )
        opening_receipt_sha256: Optional[str] = None
    elif manifest_identity.study_stage == "confirmatory_morphology":
        study_manifest = StudyManifest.load(
            study_manifest_path,
            require_status="opened",
            verify_runtime=True,
            require_clean_runtime=True,
            verify_container_digest=True,
            repository_root=repository_root,
        )
        opening = study_manifest.content["opening"]
        if "H-CELL" not in opening["permitted_claims"]:
            raise ValueError("opened study manifest does not permit the H-CELL claim")
        opening_receipt_sha256 = str(opening["opening_receipt_sha256"])
    else:  # StudyManifest.load already rejects this; retain a local fail-closed guard.
        raise ValueError("unsupported HEST study stage")
    encoder_manifest = _load_encoder_manifest(encoder_manifest_path)
    crop_manifest = _load_crop_manifest(crop_manifest_path)
    protocol = _read_json(protocol_path)
    samples = _validate_protocol(protocol, encoder_manifest, crop_manifest)
    if study_manifest.content["analysis_plan_sha256"] != _sha256_file(protocol_path):
        raise ValueError("locked study manifest does not bind the HEST protocol")
    if study_manifest.development_donors != tuple(
        protocol["development_donors"]
    ) or study_manifest.locked_test_donors != tuple(protocol["locked_test_donors"]):
        raise ValueError("locked study manifest donor partitions differ from the protocol")
    if study_manifest.content["candidate_target_gene_panel_sha256"] != _canonical_sha256(
        list(protocol["gene_ids"])
    ):
        raise ValueError("locked study manifest candidate target panel differs from the protocol")
    if study_manifest.content["type_marker_panel_sha256"] != _canonical_sha256(
        list(protocol["fine_type_marker_gene_ids"])
    ):
        raise ValueError("locked study manifest fine-type marker panel differs from the protocol")
    if study_manifest.study_stage == "measurement_development" and float(
        study_manifest.content["decision_thresholds"]["required_opposite_pool_guard_um"]
    ) != float(protocol["opposite_pool_guard_um"]):
        raise ValueError(
            "locked measurement manifest opposite-pool guard differs from the HEST protocol"
        )
    protocol_independence = _validate_protocol_independence_contract(
        protocol.get("label_target_independence")
    )
    manifest_independence = study_manifest.content["label_target_independence"]
    manifest_protocol_contract = {
        name: manifest_independence[name] for name in LABEL_TARGET_INDEPENDENCE_PROTOCOL_FIELDS
    }
    if dict(protocol_independence) != manifest_protocol_contract:
        raise ValueError(
            "locked study manifest label-target evidence differs from the HEST protocol"
        )
    independence_contract = dict(manifest_independence)
    if study_manifest.study_stage == "confirmatory_morphology":
        annotation_features = set(
            str(value) for value in independence_contract["ordered_annotation_feature_ids"]
        )
        target_features = tuple(
            str(value) for value in independence_contract["ordered_target_gene_ids"]
        )
        if (
            not target_features
            or not set(target_features) <= set(str(value) for value in protocol["gene_ids"])
            or annotation_features & set(target_features)
            or independence_contract["annotation_target_overlap_count"] != 0
        ):
            raise ValueError(
                "locked label-target evidence is not disjoint from the frozen target panel"
            )
    if study_manifest.study_stage == "measurement_development":
        samples = tuple(sample for sample in samples if sample.split_id == "development")
        source_scope = "development_donors_only"
        frozen_supported_types: tuple[str, ...] = ()
        planned_strata: set[str] = set()
    elif study_manifest.study_stage == "confirmatory_morphology":
        samples = tuple(samples)
        source_scope = "development_and_locked_after_confirmatory_opening"
        observations_contract = study_manifest.content["observations"]
        frozen_supported_types = tuple(
            str(value) for value in observations_contract["supported_fine_type_ids"]
        )
        planned_strata = set(_frozen_ita_stratum_ids(samples, frozen_supported_types))
    else:  # StudyManifest.load already rejects this; retain a local fail-closed guard.
        raise ValueError("unsupported HEST study stage")
    active_donors = tuple(sorted({sample.donor_id for sample in samples}))
    active_sample_ids = tuple(sample.sample_id for sample in samples)
    independent_artifacts: Optional[IndependentAnnotationArtifacts] = None
    if independence_contract["evidence_kind"] == "pending":
        if annotation_receipt_path is not None or annotation_predictions_path is not None:
            raise ValueError("pending label-target evidence cannot accept annotation artifacts")
    else:
        if annotation_receipt_path is None or annotation_predictions_path is None:
            raise ValueError(
                "non-pending label-target evidence requires its actual receipt and predictions"
            )
        independent_artifacts = _verify_independent_annotation_artifacts(
            annotation_receipt_path,
            annotation_predictions_path,
            independence_contract,
            prediction_donor_ids=active_donors,
        )
    annotation_artifact_inputs = {
        path
        for path in (
            None if independent_artifacts is None else independent_artifacts.receipt_path,
            None if independent_artifacts is None else independent_artifacts.predictions_path,
        )
        if path is not None
    }
    if (
        output_path
        in {
            protocol_path,
            study_manifest_path,
            encoder_manifest_path,
            crop_manifest_path,
            data_root,
            model_dir,
            plan_output_path,
            qc_output_path,
        }
        or output_path in annotation_artifact_inputs
        or plan_output_path
        in {
            protocol_path,
            study_manifest_path,
            encoder_manifest_path,
            crop_manifest_path,
            data_root,
            model_dir,
        }
        or plan_output_path in annotation_artifact_inputs
        or qc_output_path
        in {
            protocol_path,
            study_manifest_path,
            encoder_manifest_path,
            crop_manifest_path,
            data_root,
            model_dir,
            plan_output_path,
        }
        or qc_output_path in annotation_artifact_inputs
        or batch_size < 1
    ):
        raise ValueError("HEST output/input identity or batch size is invalid")
    broad_type_names = tuple(str(value) for value in protocol["broad_type_names"])
    markers = {
        str(name): tuple(str(gene) for gene in values)
        for name, values in protocol["type_markers"].items()
    }
    marker_genes = tuple(gene for name in broad_type_names for gene in markers[name])
    fine_marker_genes = tuple(str(value) for value in protocol["fine_type_marker_gene_ids"])
    target_genes = tuple(str(value) for value in protocol["gene_ids"])
    salt = str(protocol.get("pool_assignment_salt", "hest-xenium-v1"))
    full_annotation_declaration = _parse_file(
        protocol["annotation_export"], "GSE250346 annotation export"
    )
    if study_manifest.study_stage == "measurement_development":
        development_annotation = _parse_development_annotation_export(
            protocol.get("measurement_development_annotation_export"),
            active_sample_ids,
        )
        annotation_declaration = development_annotation.file
        annotation_expected_rows = development_annotation.row_count
        annotation_strict_scope = True
    else:
        annotation_declaration = full_annotation_declaration
        annotation_expected_rows = ANNOTATION_ROWS
        annotation_strict_scope = False
    annotation_path = _resolve_input(data_root, annotation_declaration)
    annotations = _read_annotations(
        annotation_path,
        allowed_sample_ids=active_sample_ids,
        independent_artifacts=independent_artifacts,
        expected_row_count=annotation_expected_rows,
        strict_sample_scope=annotation_strict_scope,
    )
    if independent_artifacts is None:
        label_source_sha256 = annotation_declaration.sha256
        label_source_kind = "gse250346_final_ct_and_lineage"
        label_field_primary = "final_CT"
        label_field_secondary = "final_lineage"
    else:
        label_source_sha256 = independent_artifacts.predictions_sha256
        label_source_kind = "independent_annotation_prediction_export"
        label_field_primary = "fine_type"
        label_field_secondary = "broad_lineage"
    rows = _Rows(
        observation_ids=[],
        cell_ids=[],
        donor_ids=[],
        sample_ids=[],
        split_ids=[],
        block_ids=[],
        roi_ids=[],
        pool_roles=[],
        type_labels=[],
        broad_type_labels=[],
        fine_type_ids=[],
        disease_statuses=[],
        site_ids=[],
        batch_ids=[],
        centres=[],
        cell_centres=[],
        nucleus_vertices=[],
        cell_vertices=[],
        nucleus_areas_um2=[],
        cell_areas_um2=[],
        nucleus_centroid_inside_cell=[],
        source_centres=[],
        annotation_he_centres=[],
        annotation_nucleus_distances_um=[],
        annotation_cell_distances_um=[],
        affine_residual_p95_um=[],
        ncount_rna=[],
        nfeature_rna=[],
        percent_negative_or_unassigned=[],
        sample_rows=[],
        targets=[],
        whole_cell_targets=[],
        target_counts=[],
        whole_cell_target_counts=[],
        nucleus_half_a_counts=[],
        nucleus_half_b_counts=[],
        whole_cell_half_a_counts=[],
        whole_cell_half_b_counts=[],
        nucleus_library_sizes=[],
        whole_cell_library_sizes=[],
        nucleus_library_size_half_a=[],
        nucleus_library_size_half_b=[],
        whole_cell_library_size_half_a=[],
        whole_cell_library_size_half_b=[],
        nucleus_detected_target_genes=[],
        whole_cell_detected_target_genes=[],
        transcript_qv_summaries=[],
        technical=[],
        cellvit=[],
        cellvit_centres=[],
        cellvit_distances_um=[],
        local_density=[],
    )
    resolved_provenance = []
    exclusion_counts: Dict[str, int] = {
        "native_cells_outside_high_qc_annotation": 0,
        "low_transcripts": 0,
        "spatial_guard": 0,
        "unsupported_donor_fine_type": 0,
    }

    for sample in samples:
        wsi_path = _resolve_input(data_root, sample.wsi)
        transcript_path = _resolve_input(data_root, sample.transcripts)
        cell_path = _resolve_input(data_root, sample.cell_seg)
        nucleus_path = _resolve_input(data_root, sample.nucleus_seg)
        cell_centres = _read_native_segmentation(cell_path)
        nucleus_centres = _read_native_segmentation(nucleus_path)
        annotated = annotations[sample.sample_id]
        if not set(annotated) <= set(cell_centres) or not set(annotated) <= set(nucleus_centres):
            raise ValueError("a high-QC annotation lacks its native Xenium cell/nucleus ID")
        registered = tuple(sorted(annotated))
        for cell_id in registered:
            fine_type = annotated[cell_id].fine_type
            if "|" in fine_type:
                raise ValueError("HEST fine type cannot contain the stratum delimiter")
            if study_manifest.study_stage == "measurement_development":
                planned_strata.add("%s|%s|%s" % (sample.donor_id, sample.sample_id, fine_type))
        exclusion_counts["native_cells_outside_high_qc_annotation"] += len(
            (set(cell_centres) | set(nucleus_centres)) - set(annotated)
        )
        coordinate_registration = _fit_coordinate_registration(
            transcript_path, sample.pixel_size_um
        )
        if coordinate_registration.residual_p95_um > float(
            protocol["maximum_affine_registration_residual_p95_um"]
        ):
            raise ValueError("HEST Xenium-to-H&E affine registration residual exceeds the protocol")
        annotation_he = coordinate_registration.transform(
            [annotated[cell_id].source_centroid for cell_id in registered]
        )
        registered_local_density = _local_density_features(
            [nucleus_centres[cell_id].centroid for cell_id in registered],
            [geometry.centroid for geometry in nucleus_centres.values()],
            sample.pixel_size_um,
        )
        registered_nucleus_centres = np.asarray(
            [nucleus_centres[cell_id].centroid for cell_id in registered], dtype=np.float64
        )
        registered_cell_centres = np.asarray(
            [cell_centres[cell_id].centroid for cell_id in registered], dtype=np.float64
        )
        annotation_nucleus_distances = (
            np.linalg.norm(annotation_he - registered_nucleus_centres, axis=1)
            * sample.pixel_size_um
        )
        annotation_cell_distances = (
            np.linalg.norm(annotation_he - registered_cell_centres, axis=1) * sample.pixel_size_um
        )
        annotation_distance_limit = float(protocol["maximum_annotation_nucleus_distance_p95_um"])
        annotation_distance_p95 = float(np.quantile(annotation_nucleus_distances, 0.95))
        registration_outlier_fraction = float(
            np.mean(annotation_nucleus_distances > annotation_distance_limit)
        )
        if annotation_distance_p95 > annotation_distance_limit:
            raise ValueError(
                "HEST annotation-to-nucleus registration distance exceeds the protocol"
            )
        if registration_outlier_fraction > float(protocol["maximum_registration_outlier_fraction"]):
            raise ValueError("HEST annotation-to-nucleus registration outlier fraction is too high")
        expression_targets = _aggregate_expression(
            transcript_path,
            registered,
            tuple(cell_centres),
            target_genes,
            marker_genes + fine_marker_genes + target_genes,
            minimum_qv=float(protocol["minimum_transcript_qv"]),
            excluded_prefixes=tuple(str(value) for value in protocol["excluded_feature_prefixes"]),
            split_salt=str(protocol["transcript_split_salt"]),
        )
        nucleus_expression = _log_cpm(
            expression_targets.nucleus_counts,
            expression_targets.nucleus_library_sizes,
        )
        whole_cell_expression = _log_cpm(
            expression_targets.whole_cell_counts,
            expression_targets.whole_cell_library_sizes,
        )
        labels = np.asarray(
            [annotated[cell_id].broad_label for cell_id in registered], dtype=np.int64
        )
        sample_start = len(rows.observation_ids)
        retained_centres = []
        for index, cell_id in enumerate(registered):
            spatial = _spatial_identity(
                sample.sample_id,
                nucleus_centres[cell_id].centroid,
                sample.pixel_size_um,
                block_um=float(protocol["spatial_block_um"]),
                roi_um=float(protocol["spatial_roi_um"]),
                guard_um=float(protocol["opposite_pool_guard_um"]),
                salt=salt,
            )
            if expression_targets.nucleus_library_sizes[index] < int(
                protocol["minimum_transcripts_per_cell"]
            ):
                exclusion_counts["low_transcripts"] += 1
                continue
            if not spatial.guard_pass:
                exclusion_counts["spatial_guard"] += 1
                continue
            scoped_cell_id = _section_scoped_cell_id(sample.sample_id, cell_id)
            rows.observation_ids.append(scoped_cell_id)
            rows.cell_ids.append(scoped_cell_id)
            rows.donor_ids.append(sample.donor_id)
            rows.sample_ids.append(sample.sample_id)
            rows.split_ids.append(sample.split_id)
            rows.block_ids.append(spatial.block_id)
            rows.roi_ids.append(spatial.roi_id)
            rows.pool_roles.append(spatial.pool_role)
            annotation = annotated[cell_id]
            rows.type_labels.append(-1)
            rows.broad_type_labels.append(int(labels[index]))
            rows.fine_type_ids.append(annotation.fine_type)
            rows.disease_statuses.append(annotation.disease_status)
            rows.site_ids.append(annotation.site_id)
            rows.batch_ids.append(annotation.batch_id)
            nucleus_geometry = nucleus_centres[cell_id]
            cell_geometry = cell_centres[cell_id]
            rows.centres.append(nucleus_geometry.centroid)
            rows.cell_centres.append(cell_geometry.centroid)
            rows.nucleus_vertices.append(nucleus_geometry.vertices)
            rows.cell_vertices.append(cell_geometry.vertices)
            rows.nucleus_areas_um2.append(nucleus_geometry.area_pixels2 * sample.pixel_size_um**2)
            rows.cell_areas_um2.append(cell_geometry.area_pixels2 * sample.pixel_size_um**2)
            rows.nucleus_centroid_inside_cell.append(
                _point_inside_polygon(nucleus_geometry.centroid, cell_geometry.vertices)
            )
            rows.source_centres.append(annotation.source_centroid)
            transformed_annotation = tuple(annotation_he[index].tolist())
            rows.annotation_he_centres.append(transformed_annotation)
            rows.annotation_nucleus_distances_um.append(
                float(
                    np.linalg.norm(annotation_he[index] - np.asarray(nucleus_geometry.centroid))
                    * sample.pixel_size_um
                )
            )
            rows.annotation_cell_distances_um.append(
                float(
                    np.linalg.norm(annotation_he[index] - np.asarray(cell_geometry.centroid))
                    * sample.pixel_size_um
                )
            )
            rows.affine_residual_p95_um.append(coordinate_registration.residual_p95_um)
            rows.ncount_rna.append(annotation.ncount_rna)
            rows.nfeature_rna.append(annotation.nfeature_rna)
            rows.percent_negative_or_unassigned.append(annotation.percent_negative_or_unassigned)
            rows.targets.append(nucleus_expression[index].astype(np.float32))
            rows.whole_cell_targets.append(whole_cell_expression[index].astype(np.float32))
            rows.target_counts.append(expression_targets.nucleus_counts[index].astype(np.float32))
            rows.whole_cell_target_counts.append(
                expression_targets.whole_cell_counts[index].astype(np.float32)
            )
            rows.nucleus_half_a_counts.append(
                expression_targets.nucleus_half_a_counts[index].astype(np.float32)
            )
            rows.nucleus_half_b_counts.append(
                expression_targets.nucleus_half_b_counts[index].astype(np.float32)
            )
            rows.whole_cell_half_a_counts.append(
                expression_targets.whole_cell_half_a_counts[index].astype(np.float32)
            )
            rows.whole_cell_half_b_counts.append(
                expression_targets.whole_cell_half_b_counts[index].astype(np.float32)
            )
            rows.nucleus_library_sizes.append(int(expression_targets.nucleus_library_sizes[index]))
            rows.whole_cell_library_sizes.append(
                int(expression_targets.whole_cell_library_sizes[index])
            )
            rows.nucleus_library_size_half_a.append(
                int(expression_targets.nucleus_library_size_half_a[index])
            )
            rows.nucleus_library_size_half_b.append(
                int(expression_targets.nucleus_library_size_half_b[index])
            )
            rows.whole_cell_library_size_half_a.append(
                int(expression_targets.whole_cell_library_size_half_a[index])
            )
            rows.whole_cell_library_size_half_b.append(
                int(expression_targets.whole_cell_library_size_half_b[index])
            )
            rows.nucleus_detected_target_genes.append(
                int(np.count_nonzero(expression_targets.nucleus_counts[index]))
            )
            rows.whole_cell_detected_target_genes.append(
                int(np.count_nonzero(expression_targets.whole_cell_counts[index]))
            )
            rows.transcript_qv_summaries.append(
                expression_targets.whole_cell_qv_summary[index].astype(np.float32)
            )
            rows.local_density.append(registered_local_density[index])
            rows.technical.append(
                math.log1p(float(expression_targets.nucleus_library_sizes[index]))
            )
            retained_centres.append(nucleus_geometry.centroid)
        sample_end = len(rows.observation_ids)
        rows.sample_rows.append((sample_start, sample_end, sample, wsi_path))
        if sample.cellvit_seg is not None:
            cellvit_path = _resolve_input(data_root, sample.cellvit_seg)
            sensitivity, names, nearest_centres, nearest_distances_um = _cellvit_sensitivity(
                np.asarray(retained_centres, dtype=np.float64).reshape(-1, 2),
                cellvit_path,
                pixel_size_um=sample.pixel_size_um,
                radius_um=float(protocol["cellvit_sensitivity_radius_um"]),
                class_names=tuple(str(value) for value in protocol["cellvit_class_names"]),
            )
            if rows.cellvit_names is None:
                rows.cellvit_names = names
            elif rows.cellvit_names != names:
                raise ValueError("CellViT sensitivity classes differ between HEST samples")
            rows.cellvit.extend(sensitivity)
            rows.cellvit_centres.extend(tuple(value) for value in nearest_centres.tolist())
            rows.cellvit_distances_um.extend(float(value) for value in nearest_distances_um)
        resolved_provenance.append(
            {
                "sample_id": sample.sample_id,
                "donor_id": sample.donor_id,
                "split_id": sample.split_id,
                "pixel_size_um": sample.pixel_size_um,
                "native_cell_count": len(cell_centres),
                "native_nucleus_count": len(nucleus_centres),
                "high_qc_annotation_count": len(annotated),
                "native_registered_intersection_count": len(registered),
                "retained_observation_count_before_fine_type_support": (sample_end - sample_start),
                "wsi_sha256": sample.wsi.sha256,
                "transcripts_sha256": sample.transcripts.sha256,
                "cell_seg_sha256": sample.cell_seg.sha256,
                "nucleus_seg_sha256": sample.nucleus_seg.sha256,
                "cellvit_seg_sha256": (
                    sample.cellvit_seg.sha256 if sample.cellvit_seg is not None else None
                ),
                "coordinate_registration_residual_p95_um": (
                    coordinate_registration.residual_p95_um
                ),
                "coordinate_registration_residual_max_um": (
                    coordinate_registration.residual_max_um
                ),
                "annotation_nucleus_distance_p50_um": float(
                    np.quantile(annotation_nucleus_distances, 0.5)
                ),
                "annotation_nucleus_distance_p95_um": annotation_distance_p95,
                "annotation_nucleus_distance_max_um": float(annotation_nucleus_distances.max()),
                "annotation_cell_distance_p95_um": float(
                    np.quantile(annotation_cell_distances, 0.95)
                ),
                "registration_outlier_fraction": registration_outlier_fraction,
                "nucleus_centroid_outside_cell_fraction": float(
                    np.mean(
                        [
                            not _point_inside_polygon(
                                nucleus_centres[cell_id].centroid,
                                cell_centres[cell_id].vertices,
                            )
                            for cell_id in registered
                        ]
                    )
                ),
                "nucleus_eligible_transcripts": (expression_targets.nucleus_eligible_transcripts),
                "whole_cell_eligible_transcripts": (
                    expression_targets.whole_cell_eligible_transcripts
                ),
            }
        )

    fine_type_names = tuple(
        sorted(
            {
                value.fine_type
                for sample_annotations in annotations.values()
                for value in sample_annotations.values()
            }
        )
    )
    fine_type_index = {name: index for index, name in enumerate(fine_type_names)}
    rows.type_labels = [fine_type_index[value] for value in rows.fine_type_ids]
    preliminary_donors = np.asarray(rows.donor_ids)
    preliminary_sections = np.asarray(rows.sample_ids)
    preliminary_roles = np.asarray(rows.pool_roles)
    preliminary_splits = np.asarray(rows.split_ids)
    preliminary_labels = np.asarray(rows.type_labels, dtype=np.int64)
    retained = np.ones(len(rows.observation_ids), dtype=np.bool_)
    if study_manifest.study_stage == "confirmatory_morphology":
        if not frozen_supported_types or not set(frozen_supported_types) <= set(fine_type_names):
            raise ValueError("confirmatory HEST source differs from the frozen H-MEAS type set")
        retained &= np.isin(np.asarray(rows.fine_type_ids), np.asarray(frozen_supported_types))
    minimum_reference = int(protocol["minimum_reference_cells_per_donor_section_type"])
    minimum_evaluation = int(protocol["minimum_evaluation_cells_per_donor_section_type"])
    for donor in sorted(set(preliminary_donors.tolist())):
        donor_mask = preliminary_donors == donor
        for section_id in sorted(set(preliminary_sections[donor_mask].tolist())):
            section_mask = donor_mask & (preliminary_sections == section_id)
            for type_index in sorted(set(preliminary_labels[section_mask].tolist())):
                local = section_mask & (preliminary_labels == type_index)
                insufficient = (
                    np.count_nonzero(local & (preliminary_roles == "reference")) < minimum_reference
                    or np.count_nonzero(local & (preliminary_roles == "evaluation"))
                    < minimum_evaluation
                )
                locked_stratum = bool(
                    study_manifest.study_stage == "confirmatory_morphology"
                    and np.all(preliminary_splits[local] == "locked_test")
                )
                if insufficient and not locked_stratum:
                    retained[local] = False
    for type_index in sorted(set(preliminary_labels[retained].tolist())):
        local = retained & (preliminary_labels == type_index)
        development_support = len(
            set(preliminary_donors[local & (preliminary_splits == "development")].tolist())
        )
        locked_support = len(
            set(preliminary_donors[local & (preliminary_splits == "locked_test")].tolist())
        )
        insufficient_development = development_support < int(
            protocol["minimum_development_donors_per_fine_type"]
        )
        if study_manifest.study_stage == "confirmatory_morphology":
            fine_type = fine_type_names[type_index]
            if fine_type in frozen_supported_types and insufficient_development:
                raise ValueError(
                    "a frozen H-MEAS fine type lost development support before confirmation"
                )
            # Locked support is an audit outcome, never a post-opening population selector.
            # Missing locked support remains in the planned denominator downstream.
            _ = locked_support
        elif insufficient_development:
            retained[preliminary_labels == type_index] = False
    exclusion_counts["unsupported_donor_fine_type"] = int(np.count_nonzero(~retained))
    old_sample_rows = tuple(rows.sample_rows)
    row_fields = (
        "observation_ids",
        "cell_ids",
        "donor_ids",
        "sample_ids",
        "split_ids",
        "block_ids",
        "roi_ids",
        "pool_roles",
        "type_labels",
        "broad_type_labels",
        "fine_type_ids",
        "disease_statuses",
        "site_ids",
        "batch_ids",
        "centres",
        "cell_centres",
        "nucleus_vertices",
        "cell_vertices",
        "nucleus_areas_um2",
        "cell_areas_um2",
        "nucleus_centroid_inside_cell",
        "source_centres",
        "annotation_he_centres",
        "annotation_nucleus_distances_um",
        "annotation_cell_distances_um",
        "affine_residual_p95_um",
        "ncount_rna",
        "nfeature_rna",
        "percent_negative_or_unassigned",
        "targets",
        "whole_cell_targets",
        "target_counts",
        "whole_cell_target_counts",
        "nucleus_half_a_counts",
        "nucleus_half_b_counts",
        "whole_cell_half_a_counts",
        "whole_cell_half_b_counts",
        "nucleus_library_sizes",
        "whole_cell_library_sizes",
        "nucleus_library_size_half_a",
        "nucleus_library_size_half_b",
        "whole_cell_library_size_half_a",
        "whole_cell_library_size_half_b",
        "nucleus_detected_target_genes",
        "whole_cell_detected_target_genes",
        "transcript_qv_summaries",
        "local_density",
        "technical",
    )
    for field in row_fields:
        values = getattr(rows, field)
        setattr(rows, field, [value for value, keep in zip(values, retained) if keep])
    if rows.cellvit_names is not None:
        rows.cellvit = [value for value, keep in zip(rows.cellvit, retained) if keep]
        rows.cellvit_centres = [
            value for value, keep in zip(rows.cellvit_centres, retained) if keep
        ]
        rows.cellvit_distances_um = [
            value for value, keep in zip(rows.cellvit_distances_um, retained) if keep
        ]
    rows.sample_rows = []
    offset = 0
    for provenance_row, (_, _, sample, path) in zip(resolved_provenance, old_sample_rows):
        count = rows.sample_ids.count(sample.sample_id)
        rows.sample_rows.append((offset, offset + count, sample, path))
        provenance_row["retained_observation_count"] = count
        offset += count

    observations = len(rows.observation_ids)
    if not observations or len(set(rows.observation_ids)) != observations:
        raise ValueError("HEST retained observation identities are empty or duplicated")
    planned_stratum_ids = tuple(sorted(planned_strata))
    planned_stratum_manifest_sha256 = _canonical_sha256(list(planned_stratum_ids))
    transcript_identity_hasher = hashlib.sha256()
    eligible_target_transcripts = 0
    for start, end, sample, _ in rows.sample_rows:
        eligible_target_transcripts += _update_transcript_identity_manifest(
            _resolve_input(data_root, sample.transcripts),
            rows.cell_ids[start:end],
            target_genes,
            minimum_qv=float(protocol["minimum_transcript_qv"]),
            section_id=sample.sample_id,
            identity_hasher=transcript_identity_hasher,
        )
    counted_eligible_targets = int(
        sum(float(np.asarray(value).sum()) for value in rows.whole_cell_target_counts)
    )
    if eligible_target_transcripts != counted_eligible_targets:
        raise ValueError("HEST transcript identity receipt differs from whole-cell target counts")
    donors = np.asarray(rows.donor_ids)
    sections = np.asarray(rows.sample_ids)
    labels = np.asarray(rows.type_labels, dtype=np.int64)
    roles = np.asarray(rows.pool_roles)
    blocks = np.asarray(rows.block_ids)
    minimum_reference = int(protocol["minimum_reference_cells_per_donor_section_type"])
    minimum_evaluation = int(protocol["minimum_evaluation_cells_per_donor_section_type"])
    for donor in active_donors:
        donor_mask = donors == str(donor)
        locked_donor = bool(
            study_manifest.study_stage == "confirmatory_morphology"
            and str(donor) in set(study_manifest.locked_test_donors)
        )
        for section_id in sorted(set(sections[donor_mask].tolist())):
            section_mask = donor_mask & (sections == section_id)
            if (
                set(roles[section_mask].tolist()) != {"reference", "evaluation"}
                and not locked_donor
            ):
                raise ValueError(
                    "each development donor/section needs spatially disjoint "
                    "reference/evaluation cells"
                )
            for type_index in sorted(set(labels[section_mask].tolist())):
                reference_count = np.count_nonzero(
                    section_mask & (roles == "reference") & (labels == type_index)
                )
                evaluation_count = np.count_nonzero(
                    section_mask & (roles == "evaluation") & (labels == type_index)
                )
                if (
                    reference_count < minimum_reference or evaluation_count < minimum_evaluation
                ) and not locked_donor:
                    raise ValueError(
                        "a HEST donor/section/type lacks the frozen reference/evaluation support"
                    )
            if set(blocks[section_mask & (roles == "reference")]) & set(
                blocks[section_mask & (roles == "evaluation")]
            ):
                raise ValueError("HEST reference/evaluation spatial blocks overlap")
    reference_split_ids, pool_roles_by_split = _reference_split_matrix(
        donors,
        sections,
        labels,
        blocks,
        roles,
        protocol["reference_splits"],
        minimum_reference=minimum_reference,
        full_support_donors=study_manifest.development_donors,
    )

    if encoder is None:
        encoder = create_frozen_encoder(model_dir, encoder_manifest, device)
    if (
        encoder.feature_width != encoder_manifest.feature_width
        or encoder.manifest_sha256 != encoder_manifest.sha256
    ):
        raise ValueError("HEST frozen encoder differs from its immutable manifest")
    crop_ids = tuple(variant.crop_id for variant in crop_manifest.variants)
    primary_crop_index = crop_ids.index(crop_manifest.primary_crop_id)
    primary_variant = crop_manifest.variants[primary_crop_index]
    image_features = np.empty(
        (observations, len(crop_manifest.variants), encoder_manifest.feature_width),
        dtype=np.float32,
    )
    crop_padding_fractions = np.empty((observations, len(crop_manifest.variants)), dtype=np.float32)
    crop_mask_fractions = np.empty_like(crop_padding_fractions)
    stain_quality_local = np.empty(
        (observations, len(STAIN_QUALITY_FEATURE_NAMES)), dtype=np.float32
    )
    nucleus_texture_features = np.empty(
        (observations, len(REGION_TEXTURE_FEATURE_NAMES)), dtype=np.float32
    )
    cell_texture_features = np.empty_like(nucleus_texture_features)
    nucleus_geometry_features = np.stack(
        [
            _classical_geometry_features(vertices, area / SOURCE_MPP**2, SOURCE_MPP)
            for vertices, area in zip(rows.nucleus_vertices, rows.nucleus_areas_um2)
        ]
    )
    cell_geometry_features = np.stack(
        [
            _classical_geometry_features(vertices, area / SOURCE_MPP**2, SOURCE_MPP)
            for vertices, area in zip(rows.cell_vertices, rows.cell_areas_um2)
        ]
    )
    cellvit_nearest_features = (
        np.empty((observations, encoder_manifest.feature_width), dtype=np.float32)
        if rows.cellvit_names is not None
        else None
    )
    cellvit_nearest_padding_fractions = (
        np.empty(observations, dtype=np.float32) if rows.cellvit_names is not None else None
    )
    coordinate_base_features = np.empty(
        (observations, len(COORDINATE_BASE_FEATURE_NAMES)), dtype=np.float32
    )
    for start, end, sample, wsi_path in rows.sample_rows:
        if start == end:
            continue
        with _TiffPatchReader(wsi_path) as slide:
            local_centres = rows.centres[start:end]
            local_nucleus_vertices = rows.nucleus_vertices[start:end]
            local_cell_vertices = rows.cell_vertices[start:end]
            for crop_index, variant in enumerate(crop_manifest.variants):
                for batch_start in range(0, end - start, batch_size):
                    batch_end = min(batch_start + batch_size, end - start)
                    rendered = [
                        _render_crop_variant(
                            slide,
                            local_centres[index],
                            local_nucleus_vertices[index],
                            local_cell_vertices[index],
                            variant,
                            crop_manifest,
                        )
                        for index in range(batch_start, batch_end)
                    ]
                    patches = np.stack([value[0] for value in rendered])
                    encoded = np.asarray(encoder.encode(patches), dtype=np.float32)
                    expected = (batch_end - batch_start, encoder_manifest.feature_width)
                    if encoded.shape != expected or not np.isfinite(encoded).all():
                        raise ValueError("HEST frozen encoder features are malformed")
                    output_slice = slice(start + batch_start, start + batch_end)
                    image_features[output_slice, crop_index] = encoded
                    crop_padding_fractions[output_slice, crop_index] = [
                        value[1] for value in rendered
                    ]
                    crop_mask_fractions[output_slice, crop_index] = [value[2] for value in rendered]
                    if crop_index == primary_crop_index:
                        primary_size = int(
                            round(primary_variant.diameter_um / crop_manifest.source_mpp)
                        )
                        for local_index, value in enumerate(rendered, start=batch_start):
                            patch = value[0]
                            centre = local_centres[local_index]
                            x0 = int(round(centre[0] - primary_size / 2.0))
                            y0 = int(round(centre[1] - primary_size / 2.0))
                            output_index = start + local_index
                            stain_quality_local[output_index] = _stain_quality_features(
                                patch, primary_variant.diameter_um
                            )
                            nucleus_texture_features[output_index] = _polygon_region_texture(
                                patch,
                                local_nucleus_vertices[local_index],
                                x0=x0,
                                y0=y0,
                            )
                            cell_texture_features[output_index] = _polygon_region_texture(
                                patch,
                                local_cell_vertices[local_index],
                                x0=x0,
                                y0=y0,
                            )
            if cellvit_nearest_features is not None:
                local_cellvit_centres = rows.cellvit_centres[start:end]
                for batch_start in range(0, end - start, batch_size):
                    batch_end = min(batch_start + batch_size, end - start)
                    rendered = [
                        _render_crop_variant(
                            slide,
                            local_cellvit_centres[index],
                            local_nucleus_vertices[index],
                            local_cell_vertices[index],
                            primary_variant,
                            crop_manifest,
                        )
                        for index in range(batch_start, batch_end)
                    ]
                    encoded = np.asarray(
                        encoder.encode(np.stack([value[0] for value in rendered])),
                        dtype=np.float32,
                    )
                    expected = (batch_end - batch_start, encoder_manifest.feature_width)
                    if encoded.shape != expected or not np.isfinite(encoded).all():
                        raise ValueError("HEST CellViT sensitivity encoder features are malformed")
                    output_slice = slice(start + batch_start, start + batch_end)
                    cellvit_nearest_features[output_slice] = encoded
                    assert cellvit_nearest_padding_fractions is not None
                    cellvit_nearest_padding_fractions[output_slice] = [
                        value[1] for value in rendered
                    ]
            xy = np.asarray(local_centres, dtype=np.float64)
            normalized = np.column_stack((xy[:, 0] / slide.width, xy[:, 1] / slide.height))
            edge_distance_um = (
                np.minimum.reduce(
                    (
                        xy[:, 0],
                        slide.width - xy[:, 0],
                        xy[:, 1],
                        slide.height - xy[:, 1],
                    )
                )
                * sample.pixel_size_um
            )
            coordinate_base_features[start:end] = np.column_stack(
                (
                    normalized,
                    normalized[:, 0] ** 2,
                    normalized[:, 1] ** 2,
                    normalized[:, 0] * normalized[:, 1],
                    np.sin(2.0 * math.pi * normalized[:, 0]),
                    np.cos(2.0 * math.pi * normalized[:, 0]),
                    np.sin(2.0 * math.pi * normalized[:, 1]),
                    np.cos(2.0 * math.pi * normalized[:, 1]),
                    np.sin(4.0 * math.pi * normalized[:, 0]),
                    np.cos(4.0 * math.pi * normalized[:, 0]),
                    np.sin(4.0 * math.pi * normalized[:, 1]),
                    np.cos(4.0 * math.pi * normalized[:, 1]),
                    edge_distance_um,
                )
            )

    section_stain_summary = np.empty_like(stain_quality_local)
    for start, end, _, _ in rows.sample_rows:
        if start < end:
            section_stain_summary[start:end] = np.median(stain_quality_local[start:end], axis=0)
    stain_features = np.column_stack((stain_quality_local, section_stain_summary))
    stain_feature_names = (
        *("local_%s" % name for name in STAIN_QUALITY_FEATURE_NAMES),
        *("section_median_%s" % name for name in STAIN_QUALITY_FEATURE_NAMES),
    )
    local_density_features = np.asarray(rows.local_density, dtype=np.float32)
    boundary_features = stain_quality_local[:, [-2, -1]]
    coordinate_features = np.column_stack(
        (coordinate_base_features, local_density_features, boundary_features)
    ).astype(np.float32)

    features = image_features[:, primary_crop_index]
    maximum_padding = float(crop_padding_fractions.max())
    if cellvit_nearest_padding_fractions is not None:
        maximum_padding = max(maximum_padding, float(cellvit_nearest_padding_fractions.max()))
    if maximum_padding > float(protocol["maximum_crop_padding_fraction"]):
        raise ValueError("HEST crop padding exceeds the frozen maximum fraction")

    target_values = np.stack(rows.targets).astype(np.float32)
    whole_cell_target_values = np.stack(rows.whole_cell_targets).astype(np.float32)
    target_counts = np.stack(rows.target_counts).astype(np.float32)
    whole_cell_target_counts = np.stack(rows.whole_cell_target_counts).astype(np.float32)
    nucleus_half_a_counts = np.stack(rows.nucleus_half_a_counts).astype(np.uint32)
    nucleus_half_b_counts = np.stack(rows.nucleus_half_b_counts).astype(np.uint32)
    whole_cell_half_a_counts = np.stack(rows.whole_cell_half_a_counts).astype(np.uint32)
    whole_cell_half_b_counts = np.stack(rows.whole_cell_half_b_counts).astype(np.uint32)
    if not np.array_equal(
        target_counts, nucleus_half_a_counts.astype(np.float32) + nucleus_half_b_counts
    ) or not np.array_equal(
        whole_cell_target_counts,
        whole_cell_half_a_counts.astype(np.float32) + whole_cell_half_b_counts,
    ):
        raise ValueError("HEST retained split-half matrices do not reconstruct target counts")
    if int(whole_cell_half_a_counts.sum() + whole_cell_half_b_counts.sum()) != (
        eligible_target_transcripts
    ):
        raise ValueError("HEST split-half totals differ from eligible target transcript receipt")
    nucleus_library_sizes = np.asarray(rows.nucleus_library_sizes, dtype=np.int64)
    whole_cell_library_sizes = np.asarray(rows.whole_cell_library_sizes, dtype=np.int64)
    nucleus_library_size_half_a = np.asarray(rows.nucleus_library_size_half_a, dtype=np.int64)
    nucleus_library_size_half_b = np.asarray(rows.nucleus_library_size_half_b, dtype=np.int64)
    whole_cell_library_size_half_a = np.asarray(rows.whole_cell_library_size_half_a, dtype=np.int64)
    whole_cell_library_size_half_b = np.asarray(rows.whole_cell_library_size_half_b, dtype=np.int64)
    if not np.array_equal(
        nucleus_library_sizes, nucleus_library_size_half_a + nucleus_library_size_half_b
    ) or not np.array_equal(
        whole_cell_library_sizes,
        whole_cell_library_size_half_a + whole_cell_library_size_half_b,
    ):
        raise ValueError("HEST retained split-half library sizes do not reconstruct totals")
    nucleus_centres = np.asarray(rows.centres, dtype=np.float64)
    native_cell_centres = np.asarray(rows.cell_centres, dtype=np.float64)
    annotation_centres = np.asarray(rows.source_centres, dtype=np.float64)
    annotation_he_centres = np.asarray(rows.annotation_he_centres, dtype=np.float64)
    cell_nucleus_distances_um = (
        np.linalg.norm(native_cell_centres - nucleus_centres, axis=1) * SOURCE_MPP
    )
    nucleus_areas_um2 = np.asarray(rows.nucleus_areas_um2, dtype=np.float64)
    cell_areas_um2 = np.asarray(rows.cell_areas_um2, dtype=np.float64)
    area_ratio = np.divide(
        nucleus_areas_um2,
        cell_areas_um2,
        out=np.zeros_like(nucleus_areas_um2),
        where=cell_areas_um2 > 0,
    )
    nuclear_morphometric_feature_names = (
        *("nucleus_%s" % name for name in GEOMETRY_FEATURE_NAMES),
        *("nucleus_%s" % name for name in REGION_TEXTURE_FEATURE_NAMES),
        "nucleus_to_cell_area_ratio",
        "cell_nucleus_centroid_distance_um",
        "nucleus_centroid_inside_cell",
    )
    nuclear_morphometric_features = np.column_stack(
        (
            nucleus_geometry_features,
            nucleus_texture_features,
            area_ratio,
            cell_nucleus_distances_um,
            rows.nucleus_centroid_inside_cell,
        )
    ).astype(np.float32)
    cell_morphometric_feature_names = (
        *("cell_%s" % name for name in GEOMETRY_FEATURE_NAMES),
        *("cell_%s" % name for name in REGION_TEXTURE_FEATURE_NAMES),
        "nucleus_to_cell_area_ratio",
        "cell_nucleus_centroid_distance_um",
    )
    cell_morphometric_features = np.column_stack(
        (
            cell_geometry_features,
            cell_texture_features,
            area_ratio,
            cell_nucleus_distances_um,
        )
    ).astype(np.float32)
    classical_morphology_feature_names = (
        *nuclear_morphometric_feature_names,
        *cell_morphometric_feature_names,
    )
    classical_morphology_features = np.column_stack(
        (nuclear_morphometric_features, cell_morphometric_features)
    ).astype(np.float32)
    if rows.cellvit_names is None:
        composition_features = local_density_features
        composition_feature_names = LOCAL_DENSITY_FEATURE_NAMES
    else:
        composition_features = np.column_stack(
            (local_density_features, np.asarray(rows.cellvit, dtype=np.float32))
        ).astype(np.float32)
        composition_feature_names = (*LOCAL_DENSITY_FEATURE_NAMES, *rows.cellvit_names)
    disease_adjustment_features, disease_adjustment_feature_names = _categorical_one_hot(
        rows.disease_statuses, "disease_status"
    )
    site_adjustment_features, site_adjustment_feature_names = _categorical_one_hot(
        rows.site_ids, "site_id"
    )
    batch_adjustment_features, batch_adjustment_feature_names = _categorical_one_hot(
        rows.batch_ids, "batch_id"
    )
    section_adjustment_features, section_adjustment_feature_names = _categorical_one_hot(
        rows.sample_ids, "section_id"
    )
    technical_covariates = np.asarray(rows.technical, dtype=np.float32)[:, None]
    full_nuisance_covariates = np.column_stack(
        (
            technical_covariates,
            section_adjustment_features,
            disease_adjustment_features,
            site_adjustment_features,
            batch_adjustment_features,
        )
    ).astype(np.float32)
    full_nuisance_covariate_names = (
        "log1p_library_size",
        *section_adjustment_feature_names,
        *disease_adjustment_feature_names,
        *site_adjustment_feature_names,
        *batch_adjustment_feature_names,
    )
    registration_qc_names = (
        "annotation_nCount_RNA",
        "annotation_nFeature_RNA",
        "annotation_percent_negative_or_unassigned",
        "native_cell_centroid_he_x",
        "native_cell_centroid_he_y",
        "native_nucleus_centroid_he_x",
        "native_nucleus_centroid_he_y",
        "annotation_xenium_centroid_x",
        "annotation_xenium_centroid_y",
        "annotation_registered_he_x",
        "annotation_registered_he_y",
        "annotation_nucleus_centroid_distance_um",
        "annotation_cell_centroid_distance_um",
        "native_cell_nucleus_centroid_distance_um",
        "native_nucleus_area_um2",
        "native_cell_area_um2",
        "native_nucleus_to_cell_area_ratio",
        "native_nucleus_centroid_inside_cell",
        "section_affine_residual_p95_um",
        "nucleus_library_size",
        "whole_cell_library_size",
        "nucleus_detected_target_genes",
        "whole_cell_detected_target_genes",
    )
    registration_qc = np.column_stack(
        (
            rows.ncount_rna,
            rows.nfeature_rna,
            rows.percent_negative_or_unassigned,
            native_cell_centres,
            nucleus_centres,
            annotation_centres,
            annotation_he_centres,
            rows.annotation_nucleus_distances_um,
            rows.annotation_cell_distances_um,
            cell_nucleus_distances_um,
            nucleus_areas_um2,
            cell_areas_um2,
            area_ratio,
            rows.nucleus_centroid_inside_cell,
            rows.affine_residual_p95_um,
            rows.nucleus_library_sizes,
            rows.whole_cell_library_sizes,
            rows.nucleus_detected_target_genes,
            rows.whole_cell_detected_target_genes,
        )
    ).astype(np.float32)
    measurement_qc_contract = (
        study_manifest.content["locked_measurement_audit"]
        if study_manifest.study_stage == "confirmatory_morphology"
        else study_manifest.content["decision_thresholds"]
    )
    annotation_nucleus_values = np.asarray(rows.annotation_nucleus_distances_um)
    annotation_cell_values = np.asarray(rows.annotation_cell_distances_um)
    nucleus_diameter_um = 2.0 * np.sqrt(nucleus_areas_um2 / np.pi)
    nearest_neighbor_um = np.asarray(local_density_features[:, 0], dtype=np.float64)
    section_ids_for_qc = np.asarray(rows.sample_ids).astype(str)
    diameter_relative_values = np.full(len(rows.observation_ids), np.inf, dtype=np.float64)
    neighbor_relative_values = np.full(len(rows.observation_ids), np.inf, dtype=np.float64)
    for section_id in sorted(set(section_ids_for_qc.tolist())):
        section = section_ids_for_qc == section_id
        valid_diameter = section & np.isfinite(nucleus_diameter_um) & (nucleus_diameter_um > 0)
        valid_neighbor = section & np.isfinite(nearest_neighbor_um) & (nearest_neighbor_um > 0)
        if valid_diameter.any():
            diameter_relative_values[section] = annotation_nucleus_values[section] / float(
                np.median(nucleus_diameter_um[valid_diameter])
            )
        if valid_neighbor.any():
            neighbor_relative_values[section] = annotation_nucleus_values[section] / float(
                np.median(nearest_neighbor_um[valid_neighbor])
            )
    registration_qc_pass = (
        annotation_nucleus_values
        <= float(measurement_qc_contract["maximum_annotation_nucleus_p95_um"])
    ) & (annotation_cell_values <= float(measurement_qc_contract["maximum_annotation_cell_p95_um"]))
    registration_qc_pass &= cell_nucleus_distances_um <= float(
        measurement_qc_contract["maximum_cell_nucleus_p95_um"]
    )
    registration_qc_pass &= diameter_relative_values <= float(
        measurement_qc_contract["maximum_registration_nucleus_diameter_ratio_p95"]
    )
    registration_qc_pass &= neighbor_relative_values <= float(
        measurement_qc_contract["maximum_registration_nearest_neighbor_ratio_p95"]
    )
    segmentation_qc_pass = np.asarray(rows.nucleus_centroid_inside_cell, dtype=np.bool_) & (
        area_ratio >= float(measurement_qc_contract["minimum_nucleus_cell_area_ratio"])
    )
    segmentation_qc_pass &= area_ratio <= float(
        measurement_qc_contract["maximum_nucleus_cell_area_ratio"]
    )
    target_qc_pass = (
        (np.asarray(rows.nucleus_library_sizes) >= int(protocol["minimum_transcripts_per_cell"]))
        & (np.asarray(rows.whole_cell_library_sizes) >= np.asarray(rows.nucleus_library_sizes))
        & (target_counts.sum(axis=1) <= np.asarray(rows.nucleus_library_sizes))
        & (whole_cell_target_counts.sum(axis=1) <= np.asarray(rows.whole_cell_library_sizes))
    )
    crop_qc_pass = np.all(
        np.isfinite(crop_padding_fractions)
        & (crop_padding_fractions >= 0.0)
        & (crop_padding_fractions <= float(measurement_qc_contract["maximum_crop_padding_p95"])),
        axis=1,
    )
    cellvit_crop_qc_pass = (
        np.ones(observations, dtype=np.bool_)
        if cellvit_nearest_padding_fractions is None
        else cellvit_nearest_padding_fractions
        <= float(measurement_qc_contract["maximum_crop_padding_p95"])
    )
    locked_measurement_qc_pass = registration_qc_pass & segmentation_qc_pass & crop_qc_pass
    if not target_qc_pass.all():
        raise ValueError("HEST target-count invariant failed")
    feature_name_prefix = encoder_manifest.encoder_id
    feature_space_id = "%s_%s_%d_hest_%s_v1" % (
        encoder_manifest.encoder_id,
        encoder_manifest.pooling_rule,
        encoder_manifest.feature_width,
        crop_manifest.primary_crop_id,
    )
    molecular_space_id = "xenium_nucleus_overlapping_log1p_cpm_10000_v1"
    registration_source_sha256 = _canonical_sha256(
        [
            (sample["sample_id"], sample["cell_seg_sha256"], sample["nucleus_seg_sha256"])
            for sample in resolved_provenance
        ]
    )
    exclusion_policy_sha256 = _canonical_sha256(
        {
            key: protocol[key]
            for key in (
                "minimum_transcripts_per_cell",
                "minimum_transcript_qv",
                "excluded_feature_prefixes",
                "type_markers",
                "fine_type_marker_gene_ids",
                "label_target_independence",
                "spatial_block_um",
                "spatial_roi_um",
                "opposite_pool_guard_um",
                "pool_assignment_salt",
                "reference_splits",
            )
        }
    )
    target_source_sha256 = _canonical_sha256(
        [(sample["sample_id"], sample["transcripts_sha256"]) for sample in resolved_provenance]
    )
    source_file_manifest_sha256 = _canonical_sha256(
        {
            "annotation_metadata": annotation_declaration.sha256,
            "annotation_parent": full_annotation_declaration.sha256,
            "label_predictions": (
                None if independent_artifacts is None else independent_artifacts.predictions_sha256
            ),
            "label_receipt": (
                None if independent_artifacts is None else independent_artifacts.receipt_sha256
            ),
            "samples": [
                {key: value for key, value in sample.items() if key.endswith("_sha256")}
                for sample in resolved_provenance
            ],
        }
    )
    segmentation_manifest_sha256 = _canonical_sha256(
        [
            (
                sample["sample_id"],
                sample["cell_seg_sha256"],
                sample["nucleus_seg_sha256"],
                sample["cellvit_seg_sha256"],
            )
            for sample in resolved_provenance
        ]
    )
    target_manifest_sha256 = _canonical_sha256(
        {
            "source_sha256": target_source_sha256,
            "gene_ids": list(target_genes),
            "broad_type_marker_gene_ids": list(marker_genes),
            "fine_type_marker_gene_ids": list(fine_marker_genes),
            "marker_target_overlap": 0,
            "minimum_qv": protocol["minimum_transcript_qv"],
            "excluded_feature_prefixes": list(protocol["excluded_feature_prefixes"]),
            "variants": [
                "nucleus_overlapping_transcripts",
                "whole_cell_assigned_transcripts",
            ],
        }
    )
    program_names = tuple(str(name) for name in protocol["target_programs"])
    gene_index = {gene: index for index, gene in enumerate(target_genes)}
    program_gene_membership = np.zeros((len(program_names), len(target_genes)), dtype=np.bool_)
    for program_index, name in enumerate(program_names):
        for gene in protocol["target_programs"][name]:
            program_gene_membership[program_index, gene_index[str(gene)]] = True
    provenance = {
        "schema": SOURCE_SCHEMA,
        "study_stage": study_manifest.study_stage,
        "study_manifest_sha256": study_manifest.sha256,
        "opening_receipt_sha256": opening_receipt_sha256,
        "source_scope": source_scope,
        "locked_donor_outcomes_materialized": (
            study_manifest.study_stage == "confirmatory_morphology"
        ),
        "protocol_sha256": _sha256_file(protocol_path),
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "encoder_manifest_sha256": encoder_manifest.sha256,
        "crop_manifest_sha256": crop_manifest.sha256,
        "model_repo": encoder_manifest.repository,
        "model_revision": encoder_manifest.revision,
        "model_config_sha256": encoder_manifest.config_sha256,
        "model_checkpoint_sha256": encoder_manifest.checkpoint_sha256,
        "annotation_source": "GSE250346 corrected Seurat metadata export",
        "annotation_sha256": annotation_declaration.sha256,
        "annotation_parent_sha256": full_annotation_declaration.sha256,
        "annotation_rows": annotation_expected_rows,
        "label_source_kind": label_source_kind,
        "label_source_sha256": label_source_sha256,
        "label_receipt_sha256": (
            None if independent_artifacts is None else independent_artifacts.receipt_sha256
        ),
        "label_field_primary": label_field_primary,
        "label_field_secondary": label_field_secondary,
        "label_target_independence": independence_contract,
        "label_target_independence_protocol": dict(protocol_independence),
        "fine_type_marker_exclusion": {
            "gene_ids": list(fine_marker_genes),
            "target_overlap": 0,
            "is_proxy_only": True,
            "establishes_full_target_independence": False,
        },
        "target_transcript_filters": {
            "primary_nucleus": (
                "overlaps_nucleus==1,qv>=20,non-control,COUNT(DISTINCT transcript_id)"
            ),
            "secondary_whole_cell": ("qv>=20,non-control,COUNT(DISTINCT transcript_id)"),
        },
        "duplicate_transcript_ids_allowed": False,
        "crop_metadata": {
            "primary_crop_id": crop_manifest.primary_crop_id,
            "crop_ids": list(crop_ids),
            "source_mpp": crop_manifest.source_mpp,
            "padding": crop_manifest.padding,
            "variants": [
                {
                    "crop_id": variant.crop_id,
                    "role": variant.role,
                    "comparison_family": variant.comparison_family,
                    "diameter_um": variant.diameter_um,
                    "inner_diameter_um": variant.inner_diameter_um,
                    "model_input_pixels": variant.model_input_pixels,
                    "effective_mpp": variant.effective_mpp,
                    "mask_mode": variant.mask_mode,
                    "fill_mode": variant.fill_mode,
                }
                for variant in crop_manifest.variants
            ],
            "blur_sigma_um": crop_manifest.blur_sigma_um,
            "random_mask_salt_sha256": hashlib.sha256(
                crop_manifest.random_mask_salt.encode("utf-8")
            ).hexdigest(),
        },
        "claim_scope": "registered_cell_local_context_112um",
        "nucleus_hypothesis_tested": False,
        "cell_intrinsic_hypothesis_tested": False,
        "authorizes_nucleus_intrinsic_claim": False,
        "native_xenium_registration_only": True,
        "cellvit_target_registration": False,
        "samples": resolved_provenance,
        "exclusion_counts": exclusion_counts,
    }
    payload: Dict[str, object] = {
        "schema_version": np.asarray(SOURCE_SCHEMA),
        "study_stage": np.asarray(study_manifest.study_stage),
        "study_manifest_sha256": np.asarray(study_manifest.sha256),
        "opening_receipt_sha256": np.asarray(opening_receipt_sha256 or ""),
        "source_scope": np.asarray(source_scope),
        "opposite_pool_guard_um": np.asarray(float(protocol["opposite_pool_guard_um"])),
        "locked_donor_outcomes_materialized": np.asarray(
            study_manifest.study_stage == "confirmatory_morphology"
        ),
        "observation_ids": np.asarray(rows.observation_ids),
        "observation_id": np.asarray(rows.observation_ids),
        "cell_id": np.asarray(rows.cell_ids),
        "donor_ids": donors,
        "donor_id": donors,
        "patient_ids": donors,
        "patient_id": donors,
        "sample_ids": np.asarray(rows.sample_ids),
        "section_ids": np.asarray(rows.sample_ids),
        "section_id": np.asarray(rows.sample_ids),
        "source_sample_ids": np.asarray(
            [SECTION_IDENTITIES[sample_id][1] for sample_id in rows.sample_ids]
        ),
        "source_sample_id": np.asarray(
            [SECTION_IDENTITIES[sample_id][1] for sample_id in rows.sample_ids]
        ),
        "split_ids": np.asarray(rows.split_ids),
        "disease_statuses": np.asarray(rows.disease_statuses),
        "disease_state": np.asarray(rows.disease_statuses),
        "site_ids": np.asarray(rows.site_ids),
        "site_id": np.asarray(rows.site_ids),
        "batch_ids": np.asarray(rows.batch_ids),
        "batch_id": np.asarray(rows.batch_ids),
        "block_ids": blocks,
        "block_id": blocks,
        "roi_ids": np.asarray(rows.roi_ids),
        "roi_id": np.asarray(rows.roi_ids),
        "pool_roles": roles,
        "pool_role": roles,
        "reference_split_ids": np.asarray(reference_split_ids),
        "pool_roles_by_split": pool_roles_by_split,
        "type_labels": labels,
        "fine_type_label": labels,
        "type_names": np.asarray(fine_type_names),
        "broad_type_labels": np.asarray(rows.broad_type_labels, dtype=np.int64),
        "broad_type_label": np.asarray(rows.broad_type_labels, dtype=np.int64),
        "broad_type_names": np.asarray(broad_type_names),
        "fine_type_ids": np.asarray(rows.fine_type_ids),
        "fine_type": np.asarray(rows.fine_type_ids),
        "fine_type_names": np.asarray(fine_type_names),
        "frozen_features": features,
        "frozen_feature_names": np.asarray(
            [
                "%s_%04d" % (feature_name_prefix, index)
                for index in range(encoder_manifest.feature_width)
            ]
        ),
        "image_features": image_features,
        "image_features_by_crop_and_encoder": image_features,
        "crop_ids": np.asarray(crop_ids),
        "crop_roles": np.asarray([variant.role for variant in crop_manifest.variants]),
        "crop_comparison_families": np.asarray(
            [variant.comparison_family for variant in crop_manifest.variants]
        ),
        "crop_diameters_um": np.asarray(
            [variant.diameter_um for variant in crop_manifest.variants], dtype=np.float32
        ),
        "crop_inner_diameters_um": np.asarray(
            [variant.inner_diameter_um for variant in crop_manifest.variants],
            dtype=np.float32,
        ),
        "crop_model_input_pixels": np.asarray(
            [variant.model_input_pixels for variant in crop_manifest.variants],
            dtype=np.int64,
        ),
        "crop_effective_mpp": np.asarray(
            [variant.effective_mpp for variant in crop_manifest.variants], dtype=np.float32
        ),
        "crop_mask_modes": np.asarray([variant.mask_mode for variant in crop_manifest.variants]),
        "crop_fill_modes": np.asarray([variant.fill_mode for variant in crop_manifest.variants]),
        "crop_padding_fractions": crop_padding_fractions,
        "crop_mask_fractions": crop_mask_fractions,
        "primary_crop_id": np.asarray(crop_manifest.primary_crop_id),
        "molecular_targets": target_values,
        "nucleus_molecular_targets": target_values,
        "whole_cell_molecular_targets": whole_cell_target_values,
        "nucleus_target_counts": target_counts,
        "whole_cell_target_counts": whole_cell_target_counts,
        "nucleus_target_counts_half_a": nucleus_half_a_counts,
        "nucleus_target_counts_half_b": nucleus_half_b_counts,
        "whole_cell_target_counts_half_a": whole_cell_half_a_counts,
        "whole_cell_target_counts_half_b": whole_cell_half_b_counts,
        "normalized_nucleus_targets": target_values,
        "normalized_whole_cell_targets": whole_cell_target_values,
        "nucleus_library_sizes": nucleus_library_sizes,
        "whole_cell_library_sizes": whole_cell_library_sizes,
        "nucleus_library_size_half_a": nucleus_library_size_half_a,
        "nucleus_library_size_half_b": nucleus_library_size_half_b,
        "whole_cell_library_size_half_a": whole_cell_library_size_half_a,
        "whole_cell_library_size_half_b": whole_cell_library_size_half_b,
        "nucleus_detected_target_genes": np.asarray(
            rows.nucleus_detected_target_genes, dtype=np.int64
        ),
        "whole_cell_detected_target_genes": np.asarray(
            rows.whole_cell_detected_target_genes, dtype=np.int64
        ),
        "gene_ids": np.asarray(target_genes),
        "ordered_gene_ids": np.asarray(target_genes),
        "type_marker_gene_ids": np.asarray(marker_genes),
        "broad_type_marker_gene_ids": np.asarray(marker_genes),
        "fine_type_marker_gene_ids": np.asarray(fine_marker_genes),
        "fine_type_marker_panel_sha256": np.asarray(_canonical_sha256(list(fine_marker_genes))),
        "label_target_independence_json": np.asarray(
            json.dumps(independence_contract, sort_keys=True, separators=(",", ":"))
        ),
        "label_target_independence_sha256": np.asarray(_canonical_sha256(independence_contract)),
        "annotation_feature_ids": np.asarray(
            independence_contract["ordered_annotation_feature_ids"], dtype=str
        ),
        "program_names": np.asarray(program_names),
        "program_gene_membership": program_gene_membership,
        "coordinate_features": coordinate_features,
        "coordinate_feature_names": np.asarray(SPATIAL_FEATURE_NAMES),
        "spatial_features": coordinate_features,
        "spatial_feature_names": np.asarray(SPATIAL_FEATURE_NAMES),
        "local_density_features": local_density_features,
        "local_density_feature_names": np.asarray(LOCAL_DENSITY_FEATURE_NAMES),
        "boundary_features": boundary_features,
        "boundary_feature_names": np.asarray(
            ["center_to_local_background_um", "central_background_fraction"]
        ),
        "technical_covariates": technical_covariates,
        "technical_covariate_names": np.asarray(["log1p_library_size"]),
        "full_nuisance_covariates": full_nuisance_covariates,
        "full_nuisance_covariate_names": np.asarray(full_nuisance_covariate_names),
        "disease_adjustment_features": disease_adjustment_features,
        "disease_adjustment_feature_names": np.asarray(disease_adjustment_feature_names),
        "site_adjustment_features": site_adjustment_features,
        "site_adjustment_feature_names": np.asarray(site_adjustment_feature_names),
        "batch_adjustment_features": batch_adjustment_features,
        "batch_adjustment_feature_names": np.asarray(batch_adjustment_feature_names),
        "section_adjustment_features": section_adjustment_features,
        "section_adjustment_feature_names": np.asarray(section_adjustment_feature_names),
        "disease_estimands": np.asarray(protocol["disease_estimands"]),
        "x_coordinate_um": (nucleus_centres[:, 0] * SOURCE_MPP).astype(np.float32),
        "y_coordinate_um": (nucleus_centres[:, 1] * SOURCE_MPP).astype(np.float32),
        "cell_centroid_x_um": (native_cell_centres[:, 0] * SOURCE_MPP).astype(np.float32),
        "cell_centroid_y_um": (native_cell_centres[:, 1] * SOURCE_MPP).astype(np.float32),
        "nucleus_centroid_x_um": (nucleus_centres[:, 0] * SOURCE_MPP).astype(np.float32),
        "nucleus_centroid_y_um": (nucleus_centres[:, 1] * SOURCE_MPP).astype(np.float32),
        "annotation_centroid_x_um": annotation_centres[:, 0].astype(np.float32),
        "annotation_centroid_y_um": annotation_centres[:, 1].astype(np.float32),
        "registration_distance_um": np.asarray(
            rows.annotation_nucleus_distances_um, dtype=np.float32
        ),
        "annotation_cell_distance_um": np.asarray(
            rows.annotation_cell_distances_um, dtype=np.float32
        ),
        "cell_nucleus_centroid_distance_um": cell_nucleus_distances_um.astype(np.float32),
        "nucleus_centroid_inside_cell": np.asarray(
            rows.nucleus_centroid_inside_cell, dtype=np.bool_
        ),
        "cell_area_um2": cell_areas_um2.astype(np.float32),
        "nucleus_area_um2": nucleus_areas_um2.astype(np.float32),
        "library_size": np.asarray(rows.nucleus_library_sizes, dtype=np.int64),
        "detected_target_genes": np.asarray(rows.nucleus_detected_target_genes, dtype=np.int64),
        "transcript_qv_summary": np.stack(rows.transcript_qv_summaries).astype(np.float32),
        "transcript_qv_summary_names": np.asarray(["minimum_qv", "median_qv", "mean_qv"]),
        "stain_features": stain_features,
        "stain_feature_names": np.asarray(stain_feature_names),
        "composition_features": composition_features,
        "composition_feature_names": np.asarray(composition_feature_names),
        "nuclear_morphometric_features": nuclear_morphometric_features,
        "nuclear_morphometric_feature_names": np.asarray(nuclear_morphometric_feature_names),
        "cell_morphometric_features": cell_morphometric_features,
        "cell_morphometric_feature_names": np.asarray(cell_morphometric_feature_names),
        "classical_morphology_features": classical_morphology_features,
        "classical_morphology_feature_names": np.asarray(classical_morphology_feature_names),
        "registration_qc_features": registration_qc,
        "registration_qc_feature_names": np.asarray(registration_qc_names),
        "registration_qc_pass": registration_qc_pass,
        "segmentation_qc_pass": segmentation_qc_pass,
        "locked_measurement_qc_pass": locked_measurement_qc_pass,
        "locked_measurement_audit_thresholds_json": np.asarray(
            json.dumps(measurement_qc_contract, sort_keys=True)
        ),
        "locked_measurement_audit_thresholds_sha256": np.asarray(
            _canonical_sha256(measurement_qc_contract)
        ),
        "registration_cardinality": np.ones(observations, dtype=np.int8),
        "target_qc_pass": target_qc_pass,
        "crop_qc_pass": crop_qc_pass,
        "cellvit_crop_qc_pass": cellvit_crop_qc_pass,
        "native_nucleus_centres_he_pixels": nucleus_centres.astype(np.float32),
        "native_cell_centres_he_pixels": native_cell_centres.astype(np.float32),
        "annotation_registered_centres_he_pixels": annotation_he_centres.astype(np.float32),
        "native_nucleus_areas_um2": nucleus_areas_um2.astype(np.float32),
        "native_cell_areas_um2": cell_areas_um2.astype(np.float32),
        "native_nucleus_centroid_inside_cell": np.asarray(
            rows.nucleus_centroid_inside_cell, dtype=np.bool_
        ),
        "feature_space_id": np.asarray(feature_space_id),
        "molecular_space_id": np.asarray(molecular_space_id),
        "feature_checkpoint_sha256": np.asarray(encoder_manifest.checkpoint_sha256),
        "feature_config_sha256": np.asarray(encoder_manifest.config_sha256),
        "encoder_revision": np.asarray(encoder_manifest.revision),
        "encoder_manifest_sha256": np.asarray(encoder_manifest.sha256),
        "crop_manifest_sha256": np.asarray(crop_manifest.sha256),
        "registration_method": np.asarray(protocol["registration_method"]),
        "encoder_name": np.asarray(encoder_manifest.repository),
        "crop_scale": np.asarray("registered_cell_local_context_112um"),
        "crop_role": np.asarray(primary_variant.role),
        "crop_diameter_um": np.asarray(primary_variant.diameter_um),
        "source_mpp": np.asarray(crop_manifest.source_mpp),
        "model_mpp": np.asarray(encoder_manifest.model_mpp),
        "model_input_pixels": np.asarray(encoder_manifest.input_pixels),
        "mask_mode": np.asarray(primary_variant.mask_mode),
        "authorizes_nucleus_intrinsic_claim": np.asarray(False),
        "nucleus_hypothesis_tested": np.asarray(False),
        "cell_intrinsic_hypothesis_tested": np.asarray(False),
        "g2_claim_scope": np.asarray("registered_cell_local_context_112um"),
        "cohort_id": np.asarray("HEST"),
        "cohort_release": np.asarray(DATASET_REVISION),
        "assay": np.asarray(protocol["assay"]),
        "observation_level": np.asarray(protocol["observation_level"]),
        "target_construction": np.asarray(protocol["target_construction"]),
        "secondary_target_construction": np.asarray("whole_cell_xenium_transcripts"),
        "label_source_sha256": np.asarray(label_source_sha256),
        "label_source_kind": np.asarray(label_source_kind),
        "annotation_receipt_sha256": np.asarray(
            "" if independent_artifacts is None else independent_artifacts.receipt_sha256
        ),
        "annotation_prediction_export_sha256": np.asarray(
            "" if independent_artifacts is None else independent_artifacts.predictions_sha256
        ),
        "source_file_manifest_sha256": np.asarray(source_file_manifest_sha256),
        "registration_source_sha256": np.asarray(registration_source_sha256),
        "registration_manifest_sha256": np.asarray(registration_source_sha256),
        "segmentation_manifest_sha256": np.asarray(segmentation_manifest_sha256),
        "exclusion_policy_sha256": np.asarray(exclusion_policy_sha256),
        "target_source_sha256": np.asarray(target_source_sha256),
        "target_manifest_sha256": np.asarray(target_manifest_sha256),
        "planned_stratum_ids": np.asarray(planned_stratum_ids),
        "planned_stratum_manifest_sha256": np.asarray(planned_stratum_manifest_sha256),
        "transcript_split_method": np.asarray("sha256-final-byte-lsb-v1"),
        "transcript_minimum_qv": np.asarray(
            float(protocol["minimum_transcript_qv"]), dtype=np.float32
        ),
        "transcript_split_salt_sha256": np.asarray(
            hashlib.sha256(str(protocol["transcript_split_salt"]).encode("utf-8")).hexdigest()
        ),
        "transcript_identity_manifest_sha256": np.asarray(transcript_identity_hasher.hexdigest()),
        "eligible_target_transcripts": np.asarray(eligible_target_transcripts, dtype=np.int64),
        "duplicate_transcript_ids": np.asarray(0, dtype=np.int64),
        "transcripts_assigned_to_multiple_cells": np.asarray(0, dtype=np.int64),
        "invalid_qv_transcripts": np.asarray(0, dtype=np.int64),
        "unknown_gene_transcripts": np.asarray(0, dtype=np.int64),
        "unknown_cell_transcripts": np.asarray(0, dtype=np.int64),
        "provenance_json": np.asarray(json.dumps(provenance, sort_keys=True)),
    }
    if rows.cellvit_names is not None:
        if len(rows.cellvit) != observations:
            raise ValueError("CellViT sensitivity rows differ from registered native cells")
        payload["cellvit_sensitivity_features"] = np.asarray(rows.cellvit, dtype=np.float32)
        payload["cellvit_sensitivity_feature_names"] = np.asarray(rows.cellvit_names)
        payload["cellvit_context_features"] = np.asarray(rows.cellvit, dtype=np.float32)
        payload["cellvit_context_feature_names"] = np.asarray(rows.cellvit_names)
        if cellvit_nearest_features is None or cellvit_nearest_padding_fractions is None:
            raise ValueError("CellViT sensitivity crop features were not constructed")
        payload["cellvit_nearest_frozen_features"] = cellvit_nearest_features
        payload["cellvit_nearest_distance_um"] = np.asarray(
            rows.cellvit_distances_um, dtype=np.float32
        )
        payload["cellvit_nearest_crop_padding_fraction"] = cellvit_nearest_padding_fractions
        payload["cellvit_nearest_crop_id"] = np.asarray(crop_manifest.primary_crop_id)
    else:
        payload["cellvit_context_features"] = np.empty((observations, 0), dtype=np.float32)
        payload["cellvit_context_feature_names"] = np.asarray([], dtype=str)
    _write_npz(output_path, payload)
    source_sha256 = _sha256_file(output_path)
    donor_sections = {
        donor: sorted(
            {
                sample_id
                for sample_id, donor_id in zip(rows.sample_ids, rows.donor_ids)
                if donor_id == donor
            }
        )
        for donor in active_donors
    }
    observation_manifest_sha256 = _canonical_sha256(
        [
            (observation_id, block_id, pool_role)
            for observation_id, block_id, pool_role in zip(
                rows.observation_ids, rows.block_ids, rows.pool_roles
            )
        ]
    )
    plan = {
        "schema": "heir.morphology_ridge_preparation_plan.v1",
        "study_stage": study_manifest.study_stage,
        "study_manifest_sha256": study_manifest.sha256,
        "opening_receipt_sha256": opening_receipt_sha256,
        "source_scope": source_scope,
        "locked_donor_outcomes_materialized": (
            study_manifest.study_stage == "confirmatory_morphology"
        ),
        "source_schema": SOURCE_SCHEMA,
        "source_schema_sha256": _canonical_sha256(SOURCE_SCHEMA),
        "source_observations_sha256": source_sha256,
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
        },
        "encoder_manifest": {
            "path": str(encoder_manifest_path),
            "sha256": encoder_manifest.sha256,
        },
        "crop_manifest": {
            "path": str(crop_manifest_path),
            "sha256": crop_manifest.sha256,
        },
        "experiment_role": "primary_hest_uni2h",
        "scientific_scope": "registered_cell_local_context_112um_association",
        "g2_claim_scope": "registered_cell_local_context_112um",
        "nucleus_hypothesis_tested": False,
        "cell_intrinsic_hypothesis_tested": False,
        "authorizes_nucleus_intrinsic_claim": False,
        "development_donors": list(study_manifest.development_donors),
        "locked_test_donors": list(study_manifest.locked_test_donors),
        "materialized_donors": list(active_donors),
        "donor_sections": donor_sections,
        "gene_ids": list(target_genes),
        "type_names": list(fine_type_names),
        "broad_type_names": list(broad_type_names),
        "type_marker_gene_ids": list(marker_genes),
        "broad_type_marker_gene_ids": list(marker_genes),
        "fine_type_marker_gene_ids": list(fine_marker_genes),
        "fine_type_marker_panel_sha256": _canonical_sha256(list(fine_marker_genes)),
        "label_target_independence": independence_contract,
        "technical_covariate_names": ["log1p_library_size"],
        "full_nuisance_covariate_names": list(full_nuisance_covariate_names),
        "frozen_feature_names": [
            "%s_%04d" % (feature_name_prefix, index)
            for index in range(encoder_manifest.feature_width)
        ],
        "crop_ids": list(crop_ids),
        "coordinate_feature_names": list(SPATIAL_FEATURE_NAMES),
        "spatial_feature_names": list(SPATIAL_FEATURE_NAMES),
        "local_density_feature_names": list(LOCAL_DENSITY_FEATURE_NAMES),
        "stain_feature_names": list(stain_feature_names),
        "composition_feature_names": list(composition_feature_names),
        "nuclear_morphometric_feature_names": list(nuclear_morphometric_feature_names),
        "cell_morphometric_feature_names": list(cell_morphometric_feature_names),
        "feature_space_id": feature_space_id,
        "feature_checkpoint_sha256": encoder_manifest.checkpoint_sha256,
        "encoder_manifest_sha256": encoder_manifest.sha256,
        "crop_manifest_sha256": crop_manifest.sha256,
        "molecular_space_id": molecular_space_id,
        "label_source_sha256": label_source_sha256,
        "label_source_kind": label_source_kind,
        "annotation_receipt_sha256": (
            None if independent_artifacts is None else independent_artifacts.receipt_sha256
        ),
        "annotation_prediction_export_sha256": (
            None if independent_artifacts is None else independent_artifacts.predictions_sha256
        ),
        "registration_source_sha256": registration_source_sha256,
        "exclusion_policy_sha256": exclusion_policy_sha256,
        "registration_method": str(protocol["registration_method"]),
        "encoder_name": encoder_manifest.repository,
        "crop_scale": "registered_cell_local_context_112um",
        "crop_metadata": {
            "primary_crop_id": crop_manifest.primary_crop_id,
            "crop_role": primary_variant.role,
            "crop_diameter_um": primary_variant.diameter_um,
            "source_mpp": crop_manifest.source_mpp,
            "model_mpp": encoder_manifest.model_mpp,
            "model_input_pixels": encoder_manifest.input_pixels,
            "mask_mode": primary_variant.mask_mode,
            "fill_mode": primary_variant.fill_mode,
            "padding": crop_manifest.padding,
            "variants": [
                {
                    "crop_id": variant.crop_id,
                    "role": variant.role,
                    "comparison_family": variant.comparison_family,
                    "diameter_um": variant.diameter_um,
                    "inner_diameter_um": variant.inner_diameter_um,
                    "model_input_pixels": variant.model_input_pixels,
                    "effective_mpp": variant.effective_mpp,
                    "mask_mode": variant.mask_mode,
                    "fill_mode": variant.fill_mode,
                }
                for variant in crop_manifest.variants
            ],
        },
        "cohort_id": "HEST",
        "cohort_release": DATASET_REVISION,
        "assay": str(protocol["assay"]),
        "observation_level": str(protocol["observation_level"]),
        "target_construction": str(protocol["target_construction"]),
        "secondary_target_construction": "whole_cell_xenium_transcripts",
        "target_source_sha256": target_source_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "target_gene_panel_sha256": _canonical_sha256(list(target_genes)),
        "program_names": list(program_names),
        "program_gene_membership": program_gene_membership.astype(int).tolist(),
        "transcript_split_method": "sha256-final-byte-lsb-v1",
        "transcript_split_salt_sha256": hashlib.sha256(
            str(protocol["transcript_split_salt"]).encode("utf-8")
        ).hexdigest(),
        "transcript_identity_manifest_sha256": transcript_identity_hasher.hexdigest(),
        "planned_stratum_ids": list(planned_stratum_ids),
        "planned_stratum_manifest_sha256": planned_stratum_manifest_sha256,
        "label_hierarchy": {
            "primary": label_field_primary,
            "secondary": label_field_secondary,
            "fine_type_names": list(fine_type_names),
            "broad_type_names": list(broad_type_names),
        },
        "source": {
            "schema": SOURCE_SCHEMA,
            "schema_sha256": _canonical_sha256(SOURCE_SCHEMA),
            "observations_sha256": source_sha256,
            "cohort_id": "HEST",
            "cohort_release": DATASET_REVISION,
            "assay": str(protocol["assay"]),
            "observation_level": str(protocol["observation_level"]),
            "donor_sections": donor_sections,
        },
        "partitions": {
            "development_donors": list(study_manifest.development_donors),
            "locked_test_donors": list(study_manifest.locked_test_donors),
            "materialized_donors": list(active_donors),
        },
        "encoder": {
            "repository": encoder_manifest.repository,
            "revision": encoder_manifest.revision,
            "checkpoint_sha256": encoder_manifest.checkpoint_sha256,
            "manifest_sha256": encoder_manifest.sha256,
            "feature_width": encoder_manifest.feature_width,
        },
        "preprocessing": {
            "implementation": "native_xenium_registered_cell_factorial_crop_ladder",
            "implementation_sha256": crop_manifest.sha256,
            "crop_role": primary_variant.role,
            "crop_diameter_um": primary_variant.diameter_um,
            "source_mpp": crop_manifest.source_mpp,
            "model_mpp": encoder_manifest.model_mpp,
            "model_input_pixels": encoder_manifest.input_pixels,
            "mask_mode": primary_variant.mask_mode,
            "primary_crop_id": crop_manifest.primary_crop_id,
            "crop_ids": list(crop_ids),
            "comparison_families": [
                variant.comparison_family for variant in crop_manifest.variants
            ],
            "effective_mpp": [variant.effective_mpp for variant in crop_manifest.variants],
        },
        "target": {
            "primary": "nucleus_overlapping_xenium_transcripts",
            "secondary": "whole_cell_xenium_transcripts",
            "normalization": "log1p_cpm_10000",
            "gene_ids": list(target_genes),
            "gene_ids_sha256": _canonical_sha256(list(target_genes)),
            "program_names": list(program_names),
            "program_gene_membership": program_gene_membership.astype(int).tolist(),
            "split_method": "sha256-final-byte-lsb-v1",
            "split_salt_sha256": hashlib.sha256(
                str(protocol["transcript_split_salt"]).encode("utf-8")
            ).hexdigest(),
            "transcript_identity_manifest_sha256": (transcript_identity_hasher.hexdigest()),
        },
        "labels": {
            "primary": label_field_primary,
            "secondary": label_field_secondary,
            "fine_type_names": list(fine_type_names),
            "broad_type_names": list(broad_type_names),
            "source_sha256": label_source_sha256,
            "source_kind": label_source_kind,
            "receipt_sha256": (
                None if independent_artifacts is None else independent_artifacts.receipt_sha256
            ),
            "fine_type_marker_gene_ids": list(fine_marker_genes),
            "label_target_independence": independence_contract,
        },
        "reference_mode": "simulated_spatially_disjoint_unpaired_rna",
        "reference_pool": {
            "construction": "same_donor_fine_type_spatial_block_pool",
            "spatially_disjoint": True,
            "minimum_per_donor_type": minimum_reference,
            "observation_manifest_sha256": observation_manifest_sha256,
        },
        "reference_splits": {
            "primary_split_id": reference_split_ids[0],
            "split_ids": list(reference_split_ids),
            "primary_evaluation_rows_fixed": True,
            "selection_unit": "spatial_block",
        },
        "nuisance_covariates": [
            "log1p_library_size",
            "section_id",
            "disease_status",
            "site_id",
            "batch_id",
        ],
        "disease_estimands": list(protocol["disease_estimands"]),
        "registration_qc_feature_names": list(registration_qc_names),
        "gate": {
            "ranks": [2, 4, 6],
            "ridge_penalties": [0.1, 1.0, 10.0, 100.0],
            "permutation_seeds": [17, 29, 41],
            "permutations_per_seed": 100,
            "minimum_support": minimum_evaluation,
            "minimum_development_donors": 5,
            "minimum_locked_donors": (
                5 if study_manifest.study_stage == "confirmatory_morphology" else 0
            ),
            "minimum_coverage_fraction": 0.7,
            "minimum_shuffled_fraction": 0.7,
            "nulls": ["within_roi_derangement", "spatial_block_reassignment"],
            "thresholds": {
                "minimum_macro_residual_coordinate_r2": 0.05,
                "minimum_shuffled_delta_r2": 0.03,
                "maximum_empirical_p": 0.01,
                "minimum_molecular_error_reduction": 0.05,
            },
        },
    }
    _write_json(plan_output_path, plan)
    padding_summary = {
        crop_id: {
            "p50": float(np.quantile(crop_padding_fractions[:, index], 0.5)),
            "p95": float(np.quantile(crop_padding_fractions[:, index], 0.95)),
            "maximum": float(crop_padding_fractions[:, index].max()),
            "masked_fraction_p50": float(np.quantile(crop_mask_fractions[:, index], 0.5)),
        }
        for index, crop_id in enumerate(crop_ids)
    }
    registration_distances = np.asarray(rows.annotation_nucleus_distances_um, dtype=np.float64)
    reference_evaluation_balance = _reference_evaluation_balance(
        donors,
        roles,
        {
            "library_size": technical_covariates,
            "spatial": coordinate_features,
            "stain": stain_features,
            "nuclear_morphology": nuclear_morphometric_features,
            "cell_morphology": cell_morphometric_features,
            "local_density": local_density_features,
        },
        {
            "fine_type": rows.fine_type_ids,
            "section": rows.sample_ids,
            "disease": rows.disease_statuses,
            "site": rows.site_ids,
            "batch": rows.batch_ids,
        },
    )
    registration_gate_pass = all(
        float(sample["coordinate_registration_residual_p95_um"])
        <= float(protocol["maximum_affine_registration_residual_p95_um"])
        and float(sample["annotation_nucleus_distance_p95_um"])
        <= float(protocol["maximum_annotation_nucleus_distance_p95_um"])
        and float(sample["registration_outlier_fraction"])
        <= float(protocol["maximum_registration_outlier_fraction"])
        for sample in resolved_provenance
    )
    qc = {
        "schema": "heir.hest_xenium_cell_qc.v1",
        "study_stage": study_manifest.study_stage,
        "study_manifest_sha256": study_manifest.sha256,
        "opening_receipt_sha256": opening_receipt_sha256,
        "source_scope": source_scope,
        "locked_donor_outcomes_materialized": (
            study_manifest.study_stage == "confirmatory_morphology"
        ),
        "source_observations": {
            "path": str(output_path),
            "sha256": source_sha256,
            "observations": observations,
        },
        "preparation_plan": {
            "path": str(plan_output_path),
            "sha256": _sha256_file(plan_output_path),
        },
        "protocol_sha256": _sha256_file(protocol_path),
        "encoder_manifest_sha256": encoder_manifest.sha256,
        "crop_manifest_sha256": crop_manifest.sha256,
        "annotation_sha256": annotation_declaration.sha256,
        "registration": {
            "method": str(protocol["registration_method"]),
            "one_to_one": True,
            "duplicate_observation_ids": 0,
            "annotation_nucleus_distance_p50_um": float(np.quantile(registration_distances, 0.5)),
            "median_annotation_nucleus_distance_um": float(
                np.quantile(registration_distances, 0.5)
            ),
            "annotation_nucleus_distance_p95_um": float(np.quantile(registration_distances, 0.95)),
            "p95_annotation_nucleus_distance_um": float(np.quantile(registration_distances, 0.95)),
            "maximum_allowed_p95_um": float(protocol["maximum_annotation_nucleus_distance_p95_um"]),
            "annotation_nucleus_distance_max_um": float(registration_distances.max()),
            "nucleus_centroid_outside_cell_fraction": float(
                1.0 - np.mean(rows.nucleus_centroid_inside_cell)
            ),
            "row_within_distance_threshold_fraction": float(np.mean(registration_qc_pass)),
            "pass": registration_gate_pass,
            "samples": resolved_provenance,
        },
        "targets": {
            "primary": "nucleus_overlapping_xenium_transcripts",
            "secondary": "whole_cell_xenium_transcripts",
            "duplicate_eligible_transcript_ids": 0,
            "unknown_noncontrol_features": 0,
            "invalid_qv_rows": 0,
            "unsegmented_high_qv_assignments": 0,
            "nucleus_eligible_transcripts": int(
                sum(int(value["nucleus_eligible_transcripts"]) for value in resolved_provenance)
            ),
            "whole_cell_eligible_transcripts": int(
                sum(int(value["whole_cell_eligible_transcripts"]) for value in resolved_provenance)
            ),
            "eligible_target_transcripts": eligible_target_transcripts,
            "transcript_identity_manifest_sha256": (transcript_identity_hasher.hexdigest()),
            "all_row_qc_pass": bool(target_qc_pass.all()),
        },
        "crops": {
            "primary_crop_id": crop_manifest.primary_crop_id,
            "g2_claim_scope": "registered_cell_local_context_112um",
            "nucleus_hypothesis_tested": False,
            "cell_intrinsic_hypothesis_tested": False,
            "common_canvas": {
                "diameter_um": 112.0,
                "effective_mpp": 0.5,
                "model_input_pixels": 224,
            },
            "resolution_sensitivity_crop_ids": ["crop_32um", "crop_64um"],
            "mask_control_fill_modes": ["white", "mean_color", "blurred"],
            "random_location_shape_matched_controls": True,
            "inpainting_control_available": False,
            "inpainting_substitute": "blurred_replacement",
            "maximum_allowed_padding_fraction": float(protocol["maximum_crop_padding_fraction"]),
            "all_row_qc_pass": bool(crop_qc_pass.all()),
            "by_crop_id": padding_summary,
        },
        "feature_families": {
            "stain_quality_columns": len(stain_feature_names),
            "nuclear_morphology_columns": len(nuclear_morphometric_feature_names),
            "cell_morphology_columns": len(cell_morphometric_feature_names),
            "spatial_columns": len(SPATIAL_FEATURE_NAMES),
            "composition_columns": len(composition_feature_names),
            "full_nuisance_columns": len(full_nuisance_covariate_names),
            "disease_estimands": ["disease_inclusive", "disease_adjusted"],
        },
        "fine_type_marker_exclusion": {
            "gene_ids": list(fine_marker_genes),
            "target_overlap": 0,
            "label_target_independence": independence_contract,
        },
        "reference_evaluation_balance": reference_evaluation_balance,
        "reference_splits": {
            split_id: {
                "reference_rows": int(
                    np.count_nonzero(pool_roles_by_split[:, index] == "reference")
                ),
                "evaluation_rows": int(
                    np.count_nonzero(pool_roles_by_split[:, index] == "evaluation")
                ),
                "excluded_rows": int(np.count_nonzero(pool_roles_by_split[:, index] == "excluded")),
                "membership_sha256": _canonical_sha256(
                    [
                        (observation_id, role)
                        for observation_id, role in zip(
                            rows.observation_ids, pool_roles_by_split[:, index].tolist()
                        )
                    ]
                ),
            }
            for index, split_id in enumerate(reference_split_ids)
        },
        "exclusion_counts": exclusion_counts,
        "planned_strata": {
            "ids": list(planned_stratum_ids),
            "manifest_sha256": planned_stratum_manifest_sha256,
        },
        "pass": bool(registration_gate_pass and target_qc_pass.all() and crop_qc_pass.all()),
    }
    _write_json(qc_output_path, qc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--encoder-manifest", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--annotation-receipt", type=Path, default=None)
    parser.add_argument("--annotation-predictions", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    build_source(
        args.protocol,
        args.study_manifest,
        args.encoder_manifest,
        args.crop_manifest,
        args.data_root,
        args.model_dir,
        args.source_output,
        args.plan_output,
        args.qc_output,
        annotation_receipt_path=args.annotation_receipt,
        annotation_predictions_path=args.annotation_predictions,
        device=args.device,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
