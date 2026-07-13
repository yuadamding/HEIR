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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from heir.features import EncoderManifest, FrozenPatchEncoder, create_frozen_encoder
from heir.features import load_encoder_manifest as _load_encoder_manifest

PROTOCOL_SCHEMA = "heir.hest_xenium_cell_protocol.v3"
SOURCE_SCHEMA = "heir.registered_observations.v3"
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
    diameter_um: float
    mask_mode: str
    inner_diameter_um: float


@dataclass(frozen=True)
class CropManifest:
    path: Path
    sha256: str
    source_mpp: float
    padding: str
    primary_crop_id: str
    variants: Tuple[CropVariant, ...]


def _load_crop_manifest(path: Path) -> CropManifest:
    resolved = path.expanduser().resolve()
    value = _read_json(resolved)
    required = {"schema", "source_mpp", "padding", "primary_crop_id", "variants"}
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
            diameter_um=float(raw.get("diameter_um", 0.0)),
            mask_mode=str(raw.get("mask_mode", "")),
            inner_diameter_um=float(raw.get("inner_diameter_um", 0.0)),
        )
        if (
            not variant.crop_id
            or not variant.role
            or variant.diameter_um <= 0
            or variant.mask_mode
            not in {"none", "nucleus", "cell", "context_ring", "target_removed", "blank"}
            or variant.inner_diameter_um < 0
            or variant.inner_diameter_um >= variant.diameter_um
        ):
            raise ValueError("crop manifest variant geometry is invalid")
        variants.append(variant)
    expected = {
        "nucleus_mask_only",
        "cell_mask_only",
        "crop_32um",
        "crop_64um",
        "crop_112um",
        "context_ring_32_to_112um",
        "context_ring_64_to_112um",
        "target_cell_removed_112um",
        "blank_patch",
    }
    crop_ids = tuple(variant.crop_id for variant in variants)
    if set(crop_ids) != expected or len(crop_ids) != len(expected):
        raise ValueError("crop manifest must contain the frozen physical crop/mask ladder")
    if str(value["primary_crop_id"]) != "crop_112um":
        raise ValueError("unmasked crop_112um must remain the context-association primary")
    if float(value["source_mpp"]) != SOURCE_MPP or value["padding"] != "white":
        raise ValueError("crop manifest source MPP or padding differs from HEST")
    return CropManifest(
        path=resolved,
        sha256=_sha256_file(resolved),
        source_mpp=float(value["source_mpp"]),
        padding=str(value["padding"]),
        primary_crop_id=str(value["primary_crop_id"]),
        variants=tuple(variants),
    )


@dataclass(frozen=True)
class InputFile:
    relative_path: str
    sha256: str


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


def _parse_file(value: object, label: str) -> InputFile:
    if not isinstance(value, Mapping):
        raise ValueError("HEST %s file declaration is missing" % label)
    relative = str(value.get("path", ""))
    digest = value.get("sha256")
    candidate = Path(relative)
    if (
        not relative
        or candidate.is_absolute()
        or ".." in candidate.parts
        or not _is_sha256(digest)
    ):
        raise ValueError("HEST %s file identity is unsafe or unpinned" % label)
    return InputFile(relative, str(digest))


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
        "scientific_scope": "nucleus_centered_local_context_association",
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
    if protocol.get("encoder_manifest_sha256") != encoder_manifest.sha256:
        raise ValueError("HEST protocol differs from the supplied encoder manifest")
    if protocol.get("crop_manifest_sha256") != crop_manifest.sha256:
        raise ValueError("HEST protocol differs from the supplied crop manifest")
    if (
        not _is_sha256(protocol.get("study_manifest_sha256"))
        or protocol.get("study_manifest_sha256") == "0" * 64
    ):
        raise ValueError("HEST protocol must bind a frozen study manifest SHA-256")
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
    gene_ids = tuple(str(value) for value in protocol.get("gene_ids", ()))
    if (
        len(flattened) != len(set(flattened))
        or not gene_ids
        or len(gene_ids) != len(set(gene_ids))
        or set(flattened) & set(gene_ids)
        or any(
            gene.startswith(prefix)
            for gene in flattened + list(gene_ids)
            for prefix in CONTROL_PREFIXES
        )
    ):
        raise ValueError("HEST marker and evaluation genes must be unique and disjoint")
    for name in (
        "minimum_transcripts_per_cell",
        "minimum_transcript_qv",
        "minimum_reference_cells_per_donor_type",
        "minimum_evaluation_cells_per_donor_type",
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
        int(protocol["minimum_reference_cells_per_donor_type"]) < 1
        or int(protocol["minimum_evaluation_cells_per_donor_type"]) < 1
    ):
        raise ValueError("HEST donor/type source support minima must be positive")
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


def _read_annotations(path: Path) -> Dict[str, Dict[str, AnnotationCell]]:
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
        "final_CT",
        "final_lineage",
        "x_centroid",
        "y_centroid",
        "nCount_RNA",
        "nFeature_RNA",
        "perc_negcontrolorunassigned",
    }
    result: Dict[str, Dict[str, AnnotationCell]] = {
        sample_id: {} for sample_id in SECTION_IDENTITIES
    }
    rows = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not required <= set(reader.fieldnames or ()):
                raise ValueError("HEST GSE250346 annotation schema is incomplete")
            for row in reader:
                sample_id = row["hest_id"]
                if sample_id not in SECTION_IDENTITIES:
                    raise ValueError("HEST annotation contains an unknown section")
                expected_donor, expected_source_sample = SECTION_IDENTITIES[sample_id]
                if row["patient"] != expected_donor or row["sample"] != expected_source_sample:
                    raise ValueError("HEST annotation differs from corrected donor identities")
                lineage = row["final_lineage"]
                fine_type = row["final_CT"].strip()
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
                rows += 1
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError("HEST GSE250346 annotation export cannot be read") from error
    if rows != ANNOTATION_ROWS or any(not values for values in result.values()):
        raise ValueError("HEST annotation must contain 938,345 cells across all 20 sections")
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
        exclusion = "".join(
            " AND NOT starts_with(t.feature_name, ?)" for _ in excluded_prefixes
        )
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
        unknown_query = """
            SELECT COUNT(*)::BIGINT
            FROM read_parquet(?) AS t
            WHERE t.qv >= ? AND (
                t.feature_name IS NULL OR (
                    NOT (t.feature_name = ANY(?))
        """ + exclusion + "))"
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
        duplicate_query = """
            SELECT COUNT(*)::BIGINT
            FROM (
                SELECT t.transcript_id
                FROM read_parquet(?) AS t
                WHERE t.qv >= ? AND t.cell_id IS NOT NULL AND t.cell_id != 'UNASSIGNED'
        """ + exclusion + """
                GROUP BY t.transcript_id
                HAVING t.transcript_id IS NULL OR COUNT(*) != 1 OR COUNT(DISTINCT t.cell_id) != 1
            ) AS duplicated
        """
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
            totals_query = """
                SELECT t.cell_id, COUNT(DISTINCT t.transcript_id)::BIGINT AS library_size
                FROM read_parquet(?) AS t
                INNER JOIN registered_cells AS r USING (cell_id)
                WHERE t.qv >= ?
            """ + overlap_clause + exclusion + " GROUP BY t.cell_id"
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
            library_half_query = """
                SELECT t.cell_id,
                       CASE WHEN RIGHT(SHA256(? || CAST(t.transcript_id AS VARCHAR)), 1)
                                      IN ('1','3','5','7','9','b','d','f')
                            THEN 1 ELSE 0 END AS split_half,
                       COUNT(DISTINCT t.transcript_id)::BIGINT AS library_size
                FROM read_parquet(?) AS t
                INNER JOIN registered_cells AS r USING (cell_id)
                WHERE t.qv >= ?
            """ + overlap_clause + exclusion + " GROUP BY t.cell_id, split_half"
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
            selected_query = """
                SELECT t.cell_id, t.feature_name,
                       CASE WHEN RIGHT(SHA256(? || CAST(t.transcript_id AS VARCHAR)), 1)
                                      IN ('1','3','5','7','9','b','d','f')
                            THEN 1 ELSE 0 END AS split_half,
                       COUNT(DISTINCT t.transcript_id)::BIGINT AS gene_count
                FROM read_parquet(?) AS t
                INNER JOIN registered_cells AS r USING (cell_id)
                WHERE t.qv >= ?
            """ + overlap_clause + """
                AND t.feature_name = ANY(?)
                GROUP BY t.cell_id, t.feature_name, split_half
            """
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
        qv_query = """
            SELECT t.cell_id, MIN(t.qv) AS minimum_qv, MEDIAN(t.qv) AS median_qv,
                   AVG(t.qv) AS mean_qv
            FROM read_parquet(?) AS t
            INNER JOIN registered_cells AS r USING (cell_id)
            WHERE t.qv >= ?
        """ + exclusion + " GROUP BY t.cell_id"
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

    try:
        import duckdb
        import pyarrow as pa
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install HEIR with the hest optional dependencies") from error
    connection = duckdb.connect(database=":memory:")
    count = 0
    try:
        connection.register(
            "retained_cells", pa.table({"cell_id": tuple(str(value) for value in cell_ids)})
        )
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


def _render_crop_variant(
    slide: _TiffPatchReader,
    centre: Tuple[float, float],
    nucleus_vertices: np.ndarray,
    cell_vertices: np.ndarray,
    variant: CropVariant,
    source_mpp: float,
) -> Tuple[np.ndarray, float, float]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install HEIR with the hest optional dependencies") from error
    size = int(round(variant.diameter_um / source_mpp))
    patch, padding_fraction, x0, y0 = slide.read_with_padding(centre, size)
    if variant.mask_mode == "none":
        return patch, padding_fraction, 0.0
    if variant.mask_mode == "blank":
        return np.full_like(patch, 255), padding_fraction, 1.0
    mask_image = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(mask_image)
    if variant.mask_mode in {"nucleus", "cell", "target_removed"}:
        vertices = nucleus_vertices if variant.mask_mode == "nucleus" else cell_vertices
        polygon = [(float(x - x0), float(y - y0)) for x, y in vertices]
        draw.polygon(polygon, fill=1)
    elif variant.mask_mode == "context_ring":
        radius = variant.inner_diameter_um / (2.0 * source_mpp)
        local_x, local_y = centre[0] - x0, centre[1] - y0
        draw.ellipse(
            (local_x - radius, local_y - radius, local_x + radius, local_y + radius),
            fill=1,
        )
    else:  # pragma: no cover - manifest validation is exhaustive
        raise ValueError("unsupported crop mask mode")
    mask = np.asarray(mask_image, dtype=np.bool_)
    if variant.mask_mode in {"nucleus", "cell"}:
        patch[~mask] = 255
    else:
        patch[mask] = 255
    return patch, padding_fraction, float(mask.mean())


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
    encoder_manifest_path: Path,
    crop_manifest_path: Path,
    data_root: Path,
    model_dir: Path,
    output_path: Path,
    plan_output_path: Path,
    qc_output_path: Path,
    *,
    device: str = "cuda",
    batch_size: int = 64,
    encoder: Optional[FrozenPatchEncoder] = None,
) -> None:
    protocol_path = protocol_path.expanduser().resolve()
    encoder_manifest_path = encoder_manifest_path.expanduser().resolve()
    crop_manifest_path = crop_manifest_path.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    plan_output_path = plan_output_path.expanduser().resolve()
    qc_output_path = qc_output_path.expanduser().resolve()
    encoder_manifest = _load_encoder_manifest(encoder_manifest_path)
    crop_manifest = _load_crop_manifest(crop_manifest_path)
    protocol = _read_json(protocol_path)
    samples = _validate_protocol(protocol, encoder_manifest, crop_manifest)
    if (
        output_path
        in {
            protocol_path,
            encoder_manifest_path,
            crop_manifest_path,
            data_root,
            model_dir,
            plan_output_path,
            qc_output_path,
        }
        or plan_output_path
        in {protocol_path, encoder_manifest_path, crop_manifest_path, data_root, model_dir}
        or qc_output_path
        in {
            protocol_path,
            encoder_manifest_path,
            crop_manifest_path,
            data_root,
            model_dir,
            plan_output_path,
        }
        or batch_size < 1
    ):
        raise ValueError("HEST output/input identity or batch size is invalid")
    broad_type_names = tuple(str(value) for value in protocol["broad_type_names"])
    markers = {
        str(name): tuple(str(gene) for gene in values)
        for name, values in protocol["type_markers"].items()
    }
    marker_genes = tuple(gene for name in broad_type_names for gene in markers[name])
    target_genes = tuple(str(value) for value in protocol["gene_ids"])
    salt = str(protocol.get("pool_assignment_salt", "hest-xenium-v1"))
    annotation_declaration = _parse_file(
        protocol["annotation_export"], "GSE250346 annotation export"
    )
    annotation_path = _resolve_input(data_root, annotation_declaration)
    annotations = _read_annotations(annotation_path)
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
    )
    resolved_provenance = []
    planned_strata: set[str] = set()
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
            planned_strata.add(
                "%s|%s|%s" % (sample.donor_id, sample.sample_id, fine_type)
            )
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
            np.linalg.norm(annotation_he - registered_cell_centres, axis=1)
            * sample.pixel_size_um
        )
        annotation_distance_limit = float(
            protocol["maximum_annotation_nucleus_distance_p95_um"]
        )
        annotation_distance_p95 = float(np.quantile(annotation_nucleus_distances, 0.95))
        registration_outlier_fraction = float(
            np.mean(annotation_nucleus_distances > annotation_distance_limit)
        )
        if annotation_distance_p95 > annotation_distance_limit:
            raise ValueError(
                "HEST annotation-to-nucleus registration distance exceeds the protocol"
            )
        if registration_outlier_fraction > float(
            protocol["maximum_registration_outlier_fraction"]
        ):
            raise ValueError("HEST annotation-to-nucleus registration outlier fraction is too high")
        expression_targets = _aggregate_expression(
            transcript_path,
            registered,
            tuple(cell_centres),
            target_genes,
            marker_genes + target_genes,
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
            rows.observation_ids.append(sample.sample_id + ":" + cell_id)
            rows.cell_ids.append(cell_id)
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
            rows.percent_negative_or_unassigned.append(
                annotation.percent_negative_or_unassigned
            )
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
            rows.nucleus_library_sizes.append(
                int(expression_targets.nucleus_library_sizes[index])
            )
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
                "retained_observation_count_before_fine_type_support": (
                    sample_end - sample_start
                ),
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
                "annotation_nucleus_distance_max_um": float(
                    annotation_nucleus_distances.max()
                ),
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
                "nucleus_eligible_transcripts": (
                    expression_targets.nucleus_eligible_transcripts
                ),
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
    preliminary_roles = np.asarray(rows.pool_roles)
    preliminary_labels = np.asarray(rows.type_labels, dtype=np.int64)
    retained = np.ones(len(rows.observation_ids), dtype=np.bool_)
    minimum_reference = int(protocol["minimum_reference_cells_per_donor_type"])
    minimum_evaluation = int(protocol["minimum_evaluation_cells_per_donor_type"])
    for donor in sorted(set(preliminary_donors.tolist())):
        donor_mask = preliminary_donors == donor
        for type_index in sorted(set(preliminary_labels[donor_mask].tolist())):
            local = donor_mask & (preliminary_labels == type_index)
            if (
                np.count_nonzero(local & (preliminary_roles == "reference"))
                < minimum_reference
                or np.count_nonzero(local & (preliminary_roles == "evaluation"))
                < minimum_evaluation
            ):
                retained[local] = False
    preliminary_splits = np.asarray(rows.split_ids)
    for type_index in sorted(set(preliminary_labels[retained].tolist())):
        local = retained & (preliminary_labels == type_index)
        development_support = len(
            set(preliminary_donors[local & (preliminary_splits == "development")].tolist())
        )
        locked_support = len(
            set(preliminary_donors[local & (preliminary_splits == "locked_test")].tolist())
        )
        if (
            development_support < int(protocol["minimum_development_donors_per_fine_type"])
            or locked_support < int(protocol["minimum_locked_donors_per_fine_type"])
        ):
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
    labels = np.asarray(rows.type_labels, dtype=np.int64)
    roles = np.asarray(rows.pool_roles)
    blocks = np.asarray(rows.block_ids)
    minimum_reference = int(protocol["minimum_reference_cells_per_donor_type"])
    minimum_evaluation = int(protocol["minimum_evaluation_cells_per_donor_type"])
    for donor in tuple(protocol["development_donors"]) + tuple(protocol["locked_test_donors"]):
        donor_mask = donors == str(donor)
        if set(roles[donor_mask].tolist()) != {"reference", "evaluation"}:
            raise ValueError("each HEST donor needs spatially disjoint reference/evaluation cells")
        for type_index in sorted(set(labels[donor_mask].tolist())):
            reference_count = np.count_nonzero(
                donor_mask & (roles == "reference") & (labels == type_index)
            )
            evaluation_count = np.count_nonzero(
                donor_mask & (roles == "evaluation") & (labels == type_index)
            )
            if reference_count < minimum_reference or evaluation_count < minimum_evaluation:
                raise ValueError("a HEST donor/type lacks the frozen reference/evaluation support")
        if set(blocks[donor_mask & (roles == "reference")]) & set(
            blocks[donor_mask & (roles == "evaluation")]
        ):
            raise ValueError("HEST reference/evaluation spatial blocks overlap")

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
    crop_padding_fractions = np.empty(
        (observations, len(crop_manifest.variants)), dtype=np.float32
    )
    crop_mask_fractions = np.empty_like(crop_padding_fractions)
    cellvit_nearest_features = (
        np.empty((observations, encoder_manifest.feature_width), dtype=np.float32)
        if rows.cellvit_names is not None
        else None
    )
    cellvit_nearest_padding_fractions = (
        np.empty(observations, dtype=np.float32) if rows.cellvit_names is not None else None
    )
    coordinate_features = np.empty((observations, 5), dtype=np.float32)
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
                            sample.pixel_size_um,
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
                    crop_mask_fractions[output_slice, crop_index] = [
                        value[2] for value in rendered
                    ]
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
                            sample.pixel_size_um,
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
            coordinate_features[start:end] = np.column_stack(
                (
                    normalized,
                    normalized[:, 0] ** 2,
                    normalized[:, 1] ** 2,
                    normalized[:, 0] * normalized[:, 1],
                )
            )

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
    nucleus_library_size_half_a = np.asarray(
        rows.nucleus_library_size_half_a, dtype=np.int64
    )
    nucleus_library_size_half_b = np.asarray(
        rows.nucleus_library_size_half_b, dtype=np.int64
    )
    whole_cell_library_size_half_a = np.asarray(
        rows.whole_cell_library_size_half_a, dtype=np.int64
    )
    whole_cell_library_size_half_b = np.asarray(
        rows.whole_cell_library_size_half_b, dtype=np.int64
    )
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
    cell_nucleus_distances_um = np.linalg.norm(
        native_cell_centres - nucleus_centres, axis=1
    ) * SOURCE_MPP
    nucleus_areas_um2 = np.asarray(rows.nucleus_areas_um2, dtype=np.float64)
    cell_areas_um2 = np.asarray(rows.cell_areas_um2, dtype=np.float64)
    area_ratio = np.divide(
        nucleus_areas_um2,
        cell_areas_um2,
        out=np.zeros_like(nucleus_areas_um2),
        where=cell_areas_um2 > 0,
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
    registration_qc_pass = (
        (np.asarray(rows.annotation_nucleus_distances_um) <= float(
            protocol["maximum_annotation_nucleus_distance_p95_um"]
        ))
        & (np.asarray(rows.affine_residual_p95_um) <= float(
            protocol["maximum_affine_registration_residual_p95_um"]
        ))
    )
    target_qc_pass = (
        (np.asarray(rows.nucleus_library_sizes) >= int(protocol["minimum_transcripts_per_cell"]))
        & (np.asarray(rows.whole_cell_library_sizes) >= np.asarray(rows.nucleus_library_sizes))
        & (target_counts.sum(axis=1) <= np.asarray(rows.nucleus_library_sizes))
        & (whole_cell_target_counts.sum(axis=1) <= np.asarray(rows.whole_cell_library_sizes))
    )
    crop_qc_pass = np.max(crop_padding_fractions, axis=1) <= float(
        protocol["maximum_crop_padding_fraction"]
    )
    if cellvit_nearest_padding_fractions is not None:
        crop_qc_pass &= cellvit_nearest_padding_fractions <= float(
            protocol["maximum_crop_padding_fraction"]
        )
    if not target_qc_pass.all() or not crop_qc_pass.all():
        raise ValueError("HEST target or crop QC invariant failed")
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
                "spatial_block_um",
                "spatial_roi_um",
                "opposite_pool_guard_um",
                "pool_assignment_salt",
            )
        }
    )
    target_source_sha256 = _canonical_sha256(
        [
            (sample["sample_id"], sample["transcripts_sha256"])
            for sample in resolved_provenance
        ]
    )
    source_file_manifest_sha256 = _canonical_sha256(
        {
            "annotation": annotation_declaration.sha256,
            "samples": [
                {
                    key: value
                    for key, value in sample.items()
                    if key.endswith("_sha256")
                }
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
    program_gene_membership = np.zeros(
        (len(program_names), len(target_genes)), dtype=np.bool_
    )
    for program_index, name in enumerate(program_names):
        for gene in protocol["target_programs"][name]:
            program_gene_membership[program_index, gene_index[str(gene)]] = True
    provenance = {
        "schema": SOURCE_SCHEMA,
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
        "annotation_rows": ANNOTATION_ROWS,
        "label_field_primary": "final_CT",
        "label_field_secondary": "final_lineage",
        "target_transcript_filters": {
            "primary_nucleus": (
                "overlaps_nucleus==1,qv>=20,non-control,COUNT(DISTINCT transcript_id)"
            ),
            "secondary_whole_cell": (
                "qv>=20,non-control,COUNT(DISTINCT transcript_id)"
            ),
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
                    "diameter_um": variant.diameter_um,
                    "inner_diameter_um": variant.inner_diameter_um,
                    "mask_mode": variant.mask_mode,
                }
                for variant in crop_manifest.variants
            ],
        },
        "claim_scope": "nucleus_centered_local_context_association",
        "authorizes_nucleus_intrinsic_claim": False,
        "native_xenium_registration_only": True,
        "cellvit_target_registration": False,
        "samples": resolved_provenance,
        "exclusion_counts": exclusion_counts,
    }
    payload: Dict[str, object] = {
        "schema_version": np.asarray(SOURCE_SCHEMA),
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
        "fine_type_marker_gene_ids": np.asarray([], dtype=str),
        "program_names": np.asarray(program_names),
        "program_gene_membership": program_gene_membership,
        "coordinate_features": coordinate_features,
        "coordinate_feature_names": np.asarray(
            ["he_x_normalized", "he_y_normalized", "he_x_squared", "he_y_squared", "he_xy"]
        ),
        "technical_covariates": np.asarray(rows.technical, dtype=np.float32)[:, None],
        "technical_covariate_names": np.asarray(["log1p_library_size"]),
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
        "detected_target_genes": np.asarray(
            rows.nucleus_detected_target_genes, dtype=np.int64
        ),
        "transcript_qv_summary": np.stack(rows.transcript_qv_summaries).astype(np.float32),
        "transcript_qv_summary_names": np.asarray(
            ["minimum_qv", "median_qv", "mean_qv"]
        ),
        "stain_features": np.empty((observations, 0), dtype=np.float32),
        "stain_feature_names": np.asarray([], dtype=str),
        "composition_features": np.empty((observations, 0), dtype=np.float32),
        "composition_feature_names": np.asarray([], dtype=str),
        "nuclear_morphometric_features": np.column_stack(
            (
                nucleus_areas_um2,
                area_ratio,
                cell_nucleus_distances_um,
                rows.nucleus_centroid_inside_cell,
            )
        ).astype(np.float32),
        "nuclear_morphometric_feature_names": np.asarray(
            [
                "nucleus_area_um2",
                "nucleus_to_cell_area_ratio",
                "cell_nucleus_centroid_distance_um",
                "nucleus_centroid_inside_cell",
            ]
        ),
        "cell_morphometric_features": cell_areas_um2.astype(np.float32)[:, None],
        "cell_morphometric_feature_names": np.asarray(["cell_area_um2"]),
        "registration_qc_features": registration_qc,
        "registration_qc_feature_names": np.asarray(registration_qc_names),
        "registration_qc_pass": registration_qc_pass,
        "registration_cardinality": np.ones(observations, dtype=np.int8),
        "target_qc_pass": target_qc_pass,
        "crop_qc_pass": crop_qc_pass,
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
        "crop_scale": np.asarray("nucleus_centered"),
        "crop_role": np.asarray(primary_variant.role),
        "crop_diameter_um": np.asarray(primary_variant.diameter_um),
        "source_mpp": np.asarray(crop_manifest.source_mpp),
        "model_mpp": np.asarray(encoder_manifest.model_mpp),
        "model_input_pixels": np.asarray(encoder_manifest.input_pixels),
        "mask_mode": np.asarray(primary_variant.mask_mode),
        "authorizes_nucleus_intrinsic_claim": np.asarray(False),
        "cohort_id": np.asarray("HEST"),
        "cohort_release": np.asarray(DATASET_REVISION),
        "assay": np.asarray(protocol["assay"]),
        "observation_level": np.asarray(protocol["observation_level"]),
        "target_construction": np.asarray(protocol["target_construction"]),
        "secondary_target_construction": np.asarray("whole_cell_xenium_transcripts"),
        "label_source_sha256": np.asarray(annotation_declaration.sha256),
        "source_file_manifest_sha256": np.asarray(source_file_manifest_sha256),
        "registration_source_sha256": np.asarray(registration_source_sha256),
        "registration_manifest_sha256": np.asarray(registration_source_sha256),
        "segmentation_manifest_sha256": np.asarray(segmentation_manifest_sha256),
        "exclusion_policy_sha256": np.asarray(exclusion_policy_sha256),
        "target_source_sha256": np.asarray(target_source_sha256),
        "target_manifest_sha256": np.asarray(target_manifest_sha256),
        "study_manifest_sha256": np.asarray(protocol["study_manifest_sha256"]),
        "planned_stratum_ids": np.asarray(planned_stratum_ids),
        "planned_stratum_manifest_sha256": np.asarray(
            planned_stratum_manifest_sha256
        ),
        "transcript_split_method": np.asarray("sha256-final-byte-lsb-v1"),
        "transcript_minimum_qv": np.asarray(
            float(protocol["minimum_transcript_qv"]), dtype=np.float32
        ),
        "transcript_split_salt_sha256": np.asarray(
            hashlib.sha256(str(protocol["transcript_split_salt"]).encode("utf-8")).hexdigest()
        ),
        "transcript_identity_manifest_sha256": np.asarray(
            transcript_identity_hasher.hexdigest()
        ),
        "eligible_target_transcripts": np.asarray(
            eligible_target_transcripts, dtype=np.int64
        ),
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
        payload["cellvit_nearest_crop_padding_fraction"] = (
            cellvit_nearest_padding_fractions
        )
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
        for donor in DEVELOPMENT_DONORS + LOCKED_TEST_DONORS
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
        "scientific_scope": "nucleus_centered_local_context_association",
        "authorizes_nucleus_intrinsic_claim": False,
        "development_donors": list(DEVELOPMENT_DONORS),
        "locked_test_donors": list(LOCKED_TEST_DONORS),
        "donor_sections": donor_sections,
        "gene_ids": list(target_genes),
        "type_names": list(fine_type_names),
        "broad_type_names": list(broad_type_names),
        "type_marker_gene_ids": list(marker_genes),
        "technical_covariate_names": ["log1p_library_size"],
        "frozen_feature_names": [
            "%s_%04d" % (feature_name_prefix, index)
            for index in range(encoder_manifest.feature_width)
        ],
        "crop_ids": list(crop_ids),
        "coordinate_feature_names": [
            "he_x_normalized",
            "he_y_normalized",
            "he_x_squared",
            "he_y_squared",
            "he_xy",
        ],
        "stain_feature_names": [],
        "composition_feature_names": [],
        "feature_space_id": feature_space_id,
        "feature_checkpoint_sha256": encoder_manifest.checkpoint_sha256,
        "encoder_manifest_sha256": encoder_manifest.sha256,
        "crop_manifest_sha256": crop_manifest.sha256,
        "molecular_space_id": molecular_space_id,
        "label_source_sha256": annotation_declaration.sha256,
        "registration_source_sha256": registration_source_sha256,
        "exclusion_policy_sha256": exclusion_policy_sha256,
        "registration_method": str(protocol["registration_method"]),
        "encoder_name": encoder_manifest.repository,
        "crop_scale": "nucleus_centered",
        "crop_metadata": {
            "primary_crop_id": crop_manifest.primary_crop_id,
            "crop_role": primary_variant.role,
            "crop_diameter_um": primary_variant.diameter_um,
            "source_mpp": crop_manifest.source_mpp,
            "model_mpp": encoder_manifest.model_mpp,
            "model_input_pixels": encoder_manifest.input_pixels,
            "mask_mode": primary_variant.mask_mode,
            "padding": crop_manifest.padding,
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
            "primary": "final_CT",
            "secondary": "final_lineage",
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
            "development_donors": list(DEVELOPMENT_DONORS),
            "locked_test_donors": list(LOCKED_TEST_DONORS),
        },
        "encoder": {
            "repository": encoder_manifest.repository,
            "revision": encoder_manifest.revision,
            "checkpoint_sha256": encoder_manifest.checkpoint_sha256,
            "manifest_sha256": encoder_manifest.sha256,
            "feature_width": encoder_manifest.feature_width,
        },
        "preprocessing": {
            "implementation": "native_xenium_nucleus_centered_physical_crop_ladder",
            "implementation_sha256": crop_manifest.sha256,
            "crop_role": primary_variant.role,
            "crop_diameter_um": primary_variant.diameter_um,
            "source_mpp": crop_manifest.source_mpp,
            "model_mpp": encoder_manifest.model_mpp,
            "model_input_pixels": encoder_manifest.input_pixels,
            "mask_mode": primary_variant.mask_mode,
            "primary_crop_id": crop_manifest.primary_crop_id,
            "crop_ids": list(crop_ids),
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
            "transcript_identity_manifest_sha256": (
                transcript_identity_hasher.hexdigest()
            ),
        },
        "labels": {
            "primary": "final_CT",
            "secondary": "final_lineage",
            "fine_type_names": list(fine_type_names),
            "broad_type_names": list(broad_type_names),
            "source_sha256": annotation_declaration.sha256,
        },
        "reference_mode": "simulated_spatially_disjoint_unpaired_rna",
        "reference_pool": {
            "construction": "same_donor_fine_type_spatial_block_pool",
            "spatially_disjoint": True,
            "minimum_per_donor_type": minimum_reference,
            "observation_manifest_sha256": observation_manifest_sha256,
        },
        "nuisance_covariates": [
            "log1p_library_size",
            "section_id",
            "disease_status",
            "site_id",
            "batch_id",
        ],
        "registration_qc_feature_names": list(registration_qc_names),
        "gate": {
            "ranks": [2, 4, 6],
            "ridge_penalties": [0.1, 1.0, 10.0, 100.0],
            "permutation_seeds": [17, 29, 41],
            "permutations_per_seed": 100,
            "minimum_support": minimum_evaluation,
            "minimum_development_donors": 5,
            "minimum_locked_donors": 5,
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
            "annotation_nucleus_distance_p50_um": float(
                np.quantile(registration_distances, 0.5)
            ),
            "median_annotation_nucleus_distance_um": float(
                np.quantile(registration_distances, 0.5)
            ),
            "annotation_nucleus_distance_p95_um": float(
                np.quantile(registration_distances, 0.95)
            ),
            "p95_annotation_nucleus_distance_um": float(
                np.quantile(registration_distances, 0.95)
            ),
            "maximum_allowed_p95_um": float(
                protocol["maximum_annotation_nucleus_distance_p95_um"]
            ),
            "annotation_nucleus_distance_max_um": float(registration_distances.max()),
            "nucleus_centroid_outside_cell_fraction": float(
                1.0 - np.mean(rows.nucleus_centroid_inside_cell)
            ),
            "row_within_distance_threshold_fraction": float(
                np.mean(registration_qc_pass)
            ),
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
            "nucleus_eligible_transcripts": int(sum(
                int(value["nucleus_eligible_transcripts"]) for value in resolved_provenance
            )),
            "whole_cell_eligible_transcripts": int(sum(
                int(value["whole_cell_eligible_transcripts"])
                for value in resolved_provenance
            )),
            "eligible_target_transcripts": eligible_target_transcripts,
            "transcript_identity_manifest_sha256": (
                transcript_identity_hasher.hexdigest()
            ),
            "all_row_qc_pass": bool(target_qc_pass.all()),
        },
        "crops": {
            "primary_crop_id": crop_manifest.primary_crop_id,
            "maximum_allowed_padding_fraction": float(
                protocol["maximum_crop_padding_fraction"]
            ),
            "all_row_qc_pass": bool(crop_qc_pass.all()),
            "by_crop_id": padding_summary,
        },
        "exclusion_counts": exclusion_counts,
        "planned_strata": {
            "ids": list(planned_stratum_ids),
            "manifest_sha256": planned_stratum_manifest_sha256,
        },
        "pass": bool(
            registration_gate_pass and target_qc_pass.all() and crop_qc_pass.all()
        ),
    }
    _write_json(qc_output_path, qc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--encoder-manifest", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    build_source(
        args.protocol,
        args.encoder_manifest,
        args.crop_manifest,
        args.data_root,
        args.model_dir,
        args.source_output,
        args.plan_output,
        args.qc_output,
        device=args.device,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
