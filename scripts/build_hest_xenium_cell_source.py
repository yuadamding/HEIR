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
from typing import Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

PROTOCOL_SCHEMA = "heir.hest_xenium_cell_protocol.v1"
SOURCE_SCHEMA = "heir.hest_xenium_cell_source.v1"
DATASET_REPO = "MahmoodLab/hest"
DATASET_REVISION = "7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec"
MODEL_REPO = "bioptimus/H-optimus-1"
MODEL_REVISION = "3592cb220dec7a150c5d7813fb56e68bd57473b9"
MODEL_CHECKPOINT_SHA256 = "c4f1e5b457ddf00679626053b0bf2899be6a19c3a04ad191c87ad1cdfd1abfe1"
FEATURE_WIDTH = 1536
HOPTIMUS_ARCHITECTURE = "vit_giant_patch14_reg4_dinov2"
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
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
        if not math.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError("HEST sample pixel size must be positive")
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


def _validate_protocol(protocol: Mapping[str, object]) -> Tuple[Sample, ...]:
    exact = {
        "schema": PROTOCOL_SCHEMA,
        "scientific_scope": "nucleus_centered_morphology_confirmation",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
        "model_input_pixels": 224,
        "model_feature_width": FEATURE_WIDTH,
        "model_mpp": 0.5,
        "normalization": "log1p_cpm_10000",
        "assay": "Xenium",
        "observation_level": "cell",
        "target_construction": "registered_cell_expression",
        "registration_method": "native_xenium_cell_id_join",
        "minimum_transcript_qv": 20.0,
    }
    for name, expected in exact.items():
        if protocol.get(name) != expected:
            raise ValueError("HEST protocol %s differs from the pinned design" % name)
    development = tuple(str(value) for value in protocol.get("development_donors", ()))
    locked = tuple(str(value) for value in protocol.get("locked_test_donors", ()))
    if development != DEVELOPMENT_DONORS or locked != LOCKED_TEST_DONORS:
        raise ValueError("HEST protocol differs from the frozen 10/5 true-donor split")
    type_names = tuple(str(value) for value in protocol.get("type_names", ()))
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
        "spatial_block_um",
        "spatial_roi_um",
        "opposite_pool_guard_um",
        "cellvit_sensitivity_radius_um",
    ):
        value = float(protocol.get(name, -1.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError("HEST protocol %s must be finite and nonnegative" % name)
    block = float(protocol["spatial_block_um"])
    roi = float(protocol["spatial_roi_um"])
    guard = float(protocol["opposite_pool_guard_um"])
    if block <= 2 * guard or roi <= 0 or block < roi:
        raise ValueError("HEST spatial block/ROI/guard design is invalid")
    mean = tuple(float(value) for value in protocol.get("model_mean", ()))
    std = tuple(float(value) for value in protocol.get("model_std", ()))
    if mean != HOPTIMUS_MEAN or std != HOPTIMUS_STD:
        raise ValueError("HEST H-Optimus-1 normalization differs from the pinned model")
    prefixes = tuple(str(value) for value in protocol.get("excluded_feature_prefixes", ()))
    if prefixes != CONTROL_PREFIXES:
        raise ValueError("HEST non-control transcript filter differs from the frozen release")
    salt = protocol.get("pool_assignment_salt")
    if not isinstance(salt, str) or not salt.strip():
        raise ValueError("HEST spatial pool assignment salt is not frozen")
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


def _polygon_centroid(wkb: bytes) -> Tuple[float, float]:
    """Return the exterior-ring centroid of a finite, little/big-endian WKB Polygon."""

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
    return float(centre[0]), float(centre[1])


def _read_native_segmentation(path: Path) -> Dict[str, Tuple[float, float]]:
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
    result: Dict[str, Tuple[float, float]] = {}
    for identity, geometry in zip(identities, geometries):
        cell_id = str(identity)
        if not cell_id or cell_id == "UNASSIGNED" or cell_id in result:
            raise ValueError("native Xenium segmentation IDs are empty or duplicated")
        result[cell_id] = _polygon_centroid(geometry)
    if not result:
        raise ValueError("native Xenium segmentation is empty")
    return result


def _read_annotations(path: Path) -> Dict[str, Dict[str, int]]:
    required = {
        "hest_id",
        "sample",
        "patient",
        "cell_id",
        "final_CT",
        "final_lineage",
        "x_centroid",
        "y_centroid",
        "nCount_RNA",
    }
    result: Dict[str, Dict[str, int]] = {sample_id: {} for sample_id in SECTION_IDENTITIES}
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
                if lineage not in TYPE_NAMES or not row["final_CT"].strip():
                    raise ValueError("HEST annotation has an unknown or empty RNA-derived label")
                cell_id = row["cell_id"]
                if not cell_id or cell_id == "UNASSIGNED" or cell_id in result[sample_id]:
                    raise ValueError("HEST annotation cell IDs are empty or duplicated")
                centroid = (float(row["x_centroid"]), float(row["y_centroid"]))
                if not np.isfinite(centroid).all() or float(row["nCount_RNA"]) <= 0:
                    raise ValueError("HEST annotation contains a malformed high-QC cell")
                result[sample_id][cell_id] = TYPE_NAMES.index(lineage)
                rows += 1
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError("HEST GSE250346 annotation export cannot be read") from error
    if rows != ANNOTATION_ROWS or any(not values for values in result.values()):
        raise ValueError("HEST annotation must contain 938,345 cells across all 20 sections")
    return result


def _aggregate_expression(
    path: Path,
    cell_ids: Sequence[str],
    genes: Sequence[str],
    *,
    minimum_qv: float,
    excluded_prefixes: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
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
    row_for_id = {value: index for index, value in enumerate(ordered_ids)}
    gene_for_name = {str(value): index for index, value in enumerate(genes)}
    if len(row_for_id) != len(ordered_ids) or len(gene_for_name) != len(genes):
        raise ValueError("HEST registered cell or gene identities are duplicated")
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("registered_cells", pa.table({"cell_id": ordered_ids}))
        exclusion = "".join(
            " AND NOT starts_with(t.feature_name, ?)" for _ in excluded_prefixes
        )
        base_parameters: list[object] = [str(path), float(minimum_qv)]
        base_parameters.extend(excluded_prefixes)
        totals_query = """
            SELECT t.cell_id, COUNT(*)::BIGINT AS library_size
            FROM read_parquet(?) AS t
            INNER JOIN registered_cells AS r USING (cell_id)
            WHERE t.qv >= ? AND t.overlaps_nucleus = 1
        """ + exclusion + " GROUP BY t.cell_id"
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
        selected_query = """
            SELECT t.cell_id, t.feature_name, COUNT(*)::BIGINT AS gene_count
            FROM read_parquet(?) AS t
            INNER JOIN registered_cells AS r USING (cell_id)
            WHERE t.qv >= ? AND t.overlaps_nucleus = 1 AND t.feature_name = ANY(?)
            GROUP BY t.cell_id, t.feature_name
        """
        counts = np.zeros((len(ordered_ids), len(genes)), dtype=np.float32)
        parameters = [str(path), float(minimum_qv), list(genes)]
        selected_result = connection.execute(selected_query, parameters)
        selected_batches = (
            selected_result.to_arrow_reader(65_536)
            if hasattr(selected_result, "to_arrow_reader")
            else selected_result.fetch_record_batch(65_536)
        )
        for batch in selected_batches:
            for cell_id, gene, count in zip(
                batch.column("cell_id").to_pylist(),
                batch.column("feature_name").to_pylist(),
                batch.column("gene_count").to_pylist(),
            ):
                counts[row_for_id[str(cell_id)], gene_for_name[str(gene)]] = float(count)
    finally:
        connection.close()
    return counts, totals


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
) -> Tuple[np.ndarray, Tuple[str, ...]]:
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
    radius_squared = radius_pixels * radius_pixels
    for row, (x, y) in enumerate(native_centres):
        centre_bin = (int(x // bin_size), int(y // bin_size))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for index in bins.get((centre_bin[0] + dx, centre_bin[1] + dy), ()):
                    distance = (centres[index, 0] - x) ** 2 + (centres[index, 1] - y) ** 2
                    if distance <= radius_squared:
                        result[row, class_index[classes[index]]] += 1.0
    names = tuple("cellvit_log1p_count_%s" % name for name in class_names)
    return np.log1p(result), names


class PatchEncoder(Protocol):
    output_width: int

    def encode(self, patches: np.ndarray) -> np.ndarray: ...


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
        if source_x1 > source_x0 and source_y1 > source_y0:
            value = np.asarray(
                self._array[source_y0:source_y1, source_x0:source_x1, :3], dtype=np.uint8
            )
            patch[
                source_y0 - y0 : source_y1 - y0,
                source_x0 - x0 : source_x1 - x0,
            ] = value
        return patch

    def close(self) -> None:
        if getattr(self, "_store", None) is not None and hasattr(self._store, "close"):
            self._store.close()
        if getattr(self, "_tiff", None) is not None:
            self._tiff.close()

    def __enter__(self) -> "_TiffPatchReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _HOptimusEncoder:
    def __init__(self, model_dir: Path, protocol: Mapping[str, object], device: str):
        try:
            import timm
            import torch
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("install HEIR with the hest optional dependencies") from error
        config_path = model_dir / "config.json"
        checkpoint_path = model_dir / "model.safetensors"
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise ValueError("local pinned H-Optimus-1 config/checkpoint is incomplete")
        if _sha256_file(checkpoint_path) != protocol["model_checkpoint_sha256"]:
            raise ValueError("H-Optimus-1 checkpoint differs from the pinned SHA-256")
        config = _read_json(config_path)
        pretrained = config.get("pretrained_cfg", {})
        if not isinstance(pretrained, Mapping):
            raise ValueError("H-Optimus-1 pretrained configuration is malformed")
        architecture = str(config.get("architecture", pretrained.get("architecture", "")))
        configured_width = int(config.get("num_features", pretrained.get("num_features", -1)))
        input_size = tuple(pretrained.get("input_size", ()))
        configured_mean = tuple(float(value) for value in pretrained.get("mean", ()))
        configured_std = tuple(float(value) for value in pretrained.get("std", ()))
        if (
            architecture != HOPTIMUS_ARCHITECTURE
            or configured_width != FEATURE_WIDTH
            or input_size != (3, 224, 224)
            or configured_mean != HOPTIMUS_MEAN
            or configured_std != HOPTIMUS_STD
        ):
            raise ValueError("H-Optimus-1 local config differs from the pinned model")
        model_args = dict(config.get("model_args", {}))
        model_args.setdefault("init_values", 1.0e-5)
        model_args.setdefault("dynamic_img_size", False)
        model_args["num_classes"] = 0
        self._model = timm.create_model(architecture, pretrained=False, **model_args)
        timm.models.load_checkpoint(self._model, str(checkpoint_path), strict=True)
        if int(getattr(self._model, "num_features", -1)) != FEATURE_WIDTH:
            raise ValueError("loaded H-Optimus-1 does not expose 1536 direct features")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self._device = torch.device(device)
        self._model.eval().to(self._device)
        self._torch = torch
        self._input_pixels = int(protocol["model_input_pixels"])
        self._mean = torch.tensor(protocol["model_mean"], dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(protocol["model_std"], dtype=torch.float32).view(1, 3, 1, 1)
        self._mean = self._mean.to(self._device)
        self._std = self._std.to(self._device)
        self.output_width = FEATURE_WIDTH

    def encode(self, patches: np.ndarray) -> np.ndarray:
        torch = self._torch
        value = torch.from_numpy(np.ascontiguousarray(patches)).permute(0, 3, 1, 2)
        value = value.to(self._device, dtype=torch.float32, non_blocking=True).div_(255.0)
        if value.shape[-2:] != (self._input_pixels, self._input_pixels):
            value = torch.nn.functional.interpolate(
                value,
                size=(self._input_pixels, self._input_pixels),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        value = (value - self._mean) / self._std
        autocast_enabled = self._device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=self._device.type, dtype=torch.float16, enabled=autocast_enabled
        ):
            encoded = self._model(value)
        if isinstance(encoded, (tuple, list)):
            encoded = encoded[0]
        if encoded.ndim == 3:
            encoded = encoded[:, 0]
        if encoded.ndim != 2 or encoded.shape[1] != self.output_width:
            raise ValueError("H-Optimus-1 output is not a 1536-dimensional CLS feature")
        return encoded.float().cpu().numpy()


@dataclass
class _Rows:
    observation_ids: list[str]
    donor_ids: list[str]
    sample_ids: list[str]
    split_ids: list[str]
    block_ids: list[str]
    roi_ids: list[str]
    pool_roles: list[str]
    type_labels: list[int]
    centres: list[Tuple[float, float]]
    sample_rows: list[Tuple[int, int, Sample, Path]]
    targets: list[np.ndarray]
    technical: list[float]
    cellvit: list[np.ndarray]
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


def build_source(
    protocol_path: Path,
    data_root: Path,
    model_dir: Path,
    output_path: Path,
    *,
    device: str = "cuda",
    batch_size: int = 64,
    encoder: Optional[PatchEncoder] = None,
) -> None:
    protocol_path = protocol_path.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    protocol = _read_json(protocol_path)
    samples = _validate_protocol(protocol)
    if output_path in {protocol_path, data_root, model_dir} or batch_size < 1:
        raise ValueError("HEST output/input identity or batch size is invalid")
    type_names = tuple(str(value) for value in protocol["type_names"])
    markers = {
        str(name): tuple(str(gene) for gene in values)
        for name, values in protocol["type_markers"].items()
    }
    marker_genes = tuple(gene for name in type_names for gene in markers[name])
    target_genes = tuple(str(value) for value in protocol["gene_ids"])
    salt = str(protocol.get("pool_assignment_salt", "hest-xenium-v1"))
    annotation_declaration = _parse_file(
        protocol["annotation_export"], "GSE250346 annotation export"
    )
    annotation_path = _resolve_input(data_root, annotation_declaration)
    annotations = _read_annotations(annotation_path)
    rows = _Rows([], [], [], [], [], [], [], [], [], [], [], [], [])
    resolved_provenance = []
    exclusion_counts: Dict[str, int] = {
        "native_cells_outside_high_qc_annotation": 0,
        "low_transcripts": 0,
        "spatial_guard": 0,
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
        exclusion_counts["native_cells_outside_high_qc_annotation"] += len(
            (set(cell_centres) | set(nucleus_centres)) - set(annotated)
        )
        counts, library = _aggregate_expression(
            transcript_path,
            registered,
            target_genes,
            minimum_qv=float(protocol["minimum_transcript_qv"]),
            excluded_prefixes=tuple(str(value) for value in protocol["excluded_feature_prefixes"]),
        )
        expression = _log_cpm(counts, library)
        labels = np.asarray([annotated[cell_id] for cell_id in registered], dtype=np.int64)
        sample_start = len(rows.observation_ids)
        retained_centres = []
        for index, cell_id in enumerate(registered):
            spatial = _spatial_identity(
                sample.sample_id,
                nucleus_centres[cell_id],
                sample.pixel_size_um,
                block_um=float(protocol["spatial_block_um"]),
                roi_um=float(protocol["spatial_roi_um"]),
                guard_um=float(protocol["opposite_pool_guard_um"]),
                salt=salt,
            )
            if library[index] < int(protocol["minimum_transcripts_per_cell"]):
                exclusion_counts["low_transcripts"] += 1
                continue
            if not spatial.guard_pass:
                exclusion_counts["spatial_guard"] += 1
                continue
            rows.observation_ids.append(sample.sample_id + ":" + cell_id)
            rows.donor_ids.append(sample.donor_id)
            rows.sample_ids.append(sample.sample_id)
            rows.split_ids.append(sample.split_id)
            rows.block_ids.append(spatial.block_id)
            rows.roi_ids.append(spatial.roi_id)
            rows.pool_roles.append(spatial.pool_role)
            rows.type_labels.append(int(labels[index]))
            rows.centres.append(nucleus_centres[cell_id])
            rows.targets.append(expression[index].astype(np.float32))
            rows.technical.append(math.log1p(float(library[index])))
            retained_centres.append(nucleus_centres[cell_id])
        sample_end = len(rows.observation_ids)
        rows.sample_rows.append((sample_start, sample_end, sample, wsi_path))
        if sample.cellvit_seg is not None:
            cellvit_path = _resolve_input(data_root, sample.cellvit_seg)
            sensitivity, names = _cellvit_sensitivity(
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
        resolved_provenance.append(
            {
                "sample_id": sample.sample_id,
                "donor_id": sample.donor_id,
                "split_id": sample.split_id,
                "pixel_size_um": sample.pixel_size_um,
                "wsi_sha256": sample.wsi.sha256,
                "transcripts_sha256": sample.transcripts.sha256,
                "cell_seg_sha256": sample.cell_seg.sha256,
                "nucleus_seg_sha256": sample.nucleus_seg.sha256,
                "cellvit_seg_sha256": (
                    sample.cellvit_seg.sha256 if sample.cellvit_seg is not None else None
                ),
            }
        )

    observations = len(rows.observation_ids)
    if not observations or len(set(rows.observation_ids)) != observations:
        raise ValueError("HEST retained observation identities are empty or duplicated")
    donors = np.asarray(rows.donor_ids)
    labels = np.asarray(rows.type_labels, dtype=np.int64)
    roles = np.asarray(rows.pool_roles)
    blocks = np.asarray(rows.block_ids)
    for donor in tuple(protocol["development_donors"]) + tuple(protocol["locked_test_donors"]):
        donor_mask = donors == str(donor)
        if set(roles[donor_mask].tolist()) != {"reference", "evaluation"}:
            raise ValueError("each HEST donor needs spatially disjoint reference/evaluation cells")
        for type_index in sorted(set(labels[donor_mask & (roles == "evaluation")].tolist())):
            if not np.any(donor_mask & (roles == "reference") & (labels == type_index)):
                raise ValueError("an evaluated HEST donor/type lacks a reference cell")
        if set(blocks[donor_mask & (roles == "reference")]) & set(
            blocks[donor_mask & (roles == "evaluation")]
        ):
            raise ValueError("HEST reference/evaluation spatial blocks overlap")

    if encoder is None:
        encoder = _HOptimusEncoder(model_dir, protocol, device)
    if encoder.output_width != FEATURE_WIDTH:
        raise ValueError("HEST frozen encoder width must be 1536")
    features = np.empty((observations, FEATURE_WIDTH), dtype=np.float32)
    coordinate_features = np.empty((observations, 5), dtype=np.float32)
    for start, end, sample, wsi_path in rows.sample_rows:
        if start == end:
            continue
        crop_pixels = int(
            round(
                float(protocol["model_input_pixels"])
                * float(protocol["model_mpp"])
                / sample.pixel_size_um
            )
        )
        with _TiffPatchReader(wsi_path) as slide:
            local_centres = rows.centres[start:end]
            for batch_start in range(0, end - start, batch_size):
                batch_end = min(batch_start + batch_size, end - start)
                patches = np.stack(
                    [
                        slide.read(centre, crop_pixels)
                        for centre in local_centres[batch_start:batch_end]
                    ]
                )
                encoded = np.asarray(encoder.encode(patches), dtype=np.float32)
                expected = (batch_end - batch_start, FEATURE_WIDTH)
                if encoded.shape != expected or not np.isfinite(encoded).all():
                    raise ValueError("HEST frozen H-Optimus-1 features are malformed")
                features[start + batch_start : start + batch_end] = encoded
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

    target_values = np.stack(rows.targets).astype(np.float32)
    provenance = {
        "schema": SOURCE_SCHEMA,
        "protocol_sha256": _sha256_file(protocol_path),
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
        "annotation_source": "GSE250346 corrected Seurat metadata export",
        "annotation_sha256": annotation_declaration.sha256,
        "annotation_rows": ANNOTATION_ROWS,
        "label_field": "final_lineage",
        "target_transcript_filter": "overlaps_nucleus==1,qv>=20,non-control",
        "native_xenium_registration_only": True,
        "cellvit_target_registration": False,
        "samples": resolved_provenance,
        "exclusion_counts": exclusion_counts,
    }
    payload: Dict[str, object] = {
        "schema_version": np.asarray(SOURCE_SCHEMA),
        "observation_ids": np.asarray(rows.observation_ids),
        "donor_ids": donors,
        "patient_ids": donors,
        "sample_ids": np.asarray(rows.sample_ids),
        "source_sample_ids": np.asarray(
            [SECTION_IDENTITIES[sample_id][1] for sample_id in rows.sample_ids]
        ),
        "split_ids": np.asarray(rows.split_ids),
        "block_ids": blocks,
        "roi_ids": np.asarray(rows.roi_ids),
        "pool_roles": roles,
        "type_labels": labels,
        "type_names": np.asarray(type_names),
        "frozen_features": features,
        "molecular_targets": target_values,
        "gene_ids": np.asarray(target_genes),
        "type_marker_gene_ids": np.asarray(marker_genes),
        "coordinate_features": coordinate_features,
        "technical_covariates": np.asarray(rows.technical, dtype=np.float32)[:, None],
        "technical_covariate_names": np.asarray(["log1p_library_size"]),
        "stain_features": np.empty((observations, 0), dtype=np.float32),
        "stain_feature_names": np.asarray([], dtype=str),
        "composition_features": np.empty((observations, 0), dtype=np.float32),
        "composition_feature_names": np.asarray([], dtype=str),
        "registration_is_one_to_one": np.ones(observations, dtype=np.bool_),
        "feature_space_id": np.asarray(
            "bioptimus/H-optimus-1@%s:cls1536" % MODEL_REVISION
        ),
        "molecular_space_id": np.asarray("xenium_cell_log1p_cpm_10000"),
        "feature_checkpoint_sha256": np.asarray(MODEL_CHECKPOINT_SHA256),
        "registration_method": np.asarray(protocol["registration_method"]),
        "encoder_name": np.asarray(MODEL_REPO),
        "crop_scale": np.asarray("nucleus_centered_112um_to_224px_0.5mpp"),
        "cohort_id": np.asarray("HEST"),
        "cohort_release": np.asarray(DATASET_REVISION),
        "assay": np.asarray(protocol["assay"]),
        "observation_level": np.asarray(protocol["observation_level"]),
        "target_construction": np.asarray(protocol["target_construction"]),
        "label_source_sha256": np.asarray(annotation_declaration.sha256),
        "registration_source_sha256": np.asarray(
            _canonical_sha256(
                [
                    (sample["sample_id"], sample["cell_seg_sha256"], sample["nucleus_seg_sha256"])
                    for sample in resolved_provenance
                ]
            )
        ),
        "exclusion_policy_sha256": np.asarray(
            _canonical_sha256(
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
        ),
        "target_source_sha256": np.asarray(
            _canonical_sha256(
                [
                    (sample["sample_id"], sample["transcripts_sha256"])
                    for sample in resolved_provenance
                ]
            )
        ),
        "provenance_json": np.asarray(json.dumps(provenance, sort_keys=True)),
    }
    if rows.cellvit_names is not None:
        if len(rows.cellvit) != observations:
            raise ValueError("CellViT sensitivity rows differ from registered native cells")
        payload["cellvit_sensitivity_features"] = np.asarray(rows.cellvit, dtype=np.float32)
        payload["cellvit_sensitivity_feature_names"] = np.asarray(rows.cellvit_names)
    _write_npz(output_path, payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    build_source(
        args.protocol,
        args.data_root,
        args.model_dir,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
