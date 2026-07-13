#!/usr/bin/env python3
"""Build the pinned HESCAPE 55-um regional source for the oracle ridge probe.

This builder deliberately does not create nucleus or cell observations.  HESCAPE lung rows are
released image/pseudo-spot pairs whose RNA target is a 55-um sum-pooled Xenium region.  Reference
and evaluation pools are separated spatially before any image features are consumed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

DATASET_REPO = "Peng-AI/hescape-pyarrow"
DATASET_REVISION = "a9abd572aa2c740e6f9abee4e197d66b652c6532"
DATASET_CONFIG = "human-lung-healthy-panel"
OFFICIAL_CODE_REVISION = "bd5470480594f1b11d21633e11aaaffe2cfbd4d4"
MODEL_REPO = "MahmoodLab/UNI2-h"
MODEL_REVISION = "d517a8dd47902dd7c308b3c36f63bce47e7b9a43"
MODEL_CONFIG_SHA256 = "8b207fbff3e34884fd225b2d52e8ff51b728a1d0ac2fe8bb2b8db8011308ac98"
MODEL_CHECKPOINT_SHA256 = "6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e"
MODEL_CHECKPOINT_BYTES = 2_725_669_217
PARQUET_MANIFEST_SHA256 = "4a87296cb7041cf577d3dd8b9210e3d65c3f43016cc267d609d93a5bd7ece389"
PARQUET_TOTAL_BYTES = 97_181_626_515
EXPECTED_SHARDS = 195
EXPECTED_ROWS = 56_689
EXPECTED_SECTIONS = 20
EXPECTED_DONORS = 15
EXPECTED_GENES = 343
FEATURE_WIDTH = 1_536
SOURCE_SCHEMA = "heir.hescape_regional_source.v1"
PLAN_SCHEMA = "heir.morphology_ridge_preparation_plan.v1"
CSV_SHA256 = {
    "train.csv": "5124fbb6f235648ba6810b7de77d925fee4022ec3225659774d1bafed017b556",
    "val.csv": "8a7ef61402d044f27fe8fa4326094b25c18abad688be98ff704d78bb32e5de32",
    "test.csv": "97aedeb29d350aca96db6399002958e266f07f9a30eb594de0af88977f115398",
}
GENE_REFERENCE_SHA256 = "77e4c5ba268601ea63a139672873d9c21f74f9e8b25c6e5314945249fd6b3484"
UNI2_MEAN = (0.485, 0.456, 0.406)
UNI2_STD = (0.229, 0.224, 0.225)
SOURCE_ANNOTATION_SHA256 = "4c4b0d159569a3ff86753b700f28a14807d00639788cb7aba2d675738e243423"
SECTION_TO_TRUE_DONOR = {
    "NCBI856": "VUILD96",
    "NCBI857": "VUILD96",
    "NCBI858": "VUILD91",
    "NCBI859": "VUILD91",
    "NCBI860": "VUILD78",
    "NCBI861": "VUILD78",
    "NCBI864": "VUILD115",
    "NCBI865": "VUILD110",
    "NCBI866": "VUILD107",
    "NCBI867": "VUILD106",
    "NCBI870": "VUILD105",
    "NCBI873": "VUILD102",
    "NCBI875": "VUHD116",
    "NCBI876": "VUHD116",
    "NCBI879": "VUHD069",
    "NCBI880": "TILD175",
    "NCBI881": "TILD117",
    "NCBI882": "TILD117",
    "NCBI883": "THD0011",
    "NCBI884": "THD0008",
}
DEVELOPMENT_DONORS = (
    "TILD175",
    "VUHD069",
    "VUHD116",
    "VUILD102",
    "VUILD105",
    "VUILD106",
    "VUILD107",
    "VUILD110",
    "VUILD115",
    "VUILD91",
)
LOCKED_DONORS = ("THD0008", "THD0011", "TILD117", "VUILD78", "VUILD96")
COMPOSITION_FEATURE_NAMES = (
    "composition_epithelial",
    "composition_immune",
    "composition_stromal",
    "composition_endothelial",
)
STAIN_FEATURE_NAMES = (
    "rgb_red_mean",
    "rgb_green_mean",
    "rgb_blue_mean",
    "rgb_red_variance",
    "rgb_green_variance",
    "rgb_blue_variance",
    "hematoxylin_mean",
    "hematoxylin_variance",
    "eosin_mean",
    "eosin_variance",
    "edge_density",
    "grayscale_entropy",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_content_sha256(path: Path) -> Tuple[str, str]:
    """Return a byte hash, or a verified Hugging Face content-addressed blob identity.

    Completed ``huggingface_hub`` snapshots are symlinks into a ``blobs`` directory whose
    64-hex filename is the repository's LFS SHA-256.  Re-reading every embedded image solely to
    recompute that hash would double the 97-GB dataset I/O.  Copied/plain files retain the strict
    byte-hash path.
    """

    if path.is_symlink():
        target = path.resolve(strict=True)
        if target.parent.name == "blobs" and re.fullmatch(r"[0-9a-f]{64}", target.name):
            if not target.is_file() or target.stat().st_size != path.stat().st_size:
                raise ValueError("Hugging Face content-addressed blob is incomplete")
            return target.name, "huggingface_lfs_content_address"
    return _sha256_file(path), "byte_sha256"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("HESCAPE protocol is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("HESCAPE protocol must be a JSON object")
    return value


def _validate_protocol(protocol: Mapping[str, object]) -> None:
    exact = {
        "schema": "heir.hescape_regional_protocol.v2",
        "scientific_scope": "regional_pseudospot_exploratory",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_config": DATASET_CONFIG,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
        "parquet_manifest_sha256": PARQUET_MANIFEST_SHA256,
        "parquet_total_bytes": PARQUET_TOTAL_BYTES,
        "source_study": "GSE250346",
        "source_annotation_sha256": SOURCE_ANNOTATION_SHA256,
        "official_hescape_split_is_donor_safe": False,
        "normalization": "log1p_cpm_10000",
        "primary_crop_source_pixels": 512,
        "primary_resize_pixels": 224,
        "technical_covariates": ["log1p_library_size"],
        "minimum_reference_per_donor_niche": 10,
        "minimum_evaluation_per_donor_niche": 10,
    }
    for name, expected in exact.items():
        if protocol.get(name) != expected:
            raise ValueError("HESCAPE protocol %s differs from the pinned regional design" % name)
    if protocol.get("authorizes_nucleus_claim") is not False:
        raise ValueError("HESCAPE regional protocol must explicitly prohibit nucleus claims")
    true_donors = {
        str(section): str(donor)
        for section, donor in dict(protocol.get("section_to_true_donor", {})).items()
    }
    development = tuple(str(value) for value in protocol.get("development_donors", ()))
    locked = tuple(str(value) for value in protocol.get("locked_test_donors", ()))
    if true_donors != SECTION_TO_TRUE_DONOR:
        raise ValueError("HESCAPE section-to-GSE250346-donor identity differs from the pin")
    if development != DEVELOPMENT_DONORS or locked != LOCKED_DONORS:
        raise ValueError("HESCAPE protocol differs from the exact 10/5 true-donor split")
    if set(development) & set(locked) or set(development + locked) != set(true_donors.values()):
        raise ValueError("HESCAPE true donors are not disjoint and exhaustive")
    roles = {donor: "development" for donor in development} | {
        donor: "locked_test" for donor in locked
    }
    if any(
        len({roles[true_donors[section]] for section in sections}) != 1
        for sections in (
            ("NCBI856", "NCBI857"),
            ("NCBI858", "NCBI859"),
            ("NCBI860", "NCBI861"),
            ("NCBI875", "NCBI876"),
            ("NCBI881", "NCBI882"),
        )
    ):
        raise ValueError("paired HESCAPE sections cross true-donor splits")
    markers = protocol.get("dominant_niche_markers")
    if not isinstance(markers, Mapping) or tuple(markers) != (
        "epithelial",
        "immune",
        "stromal",
        "endothelial",
    ):
        raise ValueError("HESCAPE RNA-only dominant-niche ontology is not frozen")
    flattened = [str(gene) for values in markers.values() for gene in values]
    if not flattened or len(flattened) != len(set(flattened)):
        raise ValueError("HESCAPE dominant-niche markers are empty or overlap")
    block = int(protocol.get("block_size_source_pixels", 0))
    roi = int(protocol.get("roi_size_source_pixels", 0))
    guard = int(protocol.get("opposite_pool_guard_source_pixels", 0))
    if block <= 2 * guard or roi <= 0 or block % roi or guard != 512:
        raise ValueError("HESCAPE spatial block/ROI/guard protocol is invalid")
    if (
        float(protocol.get("minimum_niche_score", -1)) < 0
        or float(protocol.get("minimum_niche_margin", -1)) <= 0
    ):
        raise ValueError("HESCAPE dominant-niche exclusion thresholds are invalid")


def _git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "official HESCAPE code checkout is not a readable Git repository"
        ) from error
    return result.stdout.strip()


@dataclass(frozen=True)
class OfficialSection:
    section_id: str
    donor_id: str
    hescape_patient_id: str
    split: str
    disease_state: str


def _load_official_sections(
    metadata_dir: Path, true_donor_map: Mapping[str, str]
) -> Dict[str, OfficialSection]:
    result: Dict[str, OfficialSection] = {}
    required = {
        "dataset_title",
        "gene_panel",
        "id",
        "organ",
        "disease_state",
        "patient",
        "preservation_method",
        "pixel_size_um_estimated",
        "magnification",
    }
    expected_rows = {"train.csv": 12, "val.csv": 4, "test.csv": 4}
    for filename in ("train.csv", "val.csv", "test.csv"):
        path = metadata_dir / filename
        if not path.is_file() or _sha256_file(path) != CSV_SHA256[filename]:
            raise ValueError(
                "official HESCAPE %s is missing or differs from the pinned file" % filename
            )
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or ()) != required:
                raise ValueError("official HESCAPE donor-map columns differ from the release")
            rows = list(reader)
        if len(rows) != expected_rows[filename]:
            raise ValueError("official HESCAPE %s section count differs" % filename)
        for row in rows:
            section = row["id"].strip()
            donor = row["patient"].strip()
            if (
                not re.fullmatch(r"NCBI\d+", section)
                or not re.fullmatch(r"Patient \d+", donor)
                or row["gene_panel"] != "human_lung_healthy_panel"
                or row["organ"] != "Lung"
                or row["preservation_method"] != "FFPE"
                or row["pixel_size_um_estimated"] != "0.2125"
                or row["magnification"] != "40x"
                or row["disease_state"] not in {"Healthy", "Diseased"}
                or section in result
            ):
                raise ValueError("official HESCAPE donor-map identity is ambiguous")
            result[section] = OfficialSection(
                section_id=section,
                donor_id=str(true_donor_map.get(section, "")),
                hescape_patient_id=donor,
                split=filename.removesuffix(".csv"),
                disease_state=row["disease_state"].lower(),
            )
    if set(true_donor_map) != set(result) or any(not row.donor_id for row in result.values()):
        raise ValueError("GSE250346 true-donor map does not cover every HESCAPE section exactly")
    if (
        len(result) != EXPECTED_SECTIONS
        or len({row.donor_id for row in result.values()}) != EXPECTED_DONORS
    ):
        raise ValueError("HESCAPE source identity must contain 20 sections from 15 true donors")
    return result


def _validate_donor_partitions(
    sections: Mapping[str, OfficialSection],
    development_donors: Sequence[str],
    locked_donors: Sequence[str],
) -> None:
    development = set(development_donors)
    locked = set(locked_donors)
    observed = {row.donor_id for row in sections.values()}
    if development | locked != observed or development & locked:
        raise ValueError("frozen HESCAPE donors do not partition the official cohort")
    if len(development) != 10 or len(locked) != 5:
        raise ValueError("frozen HESCAPE split must contain exactly 10/5 true donors")
    section_roles = {
        section: "development" if row.donor_id in development else "locked_test"
        for section, row in sections.items()
    }
    for donor in observed:
        if len({section_roles[key] for key, row in sections.items() if row.donor_id == donor}) != 1:
            raise ValueError("paired sections from one true donor cross frozen splits")


def _read_ordered_genes(path: Path) -> Tuple[str, ...]:
    if not path.is_file() or _sha256_file(path) != GENE_REFERENCE_SHA256:
        raise ValueError("HESCAPE ordered gene reference differs from the pinned official file")
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - exercised only without builder extras
        raise RuntimeError("install HEIR with the hescape optional dependencies") from error
    with h5py.File(path, "r") as archive:
        if "var/_index" not in archive or archive["X"].shape[1] != EXPECTED_GENES:
            raise ValueError("HESCAPE ordered gene reference is malformed")
        genes = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in archive["var/_index"][:]
        )
        if "var/gene_symbol" in archive:
            symbols = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in archive["var/gene_symbol"][:]
            )
            if symbols != genes:
                raise ValueError("HESCAPE gene symbol and expression orders disagree")
    if len(genes) != EXPECTED_GENES or len(set(genes)) != EXPECTED_GENES:
        raise ValueError("HESCAPE gene order must contain 343 unique genes")
    return genes


def _parquet_shards(directory: Path) -> Tuple[Path, ...]:
    paths = tuple(sorted(directory.glob("train-*-of-00195.parquet")))
    expected = tuple("train-%05d-of-00195.parquet" % index for index in range(EXPECTED_SHARDS))
    if tuple(path.name for path in paths) != expected:
        raise ValueError("HESCAPE lung requires the complete pinned set of 195 Parquet shards")
    if any(not path.is_file() for path in paths):
        raise ValueError("a pinned HESCAPE Parquet shard is unavailable")
    return paths


def _class_names_from_hf_metadata(metadata: bytes) -> Tuple[str, ...]:
    try:
        value = json.loads(metadata.decode("utf-8"))
        names = value["info"]["features"]["name"]["names"]
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as error:
        raise ValueError("HESCAPE Parquet lacks its class-label identity metadata") from error
    if isinstance(names, Mapping):
        ordered = tuple(str(names[str(index)]) for index in range(len(names)))
    elif isinstance(names, list):
        ordered = tuple(str(item) for item in names)
    else:
        raise ValueError("HESCAPE section class-label metadata is malformed")
    if len(ordered) != EXPECTED_SECTIONS or len(set(ordered)) != EXPECTED_SECTIONS:
        raise ValueError("HESCAPE section class-label metadata is ambiguous")
    return ordered


@dataclass(frozen=True)
class RawRows:
    section_ids: np.ndarray
    donor_ids: np.ndarray
    disease_states: np.ndarray
    coordinates: np.ndarray
    counts: np.ndarray
    shard_sha256: Tuple[str, ...]
    shard_verification: Tuple[str, ...]


def _read_rows(shards: Sequence[Path], sections: Mapping[str, OfficialSection]) -> RawRows:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - exercised only without builder extras
        raise RuntimeError("install HEIR with the hescape optional dependencies") from error
    names: Optional[Tuple[str, ...]] = None
    section_values = []
    donor_values = []
    disease_values = []
    coordinate_values = []
    count_values = []
    shard_hashes = []
    shard_verification = []
    required_columns = {
        "name",
        "image",
        "gexp",
        "cell_coords",
        "source",
        "diagnosis",
        "cancer",
        "tissue",
        "assay",
        "preservation_method",
        "stain",
        "species",
    }
    for shard in shards:
        schema = parquet.read_schema(shard)
        if not required_columns <= set(schema.names):
            raise ValueError("HESCAPE Parquet schema is incomplete")
        metadata = schema.metadata or {}
        local_names = _class_names_from_hf_metadata(metadata.get(b"huggingface", b""))
        if names is None:
            names = local_names
            if set(names) != set(sections):
                raise ValueError("Parquet section identities differ from official donor maps")
        elif names != local_names:
            raise ValueError("HESCAPE section class-label order differs between shards")
        table = parquet.read_table(
            shard,
            columns=[
                "name",
                "gexp",
                "cell_coords",
                "source",
                "diagnosis",
                "cancer",
                "tissue",
                "assay",
                "preservation_method",
                "stain",
                "species",
            ],
            use_threads=False,
        )
        rows = table.num_rows
        codes = np.asarray(table["name"].to_numpy(), dtype=np.int64)
        if np.any(codes < 0) or np.any(codes >= EXPECTED_SECTIONS):
            raise ValueError("HESCAPE contains an unknown section class label")
        local_sections = np.asarray([local_names[index] for index in codes])
        local_donors = np.asarray([sections[value].donor_id for value in local_sections])
        local_disease = np.asarray([sections[value].disease_state for value in local_sections])
        coordinates = np.asarray(table["cell_coords"].to_pylist(), dtype=np.float64).reshape(
            rows, 2
        )
        counts = np.asarray(table["gexp"].to_pylist(), dtype=np.float32).reshape(
            rows, EXPECTED_GENES
        )
        diagnoses = np.asarray(table["diagnosis"].to_pylist()).astype(str)
        identities_ok = (
            set(table["source"].to_pylist()) == {0}
            and set(table["tissue"].to_pylist()) == {0}
            and set(table["cancer"].to_pylist()) == {False}
            and set(table["assay"].to_pylist()) == {"Xenium"}
            and set(table["preservation_method"].to_pylist()) == {"FFPE"}
            and set(table["stain"].to_pylist()) == {"HnE"}
            and set(table["species"].to_pylist()) == {"Homo sapiens"}
            and np.array_equal(diagnoses, local_disease)
        )
        if (
            not identities_ok
            or not np.isfinite(coordinates).all()
            or not np.isfinite(counts).all()
            or np.any(counts < 0)
        ):
            raise ValueError(
                "HESCAPE row identity or molecular values differ from the lung release"
            )
        section_values.append(local_sections)
        donor_values.append(local_donors)
        disease_values.append(local_disease)
        coordinate_values.append(coordinates)
        count_values.append(counts)
        digest, verification = _verified_content_sha256(shard)
        shard_hashes.append(digest)
        shard_verification.append(verification)
    result = RawRows(
        section_ids=np.concatenate(section_values),
        donor_ids=np.concatenate(donor_values),
        disease_states=np.concatenate(disease_values),
        coordinates=np.concatenate(coordinate_values),
        counts=np.concatenate(count_values),
        shard_sha256=tuple(shard_hashes),
        shard_verification=tuple(shard_verification),
    )
    if len(result.counts) != EXPECTED_ROWS or set(result.section_ids.tolist()) != set(sections):
        raise ValueError("HESCAPE lung row or section count differs from the pinned release")
    identities = _observation_ids(result.section_ids, result.coordinates)
    if len(set(identities.tolist())) != EXPECTED_ROWS:
        raise ValueError("HESCAPE section/coordinate observations are not one-to-one")
    return result


def _observation_ids(sections: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            "%s:%.6f:%.6f" % (section, float(x), float(y))
            for section, (x, y) in zip(sections.astype(str), coordinates)
        ]
    )


def _log_cpm(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("RNA counts must be a finite nonnegative matrix")
    library = values.sum(axis=1, dtype=np.float64)
    valid = library > 0
    normalized = np.zeros_like(values)
    normalized[valid] = np.log1p(values[valid] * (10_000.0 / library[valid, None]))
    return normalized, np.log1p(library)[:, None], valid


@dataclass(frozen=True)
class DominantNicheFit:
    labels: np.ndarray
    scores: np.ndarray
    marker_gene_ids: Tuple[str, ...]
    marker_means: np.ndarray
    marker_scales: np.ndarray


def _fit_dominant_niches(
    log_expression: np.ndarray,
    genes: Sequence[str],
    development_mask: np.ndarray,
    marker_groups: Mapping[str, Sequence[str]],
    *,
    minimum_score: float,
    minimum_margin: float,
) -> DominantNicheFit:
    values = np.asarray(log_expression, dtype=np.float64)
    development = np.asarray(development_mask, dtype=np.bool_)
    if values.ndim != 2 or values.shape[1] != len(genes) or development.shape != (len(values),):
        raise ValueError("dominant-niche inputs are misaligned")
    if not development.any() or development.all():
        raise ValueError("dominant-niche fitting requires development and held-out rows")
    lookup = {gene: index for index, gene in enumerate(genes)}
    ordered_markers = tuple(str(gene) for group in marker_groups.values() for gene in group)
    if len(set(ordered_markers)) != len(ordered_markers) or any(
        gene not in lookup for gene in ordered_markers
    ):
        raise ValueError("every frozen dominant-niche marker must occur once in the gene panel")
    indices = np.asarray([lookup[gene] for gene in ordered_markers], dtype=np.int64)
    marker_values = values[:, indices]
    means = marker_values[development].mean(axis=0, dtype=np.float64)
    scales = marker_values[development].std(axis=0, dtype=np.float64)
    if np.any(scales <= 1.0e-8):
        raise ValueError("a frozen niche marker has no development-donor variation")
    standardized = (marker_values - means) / scales
    scores = np.empty((len(values), len(marker_groups)), dtype=np.float64)
    cursor = 0
    for type_index, group in enumerate(marker_groups.values()):
        width = len(tuple(group))
        scores[:, type_index] = standardized[:, cursor : cursor + width].mean(axis=1)
        cursor += width
    order = np.argsort(scores, axis=1, kind="stable")
    best = order[:, -1]
    best_score = scores[np.arange(len(scores)), best]
    margin = best_score - scores[np.arange(len(scores)), order[:, -2]]
    accepted = (best_score >= minimum_score) & (margin >= minimum_margin)
    labels = np.where(accepted, best, -1).astype(np.int64)
    return DominantNicheFit(
        labels=labels,
        scores=scores,
        marker_gene_ids=ordered_markers,
        marker_means=means,
        marker_scales=scales,
    )


def _spatial_bins(coordinates: np.ndarray, width: float) -> Dict[Tuple[int, int], list[int]]:
    result: Dict[Tuple[int, int], list[int]] = {}
    for index, (x, y) in enumerate(np.asarray(coordinates, dtype=np.float64)):
        key = (int(np.floor(x / width)), int(np.floor(y / width)))
        result.setdefault(key, []).append(index)
    return result


def _local_density(section_ids: np.ndarray, coordinates: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        raise ValueError("density radius must be positive")
    sections = np.asarray(section_ids).astype(str)
    coords = np.asarray(coordinates, dtype=np.float64)
    result = np.zeros(len(coords), dtype=np.float64)
    for section in sorted(set(sections.tolist())):
        selected = np.flatnonzero(sections == section)
        local = coords[selected]
        bins = _spatial_bins(local, radius)
        for local_index, point in enumerate(local):
            key = tuple(np.floor(point / radius).astype(np.int64).tolist())
            neighbors: list[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbors.extend(bins.get((key[0] + dx, key[1] + dy), ()))
            distances = local[np.asarray(neighbors, dtype=np.int64)] - point
            result[selected[local_index]] = max(
                int(np.count_nonzero(np.square(distances).sum(axis=1) <= radius * radius)) - 1,
                0,
            )
    return result


def _pool_for_block(section: str, x_block: int, y_block: int) -> str:
    digest = hashlib.sha256(
        ("heir-hescape-pool-v1\0%s\0%d\0%d" % (section, x_block, y_block)).encode()
    )
    return "reference" if digest.digest()[0] & 1 else "evaluation"


@dataclass(frozen=True)
class SpatialPools:
    block_ids: np.ndarray
    roi_ids: np.ndarray
    roles: np.ndarray
    guard_pass: np.ndarray


def _spatial_pools(
    section_ids: np.ndarray,
    donor_ids: np.ndarray,
    coordinates: np.ndarray,
    *,
    block_size: int,
    roi_size: int,
    guard: float,
) -> SpatialPools:
    sections = np.asarray(section_ids).astype(str)
    donors = np.asarray(donor_ids).astype(str)
    coords = np.asarray(coordinates, dtype=np.float64)
    if (
        coords.shape != (len(sections), 2)
        or donors.shape != sections.shape
        or block_size <= 0
        or roi_size <= 0
        or guard <= 0
    ):
        raise ValueError("spatial-pool inputs are malformed")
    block_xy = np.floor(coords / block_size).astype(np.int64)
    roi_xy = np.floor(coords / roi_size).astype(np.int64)
    block_ids = np.asarray(
        [
            "%s/%s/block_%d_%d" % (donor, section, x, y)
            for donor, section, (x, y) in zip(donors, sections, block_xy)
        ]
    )
    roi_ids = np.asarray(
        [
            "%s/%s/roi_%d_%d" % (donor, section, x, y)
            for donor, section, (x, y) in zip(donors, sections, roi_xy)
        ]
    )
    roles = np.asarray(
        [_pool_for_block(section, int(x), int(y)) for section, (x, y) in zip(sections, block_xy)]
    )
    guard_pass = np.ones(len(coords), dtype=np.bool_)
    for section in sorted(set(sections.tolist())):
        selected = np.flatnonzero(sections == section)
        local = coords[selected]
        local_roles = roles[selected]
        bins = _spatial_bins(local, guard)
        for first, point in enumerate(local):
            key = tuple(np.floor(point / guard).astype(np.int64).tolist())
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for second in bins.get((key[0] + dx, key[1] + dy), ()):
                        if second <= first or local_roles[first] == local_roles[second]:
                            continue
                        if float(np.max(np.abs(point - local[second]))) < guard:
                            guard_pass[selected[first]] = False
                            guard_pass[selected[second]] = False
    for section in sorted(set(sections.tolist())):
        retained = guard_pass & (sections == section)
        if set(roles[retained].tolist()) != {"reference", "evaluation"}:
            raise ValueError("a HESCAPE section lacks both spatially guarded pools")
    return SpatialPools(block_ids=block_ids, roi_ids=roi_ids, roles=roles, guard_pass=guard_pass)


def _coordinate_features(
    sections: np.ndarray, coordinates: np.ndarray, density: np.ndarray
) -> np.ndarray:
    section_ids = np.asarray(sections).astype(str)
    coords = np.asarray(coordinates, dtype=np.float64)
    normalized = np.zeros_like(coords)
    boundary = np.zeros(len(coords), dtype=np.float64)
    for section in sorted(set(section_ids.tolist())):
        selected = section_ids == section
        minimum = coords[selected].min(axis=0)
        maximum = coords[selected].max(axis=0)
        span = maximum - minimum
        if np.any(span <= 0):
            raise ValueError("a HESCAPE section has degenerate coordinates")
        normalized[selected] = (coords[selected] - minimum) / span
        local = normalized[selected]
        boundary[selected] = np.min(
            np.column_stack((local[:, 0], 1 - local[:, 0], local[:, 1], 1 - local[:, 1])),
            axis=1,
        )
    x = normalized[:, 0]
    y = normalized[:, 1]
    return np.column_stack((x, y, x * x, y * y, x * y, np.log1p(density), boundary))


def _supported_strata_mask(
    donors: np.ndarray,
    labels: np.ndarray,
    roles: np.ndarray,
    eligible: np.ndarray,
    *,
    minimum_reference: int,
    minimum_evaluation: int,
) -> np.ndarray:
    """Retain only donor/niche strata with prespecified support in both spatial pools."""

    donor_values = np.asarray(donors).astype(str)
    label_values = np.asarray(labels, dtype=np.int64)
    role_values = np.asarray(roles).astype(str)
    allowed = np.asarray(eligible, dtype=np.bool_)
    if not (
        donor_values.shape == label_values.shape == role_values.shape == allowed.shape
        and minimum_reference > 0
        and minimum_evaluation > 0
    ):
        raise ValueError("HESCAPE support-filter inputs are malformed")
    retained = np.zeros(len(allowed), dtype=np.bool_)
    for donor in sorted(set(donor_values[allowed].tolist())):
        for type_index in sorted(set(label_values[allowed & (donor_values == donor)].tolist())):
            local = allowed & (donor_values == donor) & (label_values == type_index)
            reference = int(np.count_nonzero(local & (role_values == "reference")))
            evaluation = int(np.count_nonzero(local & (role_values == "evaluation")))
            if reference >= minimum_reference and evaluation >= minimum_evaluation:
                retained |= local
    return retained


def _load_uni2(model_dir: Path, checkpoint: Path):
    try:
        import timm
        import torch
    except ImportError as error:  # pragma: no cover - exercised only without builder extras
        raise RuntimeError("install HEIR with the hescape optional dependencies") from error
    config_path = model_dir / "config.json"
    config = _read_json(config_path)
    pretrained = config.get("pretrained_cfg", {})
    if not isinstance(pretrained, Mapping):
        raise ValueError("UNI2-h pretrained configuration is malformed")
    architecture = config.get("architecture")
    if architecture != "vit_giant_patch14_224":
        raise ValueError("UNI2-h architecture identity differs from the pinned model")
    configured_width = config.get("num_features", pretrained.get("num_features", FEATURE_WIDTH))
    if (
        int(configured_width) != FEATURE_WIDTH
        or config.get("num_classes") != 0
        or config.get("global_pool") != "token"
    ):
        raise ValueError("UNI2-h output configuration differs from the direct 1536-vector pin")
    input_size = tuple(pretrained.get("input_size", (3, 224, 224)))
    mean = tuple(float(value) for value in pretrained.get("mean", UNI2_MEAN))
    std = tuple(float(value) for value in pretrained.get("std", UNI2_STD))
    if (
        input_size != (3, 224, 224)
        or pretrained.get("interpolation") != "bilinear"
        or mean != UNI2_MEAN
        or std != UNI2_STD
    ):
        raise ValueError("UNI2-h preprocessing differs from the official 224-pixel transform")
    model = timm.create_model(
        model_name="vit_giant_patch14_224",
        pretrained=False,
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1.0e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if int(getattr(model, "num_features", -1)) != FEATURE_WIDTH:
        raise ValueError("loaded UNI2-h does not expose 1536 direct features")
    return model


def _stain_statistics(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3 or not np.isfinite(values).all():
        raise ValueError("stain statistics require a finite RGB image")
    if values.max() > 1.0 or values.min() < 0.0:
        raise ValueError("stain-statistics RGB values must be scaled to [0,1]")
    rgb_mean = values.mean(axis=(0, 1), dtype=np.float64)
    rgb_variance = values.var(axis=(0, 1), dtype=np.float64)
    optical_density = -np.log(np.maximum(values, 1.0 / 255.0))
    stain_vectors = np.asarray(((0.65, 0.70, 0.29), (0.07, 0.99, 0.11)), dtype=np.float64)
    stain_vectors /= np.linalg.norm(stain_vectors, axis=1, keepdims=True)
    stains = optical_density @ stain_vectors.T
    stain_summary = np.asarray(
        [stains[..., 0].mean(), stains[..., 0].var(), stains[..., 1].mean(), stains[..., 1].var()]
    )
    grayscale = values @ np.asarray((0.299, 0.587, 0.114))
    horizontal = np.diff(grayscale, axis=1, prepend=grayscale[:, :1])
    vertical = np.diff(grayscale, axis=0, prepend=grayscale[:1, :])
    edge_density = float(np.mean(np.hypot(horizontal, vertical) >= 0.1))
    histogram = np.histogram(grayscale, bins=256, range=(0.0, 1.0))[0].astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = float(-np.sum(probabilities * np.log2(probabilities)) / 8.0)
    result = np.concatenate((rgb_mean, rgb_variance, stain_summary, (edge_density, entropy)))
    if result.shape != (len(STAIN_FEATURE_NAMES),) or not np.isfinite(result).all():
        raise RuntimeError("stain-statistics baseline is malformed")
    return result


def _preprocess_image(encoded: bytes, *, crop_pixels: int, resize_pixels: int):
    try:
        import torch
        from PIL import Image
    except ImportError as error:  # pragma: no cover - exercised only without builder extras
        raise RuntimeError("install HEIR with the hescape optional dependencies") from error
    with Image.open(io.BytesIO(encoded)) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width != 1024 or height != 1024 or crop_pixels != 512 or resize_pixels != 224:
            raise ValueError("HESCAPE image/crop identity differs from pinned 1024->512->224")
        left = (width - crop_pixels) // 2
        top = (height - crop_pixels) // 2
        image = image.crop((left, top, left + crop_pixels, top + crop_pixels))
        image = image.resize((resize_pixels, resize_pixels), resample=Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    stain_statistics = _stain_statistics(values.transpose(1, 2, 0))
    tensor = torch.from_numpy(values)
    mean = torch.tensor(UNI2_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(UNI2_STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std, stain_statistics


def _extract_evaluation_features(
    shards: Sequence[Path],
    rows: RawRows,
    accepted_indices: np.ndarray,
    accepted_roles: np.ndarray,
    model_dir: Path,
    *,
    crop_pixels: int,
    resize_pixels: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, Mapping[str, object], str]:
    try:
        import pyarrow.parquet as parquet
        import torch
    except ImportError as error:  # pragma: no cover - exercised only without builder extras
        raise RuntimeError("install HEIR with the hescape optional dependencies") from error
    if not torch.cuda.is_available():
        raise RuntimeError("the pinned UNI2-h extraction requires CUDA")
    if batch_size <= 0:
        raise ValueError("feature batch size must be positive")
    checkpoint = model_dir / "pytorch_model.bin"
    if not checkpoint.is_file() or not (model_dir / "config.json").is_file():
        raise ValueError("pinned UNI2-h config.json/pytorch_model.bin are required")
    if _sha256_file(model_dir / "config.json") != MODEL_CONFIG_SHA256:
        raise ValueError("UNI2-h config.json differs from the pinned model revision")
    checkpoint_sha256 = _sha256_file(checkpoint)
    if (
        checkpoint.stat().st_size != MODEL_CHECKPOINT_BYTES
        or checkpoint_sha256 != MODEL_CHECKPOINT_SHA256
    ):
        raise ValueError("UNI2-h checkpoint differs from the pinned model revision")
    model = _load_uni2(model_dir, checkpoint)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.eval().to(device=device, dtype=torch.float16)
    features = np.zeros((len(accepted_indices), FEATURE_WIDTH), dtype=np.float32)
    stain_features = np.zeros((len(accepted_indices), len(STAIN_FEATURE_NAMES)), dtype=np.float32)
    extracted = np.zeros(len(accepted_indices), dtype=np.bool_)
    source_positions = np.full(EXPECTED_ROWS, -1, dtype=np.int64)
    source_positions[accepted_indices] = np.arange(len(accepted_indices), dtype=np.int64)
    global_offset = 0
    with torch.inference_mode():
        for shard in shards:
            parquet_file = parquet.ParquetFile(shard)
            class_names = _class_names_from_hf_metadata(
                (parquet_file.schema_arrow.metadata or {}).get(b"huggingface", b"")
            )
            for batch in parquet_file.iter_batches(
                batch_size=batch_size, columns=["name", "cell_coords", "image"], use_threads=False
            ):
                batch_rows = batch.num_rows
                global_indices = np.arange(global_offset, global_offset + batch_rows)
                positions = source_positions[global_indices]
                selected = np.flatnonzero(
                    (positions >= 0) & (accepted_roles[np.maximum(positions, 0)] == "evaluation")
                )
                if len(selected):
                    codes = np.asarray(batch.column("name").to_numpy(), dtype=np.int64)[selected]
                    coords = np.asarray(
                        batch.column("cell_coords").to_pylist(), dtype=np.float64
                    ).reshape(batch_rows, 2)[selected]
                    decoded_sections = np.asarray([class_names[int(code)] for code in codes])
                    if not np.array_equal(
                        decoded_sections, rows.section_ids[global_indices[selected]]
                    ) or not np.array_equal(coords, rows.coordinates[global_indices[selected]]):
                        raise ValueError(
                            "HESCAPE image rows changed between molecular and feature passes"
                        )
                    images = batch.column("image").to_pylist()
                    tensors = []
                    stain_rows = []
                    for local_index in selected.tolist():
                        image_value = images[local_index]
                        if image_value.get("path") not in {None, ""} or not image_value.get(
                            "bytes"
                        ):
                            raise ValueError("HESCAPE image payload is not embedded one-to-one")
                        tensor, stain_row = _preprocess_image(
                            image_value["bytes"],
                            crop_pixels=crop_pixels,
                            resize_pixels=resize_pixels,
                        )
                        tensors.append(tensor)
                        stain_rows.append(stain_row)
                    model_input = torch.stack(tensors).to(
                        device=device, dtype=torch.float16, non_blocking=True
                    )
                    output = model(model_input)
                    if not isinstance(output, torch.Tensor) or output.shape != (
                        len(selected),
                        FEATURE_WIDTH,
                    ):
                        raise ValueError("UNI2-h forward is not the direct 1536-vector output")
                    output_array = output.float().cpu().numpy()
                    if not np.isfinite(output_array).all():
                        raise ValueError("UNI2-h produced non-finite direct features")
                    target_positions = positions[selected]
                    features[target_positions] = output_array
                    stain_features[target_positions] = np.asarray(stain_rows, dtype=np.float32)
                    extracted[target_positions] = True
                global_offset += batch_rows
    if global_offset != EXPECTED_ROWS:
        raise ValueError("HESCAPE image pass row count changed")
    expected_extracted = accepted_roles == "evaluation"
    if not np.array_equal(extracted, expected_extracted):
        raise RuntimeError("not every evaluation image received one direct feature")
    if np.any(features[accepted_roles == "reference"]):
        raise RuntimeError("reference-pool images must not be consumed")
    if np.any(stain_features[accepted_roles == "reference"]):
        raise RuntimeError("reference-pool stain statistics must not be consumed")
    device_evidence = {
        "device": "cuda",
        "device_name": torch.cuda.get_device_name(device),
        "inference_dtype": "float16",
        "storage_dtype": "float32",
        "direct_feature_width": FEATURE_WIDTH,
    }
    return features, stain_features, device_evidence, checkpoint_sha256


def _atomic_npz(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
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


def _validate_source_payload(payload: Mapping[str, object]) -> None:
    required = {
        "schema_version",
        "observation_ids",
        "donor_ids",
        "section_ids",
        "disease_states",
        "block_ids",
        "roi_ids",
        "pool_roles",
        "type_labels",
        "frozen_features",
        "stain_features",
        "stain_feature_names",
        "composition_features",
        "composition_feature_names",
        "molecular_targets",
        "coordinate_features",
        "technical_covariates",
        "registration_is_one_to_one",
    }
    if set(payload) != required or np.asarray(payload["schema_version"]).item() != SOURCE_SCHEMA:
        raise ValueError("HESCAPE regional source schema is incompatible with preparation")
    observations = np.asarray(payload["observation_ids"]).astype(str)
    rows = len(observations)
    if not rows or len(set(observations.tolist())) != rows:
        raise ValueError("HESCAPE regional source observations must be nonempty and unique")
    vectors = (
        "donor_ids",
        "section_ids",
        "disease_states",
        "block_ids",
        "roi_ids",
        "pool_roles",
        "type_labels",
        "registration_is_one_to_one",
    )
    if any(np.asarray(payload[name]).shape != (rows,) for name in vectors):
        raise ValueError("HESCAPE regional source vectors are misaligned")
    expected_widths = {
        "frozen_features": FEATURE_WIDTH,
        "stain_features": len(STAIN_FEATURE_NAMES),
        "composition_features": len(COMPOSITION_FEATURE_NAMES),
        "coordinate_features": 7,
        "technical_covariates": 1,
    }
    for name, width in expected_widths.items():
        matrix = np.asarray(payload[name])
        if matrix.shape != (rows, width) or not np.isfinite(matrix).all():
            raise ValueError("HESCAPE regional source %s is malformed" % name)
    targets = np.asarray(payload["molecular_targets"])
    if (
        targets.ndim != 2
        or targets.shape[0] != rows
        or not targets.shape[1]
        or not np.isfinite(targets).all()
    ):
        raise ValueError("HESCAPE regional molecular targets are malformed")
    roles = np.asarray(payload["pool_roles"]).astype(str)
    if set(roles.tolist()) != {"reference", "evaluation"}:
        raise ValueError("HESCAPE regional source requires reference and evaluation pools")
    if np.any(np.asarray(payload["frozen_features"])[roles == "reference"]) or np.any(
        np.asarray(payload["stain_features"])[roles == "reference"]
    ):
        raise ValueError("reference-pool images must remain unused in source observations")
    if (
        tuple(np.asarray(payload["stain_feature_names"]).astype(str)) != STAIN_FEATURE_NAMES
        or tuple(np.asarray(payload["composition_feature_names"]).astype(str))
        != COMPOSITION_FEATURE_NAMES
    ):
        raise ValueError("HESCAPE regional source feature names differ from the frozen protocol")
    if not np.asarray(payload["registration_is_one_to_one"], dtype=np.bool_).all():
        raise ValueError("HESCAPE released patch/expression pairing is not one-to-one")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--official-code-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    protocol_path = args.protocol.expanduser().resolve()
    parquet_dir = args.parquet_dir.expanduser().resolve()
    code_root = args.official_code_root.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    source_output = args.source_output.expanduser().resolve()
    plan_output = args.plan_output.expanduser().resolve()
    inputs = (protocol_path, parquet_dir, code_root, model_dir)
    if any(not path.exists() for path in inputs):
        raise ValueError("every pinned HESCAPE/model input must exist")
    if source_output == plan_output or source_output in inputs or plan_output in inputs:
        raise ValueError("HESCAPE source and plan outputs must be new distinct paths")

    protocol = _read_json(protocol_path)
    _validate_protocol(protocol)
    if _git_revision(code_root) != OFFICIAL_CODE_REVISION:
        raise ValueError("official HESCAPE code revision differs from the pinned checkout")
    metadata_dir = code_root / "data" / DATASET_CONFIG
    true_donor_map = {
        str(section): str(donor) for section, donor in protocol["section_to_true_donor"].items()
    }
    sections = _load_official_sections(metadata_dir, true_donor_map)
    development_donors = tuple(str(value) for value in protocol["development_donors"])
    locked_donors = tuple(str(value) for value in protocol["locked_test_donors"])
    _validate_donor_partitions(sections, development_donors, locked_donors)
    genes = _read_ordered_genes(metadata_dir / "nicheformer_reference.h5ad")
    shards = _parquet_shards(parquet_dir)
    rows = _read_rows(shards, sections)
    shard_manifest = [
        {"file": path.name, "sha256": digest, "bytes": path.stat().st_size}
        for path, digest in zip(shards, rows.shard_sha256)
    ]
    if (
        _canonical_sha256(shard_manifest) != PARQUET_MANIFEST_SHA256
        or sum(row["bytes"] for row in shard_manifest) != PARQUET_TOTAL_BYTES
    ):
        raise ValueError("HESCAPE Parquet shards differ from the checked pinned manifest")

    log_expression, log_library, nonzero_library = _log_cpm(rows.counts)
    development_mask = np.isin(rows.donor_ids, np.asarray(development_donors))
    marker_groups = {
        str(name): tuple(str(gene) for gene in values)
        for name, values in protocol["dominant_niche_markers"].items()
    }
    niche_fit = _fit_dominant_niches(
        log_expression,
        genes,
        development_mask,
        marker_groups,
        minimum_score=float(protocol["minimum_niche_score"]),
        minimum_margin=float(protocol["minimum_niche_margin"]),
    )
    pools = _spatial_pools(
        rows.section_ids,
        rows.donor_ids,
        rows.coordinates,
        block_size=int(protocol["block_size_source_pixels"]),
        roi_size=int(protocol["roi_size_source_pixels"]),
        guard=float(protocol["opposite_pool_guard_source_pixels"]),
    )
    density = _local_density(
        rows.section_ids,
        rows.coordinates,
        float(protocol["opposite_pool_guard_source_pixels"]),
    )
    controls = _coordinate_features(rows.section_ids, rows.coordinates, density)
    initially_eligible = nonzero_library & (niche_fit.labels >= 0) & pools.guard_pass
    supported = _supported_strata_mask(
        rows.donor_ids,
        niche_fit.labels,
        pools.roles,
        initially_eligible,
        minimum_reference=int(protocol["minimum_reference_per_donor_niche"]),
        minimum_evaluation=int(protocol["minimum_evaluation_per_donor_niche"]),
    )
    accepted = initially_eligible & supported
    accepted_indices = np.flatnonzero(accepted)
    if not len(accepted_indices) or np.any(niche_fit.labels[accepted_indices] < 0):
        raise ValueError("no unambiguous, guarded HESCAPE pseudo-spots remain")
    accepted_donors = rows.donor_ids[accepted]
    accepted_labels = niche_fit.labels[accepted]
    accepted_roles = pools.roles[accepted]
    for donor in development_donors + locked_donors:
        selected = accepted_donors == donor
        if not np.any(selected) or set(accepted_roles[selected].tolist()) != {
            "reference",
            "evaluation",
        }:
            raise ValueError("every frozen donor requires both guarded spatial pools")
        for type_index in sorted(set(accepted_labels[selected].tolist())):
            local = selected & (accepted_labels == type_index)
            if set(accepted_roles[local].tolist()) != {"reference", "evaluation"}:
                raise ValueError("an accepted donor/niche lacks a matched independent pool")
    for type_index in range(len(marker_groups)):
        donors = set(accepted_donors[development_mask[accepted] & (accepted_labels == type_index)])
        if len(donors) < 2:
            raise ValueError("every dominant niche needs at least two development donors")

    features, stain_features, cuda_evidence, checkpoint_sha256 = _extract_evaluation_features(
        shards,
        rows,
        accepted_indices,
        accepted_roles,
        model_dir,
        crop_pixels=int(protocol["primary_crop_source_pixels"]),
        resize_pixels=int(protocol["primary_resize_pixels"]),
        batch_size=args.batch_size,
    )
    marker_set = set(niche_fit.marker_gene_ids)
    evaluation_gene_indices = np.asarray(
        [index for index, gene in enumerate(genes) if gene not in marker_set], dtype=np.int64
    )
    evaluation_genes = tuple(genes[index] for index in evaluation_gene_indices)
    if set(evaluation_genes) & marker_set or len(evaluation_genes) + len(marker_set) != len(genes):
        raise RuntimeError("dominant-niche markers were not exactly removed from RNA targets")

    source_payload = {
        "schema_version": np.asarray(SOURCE_SCHEMA),
        "observation_ids": _observation_ids(rows.section_ids, rows.coordinates)[accepted],
        "donor_ids": accepted_donors,
        "section_ids": rows.section_ids[accepted],
        "disease_states": rows.disease_states[accepted],
        "block_ids": pools.block_ids[accepted],
        "roi_ids": pools.roi_ids[accepted],
        "pool_roles": accepted_roles,
        "type_labels": accepted_labels,
        "frozen_features": features,
        "stain_features": stain_features,
        "stain_feature_names": np.asarray(STAIN_FEATURE_NAMES),
        "composition_features": niche_fit.scores[accepted].astype(np.float32),
        "composition_feature_names": np.asarray(COMPOSITION_FEATURE_NAMES),
        "molecular_targets": log_expression[accepted][:, evaluation_gene_indices].astype(
            np.float32
        ),
        "coordinate_features": controls[accepted].astype(np.float32),
        "technical_covariates": log_library[accepted].astype(np.float32),
        "registration_is_one_to_one": np.ones(len(accepted_indices), dtype=np.bool_),
    }
    _validate_source_payload(source_payload)
    _atomic_npz(source_output, source_payload)
    source_sha256 = _sha256_file(source_output)

    registration_identity = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_config": DATASET_CONFIG,
        "rows": EXPECTED_ROWS,
        "shards": shard_manifest,
        "shard_verification_methods": sorted(set(rows.shard_verification)),
        "observation_level": "pseudo_spot_55um",
        "source_study": "GSE250346",
        "section_to_true_donor": true_donor_map,
        "source_annotation_sha256": SOURCE_ANNOTATION_SHA256,
    }
    label_identity = {
        "method": "development_fitted_continuous_marker_zscores_and_dominant_niche_v1",
        "development_donors": development_donors,
        "type_names": tuple(marker_groups),
        "marker_gene_ids": niche_fit.marker_gene_ids,
        "marker_means": niche_fit.marker_means.tolist(),
        "marker_scales": niche_fit.marker_scales.tolist(),
        "minimum_score": float(protocol["minimum_niche_score"]),
        "minimum_margin": float(protocol["minimum_niche_margin"]),
        "gene_reference_sha256": GENE_REFERENCE_SHA256,
        "composition_feature_names": COMPOSITION_FEATURE_NAMES,
    }
    exclusion_identity = {
        "ambiguous_niche_excluded": True,
        "zero_library_excluded": True,
        "opposite_pool_guard_source_pixels": protocol["opposite_pool_guard_source_pixels"],
        "guard_distance": "chebyshev_nonoverlapping_512px_footprints",
        "image_footprint_source_pixels": 512,
        "reference_images_consumed": False,
        "minimum_reference_per_donor_niche": protocol[
            "minimum_reference_per_donor_niche"
        ],
        "minimum_evaluation_per_donor_niche": protocol[
            "minimum_evaluation_per_donor_niche"
        ],
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "source_schema": SOURCE_SCHEMA,
        "source_observations_sha256": source_sha256,
        "development_donors": list(development_donors),
        "locked_test_donors": list(locked_donors),
        "type_names": list(marker_groups),
        "gene_ids": list(evaluation_genes),
        "type_marker_gene_ids": list(niche_fit.marker_gene_ids),
        "technical_covariate_names": ["log1p_library_size"],
        "feature_space_id": "uni2h_direct_1536_hescape_center512_resize224_fp16_v1",
        "feature_checkpoint_sha256": checkpoint_sha256,
        "molecular_space_id": "hescape_log1p_cpm10000_non_niche_marker_genes_v1",
        "label_source_sha256": _canonical_sha256(label_identity),
        "registration_source_sha256": _canonical_sha256(registration_identity),
        "exclusion_policy_sha256": _canonical_sha256(exclusion_identity),
        "registration_method": "released_one_to_one_histology_pseudospot_pair",
        "encoder_name": MODEL_REPO,
        "crop_scale": "full_context",
        "cohort_id": "HESCAPE",
        "cohort_release": DATASET_CONFIG,
        "assay": "Xenium",
        "observation_level": "pseudo_spot_55um",
        "target_construction": "sum_pooled_xenium_transcripts",
        "reference_mode": "simulated_spatially_disjoint_unpaired_rna",
        "scientific_scope": "regional_pseudospot_exploratory",
        "authorizes_nucleus_claim": False,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "official_csv_sha256": CSV_SHA256,
        "ordered_gene_reference_sha256": GENE_REFERENCE_SHA256,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
        "parquet_manifest_sha256": PARQUET_MANIFEST_SHA256,
        "parquet_total_bytes": PARQUET_TOTAL_BYTES,
        "parquet_verification_methods": sorted(set(rows.shard_verification)),
        "source_study": "GSE250346",
        "source_annotation_sha256": SOURCE_ANNOTATION_SHA256,
        "section_to_true_donor": true_donor_map,
        "image_protocol": {
            "source_size_pixels": 1024,
            "center_crop_pixels": 512,
            "resize_pixels": 224,
            "interpolation": "PIL_bilinear",
            "mean": UNI2_MEAN,
            "std": UNI2_STD,
            "official_transform": "Resize(224),ToTensor,Normalize(ImageNet)",
            "reference_pool_images_consumed": False,
        },
        "cuda_feature_extraction": cuda_evidence,
        "label_protocol": label_identity,
        "composition_feature_names": list(COMPOSITION_FEATURE_NAMES),
        "stain_feature_names": list(STAIN_FEATURE_NAMES),
        "stain_protocol": {
            "input": "same_deterministic_center512_resize224_RGB_as_UNI2-h",
            "rgb_statistics": "channel_mean_and_variance",
            "hematoxylin_optical_density_vector": [0.65, 0.70, 0.29],
            "eosin_optical_density_vector": [0.07, 0.99, 0.11],
            "edge_density_threshold": 0.1,
            "entropy": "normalized_256_bin_grayscale_shannon",
            "reference_pool_images_consumed": False,
        },
        "spatial_protocol": {
            "block_size_source_pixels": protocol["block_size_source_pixels"],
            "roi_size_source_pixels": protocol["roi_size_source_pixels"],
            "opposite_pool_guard_source_pixels": protocol["opposite_pool_guard_source_pixels"],
            "image_footprint_source_pixels": 512,
            "overlap_criterion": "Chebyshev center distance >= 512 for opposite pools",
            "pool_assignment": "sha256_section_block_v1",
        },
        "coordinate_feature_names": [
            "section_normalized_x",
            "section_normalized_y",
            "section_normalized_x_squared",
            "section_normalized_y_squared",
            "section_normalized_xy",
            "log1p_local_pseudospot_density_r512",
            "normalized_distance_to_section_bounding_box",
        ],
        "row_counts": {
            "release": EXPECTED_ROWS,
            "zero_library_excluded": int((~nonzero_library).sum()),
            "ambiguous_niche_excluded": int((niche_fit.labels < 0).sum()),
            "opposite_pool_guard_excluded": int((~pools.guard_pass).sum()),
            "unsupported_donor_niche_excluded": int(
                np.count_nonzero(initially_eligible & ~supported)
            ),
            "retained": int(accepted.sum()),
            "retained_reference": int(np.count_nonzero(accepted_roles == "reference")),
            "retained_evaluation": int(np.count_nonzero(accepted_roles == "evaluation")),
        },
        "parquet_shards": shard_manifest,
    }
    scientific_hashes = {
        plan["feature_checkpoint_sha256"],
        plan["label_source_sha256"],
        plan["registration_source_sha256"],
        plan["exclusion_policy_sha256"],
    }
    if len(scientific_hashes) != 4:
        raise RuntimeError("HESCAPE scientific sources are not independently identifiable")
    _atomic_json(plan_output, plan)
    print(
        json.dumps({"source": str(source_output), "plan": str(plan_output), **plan["row_counts"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
