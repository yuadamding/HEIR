#!/usr/bin/env python3
"""Build the matched Chromium-FLEX/Visium regional source for E-MTAB-14560.

The builder keeps the observation level honest: every target row is a Visium spot, not a cell.
It reads Space Ranger outputs directly, reconstructs the filtered UMI matrix from
``molecule_info.h5``, deterministically assigns each unique molecule to one of two disjoint
halves, and refuses to write a source unless the halves reconstruct the released matrix exactly.
Chromium counts and annotations are read from the CELLxGENE H5AD files with h5py alone.

H&E is the immutable base modality.  A 112-um field centred on each registered spot is encoded
with a checksum-bound, frozen H-optimus-1 model on CUDA.  Per-section embedding caches make the long
WSI pass resumable without weakening input or model identity checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import h5py
import numpy as np

from heir.features import (
    EncoderManifest,
    FrozenPatchEncoder,
    create_frozen_encoder,
    load_encoder_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO_ROOT / "configs/natcommun_matched_regional_protocol.json"
DEFAULT_DATA_ROOT = Path("/mnt/seagate/HnE/NatCommun_2025_s41467_025_59005_9")
# Martian creates symlinks inside a pipestance, which exFAT cannot represent.  Space Ranger is
# therefore staged on the local ext4 work volume; only compact immutable products go to exFAT.
DEFAULT_SPACERANGER_ROOT = Path("/storage/HEIR_work/natcommun_spaceranger")
DEFAULT_MODEL_DIR = Path("/mnt/seagate/HnE/pretrained/H-optimus-1")
DEFAULT_ENCODER_MANIFEST = REPO_ROOT / "manifests/encoders/hoptimus1.json"
DEFAULT_ENCODER_PARITY_RECEIPT = DEFAULT_MODEL_DIR / "official_local_parity.json"
ENCODER_PARITY_QUALIFIER = REPO_ROOT / "scripts/qualify_hoptimus1_parity.py"
FROZEN_ENCODER_PARITY_RECEIPT_SHA256 = (
    "a67ca37feae12a3ca444399f12dc983de01283b05f14ffe16adfcdae80a4d761"
)
FROZEN_ENCODER_PARITY_IMPLEMENTATION_SHA256 = (
    "856a3521fa8388787c43bb8cdd8a8faa202c3d3fd980aeac661f134b8e0711d1"
)
FROZEN_ENCODER_RUNTIME_SHA256: Mapping[str, str] = {
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
SECONDARY_ENCODER_MANIFEST = REPO_ROOT / "manifests/encoders/uni2h.json"
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz")

PROTOCOL_SCHEMA = "heir.natcommun_matched_regional_protocol.v1"
SOURCE_SCHEMA = "heir.natcommun_regional_source.v2"
RECEIPT_SCHEMA = "heir.natcommun_regional_source_receipt.v2"
EMBEDDING_CACHE_SCHEMA = "heir.natcommun_hoptimus1_section_cache.v1"
SPLIT_SALT = "heir-natcommun-visium-unique-umi-split-v1"
CROP_WIDTH_UM = 112.0
VISIUM_SPOT_DIAMETER_UM = 55.0
IMAGE_FEATURE_WIDTH = 1536
COORDINATE_FEATURE_NAMES = (
    "section_normalized_x",
    "section_normalized_y",
    "section_normalized_x_squared",
    "section_normalized_y_squared",
    "section_normalized_xy",
)

# These programs are frozen independently of the spatial outcomes.  They are state-oriented rather
# than broad-lineage signatures: lineage/composition is handled separately by the published
# Chromium labels and the H&E routing control.  Reliability filtering remains outer-training-only.
FROZEN_PROGRAMS: Mapping[str, tuple[str, ...]] = {
    "proliferation": (
        "MKI67",
        "TOP2A",
        "PCNA",
        "MCM2",
        "MCM3",
        "MCM4",
        "MCM5",
        "MCM6",
        "MCM7",
        "TYMS",
    ),
    "interferon_response": (
        "STAT1",
        "IFIT1",
        "IFIT2",
        "IFIT3",
        "ISG15",
        "MX1",
        "MX2",
        "OAS1",
        "OAS2",
        "IRF7",
    ),
    "hypoxia_glycolysis": (
        "CA9",
        "VEGFA",
        "SLC2A1",
        "PDK1",
        "BNIP3L",
        "EGLN3",
        "LDHB",
        "PGK1",
        "ALDOC",
        "ENO1",
    ),
    "inflammatory_activation": (
        "IL1B",
        "CXCL8",
        "CCL2",
        "CXCL2",
        "NFKBIA",
        "PTGS2",
        "ICAM1",
        "TNFAIP3",
    ),
    "cellular_stress": (
        "FOS",
        "JUN",
        "JUNB",
        "ATF3",
        "DDIT3",
        "HSPA1A",
        "HSPA1B",
        "DNAJB1",
        "GADD45B",
        "HSP90AA1",
    ),
    "fibrosis_remodeling": (
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "DCN",
        "LUM",
        "SPARC",
        "ACTA2",
        "TAGLN",
        "TIMP1",
        "MMP2",
    ),
    "epithelial_injury": (
        "KRT17",
        "KRT6A",
        "KRT6B",
        "KRT16",
        "KRT19",
        "SFN",
        "CLDN4",
        "LGALS3",
        "TACSTD2",
        "KRT8",
    ),
    "antigen_presentation": (
        "IFI30",
        "CTSS",
        "LAMP3",
        "TAP2",
        "CD74",
        "CIITA",
        "B2M",
        "TAP1",
        "PSMB8",
        "PSMB9",
    ),
}

PROGRAM_CLASSIFICATIONS: Mapping[str, str] = {
    "proliferation": "candidate_within_type_state",
    "interferon_response": "candidate_within_type_state",
    "hypoxia_glycolysis": "candidate_within_type_state",
    "inflammatory_activation": "candidate_within_type_state",
    "cellular_stress": "candidate_within_type_state",
    "fibrosis_remodeling": "context_conditioned_within_type_state",
    "epithelial_injury": "context_conditioned_within_type_state",
    "antigen_presentation": "context_conditioned_within_type_state",
}

# This exact mapping is the scientific identity of the matched experiment.  Merely retaining the
# same donor set is insufficient: a single accidental remap turns the matched arm into a wrong-bank
# arm.  Values are (donor, indication, H5AD cohort, raw H5AD donor, H&E filename).
FROZEN_SECTION_MAP: Mapping[str, tuple[str, str, str, str, str]] = {
    "B1_2": ("B1", "breast", "breast", "7", "B1_2.tif"),
    "B1_4": ("B1", "breast", "breast", "7", "B1_4.tif"),
    "B2_2": ("B2", "breast", "breast", "2", "B2_2.tif"),
    "B3_2": ("B3", "breast", "breast", "0", "B3_2.tif"),
    "B4_2": ("B4", "breast", "breast", "1", "B4_2.tif"),
    "L1_2": ("L1", "lung", "lung", "5", "L1_2.tif"),
    "L1_4": ("L1", "lung", "lung", "5", "L1_4.tif"),
    "L2_2": ("L2", "lung", "lung", "4", "L2_2.tif"),
    "L3_2": ("L3", "lung", "lung", "3", "L3_2.tif"),
    "L4_2": ("L4", "lung", "lung", "6", "L4_2.tif"),
    "D1": ("D1", "dlbcl", "dlbcl", "D1", "D1.tif"),
    "D2": ("D2", "dlbcl", "dlbcl", "D2", "D2.tif"),
    "D3": ("D3", "dlbcl", "dlbcl", "D3", "D3.tif"),
    "D4": ("D4", "dlbcl", "dlbcl", "D4", "D4.tif"),
    "D5": ("D5", "dlbcl", "dlbcl", "D5", "D5.tif"),
    "D6": ("D6", "dlbcl", "dlbcl", "D6", "D6.tif"),
}

# The CELLxGENE donor IDs for breast and lung are curator-generated integers.  The retained cell
# identifiers preserve the original study sample prefix and provide an independent, local check
# that those opaque donor IDs have not been mapped to the wrong matched bank.
FROZEN_H5AD_CELL_PREFIX: Mapping[tuple[str, str], str] = {
    ("breast", "B1"): "B1_",
    ("breast", "B2"): "B2_",
    ("breast", "B3"): "B3_",
    ("breast", "B4"): "B4_",
    ("lung", "L1"): "L1_",
    ("lung", "L2"): "L2_",
    ("lung", "L3"): "L3_",
    ("lung", "L4"): "L4_",
    **{("dlbcl", f"D{index}"): f"DLBCL_{index}_" for index in range(1, 7)},
}


def _ordered_genes() -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for members in FROZEN_PROGRAMS.values():
        for gene in members:
            if gene not in seen:
                seen.add(gene)
                result.append(gene)
    return tuple(result)


SELECTED_GENES = _ordered_genes()
PROGRAM_NAMES = tuple(FROZEN_PROGRAMS)
IMAGE_FEATURE_NAMES = tuple(f"hoptimus1_{index:04d}" for index in range(IMAGE_FEATURE_WIDTH))
BROAD_TRAINING_ONLY_PCA_DIMENSION_RANGE = (20, 50)


@dataclass(frozen=True)
class SparseCSR:
    """Canonical raw-count CSR components without a SciPy runtime dependency."""

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]


def _validate_csr(value: SparseCSR, name: str, *, allow_zeros: bool = False) -> SparseCSR:
    data = np.asarray(value.data)
    indices = np.asarray(value.indices)
    indptr = np.asarray(value.indptr)
    rows, columns = map(int, value.shape)
    if rows < 0 or columns <= 0 or data.dtype != np.int32 or indices.dtype != np.int32:
        raise ValueError(f"{name} CSR has invalid shape/count/index dtype")
    if indptr.dtype != np.int64 or indptr.shape != (rows + 1,):
        raise ValueError(f"{name} CSR indptr is malformed")
    if len(data) != len(indices) or indptr[0] != 0 or indptr[-1] != len(data):
        raise ValueError(f"{name} CSR components are inconsistent")
    if (data < 0).any() or (not allow_zeros and (data == 0).any()):
        raise ValueError(f"{name} CSR counts must be positive raw UMIs")
    if (np.diff(indptr) < 0).any() or (indices < 0).any() or (indices >= columns).any():
        raise ValueError(f"{name} CSR indexes are out of range")
    for row in range(rows):
        local = indices[int(indptr[row]) : int(indptr[row + 1])]
        if len(local) > 1 and np.any(np.diff(local) <= 0):
            raise ValueError(f"{name} CSR column indexes must be unique and sorted per row")
    return value


def _csr_sha256(value: SparseCSR) -> str:
    return _canonical_sha256(
        {
            "shape": list(value.shape),
            "data": _array_sha256(value.data),
            "indices": _array_sha256(value.indices),
            "indptr": _array_sha256(value.indptr),
        }
    )


def _csr_payload(prefix: str, value: SparseCSR) -> Mapping[str, np.ndarray]:
    _validate_csr(value, prefix)
    return {
        f"{prefix}_data": value.data,
        f"{prefix}_indices": value.indices,
        f"{prefix}_indptr": value.indptr,
        f"{prefix}_shape": np.asarray(value.shape, dtype=np.int64),
    }


def _concatenate_csr_rows(values: Sequence[SparseCSR], name: str) -> SparseCSR:
    if not values:
        raise ValueError(f"{name} requires at least one CSR part")
    columns = values[0].shape[1]
    if any(value.shape[1] != columns for value in values):
        raise ValueError(f"{name} CSR parts have inconsistent column counts")
    data = np.concatenate([value.data for value in values]).astype(np.int32, copy=False)
    indices = np.concatenate([value.indices for value in values]).astype(np.int32, copy=False)
    pointers = [np.asarray([0], dtype=np.int64)]
    offset = 0
    rows = 0
    for value in values:
        _validate_csr(value, name)
        pointers.append(value.indptr[1:] + offset)
        offset += len(value.data)
        rows += value.shape[0]
    result = SparseCSR(data, indices, np.concatenate(pointers), (rows, columns))
    return _validate_csr(result, name)


def _subset_csr_rows(value: SparseCSR, rows: np.ndarray, name: str) -> SparseCSR:
    """Select ordered CSR rows without materializing a dense spot-by-gene matrix."""

    _validate_csr(value, name)
    selected = np.asarray(rows, dtype=np.int64)
    if (
        selected.ndim != 1
        or (selected < 0).any()
        or (selected >= value.shape[0]).any()
        or (len(selected) > 1 and np.any(np.diff(selected) <= 0))
    ):
        raise ValueError(f"{name} CSR row selection must be unique, sorted, and in range")
    lengths = value.indptr[selected + 1] - value.indptr[selected]
    indptr = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)))
    data = np.empty(int(indptr[-1]), dtype=np.int32)
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    output = 0
    for row in selected:
        start, stop = int(value.indptr[row]), int(value.indptr[row + 1])
        length = stop - start
        data[output : output + length] = value.data[start:stop]
        indices[output : output + length] = value.indices[start:stop]
        output += length
    return _validate_csr(SparseCSR(data, indices, indptr, (len(selected), value.shape[1])), name)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _decode(values: object) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8")
    if array.dtype.kind == "O":
        return np.asarray(
            [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in array],
            dtype=str,
        )
    return array.astype(str)


def _constant_text(rows: int, value: object) -> np.ndarray:
    """Return a constant Unicode vector without NumPy's ``dtype=str`` U1 truncation."""

    if rows < 0:
        raise ValueError("constant text row count cannot be negative")
    text = str(value)
    if not text:
        raise ValueError("constant text value cannot be empty")
    return np.full(rows, text, dtype=f"<U{len(text)}")


def _read_h5ad_vector(group: h5py.Group, name: str) -> np.ndarray:
    node = group[name]
    if isinstance(node, h5py.Dataset):
        return _decode(node[:]) if node.dtype.kind in "SO" else np.asarray(node[:])
    if not isinstance(node, h5py.Group) or "categories" not in node or "codes" not in node:
        raise ValueError(f"unsupported H5AD vector encoding for {group.name}/{name}")
    categories_node = node["categories"]
    if not isinstance(categories_node, h5py.Dataset):
        raise ValueError(f"nested H5AD categories are unsupported for {group.name}/{name}")
    categories = _decode(categories_node[:])
    codes = np.asarray(node["codes"][:], dtype=np.int64)
    result = np.full(len(codes), "", dtype=f"<U{max(1, max(map(len, categories), default=1))}")
    valid = (codes >= 0) & (codes < len(categories))
    result[valid] = categories[codes[valid]]
    return result


def _load_protocol(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("NatCommun protocol is not valid JSON") from error
    if not isinstance(value, Mapping) or value.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("NatCommun protocol schema is unsupported")
    if (
        value.get("analysis_scope") != "retrospective_regional_non_authorizing"
        or value.get("observation_level") != "Visium_v2_spot_regional"
        or value.get("primary_endpoints")
        != [
            "reliable_fixed_molecular_programs",
            "training_donor_only_PCA_20_to_50",
            "total_expression",
            "within_type_residual_state",
        ]
        or value.get("bank_conditions")
        != ["natural_composition", "composition_depth_quality_equalized"]
    ):
        raise ValueError(
            "NatCommun protocol scope/endpoints/bank conditions differ from the frozen design"
        )
    iteration = value.get("iteration_gate")
    coverage = value.get("reference_coverage")
    encoder = value.get("h_and_e_encoder")
    if (
        not isinstance(iteration, Mapping)
        or iteration.get("one_step_fusion_updates") != 1
        or iteration.get("iterative_rounds_in_first_experiment") != 0
        or not isinstance(coverage, Mapping)
        or coverage.get("major_type_minimum_qualified_cells") != 50
        or coverage.get("unsupported_action") != "adaptive_alpha_to_zero_and_fallback_to_H_only"
        or not isinstance(encoder, Mapping)
        or encoder.get("repository") != "bioptimus/H-optimus-1"
        or encoder.get("revision") != "3592cb220dec7a150c5d7813fb56e68bd57473b9"
        or encoder.get("mode") != "frozen_CUDA_inference_only"
        or encoder.get("fine_tuning") != "prohibited"
    ):
        raise ValueError(
            "NatCommun one-step/coverage/encoder guardrails differ from the frozen design"
        )
    sections = value.get("sections")
    primary = value.get("primary_donors")
    sensitivity = value.get("failed_reference_sensitivity_donors")
    if not isinstance(sections, list) or len(sections) != 16 or not isinstance(primary, list):
        raise ValueError("NatCommun protocol section/donor declarations are malformed")
    section_ids = [str(row.get("section", "")) for row in sections if isinstance(row, Mapping)]
    donors = [str(row.get("donor", "")) for row in sections if isinstance(row, Mapping)]
    if (
        len(section_ids) != 16
        or len(set(section_ids)) != 16
        or any(not value for value in section_ids)
    ):
        raise ValueError("NatCommun protocol section IDs must be 16 unique nonempty values")
    primary_set = {str(item) for item in primary}
    if sensitivity != ["B2"] or "B2" in primary_set or set(donors) != primary_set | {"B2"}:
        raise ValueError("NatCommun protocol must retain B2 only as failed-reference sensitivity")
    for row in sections:
        if not isinstance(row, Mapping):
            raise ValueError("NatCommun protocol section row is not an object")
        required = {
            "donor",
            "indication",
            "section",
            "h_and_e",
            "h5ad",
            "h5ad_donor",
            "primary_eligible",
        }
        if not required <= set(row):
            raise ValueError("NatCommun protocol section row is incomplete")
        expected = str(row["donor"]) in primary_set
        if bool(row["primary_eligible"]) is not expected:
            raise ValueError("NatCommun primary flag differs from the donor-level declaration")
    observed_map = {
        str(row["section"]): (
            str(row["donor"]),
            str(row["indication"]),
            str(row["h5ad"]),
            str(row["h5ad_donor"]),
            str(row["h_and_e"]),
        )
        for row in sections
    }
    if observed_map != dict(FROZEN_SECTION_MAP):
        raise ValueError(
            "NatCommun section-to-donor/H5AD/H&E mapping differs from the frozen matched design"
        )
    return value


def _load_spaceranger_receipt(spaceranger_root: Path, protocol_path: Path) -> Mapping[str, object]:
    path = spaceranger_root / "run_status.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("completed Space Ranger run_status.json is required") from error
    sections = value.get("sections") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "heir.natcommun_spaceranger_run.v1"
        or value.get("status") != "complete"
        or bool(value.get("dry"))
        or value.get("spaceranger_version") != "spaceranger 4.1.0"
        or value.get("protocol_sha256") != _sha256_file(protocol_path)
        or not isinstance(sections, Mapping)
        or set(map(str, sections)) != set(FROZEN_SECTION_MAP)
    ):
        raise ValueError("Space Ranger receipt does not prove one complete frozen 16-section run")
    for section in FROZEN_SECTION_MAP:
        record = sections[section]
        if not isinstance(record, Mapping) or record.get("status") not in {
            "complete",
            "complete_existing",
        }:
            raise ValueError(f"Space Ranger receipt does not mark {section} complete")
    reference = Path(str(value.get("reference", ""))).expanduser().resolve()
    probe_set = Path(str(value.get("probe_set", ""))).expanduser().resolve()
    if (
        reference.name != "refdata-gex-GRCh38-2020-A"
        or probe_set.name != "Visium_Human_Transcriptome_Probe_Set_v2.0_GRCh38-2020-A.csv"
        or not (reference / "reference.json").is_file()
        or not probe_set.is_file()
        or value.get("reference_metadata_sha256") != _sha256_file(reference / "reference.json")
        or value.get("probe_set_sha256") != _sha256_file(probe_set)
    ):
        raise ValueError("Space Ranger receipt reference/probe identities are not reproducible")
    return value


def _fastq_input_provenance(data_root: Path, sample: str, invocation: str) -> Mapping[str, object]:
    """Bind Space Ranger read paths to the downloaded ENA inventory without rereading 218 GiB."""

    raw_root = (data_root / "arrayexpress/E-MTAB-14560/ENA_submitted").resolve()
    report = data_root / "metadata/ERP165490_ena_run_filereport.tsv"
    expected_directories = tuple(
        sorted({path.parent.resolve() for path in raw_root.rglob(f"{sample}*_R1_001.fastq.gz")})
    )
    if not expected_directories:
        raise ValueError(f"downloaded FASTQ directories are missing for {sample}")
    observed_directories = []
    for line in invocation.splitlines():
        stripped = line.strip()
        if stripped.startswith("read_path") and '"' in stripped:
            observed_directories.append(Path(stripped.split('"', 2)[1]).resolve())
    if tuple(sorted(observed_directories)) != expected_directories:
        raise ValueError(
            f"Space Ranger FASTQ read paths differ from downloaded inputs for {sample}"
        )
    submitted_sizes: dict[str, int] = {}
    with report.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            names = [
                Path(value).name for value in str(row.get("submitted_ftp", "")).split(";") if value
            ]
            sizes = [value for value in str(row.get("submitted_bytes", "")).split(";") if value]
            if len(names) != len(sizes):
                raise ValueError("ENA submitted FASTQ paths and byte sizes are misaligned")
            for name, size in zip(names, sizes):
                submitted_sizes[name] = int(size)
    files: list[Mapping[str, object]] = []
    for directory in expected_directories:
        local = sorted(directory.glob(f"{sample}*fastq.gz"))
        read_kinds = {
            kind for path in local for kind in ("R1", "R2", "I1", "I2") if f"_{kind}_" in path.name
        }
        if read_kinds != {"R1", "R2", "I1", "I2"}:
            raise ValueError(f"downloaded FASTQs are incomplete for {sample}: {directory}")
        for path in local:
            expected_bytes = submitted_sizes.get(path.name)
            actual_bytes = path.stat().st_size
            if expected_bytes is None or actual_bytes != expected_bytes:
                raise ValueError(f"downloaded FASTQ size differs from ENA for {path.name}")
            files.append(
                {"path": str(path.resolve()), "bytes": actual_bytes, "ena_run": directory.name}
            )
    return {
        "schema": "heir.natcommun_ena_fastq_inputs.v1",
        "sample": sample,
        "read_paths": [str(path) for path in expected_directories],
        "read_paths_match_invocation": True,
        "ena_inventory_path": str(report),
        "ena_inventory_sha256": _sha256_file(report),
        "files": files,
        "file_path_and_size_identity_sha256": _canonical_sha256(files),
        "all_local_sizes_match_ENA_submitted_bytes": True,
        "content_integrity_additional_evidence": (
            "Space_Ranger_completed_and_compact_matrix_molecule_outputs_are_SHA256_bound"
        ),
    }


def _section_spaceranger_provenance(
    spaceranger_root: Path,
    row: Mapping[str, object],
    data_root: Path,
    run_receipt: Mapping[str, object],
) -> Mapping[str, object]:
    section = str(row["section"])
    pipestance = spaceranger_root / section
    invocation_path = pipestance / "_invocation"
    versions_path = pipestance / "_versions"
    alignment_path = pipestance / "outs/spatial/final_alignment.json"
    alignment_qc_image = pipestance / "outs/spatial/aligned_tissue_image.jpg"
    try:
        invocation = invocation_path.read_text(encoding="utf-8")
        versions = json.loads(versions_path.read_text(encoding="utf-8"))
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Space Ranger provenance files are incomplete for {section}") from error
    if not isinstance(alignment, Mapping) or not alignment_qc_image.is_file():
        raise ValueError(f"Space Ranger alignment outputs are incomplete for {section}")
    processed = data_root / "arrayexpress/E-MTAB-14560/processed_data"
    image = (processed / str(row["h_and_e"])).resolve()
    cytassist = (processed / str(row["cytassist"])).resolve()
    expected_strings = (
        f'sample_id                 = "{section}"',
        f'sample_names:   ["{row["fastq_sample"]}"]',
        f'slide_serial_capture_area = "{row["slide"]}-{row["area"]}"',
        f'tissue_image_paths        = ["{image}"]',
        f'cytassist_image_paths     = ["{cytassist}"]',
        f'reference_path            = "{Path(str(run_receipt["reference"])).resolve()}"',
        f'target_set                = "{Path(str(run_receipt["probe_set"])).resolve()}"',
        "no_bam                    = true",
        "no_secondary_analysis     = true",
    )
    missing = [value for value in expected_strings if value not in invocation]
    if missing or versions.get("pipelines") != "4.1.0":
        raise ValueError(
            f"Space Ranger invocation differs from frozen inputs for {section}: {missing[:3]}"
        )
    fastq_provenance = _fastq_input_provenance(data_root, str(row["fastq_sample"]), invocation)
    return {
        "schema": "heir.natcommun_spaceranger_section_provenance.v1",
        "section": section,
        "invocation_path": str(invocation_path),
        "invocation_sha256": _sha256_file(invocation_path),
        "versions_path": str(versions_path),
        "versions_sha256": _sha256_file(versions_path),
        "spaceranger_pipeline_version": "4.1.0",
        "final_alignment_path": str(alignment_path),
        "final_alignment_sha256": _sha256_file(alignment_path),
        "alignment_qc_image_path": str(alignment_qc_image),
        "alignment_qc_image_sha256": _sha256_file(alignment_qc_image),
        "alignment_visual_review_required_before_exact_image_claims": True,
        "h_and_e_path": str(image),
        "h_and_e_sha256": _sha256_file(image),
        "cytassist_path": str(cytassist),
        "cytassist_sha256": _sha256_file(cytassist),
        "slide_capture_area": f"{row['slide']}-{row['area']}",
        "fastq_sample": str(row["fastq_sample"]),
        "fastq_inputs": fastq_provenance,
        "reference_metadata_sha256": run_receipt["reference_metadata_sha256"],
        "probe_set_sha256": run_receipt["probe_set_sha256"],
        "exact_invocation_fields_verified": True,
    }


def _load_encoder_parity_receipt(
    path: Path, encoder_manifest: EncoderManifest
) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "H-optimus-1 official-vs-local parity receipt is required before inference"
        ) from error
    receipt_sha256 = _sha256_file(path)
    implementation_sha256 = _sha256_file(ENCODER_PARITY_QUALIFIER)
    runtime_sha256 = {
        relative: _sha256_file(REPO_ROOT / relative) for relative in FROZEN_ENCODER_RUNTIME_SHA256
    }
    receipt_runtime = (
        value.get("production_runtime_contract") if isinstance(value, Mapping) else None
    )
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "heir.hoptimus1_official_local_parity.v1"
        or value.get("repository") != "bioptimus/H-optimus-1"
        or value.get("revision") != encoder_manifest.revision
        or value.get("encoder_manifest_sha256") != encoder_manifest.sha256
        or receipt_sha256 != FROZEN_ENCODER_PARITY_RECEIPT_SHA256
        or implementation_sha256 != FROZEN_ENCODER_PARITY_IMPLEMENTATION_SHA256
        or runtime_sha256 != dict(FROZEN_ENCODER_RUNTIME_SHA256)
        or not isinstance(receipt_runtime, Mapping)
        or receipt_runtime.get("code_sha256") != runtime_sha256
        or value.get("implementation_sha256") != implementation_sha256
        or value.get("status") != "passed"
        or value.get("passed") is not True
    ):
        raise ValueError("H-optimus-1 official-vs-local parity has not passed for this manifest")
    probe = receipt_runtime.get("resampling_probe")
    native_pixels = int(round(112.0 / 0.2125))
    probe_input = (
        np.arange(native_pixels * native_pixels * 3, dtype=np.uint32).reshape(
            native_pixels, native_pixels, 3
        )
        % 251
    ).astype(np.uint8)
    probe_output = _resize_hoptimus_batch(probe_input[None], encoder_manifest.input_pixels)[0]
    expected_probe = {
        "input_shape": list(probe_input.shape),
        "input_dtype": str(probe_input.dtype),
        "input_sha256": hashlib.sha256(probe_input.tobytes()).hexdigest(),
        "output_shape": list(probe_output.shape),
        "output_dtype": str(probe_output.dtype),
        "output_sha256": hashlib.sha256(probe_output.tobytes()).hexdigest(),
        "implementation": "Pillow.Image.Resampling.BICUBIC",
        "resampling_count": 1,
    }
    if not isinstance(probe, Mapping) or any(
        probe.get(field) != expected for field, expected in expected_probe.items()
    ):
        raise ValueError("H-optimus-1 parity receipt does not bind production resampling")
    return {
        "status": "passed",
        "schema": "heir.hoptimus1_official_local_parity.v1",
        "receipt_path": str(path),
        "receipt_sha256": receipt_sha256,
        "encoder_manifest_sha256": encoder_manifest.sha256,
        "implementation_sha256": implementation_sha256,
        "runtime_sha256": runtime_sha256,
    }


@dataclass(frozen=True)
class FilteredMatrix:
    barcodes: np.ndarray
    selected_counts: np.ndarray
    broad_counts: SparseCSR
    total_counts: np.ndarray
    selected_feature_ids: tuple[str, ...]
    broad_feature_ids: tuple[str, ...]
    all_gene_feature_ids: tuple[str, ...]


@dataclass(frozen=True)
class BroadGenePanel:
    gene_names: tuple[str, ...]
    ensembl_ids: tuple[str, ...]
    receipt: Mapping[str, object]


def _feature_type_mask(features: h5py.Group, size: int) -> np.ndarray:
    if "feature_type" not in features:
        return np.ones(size, dtype=np.bool_)
    feature_types = _decode(features["feature_type"][:])
    return np.asarray(feature_types == "Gene Expression", dtype=np.bool_)


def _unique_gene_catalog(names: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> Mapping[str, str]:
    occurrences: dict[str, list[str]] = {}
    for index in np.flatnonzero(mask):
        occurrences.setdefault(str(names[index]), []).append(str(ids[index]).split(".")[0])
    candidates = {
        gene: values[0] for gene, values in occurrences.items() if len(values) == 1 and values[0]
    }
    id_counts: dict[str, int] = {}
    for value in candidates.values():
        id_counts[value] = id_counts.get(value, 0) + 1
    return {gene: value for gene, value in candidates.items() if id_counts[value] == 1}


def _broad_gene_panel(
    protocol: Mapping[str, object],
    h5ad_root: Path,
    visium_matrices: Path | Sequence[Path],
) -> BroadGenePanel:
    """Freeze the complete metadata-only common scRNA/Visium gene universe.

    No count value is read here.  Therefore neither held-out ST expression nor held-out scRNA
    expression can influence which genes are available to an outer-training-only PCA.
    """

    files = protocol.get("h5ad_files")
    if not isinstance(files, Mapping):
        raise ValueError("NatCommun protocol h5ad_files is malformed")
    catalogs: list[tuple[str, Mapping[str, str], str]] = []
    for kind in sorted(files):
        path = h5ad_root / str(files[kind])
        with h5py.File(path, "r") as handle:
            var = handle["var"]
            names = _read_h5ad_vector(var, "feature_name").astype(str)
            ids = _read_h5ad_vector(var, "_index").astype(str)
            mask = np.ones(len(names), dtype=np.bool_)
        catalog = _unique_gene_catalog(names, ids, mask)
        catalogs.append(
            (
                f"Chromium_{kind}",
                catalog,
                _canonical_sha256(sorted(catalog.items())),
            )
        )
    matrix_paths = (
        (visium_matrices,) if isinstance(visium_matrices, Path) else tuple(visium_matrices)
    )
    if not matrix_paths:
        raise ValueError("at least one completed Visium matrix is required for common genes")
    for index, matrix_path in enumerate(matrix_paths):
        with h5py.File(matrix_path, "r") as handle:
            matrix = handle["matrix"]
            features = matrix["features"]
            names = _decode(features["name"][:])
            ids = _decode(features["id"][:])
            mask = _feature_type_mask(features, len(names))
        visium_catalog = _unique_gene_catalog(names, ids, mask)
        section = matrix_path.parent.parent.name or str(index)
        catalogs.append(
            (
                f"Visium_{section}",
                visium_catalog,
                _canonical_sha256(sorted(visium_catalog.items())),
            )
        )
    common = set(catalogs[0][1])
    for _, catalog, _ in catalogs[1:]:
        common &= set(catalog)
    consistent = [
        gene for gene in common if len({catalog[gene] for _, catalog, _ in catalogs}) == 1
    ]
    consistent.sort(key=lambda gene: (catalogs[0][1][gene], gene))
    missing_program = [gene for gene in SELECTED_GENES if gene not in set(consistent)]
    if missing_program:
        raise ValueError(
            "fixed program genes are absent or Ensembl-inconsistent across scRNA/Visium: "
            f"{missing_program}"
        )
    if len(consistent) < 256:
        raise ValueError("fewer than 256 metadata-matched genes are available for broad PCA")
    ensembl = tuple(catalogs[0][1][gene] for gene in consistent)
    receipt = {
        "schema": "heir.natcommun_broad_gene_panel.v1",
        "selection": (
            "complete_one_to_one_symbol_Ensembl_consistent_intersection_across_all_inputs"
        ),
        "uses_expression_values": False,
        "uses_spatial_outcomes": False,
        "outer_training_only_variance_selection_and_PCA_required": True,
        "gene_count": len(consistent),
        "gene_names_sha256": _canonical_sha256(consistent),
        "ensembl_ids_sha256": _canonical_sha256(list(ensembl)),
        "catalogs": [
            {"name": name, "unique_gene_count": len(catalog), "catalog_sha256": digest}
            for name, catalog, digest in catalogs
        ],
    }
    return BroadGenePanel(tuple(consistent), ensembl, receipt)


def _read_filtered_matrix(
    path: Path,
    selected_genes: Sequence[str] = SELECTED_GENES,
    broad_genes: Optional[Sequence[str]] = None,
) -> FilteredMatrix:
    """Read selected raw genes and full GEX library sizes from a 10x CSC H5."""

    with h5py.File(path, "r") as handle:
        if "matrix" not in handle:
            raise ValueError("filtered_feature_bc_matrix.h5 has no matrix group")
        matrix = handle["matrix"]
        features = matrix["features"]
        names = _decode(features["name"][:])
        ids = _decode(features["id"][:])
        gene_mask = _feature_type_mask(features, len(names))
        gene_to_rows: dict[str, list[int]] = {}
        for index in np.flatnonzero(gene_mask):
            gene_to_rows.setdefault(str(names[index]), []).append(int(index))
        broad_genes = tuple(selected_genes if broad_genes is None else broad_genes)
        required_genes = tuple(dict.fromkeys((*selected_genes, *broad_genes)))
        missing = [gene for gene in required_genes if gene not in gene_to_rows]
        duplicated = [gene for gene in required_genes if len(gene_to_rows.get(gene, ())) != 1]
        if missing or duplicated:
            raise ValueError(
                "fixed selected genes must occur exactly once in Space Ranger matrix; "
                f"missing={missing}, duplicated={duplicated}"
            )
        selected_rows = np.asarray(
            [gene_to_rows[gene][0] for gene in selected_genes], dtype=np.int64
        )
        broad_rows = np.asarray([gene_to_rows[gene][0] for gene in broad_genes], dtype=np.int64)
        row_to_selected = np.full(len(names), -1, dtype=np.int32)
        row_to_selected[selected_rows] = np.arange(len(selected_rows), dtype=np.int32)
        row_to_broad = np.full(len(names), -1, dtype=np.int32)
        row_to_broad[broad_rows] = np.arange(len(broad_rows), dtype=np.int32)
        barcodes = _decode(matrix["barcodes"][:])
        indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
        indices = matrix["indices"]
        data = matrix["data"]
        shape = tuple(int(value) for value in np.asarray(matrix["shape"][:]))
        if shape != (len(names), len(barcodes)) or len(indptr) != len(barcodes) + 1:
            raise ValueError("10x filtered matrix dimensions are inconsistent")
        selected_counts = np.zeros((len(barcodes), len(selected_genes)), dtype=np.int32)
        total_counts = np.zeros(len(barcodes), dtype=np.int64)
        broad_data_parts: list[np.ndarray] = []
        broad_index_parts: list[np.ndarray] = []
        broad_indptr = np.zeros(len(barcodes) + 1, dtype=np.int64)
        for column in range(len(barcodes)):
            start, stop = int(indptr[column]), int(indptr[column + 1])
            rows = np.asarray(indices[start:stop], dtype=np.int64)
            values = np.asarray(data[start:stop])
            if (
                not np.isfinite(values).all()
                or not np.equal(values, np.rint(values)).all()
                or (values < 0).any()
            ):
                raise ValueError("Space Ranger filtered matrix is not raw nonnegative UMI counts")
            values_i = values.astype(np.int64)
            gex = gene_mask[rows]
            total_counts[column] = int(values_i[gex].sum(dtype=np.int64))
            mapped = row_to_selected[rows]
            keep = mapped >= 0
            if keep.any():
                np.add.at(selected_counts[column], mapped[keep], values_i[keep])
            broad_mapped = row_to_broad[rows]
            broad_keep = broad_mapped >= 0
            if broad_keep.any():
                local_indices = broad_mapped[broad_keep].astype(np.int32)
                local_data = values_i[broad_keep].astype(np.int32)
                positive = local_data > 0
                local_indices, local_data = local_indices[positive], local_data[positive]
                order = np.argsort(local_indices, kind="stable")
                broad_index_parts.append(local_indices[order])
                broad_data_parts.append(local_data[order])
                broad_indptr[column + 1] = broad_indptr[column] + len(order)
            else:
                broad_indptr[column + 1] = broad_indptr[column]
        if (selected_counts < 0).any() or (total_counts > np.iinfo(np.int32).max).any():
            raise ValueError("Space Ranger count range exceeds the compact source dtype")
        broad_counts = SparseCSR(
            np.concatenate(broad_data_parts).astype(np.int32, copy=False)
            if broad_data_parts
            else np.empty(0, dtype=np.int32),
            np.concatenate(broad_index_parts).astype(np.int32, copy=False)
            if broad_index_parts
            else np.empty(0, dtype=np.int32),
            broad_indptr,
            (len(barcodes), len(broad_genes)),
        )
        _validate_csr(broad_counts, "Space Ranger broad counts")
        return FilteredMatrix(
            barcodes=np.asarray(barcodes, dtype=str),
            selected_counts=selected_counts,
            broad_counts=broad_counts,
            total_counts=total_counts.astype(np.int32),
            selected_feature_ids=tuple(str(ids[index]) for index in selected_rows),
            broad_feature_ids=tuple(str(ids[index]) for index in broad_rows),
            all_gene_feature_ids=tuple(str(ids[index]) for index in np.flatnonzero(gene_mask)),
        )


@dataclass(frozen=True)
class TissuePositions:
    barcodes: np.ndarray
    array_row_col: np.ndarray
    pixel_xy: np.ndarray


def _read_tissue_positions(path: Path, filtered_barcodes: Sequence[str]) -> TissuePositions:
    """Read Space Ranger v1/v2 tissue positions in filtered-barcode order."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError("tissue positions file is empty")
    header = rows[0]
    if "barcode" in header:
        names = {name: index for index, name in enumerate(header)}
        required = {
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        }
        if not required <= set(names):
            raise ValueError("tissue positions header is incomplete")
        values = rows[1:]

        def get(row: list[str], name: str) -> str:
            return row[names[name]]
    else:
        if any(len(row) < 6 for row in rows):
            raise ValueError("legacy tissue positions row is incomplete")
        values = rows
        legacy = {
            "barcode": 0,
            "in_tissue": 1,
            "array_row": 2,
            "array_col": 3,
            "pxl_row_in_fullres": 4,
            "pxl_col_in_fullres": 5,
        }

        def get(row: list[str], name: str) -> str:
            return row[legacy[name]]

    by_barcode: dict[str, tuple[int, int, float, float]] = {}
    for row in values:
        barcode = str(get(row, "barcode"))
        if barcode in by_barcode:
            raise ValueError("tissue positions contain a duplicated barcode")
        if int(get(row, "in_tissue")) == 1:
            by_barcode[barcode] = (
                int(get(row, "array_row")),
                int(get(row, "array_col")),
                float(get(row, "pxl_col_in_fullres")),
                float(get(row, "pxl_row_in_fullres")),
            )
    filtered = [str(value) for value in filtered_barcodes]
    if len(set(filtered)) != len(filtered):
        raise ValueError("filtered Space Ranger matrix contains duplicated barcodes")
    missing = sorted(set(filtered) - set(by_barcode))
    extra = sorted(set(by_barcode) - set(filtered))
    if missing or extra:
        raise ValueError(
            "filtered Space Ranger barcodes and in-tissue positions differ; "
            f"missing_positions={missing[:5]}, extra_positions={extra[:5]}"
        )
    ordered = filtered
    array = np.asarray([by_barcode[value][:2] for value in ordered], dtype=np.int32)
    pixels = np.asarray([by_barcode[value][2:] for value in ordered], dtype=np.float64)
    if not np.isfinite(pixels).all():
        raise ValueError("tissue positions contain non-finite registered pixels")
    return TissuePositions(np.asarray(ordered, dtype=str), array, pixels)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    values = (values + np.uint64(0x9E3779B97F4A7C15)) & mask
    values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
    values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
    return values ^ (values >> np.uint64(31))


@dataclass(frozen=True)
class MoleculeSplit:
    half_a: np.ndarray
    half_b: np.ndarray
    broad_half_a: SparseCSR
    broad_half_b: SparseCSR
    total_half_a: np.ndarray
    total_half_b: np.ndarray
    receipt: Mapping[str, object]


def _csr_replace_and_drop_zeros(expected: SparseCSR, data: np.ndarray, name: str) -> SparseCSR:
    values = np.asarray(data, dtype=np.int32)
    if values.shape != expected.data.shape or (values < 0).any():
        raise ValueError(f"{name} CSR replacement counts are malformed")
    kept_data: list[np.ndarray] = []
    kept_indices: list[np.ndarray] = []
    indptr = np.zeros(expected.shape[0] + 1, dtype=np.int64)
    for row in range(expected.shape[0]):
        start, stop = int(expected.indptr[row]), int(expected.indptr[row + 1])
        keep = values[start:stop] > 0
        kept_data.append(values[start:stop][keep])
        kept_indices.append(expected.indices[start:stop][keep])
        indptr[row + 1] = indptr[row] + int(keep.sum())
    result = SparseCSR(
        np.concatenate(kept_data).astype(np.int32, copy=False)
        if kept_data
        else np.empty(0, dtype=np.int32),
        np.concatenate(kept_indices).astype(np.int32, copy=False)
        if kept_indices
        else np.empty(0, dtype=np.int32),
        indptr,
        expected.shape,
    )
    return _validate_csr(result, name)


def _split_molecule_info(
    path: Path,
    target_barcodes: Sequence[str],
    selected_feature_ids: Sequence[str],
    broad_feature_ids: Sequence[str],
    all_gene_feature_ids: Sequence[str],
    expected_selected: np.ndarray,
    expected_broad: SparseCSR,
    expected_total: np.ndarray,
    *,
    section: str,
    salt: str = SPLIT_SALT,
    chunk_size: int = 1_000_000,
) -> MoleculeSplit:
    """Split unique Space Ranger molecules and prove exact filtered-matrix reconstruction."""

    if chunk_size <= 0:
        raise ValueError("molecule chunk size must be positive")
    if not section.strip():
        raise ValueError("section identity is required for an independent molecule split")
    expected_selected = np.asarray(expected_selected)
    expected_total = np.asarray(expected_total)
    if expected_selected.shape != (len(target_barcodes), len(selected_feature_ids)):
        raise ValueError("expected selected counts do not align with split identities")
    if expected_total.shape != (len(target_barcodes),):
        raise ValueError("expected total counts do not align with split identities")
    _validate_csr(expected_broad, "expected broad counts")
    if expected_broad.shape != (len(target_barcodes), len(broad_feature_ids)):
        raise ValueError("expected broad counts do not align with split identities")
    with h5py.File(path, "r") as handle:
        required = {
            "barcode_idx",
            "feature_idx",
            "umi",
            "barcodes",
            "features",
            "gem_group",
        }
        if not required <= set(handle):
            raise ValueError(
                "molecule_info.h5 lacks barcode/GEM-group unique-molecule identity fields"
            )
        molecule_barcodes = _decode(handle["barcodes"][:]).astype(str)

        def split_barcode(value: str) -> tuple[str, int]:
            base, separator, suffix = value.rpartition("-")
            if not separator or not base or not suffix.isdigit() or int(suffix) <= 0:
                raise ValueError(f"10x target barcode lacks a numeric GEM-group suffix: {value}")
            return base, int(suffix)

        molecule_bases = np.asarray(
            [
                value.rpartition("-")[0] if value.rpartition("-")[2].isdigit() else value
                for value in molecule_barcodes
            ],
            dtype=str,
        )
        if len(set(molecule_bases.tolist())) != len(molecule_bases):
            raise ValueError("molecule_info barcode sequence catalog is not unique")
        molecule_base_lookup = {value: index for index, value in enumerate(molecule_bases)}
        barcode_maps: dict[int, np.ndarray] = {}
        missing_targets: list[str] = []
        for target_index, target in enumerate(map(str, target_barcodes)):
            base, gem = split_barcode(target)
            molecule_index = molecule_base_lookup.get(base)
            if molecule_index is None:
                missing_targets.append(target)
                continue
            mapper = barcode_maps.setdefault(
                gem, np.full(len(molecule_barcodes), -1, dtype=np.int32)
            )
            if mapper[molecule_index] >= 0:
                raise ValueError("target barcodes duplicate one sequence/GEM-group identity")
            mapper[molecule_index] = target_index
        if missing_targets:
            raise ValueError(f"molecule info is missing target barcodes: {missing_targets[:5]}")
        features = handle["features"]
        molecule_feature_ids = _decode(features["id"][:])
        selected_by_id = {
            str(value).split(".")[0]: index for index, value in enumerate(selected_feature_ids)
        }
        broad_by_id = {
            str(value).split(".")[0]: index for index, value in enumerate(broad_feature_ids)
        }
        all_gene = {str(value).split(".")[0] for value in all_gene_feature_ids}
        feature_map = np.full(len(molecule_feature_ids), -1, dtype=np.int32)
        broad_feature_map = np.full(len(molecule_feature_ids), -1, dtype=np.int32)
        gene_feature = np.zeros(len(molecule_feature_ids), dtype=np.bool_)
        for index, value in enumerate(molecule_feature_ids):
            key = str(value).split(".")[0]
            if key in all_gene:
                gene_feature[index] = True
            if key in selected_by_id:
                feature_map[index] = selected_by_id[key]
            if key in broad_by_id:
                broad_feature_map[index] = broad_by_id[key]
        if sum(feature_map >= 0) != len(selected_feature_ids):
            raise ValueError("molecule info does not contain every fixed selected feature")
        if sum(broad_feature_map >= 0) != len(broad_feature_ids):
            raise ValueError("molecule info does not contain every broad common feature")
        size = len(handle["umi"])
        if len(handle["barcode_idx"]) != size or len(handle["feature_idx"]) != size:
            raise ValueError("molecule info identity vectors have inconsistent lengths")
        has_library = "library_idx" in handle
        if len(handle["gem_group"]) != size:
            raise ValueError("molecule info GEM-group vector is misaligned")
        if "count" in handle and len(handle["count"]) != size:
            raise ValueError("molecule info read-count vector is misaligned")
        half_a = np.zeros(expected_selected.shape, dtype=np.int32)
        half_b = np.zeros(expected_selected.shape, dtype=np.int32)
        total_a = np.zeros(len(target_barcodes), dtype=np.int32)
        total_b = np.zeros(len(target_barcodes), dtype=np.int32)
        expected_rows = np.repeat(
            np.arange(expected_broad.shape[0], dtype=np.int64), np.diff(expected_broad.indptr)
        )
        expected_keys = expected_rows * expected_broad.shape[1] + expected_broad.indices.astype(
            np.int64
        )
        broad_a_data = np.zeros(len(expected_broad.data), dtype=np.int32)
        broad_b_data = np.zeros(len(expected_broad.data), dtype=np.int32)
        effective_salt = f"{salt}:{section}"
        salt_u64 = np.uint64(
            int.from_bytes(hashlib.sha256(effective_salt.encode()).digest()[:8], "little")
        )
        selected_a_count = selected_b_count = total_molecules = 0
        for start in range(0, size, chunk_size):
            stop = min(size, start + chunk_size)
            barcode_index = np.asarray(handle["barcode_idx"][start:stop], dtype=np.int64)
            feature_index = np.asarray(handle["feature_idx"][start:stop], dtype=np.int64)
            umi = np.asarray(handle["umi"][start:stop], dtype=np.uint64)
            if (
                (barcode_index < 0).any()
                or (barcode_index >= len(molecule_barcodes)).any()
                or (feature_index < 0).any()
                or (feature_index >= len(feature_map)).any()
            ):
                raise ValueError("molecule info contains out-of-range identity indexes")
            if "count" in handle:
                reads = np.asarray(handle["count"][start:stop])
                if (reads <= 0).any():
                    raise ValueError("molecule info contains a zero-read molecule")
            library = (
                np.asarray(handle["library_idx"][start:stop], dtype=np.uint64)
                if has_library
                else np.zeros(stop - start, dtype=np.uint64)
            )
            gem_group = np.asarray(handle["gem_group"][start:stop], dtype=np.uint64)
            spot = np.full(stop - start, -1, dtype=np.int32)
            for gem in np.unique(gem_group):
                mapper = barcode_maps.get(int(gem))
                if mapper is not None:
                    local = gem_group == gem
                    spot[local] = mapper[barcode_index[local]]
            is_gene = gene_feature[feature_index]
            valid = (spot >= 0) & is_gene
            if not valid.any():
                continue
            # A fixed vectorized mixer avoids Python's randomized hash and remains fast on tens
            # of millions of molecules.  Library identity prevents pooled-library UMI aliases.
            key = (
                umi
                ^ (barcode_index.astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93))
                ^ (feature_index.astype(np.uint64) * np.uint64(0xA5A3564E27F8862F))
                ^ ((library + np.uint64(1)) * np.uint64(0x9E3779B185EBCA87))
                ^ ((gem_group + np.uint64(1)) * np.uint64(0x94D049BB133111EB))
                ^ salt_u64
            )
            in_a = (_splitmix64(key) & np.uint64(1)) == 0
            for target, assignment in ((total_a, in_a), (total_b, ~in_a)):
                rows = spot[valid & assignment]
                target += np.bincount(rows, minlength=len(target)).astype(np.int32)
            selected = valid & (feature_map[feature_index] >= 0)
            for target, assignment in ((half_a, in_a), (half_b, ~in_a)):
                keep = selected & assignment
                flat = (
                    spot[keep].astype(np.int64) * len(selected_feature_ids)
                    + feature_map[feature_index[keep]]
                )
                target += (
                    np.bincount(flat, minlength=target.size).reshape(target.shape).astype(np.int32)
                )
            selected_a_count += int(np.sum(selected & in_a))
            selected_b_count += int(np.sum(selected & ~in_a))
            broad = valid & (broad_feature_map[feature_index] >= 0)
            if broad.any():
                broad_keys = spot[broad].astype(np.int64) * expected_broad.shape[
                    1
                ] + broad_feature_map[feature_index[broad]].astype(np.int64)
                slots = np.searchsorted(expected_keys, broad_keys)
                if (slots >= len(expected_keys)).any() or not np.array_equal(
                    expected_keys[slots], broad_keys
                ):
                    raise ValueError(
                        "molecule info broad counts contain a key absent from filtered matrix"
                    )
                assignments = in_a[broad]
                for target, keep in ((broad_a_data, assignments), (broad_b_data, ~assignments)):
                    unique, counts = np.unique(slots[keep], return_counts=True)
                    target[unique] += counts.astype(np.int32)
            total_molecules += int(np.sum(valid))
    reconstructed = half_a.astype(np.int64) + half_b.astype(np.int64)
    total_reconstructed = total_a.astype(np.int64) + total_b.astype(np.int64)
    if not np.array_equal(reconstructed, expected_selected.astype(np.int64)):
        difference = reconstructed - expected_selected.astype(np.int64)
        raise ValueError(
            "unique molecule halves do not reconstruct selected filtered UMI counts exactly; "
            f"absolute_difference={int(np.abs(difference).sum())}"
        )
    if not np.array_equal(total_reconstructed, expected_total.astype(np.int64)):
        difference = total_reconstructed - expected_total.astype(np.int64)
        raise ValueError(
            "unique molecule halves do not reconstruct full filtered GEX library sizes exactly; "
            f"absolute_difference={int(np.abs(difference).sum())}"
        )
    if not np.array_equal(
        broad_a_data.astype(np.int64) + broad_b_data.astype(np.int64),
        expected_broad.data.astype(np.int64),
    ):
        raise ValueError(
            "unique molecule halves do not reconstruct broad common-gene counts exactly"
        )
    broad_half_a = _csr_replace_and_drop_zeros(expected_broad, broad_a_data, "broad half A")
    broad_half_b = _csr_replace_and_drop_zeros(expected_broad, broad_b_data, "broad half B")
    receipt = {
        "schema": "heir.natcommun_unique_umi_split.v2",
        "algorithm": "splitmix64_parity_over_section_barcode_feature_umi_library_gem_group",
        "section": section,
        "salt": effective_salt,
        "gem_group_in_identity": True,
        "barcode_join": "sequence_plus_numeric_GEM_group_suffix",
        "source_semantics": (
            "one molecule_info row is one corrected unique UMI; read count is not expression weight"
        ),
        "target_barcode_count": len(target_barcodes),
        "selected_gene_count": len(selected_feature_ids),
        "selected_half_a_molecules": selected_a_count,
        "selected_half_b_molecules": selected_b_count,
        "total_target_gex_molecules": total_molecules,
        "halves_are_disjoint_by_construction": True,
        "selected_reconstruction_exact": True,
        "total_library_reconstruction_exact": True,
        "broad_common_gene_reconstruction_exact": True,
        "full_counts_sha256": _array_sha256(expected_selected.astype(np.int32)),
        "half_a_sha256": _array_sha256(half_a),
        "half_b_sha256": _array_sha256(half_b),
        "broad_full_csr_sha256": _csr_sha256(expected_broad),
        "broad_half_a_csr_sha256": _csr_sha256(broad_half_a),
        "broad_half_b_csr_sha256": _csr_sha256(broad_half_b),
    }
    return MoleculeSplit(half_a, half_b, broad_half_a, broad_half_b, total_a, total_b, receipt)


@dataclass(frozen=True)
class ChromiumData:
    counts: np.ndarray
    broad_counts: SparseCSR
    total_counts: np.ndarray
    cell_ids: np.ndarray
    donor_ids: np.ndarray
    raw_h5ad_donor_ids: np.ndarray
    sample_ids: np.ndarray
    indication_ids: np.ndarray
    primary_eligible: np.ndarray
    level1: np.ndarray
    level2: np.ndarray
    level3: np.ndarray
    level4: np.ndarray
    n_features: np.ndarray
    percent_mt: np.ndarray
    percent_ribo: np.ndarray
    percent_hb: np.ndarray
    dv200: np.ndarray
    block_age_months: np.ndarray
    gene_ensembl_ids: tuple[str, ...]
    broad_gene_ensembl_ids: tuple[str, ...]
    input_receipts: tuple[Mapping[str, object], ...]


def _numeric_obs(obs: h5py.Group, name: str, rows: int, default: float = np.nan) -> np.ndarray:
    if name not in obs:
        return np.full(rows, default, dtype=np.float32)
    values = np.asarray(_read_h5ad_vector(obs, name), dtype=np.float64)
    if values.shape != (rows,):
        raise ValueError(f"H5AD obs/{name} is misaligned")
    return values.astype(np.float32)


def _csr_selected_counts(
    matrix: h5py.Group,
    selected_columns: np.ndarray,
    broad_columns: Optional[np.ndarray] = None,
    *,
    row_chunk: int = 2048,
) -> tuple[np.ndarray, np.ndarray, SparseCSR]:
    shape = tuple(int(value) for value in np.asarray(matrix.attrs.get("shape", ())))
    if len(shape) != 2 or matrix.attrs.get("encoding-type") != "csr_matrix":
        raise ValueError("H5AD X must be anndata CSR encoding")
    rows, columns = shape
    column_map = np.full(columns, -1, dtype=np.int32)
    column_map[selected_columns] = np.arange(len(selected_columns), dtype=np.int32)
    broad_columns = (
        selected_columns if broad_columns is None else np.asarray(broad_columns, dtype=np.int64)
    )
    broad_map = np.full(columns, -1, dtype=np.int32)
    broad_map[broad_columns] = np.arange(len(broad_columns), dtype=np.int32)
    result = np.zeros((rows, len(selected_columns)), dtype=np.int32)
    totals = np.zeros(rows, dtype=np.int64)
    broad_data_parts: list[np.ndarray] = []
    broad_index_parts: list[np.ndarray] = []
    broad_indptr = np.zeros(rows + 1, dtype=np.int64)
    indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
    if len(indptr) != rows + 1:
        raise ValueError("H5AD CSR indptr is inconsistent")
    for row_start in range(0, rows, row_chunk):
        row_stop = min(rows, row_start + row_chunk)
        start, stop = int(indptr[row_start]), int(indptr[row_stop])
        indices = np.asarray(matrix["indices"][start:stop], dtype=np.int64)
        values = np.asarray(matrix["data"][start:stop])
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.equal(values, np.rint(values)).all()
        ):
            raise ValueError("H5AD X is not raw nonnegative integer Chromium counts")
        values_i = values.astype(np.int64)
        repeats = np.diff(indptr[row_start : row_stop + 1])
        local_rows = np.repeat(np.arange(row_stop - row_start, dtype=np.int64), repeats)
        if len(local_rows) != len(indices) or (indices < 0).any() or (indices >= columns).any():
            raise ValueError("H5AD CSR indexes are malformed")
        np.add.at(totals[row_start:row_stop], local_rows, values_i)
        selected = column_map[indices]
        keep = selected >= 0
        if keep.any():
            np.add.at(
                result[row_start:row_stop], (local_rows[keep], selected[keep]), values_i[keep]
            )
        local_indptr = indptr[row_start : row_stop + 1] - start
        mapped_broad = broad_map[indices]
        for local_row in range(row_stop - row_start):
            local_start, local_stop = int(local_indptr[local_row]), int(local_indptr[local_row + 1])
            local_columns = mapped_broad[local_start:local_stop]
            local_values = values_i[local_start:local_stop]
            broad_keep = (local_columns >= 0) & (local_values > 0)
            retained_columns = local_columns[broad_keep].astype(np.int32)
            retained_values = local_values[broad_keep].astype(np.int32)
            order = np.argsort(retained_columns, kind="stable")
            broad_index_parts.append(retained_columns[order])
            broad_data_parts.append(retained_values[order])
            output_row = row_start + local_row
            broad_indptr[output_row + 1] = broad_indptr[output_row] + len(order)
    if (totals > np.iinfo(np.int32).max).any():
        raise ValueError("Chromium library size exceeds compact source dtype")
    broad = SparseCSR(
        np.concatenate(broad_data_parts).astype(np.int32, copy=False)
        if broad_data_parts
        else np.empty(0, dtype=np.int32),
        np.concatenate(broad_index_parts).astype(np.int32, copy=False)
        if broad_index_parts
        else np.empty(0, dtype=np.int32),
        broad_indptr,
        (rows, len(broad_columns)),
    )
    return result, totals.astype(np.int32), _validate_csr(broad, "Chromium broad counts")


def _read_chromium_h5ads(
    protocol: Mapping[str, object],
    h5ad_root: Path,
    selected_genes: Sequence[str] = SELECTED_GENES,
    broad_panel: Optional[BroadGenePanel] = None,
) -> ChromiumData:
    sections = protocol["sections"]
    files = protocol.get("h5ad_files")
    if not isinstance(files, Mapping):
        raise ValueError("NatCommun protocol h5ad_files is malformed")
    primary = {str(value) for value in protocol["primary_donors"]}
    donor_maps: dict[str, dict[str, str]] = {}
    for row in sections:
        kind = str(row["h5ad"])
        raw = str(row["h5ad_donor"])
        donor = str(row["donor"])
        existing = donor_maps.setdefault(kind, {}).get(raw)
        if existing is not None and existing != donor:
            raise ValueError("one H5AD donor maps to multiple study donors")
        donor_maps[kind][raw] = donor
    parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "counts",
            "total_counts",
            "cell_ids",
            "donor_ids",
            "indication_ids",
            "primary_eligible",
            "raw_h5ad_donor_ids",
            "sample_ids",
            "level1",
            "level2",
            "level3",
            "level4",
            "n_features",
            "percent_mt",
            "percent_ribo",
            "percent_hb",
            "dv200",
            "block_age_months",
        )
    }
    common_ensembl: Optional[tuple[str, ...]] = None
    broad_parts: list[SparseCSR] = []
    receipts: list[Mapping[str, object]] = []
    for kind in sorted(donor_maps):
        if kind not in files:
            raise ValueError(f"protocol lacks H5AD file for {kind}")
        path = h5ad_root / str(files[kind])
        with h5py.File(path, "r") as handle:
            obs, var = handle["obs"], handle["var"]
            genes = _read_h5ad_vector(var, "feature_name").astype(str)
            ensembl = _read_h5ad_vector(var, "_index").astype(str)
            gene_rows: dict[str, list[int]] = {}
            for index, gene in enumerate(genes):
                gene_rows.setdefault(str(gene), []).append(index)
            missing = [gene for gene in selected_genes if gene not in gene_rows]
            duplicated = [gene for gene in selected_genes if len(gene_rows.get(gene, ())) != 1]
            if missing or duplicated:
                raise ValueError(
                    f"fixed selected genes must occur once in {path.name}; "
                    f"missing={missing}, duplicated={duplicated}"
                )
            columns = np.asarray([gene_rows[gene][0] for gene in selected_genes], dtype=np.int64)
            broad_genes = tuple(selected_genes) if broad_panel is None else broad_panel.gene_names
            missing_broad = [gene for gene in broad_genes if len(gene_rows.get(gene, ())) != 1]
            if missing_broad:
                raise ValueError(f"broad common genes differ in {path.name}: {missing_broad[:10]}")
            selected_ensembl = tuple(str(ensembl[index]) for index in columns)
            broad_columns = np.asarray([gene_rows[gene][0] for gene in broad_genes], dtype=np.int64)
            broad_ensembl = tuple(str(ensembl[index]).split(".")[0] for index in broad_columns)
            expected_broad_ensembl = (
                tuple(value.split(".")[0] for value in selected_ensembl)
                if broad_panel is None
                else broad_panel.ensembl_ids
            )
            if broad_ensembl != expected_broad_ensembl:
                raise ValueError(f"broad gene Ensembl identities differ in {path.name}")
            if common_ensembl is None:
                common_ensembl = selected_ensembl
            elif common_ensembl != selected_ensembl:
                raise ValueError(
                    "selected gene Ensembl identities differ across Chromium H5AD files"
                )
            counts, totals, broad_counts = _csr_selected_counts(handle["X"], columns, broad_columns)
            broad_parts.append(broad_counts)
            rows = len(counts)
            raw_donors = _read_h5ad_vector(obs, "donor_id").astype(str)
            unknown = sorted(set(raw_donors) - set(donor_maps[kind]))
            if unknown:
                raise ValueError(f"unmapped {kind} H5AD donors: {unknown}")
            donors = np.asarray([donor_maps[kind][value] for value in raw_donors], dtype=str)
            levels = [
                _read_h5ad_vector(obs, name).astype(str)
                for name in ("Level1", "Level2", "Level3", "Harmonised_Level4")
            ]
            cell_ids = _read_h5ad_vector(obs, "_index").astype(str)
            if len(set(cell_ids)) != rows:
                raise ValueError(f"H5AD cell IDs are not unique within {kind}")
            verified_prefixes: dict[str, str] = {}
            if kind in {"breast", "lung", "dlbcl"}:
                for donor in sorted(set(donors)):
                    prefix = FROZEN_H5AD_CELL_PREFIX.get((kind, donor))
                    if (
                        prefix is None
                        or not np.char.startswith(cell_ids[donors == donor], prefix).all()
                    ):
                        raise ValueError(
                            f"{kind} H5AD cell prefixes contradict the frozen mapping for {donor}"
                        )
                    verified_prefixes[donor] = prefix
            parts["counts"].append(counts)
            parts["total_counts"].append(totals)
            parts["cell_ids"].append(
                np.asarray([f"{kind}:{value}" for value in cell_ids], dtype=str)
            )
            parts["donor_ids"].append(donors)
            parts["raw_h5ad_donor_ids"].append(raw_donors)
            parts["sample_ids"].append(_read_h5ad_vector(obs, "sample_id").astype(str))
            parts["indication_ids"].append(_constant_text(rows, kind))
            parts["primary_eligible"].append(
                np.asarray([value in primary for value in donors], dtype=np.bool_)
            )
            for name, values in zip(("level1", "level2", "level3", "level4"), levels):
                parts[name].append(values)
            parts["n_features"].append(_numeric_obs(obs, "nFeature_RNA", rows))
            parts["percent_mt"].append(_numeric_obs(obs, "percent_mt", rows))
            parts["percent_ribo"].append(_numeric_obs(obs, "percent_ribo", rows))
            parts["percent_hb"].append(_numeric_obs(obs, "percent_hb", rows))
            parts["dv200"].append(_read_h5ad_vector(obs, "DV200_percent").astype(str))
            parts["block_age_months"].append(_read_h5ad_vector(obs, "block_age_months").astype(str))
        receipts.append(
            {
                "h5ad_kind": kind,
                "path": str(path),
                "sha256": _sha256_file(path),
                "cell_count": rows,
                "raw_count_matrix": "X",
                "ambient_corrected_layer_not_used": "layers/SoupX",
                "donor_mapping": donor_maps[kind],
                "cell_identity_prefixes_verified": verified_prefixes,
                "broad_common_gene_count": len(broad_genes),
                "broad_counts_csr_sha256": _csr_sha256(broad_counts),
            }
        )
    merged = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    if common_ensembl is None or len(set(merged["cell_ids"].tolist())) != len(merged["cell_ids"]):
        raise ValueError("Chromium cell identities are not globally unique")
    if set(merged["donor_ids"].tolist()) != primary | {"B2"}:
        raise ValueError("Chromium H5ADs do not cover every declared donor including B2")
    merged_broad = _concatenate_csr_rows(broad_parts, "merged Chromium broad counts")
    return ChromiumData(
        counts=merged["counts"],
        broad_counts=merged_broad,
        total_counts=merged["total_counts"],
        cell_ids=merged["cell_ids"],
        donor_ids=merged["donor_ids"],
        raw_h5ad_donor_ids=merged["raw_h5ad_donor_ids"],
        sample_ids=merged["sample_ids"],
        indication_ids=merged["indication_ids"],
        primary_eligible=merged["primary_eligible"],
        level1=merged["level1"],
        level2=merged["level2"],
        level3=merged["level3"],
        level4=merged["level4"],
        n_features=merged["n_features"],
        percent_mt=merged["percent_mt"],
        percent_ribo=merged["percent_ribo"],
        percent_hb=merged["percent_hb"],
        dv200=merged["dv200"],
        block_age_months=merged["block_age_months"],
        gene_ensembl_ids=common_ensembl,
        broad_gene_ensembl_ids=(
            tuple(value.split(".")[0] for value in common_ensembl)
            if broad_panel is None
            else broad_panel.ensembl_ids
        ),
        input_receipts=tuple(receipts),
    )


def _coordinate_features(section_ids: Sequence[str], pixel_xy: np.ndarray) -> np.ndarray:
    sections = np.asarray(section_ids).astype(str)
    pixels = np.asarray(pixel_xy, dtype=np.float64)
    if pixels.shape != (len(sections), 2) or not np.isfinite(pixels).all():
        raise ValueError("registered pixel coordinates are malformed")
    result = np.zeros((len(sections), len(COORDINATE_FEATURE_NAMES)), dtype=np.float32)
    for section in sorted(set(sections)):
        mask = sections == section
        local = pixels[mask]
        span = np.ptp(local, axis=0)
        normalized = np.divide(
            local - np.min(local, axis=0), span, out=np.zeros_like(local), where=span > 0
        )
        result[mask] = np.column_stack(
            (
                normalized[:, 0],
                normalized[:, 1],
                normalized[:, 0] ** 2,
                normalized[:, 1] ** 2,
                normalized[:, 0] * normalized[:, 1],
            )
        )
    return result


class _TiffRegionReader:
    """Region reader that avoids re-decoding pathological full-image TIFF strips."""

    MAX_WHOLE_IMAGE_BYTES = 4 * 1024**3
    MAX_REGION_CHUNK_BYTES = 64 * 1024**2

    def __init__(self, path: Path):
        try:
            import tifffile
            import zarr
        except ImportError as error:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "install HEIR HEST image dependencies (tifffile and zarr)"
            ) from error
        self._tiff = tifffile.TiffFile(path)
        series = self._tiff.series[0]
        self._axes = str(series.axes)
        # ``series.aszarr()`` exposes a Zarr group for pyramidal DLBCL WSIs.  Explicit level-zero
        # selection produces the array contract used for both pyramids and flat breast/lung TIFFs.
        self._store = series.aszarr(level=0)
        self._array = zarr.open(self._store, mode="r")
        shape = tuple(int(value) for value in self._array.shape)
        if self._axes in {"YXS", "YXC"} and len(shape) == 3 and shape[2] >= 3:
            self.height, self.width = shape[:2]
            self._channel_first = False
        elif self._axes in {"SYX", "CYX"} and len(shape) == 3 and shape[0] >= 3:
            self.height, self.width = shape[1:]
            self._channel_first = True
        else:
            self.close()
            raise ValueError(f"unsupported H&E TIFF axes/shape: {self._axes}/{shape}")
        self.decoded_bytes = int(np.prod(shape, dtype=np.int64)) * int(self._array.dtype.itemsize)
        self.chunk_shape = tuple(int(value) for value in self._array.chunks)
        chunk_bytes = int(np.prod(self.chunk_shape, dtype=np.int64)) * int(
            self._array.dtype.itemsize
        )
        self._whole_image: Optional[np.ndarray] = None
        self._decode_whole_on_first_crop = False
        if chunk_bytes > self.MAX_REGION_CHUNK_BYTES:
            if self.decoded_bytes > self.MAX_WHOLE_IMAGE_BYTES:
                self.close()
                raise MemoryError(
                    "non-tiled H&E would require more than 4 GiB to decode once; create a tiled "
                    "lossless cache before embedding"
                )
            # A Zarr request against a one-strip TIFF decodes the entire image.  Holding that one
            # decoded image is vastly safer than repeating a 0.2--2 GiB decode for every spot.
            self._decode_whole_on_first_crop = True
            self.storage_mode = "whole_image_decoded_once_for_large_strip"
        else:
            self.storage_mode = "tile_or_small_strip_region_reads"

    def crop(self, center_xy: Sequence[float], width: int) -> np.ndarray:
        if width <= 0:
            raise ValueError("H&E crop width must be positive")
        x, y = map(float, center_xy)
        left = int(np.floor(x - width / 2.0))
        top = int(np.floor(y - width / 2.0))
        right, bottom = left + width, top + width
        source_left, source_top = max(0, left), max(0, top)
        source_right, source_bottom = min(self.width, right), min(self.height, bottom)
        output = np.full((width, width, 3), 255, dtype=np.uint8)
        if source_left < source_right and source_top < source_bottom:
            if self._decode_whole_on_first_crop and self._whole_image is None:
                self._whole_image = np.asarray(self._array[:])
            if self._channel_first:
                source = self._whole_image if self._whole_image is not None else self._array
                region = np.asarray(
                    source[:3, source_top:source_bottom, source_left:source_right]
                ).transpose(1, 2, 0)
            else:
                source = self._whole_image if self._whole_image is not None else self._array
                region = np.asarray(source[source_top:source_bottom, source_left:source_right, :3])
            if region.dtype != np.uint8:
                raise ValueError("H&E TIFF must decode to uint8 RGB")
            output[
                source_top - top : source_bottom - top,
                source_left - left : source_right - left,
            ] = region
        return output

    def close(self) -> None:
        self._whole_image = None
        store = getattr(self, "_store", None)
        if store is not None and hasattr(store, "close"):
            store.close()
        tiff = getattr(self, "_tiff", None)
        if tiff is not None:
            tiff.close()

    def __enter__(self) -> "_TiffRegionReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _spot_crop_pixels(scalefactors_path: Path) -> tuple[int, Mapping[str, object]]:
    value = json.loads(scalefactors_path.read_text(encoding="utf-8"))
    diameter = float(value["spot_diameter_fullres"])
    width = int(round(diameter * CROP_WIDTH_UM / VISIUM_SPOT_DIAMETER_UM))
    if not np.isfinite(diameter) or diameter <= 0 or width <= 0:
        raise ValueError("Space Ranger spot diameter is invalid")
    return width, {
        "physical_width_um": CROP_WIDTH_UM,
        "nominal_visium_spot_diameter_um": VISIUM_SPOT_DIAMETER_UM,
        "spot_diameter_fullres_pixels": diameter,
        "crop_width_fullres_pixels": width,
        "registration_coordinates": "tissue_positions pxl_col/row_in_fullres",
        "padding": "white_RGB_255_outside_WSI",
    }


def _resize_hoptimus_batch(patches: np.ndarray, input_pixels: int) -> np.ndarray:
    """Use the same single Pillow-bicubic canvas conversion qualified by parity."""

    values = np.asarray(patches)
    if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ValueError("registered H&E patches must be NHWC uint8 RGB")
    if values.shape[1:3] == (input_pixels, input_pixels):
        return np.ascontiguousarray(values)
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("install Pillow for qualified H-optimus-1 resampling") from error
    return np.stack(
        [
            np.asarray(
                Image.fromarray(patch, mode="RGB").resize(
                    (input_pixels, input_pixels), resample=Image.Resampling.BICUBIC
                ),
                dtype=np.uint8,
            )
            for patch in values
        ]
    )


def _h_and_e_observation_selection(
    image_path: Path,
    barcodes: np.ndarray,
    pixel_xy: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Exclude queries whose registered center has no corresponding H&E pixel.

    This mask uses only registration coordinates and the image canvas, never expression values,
    reference labels, or model outputs.  It is applied identically to every benchmark arm.
    """

    identities = np.asarray(barcodes).astype(str)
    pixels = np.asarray(pixel_xy, dtype=np.float64)
    if pixels.shape != (len(identities), 2) or not np.isfinite(pixels).all():
        raise ValueError("registered H&E observation identities/coordinates are malformed")
    with _TiffRegionReader(image_path) as reader:
        retained = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < reader.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < reader.height)
        )
        dimensions = [reader.width, reader.height]
    if not retained.any():
        raise ValueError("no registered query center overlaps the H&E image canvas")
    excluded = identities[~retained]
    return retained, {
        "schema": "heir.natcommun_h_and_e_observation_selection.v1",
        "criterion": "registered_spot_center_inside_original_H_and_E_canvas",
        "uses_expression_values": False,
        "uses_spatial_outcomes": False,
        "uses_reference_labels": False,
        "applied_identically_to_all_benchmark_arms": True,
        "image_dimensions_pixels": dimensions,
        "input_spot_count": len(identities),
        "retained_spot_count": int(retained.sum()),
        "excluded_spot_count": int((~retained).sum()),
        "retained_mask_sha256": _array_sha256(retained),
        "excluded_barcode_ids": excluded.tolist(),
        "excluded_barcode_ids_sha256": _canonical_sha256(excluded.tolist()),
    }


def _registration_qc(
    reader: _TiffRegionReader, pixel_xy: np.ndarray, crop_width: int
) -> Mapping[str, object]:
    pixels = np.asarray(pixel_xy, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not len(pixels) or not np.isfinite(pixels).all():
        raise ValueError("registered H&E spot centers are malformed")
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < reader.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < reader.height)
    )
    if not inside.all():
        raise ValueError(
            f"{int((~inside).sum())} registered spot centers fall outside the full-resolution H&E"
        )
    left = np.floor(pixels[:, 0] - crop_width / 2.0).astype(np.int64)
    top = np.floor(pixels[:, 1] - crop_width / 2.0).astype(np.int64)
    right, bottom = left + crop_width, top + crop_width
    retained_width = np.maximum(0, np.minimum(reader.width, right) - np.maximum(0, left))
    retained_height = np.maximum(0, np.minimum(reader.height, bottom) - np.maximum(0, top))
    retained_fraction = retained_width * retained_height / float(crop_width * crop_width)
    padding = 1.0 - retained_fraction
    if np.max(padding) > 0.75:
        raise ValueError("a registered H&E crop is more than 75% outside the source image")
    return {
        "image_width_pixels": reader.width,
        "image_height_pixels": reader.height,
        "all_spot_centers_inside_image": True,
        "spots_with_any_padding": int(np.sum(padding > 0)),
        "mean_padding_fraction": float(np.mean(padding)),
        "maximum_padding_fraction": float(np.max(padding)),
        "reader_storage_mode": reader.storage_mode,
        "reader_zarr_chunk_shape": list(reader.chunk_shape),
        "decoded_image_bytes_if_whole_cached": reader.decoded_bytes,
    }


def _section_embeddings(
    *,
    section: str,
    image_path: Path,
    scalefactors_path: Path,
    barcodes: np.ndarray,
    pixel_xy: np.ndarray,
    encoder: FrozenPatchEncoder,
    encoder_manifest: EncoderManifest,
    cache_dir: Path,
    batch_size: int,
    image_sha256: Optional[str] = None,
    encoder_parity: Optional[Mapping[str, object]] = None,
) -> tuple[np.ndarray, Mapping[str, object]]:
    if batch_size <= 0:
        raise ValueError("embedding batch size must be positive")
    width, crop_receipt = _spot_crop_pixels(scalefactors_path)
    with _TiffRegionReader(image_path) as reader:
        registration = _registration_qc(reader, pixel_xy, width)
        resolved_image_sha256 = image_sha256 or _sha256_file(image_path)
        if len(resolved_image_sha256) != 64:
            raise ValueError("H&E image SHA-256 identity is malformed")
        image_identity = {
            "path": str(image_path),
            "bytes": image_path.stat().st_size,
            "sha256": resolved_image_sha256,
        }
        identity = {
            "schema": EMBEDDING_CACHE_SCHEMA,
            "builder_implementation_sha256": _sha256_file(Path(__file__).resolve()),
            "section": section,
            "image": image_identity,
            "scalefactors_path": str(scalefactors_path),
            "scalefactors_sha256": _sha256_file(scalefactors_path),
            "barcodes_sha256": _array_sha256(np.asarray(barcodes, dtype="S")),
            "pixel_xy_sha256": _array_sha256(np.asarray(pixel_xy, dtype=np.float64)),
            "crop": crop_receipt,
            "encoder_resampling": {
                "source_canvas_pixels": [width, width],
                "target_canvas_pixels": [
                    encoder_manifest.input_pixels,
                    encoder_manifest.input_pixels,
                ],
                "implementation": "Pillow bicubic",
                "resampling_count": int(width != encoder_manifest.input_pixels),
                "qualified_against_official_loader": True,
            },
            "registration_qc": registration,
            "encoder_manifest": str(encoder_manifest.path),
            "encoder_manifest_sha256": encoder_manifest.sha256,
            "official_local_parity": (
                dict(encoder_parity)
                if encoder_parity is not None
                else {"status": "fixture_unqualified"}
            ),
            "encoder_id": encoder_manifest.encoder_id,
            "feature_width": encoder_manifest.feature_width,
            "device": "cuda",
            "fine_tuning": "none_frozen_eval_inference",
            "cache_feature_dtype": "float16",
        }
        identity_sha = _canonical_sha256(identity)
        cache_path = cache_dir / f"{section}.npz"
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as archive:
                    cached_identity = str(np.asarray(archive["cache_identity_sha256"]).item())
                    cached_barcodes = archive["barcodes"].astype(str)
                    features = np.asarray(archive["image_features"])
                if (
                    cached_identity == identity_sha
                    and np.array_equal(cached_barcodes, barcodes.astype(str))
                    and features.shape == (len(barcodes), encoder_manifest.feature_width)
                    and features.dtype == np.float16
                    and np.isfinite(features).all()
                ):
                    return features, {
                        **identity,
                        "cache_status": "reused",
                        "cache_path": str(cache_path),
                    }
            except (OSError, KeyError, ValueError):
                pass
        features = np.empty((len(barcodes), encoder_manifest.feature_width), dtype=np.float16)
        for start in range(0, len(barcodes), batch_size):
            stop = min(len(barcodes), start + batch_size)
            patches = _resize_hoptimus_batch(
                np.stack([reader.crop(pixel_xy[index], width) for index in range(start, stop)]),
                encoder_manifest.input_pixels,
            )
            encoded = np.asarray(encoder.encode(patches), dtype=np.float32)
            if (
                encoded.shape != (stop - start, encoder_manifest.feature_width)
                or not np.isfinite(encoded).all()
            ):
                raise ValueError("H-optimus-1 encoder returned malformed section features")
            features[start:stop] = encoded.astype(np.float16)
    if not np.isfinite(features).all():
        raise ValueError("float16 H-optimus-1 cache contains non-finite values")
    _atomic_npz(
        cache_path,
        {
            "cache_identity_sha256": np.asarray(identity_sha),
            "barcodes": np.asarray(barcodes),
            "image_features": features,
            "cache_receipt_json": np.asarray(json.dumps(identity, sort_keys=True, allow_nan=False)),
        },
    )
    return features, {**identity, "cache_status": "created", "cache_path": str(cache_path)}


def _program_membership(genes: Sequence[str] = SELECTED_GENES) -> np.ndarray:
    if not 3 <= len(PROGRAM_NAMES) <= 8:
        raise ValueError("scientific protocol requires three to eight fixed candidate programs")
    index = {str(gene): value for value, gene in enumerate(genes)}
    membership = np.zeros((len(PROGRAM_NAMES), len(genes)), dtype=np.bool_)
    for row, name in enumerate(PROGRAM_NAMES):
        missing = [gene for gene in FROZEN_PROGRAMS[name] if gene not in index]
        if missing:
            raise ValueError(f"frozen program {name} is outside selected genes: {missing}")
        membership[row, [index[gene] for gene in FROZEN_PROGRAMS[name]]] = True
    return membership


def _reference_coverage(chromium: ChromiumData) -> Mapping[str, np.ndarray]:
    keys = sorted(set(zip(chromium.donor_ids.astype(str), chromium.level1.astype(str))))
    donors, types, counts, primary = [], [], [], []
    for donor, cell_type in keys:
        mask = (chromium.donor_ids == donor) & (chromium.level1 == cell_type)
        donors.append(donor)
        types.append(cell_type)
        counts.append(int(mask.sum()))
        primary.append(bool(chromium.primary_eligible[mask][0]))
    return {
        "reference_coverage_donor_ids": np.asarray(donors),
        "reference_coverage_level1_type_ids": np.asarray(types),
        "reference_coverage_cell_counts": np.asarray(counts, dtype=np.int32),
        "reference_coverage_primary_eligible": np.asarray(primary, dtype=np.bool_),
    }


def _manual_pathology_labels(
    path: Optional[Path],
    section_ids: np.ndarray,
    barcode_ids: np.ndarray,
    provenance_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    labels = np.full(len(section_ids), "__unavailable__", dtype="<U64")
    grouped = np.full(len(section_ids), "__unavailable__", dtype="<U64")
    if path is None:
        if provenance_path is not None:
            raise ValueError("pathology provenance was supplied without an annotation CSV")
        return (
            labels,
            grouped,
            {
                "status": "unavailable",
                "reason": (
                    "Supplementary_Data.xlsx SuppData5 contains only section-level aggregate label "
                    "counts, not a barcode-to-pathologist-label mapping; no Visium-derived "
                    "clusters used"
                ),
                "outcome_independent_manual_labels_only": True,
            },
        )
    if provenance_path is None:
        raise ValueError(
            "manual pathology CSV requires a frozen blinded H&E-only provenance manifest"
        )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("manual pathology provenance manifest is invalid") from error
    annotation_sha = _sha256_file(path)
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("schema") != "heir.h_and_e_pathology_annotations.v1"
        or provenance.get("annotation_file_sha256") != annotation_sha
        or provenance.get("source_modality") != "H&E_only"
        or provenance.get("barcode_keyed") is not True
        or provenance.get("blinded_to_spatial_transcriptomics") is not True
        or provenance.get("uses_spatial_expression") is not False
        or provenance.get("uses_Visium_clusters") is not False
        or provenance.get("uses_Cell2location") is not False
    ):
        raise ValueError(
            "manual pathology provenance does not prove blinded outcome-independent H&E labels"
        )
    by_key: dict[tuple[str, str], tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "section_id",
            "barcode_id",
            "pathologist_annotation",
            "grouped_pathology_annotation",
        }
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("manual pathology CSV is missing required identity/label columns")
        for row in reader:
            key = (str(row["section_id"]), str(row["barcode_id"]))
            if key in by_key:
                raise ValueError("manual pathology CSV contains duplicate section/barcode identity")
            by_key[key] = (
                str(row["pathologist_annotation"]),
                str(row["grouped_pathology_annotation"]),
            )
    expected_keys = set(zip(section_ids.astype(str), barcode_ids.astype(str)))
    if set(by_key) != expected_keys:
        raise ValueError("manual pathology CSV must cover exactly every retained section/barcode")
    matched = 0
    for index, key in enumerate(zip(section_ids.astype(str), barcode_ids.astype(str))):
        value = by_key.get(key)
        if value is not None:
            labels[index], grouped[index] = value
            matched += 1
    if matched == 0:
        raise ValueError("manual pathology CSV does not match any retained Visium spot")
    return (
        labels,
        grouped,
        {
            "status": "available_complete",
            "path": str(path),
            "sha256": annotation_sha,
            "matched_spots": matched,
            "total_spots": len(labels),
            "outcome_independent_manual_labels_only": True,
            "visium_derived_clusters_used": False,
            "provenance_manifest": str(provenance_path),
            "provenance_manifest_sha256": _sha256_file(provenance_path),
            "blinded_to_spatial_transcriptomics": True,
        },
    )


def _csr_from_payload(payload: Mapping[str, object], prefix: str) -> SparseCSR:
    shape_array = np.asarray(payload[f"{prefix}_shape"], dtype=np.int64)
    if shape_array.shape != (2,):
        raise ValueError(f"{prefix} CSR shape is malformed")
    return _validate_csr(
        SparseCSR(
            np.asarray(payload[f"{prefix}_data"]),
            np.asarray(payload[f"{prefix}_indices"]),
            np.asarray(payload[f"{prefix}_indptr"]),
            (int(shape_array[0]), int(shape_array[1])),
        ),
        prefix,
    )


def _csr_sum_equals(full: SparseCSR, first: SparseCSR, second: SparseCSR) -> bool:
    if full.shape != first.shape or full.shape != second.shape:
        return False
    for row in range(full.shape[0]):
        fs, fe = int(full.indptr[row]), int(full.indptr[row + 1])
        a_s, a_e = int(first.indptr[row]), int(first.indptr[row + 1])
        b_s, b_e = int(second.indptr[row]), int(second.indptr[row + 1])
        columns = np.union1d(first.indices[a_s:a_e], second.indices[b_s:b_e])
        expected_columns = full.indices[fs:fe]
        if not np.array_equal(columns, expected_columns):
            return False
        summed = np.zeros(len(columns), dtype=np.int64)
        if a_e > a_s:
            summed[np.searchsorted(columns, first.indices[a_s:a_e])] += first.data[a_s:a_e]
        if b_e > b_s:
            summed[np.searchsorted(columns, second.indices[b_s:b_e])] += second.data[b_s:b_e]
        if not np.array_equal(summed, full.data[fs:fe].astype(np.int64)):
            return False
    return True


def _validate_source(payload: Mapping[str, object], protocol: Mapping[str, object]) -> None:
    genes = np.asarray(payload["gene_ids"]).astype(str)
    full = np.asarray(payload["st_counts_full"])
    first = np.asarray(payload["st_counts_half_a"])
    second = np.asarray(payload["st_counts_half_b"])
    rows = len(np.asarray(payload["spot_ids"]))
    if tuple(genes) != SELECTED_GENES or full.shape != (rows, len(genes)):
        raise ValueError("source fixed gene/count dimensions are malformed")
    if full.dtype != np.int32 or first.dtype != np.int32 or second.dtype != np.int32:
        raise ValueError("source spatial counts must retain raw int32 UMI values")
    if not np.array_equal(full.astype(np.int64), first.astype(np.int64) + second.astype(np.int64)):
        raise ValueError("source spatial split halves do not reconstruct full counts")
    total = np.asarray(payload["st_total_umi_counts_full"])
    if not np.array_equal(
        total.astype(np.int64),
        np.asarray(payload["st_total_umi_counts_half_a"], dtype=np.int64)
        + np.asarray(payload["st_total_umi_counts_half_b"], dtype=np.int64),
    ):
        raise ValueError("source total-library split halves do not reconstruct")
    broad_genes = np.asarray(payload["broad_gene_ids"]).astype(str)
    broad_ensembl = np.asarray(payload["broad_gene_ensembl_ids"]).astype(str)
    if (
        len(broad_genes) < 256
        or len(broad_genes) != len(broad_ensembl)
        or len(set(broad_genes.tolist())) != len(broad_genes)
        or len(set(broad_ensembl.tolist())) != len(broad_ensembl)
    ):
        raise ValueError("source broad common-gene identities are malformed")
    broad_full = _csr_from_payload(payload, "st_broad_counts_full")
    broad_a = _csr_from_payload(payload, "st_broad_counts_half_a")
    broad_b = _csr_from_payload(payload, "st_broad_counts_half_b")
    if broad_full.shape != (rows, len(broad_genes)) or not _csr_sum_equals(
        broad_full, broad_a, broad_b
    ):
        raise ValueError("source broad spatial split halves do not reconstruct full counts")
    row_fields = (
        "barcode_ids",
        "donor_ids",
        "section_ids",
        "indication_ids",
        "spot_primary_eligible",
        "pixel_xy",
        "array_row_col",
        "coordinate_features",
        "image_features",
        "spot_pathologist_annotation",
        "spot_grouped_pathology_annotation",
    )
    if any(len(np.asarray(payload[name])) != rows for name in row_fields):
        raise ValueError("source spot fields are not row aligned")
    if np.asarray(payload["image_features"]).shape != (rows, IMAGE_FEATURE_WIDTH):
        raise ValueError("source H-optimus-1 features have the wrong width")
    section_contract = {
        str(row["section"]): (
            str(row["donor"]),
            str(row["indication"]),
            bool(row["primary_eligible"]),
        )
        for row in protocol["sections"]
    }
    section = np.asarray(payload["section_ids"]).astype(str)
    donor = np.asarray(payload["donor_ids"]).astype(str)
    indication = np.asarray(payload["indication_ids"]).astype(str)
    spot_primary = np.asarray(payload["spot_primary_eligible"], dtype=np.bool_)
    if set(section.tolist()) != set(section_contract):
        raise ValueError("source section IDs differ from the frozen section contract")
    for section_id, (expected_donor, expected_indication, expected_eligible) in (
        section_contract.items()
    ):
        mask = section == section_id
        if (
            not mask.any()
            or not np.all(donor[mask] == expected_donor)
            or not np.all(indication[mask] == expected_indication)
            or not np.all(spot_primary[mask] == expected_eligible)
        ):
            raise ValueError("source spot cohort labels differ from the frozen section contract")
    primary = {str(value) for value in protocol["primary_donors"]}
    expected_primary = np.asarray([value in primary for value in donor])
    if not np.array_equal(expected_primary, spot_primary):
        raise ValueError("source primary flags are not donor scoped")
    if "B2" not in set(donor) or spot_primary[donor == "B2"].any():
        raise ValueError("source must retain B2 only as a non-primary sensitivity")
    sc_counts = np.asarray(payload["sc_counts"])
    sc_rows = len(np.asarray(payload["sc_cell_ids"]))
    if sc_counts.shape != (sc_rows, len(genes)) or sc_counts.dtype != np.int32:
        raise ValueError("source Chromium raw counts are malformed")
    sc_donor = np.asarray(payload["sc_donor_ids"]).astype(str)
    sc_indication = np.asarray(payload["sc_indication_ids"]).astype(str)
    sc_primary = np.asarray(payload["sc_primary_eligible"], dtype=np.bool_)
    donor_contract = {
        expected_donor: expected_indication
        for expected_donor, expected_indication, _ in section_contract.values()
    }
    if (
        len(sc_donor) != sc_rows
        or len(sc_indication) != sc_rows
        or len(sc_primary) != sc_rows
        or set(sc_donor.tolist()) != set(donor_contract)
    ):
        raise ValueError("source Chromium cohort labels are malformed")
    for expected_donor, expected_indication in donor_contract.items():
        mask = sc_donor == expected_donor
        if (
            not mask.any()
            or not np.all(sc_indication[mask] == expected_indication)
            or not np.all(sc_primary[mask] == (expected_donor in primary))
        ):
            raise ValueError("source Chromium cohort labels differ from the frozen donor contract")
    sc_broad = _csr_from_payload(payload, "sc_broad_counts")
    if sc_broad.shape != (sc_rows, len(broad_genes)):
        raise ValueError("source Chromium broad CSR is not cell/gene aligned")
    if np.asarray(payload["program_gene_membership"]).shape != (len(PROGRAM_NAMES), len(genes)):
        raise ValueError("source fixed program membership is malformed")
    if tuple(np.asarray(payload["program_classifications"]).astype(str)) != tuple(
        PROGRAM_CLASSIFICATIONS[name] for name in PROGRAM_NAMES
    ):
        raise ValueError("source fixed program classifications are malformed")
    if np.asarray(payload["blank_image_feature_vector"]).shape != (IMAGE_FEATURE_WIDTH,):
        raise ValueError("source blank-image embedding is malformed")


def run(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.expanduser().resolve()
    protocol = _load_protocol(protocol_path)
    data_root = args.data_root.expanduser().resolve()
    spaceranger_root = args.spaceranger_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    cache_dir = (args.cache_dir or output.parent / "section_embedding_cache").expanduser().resolve()
    h5ad_root = data_root / "cellxgene"
    processed = data_root / "arrayexpress/E-MTAB-14560/processed_data"
    spaceranger_receipt = _load_spaceranger_receipt(spaceranger_root, protocol_path)
    visium_matrices = tuple(
        spaceranger_root / str(row["section"]) / "outs/filtered_feature_bc_matrix.h5"
        for row in protocol["sections"]
    )
    missing_matrices = [str(path) for path in visium_matrices if not path.is_file()]
    if missing_matrices:
        raise FileNotFoundError(
            "completed Space Ranger matrices are missing: " + ", ".join(missing_matrices[:3])
        )
    broad_panel = _broad_gene_panel(protocol, h5ad_root, visium_matrices)
    encoder_manifest = load_encoder_manifest(args.encoder_manifest.expanduser().resolve())
    if (
        encoder_manifest.repository != "bioptimus/H-optimus-1"
        or encoder_manifest.feature_width != IMAGE_FEATURE_WIDTH
        or encoder_manifest.fine_tuning != "prohibited"
        or encoder_manifest.input_pixels != 224
        or not np.isclose(
            encoder_manifest.input_pixels * encoder_manifest.model_mpp,
            CROP_WIDTH_UM,
            rtol=0.0,
            atol=1.0e-6,
        )
    ):
        raise ValueError(
            "NatCommun source requires the frozen 1536-wide H-optimus-1 manifest "
            "at its 112-um field"
        )
    if args.device != "cuda":
        raise ValueError("NatCommun scientific source requires CUDA H-optimus-1 inference")
    parity_receipt = _load_encoder_parity_receipt(
        args.encoder_parity_receipt.expanduser().resolve(), encoder_manifest
    )
    secondary_encoder_manifest = load_encoder_manifest(SECONDARY_ENCODER_MANIFEST)
    if secondary_encoder_manifest.repository != "MahmoodLab/UNI2-h":
        raise ValueError("prespecified UNI2-h secondary comparator manifest is malformed")
    chromium = _read_chromium_h5ads(protocol, h5ad_root, SELECTED_GENES, broad_panel)
    encoder = create_frozen_encoder(args.model_dir.expanduser().resolve(), encoder_manifest, "cuda")
    blank_patch = np.full(
        (1, encoder_manifest.input_pixels, encoder_manifest.input_pixels, 3), 255, dtype=np.uint8
    )
    blank_feature = np.asarray(encoder.encode(blank_patch), dtype=np.float32)
    if blank_feature.shape != (1, IMAGE_FEATURE_WIDTH) or not np.isfinite(blank_feature).all():
        raise ValueError("H-optimus-1 blank-patch embedding is malformed")
    blank_receipt = {
        "schema": "heir.natcommun_blank_image_control.v1",
        "input": "constant_white_RGB_uint8_255",
        "input_shape": list(blank_patch.shape[1:]),
        "uses_exact_encoder_preprocessing": True,
        "encoder_manifest_sha256": encoder_manifest.sha256,
        "device": "cuda",
        "feature_sha256_float32": _array_sha256(blank_feature[0]),
        "stored_dtype": "float16",
    }

    spot_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "spot_ids",
            "barcode_ids",
            "donor_ids",
            "section_ids",
            "indication_ids",
            "spot_primary_eligible",
            "pixel_xy",
            "array_row_col",
            "st_counts_full",
            "st_counts_half_a",
            "st_counts_half_b",
            "st_total_umi_counts_full",
            "st_total_umi_counts_half_a",
            "st_total_umi_counts_half_b",
            "image_features",
        )
    }
    section_receipts: list[Mapping[str, object]] = []
    broad_full_parts: list[SparseCSR] = []
    broad_half_a_parts: list[SparseCSR] = []
    broad_half_b_parts: list[SparseCSR] = []
    for row in protocol["sections"]:
        section = str(row["section"])
        outs = spaceranger_root / section / "outs"
        matrix_path = outs / "filtered_feature_bc_matrix.h5"
        molecule_path = outs / "molecule_info.h5"
        positions_path = outs / "spatial/tissue_positions.csv"
        if not positions_path.is_file():
            legacy = outs / "spatial/tissue_positions_list.csv"
            positions_path = legacy if legacy.is_file() else positions_path
        scalefactors_path = outs / "spatial/scalefactors_json.json"
        image_path = processed / str(row["h_and_e"])
        for required in (matrix_path, molecule_path, positions_path, scalefactors_path, image_path):
            if not required.is_file() or required.stat().st_size == 0:
                raise FileNotFoundError(f"missing completed section input: {required}")
        processing_provenance = _section_spaceranger_provenance(
            spaceranger_root, row, data_root, spaceranger_receipt
        )
        filtered = _read_filtered_matrix(matrix_path, SELECTED_GENES, broad_panel.gene_names)
        if tuple(value.split(".")[0] for value in filtered.selected_feature_ids) != tuple(
            value.split(".")[0] for value in chromium.gene_ensembl_ids
        ):
            raise ValueError(
                f"selected Ensembl identities differ between {section} Visium and Chromium"
            )
        if tuple(value.split(".")[0] for value in filtered.broad_feature_ids) != tuple(
            value.split(".")[0] for value in broad_panel.ensembl_ids
        ):
            raise ValueError(
                f"broad Ensembl identities differ between {section} Visium and Chromium"
            )
        positions = _read_tissue_positions(positions_path, filtered.barcodes)
        barcode_index = {value: index for index, value in enumerate(filtered.barcodes.astype(str))}
        filtered_order = np.asarray(
            [barcode_index[value] for value in positions.barcodes.astype(str)], dtype=np.int64
        )
        selection_mask, selection_receipt = _h_and_e_observation_selection(
            image_path, positions.barcodes, positions.pixel_xy
        )
        retained_rows = np.flatnonzero(selection_mask).astype(np.int64)
        keep = filtered_order[retained_rows]
        positions = TissuePositions(
            positions.barcodes[retained_rows],
            positions.array_row_col[retained_rows],
            positions.pixel_xy[retained_rows],
        )
        full = filtered.selected_counts[keep]
        total = filtered.total_counts[keep]
        broad_counts = _subset_csr_rows(
            filtered.broad_counts, keep, f"{section} retained spatial broad counts"
        )
        split = _split_molecule_info(
            molecule_path,
            positions.barcodes,
            filtered.selected_feature_ids,
            filtered.broad_feature_ids,
            filtered.all_gene_feature_ids,
            full,
            broad_counts,
            total,
            section=section,
            chunk_size=args.molecule_chunk_size,
        )
        features, embedding_receipt = _section_embeddings(
            section=section,
            image_path=image_path,
            scalefactors_path=scalefactors_path,
            barcodes=positions.barcodes,
            pixel_xy=positions.pixel_xy,
            encoder=encoder,
            encoder_manifest=encoder_manifest,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            image_sha256=str(processing_provenance["h_and_e_sha256"]),
            encoder_parity=parity_receipt,
        )
        count = len(positions.barcodes)
        spot_parts["spot_ids"].append(
            np.asarray([f"{section}:{value}" for value in positions.barcodes])
        )
        spot_parts["barcode_ids"].append(positions.barcodes)
        spot_parts["donor_ids"].append(_constant_text(count, row["donor"]))
        spot_parts["section_ids"].append(_constant_text(count, section))
        spot_parts["indication_ids"].append(_constant_text(count, row["indication"]))
        spot_parts["spot_primary_eligible"].append(
            np.full(count, bool(row["primary_eligible"]), dtype=np.bool_)
        )
        spot_parts["pixel_xy"].append(positions.pixel_xy)
        spot_parts["array_row_col"].append(positions.array_row_col)
        spot_parts["st_counts_full"].append(full)
        spot_parts["st_counts_half_a"].append(split.half_a)
        spot_parts["st_counts_half_b"].append(split.half_b)
        spot_parts["st_total_umi_counts_full"].append(total)
        spot_parts["st_total_umi_counts_half_a"].append(split.total_half_a)
        spot_parts["st_total_umi_counts_half_b"].append(split.total_half_b)
        spot_parts["image_features"].append(features)
        broad_full_parts.append(broad_counts)
        broad_half_a_parts.append(split.broad_half_a)
        broad_half_b_parts.append(split.broad_half_b)
        section_receipts.append(
            {
                "section": section,
                "donor": str(row["donor"]),
                "primary_eligible": bool(row["primary_eligible"]),
                "spot_count": count,
                "filtered_matrix": {"path": str(matrix_path), "sha256": _sha256_file(matrix_path)},
                "selected_feature_ids": list(filtered.selected_feature_ids),
                "molecule_info": {
                    "path": str(molecule_path),
                    "sha256": _sha256_file(molecule_path),
                },
                "tissue_positions": {
                    "path": str(positions_path),
                    "sha256": _sha256_file(positions_path),
                },
                "h_and_e_observation_selection": selection_receipt,
                "umi_split": split.receipt,
                "embedding": embedding_receipt,
                "spaceranger_provenance": processing_provenance,
            }
        )
    spots = {name: np.concatenate(values, axis=0) for name, values in spot_parts.items()}
    broad_full = _concatenate_csr_rows(broad_full_parts, "spatial broad full")
    broad_half_a = _concatenate_csr_rows(broad_half_a_parts, "spatial broad half A")
    broad_half_b = _concatenate_csr_rows(broad_half_b_parts, "spatial broad half B")
    if not _csr_sum_equals(broad_full, broad_half_a, broad_half_b):
        raise ValueError("concatenated broad spatial halves do not reconstruct full counts")
    coordinates = _coordinate_features(spots["section_ids"], spots["pixel_xy"])
    manual_path = (
        args.pathology_annotations.expanduser().resolve() if args.pathology_annotations else None
    )
    manual_provenance = (
        args.pathology_annotation_manifest.expanduser().resolve()
        if args.pathology_annotation_manifest
        else None
    )
    pathologist, grouped_pathology, pathology_receipt = _manual_pathology_labels(
        manual_path, spots["section_ids"], spots["barcode_ids"], manual_provenance
    )
    program_membership = _program_membership()
    source_receipt = {
        "schema": RECEIPT_SCHEMA,
        "builder_implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256_file(protocol_path),
        "analysis_scope": protocol["analysis_scope"],
        "observation_level": "Visium_v2_spot_regional_not_cellular",
        "gene_selection": {
            "method": "outcome_independent_prespecified_state_program_union",
            "ordered_gene_ids": list(SELECTED_GENES),
            "ordered_gene_ids_sha256": _canonical_sha256(list(SELECTED_GENES)),
            "programs": {
                name: {
                    "genes": list(FROZEN_PROGRAMS[name]),
                    "classification": PROGRAM_CLASSIFICATIONS[name],
                }
                for name in PROGRAM_NAMES
            },
            "reliability_selection_deferred_to_outer_training_donors": True,
            "fixed_candidate_program_count": len(PROGRAM_NAMES),
            "broader_latent": {
                "definition": (
                    "PCA_fit_on_outer_training_donors_only_separate_from_program_endpoints"
                ),
                "allowed_component_range": list(BROAD_TRAINING_ONLY_PCA_DIMENSION_RANGE),
                "source_stores_raw_counts_not_a_globally_fit_latent": True,
                "common_gene_panel": broad_panel.receipt,
                "storage": "canonical_int32_CSR_no_global_densification",
                "spatial_full_csr_sha256": _csr_sha256(broad_full),
                "spatial_half_a_csr_sha256": _csr_sha256(broad_half_a),
                "spatial_half_b_csr_sha256": _csr_sha256(broad_half_b),
                "chromium_csr_sha256": _csr_sha256(chromium.broad_counts),
            },
        },
        "spatial_counts": "raw_filtered_Space_Ranger_unique_UMIs",
        "chromium_counts": "raw_CELLxGENE_H5AD_X_not_SoupX",
        "measurement_endpoint_guardrail": {
            "raw_half_vs_half_loss_is_not_the_measurement_floor": True,
            "required_benchmark_correction": (
                "explicit_full_depth_noise_correction_or_cross_fitting"
            ),
            "adjacent_section_replicates": "B1_and_L1_secondary_only",
        },
        "natural_reference_bank_composition_preserved": True,
        "composition_equalization_fields": ["sc_donor_ids", "sc_level1_type_ids"],
        "within_type_state_fields": [
            "sc_level1_type_ids",
            "sc_level2_type_ids",
            "sc_level3_type_ids",
            "sc_level4_type_ids",
        ],
        "type_label_provenance": {
            "source": "published_CELLxGENE_H5AD_Level1_Level2_Level3_Harmonised_Level4",
            "allowed_reference_use": (
                "reference_bank_stratification_and_training_donor_only_routing"
            ),
            "held_out_spatial_type_routing": "must_be_predicted_by_a_training_donor_only_model",
            "held_out_ST_or_type_label_routing_prohibited": True,
            "cell2location_or_Visium_cluster_labels_used": False,
        },
        "primary_donors": list(protocol["primary_donors"]),
        "sensitivity_only_donors": ["B2"],
        "matched_cohort_identity": {
            "frozen_section_map": {
                section: {
                    "donor": values[0],
                    "indication": values[1],
                    "h5ad_cohort": values[2],
                    "raw_h5ad_donor": values[3],
                    "h_and_e_filename": values[4],
                }
                for section, values in FROZEN_SECTION_MAP.items()
            },
            "h5ad_cell_identity_prefixes": {
                f"{kind}:{donor}": prefix
                for (kind, donor), prefix in FROZEN_H5AD_CELL_PREFIX.items()
            },
            "mapping_guardrail": (
                "protocol_exact_match_plus_independent_CELLxGENE_cell_ID_prefix_check"
            ),
        },
        "encoder": {
            "manifest": str(encoder_manifest.path),
            "manifest_sha256": encoder_manifest.sha256,
            "repository": encoder_manifest.repository,
            "revision": encoder_manifest.revision,
            "fine_tuning": "none",
            "device": "cuda",
            "stored_feature_dtype": "float16",
            "official_local_parity": parity_receipt,
        },
        "encoder_roles": {
            "primary": {
                "repository": encoder_manifest.repository,
                "revision": encoder_manifest.revision,
                "manifest_sha256": encoder_manifest.sha256,
                "status": "frozen_primary_run",
            },
            "secondary_comparator": {
                "repository": secondary_encoder_manifest.repository,
                "revision": secondary_encoder_manifest.revision,
                "manifest_sha256": secondary_encoder_manifest.sha256,
                "status": "prespecified_not_run_in_primary_source",
            },
        },
        "blank_image_control": blank_receipt,
        "pathology_annotations": pathology_receipt,
        "chromium_inputs": list(chromium.input_receipts),
        "spaceranger_run": {
            "receipt": str(spaceranger_root / "run_status.json"),
            "receipt_sha256": _sha256_file(spaceranger_root / "run_status.json"),
            "version": spaceranger_receipt["spaceranger_version"],
            "reference_metadata_sha256": spaceranger_receipt["reference_metadata_sha256"],
            "probe_set_sha256": spaceranger_receipt["probe_set_sha256"],
            "all_frozen_sections_complete": True,
        },
        "sections": section_receipts,
        "source_limit": "regional Visium validation cannot authorize cell-level hypotheses",
    }
    payload: dict[str, object] = {
        "schema_version": np.asarray(SOURCE_SCHEMA),
        "analysis_scope": np.asarray(str(protocol["analysis_scope"])),
        "observation_level": np.asarray("Visium_v2_spot_regional"),
        "reference_modality": np.asarray(str(protocol["reference_modality"])),
        "primary_endpoint_names": np.asarray(protocol["primary_endpoints"]),
        "bank_condition_names": np.asarray(protocol["bank_conditions"]),
        "primary_donor_ids": np.asarray(protocol["primary_donors"]),
        "sensitivity_donor_ids": np.asarray(protocol["failed_reference_sensitivity_donors"]),
        "gene_ids": np.asarray(SELECTED_GENES),
        "gene_ensembl_ids": np.asarray(chromium.gene_ensembl_ids),
        "broad_gene_ids": np.asarray(broad_panel.gene_names),
        "broad_gene_ensembl_ids": np.asarray(broad_panel.ensembl_ids),
        "program_names": np.asarray(PROGRAM_NAMES),
        "program_classifications": np.asarray(
            [PROGRAM_CLASSIFICATIONS[name] for name in PROGRAM_NAMES]
        ),
        "program_gene_membership": program_membership,
        **spots,
        "coordinate_features": coordinates,
        "coordinate_feature_names": np.asarray(COORDINATE_FEATURE_NAMES),
        "image_feature_names": np.asarray(IMAGE_FEATURE_NAMES),
        "blank_image_feature_vector": blank_feature[0].astype(np.float16),
        "blank_image_receipt_json": np.asarray(
            json.dumps(blank_receipt, sort_keys=True, allow_nan=False)
        ),
        "spot_pathologist_annotation": pathologist,
        "spot_grouped_pathology_annotation": grouped_pathology,
        "pathology_annotation_receipt_json": np.asarray(
            json.dumps(pathology_receipt, sort_keys=True, allow_nan=False)
        ),
        "sc_counts": chromium.counts,
        **_csr_payload("st_broad_counts_full", broad_full),
        **_csr_payload("st_broad_counts_half_a", broad_half_a),
        **_csr_payload("st_broad_counts_half_b", broad_half_b),
        **_csr_payload("sc_broad_counts", chromium.broad_counts),
        "sc_total_umi_counts": chromium.total_counts,
        "sc_cell_ids": chromium.cell_ids,
        "sc_donor_ids": chromium.donor_ids,
        "sc_raw_h5ad_donor_ids": chromium.raw_h5ad_donor_ids,
        "sc_sample_ids": chromium.sample_ids,
        "sc_indication_ids": chromium.indication_ids,
        "sc_primary_eligible": chromium.primary_eligible,
        "sc_level1_type_ids": chromium.level1,
        "sc_level2_type_ids": chromium.level2,
        "sc_level3_type_ids": chromium.level3,
        "sc_level4_type_ids": chromium.level4,
        "sc_n_features_rna": chromium.n_features,
        "sc_percent_mt": chromium.percent_mt,
        "sc_percent_ribo": chromium.percent_ribo,
        "sc_percent_hb": chromium.percent_hb,
        "sc_dv200_percent": chromium.dv200,
        "sc_block_age_months": chromium.block_age_months,
        **_reference_coverage(chromium),
        "source_receipt_json": np.asarray(
            json.dumps(source_receipt, sort_keys=True, allow_nan=False)
        ),
    }
    _validate_source(payload, protocol)
    _atomic_npz(output, payload)
    source_sha = _sha256_file(output)
    _atomic_json(
        output.with_suffix(".receipt.json"),
        {
            **source_receipt,
            "source": str(output),
            "source_sha256": source_sha,
            "spot_count": len(spots["spot_ids"]),
            "chromium_cell_count": len(chromium.cell_ids),
        },
    )
    print(
        json.dumps(
            {
                "source": str(output),
                "sha256": source_sha,
                "spots": len(spots["spot_ids"]),
                "chromium_cells": len(chromium.cell_ids),
                "schema": SOURCE_SCHEMA,
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--spaceranger-root", type=Path, default=DEFAULT_SPACERANGER_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--encoder-manifest", type=Path, default=DEFAULT_ENCODER_MANIFEST)
    parser.add_argument(
        "--encoder-parity-receipt", type=Path, default=DEFAULT_ENCODER_PARITY_RECEIPT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--pathology-annotations", type=Path)
    parser.add_argument("--pathology-annotation-manifest", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--molecule-chunk-size", type=int, default=1_000_000)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(parse_args()))
