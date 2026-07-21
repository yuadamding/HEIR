#!/usr/bin/env python3
"""Run the post-protocol spatialDLPFC biological-mechanism experiment.

This runner is deliberately narrow: it tests whether a matched snRNA reference
improves an H&E-only predictor (M3 versus M0), and whether that improvement
depends on spatially aligned morphology (M3 versus within-section shuffled M4).
The processed spatialDLPFC target has already been opened, the source lacks
full-resolution 0.5-um/px H&E, and no molecule-level split is available.
Consequently every result is exploratory, protocol-deviating, and
nonconfirmatory; this script never estimates or reports an ST floor.

Stages are separated so ``fit-predict`` cannot open held-out ST counts.  Large
artifacts live below ``--output`` (on /mnt/seagate by default), not in Git.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Set conservative process-wide defaults before NumPy/Torch initialize pools.
for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import h5py
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from heir.evaluation import generative_fusion as core
from heir.features import create_frozen_encoder, load_encoder_manifest

SCHEMA = "heir.spatialdlpfc_exploratory.v2"
FEATURE_SCHEMA = "heir.spatialdlpfc_hoptimus1_mesoscopic_features.v2"
PREPARED_SCHEMA = "heir.spatialdlpfc_exploratory_prepared.v2"
PREDICTION_SCHEMA = "heir.spatialdlpfc_exploratory_predictions.v2"
SCORE_SCHEMA = "heir.spatialdlpfc_exploratory_scores.v2"
ANALYSIS_SCOPE = "exploratory_protocol_deviating_nonconfirmatory"
HOPTIMUS_REPOSITORY = "bioptimus/H-optimus-1"
HOPTIMUS_REVISION = "3592cb220dec7a150c5d7813fb56e68bd57473b9"
HOPTIMUS_MANIFEST_SHA256 = "f6852288e1ae146a4865bf19e38ce994c0be9ce1c2bfa09bdf77747043ac8fd9"
DEFAULT_SOURCE = Path("/mnt/seagate/HEIR_runs/spatialdlpfc_exploratory/source/source.h5")
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/spatialdlpfc_exploratory")
DEFAULT_MODEL_DIR = Path("/mnt/seagate/HnE/pretrained/H-optimus-1")
DEFAULT_ENCODER_MANIFEST = REPO_ROOT / "manifests/encoders/hoptimus1.json"
DEFAULT_PANEL = REPO_ROOT / "configs/spatialdlpfc_exploratory_gene_panel.json"
DEFAULT_PROTOCOL = REPO_ROOT / "configs/spatialdlpfc_exploratory_protocol_v2.json"
NATCOMM_RUNNER = REPO_ROOT / "scripts/benchmark_natcommun_generative_development.py"

# Frozen before this exploratory run; section choice did not inspect target counts.
QUERY_SECTIONS: Mapping[str, tuple[str, str]] = {
    "Br2720": ("V10U24-091_C1", "snRNA_Br2720_post"),
    "Br2743": ("V19B23-074_A1", "snRNA_Br2743_mid"),
    "Br3942": ("V19B23-074_B1", "snRNA_Br3942_mid"),
    "Br6423": ("V19B23-073_C1", "snRNA_Br6423_post"),
    "Br6432": ("V10B01-002_B1", "snRNA_Br6432_ant"),
    "Br6471": ("V10B01-052_A1", "snRNA_Br6471_mid"),
    "Br6522": ("V10B01-053_B1", "snRNA_Br6522_post"),
    "Br8325": ("V10U24-094_C1", "snRNA_Br8325_ant"),
    "Br8492": ("V19B23-073_D1", "snRNA_Br8492_post"),
    "Br8667": ("V10U24-094_D1", "snRNA_Br8667_ant"),
}
BR2720 = "Br2720"
MAIN_CLEAN_DONORS = tuple(donor for donor in QUERY_SECTIONS if donor != BR2720)
SENSITIVITY_DONORS = tuple(QUERY_SECTIONS)
MODEL_ARMS = ("M0", "M3", "M4")
SPOT_CAP_PER_SECTION = 2048
REFERENCE_CAP_PER_SAMPLE_TYPE = 512
PATCH_PIXELS = 224
GRID_STRIDE_PIXELS = 112
FEATURE_BATCH_SIZE = 1
FROZEN_PROTOCOL_SHA256 = "7a33d4e1b786bf24fd0c17a1cbefffc1ef6365fd241e50ad863dfa23e7dd430e"
V1_PROTOCOL_SHA256 = "9c2a717c887ce2c12e274d87433ceea0b3ef1528c57f8ba9e92779913f20b2f6"
PANEL_GENE_COUNT = 248


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, object], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        if private:
            os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _semantic_hash(arrays: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        if value.dtype.kind == "O":
            raise TypeError(f"object array is prohibited: {name}")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.dtype.str.encode())
        digest.update(_json_bytes(list(value.shape)))
        if value.dtype.kind == "U":
            digest.update(_json_bytes(value.tolist()))
        else:
            digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _text_array(dataset: h5py.Dataset) -> np.ndarray:
    values = np.asarray(dataset[...])
    if values.ndim != 1:
        raise ValueError(f"{dataset.name} must be one-dimensional")
    return np.asarray(
        [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values],
        dtype=str,
    )


def _scalar_text(value: object) -> str:
    array = np.asarray(value).reshape(-1)
    if len(array) != 1:
        raise ValueError("expected one scalar string")
    item = array[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _safe_name(value: str) -> str:
    stem = "".join(character if character.isalnum() else "_" for character in value).strip("_")
    return f"{stem[:60]}_{hashlib.sha256(value.encode()).hexdigest()[:10]}"


def _import_script(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_resources(args: argparse.Namespace) -> None:
    if not 1 <= int(args.cpu_threads) <= 4:
        raise ValueError("--cpu-threads must be between 1 and 4")
    if not 0.0 < float(args.gpu_memory_fraction) <= 0.60:
        raise ValueError("--gpu-memory-fraction must be in (0, 0.60]")
    if int(args.fit_batch_size) < 1:
        raise ValueError("--fit-batch-size must be positive")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = str(args.cpu_threads)
    torch.set_num_threads(int(args.cpu_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required unless --device cpu is explicit")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one GPU must be visible to this process")
        device = torch.device(args.device)
        if device.index not in (None, 0):
            raise ValueError("only one visible GPU (cuda:0) may be used")
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_memory_fraction), 0)


def _assert_no_unrelated_gpu_process() -> None:
    """Fail closed rather than contend with another compute process."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise RuntimeError("cannot verify the frozen GPU contention rule with nvidia-smi") from error
    active = {
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip() and line.strip().isdigit()
    }
    unrelated = sorted(active - {os.getpid()})
    if unrelated:
        raise RuntimeError(f"unrelated GPU compute processes are active: {unrelated}")


def _reject_encoder(value: object) -> None:
    normalized = str(value).casefold().replace("-", "").replace("_", "")
    if "uni2" in normalized:
        raise ValueError("UNI2-h is explicitly prohibited in this experiment")
    if normalized not in {"hoptimus1", "bioptimus/hoptimus1"}:
        raise ValueError("the only accepted encoder is bioptimus/H-optimus-1")


def _validate_hoptimus(args: argparse.Namespace):
    _reject_encoder(args.encoder)
    manifest = load_encoder_manifest(args.encoder_manifest)
    _reject_encoder(manifest.repository)
    if (
        manifest.repository != HOPTIMUS_REPOSITORY
        or manifest.revision != HOPTIMUS_REVISION
        or manifest.sha256 != HOPTIMUS_MANIFEST_SHA256
        or manifest.input_pixels != PATCH_PIXELS
        or manifest.feature_width != 1536
        or manifest.fine_tuning != "prohibited"
    ):
        raise ValueError("encoder manifest does not match the frozen H-optimus-1 primary")
    checkpoint = args.model_dir / manifest.checkpoint_filename
    if not checkpoint.is_file() or _sha256(checkpoint) != manifest.checkpoint_sha256:
        raise ValueError("local H-optimus-1 checkpoint is missing or checksum-mismatched")
    return manifest


def _load_protocol(args: argparse.Namespace) -> tuple[Mapping[str, object], str]:
    observed_sha = _sha256(args.protocol)
    if observed_sha != FROZEN_PROTOCOL_SHA256:
        raise ValueError("exploratory protocol differs from its frozen SHA-256")
    payload = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "heir.spatialdlpfc_exploratory_protocol.v2"
        or payload.get("analysis_status") != ANALYSIS_SCOPE
    ):
        raise ValueError("exploratory protocol schema/status is invalid")
    revision = payload.get("revision", {})
    if (
        revision.get("predecessor_sha256") != V1_PROTOCOL_SHA256
        or revision.get("v1_outputs_written") is not False
        or revision.get("v1_models_fit") is not False
        or revision.get("v1_targets_scored") is not False
        or "8_of_256" not in str(revision.get("trigger", ""))
    ):
        raise ValueError("v2 protocol does not preserve the frozen v1 preflight trigger")
    queries = {
        str(row["donor"]): (str(row["section"]), str(row["reference_sample"]))
        for row in payload.get("selected_queries", ())
    }
    outer = payload.get("outer_validation", {})
    limits = payload.get("resource_limits", {})
    image = payload.get("image_proxy", {})
    encoder = payload.get("encoder", {})
    if queries != dict(QUERY_SECTIONS):
        raise ValueError("runner query selection differs from the frozen protocol")
    expected = {
        "base_seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.fit_batch_size,
        "latent_dimensions": args.latent_dim,
        "spot_cap_per_section": SPOT_CAP_PER_SECTION,
        "reference_cell_cap_per_sample_and_broad_type": REFERENCE_CAP_PER_SAMPLE_TYPE,
    }
    if any(int(outer.get(name, -1)) != int(value) for name, value in expected.items()):
        raise ValueError("CLI training/cap settings differ from the frozen protocol")
    if (
        int(image.get("crop_pixels", -1)) != PATCH_PIXELS
        or int(image.get("grid_stride_pixels", -1)) != GRID_STRIDE_PIXELS
        or bool(image.get("scale_qualified", True))
        or int(limits.get("feature_batch_size", -1)) != FEATURE_BATCH_SIZE
        or int(limits.get("maximum_CPU_threads", -1)) < int(args.cpu_threads)
        or float(limits.get("maximum_GPU_memory_fraction", -1)) < float(args.gpu_memory_fraction)
        or encoder.get("repository") != HOPTIMUS_REPOSITORY
        or encoder.get("revision") != HOPTIMUS_REVISION
        or encoder.get("UNI2_h") != "forbidden_not_run"
    ):
        raise ValueError("CLI/image/encoder resources differ from the frozen protocol")
    panel = payload.get("immutable_inputs", {}).get("frozen_gene_panel", {})
    if panel.get("sha256") != _sha256(args.panel) or int(panel.get("size", -1)) != PANEL_GENE_COUNT:
        raise ValueError("panel differs from the protocol-bound frozen artifact")
    return payload, observed_sha


def _required_source(handle: h5py.File) -> None:
    required = {
        "/panel/genes",
        "/spots/id",
        "/spots/barcode",
        "/spots/donor",
        "/spots/section",
        "/spots/sample_id",
        "/spots/position",
        "/spots/x_fullres",
        "/spots/y_fullres",
        "/spots/array_row",
        "/spots/array_col",
        "/spots/library",
        "/spots/counts",
        "/reference/id",
        "/reference/donor",
        "/reference/sample_id",
        "/reference/position",
        "/reference/cell_type",
        "/reference/library",
        "/reference/counts",
        "/images/section",
        "/images/path",
        "/images/width",
        "/images/height",
        "/images/scale_factor",
    }
    missing = sorted(name for name in required if name not in handle)
    if missing:
        raise ValueError(f"source HDF5 schema is incomplete: {missing}")


def _source_metadata(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        _required_source(handle)
        result = {"genes": _text_array(handle["/panel/genes"])}
        for name in ("id", "barcode", "donor", "section", "sample_id", "position"):
            result[f"spot_{name}"] = _text_array(handle[f"/spots/{name}"])
        for name in ("x_fullres", "y_fullres", "array_row", "array_col", "library"):
            result[f"spot_{name}"] = np.asarray(handle[f"/spots/{name}"][...])
        for name in ("id", "donor", "sample_id", "position", "cell_type"):
            result[f"reference_{name}"] = _text_array(handle[f"/reference/{name}"])
        result["reference_library"] = np.asarray(handle["/reference/library"][...])
        result["image_section"] = _text_array(handle["/images/section"])
        result["image_path"] = _text_array(handle["/images/path"])
        for name in ("width", "height", "scale_factor"):
            result[f"image_{name}"] = np.asarray(handle[f"/images/{name}"][...])
        spot_shape = handle["/spots/counts"].shape
        reference_shape = handle["/reference/counts"].shape
    spots, genes = len(result["spot_id"]), len(result["genes"])
    cells = len(result["reference_id"])
    if spot_shape != (spots, genes) or reference_shape != (cells, genes):
        raise ValueError("counts must be observation-by-panel-gene without transposition")
    for prefix, rows, names in (
        (
            "spot_",
            spots,
            ("id", "barcode", "donor", "section", "sample_id", "position", "x_fullres", "y_fullres", "array_row", "array_col", "library"),
        ),
        ("reference_", cells, ("id", "donor", "sample_id", "position", "cell_type", "library")),
        ("image_", len(result["image_section"]), ("section", "path", "width", "height", "scale_factor")),
    ):
        for name in names:
            if np.asarray(result[f"{prefix}{name}"]).shape != (rows,):
                raise ValueError(f"{prefix}{name} is not row-aligned")
    if len(set(result["spot_id"].tolist())) != spots:
        raise ValueError("/spots/id must be globally unique")
    if len(set(result["reference_id"].tolist())) != cells:
        raise ValueError("/reference/id must be globally unique")
    if len(set(result["image_section"].tolist())) != len(result["image_section"]):
        raise ValueError("/images/section must be unique")
    for name in (
        "genes",
        "spot_id",
        "spot_barcode",
        "spot_donor",
        "spot_section",
        "spot_sample_id",
        "spot_position",
        "reference_id",
        "reference_donor",
        "reference_sample_id",
        "reference_position",
        "reference_cell_type",
        "image_section",
        "image_path",
    ):
        if np.any(np.char.strip(np.asarray(result[name], dtype=str)) == ""):
            raise ValueError(f"{name} contains an empty required label")
    for name in ("spot_x_fullres", "spot_y_fullres", "spot_library", "reference_library", "image_scale_factor"):
        values = np.asarray(result[name], dtype=np.float64)
        if not np.all(np.isfinite(values)) or ("library" in name and np.any(values <= 0)):
            raise ValueError(f"{name} must be finite and libraries positive")
        if "library" in name and np.any(values != np.floor(values)):
            raise ValueError(f"{name} must contain raw integer library totals")
    for name in ("image_width", "image_height"):
        values = np.asarray(result[name], dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values < 1) or np.any(values != np.floor(values)):
            raise ValueError(f"{name} must contain positive integer dimensions")
    if np.any(np.asarray(result["image_scale_factor"], dtype=float) <= 0):
        raise ValueError("image scale factors must be positive")
    for name in ("spot_array_row", "spot_array_col"):
        values = np.asarray(result[name], dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values != np.floor(values)):
            raise ValueError(f"{name} must contain finite integer lattice coordinates")
    return result


def _frozen_panel(args: argparse.Namespace, observed: Sequence[str]) -> tuple[np.ndarray, str]:
    payload = json.loads(args.panel.read_text(encoding="utf-8"))
    genes = np.asarray(payload.get("gene_ids", ()), dtype=str)
    if (
        payload.get("schema") != "heir.spatialdlpfc_annotation_compatibility_panel.v1"
        or payload.get("analysis_status") != ANALYSIS_SCOPE
        or int(payload.get("source_gene_count", -1)) != 256
        or int(payload.get("gene_count", -1)) != PANEL_GENE_COUNT
        or len(genes) != PANEL_GENE_COUNT
        or len(set(genes.tolist())) != len(genes)
        or payload.get("predecessor_protocol_sha256") != V1_PROTOCOL_SHA256
        or len(payload.get("excluded_unmappable_gene_ids", ())) != 8
    ):
        raise ValueError("frozen 248-gene annotation-compatibility panel contract is invalid")
    identity = hashlib.sha256(("\n".join(genes.tolist()) + "\n").encode()).hexdigest()
    if identity != payload.get("identity_sha256"):
        raise ValueError("compatibility panel ordered-gene identity is invalid")
    if not np.array_equal(genes, np.asarray(observed, dtype=str)):
        raise ValueError("source panel differs in membership/order from the frozen panel")
    return genes, _sha256(args.panel)


def _grid_centres(length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("image dimensions must be positive")
    centres = np.arange(GRID_STRIDE_PIXELS, length, GRID_STRIDE_PIXELS, dtype=np.int64)
    if not len(centres):
        centres = np.asarray([GRID_STRIDE_PIXELS], dtype=np.int64)
    if centres[-1] + PATCH_PIXELS // 2 < length:
        centres = np.append(centres, centres[-1] + GRID_STRIDE_PIXELS)
    return centres


def _crop(image: np.ndarray, centre_x: int, centre_y: int) -> np.ndarray:
    radius = PATCH_PIXELS // 2
    left, top = int(centre_x) - radius, int(centre_y) - radius
    right, bottom = left + PATCH_PIXELS, top + PATCH_PIXELS
    patch = np.full((PATCH_PIXELS, PATCH_PIXELS, 3), 255, dtype=np.uint8)
    source_left, source_top = max(0, left), max(0, top)
    source_right, source_bottom = min(image.shape[1], right), min(image.shape[0], bottom)
    if source_left < source_right and source_top < source_bottom:
        patch[
            source_top - top : source_bottom - top,
            source_left - left : source_right - left,
        ] = image[source_top:source_bottom, source_left:source_right]
    return patch


def _nearest(values: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1).astype(np.int64)


def _estimate_hires_scale_receipt(
    source: Mapping[str, np.ndarray], rows: np.ndarray, scale: float
) -> Mapping[str, float | str]:
    """Estimate scale from the fixed 100-um Visium centre lattice, never outcomes."""

    array_row = np.asarray(source["spot_array_row"][rows], dtype=np.int64)
    array_col = np.asarray(source["spot_array_col"][rows], dtype=np.int64)
    x = np.asarray(source["spot_x_fullres"][rows], dtype=np.float64)
    y = np.asarray(source["spot_y_fullres"][rows], dtype=np.float64)
    lookup = {(int(row), int(col)): index for index, (row, col) in enumerate(zip(array_row, array_col))}
    distances: list[float] = []
    # 10x Visium's staggered lattice uses horizontal +/-2 and diagonal +/-1
    # array-coordinate steps for neighbouring 100-um spot centres.
    for index, (row, col) in enumerate(zip(array_row, array_col)):
        for delta_row, delta_col in ((0, 2), (1, -1), (1, 1)):
            neighbour = lookup.get((int(row + delta_row), int(col + delta_col)))
            if neighbour is not None:
                distances.append(float(np.hypot(x[index] - x[neighbour], y[index] - y[neighbour])))
    if not distances or not np.isfinite(distances).all():
        raise ValueError("cannot estimate hires scale from the Visium 100-um centre lattice")
    fullres_pixels_per_100_um = float(np.median(distances))
    if fullres_pixels_per_100_um <= 0 or scale <= 0:
        raise ValueError("Visium/image scale estimate is non-positive")
    fullres_mpp = 100.0 / fullres_pixels_per_100_um
    hires_mpp = fullres_mpp / scale
    return {
        "estimated_microns_per_pixel": hires_mpp,
        "estimated_fullres_microns_per_pixel": fullres_mpp,
        "native_pixels_per_nominal_112_um_field": 112.0 / hires_mpp,
        "upsampling_ratio": hires_mpp / 0.5,
        "upsampling_ratio_to_0.5_um_per_px": hires_mpp / 0.5,
        "estimation_method": "median_fullres_distance_of_100um_Visium_lattice_neighbours_then_hires_scale_factor",
    }


def features(args: argparse.Namespace) -> Mapping[str, object]:
    """Encode native hires-image crops; this stage never reads molecular counts."""

    _, protocol_sha = _load_protocol(args)
    manifest = _validate_hoptimus(args)
    source = _source_metadata(args.source)
    _, panel_sha = _frozen_panel(args, source["genes"])
    source_sha = _sha256(args.source)
    encoder = None
    old_manifest_path = args.output / "features" / "manifest.json"
    old_manifest = (
        json.loads(old_manifest_path.read_text(encoding="utf-8"))
        if args.resume and old_manifest_path.is_file()
        else {}
    )
    image_lookup = {
        section: index for index, section in enumerate(source["image_section"].tolist())
    }
    section_receipts: dict[str, object] = {}
    for section in sorted(set(source["spot_section"].tolist())):
        if section not in image_lookup:
            raise ValueError(f"section {section} has no exported hires image")
        image_index = image_lookup[section]
        image_path = Path(source["image_path"][image_index]).expanduser()
        if not image_path.is_absolute():
            image_path = args.source.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_sha = _sha256(image_path)
        keep = np.flatnonzero(source["spot_section"] == section)
        scale = float(source["image_scale_factor"][image_index])
        scale_receipt = _estimate_hires_scale_receipt(source, keep, scale)
        identity = {
            "schema": FEATURE_SCHEMA,
            "analysis_scope": ANALYSIS_SCOPE,
            "section": section,
            "source_sha256": source_sha,
            "protocol_sha256": protocol_sha,
            "panel_sha256": panel_sha,
            "image_path": str(image_path),
            "image_sha256": image_sha,
            "image_width": int(source["image_width"][image_index]),
            "image_height": int(source["image_height"][image_index]),
            "native_image_dimensions": [
                int(source["image_width"][image_index]),
                int(source["image_height"][image_index]),
            ],
            "scale_factor": scale,
            "fullres_to_hires_scale_factor": scale,
            "encoder_repository": manifest.repository,
            "encoder_revision": manifest.revision,
            "encoder_manifest_sha256": manifest.sha256,
            "encoder_checkpoint_sha256": manifest.checkpoint_sha256,
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "encoder_batch_size": FEATURE_BATCH_SIZE,
            "patch_pixels": PATCH_PIXELS,
            "grid_stride_pixels": GRID_STRIDE_PIXELS,
            "grid_encoding": "unique_grid_cells_assigned_to_at_least_one_spot",
            "pixel_size_qualification": "failed_not_native_0.5_um_per_px",
            "representation_scope": "scale_unqualified_mesoscopic_proxy",
            **scale_receipt,
            "spot_coordinate_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    np.column_stack(
                        (
                            source["spot_x_fullres"][keep],
                            source["spot_y_fullres"][keep],
                            source["spot_array_row"][keep],
                            source["spot_array_col"][keep],
                        )
                    ),
                    dtype="<f8",
                ).tobytes()
            ).hexdigest(),
            "spot_ids_sha256": hashlib.sha256(
                "\n".join(source["spot_id"][keep].tolist()).encode()
            ).hexdigest(),
        }
        identity_sha = hashlib.sha256(_json_bytes(identity)).hexdigest()
        cache_path = args.output / "features" / "sections" / f"{_safe_name(section)}.npz"
        if args.resume and cache_path.is_file():
            cached = _load_npz(cache_path)
            prior = old_manifest.get("sections", {}).get(section, {})
            cache_file_sha = _sha256(cache_path)
            if (
                prior.get("cache_sha256") == cache_file_sha
                and _scalar_text(cached.get("cache_identity_sha256", "")) == identity_sha
            ):
                if (
                    np.asarray(cached.get("spot_features", np.empty(0))).shape
                    == (len(keep), manifest.feature_width)
                    and np.array_equal(np.asarray(cached["spot_ids"]).astype(str), source["spot_id"][keep])
                ):
                    section_receipts[section] = {
                        **identity,
                        "cache_path": str(cache_path),
                        "cache_sha256": cache_file_sha,
                        "cache_status": "reused",
                    }
                    continue
        with Image.open(image_path) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        expected = (int(source["image_height"][image_index]), int(source["image_width"][image_index]))
        if rgb.shape[:2] != expected:
            raise ValueError(f"image dimensions differ from source metadata for {section}")
        x_grid, y_grid = _grid_centres(rgb.shape[1]), _grid_centres(rgb.shape[0])
        full_grid_xy = np.asarray([(x, y) for y in y_grid for x in x_grid], dtype=np.int32)
        x_hires = np.asarray(source["spot_x_fullres"][keep], dtype=np.float64) * scale
        y_hires = np.asarray(source["spot_y_fullres"][keep], dtype=np.float64) * scale
        x_index, y_index = _nearest(x_hires, x_grid), _nearest(y_hires, y_grid)
        assigned_full = y_index * len(x_grid) + x_index
        # Encoding only grid cells actually assigned to a spot preserves every
        # spot's patch exactly while avoiding thousands of unused background
        # patches on the downsampled whole-image canvas.
        used_full = np.unique(assigned_full)
        grid_xy = full_grid_xy[used_full]
        assigned = np.searchsorted(used_full, assigned_full)
        grid_features = np.empty((len(grid_xy), manifest.feature_width), dtype=np.float16)
        if encoder is None:
            if str(args.device).startswith("cuda"):
                _assert_no_unrelated_gpu_process()
            encoder = create_frozen_encoder(args.model_dir, manifest, str(args.device))
        for row, (centre_x, centre_y) in enumerate(grid_xy):
            # H-optimus-1 is intentionally batch one to stay inside the 60% GPU cap.
            encoded = encoder.encode(_crop(rgb, int(centre_x), int(centre_y))[None])
            grid_features[row] = np.asarray(encoded[0], dtype=np.float16)
        outside = (x_hires < 0) | (x_hires >= rgb.shape[1]) | (y_hires < 0) | (y_hires >= rgb.shape[0])
        arrays = {
            "schema": np.asarray(FEATURE_SCHEMA),
            "cache_identity_sha256": np.asarray(identity_sha),
            "cache_receipt_json": np.asarray(json.dumps(identity, sort_keys=True, allow_nan=False)),
            "spot_ids": source["spot_id"][keep],
            "spot_features": grid_features[assigned],
            "assigned_grid_index": assigned.astype(np.int32),
            "assigned_full_grid_index": assigned_full.astype(np.int32),
            "grid_xy_hires_pixels": grid_xy,
            "grid_features": grid_features,
            "out_of_image_coordinate": outside,
        }
        _atomic_npz(cache_path, arrays)
        section_receipts[section] = {
            **identity,
            "cache_path": str(cache_path),
            "cache_sha256": _sha256(cache_path),
            "cache_semantic_sha256": _semantic_hash(arrays),
            "grid_patch_count": len(grid_xy),
            "full_grid_patch_count": len(full_grid_xy),
            "skipped_unassigned_grid_patch_count": len(full_grid_xy) - len(grid_xy),
            "spot_count": len(keep),
            "out_of_image_spot_count": int(outside.sum()),
            "cache_status": "created",
        }
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    receipt = {
        "schema": FEATURE_SCHEMA,
        "analysis_scope": ANALYSIS_SCOPE,
        "source_path": str(args.source.resolve()),
        "source_sha256": source_sha,
        "protocol_path": str(args.protocol),
        "protocol_sha256": protocol_sha,
        "gene_panel": {
            "size": PANEL_GENE_COUNT,
            "revision": "v1_failed_annotation_preflight; v2_removed_only_8_unmappable_symbols",
            "target_values_used_for_revision": False,
        },
        "encoder": {
            "repository": manifest.repository,
            "revision": manifest.revision,
            "manifest_sha256": manifest.sha256,
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "batch_size": FEATURE_BATCH_SIZE,
        },
        "image_sampling": {
            "patch_pixels": PATCH_PIXELS,
            "stride_pixels": GRID_STRIDE_PIXELS,
            "assignment": "nearest_grid_centre_after_fullres_to_hires_scale",
            "resampling": "none_native_hires_pixels_white_edge_padding",
            "pixel_size_qualified_for_H_optimus_0.5_um_per_px": False,
            "interpretation": "mesoscopic_regional_proxy_not_cell_morphology",
        },
        "sections": section_receipts,
        "all_sections_complete": len(section_receipts) == len(set(source["spot_section"].tolist())),
    }
    _atomic_json(args.output / "features" / "manifest.json", receipt)
    encoder = None
    gc.collect()
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return receipt


def _stable_cap(indices: Iterable[int], identifiers: np.ndarray, cap: int, _group: str) -> np.ndarray:
    rows = np.asarray(list(indices), dtype=np.int64)
    if len(rows) <= cap:
        return np.sort(rows)
    ranked = sorted(
        rows.tolist(),
        key=lambda row: (
            hashlib.sha256(str(identifiers[row]).encode()).digest(),
            str(identifiers[row]),
        ),
    )
    return np.sort(np.asarray(ranked[:cap], dtype=np.int64))


def _validated_counts(values: object, name: str) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    if (
        numeric.ndim != 2
        or not np.all(np.isfinite(numeric))
        or np.any(numeric < 0)
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric > np.iinfo(np.int32).max)
    ):
        raise ValueError(f"{name} must contain finite non-negative int32-range raw counts")
    return numeric.astype(np.int32)


def _read_count_rows(dataset: h5py.Dataset | np.ndarray, rows: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows) or np.any(np.diff(rows) < 0):
        raise ValueError(f"{name} rows must be non-empty and sorted")
    return _validated_counts(np.asarray(dataset[rows, :]), name)


def _selected_features(args: argparse.Namespace, source: Mapping[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    result = np.empty((len(rows), 1536), dtype=np.float16)
    row_to_output = {int(row): index for index, row in enumerate(rows.tolist())}
    for section in sorted(set(source["spot_section"][rows].tolist())):
        section_rows = rows[source["spot_section"][rows] == section]
        cache_path = args.output / "features" / "sections" / f"{_safe_name(section)}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"feature cache absent for {section}; run features first")
        cache = _load_npz(cache_path)
        cache_ids = np.asarray(cache["spot_ids"]).astype(str)
        cache_index = {identifier: index for index, identifier in enumerate(cache_ids.tolist())}
        cache_features = np.asarray(cache["spot_features"], dtype=np.float16)
        for source_row in section_rows:
            identifier = str(source["spot_id"][source_row])
            if identifier not in cache_index:
                raise ValueError(f"feature cache lacks spot {identifier}")
            result[row_to_output[int(source_row)]] = cache_features[cache_index[identifier]]
    if not np.all(np.isfinite(result)):
        raise ValueError("selected H-optimus-1 features contain non-finite values")
    return result


def prepare(args: argparse.Namespace) -> Mapping[str, object]:
    """Seal public fit inputs separately from score-only held-out counts."""

    _, protocol_sha = _load_protocol(args)
    feature_manifest_path = args.output / "features" / "manifest.json"
    if not feature_manifest_path.is_file():
        raise FileNotFoundError("run the features stage first")
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    source = _source_metadata(args.source)
    genes, panel_sha = _frozen_panel(args, source["genes"])
    source_sha = _sha256(args.source)
    if feature_manifest.get("source_sha256") != source_sha:
        raise ValueError("feature caches are not bound to the current source")
    feature_manifest_identity = hashlib.sha256(
        _json_bytes(
            {
                "source_sha256": feature_manifest.get("source_sha256"),
                "protocol_sha256": feature_manifest.get("protocol_sha256"),
                "encoder": feature_manifest.get("encoder"),
                "image_sampling": feature_manifest.get("image_sampling"),
                "section_cache_sha256": {
                    section: receipt.get("cache_sha256")
                    for section, receipt in feature_manifest.get("sections", {}).items()
                },
            }
        )
    ).hexdigest()
    # The half-split program diagnostic is prohibited/disabled for this source.
    # These inert shape-compatible arrays are never consumed by model fitting.
    program_names = np.asarray(["disabled_not_run"], dtype=str)
    program_membership = np.zeros((1, len(genes)), dtype=bool)
    all_sections = sorted(set(source["spot_section"].tolist()))
    fold_receipts: dict[str, object] = {}
    with h5py.File(args.source, "r") as handle:
        _required_source(handle)
        all_spot_counts: np.ndarray | None = None
        all_reference_counts: np.ndarray | None = None
        for heldout in SENSITIVITY_DONORS:  # serial by construction
            query_section, matched_sample = QUERY_SECTIONS[heldout]
            if not matched_sample.startswith("snRNA_"):
                raise ValueError("frozen reference labels must use the snRNA_ provenance prefix")
            source_matched_sample = matched_sample.removeprefix("snRNA_")
            expected_position = source_matched_sample.rsplit("_", 1)[-1]
            query_candidates = np.flatnonzero(
                (source["spot_donor"] == heldout) & (source["spot_section"] == query_section)
            )
            if not len(query_candidates):
                raise ValueError(f"frozen query section is absent: {heldout}/{query_section}")
            if set(source["spot_position"][query_candidates].tolist()) != {expected_position}:
                raise ValueError("frozen query section/reference region labels are inconsistent")
            query_rows = _stable_cap(
                query_candidates, source["spot_id"], SPOT_CAP_PER_SECTION, f"query:{heldout}:{query_section}"
            )
            train_groups: list[np.ndarray] = []
            for section in all_sections:
                section_rows = np.flatnonzero(
                    (source["spot_section"] == section) & (source["spot_donor"] != heldout)
                )
                if len(section_rows):
                    train_groups.append(
                        _stable_cap(section_rows, source["spot_id"], SPOT_CAP_PER_SECTION, f"train:{heldout}:{section}")
                    )
            train_rows = np.sort(np.concatenate(train_groups))
            if heldout in set(source["spot_donor"][train_rows].tolist()):
                raise RuntimeError("held-out donor leaked into ST training rows")
            # Br2720 never contributes to training/wrong/generic reference banks.
            train_reference_candidates = np.flatnonzero(
                (source["reference_donor"] != heldout)
                & (source["reference_donor"] != BR2720)
                & (source["reference_cell_type"] != "drop")
            )
            reference_groups: list[np.ndarray] = []
            group_keys = sorted(
                set(
                    zip(
                        source["reference_sample_id"][train_reference_candidates].tolist(),
                        source["reference_cell_type"][train_reference_candidates].tolist(),
                    )
                )
            )
            for sample_id, cell_type in group_keys:
                local = train_reference_candidates[
                    (source["reference_sample_id"][train_reference_candidates] == sample_id)
                    & (source["reference_cell_type"][train_reference_candidates] == cell_type)
                ]
                reference_groups.append(
                    _stable_cap(local, source["reference_id"], REFERENCE_CAP_PER_SAMPLE_TYPE, f"train_reference:{heldout}:{sample_id}:{cell_type}")
                )
            train_reference_rows = np.sort(np.concatenate(reference_groups))
            matched_candidates = np.flatnonzero(
                (source["reference_donor"] == heldout)
                & (source["reference_sample_id"] == source_matched_sample)
                & (source["reference_cell_type"] != "drop")
            )
            if not len(matched_candidates):
                raise ValueError(
                    f"matched snRNA sample is absent: frozen={matched_sample}, source={source_matched_sample}"
                )
            if set(source["reference_position"][matched_candidates].tolist()) != {expected_position}:
                raise ValueError("matched reference sample position differs from the query region")
            matched_groups: list[np.ndarray] = []
            for cell_type in sorted(set(source["reference_cell_type"][matched_candidates].tolist())):
                local = matched_candidates[source["reference_cell_type"][matched_candidates] == cell_type]
                matched_groups.append(
                    _stable_cap(local, source["reference_id"], REFERENCE_CAP_PER_SAMPLE_TYPE, f"matched_reference:{heldout}:{matched_sample}:{cell_type}")
                )
            matched_rows = np.sort(np.concatenate(matched_groups))
            public_path = args.output / "prepared" / "public" / f"{heldout}.npz"
            secret_path = args.output / "prepared" / "secret" / f"{heldout}.npz"
            fold_receipt_path = args.output / "prepared" / "folds" / f"{heldout}.json"
            preparation_identity = hashlib.sha256(
                _json_bytes(
                    {
                        "source_sha256": source_sha,
                        "feature_manifest_identity_sha256": feature_manifest_identity,
                        "panel_sha256": panel_sha,
                        "protocol_sha256": protocol_sha,
                        "heldout": heldout,
                        "query_section": query_section,
                        "matched_source_sample": source_matched_sample,
                        "train_spot_ids_sha256": hashlib.sha256(
                            "\n".join(source["spot_id"][train_rows].tolist()).encode()
                        ).hexdigest(),
                        "query_spot_ids_sha256": hashlib.sha256(
                            "\n".join(source["spot_id"][query_rows].tolist()).encode()
                        ).hexdigest(),
                        "train_reference_ids_sha256": hashlib.sha256(
                            "\n".join(source["reference_id"][train_reference_rows].tolist()).encode()
                        ).hexdigest(),
                        "matched_reference_ids_sha256": hashlib.sha256(
                            "\n".join(source["reference_id"][matched_rows].tolist()).encode()
                        ).hexdigest(),
                        "spot_cap": SPOT_CAP_PER_SECTION,
                        "reference_cap": REFERENCE_CAP_PER_SAMPLE_TYPE,
                    }
                )
            ).hexdigest()
            if (
                args.resume
                and fold_receipt_path.is_file()
                and public_path.is_file()
                and secret_path.is_file()
            ):
                old = json.loads(fold_receipt_path.read_text(encoding="utf-8"))
                if old.get("preparation_identity_sha256") == preparation_identity:
                    old_public, old_secret = _load_npz(public_path), _load_npz(secret_path)
                    if (
                        old.get("public_semantic_sha256") == _semantic_hash(old_public)
                        and old.get("secret_semantic_sha256") == _semantic_hash(old_secret)
                    ):
                        fold_receipts[heldout] = old
                        del old_public, old_secret
                        continue
            if all_spot_counts is None or all_reference_counts is None:
                # At ~200 MB for the frozen 248-gene panel, one sequential read
                # is safer and much faster than HDF5 fancy indexing per fold.
                all_spot_counts = _validated_counts(handle["/spots/counts"][...], "all ST")
                all_reference_counts = _validated_counts(
                    handle["/reference/counts"][...], "all snRNA"
                )
            train_counts = _read_count_rows(all_spot_counts, train_rows, "train ST")
            query_counts = _read_count_rows(all_spot_counts, query_rows, "query ST")
            train_sc_counts = _read_count_rows(all_reference_counts, train_reference_rows, "train snRNA")
            matched_counts = _read_count_rows(all_reference_counts, matched_rows, "matched snRNA")
            train_library = np.asarray(source["spot_library"][train_rows], dtype=np.float32)
            query_library = np.asarray(source["spot_library"][query_rows], dtype=np.float32)
            train_sc_library = np.asarray(source["reference_library"][train_reference_rows], dtype=np.float32)
            matched_library = np.asarray(source["reference_library"][matched_rows], dtype=np.float32)
            for name, counts, library in (
                ("train ST", train_counts, train_library),
                ("query ST", query_counts, query_library),
                ("train snRNA", train_sc_counts, train_sc_library),
                ("matched snRNA", matched_counts, matched_library),
            ):
                if np.any(counts.sum(axis=1) > library + 1.0e-5):
                    raise ValueError(f"{name} panel counts exceed registered full-library exposure")
            # The reused public-fold schema requires reconstructing A/B fields,
            # but this cohort has no molecule-level halves.  Full+zero is an
            # explicit inert placeholder; both diagnostic consumers are
            # disabled in fit_predict below.
            train_half_a = train_counts.copy()
            train_half_b = np.zeros_like(train_counts)
            train_library_a = train_library.copy()
            train_library_b = np.zeros_like(train_library)
            wrong_donors = np.asarray(sorted(set(source["reference_donor"][train_reference_rows].tolist())), dtype=str)
            if heldout in set(wrong_donors.tolist()) or BR2720 in set(wrong_donors.tolist()):
                raise RuntimeError("held-out/Br2720 donor leaked into training reference bank")
            train_features = _selected_features(args, source, train_rows)
            query_features = _selected_features(args, source, query_rows)
            public = {
                "schema": np.asarray(PREPARED_SCHEMA),
                "heldout_donor": np.asarray(heldout),
                "gene_ids": genes,
                "train_spot_ids": source["spot_id"][train_rows],
                "train_donor_ids": source["spot_donor"][train_rows],
                "train_section_ids": source["spot_section"][train_rows],
                "train_indication_ids": np.repeat("DLPFC", len(train_rows)),
                "train_image": train_features,
                "train_coordinates": np.column_stack((source["spot_array_col"][train_rows], source["spot_array_row"][train_rows])).astype(np.float32),
                "train_st_counts": train_counts,
                "train_st_library": train_library,
                "train_st_half_a": train_half_a,
                "train_st_half_b": train_half_b,
                "train_st_library_half_a": train_library_a,
                "train_st_library_half_b": train_library_b,
                "train_sc_counts": train_sc_counts,
                "train_sc_library": train_sc_library,
                "train_sc_cell_ids": source["reference_id"][train_reference_rows],
                "train_sc_donor_ids": source["reference_donor"][train_reference_rows],
                "train_sc_indication_ids": np.repeat("DLPFC", len(train_reference_rows)),
                "train_sc_type_ids": source["reference_cell_type"][train_reference_rows],
                "query_spot_ids": source["spot_id"][query_rows],
                "query_section_ids": source["spot_section"][query_rows],
                "query_indication_ids": np.repeat("DLPFC", len(query_rows)),
                "query_image": query_features,
                "query_coordinates": np.column_stack((source["spot_array_col"][query_rows], source["spot_array_row"][query_rows])).astype(np.float32),
                "query_blank_image": np.zeros_like(query_features),
                "query_shuffle_index": np.roll(np.arange(len(query_rows), dtype=np.int64), 1),
                "matched_sc_counts": matched_counts,
                "matched_sc_library": matched_library,
                "matched_sc_cell_ids": source["reference_id"][matched_rows],
                "matched_sc_donor_ids": source["reference_donor"][matched_rows],
                "matched_sc_indication_ids": np.repeat("DLPFC", len(matched_rows)),
                "matched_sc_type_ids": source["reference_cell_type"][matched_rows],
                "wrong_train_sc_index": np.arange(len(train_reference_rows), dtype=np.int64),
                "wrong_donor_ids": wrong_donors,
                "program_names": program_names,
                "program_gene_membership": program_membership,
                "mechanical_half_status": np.asarray("inert_full_plus_zero_schema_placeholder_never_used_or_scored"),
                "image_scale_qualification": np.asarray("failed_mesoscopic_proxy"),
            }
            secret = {
                "schema": np.asarray(PREPARED_SCHEMA),
                "heldout_donor": np.asarray(heldout),
                "gene_ids": genes,
                "heldout_spot_ids": source["spot_id"][query_rows],
                "heldout_section_ids": source["spot_section"][query_rows],
                "heldout_st_counts": query_counts,
                "heldout_st_library": query_library,
                "primary_score_eligible": query_library > 0,
            }
            _atomic_npz(public_path, public)
            _atomic_npz(secret_path, secret, private=True)
            fold_receipt = {
                "query_section": query_section,
                "matched_reference_frozen_label": matched_sample,
                "matched_reference_source_sample_id": source_matched_sample,
                "matched_position": expected_position,
                "seed": _fold_seed(args.seed, heldout),
                "public_path": str(public_path),
                "public_semantic_sha256": _semantic_hash(public),
                "secret_path": str(secret_path),
                "secret_semantic_sha256": _semantic_hash(secret),
                "train_spots": len(train_rows),
                "query_spots": len(query_rows),
                "train_reference_nuclei": len(train_reference_rows),
                "matched_reference_nuclei": len(matched_rows),
                "heldout_donor_absent_from_all_training_modalities": True,
                "Br2720_absent_from_training_reference_bank": True,
                "preparation_identity_sha256": preparation_identity,
            }
            _atomic_json(fold_receipt_path, fold_receipt)
            fold_receipts[heldout] = fold_receipt
            del public, secret, train_features, query_features
    receipt = {
        "schema": PREPARED_SCHEMA,
        "analysis_scope": ANALYSIS_SCOPE,
        "source_path": str(args.source.resolve()),
        "source_sha256": source_sha,
        "feature_manifest_identity_sha256": feature_manifest_identity,
        "protocol_path": str(args.protocol),
        "protocol_sha256": protocol_sha,
        "panel_path": str(args.panel.resolve()),
        "panel_sha256": panel_sha,
        "panel_gene_count": PANEL_GENE_COUNT,
        "v1_revision_trigger": (
            "preflight_failed_before_export_fit_or_scoring_because_8_of_256_symbols_"
            "were_absent_from_SPE_gene_annotation; no_expression_values_or_losses_used"
        ),
        "base_seed": args.seed,
        "caps": {"spots_per_section": SPOT_CAP_PER_SECTION, "nuclei_per_sample_cell_type": REFERENCE_CAP_PER_SAMPLE_TYPE},
        "query_sections": {donor: values[0] for donor, values in QUERY_SECTIONS.items()},
        "main_clean_donors": list(MAIN_CLEAN_DONORS),
        "sensitivity_donors": list(SENSITIVITY_DONORS),
        "main_clean_rule": "exclude_Br2720_from_scoring_and_all_training_reference_banks",
        "Br2720_inclusion_sensitivity_design": (
            "reuse_the_nine_clean_predictions_and_add_one_Br2720_heldout_fold; "
            "the_other_nine_folds_are_not_refit_with_Br2720_in_the_reference_bank"
        ),
        "target_isolation": "fit_predict_reads_public_npz_only; score_opens_secret_after_predictions_are_sealed",
        "count_half_fields": "inert_full_plus_zero_schema_placeholder; diagnostic_consumer_disabled; M8_disabled; never_scored",
        "folds": fold_receipts,
        "all_folds_complete": len(fold_receipts) == len(SENSITIVITY_DONORS),
    }
    _atomic_json(args.output / "prepared" / "manifest.json", receipt)
    return receipt


def _fold_seed(base_seed: int, donor: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base_seed}:{donor}".encode()).digest()[:4], "little")


def fit_predict(args: argparse.Namespace) -> Mapping[str, object]:
    """Fit LODO folds serially; this function has no path to score-only targets."""

    _, protocol_sha = _load_protocol(args)
    prepared_path = args.output / "prepared" / "manifest.json"
    if not prepared_path.is_file():
        raise FileNotFoundError("run prepare first")
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if int(prepared.get("base_seed", -1)) != int(args.seed):
        raise ValueError("seed differs from the prepared target-isolation contract")
    _validate_hoptimus(args)
    if str(args.device).startswith("cuda"):
        _assert_no_unrelated_gpu_process()
    nat = _import_script(NATCOMM_RUNNER, "heir_spatialdlpfc_natcommun_orchestrator")
    # The reused orchestrator otherwise fits M8 parameters from artificial count
    # halves or computes half-split reliability diagnostics. Disable both paths
    # completely: this cohort has no molecule split.
    nat.fit_m8_cross_half_predictor = lambda *unused_args, **unused_kwargs: {}
    nat.fit_training_diagnostics = lambda *unused_args, **unused_kwargs: {}
    receipts: dict[str, object] = {}
    runner_sha = _sha256(Path(__file__).resolve())
    core_sha = _sha256(Path(core.__file__).resolve())
    for heldout in SENSITIVITY_DONORS:  # no parallel folds
        fold = prepared["folds"][heldout]
        public_path = Path(fold["public_path"])
        public = _load_npz(public_path)
        if _semantic_hash(public) != fold["public_semantic_sha256"]:
            raise ValueError(f"public fold changed after isolation: {heldout}")
        identity = hashlib.sha256(
            _json_bytes(
                {
                    "public_semantic_sha256": fold["public_semantic_sha256"],
                    "seed": fold["seed"],
                    "epochs": args.epochs,
                    "fit_batch_size": args.fit_batch_size,
                    "latent_dim": args.latent_dim,
                    "runner_sha256": runner_sha,
                    "core_sha256": core_sha,
                    "protocol_sha256": protocol_sha,
                    "arms": MODEL_ARMS,
                    "M8": "disabled",
                }
            )
        ).hexdigest()
        prediction_path = args.output / "predictions" / f"{heldout}.npz"
        receipt_path = args.output / "predictions" / f"{heldout}.json"
        if args.resume and prediction_path.is_file() and receipt_path.is_file():
            old = json.loads(receipt_path.read_text(encoding="utf-8"))
            cached = _load_npz(prediction_path)
            if old.get("identity_sha256") == identity and old.get("prediction_semantic_sha256") == _semantic_hash(cached):
                receipts[heldout] = old
                continue
        generated = nat.fit_predict_one_fold(
            public,
            device=str(args.device),
            epochs=int(args.epochs),
            batch_size=int(args.fit_batch_size),
            latent_dim=int(args.latent_dim),
            seed=int(fold["seed"]),
        )
        selected: dict[str, object] = {
            "schema": np.asarray(PREDICTION_SCHEMA),
            "heldout_donor": np.asarray(heldout),
            "query_spot_ids": np.asarray(generated["query_spot_ids"]),
            "gene_ids": np.asarray(generated["gene_ids"]),
            "prediction_scale": np.asarray(generated["prediction_scale"]),
            "training_only_dispersion": np.asarray(generated["training_only_dispersion"], dtype=np.float32),
            "rate_M0": np.asarray(generated["rate_M0"], dtype=np.float32),
            "rate_M3": np.asarray(generated["rate_M3"], dtype=np.float32),
            "rate_M4": np.asarray(generated["rate_M4"], dtype=np.float32),
            "m4_shuffle_index": np.asarray(generated["m4_shuffle_index"], dtype=np.int64),
            "m4_composition_stratum": np.asarray(generated["m4_composition_stratum"], dtype=np.int64),
            "query_H_composition": np.asarray(generated["query_H_composition"], dtype=np.float32),
            "reference_model_type_names": np.asarray(generated["reference_model_type_names"]),
            "matched_observed_type_names": np.asarray(generated["matched_observed_type_names"]),
            "matched_reference_effective_sample_size": np.asarray(generated["matched_reference_effective_sample_size"]),
            "matched_reference_cell_count": np.asarray(generated["matched_reference_cell_count"]),
        }
        if any(name.casefold().startswith(("m8", "rate_m1", "rate_m2", "rate_m5", "rate_m6", "rate_m7", "rate_bleep")) for name in selected):
            raise RuntimeError("non-target model arm escaped the compact prediction artifact")
        _atomic_npz(prediction_path, selected)
        receipt = {
            "schema": PREDICTION_SCHEMA,
            "analysis_scope": ANALYSIS_SCOPE,
            "heldout_donor": heldout,
            "identity_sha256": identity,
            "public_semantic_sha256": fold["public_semantic_sha256"],
            "prediction_path": str(prediction_path),
            "prediction_semantic_sha256": _semantic_hash(selected),
            "heldout_ST_opened": False,
            "arms_retained": list(MODEL_ARMS),
            "M8_and_ST_floor": "disabled_not_computed_not_reported",
            "count_half_diagnostics": "disabled; inert_full_plus_zero_schema_fields_not_consumed",
            "orchestrator_auxiliary_arms": "computed_in_memory_by_reused_frozen_fit_then_discarded",
            "encoder": HOPTIMUS_REPOSITORY,
            "encoder_batch_size": FEATURE_BATCH_SIZE,
            "fit_batch_size": args.fit_batch_size,
            "device": str(args.device),
            "cpu_threads": args.cpu_threads,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "serial_fold_execution": True,
            "artifact_complete": True,
            "protocol_sha256": protocol_sha,
            "panel_gene_count": len(public["gene_ids"]),
        }
        _atomic_json(receipt_path, receipt)
        receipts[heldout] = receipt
        del generated, selected, public
        gc.collect()
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    output = {
        "schema": PREDICTION_SCHEMA,
        "analysis_scope": ANALYSIS_SCOPE,
        "folds": receipts,
        "all_folds_complete": len(receipts) == len(SENSITIVITY_DONORS),
        "serial_fold_execution": True,
        "protocol_sha256": protocol_sha,
    }
    _atomic_json(args.output / "predictions" / "manifest.json", output)
    return output


def _nb2_rows(counts: np.ndarray, mean: np.ndarray, theta: np.ndarray) -> np.ndarray:
    result = core.nb2_deviance(counts, mean, theta, reduction="none")
    if torch.is_tensor(result):
        result = result.detach().cpu().numpy()
    values = np.asarray(result, dtype=np.float64)
    if values.shape != counts.shape:
        raise RuntimeError("NB2 deviance did not return a spot-by-gene matrix")
    return values.mean(axis=1)


def _sign_flip(differences: Sequence[float]) -> Mapping[str, object]:
    result = core.exact_sign_flip_test(differences, alternative="greater")
    return {
        "mean_difference": result.statistic,
        "p_value": result.p_value,
        "confidence_interval": list(result.confidence_interval),
        "positive_fraction": result.positive_fraction,
        "donors": result.donors,
        "status": "descriptive_only_not_registered_inference",
    }


def _aggregate_tests(rows: Sequence[Mapping[str, object]], donors: Sequence[str]) -> Mapping[str, object]:
    selected = {str(row["donor"]): row for row in rows if str(row["donor"]) in set(donors)}
    if set(selected) != set(donors):
        raise ValueError("score rows do not cover the requested donor analysis")
    m0_minus_m3 = [float(selected[donor]["loss_M0"]) - float(selected[donor]["loss_M3"]) for donor in donors]
    m4_minus_m3 = [float(selected[donor]["loss_M4"]) - float(selected[donor]["loss_M3"]) for donor in donors]
    return {
        "donors": list(donors),
        "donor_count": len(donors),
        "M3_vs_M0_positive_favors_M3": _sign_flip(m0_minus_m3),
        "M3_vs_M4_positive_supports_spatially_aligned_morphology": _sign_flip(m4_minus_m3),
        "donor_differences": {
            "M0_minus_M3": dict(zip(donors, m0_minus_m3)),
            "M4_minus_M3": dict(zip(donors, m4_minus_m3)),
        },
    }


def score(args: argparse.Namespace) -> Mapping[str, object]:
    """Open score-only targets only after every fold prediction is sealed."""

    _, protocol_sha = _load_protocol(args)
    prepared = json.loads((args.output / "prepared" / "manifest.json").read_text(encoding="utf-8"))
    predictions = json.loads((args.output / "predictions" / "manifest.json").read_text(encoding="utf-8"))
    if not predictions.get("all_folds_complete"):
        raise RuntimeError("all serial fold predictions must be sealed before scoring")
    rows: list[Mapping[str, object]] = []
    for heldout in SENSITIVITY_DONORS:
        fold = prepared["folds"][heldout]
        secret = _load_npz(Path(fold["secret_path"]))
        if _semantic_hash(secret) != fold["secret_semantic_sha256"]:
            raise ValueError(f"score-only target changed after preparation: {heldout}")
        prediction_path = Path(predictions["folds"][heldout]["prediction_path"])
        prediction = _load_npz(prediction_path)
        if predictions["folds"][heldout]["prediction_semantic_sha256"] != _semantic_hash(prediction):
            raise ValueError(f"prediction changed before target opening: {heldout}")
        if not np.array_equal(prediction["query_spot_ids"].astype(str), secret["heldout_spot_ids"].astype(str)):
            raise ValueError("prediction/target spot order differs")
        if not np.array_equal(prediction["gene_ids"].astype(str), secret["gene_ids"].astype(str)):
            raise ValueError("prediction/target panel order differs")
        eligible = np.asarray(secret["primary_score_eligible"], dtype=bool)
        counts = np.asarray(secret["heldout_st_counts"])[eligible]
        library = np.asarray(secret["heldout_st_library"], dtype=np.float64)[eligible]
        sections = np.asarray(secret["heldout_section_ids"]).astype(str)[eligible]
        theta = np.asarray(prediction["training_only_dispersion"], dtype=np.float64)
        arm_section_losses: dict[str, dict[str, float]] = {}
        arm_donor_losses: dict[str, float] = {}
        for arm in MODEL_ARMS:
            rates = np.asarray(prediction[f"rate_{arm}"], dtype=np.float64)[eligible]
            means = np.maximum(rates * library[:, None], 1.0e-8)
            spot_loss = _nb2_rows(counts, means, theta)
            section_losses = {
                section: float(np.mean(spot_loss[sections == section]))
                for section in sorted(set(sections.tolist()))
            }
            arm_section_losses[arm] = section_losses
            arm_donor_losses[arm] = float(np.mean(list(section_losses.values())))
        rows.append(
            {
                "donor": heldout,
                "query_section": QUERY_SECTIONS[heldout][0],
                "eligible_spots": int(eligible.sum()),
                "aggregation": "spot_gene_mean_NB2_deviance_then_section_mean_then_donor_mean",
                "section_losses": arm_section_losses,
                **{f"loss_{arm}": arm_donor_losses[arm] for arm in MODEL_ARMS},
                "M0_minus_M3": arm_donor_losses["M0"] - arm_donor_losses["M3"],
                "M4_minus_M3": arm_donor_losses["M4"] - arm_donor_losses["M3"],
            }
        )
    output = {
        "schema": SCORE_SCHEMA,
        "analysis_scope": ANALYSIS_SCOPE,
        "endpoint": "training_dispersion_NB2_deviance",
        "gene_panel_size": PANEL_GENE_COUNT,
        "v1_revision_trigger": (
            "annotation_only_preflight_removed_8_unmappable_symbols_before_any_export_fit_or_scoring"
        ),
        "aggregation": "section_then_donor",
        "folds": rows,
        "main_clean": _aggregate_tests(rows, MAIN_CLEAN_DONORS),
        "Br2720_inclusion_sensitivity": _aggregate_tests(rows, SENSITIVITY_DONORS),
        "inference_status": "descriptive_exact_sign_flip_only; target_was_opened_and_analysis_not_registered",
        "ST_floor": "not_estimable_not_computed_not_reported",
        "cell_level_hypotheses": "not_tested",
        "protocol_sha256": protocol_sha,
    }
    _atomic_json(args.output / "scores" / "scores.json", output)
    return output


def report(args: argparse.Namespace) -> str:
    _, protocol_sha = _load_protocol(args)
    scores_path = args.output / "scores" / "scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    feature_manifest_path = args.output / "features" / "manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    section_receipts = tuple(feature_manifest["sections"].values())
    estimated_mpp = [float(value["estimated_microns_per_pixel"]) for value in section_receipts]
    native_field_pixels = [
        float(value["native_pixels_per_nominal_112_um_field"]) for value in section_receipts
    ]
    upsampling = [float(value["upsampling_ratio"]) for value in section_receipts]
    lines = [
        "# spatialDLPFC exploratory biological-hypothesis report (protocol v2)",
        "",
        f"Status: **{ANALYSIS_SCOPE}**.",
        "",
        "This run asks only whether matched snRNA refinement improves an H&E-only predictor "
        "(M3 vs M0), and whether the gain depends on spatially aligned H&E (M3 vs shuffled M4).",
        "",
        "## Annotation-only protocol revision",
        "",
        "Protocol v1 stopped during preflight, before source export, model fitting, or scoring, because 8 of its 256 external panel symbols had no SPE gene-annotation match. Protocol v2 preserves external order and removes only those eight symbols, yielding 248 genes. This compatibility filter used annotation membership only—no expression values, predictions, or losses.",
        "",
        "## Image-proxy qualification",
        "",
        f"Across {len(section_receipts)} embedded hires sections, estimated scale ranges from {min(estimated_mpp):.4g} to {max(estimated_mpp):.4g} um/px. A nominal 112-um field therefore spans only {min(native_field_pixels):.4g} to {max(native_field_pixels):.4g} native pixels, implying {min(upsampling):.4g}x to {max(upsampling):.4g}x scale mismatch relative to H-optimus-1's 0.5 um/px input. Qualification remains failed; features are mesoscopic proxies.",
        "",
        "## Donor-balanced NB2 results",
        "",
        "| Donor | Query section | Spots | M0 | M3 | M4 | M0-M3 | M4-M3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scores["folds"]:
        lines.append(
            f"| {row['donor']} | {row['query_section']} | {row['eligible_spots']} | "
            f"{row['loss_M0']:.6g} | {row['loss_M3']:.6g} | {row['loss_M4']:.6g} | "
            f"{row['M0_minus_M3']:.6g} | {row['M4_minus_M3']:.6g} |"
        )
    lines.extend(["", "## Descriptive paired summaries", ""])
    for label, key in (
        ("Main clean (Br2720 excluded from scoring/reference)", "main_clean"),
        ("Br2720 inclusion sensitivity", "Br2720_inclusion_sensitivity"),
    ):
        value = scores[key]
        first = value["M3_vs_M0_positive_favors_M3"]
        second = value["M3_vs_M4_positive_supports_spatially_aligned_morphology"]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- M0-M3 mean: {first['mean_difference']:.6g}; exact sign-flip p={first['p_value']:.6g}; "
                f"95% descriptive CI=[{first['confidence_interval'][0]:.6g}, {first['confidence_interval'][1]:.6g}]; "
                f"positive fraction={first['positive_fraction']:.3f}; n={first['donors']}.",
                f"- M4-M3 mean: {second['mean_difference']:.6g}; exact sign-flip p={second['p_value']:.6g}; "
                f"95% descriptive CI=[{second['confidence_interval'][0]:.6g}, {second['confidence_interval'][1]:.6g}]; "
                f"positive fraction={second['positive_fraction']:.3f}; n={second['donors']}.",
                "- P-values and intervals are descriptive, not confirmatory inference.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- The processed target and its structural metadata were already opened, and this subset was frozen post-protocol. There is no independent T0 cohort here.",
            "- The v2 248-gene endpoint follows an annotation-only post-opening compatibility revision from the failed 256-gene v1 preflight; it remains nonconfirmatory.",
            "- Embedded Space Ranger hires images are downsampled and not qualified at the frozen H-optimus-1 0.5 um/px scale. Native 224x224 hires crops are mesoscopic regional proxies, not cell morphology.",
            "- The experiment has no nucleus segmentation, cell-resolved molecular ground truth, or verified independent aliquot chain; it cannot test blocked cell-level hypotheses.",
            "- No original molecule_info split exists. Inert full-plus-zero schema placeholders are not consumed; count-half diagnostics and M8 are disabled, and no ST floor is estimated or reported.",
            "- Only H-optimus-1 is accepted. UNI2-h is rejected by the runner.",
            "- All held-out-donor ST and snRNA training rows are excluded; folds run serially with <=4 CPU threads, one GPU, and <=60% per-process GPU memory.",
            "- The 10-donor sensitivity reuses the nine clean predictions and adds the Br2720 held-out fold. The other nine folds are not refit with Br2720 in any reference bank.",
            "- Reusing the frozen NatCommun fit currently computes auxiliary model arms in memory as implementation overhead, then discards them. Only M0/M3/M4 are persisted or scored; M8 is disabled before fitting.",
            "",
            "## Artifact provenance",
            "",
            f"- Score artifact: `{scores_path}`",
            f"- Score SHA-256: `{_sha256(scores_path)}`",
            f"- Source: `{args.source}`",
            f"- Source SHA-256: `{_sha256(args.source)}`",
            f"- Frozen protocol SHA-256: `{protocol_sha}`",
            f"- Feature manifest SHA-256: `{_sha256(feature_manifest_path)}`",
            "",
        ]
    )
    rendered = "\n".join(lines)
    _atomic_text(args.output / "REPORT.md", rendered)
    return rendered


def all_stages(args: argparse.Namespace) -> Mapping[str, object]:
    feature_receipt = features(args)
    prepared = prepare(args)
    predictions = fit_predict(args)
    scores = score(args)
    report(args)
    return {
        "features_complete": bool(feature_receipt["all_sections_complete"]),
        "prepared_complete": bool(prepared["all_folds_complete"]),
        "predictions_complete": bool(predictions["all_folds_complete"]),
        "scores_complete": len(scores["folds"]) == len(SENSITIVITY_DONORS),
        "report_path": str(args.output / "REPORT.md"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("features", "prepare", "fit-predict", "score", "report", "all"))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--encoder", default="hoptimus1")
    parser.add_argument("--encoder-manifest", type=Path, default=DEFAULT_ENCODER_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.60)
    parser.add_argument("--fit-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.panel = args.panel.expanduser().resolve()
    args.protocol = args.protocol.expanduser().resolve()
    args.encoder_manifest = args.encoder_manifest.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if int(args.epochs) < 1 or int(args.latent_dim) < 1:
        raise ValueError("epochs and latent dimension must be positive")
    _load_protocol(args)
    configure_resources(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    handlers = {
        "features": features,
        "prepare": prepare,
        "fit-predict": fit_predict,
        "score": score,
        "report": report,
        "all": all_stages,
    }
    result = handlers[args.stage](args)
    if isinstance(result, Mapping):
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
