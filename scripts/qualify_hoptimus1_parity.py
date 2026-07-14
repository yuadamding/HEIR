#!/usr/bin/env python3
"""Qualify the pinned local H-optimus-1 loader against the official Hub loader.

This is an implementation qualification, not a biological endpoint.  It uses a
small deterministic HEST patch suite and writes metrics-only JSON/Markdown
receipts.  Model weights and image patches are never written to the receipt.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from heir.features import load_encoder_manifest
from heir.features.base import EncoderManifest, sha256_file, verified_model_file
from heir.features.hoptimus1 import HOptimus1Encoder

SCHEMA = "heir.hoptimus1_official_local_parity.v1"
REPOSITORY = "bioptimus/H-optimus-1"
REVISION = "3592cb220dec7a150c5d7813fb56e68bd57473b9"
ARCHITECTURE = "vit_giant_patch14_reg4_dinov2"
MANIFEST_SHA256 = "f6852288e1ae146a4865bf19e38ce994c0be9ce1c2bfa09bdf77747043ac8fd9"
CONFIG_SHA256 = "b10c4f37ce804ff58bec7f2ffd35cc29ecbfdb2b96ac81f3c1b3e37e2b27616e"
CHECKPOINT_SHA256 = "c4f1e5b457ddf00679626053b0bf2899be6a19c3a04ad191c87ad1cdfd1abfe1"
README_SHA256 = "43e14486f058782912d63d22ce27e2f1252eaf9e55ae8e2fd8ed416444dbc45e"
HEST_DATASET_REPOSITORY = "MahmoodLab/hest"
HEST_DATASET_REVISION = "7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec"
SOURCE_MPP = 0.2125
FIELD_OF_VIEW_UM = 112.0
MODEL_MPP = 0.5
INPUT_PIXELS = 224
FEATURE_WIDTH = 1536
FP32_MINIMUM_COSINE = 0.999999
FP16_MINIMUM_COSINE = 0.9999
PATCH_CATEGORIES = (
    "natural_hest_tissue",
    "mostly_background",
    "dark_hematoxylin_rich",
    "low_contrast_tissue",
    "padded_border",
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "manifests/encoders/hoptimus1.json"
DEFAULT_MODEL_DIR = Path("/mnt/seagate/HnE/pretrained/H-optimus-1")
DEFAULT_HEST_WSI = Path("/mnt/seagate/HnE/HEST/hest-lung-xenium/wsis/NCBI856.tif")
DEFAULT_HF_CACHE = Path("/mnt/seagate/HnE/pretrained/huggingface-cache")
DEFAULT_OUTPUT_JSON = DEFAULT_MODEL_DIR / "official_local_parity.json"
DEFAULT_OUTPUT_MARKDOWN = DEFAULT_MODEL_DIR / "official_local_parity.md"
PRODUCTION_RUNTIME_FILES = (
    "src/heir/features/__init__.py",
    "src/heir/features/base.py",
    "src/heir/features/hoptimus1.py",
)


@dataclass(frozen=True)
class Candidate:
    """One level-zero WSI location with deterministic image-content scores."""

    x: int
    y: int
    tissue_fraction: float
    mean_luminance: float
    luminance_std: float
    hematoxylin_p90: float
    dark_fraction: float


@dataclass(frozen=True)
class PatchReceipt:
    """Audit metadata for one in-memory patch; pixels are intentionally omitted."""

    category: str
    center_level0_pixels: tuple[int, int]
    source_mpp: float
    field_of_view_um: float
    native_crop_pixels: int
    model_mpp: float
    model_input_pixels: int
    resampling: str
    resampling_count: int
    padding_fraction: float
    tissue_fraction: float
    mean_luminance: float
    luminance_std: float
    hematoxylin_p90: float
    dark_fraction: float
    input_rgb_sha256: str


class TiffPatchReader:
    """Read bounded level-zero regions from a tiled HEST WSI."""

    def __init__(self, path: Path):
        try:
            import tifffile
            import zarr
        except ImportError as error:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("install HEIR with the hest optional dependencies") from error
        self.path = path.expanduser().resolve()
        self._tiff = tifffile.TiffFile(self.path)
        self._store = self._tiff.series[0].aszarr(level=0)
        self._array = zarr.open(self._store, mode="r")
        axes = self._tiff.series[0].axes
        if axes not in {"YXS", "YXC"} or len(self._array.shape) != 3:
            self.close()
            raise ValueError("HEST WSI level zero must be an RGB YXS/YXC image")
        if int(self._array.shape[2]) < 3:
            self.close()
            raise ValueError("HEST WSI must contain at least three colour channels")
        self.height = int(self._array.shape[0])
        self.width = int(self._array.shape[1])

    def read_with_padding(self, center: tuple[int, int], size: int) -> tuple[np.ndarray, float]:
        if size <= 0:
            raise ValueError("crop size must be positive")
        x, y = center
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("patch center is outside the WSI")
        x0 = int(round(x - size / 2.0))
        y0 = int(round(y - size / 2.0))
        x1, y1 = x0 + size, y0 + size
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x1, self.width), min(y1, self.height)
        patch = np.full((size, size, 3), 255, dtype=np.uint8)
        copied = max(sx1 - sx0, 0) * max(sy1 - sy0, 0)
        if sx1 > sx0 and sy1 > sy0:
            patch[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = np.asarray(
                self._array[sy0:sy1, sx0:sx1, :3], dtype=np.uint8
            )
        return patch, 1.0 - copied / float(size * size)

    def close(self) -> None:
        if getattr(self, "_store", None) is not None and hasattr(self._store, "close"):
            self._store.close()
        if getattr(self, "_tiff", None) is not None:
            self._tiff.close()

    def __enter__(self) -> "TiffPatchReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _production_runtime_contract() -> Mapping[str, object]:
    """Bind the adapter runtime and a behavioral probe of production resampling."""

    try:
        import PIL
        from PIL import Image
    except ImportError as error:  # pragma: no cover - required by real qualification
        raise RuntimeError("install Pillow for H-optimus-1 parity qualification") from error
    native_pixels = int(round(FIELD_OF_VIEW_UM / SOURCE_MPP))
    probe = (
        np.arange(native_pixels * native_pixels * 3, dtype=np.uint32).reshape(
            native_pixels, native_pixels, 3
        )
        % 251
    ).astype(np.uint8)
    resized = np.asarray(
        Image.fromarray(probe, mode="RGB").resize(
            (INPUT_PIXELS, INPUT_PIXELS), resample=Image.Resampling.BICUBIC
        ),
        dtype=np.uint8,
    )
    return {
        "code_sha256": {
            relative: sha256_file(REPOSITORY_ROOT / relative)
            for relative in PRODUCTION_RUNTIME_FILES
        },
        "resampling_probe": {
            "input_shape": list(probe.shape),
            "input_dtype": str(probe.dtype),
            "input_sha256": _sha256_bytes(probe),
            "output_shape": list(resized.shape),
            "output_dtype": str(resized.dtype),
            "output_sha256": _sha256_bytes(resized),
            "implementation": "Pillow.Image.Resampling.BICUBIC",
            "pillow_version": PIL.__version__,
            "resampling_count": 1,
        },
    }


def _image_scores(patch: np.ndarray) -> Mapping[str, float]:
    values = np.asarray(patch, dtype=np.float64) / 255.0
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("patch must be HWC RGB")
    luminance = 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    tissue = luminance < 0.92
    optical_density = -np.log(np.clip((values * 255.0 + 1.0) / 256.0, 1.0e-6, 1.0))
    hematoxylin = (
        0.650 * optical_density[..., 0]
        + 0.704 * optical_density[..., 1]
        + 0.286 * optical_density[..., 2]
    )
    return {
        "tissue_fraction": float(tissue.mean()),
        "mean_luminance": float(luminance.mean()),
        "luminance_std": float(luminance.std()),
        "hematoxylin_p90": float(np.quantile(hematoxylin, 0.90)),
        "dark_fraction": float((luminance < 0.45).mean()),
    }


def _native_crop_pixels() -> int:
    value = int(round(FIELD_OF_VIEW_UM / SOURCE_MPP))
    if value <= INPUT_PIXELS:
        raise AssertionError("HEST level-zero crop unexpectedly needs no downsampling")
    return value


def _grid_centers(
    width: int, height: int, crop_pixels: int, grid_size: int
) -> tuple[tuple[int, int], ...]:
    if not 4 <= grid_size <= 24:
        raise ValueError("grid size must be between 4 and 24")
    margin = int(math.ceil(crop_pixels / 2.0))
    if width <= 2 * margin or height <= 2 * margin:
        raise ValueError("HEST WSI is too small for an unpadded 112-um patch grid")
    xs = np.rint(np.linspace(margin, width - margin - 1, grid_size)).astype(int)
    ys = np.rint(np.linspace(margin, height - margin - 1, grid_size)).astype(int)
    return tuple((int(x), int(y)) for y in ys for x in xs)


def _scan_candidates(reader: TiffPatchReader, grid_size: int) -> tuple[Candidate, ...]:
    crop_pixels = _native_crop_pixels()
    result = []
    for center in _grid_centers(reader.width, reader.height, crop_pixels, grid_size):
        patch, padding = reader.read_with_padding(center, crop_pixels)
        if padding != 0.0:
            raise AssertionError("interior qualification grid unexpectedly produced padding")
        result.append(Candidate(x=center[0], y=center[1], **_image_scores(patch)))
    return tuple(result)


def _first_unused(
    candidates: Sequence[Candidate],
    used: set[tuple[int, int]],
    key: Callable[[Candidate], tuple[float, ...]],
    predicate: Callable[[Candidate], bool] = lambda _candidate: True,
) -> Candidate:
    eligible = [
        candidate
        for candidate in candidates
        if (candidate.x, candidate.y) not in used and predicate(candidate)
    ]
    if not eligible:
        raise ValueError("deterministic HEST grid lacks a required patch regime")
    selected = min(eligible, key=key)
    used.add((selected.x, selected.y))
    return selected


def _select_candidates(candidates: Sequence[Candidate]) -> Mapping[str, Candidate]:
    if len(candidates) < 4:
        raise ValueError("at least four interior HEST candidates are required")
    used: set[tuple[int, int]] = set()
    background = _first_unused(
        candidates,
        used,
        lambda value: (
            value.tissue_fraction,
            -value.mean_luminance,
            float(value.y),
            float(value.x),
        ),
    )
    dark = _first_unused(
        candidates,
        used,
        lambda value: (
            -value.hematoxylin_p90,
            -value.dark_fraction,
            float(value.y),
            float(value.x),
        ),
    )
    low_contrast = _first_unused(
        candidates,
        used,
        lambda value: (
            value.luminance_std,
            -value.tissue_fraction,
            float(value.y),
            float(value.x),
        ),
        predicate=lambda value: value.tissue_fraction >= 0.05,
    )
    natural = _first_unused(
        candidates,
        used,
        lambda value: (
            -value.tissue_fraction,
            -value.luminance_std,
            float(value.y),
            float(value.x),
        ),
        predicate=lambda value: value.tissue_fraction >= 0.10,
    )
    return {
        "natural_hest_tissue": natural,
        "mostly_background": background,
        "dark_hematoxylin_rich": dark,
        "low_contrast_tissue": low_contrast,
    }


def _resample_once(patch: np.ndarray) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("install Pillow to construct the parity patch suite") from error
    values = np.asarray(patch)
    expected = _native_crop_pixels()
    if values.shape != (expected, expected, 3) or values.dtype != np.uint8:
        raise ValueError("physical HEST crop must be the native 112-um uint8 RGB canvas")
    return np.asarray(
        Image.fromarray(values, mode="RGB").resize(
            (INPUT_PIXELS, INPUT_PIXELS), resample=Image.Resampling.BICUBIC
        ),
        dtype=np.uint8,
    )


def _patch_receipt(
    category: str,
    center: tuple[int, int],
    patch: np.ndarray,
    resized: np.ndarray,
    padding_fraction: float,
) -> PatchReceipt:
    scores = _image_scores(patch)
    return PatchReceipt(
        category=category,
        center_level0_pixels=center,
        source_mpp=SOURCE_MPP,
        field_of_view_um=FIELD_OF_VIEW_UM,
        native_crop_pixels=_native_crop_pixels(),
        model_mpp=MODEL_MPP,
        model_input_pixels=INPUT_PIXELS,
        resampling="Pillow bicubic",
        resampling_count=1,
        padding_fraction=float(padding_fraction),
        input_rgb_sha256=_sha256_bytes(resized),
        **scores,
    )


def _build_patch_suite(
    reader: TiffPatchReader,
    grid_size: int,
    *,
    resize: Callable[[np.ndarray], np.ndarray] = _resample_once,
) -> tuple[np.ndarray, tuple[PatchReceipt, ...]]:
    selected = _select_candidates(_scan_candidates(reader, grid_size))
    crop_pixels = _native_crop_pixels()
    patches = []
    receipts = []
    for category in PATCH_CATEGORIES[:-1]:
        candidate = selected[category]
        center = (candidate.x, candidate.y)
        native, padding = reader.read_with_padding(center, crop_pixels)
        resized = np.asarray(resize(native), dtype=np.uint8)
        if resized.shape != (INPUT_PIXELS, INPUT_PIXELS, 3):
            raise ValueError("resampler did not produce an exact 224-pixel RGB input")
        patches.append(resized)
        receipts.append(_patch_receipt(category, center, native, resized, padding))
    border_center = (0, 0)
    native, padding = reader.read_with_padding(border_center, crop_pixels)
    if padding <= 0.0:
        raise ValueError("padded-border qualification patch contains no padding")
    resized = np.asarray(resize(native), dtype=np.uint8)
    if resized.shape != (INPUT_PIXELS, INPUT_PIXELS, 3):
        raise ValueError("resampler did not produce an exact 224-pixel RGB input")
    patches.append(resized)
    receipts.append(_patch_receipt(PATCH_CATEGORIES[-1], border_center, native, resized, padding))
    result = np.stack(patches)
    if result.shape != (len(PATCH_CATEGORIES), INPUT_PIXELS, INPUT_PIXELS, 3):
        raise AssertionError("qualification patch suite shape is inconsistent")
    return result, tuple(receipts)


def _validate_contract(manifest: EncoderManifest, model_dir: Path) -> None:
    expected = {
        "sha256": MANIFEST_SHA256,
        "repository": REPOSITORY,
        "revision": REVISION,
        "architecture": ARCHITECTURE,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "feature_width": FEATURE_WIDTH,
        "input_pixels": INPUT_PIXELS,
        "model_mpp": MODEL_MPP,
        "fine_tuning": "prohibited",
    }
    actual = {name: getattr(manifest, name) for name in expected}
    if actual != expected or not manifest.available:
        raise ValueError("H-optimus-1 manifest differs from the frozen parity contract")
    if (
        manifest.checkpoint_filename != "model.safetensors"
        or manifest.config_filename != "config.json"
    ):
        raise ValueError("parity qualification requires config.json and model.safetensors")
    verified_model_file(model_dir, manifest.checkpoint_filename, CHECKPOINT_SHA256)
    verified_model_file(model_dir, manifest.config_filename, CONFIG_SHA256)
    verified_model_file(model_dir, "README.md", README_SHA256)


def _configure_torch() -> object:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - required runtime dependency
        raise RuntimeError("install HEIR's torch dependency") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FP16-versus-FP32 parity arm")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    return torch


def _official_loader(cache_dir: Path, device: str) -> object:
    try:
        import timm
    except ImportError as error:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("install HEIR with the hest optional dependencies") from error
    model = timm.create_model(
        f"hf-hub:{REPOSITORY}@{REVISION}",
        pretrained=True,
        init_values=1.0e-5,
        dynamic_img_size=False,
        cache_dir=str(cache_dir.expanduser().resolve()),
    )
    return model.eval().to(device)


def _local_loader(model_dir: Path, manifest: EncoderManifest, device: str) -> HOptimus1Encoder:
    return HOptimus1Encoder(model_dir, manifest, device=device)


def _encode_model(
    model: object,
    patches: np.ndarray,
    manifest: EncoderManifest,
    precision: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch

    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    if not 1 <= batch_size <= 4:
        raise ValueError("batch size must be between one and four")
    mean = torch.tensor(manifest.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(manifest.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    outputs = []
    for start in range(0, len(patches), batch_size):
        values = np.ascontiguousarray(patches[start : start + batch_size])
        if values.shape[1:] != (INPUT_PIXELS, INPUT_PIXELS, 3) or values.dtype != np.uint8:
            raise ValueError("encoder parity inputs must already be 224-pixel uint8 RGB")
        tensor = torch.from_numpy(values).permute(0, 3, 1, 2).to(device=device)
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
        tensor = (tensor - mean) / std
        use_amp = precision == "fp16"
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp),
        ):
            output = model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise ValueError("H-optimus-1 output is not a tensor")
        if output.ndim == 3:
            output = output[:, 0]
        outputs.append(output.float().cpu().numpy())
    result = np.concatenate(outputs, axis=0)
    _validate_embedding_matrix(result, len(patches))
    return result


def _validate_embedding_matrix(values: np.ndarray, rows: int) -> None:
    array = np.asarray(values)
    if array.shape != (rows, FEATURE_WIDTH):
        raise ValueError("H-optimus-1 output must have finite shape N x 1536")
    if not np.isfinite(array).all():
        raise ValueError("H-optimus-1 output contains non-finite values")


def _release_model(model: object) -> None:
    if isinstance(model, HOptimus1Encoder):
        model._model.to("cpu")
        return
    if hasattr(model, "to"):
        model.to("cpu")


def _clear_allocators() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover
        pass


def _run_sequential_inference(
    patches: np.ndarray,
    load_official: Callable[[], object],
    load_local: Callable[[], object],
    encode: Callable[[object, str], np.ndarray],
    release: Callable[[object], None] = _release_model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, int]]:
    """Load official then local; at most one giant model is live at any time."""

    import torch

    peaks: dict[str, int] = {}
    torch.cuda.reset_peak_memory_stats()
    official = load_official()
    try:
        official_fp32 = encode(official, "fp32")
        peaks["official_fp32_cuda_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    finally:
        release(official)
        del official
        _clear_allocators()
    torch.cuda.reset_peak_memory_stats()
    local = load_local()
    try:
        local_fp32 = encode(local, "fp32")
        local_fp16 = encode(local, "fp16")
        peaks["local_fp32_fp16_cuda_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    finally:
        release(local)
        del local
        _clear_allocators()
    return official_fp32, local_fp32, local_fp16, peaks


def _comparison(first: np.ndarray, second: np.ndarray) -> Mapping[str, float]:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("embedding comparisons require equal two-dimensional arrays")
    numerator = np.sum(first.astype(np.float64) * second.astype(np.float64), axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    if np.any(denominator <= 0.0):
        raise ValueError("embedding cosine is undefined for a zero vector")
    cosine = numerator / denominator
    absolute = np.abs(first.astype(np.float64) - second.astype(np.float64))
    return {
        "minimum_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "mean_absolute_error": float(absolute.mean()),
        "maximum_absolute_error": float(absolute.max()),
    }


def _feature_sha256(values: np.ndarray) -> str:
    return _sha256_bytes(np.asarray(values, dtype="<f4"))


def _maximum_resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if os.uname().sysname != "Darwin" else value


def qualify(args: argparse.Namespace) -> Mapping[str, object]:
    torch = _configure_torch()
    manifest = load_encoder_manifest(args.manifest)
    _validate_contract(manifest, args.model_dir)
    with TiffPatchReader(args.hest_wsi) as reader:
        patches, patch_receipts = _build_patch_suite(reader, args.grid_size)
        wsi_dimensions = [reader.width, reader.height]

    def encode(model: object, precision: str) -> np.ndarray:
        if isinstance(model, HOptimus1Encoder):
            if precision == "fp16":
                values = np.asarray(model.encode(patches), dtype=np.float32)
                _validate_embedding_matrix(values, len(patches))
                return values
            model = model._model
        return _encode_model(model, patches, manifest, precision, args.batch_size, args.device)

    official, local_fp32, local_fp16, gpu_peaks = _run_sequential_inference(
        patches,
        lambda: _official_loader(args.hf_cache_dir, args.device),
        lambda: _local_loader(args.model_dir, manifest, args.device),
        encode,
    )
    _validate_embedding_matrix(official, len(patches))
    _validate_embedding_matrix(local_fp32, len(patches))
    _validate_embedding_matrix(local_fp16, len(patches))
    fp32 = dict(_comparison(official, local_fp32))
    fp16 = dict(_comparison(local_fp32, local_fp16))
    fp32["minimum_required_cosine"] = FP32_MINIMUM_COSINE
    fp16["minimum_required_cosine"] = FP16_MINIMUM_COSINE
    fp32["passed"] = fp32["minimum_cosine"] >= FP32_MINIMUM_COSINE
    fp16["passed"] = fp16["minimum_cosine"] >= FP16_MINIMUM_COSINE
    passed = bool(fp32["passed"] and fp16["passed"])
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "scope": "encoder_implementation_only_not_a_biological_result",
        "passed": passed,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "production_runtime_contract": _production_runtime_contract(),
        "repository": REPOSITORY,
        "revision": REVISION,
        "encoder_manifest_sha256": manifest.sha256,
        "model": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "architecture": ARCHITECTURE,
            "manifest_path": str(Path(args.manifest).expanduser().resolve()),
            "manifest_sha256": manifest.sha256,
            "config_filename": manifest.config_filename,
            "config_sha256": manifest.config_sha256,
            "official_readme_filename": "README.md",
            "official_readme_sha256": README_SHA256,
            "checkpoint_filename": manifest.checkpoint_filename,
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "checkpoint_format": "safetensors",
            "official_loader": f"timm hf-hub:{REPOSITORY}@{REVISION}",
            "local_loader": "heir.features.hoptimus1.HOptimus1Encoder",
            "local_fp16_path": "HOptimus1Encoder.encode_exact_biological_path",
            "fine_tuning": manifest.fine_tuning,
        },
        "hest_patch_suite": {
            "dataset_repository": HEST_DATASET_REPOSITORY,
            "dataset_revision": HEST_DATASET_REVISION,
            "wsi_path": str(Path(args.hest_wsi).expanduser().resolve()),
            "wsi_sha256": sha256_file(Path(args.hest_wsi).expanduser().resolve()),
            "wsi_level0_dimensions": wsi_dimensions,
            "selection_grid_size": args.grid_size,
            "categories": list(PATCH_CATEGORIES),
            "patches": [asdict(value) for value in patch_receipts],
            "pixels_or_model_bytes_in_receipt": False,
        },
        "embedding_contract": {
            "shape": list(official.shape),
            "dtype_at_receipt_boundary": "float32",
            "all_finite": True,
            "official_fp32_sha256": _feature_sha256(official),
            "local_fp32_sha256": _feature_sha256(local_fp32),
            "local_fp16_to_float32_sha256": _feature_sha256(local_fp16),
        },
        "comparisons": {
            "official_fp32_vs_local_fp32": fp32,
            "local_fp32_vs_local_fp16": fp16,
        },
        "resources": {
            "sequential_model_loading": True,
            "simultaneously_live_giant_models": 1,
            "batch_size": args.batch_size,
            "device": args.device,
            "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "tf32_enabled": False,
            "gpu_peak_bytes": gpu_peaks,
            "process_maximum_resident_bytes": _maximum_resident_bytes(),
        },
    }
    return receipt


def _atomic_write(path: Path, text: str) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _markdown(receipt: Mapping[str, object]) -> str:
    model = receipt["model"]
    suite = receipt["hest_patch_suite"]
    comparisons = receipt["comparisons"]
    resources = receipt["resources"]
    runtime = receipt["production_runtime_contract"]
    assert isinstance(model, Mapping)
    assert isinstance(suite, Mapping)
    assert isinstance(comparisons, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(runtime, Mapping)
    fp32 = comparisons["official_fp32_vs_local_fp32"]
    fp16 = comparisons["local_fp32_vs_local_fp16"]
    assert isinstance(fp32, Mapping) and isinstance(fp16, Mapping)
    lines = [
        "# H-optimus-1 embedding parity qualification",
        "",
        f"**Status:** {receipt['status']}",
        "",
        "This receipt qualifies encoder implementation only; it is not a biological result.",
        "",
        "## Frozen identity",
        "",
        f"- Repository/revision: `{model['repository']}@{model['revision']}`",
        f"- Manifest SHA-256: `{model['manifest_sha256']}`",
        f"- Config SHA-256: `{model['config_sha256']}`",
        f"- Safetensors SHA-256: `{model['checkpoint_sha256']}`",
        f"- Official README SHA-256: `{model['official_readme_sha256']}`",
        "- Fine-tuning: prohibited for this qualification",
        "- Production runtime contract SHA-256: `"
        + hashlib.sha256(
            json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        + "`",
        "",
        "## Patch geometry",
        "",
        (
            f"Each patch is a 112-um HEST level-zero crop at {SOURCE_MPP} um/pixel, "
            f"resampled exactly once to {INPUT_PIXELS} x {INPUT_PIXELS} pixels "
            f"({MODEL_MPP} um/pixel)."
        ),
        "",
        "| Patch regime | Padding fraction | Tissue fraction | Input SHA-256 |",
        "|---|---:|---:|---|",
    ]
    patches = suite["patches"]
    assert isinstance(patches, list)
    for patch in patches:
        assert isinstance(patch, Mapping)
        lines.append(
            "| {category} | {padding:.6f} | {tissue:.6f} | `{digest}` |".format(
                category=patch["category"],
                padding=float(patch["padding_fraction"]),
                tissue=float(patch["tissue_fraction"]),
                digest=patch["input_rgb_sha256"],
            )
        )
    lines.extend(
        [
            "",
            "## Parity metrics",
            "",
            "| Comparison | Minimum cosine | Required | MAE | Maximum absolute error | Pass |",
            "|---|---:|---:|---:|---:|---|",
            (
                "| Official FP32 vs local FP32 | {cos:.9f} | {required:.6f} | "
                "{mae:.3e} | {maximum:.3e} | {passed} |"
            ).format(
                cos=float(fp32["minimum_cosine"]),
                required=float(fp32["minimum_required_cosine"]),
                mae=float(fp32["mean_absolute_error"]),
                maximum=float(fp32["maximum_absolute_error"]),
                passed=fp32["passed"],
            ),
            (
                "| Local FP32 vs local FP16 | {cos:.9f} | {required:.4f} | "
                "{mae:.3e} | {maximum:.3e} | {passed} |"
            ).format(
                cos=float(fp16["minimum_cosine"]),
                required=float(fp16["minimum_required_cosine"]),
                mae=float(fp16["mean_absolute_error"]),
                maximum=float(fp16["maximum_absolute_error"]),
                passed=fp16["passed"],
            ),
            "",
            "## Resource boundary",
            "",
            (
                f"Official and local models were loaded sequentially with batch size "
                f"{resources['batch_size']}; at most one giant model was live at a time."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hest-wsi", type=Path, default=DEFAULT_HEST_WSI)
    parser.add_argument("--hf-cache-dir", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--grid-size", type=int, choices=range(4, 25), default=12)
    args = parser.parse_args(argv)
    if args.output_json.expanduser().resolve() == args.output_markdown.expanduser().resolve():
        parser.error("JSON and Markdown outputs must be different files")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    receipt = qualify(args)
    _atomic_write(args.output_json, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _atomic_write(args.output_markdown, _markdown(receipt))
    return 0 if bool(receipt["passed"]) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
