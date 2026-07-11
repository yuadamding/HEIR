"""Frozen CUDA pathology features from bounded, nucleus-centred WSI reads.

The default scientific encoder is OmiCLIP/Loki's CoCa ViT-L/14 visual tower.
It is loaded without the text or multimodal towers so the published checkpoint
fits comfortably on a 10 GB GPU.  A clearly labelled ImageNet ResNet-50
baseline is also available for ablations.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..utils import resolve_device, sha256_file
from .slides import SlideBackend, tissue_fraction

PathLike = Union[str, os.PathLike]

FEATURE_CONTRACT = "heir.nucleus_pathology_features"
FEATURE_CONTRACT_VERSION = 1
OMICLIP_MODEL_NAME = "coca_ViT-L-14"
OMICLIP_ENCODER_NAME = "omiclip-loki-coca-vit-l-14"
IMAGENET_ENCODER_NAME = "torchvision-resnet50-imagenet1k-v2-baseline"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _string_array(values: Sequence[object]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max((len(value) for value in strings), default=1)
    return np.asarray(strings, dtype="<U%d" % width)


@dataclass(frozen=True)
class EncoderDescriptor:
    """Exact identity and input contract for a frozen image encoder."""

    name: str
    architecture: str
    package: str
    package_version: str
    checkpoint_sha256: str
    output_dim: int
    input_size: int
    normalization_mean: Tuple[float, float, float]
    normalization_std: Tuple[float, float, float]
    scientific_role: str

    def __post_init__(self) -> None:
        if not self.name or not self.architecture or not self.package:
            raise ValueError("encoder identity fields cannot be empty")
        if len(self.checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must contain a full SHA-256 digest")
        if self.output_dim <= 0 or self.input_size <= 0:
            raise ValueError("encoder dimensions must be positive")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("encoder normalization must contain three channels")
        if any(value <= 0 for value in self.normalization_std):
            raise ValueError("encoder normalization standard deviations must be positive")


@dataclass(frozen=True)
class ExtractionTelemetry:
    """Measured feature-extraction throughput and bounded-memory telemetry."""

    device: str
    device_name: str
    mixed_precision: bool
    amp_dtype: str
    nuclei: int
    scales: int
    batches: int
    batch_size: int
    model_load_seconds: float
    patch_read_seconds: float
    encode_seconds: float
    total_seconds: float
    nuclei_per_second: float
    images_per_second: float
    steady_state_nuclei_per_second: float
    encoder_images_per_second: float
    peak_cuda_memory_bytes: int


@dataclass(frozen=True)
class ExtractedPathologyFeatures:
    nucleus_ids: np.ndarray
    centroids_px: np.ndarray
    features: np.ndarray
    feature_names: Tuple[str, ...]
    tissue_fraction_by_scale: np.ndarray
    feature_space_id: str
    descriptor: EncoderDescriptor
    patch_diameters_um: Tuple[float, ...]
    native_mpp: Tuple[float, float]
    slide_backend: str
    telemetry: ExtractionTelemetry

    def __post_init__(self) -> None:
        identifiers = _string_array(self.nucleus_ids.tolist())
        centroids = np.asarray(self.centroids_px, dtype=np.float64)
        features = np.asarray(self.features, dtype=np.float32)
        fractions = np.asarray(self.tissue_fraction_by_scale, dtype=np.float32)
        count = len(identifiers)
        if len(set(identifiers.tolist())) != count:
            raise ValueError("nucleus IDs must be unique")
        if centroids.shape != (count, 2) or not np.isfinite(centroids).all():
            raise ValueError("centroids_px must be finite with shape (nuclei, 2)")
        if features.shape != (count, len(self.feature_names)):
            raise ValueError("features and feature_names are inconsistent")
        if not np.isfinite(features).all():
            raise ValueError("pathology features must be finite")
        if fractions.shape != (count, len(self.patch_diameters_um)):
            raise ValueError("tissue fractions must have one column per scale")
        if not np.isfinite(fractions).all() or np.any(fractions < 0) or np.any(fractions > 1):
            raise ValueError("tissue fractions must lie in [0, 1]")
        object.__setattr__(self, "nucleus_ids", identifiers)
        object.__setattr__(self, "centroids_px", centroids)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "tissue_fraction_by_scale", fractions)


class _NormalizedVisualTower(nn.Module):
    """Return only the normalized pooled vector from a frozen visual tower."""

    def __init__(self, tower: nn.Module) -> None:
        super().__init__()
        self.tower = tower

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.tower(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise RuntimeError("image encoder did not return a two-dimensional tensor")
        # Normalize in float32 even under AMP. This limits numerical drift in
        # the stored cross-slide feature space while retaining fast FP16 tower
        # execution.
        return F.normalize(output.float(), p=2, dim=-1)


def _torch_load_checkpoint(path: Path, trust_checkpoint: bool) -> Mapping[str, Any]:
    """Load tensor checkpoints safely unless a raw published archive is trusted.

    The published Loki archive also contains optimizer/scaler NumPy metadata,
    which torch 2.3 cannot load in weights-only mode.  Unsafe pickle loading is
    therefore never implicit and requires the dedicated CLI flag.
    """

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except (pickle.UnpicklingError, RuntimeError) as error:
        if not trust_checkpoint:
            raise RuntimeError(
                "checkpoint is not weights-only safe; use --trust-checkpoint only for a "
                "checkpoint obtained from a verified source"
            ) from error
        payload = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise ValueError("encoder checkpoint root must be a mapping")
    return payload


def _state_dict(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    candidate: Any = payload.get("state_dict", payload)
    if not isinstance(candidate, Mapping):
        raise ValueError("encoder checkpoint has no state_dict mapping")
    if not candidate or not all(isinstance(value, torch.Tensor) for value in candidate.values()):
        raise ValueError("encoder state_dict must contain tensors only")
    return candidate


def load_omiclip_visual_encoder(
    checkpoint: PathLike,
    *,
    device: Union[str, torch.device] = "auto",
    input_size: int = 224,
    trust_checkpoint: bool = False,
) -> Tuple[nn.Module, EncoderDescriptor]:
    """Load only Loki/OmiCLIP's pretrained CoCa ViT-L/14 visual tower."""

    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    if input_size != 224:
        raise ValueError("the published OmiCLIP checkpoint requires input_size=224")
    try:
        import open_clip  # type: ignore
        from open_clip.model import _build_vision_tower  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "OmiCLIP extraction requires open-clip-torch==2.26.1; install the pathology extra"
        ) from error

    package_version = importlib.metadata.version("open-clip-torch")
    if package_version != "2.26.1":
        raise RuntimeError(
            "OmiCLIP visual-only loading is pinned to open-clip-torch==2.26.1, found %s"
            % package_version
        )
    model_config = open_clip.get_model_config(OMICLIP_MODEL_NAME)
    if not isinstance(model_config, Mapping):
        raise RuntimeError("open_clip does not provide the expected OmiCLIP architecture")
    vision_config = dict(model_config["vision_cfg"])
    vision_config["image_size"] = int(input_size)
    output_dim = int(model_config["embed_dim"])
    tower = _build_vision_tower(output_dim, vision_config, quick_gelu=False, cast_dtype=None)

    payload = _torch_load_checkpoint(source, trust_checkpoint=trust_checkpoint)
    raw_state = _state_dict(payload)
    visual_state: Dict[str, torch.Tensor] = {}
    for raw_name, value in raw_state.items():
        name = str(raw_name)
        if name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("visual."):
            visual_state[name[len("visual.") :]] = value
        elif str(payload.get("schema", "")).startswith("heir.omiclip_visual"):
            visual_state[name] = value
    if not visual_state:
        raise ValueError("checkpoint contains no OmiCLIP visual-tower parameters")
    incompatible = tower.load_state_dict(visual_state, strict=True)
    if (
        incompatible.missing_keys or incompatible.unexpected_keys
    ):  # pragma: no cover - strict raises
        raise ValueError("OmiCLIP visual checkpoint does not match the pinned architecture")
    del visual_state, raw_state, payload

    resolved_device = resolve_device(str(device))
    model = _NormalizedVisualTower(tower).eval().requires_grad_(False).to(resolved_device)
    if resolved_device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    descriptor = EncoderDescriptor(
        name=OMICLIP_ENCODER_NAME,
        architecture=OMICLIP_MODEL_NAME,
        package="open-clip-torch",
        package_version=package_version,
        checkpoint_sha256=sha256_file(source),
        output_dim=output_dim,
        input_size=int(input_size),
        normalization_mean=CLIP_MEAN,
        normalization_std=CLIP_STD,
        scientific_role="pretrained histology-to-spatial-transcriptomics foundation encoder",
    )
    return model, descriptor


def load_imagenet_resnet50_encoder(
    *,
    device: Union[str, torch.device] = "auto",
    input_size: int = 224,
) -> Tuple[nn.Module, EncoderDescriptor]:
    """Load the explicit ImageNet-only baseline (not a pathology foundation model)."""

    try:
        import torchvision  # type: ignore
        from torchvision.models import ResNet50_Weights, resnet50  # type: ignore
    except ImportError as error:
        raise RuntimeError("the ImageNet baseline requires torchvision") from error
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    output_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    checkpoint_name = Path(weights.url).name
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    if not checkpoint_path.is_file():  # pragma: no cover - torchvision just downloaded it
        raise RuntimeError("torchvision did not retain its ImageNet checkpoint in the hub cache")
    resolved_device = resolve_device(str(device))
    wrapped = _NormalizedVisualTower(model).eval().requires_grad_(False).to(resolved_device)
    if resolved_device.type == "cuda":
        wrapped = wrapped.to(memory_format=torch.channels_last)
    descriptor = EncoderDescriptor(
        name=IMAGENET_ENCODER_NAME,
        architecture="resnet50",
        package="torchvision",
        package_version=torchvision.__version__,
        checkpoint_sha256=sha256_file(checkpoint_path),
        output_dim=output_dim,
        input_size=int(input_size),
        normalization_mean=IMAGENET_MEAN,
        normalization_std=IMAGENET_STD,
        scientific_role="ImageNet-only baseline; not pathology pretrained",
    )
    return wrapped, descriptor


def pathology_feature_space_id(
    descriptor: EncoderDescriptor,
    patch_diameters_um: Sequence[float],
    *,
    mixed_precision: bool,
) -> str:
    """Hash every preprocessing choice that can change the stored features."""

    scales = tuple(float(value) for value in patch_diameters_um)
    if not scales or any(not np.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("patch diameters must be finite and positive")
    payload = {
        "contract": FEATURE_CONTRACT,
        "contract_version": FEATURE_CONTRACT_VERSION,
        "descriptor": asdict(descriptor),
        "patch_diameters_um": scales,
        "resize": "bounded-wsi-read+bicubic-to-square",
        "pooled_feature_normalization": "l2-float32-per-scale",
        "scale_fusion": "concatenate-in-declared-order",
        "inference_precision": "cuda-fp16-amp" if mixed_precision else "float32",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return "%s:sha256:%s" % (descriptor.name, digest)


def _level_for_patch(slide: SlideBackend, diameter_um: float, output_size: int) -> int:
    native_width = diameter_um / slide.mpp_x
    native_height = diameter_um / slide.mpp_y
    ideal_downsample = math.sqrt(native_width * native_height) / float(output_size)
    # Pick the closest pyramid level in log space. Reads stay bounded even for
    # a context field spanning thousands of native pixels.
    return min(
        range(slide.level_count),
        key=lambda index: abs(
            math.log(max(slide.level_downsamples[index], 1.0e-12))
            - math.log(max(ideal_downsample, 1.0e-12))
        ),
    )


def _read_centered_patch(
    slide: SlideBackend,
    centroid_px: np.ndarray,
    diameter_um: float,
    output_size: int,
) -> np.ndarray:
    native_width = diameter_um / slide.mpp_x
    native_height = diameter_um / slide.mpp_y
    x0 = float(centroid_px[0]) - 0.5 * native_width
    y0 = float(centroid_px[1]) - 0.5 * native_height
    level = _level_for_patch(slide, diameter_um, output_size)
    downsample = slide.level_downsamples[level]
    read_width = max(1, int(math.ceil(native_width / downsample)))
    read_height = max(1, int(math.ceil(native_height / downsample)))
    patch = slide.read_region((x0, y0), level, (read_width, read_height), mode="RGB")
    if patch.size != (output_size, output_size):
        from PIL import Image

        resampling = getattr(Image, "Resampling", Image).BICUBIC
        patch = patch.resize((output_size, output_size), resample=resampling)
    pixels = np.asarray(patch, dtype=np.uint8).copy()
    # OpenSlide/Pillow pad out-of-bounds regions with black. Histology glass is
    # white, so explicitly repair only the mathematically out-of-slide area.
    left = int(math.ceil(max(0.0, -x0) / native_width * output_size))
    top = int(math.ceil(max(0.0, -y0) / native_height * output_size))
    right = int(
        math.ceil(max(0.0, x0 + native_width - slide.dimensions[0]) / native_width * output_size)
    )
    bottom = int(
        math.ceil(max(0.0, y0 + native_height - slide.dimensions[1]) / native_height * output_size)
    )
    if left:
        pixels[:, : min(left, output_size)] = 255
    if right:
        pixels[:, max(0, output_size - right) :] = 255
    if top:
        pixels[: min(top, output_size), :] = 255
    if bottom:
        pixels[max(0, output_size - bottom) :, :] = 255
    return pixels


def _devices_compatible(actual: torch.device, requested: torch.device) -> bool:
    """Treat CUDA's implicit current-device spelling as its indexed spelling."""

    if actual.type != requested.type:
        return False
    return actual.index is None or requested.index is None or actual.index == requested.index


def extract_nucleus_pathology_features(
    slide: SlideBackend,
    nucleus_ids: Sequence[object],
    centroids_px: np.ndarray,
    encoder: nn.Module,
    descriptor: EncoderDescriptor,
    *,
    patch_diameters_um: Sequence[float] = (32.0, 128.0),
    batch_size: int = 64,
    device: Union[str, torch.device] = "auto",
    mixed_precision: Optional[bool] = None,
    model_load_seconds: float = 0.0,
) -> ExtractedPathologyFeatures:
    """Extract multi-scale features without materializing a whole slide image."""

    identifiers = _string_array(nucleus_ids)
    coordinates = np.asarray(centroids_px, dtype=np.float64)
    if coordinates.shape != (len(identifiers), 2) or not np.isfinite(coordinates).all():
        raise ValueError("centroids_px must be finite with shape (nuclei, 2)")
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("nucleus IDs must be unique")
    scales = tuple(float(value) for value in patch_diameters_um)
    if not scales or any(not np.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("patch diameters must be finite and positive")
    if len(set(scales)) != len(scales):
        raise ValueError("patch diameters must be unique")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resolved_device = resolve_device(str(device))
    use_amp = resolved_device.type == "cuda" if mixed_precision is None else bool(mixed_precision)
    if use_amp and resolved_device.type != "cuda":
        raise ValueError("mixed precision is currently supported only on CUDA")
    try:
        model_device = next(encoder.parameters()).device
    except StopIteration:
        model_device = resolved_device
    if not _devices_compatible(model_device, resolved_device):
        raise ValueError("encoder parameters are not on the requested extraction device")

    count = len(identifiers)
    feature_width = len(scales) * descriptor.output_dim
    output = np.empty((count, feature_width), dtype=np.float32)
    fractions = np.empty((count, len(scales)), dtype=np.float32)
    mean = torch.tensor(descriptor.normalization_mean, device=resolved_device).view(1, 3, 1, 1)
    std = torch.tensor(descriptor.normalization_std, device=resolved_device).view(1, 3, 1, 1)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    extraction_start = time.perf_counter()
    read_seconds = 0.0
    encode_seconds = 0.0
    batches = 0

    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            read_start = time.perf_counter()
            pixels = np.empty(
                ((stop - start) * len(scales), descriptor.input_size, descriptor.input_size, 3),
                dtype=np.uint8,
            )
            for local_index, centroid in enumerate(coordinates[start:stop]):
                for scale_index, diameter_um in enumerate(scales):
                    patch = _read_centered_patch(
                        slide,
                        centroid,
                        diameter_um,
                        descriptor.input_size,
                    )
                    flat_index = local_index * len(scales) + scale_index
                    pixels[flat_index] = patch
                    fractions[start + local_index, scale_index] = tissue_fraction(patch)
            read_seconds += time.perf_counter() - read_start

            encode_start = time.perf_counter()
            host_pixels = torch.from_numpy(pixels)
            if resolved_device.type == "cuda":
                host_pixels = host_pixels.pin_memory()
            images = host_pixels.to(resolved_device, non_blocking=resolved_device.type == "cuda")
            images = images.permute(0, 3, 1, 2).float().div_(255.0)
            images.sub_(mean).div_(std)
            if resolved_device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                encoded = encoder(images)
            if encoded.shape != ((stop - start) * len(scales), descriptor.output_dim):
                raise RuntimeError(
                    "encoder output shape %s does not match descriptor width %d"
                    % (tuple(encoded.shape), descriptor.output_dim)
                )
            fused = encoded.reshape(stop - start, len(scales) * descriptor.output_dim)
            output[start:stop] = fused.cpu().numpy().astype(np.float32, copy=False)
            if resolved_device.type == "cuda":
                torch.cuda.synchronize(resolved_device)
            encode_seconds += time.perf_counter() - encode_start
            batches += 1

    total_seconds = time.perf_counter() - extraction_start + float(model_load_seconds)
    device_name = (
        torch.cuda.get_device_name(resolved_device) if resolved_device.type == "cuda" else "CPU"
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(resolved_device))
        if resolved_device.type == "cuda"
        else 0
    )
    telemetry = ExtractionTelemetry(
        device=str(resolved_device),
        device_name=device_name,
        mixed_precision=use_amp,
        amp_dtype="float16" if use_amp else "disabled",
        nuclei=count,
        scales=len(scales),
        batches=batches,
        batch_size=batch_size,
        model_load_seconds=float(model_load_seconds),
        patch_read_seconds=read_seconds,
        encode_seconds=encode_seconds,
        total_seconds=total_seconds,
        nuclei_per_second=count / max(total_seconds, 1.0e-12),
        images_per_second=(count * len(scales)) / max(total_seconds, 1.0e-12),
        steady_state_nuclei_per_second=count / max(read_seconds + encode_seconds, 1.0e-12),
        encoder_images_per_second=(count * len(scales)) / max(encode_seconds, 1.0e-12),
        peak_cuda_memory_bytes=peak_memory,
    )
    names = tuple(
        "%s_%gum_%04d" % (descriptor.name, diameter, feature_index)
        for diameter in scales
        for feature_index in range(descriptor.output_dim)
    )
    return ExtractedPathologyFeatures(
        nucleus_ids=identifiers,
        centroids_px=coordinates,
        features=output,
        feature_names=names,
        tissue_fraction_by_scale=fractions,
        feature_space_id=pathology_feature_space_id(
            descriptor,
            scales,
            mixed_precision=use_amp,
        ),
        descriptor=descriptor,
        patch_diameters_um=scales,
        native_mpp=slide.native_mpp,
        slide_backend=slide.backend_name,
        telemetry=telemetry,
    )


def with_peak_memory(
    result: ExtractedPathologyFeatures,
    peak_cuda_memory_bytes: int,
) -> ExtractedPathologyFeatures:
    """Replace telemetry with a peak measured across model load and extraction."""

    return replace(
        result,
        telemetry=replace(
            result.telemetry,
            peak_cuda_memory_bytes=int(peak_cuda_memory_bytes),
        ),
    )


def save_pathology_feature_npz(
    result: ExtractedPathologyFeatures,
    path: PathLike,
    *,
    slide_sha256: str,
    nuclei_sha256: str,
    overwrite: bool = False,
) -> Path:
    """Atomically persist a pickle-free, self-identifying feature bundle."""

    if len(slide_sha256) != 64 or len(nuclei_sha256) != 64:
        raise ValueError("source SHA-256 digests must be complete")
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("pathology feature output must use .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(str(destination))
    descriptor_json = json.dumps(
        asdict(result.descriptor), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    telemetry_json = json.dumps(
        asdict(result.telemetry), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    arrays: Dict[str, np.ndarray] = {
        "__contract__": np.asarray(FEATURE_CONTRACT),
        "__version__": np.asarray(FEATURE_CONTRACT_VERSION, dtype=np.int64),
        "nucleus_ids": result.nucleus_ids,
        "features": result.features,
        "feature_names": _string_array(result.feature_names),
        "centroids_px": result.centroids_px,
        "tissue_fraction": result.tissue_fraction_by_scale.min(axis=1),
        "tissue_fraction_by_scale": result.tissue_fraction_by_scale,
        "feature_space_id": np.asarray(result.feature_space_id),
        "encoder_descriptor_json": np.asarray(descriptor_json),
        "patch_diameters_um": np.asarray(result.patch_diameters_um, dtype=np.float64),
        "native_mpp": np.asarray(result.native_mpp, dtype=np.float64),
        "slide_backend": np.asarray(result.slide_backend),
        "slide_sha256": np.asarray(slide_sha256),
        "nuclei_sha256": np.asarray(nuclei_sha256),
        "telemetry_json": np.asarray(telemetry_json),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".npz.tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            # Encoder embeddings are poorly compressible. An uncompressed NPZ
            # avoids a long single-threaded CPU bottleneck while remaining a
            # safe, mmap-free NumPy archive.
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "FEATURE_CONTRACT",
    "FEATURE_CONTRACT_VERSION",
    "OMICLIP_ENCODER_NAME",
    "IMAGENET_ENCODER_NAME",
    "EncoderDescriptor",
    "ExtractionTelemetry",
    "ExtractedPathologyFeatures",
    "load_omiclip_visual_encoder",
    "load_imagenet_resnet50_encoder",
    "pathology_feature_space_id",
    "extract_nucleus_pathology_features",
    "with_peak_memory",
    "save_pathology_feature_npz",
]
