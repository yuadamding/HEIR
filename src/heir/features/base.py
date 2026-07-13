"""Manifest-bound interfaces for frozen histology patch encoders."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
ENCODER_MANIFEST_SCHEMA = "heir.encoder_manifest.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, name: str, *, allow_unavailable: bool = False) -> str:
    digest = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("encoder manifest %s must be a lowercase SHA-256" % name)
    if digest == "0" * 64 and not allow_unavailable:
        raise ValueError("available encoder manifest %s cannot use the unavailable sentinel" % name)
    return digest


def _triplet(value: object, name: str, *, positive: bool = False) -> Tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("encoder manifest %s must contain three numbers" % name)
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not np.isfinite(result).all():
        raise ValueError("encoder manifest %s must contain three finite numbers" % name)
    if positive and any(item <= 0 for item in result):
        raise ValueError("encoder manifest %s values must be positive" % name)
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class EncoderManifest:
    """Exact encoder identity and preprocessing accepted by one experiment arm."""

    path: Path
    sha256: str
    encoder_id: str
    availability: str
    status_reason: str
    implementation: str
    repository: str
    revision: str
    architecture: str
    checkpoint_filename: str
    checkpoint_sha256: str
    config_filename: str
    config_sha256: str
    feature_width: int
    input_pixels: int
    model_mpp: float
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    interpolation: str
    pooling_rule: str
    license: str
    known_training_datasets: Tuple[str, ...]
    evaluation_overlap: str
    fine_tuning: str

    @property
    def available(self) -> bool:
        return self.availability == "available"


def load_encoder_manifest(path: PathLike, *, require_available: bool = True) -> EncoderManifest:
    """Load a frozen encoder manifest without contacting a model hub."""

    resolved = Path(path).expanduser().resolve()
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("encoder manifest is not valid JSON") from error
    if not isinstance(raw, Mapping) or raw.get("schema") != ENCODER_MANIFEST_SCHEMA:
        raise ValueError("encoder manifest schema is unsupported")
    required = {
        "encoder_id",
        "availability",
        "implementation",
        "repository",
        "revision",
        "architecture",
        "checkpoint_filename",
        "checkpoint_sha256",
        "feature_width",
        "input_pixels",
        "model_mpp",
        "normalization",
        "interpolation",
        "pooling_rule",
        "license",
        "known_training_datasets",
        "evaluation_overlap",
        "fine_tuning",
    }
    if not required <= set(raw):
        raise ValueError("encoder manifest is incomplete")
    availability = str(raw["availability"])
    if availability not in {"available", "inaccessible"}:
        raise ValueError("encoder manifest availability is unsupported")
    status_reason = str(raw.get("status_reason", ""))
    if availability == "inaccessible" and not status_reason.strip():
        raise ValueError("inaccessible encoder manifest requires a status reason")
    if require_available and availability != "available":
        raise ValueError("encoder is inaccessible: %s" % status_reason)
    checkpoint_filename = str(raw["checkpoint_filename"])
    config_filename = str(raw.get("config_filename", ""))
    if Path(checkpoint_filename).name != checkpoint_filename or not checkpoint_filename:
        raise ValueError("encoder manifest checkpoint filename must be a local basename")
    if config_filename and Path(config_filename).name != config_filename:
        raise ValueError("encoder manifest config filename must be a local basename")
    checkpoint_sha = _sha256(
        raw["checkpoint_sha256"],
        "checkpoint_sha256",
        allow_unavailable=availability == "inaccessible",
    )
    config_sha_raw = raw.get("config_sha256", "0" * 64)
    config_sha = _sha256(
        config_sha_raw,
        "config_sha256",
        allow_unavailable=availability == "inaccessible" or not config_filename,
    )
    normalization = raw["normalization"]
    if not isinstance(normalization, Mapping) or not {"mean", "std"} <= set(normalization):
        raise ValueError("encoder manifest normalization is incomplete")
    mean = _triplet(normalization["mean"], "normalization.mean")
    std = _triplet(normalization["std"], "normalization.std", positive=True)
    feature_width = int(raw["feature_width"])
    input_pixels = int(raw["input_pixels"])
    model_mpp = float(raw["model_mpp"])
    if feature_width <= 0 or input_pixels <= 0 or not np.isfinite(model_mpp) or model_mpp <= 0:
        raise ValueError("encoder manifest dimensions are invalid")
    overlap = str(raw["evaluation_overlap"])
    fine_tuning = str(raw["fine_tuning"])
    if overlap not in {"none_known", "possible", "known", "unknown"}:
        raise ValueError("encoder manifest evaluation overlap is unsupported")
    if fine_tuning not in {"prohibited", "development_only", "allowed"}:
        raise ValueError("encoder manifest fine-tuning policy is unsupported")
    training = raw["known_training_datasets"]
    if not isinstance(training, list) or any(not str(value).strip() for value in training):
        raise ValueError("encoder manifest training datasets must be a string list")
    return EncoderManifest(
        path=resolved,
        sha256=sha256_file(resolved),
        encoder_id=str(raw["encoder_id"]),
        availability=availability,
        status_reason=status_reason,
        implementation=str(raw["implementation"]),
        repository=str(raw["repository"]),
        revision=str(raw["revision"]),
        architecture=str(raw["architecture"]),
        checkpoint_filename=checkpoint_filename,
        checkpoint_sha256=checkpoint_sha,
        config_filename=config_filename,
        config_sha256=config_sha,
        feature_width=feature_width,
        input_pixels=input_pixels,
        model_mpp=model_mpp,
        mean=mean,
        std=std,
        interpolation=str(raw["interpolation"]),
        pooling_rule=str(raw["pooling_rule"]),
        license=str(raw["license"]),
        known_training_datasets=tuple(str(value) for value in training),
        evaluation_overlap=overlap,
        fine_tuning=fine_tuning,
    )


class FrozenPatchEncoder(Protocol):
    """Minimal contract used by registered-observation builders."""

    feature_width: int
    manifest_sha256: str

    def encode(self, patches: np.ndarray) -> np.ndarray:
        """Encode an NHWC uint8 batch into a finite two-dimensional feature matrix."""


def verified_model_file(model_dir: PathLike, filename: str, expected_sha256: str) -> Path:
    """Resolve one local model file and reject missing or altered bytes."""

    root = Path(model_dir).expanduser().resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("encoder model file escapes the model directory") from error
    if not candidate.is_file() or sha256_file(candidate) != expected_sha256:
        raise ValueError(
            "encoder model file is missing or differs from its manifest: %s" % candidate
        )
    return candidate


def load_local_state_dict(path: PathLike) -> Mapping[str, object]:
    """Load a local torch or safetensors state dict without hub fallback."""

    resolved = Path(path).expanduser().resolve()
    if resolved.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("install safetensors for this encoder checkpoint") from error
        state = load_file(str(resolved), device="cpu")
    else:
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("install torch for this encoder checkpoint") from error
        state = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("encoder checkpoint does not contain a state dictionary")
    nested = state.get("state_dict")
    return nested if isinstance(nested, Mapping) else state


class TorchPatchEncoder:
    """Shared deterministic preprocessing and output pooling for frozen torch models."""

    def __init__(self, model: object, manifest: EncoderManifest, device: str):
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("install the selected HEIR encoder dependencies") from error
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self._torch = torch
        self._device = torch.device(device)
        self._model = model.eval().to(self._device)
        self._input_pixels = manifest.input_pixels
        self._interpolation = manifest.interpolation
        if self._interpolation not in {"bilinear", "bicubic"}:
            raise ValueError("encoder interpolation must be bilinear or bicubic")
        self._pooling_rule = manifest.pooling_rule
        self._mean = torch.tensor(manifest.mean, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(manifest.std, dtype=torch.float32).view(1, 3, 1, 1)
        self._mean = self._mean.to(self._device)
        self._std = self._std.to(self._device)
        self.feature_width = manifest.feature_width
        self.manifest_sha256 = manifest.sha256

    def _pool(self, output: object):
        torch = self._torch
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise ValueError("encoder output is not a tensor")
        if self._pooling_rule == "direct_features":
            if output.ndim == 3:
                output = output[:, 0]
        elif self._pooling_rule == "cls_token":
            if output.ndim != 3:
                raise ValueError("CLS pooling requires token-level encoder output")
            output = output[:, 0]
        elif self._pooling_rule == "cls_plus_patch_mean":
            if output.ndim != 3 or output.shape[1] < 2:
                raise ValueError("CLS-plus-mean pooling requires token-level encoder output")
            prefix_tokens = int(getattr(self._model, "num_prefix_tokens", 1))
            output = torch.cat((output[:, 0], output[:, prefix_tokens:].mean(1)), dim=-1)
        else:
            raise ValueError("encoder pooling rule is unsupported")
        return output

    def encode(self, patches: np.ndarray) -> np.ndarray:
        torch = self._torch
        values = np.asarray(patches)
        if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
            raise ValueError("encoder patches must be NHWC uint8 RGB")
        tensor = torch.from_numpy(np.ascontiguousarray(values)).permute(0, 3, 1, 2)
        tensor = tensor.to(self._device, dtype=torch.float32, non_blocking=True).div_(255.0)
        if tensor.shape[-2:] != (self._input_pixels, self._input_pixels):
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(self._input_pixels, self._input_pixels),
                mode=self._interpolation,
                align_corners=False,
                antialias=True,
            )
        tensor = (tensor - self._mean) / self._std
        use_amp = self._device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=self._device.type, dtype=torch.float16, enabled=use_amp
        ):
            output = self._pool(self._model(tensor))
        if output.ndim != 2 or output.shape[1] != self.feature_width:
            raise ValueError("encoder output width differs from its manifest")
        result = output.float().cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("encoder produced non-finite features")
        return result
