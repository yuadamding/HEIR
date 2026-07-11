"""Generate cached cell/context embeddings without duplicating WSI crops."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from ..image.slides import SlideBackend
from ..utils import resolve_device


@dataclass(frozen=True)
class CropSpec:
    name: str
    window_um: float
    output_pixels: int = 224

    def __post_init__(self) -> None:
        if not self.name or self.window_um <= 0 or self.output_pixels <= 0:
            raise ValueError("crop name and dimensions must be positive")


class FrozenPatchEncoder:
    """Thin wrapper around any image encoder with a deterministic preprocessor."""

    def __init__(
        self,
        model: nn.Module,
        preprocess: Callable[[Image.Image], Tensor],
        device: str = "auto",
        output_key: Optional[str] = None,
        mixed_precision: Optional[bool] = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.preprocess = preprocess
        self.output_key = output_key
        self.mixed_precision = (
            self.device.type == "cuda" if mixed_precision is None else bool(mixed_precision)
        )

    @torch.no_grad()
    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("images cannot be empty")
        batch = torch.stack([self.preprocess(image) for image in images])
        if self.device.type == "cuda":
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        else:
            batch = batch.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
            enabled=self.mixed_precision,
        ):
            output = self.model(batch)
        if self.output_key is not None:
            if not isinstance(output, dict) or self.output_key not in output:
                raise ValueError("encoder output does not contain %s" % self.output_key)
            output = output[self.output_key]
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, Tensor):
            raise TypeError("image encoder must return a Tensor, tuple, or mapping")
        if output.ndim > 2:
            output = output.flatten(start_dim=2).mean(dim=-1)
        if output.ndim != 2 or output.shape[0] != len(images):
            raise ValueError("encoder output must have shape (images, features)")
        output = torch.nn.functional.normalize(output.float(), dim=-1)
        return output.cpu().numpy().astype(np.float32)


def _read_centered(slide: SlideBackend, center_um: np.ndarray, spec: CropSpec) -> Image.Image:
    half = spec.window_um / 2.0
    origin = center_um - half
    return slide.read_region_um(
        origin_um=origin,
        size_um=(spec.window_um, spec.window_um),
        output_size=(spec.output_pixels, spec.output_pixels),
    )


def extract_multiscale_features(
    slide: SlideBackend,
    centroids_um: np.ndarray,
    encoder: FrozenPatchEncoder,
    crop_specs: Sequence[CropSpec],
    batch_size: int = 64,
    morphology: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Read crops on demand and concatenate normalized embeddings by scale."""

    coordinates = np.asarray(centroids_um, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError("centroids_um must have shape (nuclei, 2)")
    if not crop_specs or batch_size <= 0:
        raise ValueError("at least one crop spec and a positive batch size are required")
    morphology_values = None
    if morphology is not None:
        morphology_values = np.asarray(morphology, dtype=np.float32)
        if morphology_values.ndim != 2 or morphology_values.shape[0] != len(coordinates):
            raise ValueError("morphology must align to nuclei")
    scale_features: List[np.ndarray] = []
    for spec in crop_specs:
        batches: List[np.ndarray] = []
        for start in range(0, len(coordinates), batch_size):
            stop = min(start + batch_size, len(coordinates))
            images = [
                _read_centered(slide, coordinate, spec) for coordinate in coordinates[start:stop]
            ]
            batches.append(encoder.encode(images))
            for image in images:
                image.close()
        scale_features.append(np.concatenate(batches, axis=0))
    if morphology_values is not None:
        mean = morphology_values.mean(axis=0, keepdims=True)
        std = morphology_values.std(axis=0, keepdims=True)
        scale_features.append((morphology_values - mean) / np.maximum(std, 1.0e-6))
    result = np.concatenate(scale_features, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("extracted features contain non-finite values")
    return result
