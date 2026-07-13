"""Frozen, manifest-bound histology feature encoders."""

from pathlib import Path

from .base import EncoderManifest, FrozenPatchEncoder, load_encoder_manifest
from .h0mini import H0MiniEncoder
from .handcrafted import HandcraftedPatchEncoder
from .hoptimus1 import HOptimus1Encoder
from .uni2h import UNI2HEncoder


def create_frozen_encoder(
    model_dir: Path, manifest: EncoderManifest, device: str = "cuda"
) -> FrozenPatchEncoder:
    """Construct the local adapter selected by an immutable manifest."""

    if manifest.implementation == "uni2h_timm":
        return UNI2HEncoder(model_dir, manifest, device)
    if manifest.repository == "bioptimus/H-optimus-1":
        return HOptimus1Encoder(model_dir, manifest, device)
    if manifest.repository == "bioptimus/H0-mini":
        return H0MiniEncoder(model_dir, manifest, device)
    raise ValueError("encoder manifest does not select a supported local adapter")

__all__ = [
    "EncoderManifest",
    "FrozenPatchEncoder",
    "H0MiniEncoder",
    "HOptimus1Encoder",
    "HandcraftedPatchEncoder",
    "UNI2HEncoder",
    "create_frozen_encoder",
    "load_encoder_manifest",
]
