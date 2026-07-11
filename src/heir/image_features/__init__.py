"""Streaming multiscale crop and frozen-encoder feature extraction."""

from .extract import CropSpec, FrozenPatchEncoder, extract_multiscale_features

__all__ = ["CropSpec", "FrozenPatchEncoder", "extract_multiscale_features"]
