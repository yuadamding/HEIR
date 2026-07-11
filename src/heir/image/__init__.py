"""Histology access, coordinate calibration, nuclei, and spatial graphs."""

from importlib import import_module

from .coordinates import (
    AffineTransform2D,
    NativePixelMicronTransform,
    PixelMicronTransform,
    microns_to_native_pixels,
    native_pixels_to_microns,
    normalize_mpp,
)
from .graph import (
    EDGE_FEATURE_NAMES,
    SpatialGraph,
    boundary_aware_edge_weights,
    build_nucleus_graph,
    build_spatial_graph,
)
from .nuclei import (
    ConservationError,
    FeatureBundle,
    NucleusTable,
    SpotAggregation,
    SpotAssignment,
    aggregate_nuclei_to_spots,
    aggregate_to_spots,
    assign_nuclei_to_spots,
    assign_nuclei_to_visium_spots,
    canonical_nucleus_ids,
    check_spot_conservation,
    load_feature_bundle,
    load_nuclei,
    load_nucleus_table,
)

_SLIDE_EXPORTS = {
    "SlideBackend",
    "OpenSlideBackend",
    "PILSlideBackend",
    "open_slide",
    "ExposureMetrics",
    "PatchQC",
    "tissue_fraction",
    "blur_score",
    "exposure_metrics",
    "assess_patch_qc",
    "patch_qc",
}

_FEATURE_EXPORTS = {
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
}


def __getattr__(name: str):
    """Load Pillow/OpenSlide-facing symbols only when they are requested."""

    if name in _SLIDE_EXPORTS:
        module = import_module(".slides", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _FEATURE_EXPORTS:
        module = import_module(".features", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()).union(_SLIDE_EXPORTS).union(_FEATURE_EXPORTS))


__all__ = [
    "AffineTransform2D",
    "NativePixelMicronTransform",
    "PixelMicronTransform",
    "normalize_mpp",
    "native_pixels_to_microns",
    "microns_to_native_pixels",
    "SlideBackend",
    "OpenSlideBackend",
    "PILSlideBackend",
    "open_slide",
    "ExposureMetrics",
    "PatchQC",
    "tissue_fraction",
    "blur_score",
    "exposure_metrics",
    "assess_patch_qc",
    "patch_qc",
    "NucleusTable",
    "canonical_nucleus_ids",
    "load_nuclei",
    "load_nucleus_table",
    "FeatureBundle",
    "load_feature_bundle",
    "SpotAssignment",
    "assign_nuclei_to_visium_spots",
    "assign_nuclei_to_spots",
    "ConservationError",
    "SpotAggregation",
    "check_spot_conservation",
    "aggregate_nuclei_to_spots",
    "aggregate_to_spots",
    "EDGE_FEATURE_NAMES",
    "SpatialGraph",
    "boundary_aware_edge_weights",
    "build_spatial_graph",
    "build_nucleus_graph",
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
