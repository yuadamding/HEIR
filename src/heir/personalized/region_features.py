"""Frozen H&E region features bound to image, grid, and local encoder identities.

Only image pixels enter this cache. Registered pixels have no implied physical
calibration, even when the encoder manifest specifies its preferred model MPP.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from heir.features.base import (
    EncoderManifest,
    FrozenPatchEncoder,
    load_encoder_manifest,
    sha256_file,
)

from .regions import RegionGrid, _rgb_image

_SCHEMA = "heir.frozen_region_features.v1"
_ARRAYS = {"features", "region_ids", "coordinates", "metadata_json"}
_RGB_SETTINGS = {"encoder_name", "encoder_manifest_sha256"}
_CROP = "selected_half_open_grid_bins_clipped_to_image_no_padding_no_flip"
_RESIZE = "encoder_resizes_each_actual_crop_once_if_needed_to_square_model_input"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _geometry(grid):
    settings = {key: value for key, value in grid.settings.items() if key not in _RGB_SETTINGS}
    shape, size = settings.get("image_shape", []), settings.get("bin_size_pixels")
    keys, coordinates = np.asarray(grid.bin_keys), np.asarray(grid.coordinates)
    fractions, ids = np.asarray(grid.tissue_fraction), tuple(grid.region_ids)
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or len(shape) != 2
        or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in shape)
        or settings.get("schema") != "heir.he_region_grid.v1"
        or settings.get("prediction_unit") != "tissue_region"
        or settings.get("coordinate_units") != "registered_pixel"
        or settings.get("origin_pixels") != [0, 0]
        or settings.get("array_row_is_y") is not True
        or settings.get("display_origin") != "lower"
        or settings.get("edge_bins") != "clip_to_image_no_padding"
        or settings.get("contains_spatial_rna_derived_features") is not False
        or settings.get("contains_protein_derived_features") is not False
        or not ids
        or len(set(ids)) != len(ids)
        or keys.shape != (len(ids), 2)
        or keys.dtype.kind not in "iu"
        or coordinates.shape != (len(ids), 2)
        or not np.isfinite(coordinates).all()
        or fractions.shape != (len(ids),)
        or not np.isfinite(fractions).all()
        or np.any((fractions < 0) | (fractions > 1))
        or len(grid.component_ids) != len(ids)
        or any(not isinstance(v, str) or not v.strip() for v in grid.component_ids)
    ):
        raise ValueError("invalid H&E region grid geometry or provenance")
    extent = np.asarray(shape[::-1])
    if np.any(keys < 0) or np.any(keys >= (extent + size - 1) // size):
        raise ValueError("H&E region grid bin lies outside the declared image")
    lower = keys * size
    upper = np.minimum(lower + size, extent)
    centers = (lower + upper) / 2
    if (
        ids != tuple(f"he_region:{x}:{y}" for x, y in keys)
        or len(set(map(tuple, keys))) != len(ids)
        or not np.array_equal(coordinates, centers)
    ):
        raise ValueError("H&E region IDs/coordinates do not match their clipped grid bins")
    result = {
        "settings": settings,
        "region_ids": list(ids),
        "coordinates": coordinates.tolist(),
        "bin_keys": keys.tolist(),
        "component_ids": list(grid.component_ids),
        "tissue_fraction": fractions.tolist(),
    }
    _json(result)  # Reject nonfinite or non-JSON settings before hashing them.
    return result


def grid_identity(grid: RegionGrid) -> str:
    """Hash ordered finite geometry, excluding the replaceable RGB feature identity."""
    return hashlib.sha256(_json(_geometry(grid)).encode("utf-8")).hexdigest()


def _feature_digest(features):
    return hashlib.sha256(features.tobytes(order="C")).hexdigest()


def write_feature_cache(
    image_path,
    grid: RegionGrid,
    output_npz,
    encoder: FrozenPatchEncoder,
    manifest: EncoderManifest,
    *,
    expected_image_sha256: str,
    batch_size: int = 4,
) -> dict:
    """Encode the actual selected, clipped H&E bins and exclusively create a cache.

    The existing encoder performs its declared resize and normalization. Equal
    crop shapes share batches; partial edge crops are never padded or flipped.
    Checkpoint/config verification belongs to the local encoder factory caller.
    """
    expected_image_sha256 = _digest(expected_image_sha256, "expected image digest")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    output = Path(output_npz)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"feature cache already exists: {output}")
    if not isinstance(manifest, EncoderManifest) or manifest != load_encoder_manifest(
        manifest.path
    ):
        raise ValueError("encoder manifest object differs from its verified source file")
    if (
        encoder.manifest_sha256 != manifest.sha256
        or encoder.feature_width != manifest.feature_width
    ):
        raise ValueError("encoder identity or feature width differs from its manifest")
    geometry = _geometry(grid)
    identity = grid_identity(grid)
    image_path = Path(image_path)
    if sha256_file(image_path) != expected_image_sha256:
        raise ValueError("H&E image checksum differs from its expected source")
    image = _rgb_image(image_path)
    if list(image.shape[:2]) != geometry["settings"]["image_shape"]:
        raise ValueError("actual H&E image dimensions differ from the region grid")
    size = geometry["settings"]["bin_size_pixels"]
    height, width = image.shape[:2]
    batches = {}
    for row, (x, y) in enumerate(grid.bin_keys):
        x, y = int(x * size), int(y * size)
        bounds = (x, y, min(x + size, width), min(y + size, height))
        crop_shape = (bounds[3] - y, bounds[2] - x)
        batches.setdefault(crop_shape, []).append((row, bounds))
    features = np.empty((len(grid.region_ids), manifest.feature_width), dtype=np.float32)
    for members in batches.values():
        for start in range(0, len(members), batch_size):
            batch = members[start : start + batch_size]
            patches = np.stack([image[y:top, x:right] for _, (x, y, right, top) in batch])
            values = np.asarray(encoder.encode(patches))
            if (
                values.shape != (len(batch), manifest.feature_width)
                or values.dtype.kind != "f"
                or not np.isfinite(values).all()
            ):
                raise ValueError("encoder returned invalid region feature shape, dtype, or values")
            features[[row for row, _ in batch]] = values
    if not np.isfinite(features).all():
        raise ValueError("region features are not finite after float32 conversion")
    if sha256_file(image_path) != expected_image_sha256 or grid_identity(grid) != identity:
        raise ValueError("H&E image or region grid changed during encoding")
    manifest_text = manifest.path.read_bytes().decode("utf-8")
    if hashlib.sha256(manifest_text.encode("utf-8")).hexdigest() != manifest.sha256:
        raise ValueError("encoder manifest changed during encoding")
    metadata = {
        "schema": _SCHEMA,
        "image_sha256": expected_image_sha256,
        "grid_sha256": identity,
        "encoder_manifest_sha256": manifest.sha256,
        "encoder_manifest_json": manifest_text,
        "encoder": json.loads(manifest_text),
        "frozen": True,
        "feature_dim": manifest.feature_width,
        "feature_shape": list(features.shape),
        "feature_dtype": "float32",
        "feature_sha256": _feature_digest(features),
        "feature_hash_encoding": "float32_C_order_raw_bytes",
        "batch_size": batch_size,
        "prediction_unit": "tissue_region",
        "coordinate_units": "registered_pixel",
        "crop_definition": _CROP,
        "array_row_is_y": True,
        "native_crop_shapes": [list(s) for s in sorted(batches)],
        "input_pixels": manifest.input_pixels,
        "interpolation": manifest.interpolation,
        "resize": _RESIZE,
        "preferred_model_mpp": manifest.model_mpp,
        "physical_scale_verified": False,
        "contains_spatial_rna_derived_features": False,
        "contains_protein_derived_features": False,
    }
    metadata_json = _json(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        try:
            np.savez_compressed(
                handle,
                features=features,
                region_ids=np.asarray(grid.region_ids, dtype=str),
                coordinates=np.asarray(grid.coordinates, dtype=np.float32),
                metadata_json=np.asarray(metadata_json),
            )
        except BaseException:
            output.unlink()
            raise
    return metadata


def load_feature_cache(
    path,
    grid: RegionGrid,
    *,
    expected_image_sha256: str,
    expected_encoder_manifest_sha256: str,
) -> tuple[np.ndarray, dict]:
    """Load exactly the declared ordered region axis; never fall back to RGB features."""
    expected_image_sha256 = _digest(expected_image_sha256, "expected image digest")
    expected_encoder_manifest_sha256 = _digest(
        expected_encoder_manifest_sha256, "expected encoder manifest digest"
    )
    identity = grid_identity(grid)
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(_ARRAYS) or set(archive.files) != _ARRAYS:
            raise ValueError("feature cache requires the exact numeric and Unicode array contract")
        features, ids, coordinates, raw = (
            archive[key] for key in ("features", "region_ids", "coordinates", "metadata_json")
        )
        if raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("feature cache metadata must be a Unicode JSON scalar")
        try:
            metadata = json.loads(raw.item())
            manifest_text = metadata["encoder_manifest_json"]
            encoder_document = json.loads(manifest_text)
            if not isinstance(encoder_document, dict):
                raise ValueError("encoder manifest must be a JSON object")
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("feature cache metadata or encoder identity is invalid") from exc
        if (
            metadata.get("schema") != _SCHEMA
            or metadata.get("image_sha256") != expected_image_sha256
            or metadata.get("grid_sha256") != identity
            or metadata.get("encoder_manifest_sha256") != expected_encoder_manifest_sha256
            or hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
            != expected_encoder_manifest_sha256
            or metadata.get("encoder") != encoder_document
            or metadata.get("frozen") is not True
            or encoder_document.get("schema") != "heir.encoder_manifest.v1"
            or encoder_document.get("availability") != "available"
            or metadata.get("prediction_unit") != "tissue_region"
            or metadata.get("coordinate_units") != "registered_pixel"
            or metadata.get("physical_scale_verified") is not False
            or metadata.get("contains_spatial_rna_derived_features") is not False
            or metadata.get("contains_protein_derived_features") is not False
        ):
            raise ValueError("feature cache source, grid, or encoder identity mismatch")
        width = encoder_document.get("feature_width")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width < 1
            or features.dtype != np.dtype("float32")
            or not np.isfinite(features).all()
            or features.shape != (len(grid.region_ids), width)
            or metadata.get("feature_dim") != width
            or metadata.get("feature_shape") != list(features.shape)
            or metadata.get("feature_dtype") != "float32"
            or metadata.get("feature_hash_encoding") != "float32_C_order_raw_bytes"
            or metadata.get("feature_sha256") != _feature_digest(features)
            or ids.dtype.kind != "U"
            or ids.shape != (len(grid.region_ids),)
            or tuple(ids.tolist()) != tuple(grid.region_ids)
            or coordinates.dtype != np.dtype("float32")
            or coordinates.shape != (len(grid.region_ids), 2)
            or not np.array_equal(coordinates, grid.coordinates)
            or metadata.get("input_pixels") != encoder_document.get("input_pixels")
            or metadata.get("interpolation") != encoder_document.get("interpolation")
            or metadata.get("preferred_model_mpp") != encoder_document.get("model_mpp")
            or metadata.get("crop_definition") != _CROP
            or metadata.get("resize") != _RESIZE
            or metadata.get("array_row_is_y") is not True
        ):
            raise ValueError(
                "feature cache arrays or preprocessing differ from the declared contract"
            )
        _json(metadata)
        return features.copy(), metadata
