#!/usr/bin/env python3
"""Build the separately scored UNI2-h NatCommun encoder sensitivity.

The exact frozen NatCommun ``source.npz`` supplies spot rows and registered H&E
geometry.  For every source spot this script encodes (1) its natural registered
112-um canvas and (2) the same canvas with pixels outside the centred 55-um
Visium footprint whitened.  Both native canvases are passed directly to
``UNI2HEncoder.encode``; its manifest-bound bilinear interpolation is therefore
the only resize.

This is a secondary encoder sensitivity, not an official/local parity
qualification and not a cell-level experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.features import EncoderManifest, FrozenPatchEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
V1_BUILDER_PATH = REPO_ROOT / "scripts/build_natcommun_regional_source.py"
FROZEN_V1_BUILDER_SHA256 = "3b6006f61c72cb46029366e30f1510195109bd17d6471d6b2b4c4d0f55c5fdbb"
UNI2_ADAPTER_PATH = REPO_ROOT / "src/heir/features/uni2h.py"
ENCODER_BASE_PATH = REPO_ROOT / "src/heir/features/base.py"
ENCODER_FACTORY_PATH = REPO_ROOT / "src/heir/features/__init__.py"

DEFAULT_SOURCE = Path("/mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz")
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/natcommun_regional_source/uni2h_sensitivity.npz")
DEFAULT_MODEL_DIR = Path("/mnt/seagate/HnE/pretrained/UNI2-h")
DEFAULT_ENCODER_MANIFEST = REPO_ROOT / "manifests/encoders/uni2h.json"

OUTPUT_SCHEMA = "heir.natcommun_uni2h_sensitivity.v1"
CACHE_SCHEMA = "heir.natcommun_uni2h_section_cache.v1"
RECEIPT_SCHEMA = "heir.natcommun_uni2h_sensitivity_receipt.v1"
SOURCE_FIELD_UM = 112.0
TARGET_FIELD_UM = 55.0
MAX_BATCH_SIZE = 8
FEATURE_STAT_CHUNK_ROWS = 1024


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_v1_builder() -> ModuleType:
    if _sha256_path(V1_BUILDER_PATH) != FROZEN_V1_BUILDER_SHA256:
        raise RuntimeError(
            "frozen NatCommun v1 builder hash differs; refusing unqualified helper import"
        )
    name = "_heir_frozen_natcommun_regional_source_for_uni2h"
    spec = importlib.util.spec_from_file_location(name, V1_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen NatCommun v1 builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V1 = _load_frozen_v1_builder()


@dataclass(frozen=True)
class SourceContract:
    path: Path
    sha256: str
    schema: str
    spot_ids: np.ndarray
    barcode_ids: np.ndarray
    section_ids: np.ndarray
    pixel_xy: np.ndarray
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class SectionContract:
    section: str
    row_indices: np.ndarray
    spot_ids: np.ndarray
    barcodes: np.ndarray
    pixel_xy: np.ndarray
    receipt: Mapping[str, object]


def _scalar_text(value: object, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar string")
    return str(array.item())


def _require_sha256(value: str, name: str) -> str:
    normalized = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return normalized


def _feature_stats(
    features: object,
    name: str,
    *,
    expected_rows: Optional[int] = None,
    expected_width: Optional[int] = None,
) -> Mapping[str, object]:
    values = np.asarray(features)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError(f"{name} needs at least two feature rows")
    if expected_rows is not None and values.shape[0] != expected_rows:
        raise ValueError(f"{name} row count differs from its bound identities")
    if expected_width is not None and values.shape[1] != expected_width:
        raise ValueError(f"{name} width differs from the UNI2-h manifest")

    width = values.shape[1]
    totals = np.zeros(width, dtype=np.float64)
    sum_squares = np.zeros(width, dtype=np.float64)
    minimum = np.full(width, np.inf, dtype=np.float32)
    maximum = np.full(width, -np.inf, dtype=np.float32)
    minimum_row_norm = np.inf
    maximum_row_norm = -np.inf
    for start in range(0, len(values), FEATURE_STAT_CHUNK_ROWS):
        block = np.asarray(values[start : start + FEATURE_STAT_CHUNK_ROWS], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"{name} contains non-finite values")
        totals += block.sum(axis=0, dtype=np.float64)
        sum_squares += np.square(block, dtype=np.float64).sum(axis=0, dtype=np.float64)
        minimum = np.minimum(minimum, block.min(axis=0))
        maximum = np.maximum(maximum, block.max(axis=0))
        row_norms = np.square(block, dtype=np.float64).sum(axis=1)
        minimum_row_norm = min(minimum_row_norm, float(row_norms.min()))
        maximum_row_norm = max(maximum_row_norm, float(row_norms.max()))

    centered_energy = float(
        np.maximum(0.0, sum_squares - np.square(totals) / float(len(values))).sum()
    )
    variable_dimensions = int(np.count_nonzero(maximum > minimum))
    if minimum_row_norm <= 0 or variable_dimensions == 0 or centered_energy <= 0:
        raise ValueError(f"{name} is degenerate")
    return {
        "rows": int(values.shape[0]),
        "width": int(width),
        "dtype": str(values.dtype),
        "finite": True,
        "minimum_squared_row_norm": float(minimum_row_norm),
        "maximum_squared_row_norm": float(maximum_row_norm),
        "variable_feature_dimensions": variable_dimensions,
        "total_centered_feature_energy": centered_energy,
        "array_sha256": V1._array_sha256(values),
    }


def _load_source(path: Path, expected_sha256: str) -> SourceContract:
    expected = _require_sha256(expected_sha256, "source SHA-256")
    actual = _sha256_path(path)
    if actual != expected:
        raise ValueError("NatCommun source SHA-256 differs from --source-sha256")
    try:
        with np.load(path, allow_pickle=False) as archive:
            schema = _scalar_text(archive["schema_version"], "source schema_version")
            spot_ids = np.asarray(archive["spot_ids"]).astype(str)
            barcodes = np.asarray(archive["barcode_ids"]).astype(str)
            sections = np.asarray(archive["section_ids"]).astype(str)
            pixel_xy = np.asarray(archive["pixel_xy"], dtype=np.float64)
            receipt_text = _scalar_text(
                archive["source_receipt_json"], "source source_receipt_json"
            )
    except (KeyError, OSError, ValueError) as error:
        raise ValueError("NatCommun source lacks the frozen row/receipt contract") from error
    try:
        receipt = json.loads(receipt_text)
    except json.JSONDecodeError as error:
        raise ValueError("NatCommun embedded source receipt is invalid JSON") from error

    rows = len(spot_ids)
    if (
        schema != V1.SOURCE_SCHEMA
        or not isinstance(receipt, Mapping)
        or receipt.get("schema") != V1.RECEIPT_SCHEMA
        or receipt.get("builder_implementation_sha256") != FROZEN_V1_BUILDER_SHA256
        or receipt.get("observation_level") != "Visium_v2_spot_regional_not_cellular"
    ):
        raise ValueError("source was not produced by the hash-frozen NatCommun v1 builder")
    if (
        spot_ids.ndim != 1
        or not rows
        or len(set(spot_ids.tolist())) != rows
        or barcodes.shape != (rows,)
        or sections.shape != (rows,)
        or pixel_xy.shape != (rows, 2)
        or not np.isfinite(pixel_xy).all()
    ):
        raise ValueError("source spot identities or registered coordinates are malformed")
    expected_spot_ids = np.char.add(np.char.add(sections, ":"), barcodes)
    if not np.array_equal(spot_ids, expected_spot_ids):
        raise ValueError("source spot_ids are not exactly section:barcode row aligned")
    return SourceContract(
        path=path,
        sha256=actual,
        schema=schema,
        spot_ids=spot_ids,
        barcode_ids=barcodes,
        section_ids=sections,
        pixel_xy=pixel_xy,
        receipt=receipt,
    )


def _section_contracts(source: SourceContract) -> tuple[SectionContract, ...]:
    receipts = source.receipt.get("sections")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("source receipt has no section registration identities")
    by_section: dict[str, Mapping[str, object]] = {}
    for item in receipts:
        if not isinstance(item, Mapping):
            raise ValueError("source section receipt is malformed")
        section = str(item.get("section", ""))
        if not section or section in by_section:
            raise ValueError("source section receipt identities are empty or duplicated")
        by_section[section] = item

    ordered = list(dict.fromkeys(source.section_ids.tolist()))
    if ordered != [str(item.get("section")) for item in receipts]:
        raise ValueError("source row section order differs from its embedded receipt")
    contracts = []
    previous_stop = 0
    for section in ordered:
        rows = np.flatnonzero(source.section_ids == section).astype(np.int64)
        if not np.array_equal(rows, np.arange(previous_stop, previous_stop + len(rows))):
            raise ValueError("source section rows are not in frozen contiguous order")
        previous_stop += len(rows)
        receipt = by_section[section]
        embedding = receipt.get("embedding")
        if (
            int(receipt.get("spot_count", -1)) != len(rows)
            or not isinstance(embedding, Mapping)
            or embedding.get("barcodes_sha256")
            != V1._array_sha256(np.asarray(source.barcode_ids[rows], dtype="S"))
            or embedding.get("pixel_xy_sha256")
            != V1._array_sha256(np.asarray(source.pixel_xy[rows], dtype=np.float64))
        ):
            raise ValueError(f"source section row hashes differ for {section}")
        contracts.append(
            SectionContract(
                section=section,
                row_indices=rows,
                spot_ids=source.spot_ids[rows],
                barcodes=source.barcode_ids[rows],
                pixel_xy=source.pixel_xy[rows],
                receipt=receipt,
            )
        )
    return tuple(contracts)


def _registered_inner_bounds(
    center_xy: Sequence[float], outer_width: int, inner_width: int
) -> tuple[int, int, int, int]:
    """Map independently floored registered crops into the outer canvas."""

    if outer_width <= 0 or inner_width <= 0 or inner_width > outer_width:
        raise ValueError("registered inner/outer crop widths are invalid")
    x, y = map(float, center_xy)
    if not np.isfinite([x, y]).all():
        raise ValueError("registered crop center is non-finite")
    outer_left = int(np.floor(x - outer_width / 2.0))
    outer_top = int(np.floor(y - outer_width / 2.0))
    inner_left = int(np.floor(x - inner_width / 2.0))
    inner_top = int(np.floor(y - inner_width / 2.0))
    left, top = inner_left - outer_left, inner_top - outer_top
    right, bottom = left + inner_width, top + inner_width
    if left < 0 or top < 0 or right > outer_width or bottom > outer_width:
        raise ValueError("registered 55-um square falls outside the 112-um canvas")
    return left, top, right, bottom


def _whiten_outside_registered_square(
    patch: object, center_xy: Sequence[float], inner_width: int
) -> np.ndarray:
    source = np.asarray(patch)
    if (
        source.ndim != 3
        or source.shape[0] != source.shape[1]
        or source.shape[2] != 3
        or source.dtype != np.uint8
    ):
        raise ValueError("registered source patch must be square uint8 RGB")
    left, top, right, bottom = _registered_inner_bounds(center_xy, source.shape[0], inner_width)
    output = np.full_like(source, 255)
    output[top:bottom, left:right] = source[top:bottom, left:right]
    return output


def _registration_equal(expected: Mapping[str, object], actual: Mapping[str, object]) -> bool:
    if set(expected) != set(actual):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            if not np.isclose(expected_value, actual_value, rtol=0.0, atol=1.0e-12):
                return False
        elif expected_value != actual_value:
            return False
    return True


def _section_files(
    section: SectionContract,
) -> tuple[Path, int, int, Mapping[str, object], Mapping[str, object]]:
    embedding = section.receipt.get("embedding")
    processing = section.receipt.get("spaceranger_provenance")
    positions = section.receipt.get("tissue_positions")
    if not all(isinstance(item, Mapping) for item in (embedding, processing, positions)):
        raise ValueError(f"source registration provenance is absent for {section.section}")
    assert isinstance(embedding, Mapping)
    assert isinstance(processing, Mapping)
    assert isinstance(positions, Mapping)
    image_identity = embedding.get("image")
    crop = embedding.get("crop")
    registration = embedding.get("registration_qc")
    if not all(isinstance(item, Mapping) for item in (image_identity, crop, registration)):
        raise ValueError(f"source embedding identity is malformed for {section.section}")
    assert isinstance(image_identity, Mapping)
    assert isinstance(crop, Mapping)
    assert isinstance(registration, Mapping)

    paths = {
        "h_and_e": Path(str(image_identity.get("path", ""))).expanduser().resolve(),
        "scalefactors": Path(str(embedding.get("scalefactors_path", ""))).expanduser().resolve(),
        "tissue_positions": Path(str(positions.get("path", ""))).expanduser().resolve(),
        "spaceranger_final_alignment": Path(str(processing.get("final_alignment_path", "")))
        .expanduser()
        .resolve(),
        "spaceranger_alignment_qc_image": Path(str(processing.get("alignment_qc_image_path", "")))
        .expanduser()
        .resolve(),
    }
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()):
        raise FileNotFoundError(f"registered source files are missing for {section.section}")
    hashes = {name: _sha256_path(path) for name, path in paths.items()}
    if (
        processing.get("schema") != "heir.natcommun_spaceranger_section_provenance.v1"
        or processing.get("exact_invocation_fields_verified") is not True
        or hashes["h_and_e"] != image_identity.get("sha256")
        or hashes["h_and_e"] != processing.get("h_and_e_sha256")
        or paths["h_and_e"] != Path(str(processing.get("h_and_e_path", ""))).expanduser().resolve()
        or hashes["scalefactors"] != embedding.get("scalefactors_sha256")
        or hashes["tissue_positions"] != positions.get("sha256")
        or hashes["spaceranger_final_alignment"] != processing.get("final_alignment_sha256")
        or hashes["spaceranger_alignment_qc_image"] != processing.get("alignment_qc_image_sha256")
    ):
        raise ValueError(f"registered H&E/scalefactor identity changed for {section.section}")

    outer_width, computed_crop = V1._spot_crop_pixels(paths["scalefactors"])
    if crop != computed_crop or crop.get("physical_width_um") != SOURCE_FIELD_UM:
        raise ValueError(f"source 112-um crop identity changed for {section.section}")
    inner_width = int(round(float(computed_crop["spot_diameter_fullres_pixels"])))
    if inner_width <= 0 or inner_width > outer_width:
        raise ValueError(f"55-um pixel width is invalid for {section.section}")
    identity = {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
        "spaceranger_exact_invocation_fields_verified": True,
        "source_crop": computed_crop,
        "target_crop": {
            "construction": "white_outside_registered_center_square_on_native_112um_canvas",
            "source_canvas_physical_width_um": SOURCE_FIELD_UM,
            "retained_center_physical_width_um": TARGET_FIELD_UM,
            "source_canvas_fullres_pixels": outer_width,
            "retained_center_fullres_pixels": inner_width,
            "centering_rule": "independent_floor_registered_bounds",
            "outside_value": "white_RGB_uint8_255",
        },
    }
    return paths["h_and_e"], outer_width, inner_width, registration, identity


def _encoder_identity(manifest: EncoderManifest, model_dir: Path) -> Mapping[str, object]:
    checkpoint = (model_dir / manifest.checkpoint_filename).resolve()
    config = (model_dir / manifest.config_filename).resolve()
    if (
        manifest.repository != "MahmoodLab/UNI2-h"
        or manifest.implementation != "uni2h_timm"
        or manifest.architecture != "vit_giant_patch14_224"
        or manifest.feature_width != 1536
        or manifest.input_pixels != 224
        or manifest.interpolation != "bilinear"
        or manifest.pooling_rule != "direct_features"
        or manifest.fine_tuning != "prohibited"
    ):
        raise ValueError("encoder manifest is not the supported frozen UNI2-h adapter")
    if (
        not checkpoint.is_file()
        or _sha256_path(checkpoint) != manifest.checkpoint_sha256
        or not config.is_file()
        or _sha256_path(config) != manifest.config_sha256
    ):
        raise ValueError("UNI2-h checkpoint/config differs from the encoder manifest")
    return {
        "repository": manifest.repository,
        "revision": manifest.revision,
        "architecture": manifest.architecture,
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "config_path": str(config),
        "config_sha256": manifest.config_sha256,
        "feature_width": manifest.feature_width,
        "input_pixels": manifest.input_pixels,
        "model_mpp": manifest.model_mpp,
        "normalization": {"mean": list(manifest.mean), "std": list(manifest.std)},
        "interpolation": manifest.interpolation,
        "pooling_rule": manifest.pooling_rule,
        "license": manifest.license,
        "known_training_datasets": list(manifest.known_training_datasets),
        "evaluation_overlap": manifest.evaluation_overlap,
        "device": "cuda",
        "fine_tuning": "none_frozen_eval_inference",
        "official_local_parity_claim": "none_not_assessed",
        "qualification_role": "manifest_hash_bound_secondary_sensitivity",
    }


def _blank_feature(
    encoder: FrozenPatchEncoder, manifest: EncoderManifest
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Encode the all-white control shared by both crop arms."""

    white = np.full((1, manifest.input_pixels, manifest.input_pixels, 3), 255, dtype=np.uint8)
    encoded = np.asarray(encoder.encode(white), dtype=np.float32)
    if encoded.shape != (1, manifest.feature_width) or not np.isfinite(encoded).all():
        raise ValueError("UNI2-h returned an invalid blank-image feature vector")
    stored = encoded[0].astype(np.float16)
    squared_norm = float(np.square(stored.astype(np.float64)).sum())
    if not np.isfinite(stored).all() or squared_norm <= 0:
        raise ValueError("UNI2-h blank-image feature vector is degenerate")
    return stored, {
        "construction": "all_white_RGB_uint8_255_at_manifest_input_pixels",
        "applies_to": ["natural_112um", "centered_55um_whitened"],
        "semantic_reason": "an_all_white_canvas_is_identical_under_both_crop_constructions",
        "input_shape": list(white.shape),
        "stored_dtype": str(stored.dtype),
        "finite": True,
        "squared_norm": squared_norm,
        "array_sha256": V1._array_sha256(stored),
    }


def _section_features(
    *,
    source: SourceContract,
    section: SectionContract,
    encoder: FrozenPatchEncoder,
    manifest: EncoderManifest,
    encoder_identity: Mapping[str, object],
    implementation: Mapping[str, str],
    cache_dir: Path,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    image_path, outer_width, inner_width, expected_registration, file_identity = _section_files(
        section
    )
    with V1._TiffRegionReader(image_path) as reader:
        registration = V1._registration_qc(reader, section.pixel_xy, outer_width)
        if not _registration_equal(expected_registration, registration):
            raise ValueError(f"registered H&E geometry changed for {section.section}")
        identity = {
            "schema": CACHE_SCHEMA,
            "implementation": dict(implementation),
            "frozen_v1_builder": {
                "path": str(V1_BUILDER_PATH),
                "sha256": FROZEN_V1_BUILDER_SHA256,
            },
            "source_sha256": source.sha256,
            "section": section.section,
            "row_indices_sha256": V1._array_sha256(section.row_indices),
            "spot_ids_sha256": V1._array_sha256(np.asarray(section.spot_ids, dtype="S")),
            "barcodes_sha256": V1._array_sha256(np.asarray(section.barcodes, dtype="S")),
            "pixel_xy_sha256": V1._array_sha256(np.asarray(section.pixel_xy, dtype=np.float64)),
            **file_identity,
            "registration_qc": registration,
            "encoder": dict(encoder_identity),
            "preprocessing": {
                "input_to_encoder": "native_registered_uint8_RGB_canvas",
                "explicit_pre_encoder_resize": False,
                "encoder_internal_resize": "torch_bilinear_align_corners_false_antialias_true",
                "natural_112um": "unmodified_registered_canvas",
                "centered_55um": "same_canvas_white_outside_registered_55um_square",
            },
            "batch_size_bound": {"requested": batch_size, "maximum": MAX_BATCH_SIZE},
            "stored_feature_dtype": "float16",
        }
        identity_sha256 = V1._canonical_sha256(identity)
        if re.fullmatch(r"[A-Za-z0-9_.-]+", section.section) is None:
            raise ValueError("section identity is unsafe for a cache filename")
        cache_path = cache_dir / f"{section.section}.uni2h_112um_55um.npz"
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as archive:
                    cached_identity = _scalar_text(
                        archive["cache_identity_sha256"], "cache identity"
                    )
                    cached_spots = np.asarray(archive["spot_ids"]).astype(str)
                    natural = np.asarray(archive["image_features_112um"])
                    centered = np.asarray(archive["image_features_55um"])
                    natural_sha256 = _scalar_text(
                        archive["image_features_112um_sha256"],
                        "cached UNI2-h 112-um feature SHA-256",
                    )
                    centered_sha256 = _scalar_text(
                        archive["image_features_55um_sha256"],
                        "cached UNI2-h 55-um feature SHA-256",
                    )
                natural_stats = _feature_stats(
                    natural,
                    f"{section.section} cached UNI2-h 112-um features",
                    expected_rows=len(section.spot_ids),
                    expected_width=manifest.feature_width,
                )
                centered_stats = _feature_stats(
                    centered,
                    f"{section.section} cached UNI2-h 55-um features",
                    expected_rows=len(section.spot_ids),
                    expected_width=manifest.feature_width,
                )
                if (
                    natural.dtype == np.float16
                    and centered.dtype == np.float16
                    and cached_identity == identity_sha256
                    and np.array_equal(cached_spots, section.spot_ids)
                    and natural_sha256 == natural_stats["array_sha256"]
                    and centered_sha256 == centered_stats["array_sha256"]
                ):
                    return (
                        natural,
                        centered,
                        {
                            **identity,
                            "cache_status": "reused",
                            "cache_path": str(cache_path),
                            "cache_sha256": _sha256_path(cache_path),
                            "feature_stats": {
                                "natural_112um": natural_stats,
                                "centered_55um": centered_stats,
                            },
                        },
                    )
            except (KeyError, OSError, ValueError):
                pass

        natural = np.empty((len(section.spot_ids), manifest.feature_width), dtype=np.float16)
        centered = np.empty_like(natural)
        for start in range(0, len(section.spot_ids), batch_size):
            stop = min(len(section.spot_ids), start + batch_size)
            natural_patches = []
            centered_patches = []
            for index in range(start, stop):
                patch = reader.crop(section.pixel_xy[index], outer_width)
                natural_patches.append(patch)
                centered_patches.append(
                    _whiten_outside_registered_square(patch, section.pixel_xy[index], inner_width)
                )
            natural_block = np.asarray(encoder.encode(np.stack(natural_patches)))
            centered_block = np.asarray(encoder.encode(np.stack(centered_patches)))
            shape = (stop - start, manifest.feature_width)
            if natural_block.shape != shape or centered_block.shape != shape:
                raise ValueError(f"UNI2-h returned the wrong shape for {section.section}")
            if not np.isfinite(natural_block).all() or not np.isfinite(centered_block).all():
                raise ValueError(f"UNI2-h returned non-finite features for {section.section}")
            natural[start:stop] = natural_block.astype(np.float16)
            centered[start:stop] = centered_block.astype(np.float16)

    natural_stats = _feature_stats(
        natural,
        f"{section.section} UNI2-h 112-um features",
        expected_rows=len(section.spot_ids),
        expected_width=manifest.feature_width,
    )
    centered_stats = _feature_stats(
        centered,
        f"{section.section} UNI2-h 55-um features",
        expected_rows=len(section.spot_ids),
        expected_width=manifest.feature_width,
    )
    V1._atomic_npz(
        cache_path,
        {
            "cache_identity_sha256": np.asarray(identity_sha256),
            "spot_ids": section.spot_ids,
            "image_features_112um": natural,
            "image_features_55um": centered,
            "image_features_112um_sha256": np.asarray(natural_stats["array_sha256"]),
            "image_features_55um_sha256": np.asarray(centered_stats["array_sha256"]),
            "cache_receipt_json": np.asarray(json.dumps(identity, sort_keys=True, allow_nan=False)),
        },
    )
    return (
        natural,
        centered,
        {
            **identity,
            "cache_status": "created",
            "cache_path": str(cache_path),
            "cache_sha256": _sha256_path(cache_path),
            "feature_stats": {
                "natural_112um": natural_stats,
                "centered_55um": centered_stats,
            },
        },
    )


def run(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("NatCommun UNI2-h sensitivity requires CUDA")
    if args.batch_size <= 0 or args.batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
    if _sha256_path(V1_BUILDER_PATH) != FROZEN_V1_BUILDER_SHA256:
        raise RuntimeError("frozen NatCommun v1 builder changed after helper import")

    source_path = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == source_path:
        raise ValueError("UNI2-h supplement output must not overwrite its frozen source")
    cache_dir = (args.cache_dir or output.parent / "uni2h_section_cache").expanduser().resolve()
    source = _load_source(source_path, args.source_sha256)
    sections = _section_contracts(source)

    manifest = V1.load_encoder_manifest(args.encoder_manifest.expanduser().resolve())
    model_dir = args.model_dir.expanduser().resolve()
    encoder_identity = _encoder_identity(manifest, model_dir)
    implementation = {
        "builder_sha256": _sha256_path(Path(__file__).resolve()),
        "uni2h_adapter_sha256": _sha256_path(UNI2_ADAPTER_PATH),
        "encoder_base_sha256": _sha256_path(ENCODER_BASE_PATH),
        "encoder_factory_sha256": _sha256_path(ENCODER_FACTORY_PATH),
    }
    encoder = V1.create_frozen_encoder(model_dir, manifest, device="cuda")
    blank_feature, blank_receipt = _blank_feature(encoder, manifest)

    natural = np.empty((len(source.spot_ids), manifest.feature_width), dtype=np.float16)
    centered = np.empty_like(natural)
    written = np.zeros(len(source.spot_ids), dtype=np.bool_)
    section_receipts = []
    for section in sections:
        section_natural, section_centered, receipt = _section_features(
            source=source,
            section=section,
            encoder=encoder,
            manifest=manifest,
            encoder_identity=encoder_identity,
            implementation=implementation,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
        )
        if written[section.row_indices].any():
            raise ValueError("section row assignments overlap")
        natural[section.row_indices] = section_natural
        centered[section.row_indices] = section_centered
        written[section.row_indices] = True
        section_receipts.append(receipt)
    if not written.all():
        raise ValueError("section caches do not cover every source spot exactly once")

    feature_stats = {
        "natural_112um": _feature_stats(
            natural,
            "assembled UNI2-h 112-um features",
            expected_rows=len(source.spot_ids),
            expected_width=manifest.feature_width,
        ),
        "centered_55um": _feature_stats(
            centered,
            "assembled UNI2-h 55-um features",
            expected_rows=len(source.spot_ids),
            expected_width=manifest.feature_width,
        ),
    }
    if _sha256_path(source.path) != source.sha256:
        raise ValueError("frozen NatCommun source changed during UNI2-h inference")
    if _sha256_path(V1_BUILDER_PATH) != FROZEN_V1_BUILDER_SHA256:
        raise RuntimeError("frozen NatCommun v1 helper changed during inference")
    if implementation != {
        "builder_sha256": _sha256_path(Path(__file__).resolve()),
        "uni2h_adapter_sha256": _sha256_path(UNI2_ADAPTER_PATH),
        "encoder_base_sha256": _sha256_path(ENCODER_BASE_PATH),
        "encoder_factory_sha256": _sha256_path(ENCODER_FACTORY_PATH),
    }:
        raise RuntimeError("UNI2-h implementation changed during inference")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "implementation": implementation,
        "frozen_v1_builder": {
            "path": str(V1_BUILDER_PATH),
            "sha256": FROZEN_V1_BUILDER_SHA256,
            "imported_helpers": [
                "_TiffRegionReader",
                "_spot_crop_pixels",
                "_registration_qc",
                "_array_sha256",
                "_canonical_sha256",
                "_atomic_npz",
            ],
        },
        "source": {
            "path": str(source.path),
            "sha256": source.sha256,
            "schema": source.schema,
            "builder_implementation_sha256": source.receipt["builder_implementation_sha256"],
            "spot_count": len(source.spot_ids),
            "spot_ids_sha256": V1._array_sha256(np.asarray(source.spot_ids, dtype="S")),
        },
        "encoder": encoder_identity,
        "preprocessing": {
            "natural_112um": "registered_native_canvas_passed_directly_to_encoder.encode",
            "centered_55um": (
                "same_native_canvas_white_outside_registered_center_55um_then_encoder.encode"
            ),
            "explicit_pre_encoder_resize": False,
            "only_resize": "UNI2HEncoder_manifest_bound_bilinear_interpolation",
            "official_local_parity_claim": "none_not_assessed",
        },
        "blank_image_control": blank_receipt,
        "execution": {
            "device": "cuda",
            "batch_size": args.batch_size,
            "maximum_allowed_batch_size": MAX_BATCH_SIZE,
            "per_section_resumable_caches": True,
        },
        "sections": section_receipts,
        "feature_stats": feature_stats,
        "row_alignment": {
            "output_spot_ids_exactly_equal_source": True,
            "all_source_rows_written_exactly_once": True,
        },
        "scientific_role": {
            "encoder": "secondary_sensitivity_scored_separately_from_H_optimus_1",
            "observation_level": "regional_Visium_v2_spot_not_cellular",
            "not_authorized": [
                "pooling_UNI2_h_with_H_optimus_1_primary_results",
                "official_local_UNI2_h_parity_claim",
                "cell_level_hypothesis_authorization",
                "model_selection_on_held_out_donors",
            ],
        },
    }
    V1._atomic_npz(
        output,
        {
            "schema_version": np.asarray(OUTPUT_SCHEMA),
            "spot_ids": source.spot_ids,
            "image_features_112um": natural,
            "image_features_55um": centered,
            "blank_image_feature_vector": blank_feature,
            "source_sha256": np.asarray(source.sha256),
            "receipt_json": np.asarray(json.dumps(receipt, sort_keys=True, allow_nan=False)),
        },
    )
    output_sha256 = _sha256_path(output)
    V1._atomic_json(
        output.with_suffix(".receipt.json"),
        {**receipt, "output": str(output), "output_sha256": output_sha256},
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": output_sha256,
                "schema": OUTPUT_SCHEMA,
                "spots": len(source.spot_ids),
                "sections": len(sections),
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--encoder-manifest", type=Path, default=DEFAULT_ENCODER_MANIFEST)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(parse_args()))
