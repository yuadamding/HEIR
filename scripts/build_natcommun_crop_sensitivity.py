#!/usr/bin/env python3
"""Build the registered 55-um H-optimus-1 crop-sensitivity supplement.

The frozen NatCommun source stores H-optimus-1 features from a registered 112-um
canvas.  This supplement keeps that same canvas and qualified resize path, but
whitens every pixel outside the registered, centred 55-um Visium footprint.  It
therefore tests field of view without changing model magnification.

The script intentionally imports the frozen v1 source builder after verifying
its implementation hash.  Registration, TIFF reading, resampling, manifest and
parity checks consequently use the already-qualified implementation instead of
an independent copy.
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

DEFAULT_SOURCE = Path("/mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz")
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/natcommun_regional_source/crop_sensitivity_55um.npz")
DEFAULT_MODEL_DIR = Path("/mnt/seagate/HnE/pretrained/H-optimus-1")
DEFAULT_ENCODER_MANIFEST = REPO_ROOT / "manifests/encoders/hoptimus1.json"
DEFAULT_ENCODER_PARITY_RECEIPT = DEFAULT_MODEL_DIR / "official_local_parity.json"

SUPPLEMENT_SCHEMA = "heir.natcommun_crop_sensitivity.v1"
CACHE_SCHEMA = "heir.natcommun_crop_sensitivity_section_cache.v1"
RECEIPT_SCHEMA = "heir.natcommun_crop_sensitivity_receipt.v1"
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
    actual = _sha256_path(V1_BUILDER_PATH)
    if actual != FROZEN_V1_BUILDER_SHA256:
        raise RuntimeError(
            "frozen NatCommun v1 builder hash differs; refusing unqualified helper import"
        )
    name = "_heir_frozen_natcommun_regional_source_v1"
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
    native_feature_stats: Mapping[str, object]


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
    if values.ndim != 2 or not len(values):
        raise ValueError(f"{name} must be a non-empty row-by-feature matrix")
    if expected_rows is not None and values.shape[0] != expected_rows:
        raise ValueError(f"{name} row count differs from its bound identities")
    if expected_width is not None and values.shape[1] != expected_width:
        raise ValueError(f"{name} width differs from the H-optimus-1 manifest")
    if values.shape[0] < 2:
        raise ValueError(f"{name} needs at least two rows for a nondegeneracy check")

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

    centered_sum_squares = np.maximum(0.0, sum_squares - np.square(totals) / float(len(values)))
    variable_dimensions = int(np.count_nonzero(maximum > minimum))
    total_centered_energy = float(centered_sum_squares.sum())
    if minimum_row_norm <= 0 or variable_dimensions == 0 or total_centered_energy <= 0:
        raise ValueError(f"{name} is degenerate")
    return {
        "rows": int(values.shape[0]),
        "width": int(width),
        "dtype": str(values.dtype),
        "finite": True,
        "minimum_squared_row_norm": float(minimum_row_norm),
        "maximum_squared_row_norm": float(maximum_row_norm),
        "variable_feature_dimensions": variable_dimensions,
        "total_centered_feature_energy": total_centered_energy,
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
            native_features = np.asarray(archive["image_features"])
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
        or native_features.dtype != np.float16
    ):
        raise ValueError("source spot identities or registered coordinates are malformed")
    expected_spot_ids = np.char.add(np.char.add(sections, ":"), barcodes)
    if not np.array_equal(spot_ids, expected_spot_ids):
        raise ValueError("source spot_ids are not exactly section:barcode row aligned")
    global_native_stats = _feature_stats(
        native_features,
        "source 112-um image_features",
        expected_rows=rows,
        expected_width=V1.IMAGE_FEATURE_WIDTH,
    )
    section_native_stats = {}
    for section in dict.fromkeys(sections.tolist()):
        section_rows = np.flatnonzero(sections == section)
        section_native_stats[section] = _feature_stats(
            native_features[section_rows],
            f"source {section} 112-um image_features",
            expected_rows=len(section_rows),
            expected_width=V1.IMAGE_FEATURE_WIDTH,
        )
    native_stats = {
        "global": global_native_stats,
        "per_section": section_native_stats,
    }
    return SourceContract(
        path=path,
        sha256=actual,
        schema=schema,
        spot_ids=spot_ids,
        barcode_ids=barcodes,
        section_ids=sections,
        pixel_xy=pixel_xy,
        receipt=receipt,
        native_feature_stats=native_stats,
    )


def _validate_encoder_identity(
    source: SourceContract,
    manifest: EncoderManifest,
    parity: Mapping[str, object],
) -> Mapping[str, object]:
    source_encoder = source.receipt.get("encoder")
    roles = source.receipt.get("encoder_roles")
    source_primary = roles.get("primary") if isinstance(roles, Mapping) else None
    source_parity = (
        source_encoder.get("official_local_parity") if isinstance(source_encoder, Mapping) else None
    )
    parity_fields = (
        "status",
        "schema",
        "receipt_sha256",
        "encoder_manifest_sha256",
        "implementation_sha256",
        "runtime_sha256",
    )
    if (
        not isinstance(source_encoder, Mapping)
        or not isinstance(source_primary, Mapping)
        or not isinstance(source_parity, Mapping)
        or source_encoder.get("repository") != "bioptimus/H-optimus-1"
        or source_encoder.get("revision") != manifest.revision
        or source_encoder.get("manifest_sha256") != manifest.sha256
        or source_encoder.get("device") != "cuda"
        or source_primary.get("repository") != source_encoder.get("repository")
        or source_primary.get("revision") != source_encoder.get("revision")
        or source_primary.get("manifest_sha256") != source_encoder.get("manifest_sha256")
        or any(source_parity.get(field) != parity.get(field) for field in parity_fields)
    ):
        raise ValueError("source H-optimus-1 manifest/parity identity differs from this run")
    return {
        "repository": manifest.repository,
        "revision": manifest.revision,
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "architecture": manifest.architecture,
        "checkpoint_filename": manifest.checkpoint_filename,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "config_filename": manifest.config_filename,
        "config_sha256": manifest.config_sha256,
        "feature_width": manifest.feature_width,
        "input_pixels": manifest.input_pixels,
        "model_mpp": manifest.model_mpp,
        "device": "cuda",
        "fine_tuning": "none_frozen_eval_inference",
        "official_local_parity": dict(parity),
    }


def _section_contracts(source: SourceContract) -> tuple[SectionContract, ...]:
    section_receipts = source.receipt.get("sections")
    if not isinstance(section_receipts, list) or not section_receipts:
        raise ValueError("source receipt has no section registration identities")
    receipt_by_section: dict[str, Mapping[str, object]] = {}
    for item in section_receipts:
        if not isinstance(item, Mapping):
            raise ValueError("source section receipt is malformed")
        section = str(item.get("section", ""))
        if not section or section in receipt_by_section:
            raise ValueError("source section receipt identities are empty or duplicated")
        receipt_by_section[section] = item

    ordered_sections = list(dict.fromkeys(source.section_ids.tolist()))
    if ordered_sections != [str(item.get("section")) for item in section_receipts]:
        raise ValueError("source row section order differs from its embedded receipt")
    if set(ordered_sections) != set(receipt_by_section):
        raise ValueError("source row sections differ from its embedded receipt")

    contracts = []
    previous_stop = 0
    for section in ordered_sections:
        rows = np.flatnonzero(source.section_ids == section).astype(np.int64)
        if not np.array_equal(rows, np.arange(previous_stop, previous_stop + len(rows))):
            raise ValueError("source section rows are not in frozen contiguous order")
        previous_stop += len(rows)
        receipt = receipt_by_section[section]
        if int(receipt.get("spot_count", -1)) != len(rows):
            raise ValueError(f"source section spot count differs for {section}")
        embedding = receipt.get("embedding")
        source_encoder = source.receipt.get("encoder")
        source_parity = (
            source_encoder.get("official_local_parity")
            if isinstance(source_encoder, Mapping)
            else None
        )
        if (
            not isinstance(embedding, Mapping)
            or not isinstance(source_encoder, Mapping)
            or not isinstance(source_parity, Mapping)
            or embedding.get("encoder_manifest_sha256") != source_encoder.get("manifest_sha256")
            or embedding.get("official_local_parity") != source_parity
            or embedding.get("device") != "cuda"
            or int(embedding.get("feature_width", -1)) != V1.IMAGE_FEATURE_WIDTH
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
    patch: object,
    center_xy: Sequence[float],
    inner_width: int,
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


def _section_file_contract(
    section: SectionContract,
) -> tuple[Path, Path, int, int, Mapping[str, object], Mapping[str, object]]:
    embedding = section.receipt.get("embedding")
    processing = section.receipt.get("spaceranger_provenance")
    positions = section.receipt.get("tissue_positions")
    if (
        not isinstance(embedding, Mapping)
        or not isinstance(processing, Mapping)
        or not isinstance(positions, Mapping)
    ):
        raise ValueError(f"source registration provenance is absent for {section.section}")
    image_identity = embedding.get("image")
    crop = embedding.get("crop")
    registration = embedding.get("registration_qc")
    if (
        not isinstance(image_identity, Mapping)
        or not isinstance(crop, Mapping)
        or not isinstance(registration, Mapping)
    ):
        raise ValueError(f"source embedding identity is malformed for {section.section}")
    image_path = Path(str(image_identity.get("path", ""))).expanduser().resolve()
    scalefactors_path = Path(str(embedding.get("scalefactors_path", ""))).expanduser().resolve()
    positions_path = Path(str(positions.get("path", ""))).expanduser().resolve()
    alignment_path = Path(str(processing.get("final_alignment_path", ""))).expanduser().resolve()
    alignment_qc_path = (
        Path(str(processing.get("alignment_qc_image_path", ""))).expanduser().resolve()
    )
    registration_paths = (
        image_path,
        scalefactors_path,
        positions_path,
        alignment_path,
        alignment_qc_path,
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in registration_paths):
        raise FileNotFoundError(f"registered source files are missing for {section.section}")
    image_sha256 = _sha256_path(image_path)
    scalefactors_sha256 = _sha256_path(scalefactors_path)
    positions_sha256 = _sha256_path(positions_path)
    alignment_sha256 = _sha256_path(alignment_path)
    alignment_qc_sha256 = _sha256_path(alignment_qc_path)
    if (
        processing.get("schema") != "heir.natcommun_spaceranger_section_provenance.v1"
        or processing.get("exact_invocation_fields_verified") is not True
        or image_sha256 != image_identity.get("sha256")
        or image_sha256 != processing.get("h_and_e_sha256")
        or image_path != Path(str(processing.get("h_and_e_path", ""))).expanduser().resolve()
        or scalefactors_sha256 != embedding.get("scalefactors_sha256")
        or positions_sha256 != positions.get("sha256")
        or alignment_sha256 != processing.get("final_alignment_sha256")
        or alignment_qc_sha256 != processing.get("alignment_qc_image_sha256")
    ):
        raise ValueError(f"registered H&E/scalefactor identity changed for {section.section}")
    outer_width, computed_crop = V1._spot_crop_pixels(scalefactors_path)
    if crop != computed_crop or crop.get("physical_width_um") != SOURCE_FIELD_UM:
        raise ValueError(f"source 112-um crop identity changed for {section.section}")
    spot_diameter = float(computed_crop["spot_diameter_fullres_pixels"])
    inner_width = int(round(spot_diameter))
    if inner_width <= 0 or inner_width > outer_width:
        raise ValueError(f"55-um pixel width is invalid for {section.section}")
    target_crop = {
        "construction": "white_outside_registered_center_square_on_native_112um_canvas",
        "source_canvas_physical_width_um": SOURCE_FIELD_UM,
        "retained_center_physical_width_um": TARGET_FIELD_UM,
        "source_canvas_fullres_pixels": outer_width,
        "retained_center_fullres_pixels": inner_width,
        "centering_rule": ("independent_floor(center_minus_width_over_two)_registered_bounds"),
        "outside_value": "white_RGB_uint8_255",
        "separate_55um_resize": False,
    }
    return (
        image_path,
        scalefactors_path,
        outer_width,
        inner_width,
        registration,
        {
            "image_path": str(image_path),
            "image_sha256": image_sha256,
            "scalefactors_path": str(scalefactors_path),
            "scalefactors_sha256": scalefactors_sha256,
            "tissue_positions_path": str(positions_path),
            "tissue_positions_sha256": positions_sha256,
            "spaceranger_final_alignment_path": str(alignment_path),
            "spaceranger_final_alignment_sha256": alignment_sha256,
            "spaceranger_alignment_qc_image_path": str(alignment_qc_path),
            "spaceranger_alignment_qc_image_sha256": alignment_qc_sha256,
            "spaceranger_exact_invocation_fields_verified": True,
            "alignment_visual_review_required_before_exact_image_claims": processing.get(
                "alignment_visual_review_required_before_exact_image_claims"
            ),
            "source_crop": computed_crop,
            "target_crop": target_crop,
        },
    )


def _section_embeddings(
    *,
    source: SourceContract,
    section: SectionContract,
    encoder: FrozenPatchEncoder,
    manifest: EncoderManifest,
    encoder_identity: Mapping[str, object],
    cache_dir: Path,
    batch_size: int,
) -> tuple[np.ndarray, Mapping[str, object]]:
    (
        image_path,
        _scalefactors_path,
        outer_width,
        inner_width,
        expected_registration,
        file_identity,
    ) = _section_file_contract(section)
    with V1._TiffRegionReader(image_path) as reader:
        registration = V1._registration_qc(reader, section.pixel_xy, outer_width)
        if not _registration_equal(expected_registration, registration):
            raise ValueError(f"registered H&E geometry changed for {section.section}")
        identity = {
            "schema": CACHE_SCHEMA,
            "builder_implementation_sha256": _sha256_path(Path(__file__).resolve()),
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
            "resampling": {
                "source_canvas_pixels": [outer_width, outer_width],
                "target_canvas_pixels": [manifest.input_pixels, manifest.input_pixels],
                "implementation": "frozen_v1_Pillow.Image.Resampling.BICUBIC",
                "resampling_count": int(outer_width != manifest.input_pixels),
                "qualified_against_official_loader": True,
            },
            "batch_size_bound": {"requested": batch_size, "maximum": MAX_BATCH_SIZE},
            "stored_feature_dtype": "float16",
        }
        identity_sha256 = V1._canonical_sha256(identity)
        if re.fullmatch(r"[A-Za-z0-9_.-]+", section.section) is None:
            raise ValueError("section identity is unsafe for a cache filename")
        cache_path = cache_dir / f"{section.section}.55um_masked_context.npz"
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as archive:
                    cached_identity = _scalar_text(
                        archive["cache_identity_sha256"], "cache identity"
                    )
                    cached_spot_ids = np.asarray(archive["spot_ids"]).astype(str)
                    cached_features = np.asarray(archive["image_features_55um"])
                    cached_feature_sha256 = _scalar_text(
                        archive["image_features_sha256"],
                        "cached feature SHA-256",
                    )
                if cached_features.dtype != np.float16:
                    raise ValueError("cached crop-sensitivity features are not float16")
                stats = _feature_stats(
                    cached_features,
                    f"{section.section} cached 55-um features",
                    expected_rows=len(section.spot_ids),
                    expected_width=manifest.feature_width,
                )
                if (
                    cached_identity == identity_sha256
                    and np.array_equal(cached_spot_ids, section.spot_ids)
                    and cached_feature_sha256 == stats["array_sha256"]
                ):
                    return cached_features, {
                        **identity,
                        "cache_status": "reused",
                        "cache_path": str(cache_path),
                        "cache_sha256": _sha256_path(cache_path),
                        "feature_stats": stats,
                    }
            except (KeyError, OSError, ValueError):
                pass

        features = np.empty((len(section.spot_ids), manifest.feature_width), dtype=np.float16)
        for start in range(0, len(section.spot_ids), batch_size):
            stop = min(len(section.spot_ids), start + batch_size)
            patches = []
            for index in range(start, stop):
                patch = reader.crop(section.pixel_xy[index], outer_width)
                patches.append(
                    _whiten_outside_registered_square(patch, section.pixel_xy[index], inner_width)
                )
            resized = V1._resize_hoptimus_batch(np.stack(patches), manifest.input_pixels)
            encoded = np.asarray(encoder.encode(resized), dtype=np.float32)
            if encoded.shape != (stop - start, manifest.feature_width):
                raise ValueError(f"H-optimus-1 returned the wrong shape for {section.section}")
            if not np.isfinite(encoded).all():
                raise ValueError(f"H-optimus-1 returned non-finite features for {section.section}")
            features[start:stop] = encoded.astype(np.float16)

    stats = _feature_stats(
        features,
        f"{section.section} 55-um features",
        expected_rows=len(section.spot_ids),
        expected_width=manifest.feature_width,
    )
    V1._atomic_npz(
        cache_path,
        {
            "cache_identity_sha256": np.asarray(identity_sha256),
            "spot_ids": section.spot_ids,
            "image_features_55um": features,
            "image_features_sha256": np.asarray(stats["array_sha256"]),
            "cache_receipt_json": np.asarray(json.dumps(identity, sort_keys=True, allow_nan=False)),
        },
    )
    return features, {
        **identity,
        "cache_status": "created",
        "cache_path": str(cache_path),
        "cache_sha256": _sha256_path(cache_path),
        "feature_stats": stats,
    }


def run(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("NatCommun H-optimus-1 crop sensitivity requires CUDA")
    if args.batch_size <= 0 or args.batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
    if _sha256_path(V1_BUILDER_PATH) != FROZEN_V1_BUILDER_SHA256:
        raise RuntimeError("frozen NatCommun v1 builder changed after helper import")

    source_path = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == source_path:
        raise ValueError("crop supplement output must not overwrite its frozen source")
    cache_dir = args.cache_dir or output.parent / "crop_sensitivity_section_cache"
    cache_dir = cache_dir.expanduser().resolve()
    source = _load_source(source_path, args.source_sha256)

    manifest = V1.load_encoder_manifest(args.encoder_manifest.expanduser().resolve())
    if (
        manifest.repository != "bioptimus/H-optimus-1"
        or manifest.feature_width != V1.IMAGE_FEATURE_WIDTH
        or manifest.input_pixels != 224
        or manifest.fine_tuning != "prohibited"
        or not np.isclose(
            manifest.input_pixels * manifest.model_mpp,
            SOURCE_FIELD_UM,
            rtol=0.0,
            atol=1.0e-6,
        )
    ):
        raise ValueError("crop sensitivity requires the frozen 112-um H-optimus-1 manifest")
    parity = V1._load_encoder_parity_receipt(
        args.encoder_parity_receipt.expanduser().resolve(), manifest
    )
    encoder_identity = _validate_encoder_identity(source, manifest, parity)
    sections = _section_contracts(source)
    encoder = V1.create_frozen_encoder(
        args.model_dir.expanduser().resolve(), manifest, device="cuda"
    )

    features = np.empty((len(source.spot_ids), manifest.feature_width), dtype=np.float16)
    written = np.zeros(len(source.spot_ids), dtype=np.bool_)
    section_receipts = []
    for section in sections:
        section_features, receipt = _section_embeddings(
            source=source,
            section=section,
            encoder=encoder,
            manifest=manifest,
            encoder_identity=encoder_identity,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
        )
        if written[section.row_indices].any():
            raise ValueError("section row assignments overlap")
        features[section.row_indices] = section_features
        written[section.row_indices] = True
        section_receipts.append(receipt)
    if not written.all():
        raise ValueError("section caches do not cover every source spot exactly once")
    feature_stats = _feature_stats(
        features,
        "assembled 55-um image_features",
        expected_rows=len(source.spot_ids),
        expected_width=manifest.feature_width,
    )
    if _sha256_path(source.path) != source.sha256:
        raise ValueError("frozen NatCommun source changed during crop-sensitivity inference")
    if _sha256_path(V1_BUILDER_PATH) != FROZEN_V1_BUILDER_SHA256:
        raise RuntimeError("frozen NatCommun v1 helper changed during inference")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "builder_implementation_sha256": _sha256_path(Path(__file__).resolve()),
        "frozen_v1_builder": {
            "path": str(V1_BUILDER_PATH),
            "sha256": FROZEN_V1_BUILDER_SHA256,
            "imported_helpers": [
                "_TiffRegionReader",
                "_spot_crop_pixels",
                "_registration_qc",
                "_resize_hoptimus_batch",
                "_load_encoder_parity_receipt",
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
            "native_112um_feature_stats": source.native_feature_stats,
        },
        "crop_construction": {
            "source_canvas_physical_width_um": SOURCE_FIELD_UM,
            "retained_center_physical_width_um": TARGET_FIELD_UM,
            "operation": (
                "extract_the_registered_112um_canvas_then_whiten_everything_outside_the_"
                "independently_registered_centered_55um_square"
            ),
            "white_value": "RGB_uint8_255",
            "resize_after_masking": (
                "same_single_frozen_v1_Pillow_bicubic_112um_canvas_to_224_pixels"
            ),
            "separate_55um_crop_resize_prohibited": True,
            "model_magnification_unchanged": True,
        },
        "encoder": encoder_identity,
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
        "scientific_scope": ("regional_Visium_crop_sensitivity_only_not_cell_level_authorization"),
    }
    payload = {
        "schema_version": np.asarray(SUPPLEMENT_SCHEMA),
        "spot_ids": source.spot_ids,
        "image_features_55um": features,
        "source_sha256": np.asarray(source.sha256),
        "receipt_json": np.asarray(json.dumps(receipt, sort_keys=True, allow_nan=False)),
    }
    V1._atomic_npz(output, payload)
    output_sha256 = _sha256_path(output)
    V1._atomic_json(
        output.with_suffix(".receipt.json"),
        {
            **receipt,
            "output": str(output),
            "output_sha256": output_sha256,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": output_sha256,
                "schema": SUPPLEMENT_SCHEMA,
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
    parser.add_argument(
        "--encoder-parity-receipt", type=Path, default=DEFAULT_ENCODER_PARITY_RECEIPT
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(parse_args()))
