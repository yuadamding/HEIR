from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_builder():
    path = Path(__file__).parents[1] / "scripts" / "build_natcommun_crop_sensitivity.py"
    spec = importlib.util.spec_from_file_location("build_natcommun_crop_sensitivity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fixture(
    path: Path,
    *,
    spot_ids: np.ndarray | None = None,
) -> tuple[Path, str]:
    sections = np.asarray(["S1", "S1", "S2", "S2"])
    barcodes = np.asarray(["A-1", "B-1", "C-1", "D-1"])
    spots = (
        np.asarray(spot_ids)
        if spot_ids is not None
        else np.asarray(["S1:A-1", "S1:B-1", "S2:C-1", "S2:D-1"])
    )
    pixels = np.asarray([[20.0, 20.0], [30.0, 30.0], [25.0, 25.0], [35.0, 35.0]])
    features = np.tile(
        np.linspace(0.25, 1.25, builder.V1.IMAGE_FEATURE_WIDTH, dtype=np.float32),
        (4, 1),
    )
    features += np.arange(4, dtype=np.float32)[:, None] * 0.5
    parity = {
        "status": "passed",
        "schema": "heir.hoptimus1_official_local_parity.v1",
        "receipt_sha256": "a" * 64,
        "encoder_manifest_sha256": "b" * 64,
        "implementation_sha256": "c" * 64,
        "runtime_sha256": {"runtime.py": "d" * 64},
    }
    section_receipts = []
    for section in ("S1", "S2"):
        rows = np.flatnonzero(sections == section)
        section_receipts.append(
            {
                "section": section,
                "spot_count": len(rows),
                "embedding": {
                    "encoder_manifest_sha256": "b" * 64,
                    "official_local_parity": parity,
                    "device": "cuda",
                    "feature_width": builder.V1.IMAGE_FEATURE_WIDTH,
                    "barcodes_sha256": builder.V1._array_sha256(
                        np.asarray(barcodes[rows], dtype="S")
                    ),
                    "pixel_xy_sha256": builder.V1._array_sha256(
                        np.asarray(pixels[rows], dtype=np.float64)
                    ),
                },
            }
        )
    receipt = {
        "schema": builder.V1.RECEIPT_SCHEMA,
        "builder_implementation_sha256": builder.FROZEN_V1_BUILDER_SHA256,
        "observation_level": "Visium_v2_spot_regional_not_cellular",
        "encoder": {
            "manifest_sha256": "b" * 64,
            "official_local_parity": parity,
        },
        "sections": section_receipts,
    }
    np.savez_compressed(
        path,
        schema_version=np.asarray(builder.V1.SOURCE_SCHEMA),
        spot_ids=spots,
        barcode_ids=barcodes,
        section_ids=sections,
        pixel_xy=pixels,
        image_features=features.astype(np.float16),
        source_receipt_json=np.asarray(json.dumps(receipt, sort_keys=True)),
    )
    return path, _sha256(path)


def test_frozen_helper_import_and_public_output_contract_are_explicit() -> None:
    assert _sha256(builder.V1_BUILDER_PATH) == builder.FROZEN_V1_BUILDER_SHA256
    assert builder.SUPPLEMENT_SCHEMA == "heir.natcommun_crop_sensitivity.v1"
    assert builder.MAX_BATCH_SIZE == 8
    assert builder.SOURCE_FIELD_UM == 112.0
    assert builder.TARGET_FIELD_UM == 55.0


def test_registered_mask_keeps_only_exact_inner_square_without_mutating_source() -> None:
    patch = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
    original = patch.copy()
    # Independent registered floors place this odd-width square at x=[3,8), y=[2,7).
    masked = builder._whiten_outside_registered_square(patch, (12.75, 9.25), 5)
    left, top, right, bottom = builder._registered_inner_bounds((12.75, 9.25), 10, 5)
    expected = np.full_like(patch, 255)
    expected[top:bottom, left:right] = patch[top:bottom, left:right]
    np.testing.assert_array_equal(masked, expected)
    np.testing.assert_array_equal(patch, original)
    assert (left, top, right, bottom) == (3, 2, 8, 7)


def test_features_fail_closed_on_nonfinite_zero_or_between_row_degeneracy() -> None:
    valid = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.5, 3.0]], dtype=np.float16)
    stats = builder._feature_stats(valid, "valid", expected_rows=2, expected_width=3)
    assert stats["finite"] is True
    assert stats["variable_feature_dimensions"] == 1

    with pytest.raises(ValueError, match="non-finite"):
        builder._feature_stats(np.asarray([[1.0, 2.0], [np.nan, 3.0]]), "bad")
    with pytest.raises(ValueError, match="degenerate"):
        builder._feature_stats(np.ones((2, 3), dtype=np.float32), "constant")
    with pytest.raises(ValueError, match="degenerate"):
        builder._feature_stats(np.zeros((2, 3), dtype=np.float32), "zero")


def test_source_sha_row_alignment_and_section_hashes_are_fail_closed(tmp_path: Path) -> None:
    source_path, source_sha = _source_fixture(tmp_path / "source.npz")
    source = builder._load_source(source_path, source_sha)
    contracts = builder._section_contracts(source)
    assert [item.section for item in contracts] == ["S1", "S2"]
    assert np.array_equal(contracts[0].spot_ids, ["S1:A-1", "S1:B-1"])

    with pytest.raises(ValueError, match="differs from --source-sha256"):
        builder._load_source(source_path, "0" * 64)

    bad_path, bad_sha = _source_fixture(
        tmp_path / "bad_source.npz",
        spot_ids=np.asarray(["wrong", "S1:B-1", "S2:C-1", "S2:D-1"]),
    )
    with pytest.raises(ValueError, match="section:barcode row aligned"):
        builder._load_source(bad_path, bad_sha)

    changed = source.receipt["sections"][0]["embedding"]
    changed["barcodes_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="row hashes differ"):
        builder._section_contracts(source)


def test_source_encoder_identity_must_match_exact_parity_receipt(tmp_path: Path) -> None:
    source_path, source_sha = _source_fixture(tmp_path / "source.npz")
    source = builder._load_source(source_path, source_sha)
    parity = {
        "status": "passed",
        "schema": "heir.hoptimus1_official_local_parity.v1",
        "receipt_sha256": "a" * 64,
        "encoder_manifest_sha256": "b" * 64,
        "implementation_sha256": "c" * 64,
        "runtime_sha256": {"runtime.py": "d" * 64},
    }
    manifest = SimpleNamespace(
        repository="bioptimus/H-optimus-1",
        revision="revision",
        path=tmp_path / "manifest.json",
        sha256="b" * 64,
        architecture="fixture",
        checkpoint_filename="model.safetensors",
        checkpoint_sha256="1" * 64,
        config_filename="config.json",
        config_sha256="2" * 64,
        feature_width=1536,
        input_pixels=224,
        model_mpp=0.5,
    )
    source.receipt["encoder"] = {
        "repository": manifest.repository,
        "revision": manifest.revision,
        "manifest_sha256": manifest.sha256,
        "device": "cuda",
        "official_local_parity": dict(parity),
    }
    source.receipt["encoder_roles"] = {
        "primary": {
            "repository": manifest.repository,
            "revision": manifest.revision,
            "manifest_sha256": manifest.sha256,
        }
    }
    identity = builder._validate_encoder_identity(source, manifest, parity)
    assert identity["official_local_parity"] == parity

    changed = dict(parity)
    changed["receipt_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="manifest/parity identity differs"):
        builder._validate_encoder_identity(source, manifest, changed)


class _FixtureEncoder:
    def __init__(self) -> None:
        self.calls = 0
        self.patches: list[np.ndarray] = []

    def encode(self, patches: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.patches.append(patches.copy())
        means = patches.mean(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
        return np.column_stack((means, means + 1.0, means + 2.0)).astype(np.float32)


def test_section_cache_is_registration_source_and_crop_identity_bound(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("zarr")
    image_path = tmp_path / "S1.tif"
    grid = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    image = np.stack((grid % 251, (grid * 2) % 251, (grid * 3) % 251), axis=-1).astype(np.uint8)
    tifffile.imwrite(image_path, image, photometric="rgb", tile=(16, 16))
    scalefactors_path = tmp_path / "scalefactors_json.json"
    scalefactors_path.write_text(json.dumps({"spot_diameter_fullres": 11.0}), encoding="utf-8")
    positions_path = tmp_path / "tissue_positions.csv"
    positions_path.write_text("fixture positions\n", encoding="utf-8")
    alignment_path = tmp_path / "final_alignment.json"
    alignment_path.write_text('{"fixture": true}\n', encoding="utf-8")
    alignment_qc_path = tmp_path / "aligned_tissue_image.jpg"
    alignment_qc_path.write_bytes(b"fixture-qc-image")
    pixels = np.asarray([[20.25, 20.75], [40.25, 40.75]], dtype=np.float64)
    barcodes = np.asarray(["A-1", "B-1"])
    spot_ids = np.asarray(["S1:A-1", "S1:B-1"])
    outer_width, crop = builder.V1._spot_crop_pixels(scalefactors_path)
    with builder.V1._TiffRegionReader(image_path) as reader:
        registration = builder.V1._registration_qc(reader, pixels, outer_width)
    image_sha = _sha256(image_path)
    receipt = {
        "section": "S1",
        "spot_count": 2,
        "embedding": {
            "image": {"path": str(image_path), "sha256": image_sha},
            "scalefactors_path": str(scalefactors_path),
            "scalefactors_sha256": _sha256(scalefactors_path),
            "barcodes_sha256": builder.V1._array_sha256(np.asarray(barcodes, dtype="S")),
            "pixel_xy_sha256": builder.V1._array_sha256(pixels),
            "crop": crop,
            "registration_qc": registration,
        },
        "spaceranger_provenance": {
            "schema": "heir.natcommun_spaceranger_section_provenance.v1",
            "h_and_e_path": str(image_path),
            "h_and_e_sha256": image_sha,
            "final_alignment_path": str(alignment_path),
            "final_alignment_sha256": _sha256(alignment_path),
            "alignment_qc_image_path": str(alignment_qc_path),
            "alignment_qc_image_sha256": _sha256(alignment_qc_path),
            "alignment_visual_review_required_before_exact_image_claims": True,
            "exact_invocation_fields_verified": True,
        },
        "tissue_positions": {
            "path": str(positions_path),
            "sha256": _sha256(positions_path),
        },
    }
    source = builder.SourceContract(
        path=tmp_path / "source.npz",
        sha256="e" * 64,
        schema=builder.V1.SOURCE_SCHEMA,
        spot_ids=spot_ids,
        barcode_ids=barcodes,
        section_ids=np.asarray(["S1", "S1"]),
        pixel_xy=pixels,
        receipt={"sections": [receipt]},
        native_feature_stats={},
    )
    section = builder.SectionContract(
        section="S1",
        row_indices=np.asarray([0, 1], dtype=np.int64),
        spot_ids=spot_ids,
        barcodes=barcodes,
        pixel_xy=pixels,
        receipt=receipt,
    )
    manifest = SimpleNamespace(feature_width=3, input_pixels=224)
    encoder = _FixtureEncoder()
    first, first_receipt = builder._section_embeddings(
        source=source,
        section=section,
        encoder=encoder,
        manifest=manifest,
        encoder_identity={"manifest_sha256": "a" * 64, "device": "cuda"},
        cache_dir=tmp_path / "cache",
        batch_size=2,
    )
    assert first.shape == (2, 3)
    assert first.dtype == np.float16
    assert encoder.calls == 1
    assert encoder.patches[0].shape == (2, 224, 224, 3)
    assert np.all(encoder.patches[0][:, 0, 0] == 255)
    assert first_receipt["target_crop"]["retained_center_fullres_pixels"] == 11
    assert first_receipt["target_crop"]["source_canvas_fullres_pixels"] == 22
    assert first_receipt["cache_status"] == "created"

    second, second_receipt = builder._section_embeddings(
        source=source,
        section=section,
        encoder=encoder,
        manifest=manifest,
        encoder_identity={"manifest_sha256": "a" * 64, "device": "cuda"},
        cache_dir=tmp_path / "cache",
        batch_size=2,
    )
    np.testing.assert_array_equal(second, first)
    assert encoder.calls == 1
    assert second_receipt["cache_status"] == "reused"

    cache_path = tmp_path / "cache" / "S1.55um_masked_context.npz"
    with np.load(cache_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    assert str(payload["image_features_sha256"].item()) == builder.V1._array_sha256(first)
    tampered = payload["image_features_55um"].copy()
    tampered[0, 0] += np.float16(1.0)
    payload["image_features_55um"] = tampered
    builder.V1._atomic_npz(cache_path, payload)

    repaired, repaired_receipt = builder._section_embeddings(
        source=source,
        section=section,
        encoder=encoder,
        manifest=manifest,
        encoder_identity={"manifest_sha256": "a" * 64, "device": "cuda"},
        cache_dir=tmp_path / "cache",
        batch_size=2,
    )
    np.testing.assert_array_equal(repaired, first)
    assert encoder.calls == 2
    assert repaired_receipt["cache_status"] == "created"

    receipt["embedding"]["registration_qc"] = {
        **registration,
        "all_spot_centers_inside_image": False,
    }
    with pytest.raises(ValueError, match="registered H&E geometry changed"):
        builder._section_embeddings(
            source=source,
            section=section,
            encoder=encoder,
            manifest=manifest,
            encoder_identity={"manifest_sha256": "a" * 64, "device": "cuda"},
            cache_dir=tmp_path / "cache",
            batch_size=2,
        )


def test_cli_exposes_cuda_only_and_exact_source_hash() -> None:
    args = builder.parse_args(["--source-sha256", "a" * 64])
    assert args.device == "cuda"
    assert args.batch_size <= builder.MAX_BATCH_SIZE
    with pytest.raises(SystemExit):
        builder.parse_args(["--source-sha256", "a" * 64, "--device", "cpu"])
