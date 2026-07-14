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
    path = Path(__file__).parents[1] / "scripts" / "build_natcommun_uni2_sensitivity.py"
    spec = importlib.util.spec_from_file_location("build_natcommun_uni2_sensitivity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fixture(path: Path, *, spot_ids: np.ndarray | None = None) -> tuple[Path, str]:
    sections = np.asarray(["S1", "S1", "S2", "S2"])
    barcodes = np.asarray(["A-1", "B-1", "C-1", "D-1"])
    spots = (
        np.asarray(spot_ids)
        if spot_ids is not None
        else np.asarray(["S1:A-1", "S1:B-1", "S2:C-1", "S2:D-1"])
    )
    pixels = np.asarray([[20.0, 20.0], [30.0, 30.0], [25.0, 25.0], [35.0, 35.0]])
    section_receipts = []
    for section in ("S1", "S2"):
        rows = np.flatnonzero(sections == section)
        section_receipts.append(
            {
                "section": section,
                "spot_count": len(rows),
                "embedding": {
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
        "sections": section_receipts,
    }
    np.savez_compressed(
        path,
        schema_version=np.asarray(builder.V1.SOURCE_SCHEMA),
        spot_ids=spots,
        barcode_ids=barcodes,
        section_ids=sections,
        pixel_xy=pixels,
        source_receipt_json=np.asarray(json.dumps(receipt, sort_keys=True)),
    )
    return path, _sha256(path)


def test_frozen_registration_helper_and_secondary_contract_are_explicit() -> None:
    assert _sha256(builder.V1_BUILDER_PATH) == builder.FROZEN_V1_BUILDER_SHA256
    assert builder.OUTPUT_SCHEMA == "heir.natcommun_uni2h_sensitivity.v1"
    assert builder.SOURCE_FIELD_UM == 112.0
    assert builder.TARGET_FIELD_UM == 55.0
    assert builder.MAX_BATCH_SIZE == 8

    manifest = builder.V1.load_encoder_manifest(builder.DEFAULT_ENCODER_MANIFEST)
    assert manifest.repository == "MahmoodLab/UNI2-h"
    assert manifest.interpolation == "bilinear"
    assert manifest.feature_width == 1536


def test_registered_whitening_uses_independent_floors_and_preserves_input() -> None:
    patch = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
    original = patch.copy()
    whitened = builder._whiten_outside_registered_square(patch, (12.75, 9.25), 5)
    left, top, right, bottom = builder._registered_inner_bounds((12.75, 9.25), 10, 5)
    expected = np.full_like(patch, 255)
    expected[top:bottom, left:right] = patch[top:bottom, left:right]
    np.testing.assert_array_equal(whitened, expected)
    np.testing.assert_array_equal(patch, original)
    assert (left, top, right, bottom) == (3, 2, 8, 7)


def test_source_hash_row_alignment_and_section_hashes_fail_closed(tmp_path: Path) -> None:
    source_path, source_sha = _source_fixture(tmp_path / "source.npz")
    source = builder._load_source(source_path, source_sha)
    sections = builder._section_contracts(source)
    assert [section.section for section in sections] == ["S1", "S2"]
    assert np.array_equal(sections[0].spot_ids, ["S1:A-1", "S1:B-1"])

    with pytest.raises(ValueError, match="differs from --source-sha256"):
        builder._load_source(source_path, "0" * 64)
    bad_path, bad_sha = _source_fixture(
        tmp_path / "bad.npz",
        spot_ids=np.asarray(["wrong", "S1:B-1", "S2:C-1", "S2:D-1"]),
    )
    with pytest.raises(ValueError, match="section:barcode row aligned"):
        builder._load_source(bad_path, bad_sha)

    source.receipt["sections"][0]["embedding"]["pixel_xy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="row hashes differ"):
        builder._section_contracts(source)


def test_feature_stats_reject_nonfinite_zero_and_between_row_degeneracy() -> None:
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


def test_encoder_identity_binds_model_files_without_claiming_parity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.bin"
    config = tmp_path / "config.json"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text('{"fixture": true}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = SimpleNamespace(
        repository="MahmoodLab/UNI2-h",
        implementation="uni2h_timm",
        architecture="vit_giant_patch14_224",
        feature_width=1536,
        input_pixels=224,
        interpolation="bilinear",
        pooling_rule="direct_features",
        fine_tuning="prohibited",
        checkpoint_filename=checkpoint.name,
        checkpoint_sha256=_sha256(checkpoint),
        config_filename=config.name,
        config_sha256=_sha256(config),
        revision="revision",
        path=manifest_path,
        sha256="a" * 64,
        model_mpp=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        license="CC-BY-NC-ND-4.0",
        known_training_datasets=(),
        evaluation_overlap="unknown",
    )
    identity = builder._encoder_identity(manifest, tmp_path)
    assert identity["checkpoint_sha256"] == _sha256(checkpoint)
    assert identity["config_sha256"] == _sha256(config)
    assert identity["official_local_parity_claim"] == "none_not_assessed"

    config.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the encoder manifest"):
        builder._encoder_identity(manifest, tmp_path)


class _FixtureEncoder:
    def __init__(self) -> None:
        self.patches: list[np.ndarray] = []

    def encode(self, patches: np.ndarray) -> np.ndarray:
        self.patches.append(patches.copy())
        means = patches.mean(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
        return np.column_stack((means, means + 1.0, means + 2.0)).astype(np.float32)


def test_blank_control_is_manifest_bound_finite_and_nonzero() -> None:
    encoder = _FixtureEncoder()
    manifest = SimpleNamespace(input_pixels=14, feature_width=3)
    blank, receipt = builder._blank_feature(encoder, manifest)
    assert blank.shape == (3,)
    assert blank.dtype == np.float16
    assert np.isfinite(blank).all()
    assert receipt["input_shape"] == [1, 14, 14, 3]
    assert receipt["applies_to"] == ["natural_112um", "centered_55um_whitened"]
    assert receipt["array_sha256"] == builder.V1._array_sha256(blank)
    assert receipt["squared_norm"] > 0


def test_section_cache_encodes_native_natural_and_whitened_canvases(tmp_path: Path) -> None:
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
    )
    section = builder.SectionContract(
        section="S1",
        row_indices=np.asarray([0, 1], dtype=np.int64),
        spot_ids=spot_ids,
        barcodes=barcodes,
        pixel_xy=pixels,
        receipt=receipt,
    )
    encoder = _FixtureEncoder()
    manifest = SimpleNamespace(feature_width=3)
    natural, centered, first_receipt = builder._section_features(
        source=source,
        section=section,
        encoder=encoder,
        manifest=manifest,
        encoder_identity={"manifest_sha256": "a" * 64, "device": "cuda"},
        implementation={"builder_sha256": "b" * 64},
        cache_dir=tmp_path / "cache",
        batch_size=2,
    )
    assert natural.shape == centered.shape == (2, 3)
    assert natural.dtype == centered.dtype == np.float16
    assert len(encoder.patches) == 2
    assert encoder.patches[0].shape == (2, outer_width, outer_width, 3)
    assert encoder.patches[1].shape == (2, outer_width, outer_width, 3)
    assert outer_width != 224
    assert np.any(encoder.patches[0][:, 0, 0] != 255)
    assert np.all(encoder.patches[1][:, 0, 0] == 255)
    assert not np.array_equal(natural, centered)
    assert first_receipt["preprocessing"]["explicit_pre_encoder_resize"] is False
    assert first_receipt["cache_status"] == "created"

    cached_natural, cached_centered, second_receipt = builder._section_features(
        source=source,
        section=section,
        encoder=encoder,
        manifest=manifest,
        encoder_identity={"manifest_sha256": "a" * 64, "device": "cuda"},
        implementation={"builder_sha256": "b" * 64},
        cache_dir=tmp_path / "cache",
        batch_size=2,
    )
    np.testing.assert_array_equal(cached_natural, natural)
    np.testing.assert_array_equal(cached_centered, centered)
    assert len(encoder.patches) == 2
    assert second_receipt["cache_status"] == "reused"

    cache_path = tmp_path / "cache" / "S1.uni2h_112um_55um.npz"
    with np.load(cache_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    assert str(payload["image_features_112um_sha256"].item()) == (builder.V1._array_sha256(natural))
    assert str(payload["image_features_55um_sha256"].item()) == (builder.V1._array_sha256(centered))

    tampered_natural = payload["image_features_112um"].copy()
    tampered_natural[0, 0] += np.float16(1.0)
    payload["image_features_112um"] = tampered_natural
    builder.V1._atomic_npz(cache_path, payload)
    repaired_natural, repaired_centered, natural_repair_receipt = builder._section_features(
        source=source,
        section=section,
        encoder=encoder,
        manifest=manifest,
        encoder_identity={"manifest_sha256": "a" * 64, "device": "cuda"},
        implementation={"builder_sha256": "b" * 64},
        cache_dir=tmp_path / "cache",
        batch_size=2,
    )
    np.testing.assert_array_equal(repaired_natural, natural)
    np.testing.assert_array_equal(repaired_centered, centered)
    assert len(encoder.patches) == 4
    assert natural_repair_receipt["cache_status"] == "created"

    with np.load(cache_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    tampered_centered = payload["image_features_55um"].copy()
    tampered_centered[0, 0] += np.float16(1.0)
    payload["image_features_55um"] = tampered_centered
    builder.V1._atomic_npz(cache_path, payload)
    final_natural, final_centered, centered_repair_receipt = builder._section_features(
        source=source,
        section=section,
        encoder=encoder,
        manifest=manifest,
        encoder_identity={"manifest_sha256": "a" * 64, "device": "cuda"},
        implementation={"builder_sha256": "b" * 64},
        cache_dir=tmp_path / "cache",
        batch_size=2,
    )
    np.testing.assert_array_equal(final_natural, natural)
    np.testing.assert_array_equal(final_centered, centered)
    assert len(encoder.patches) == 6
    assert centered_repair_receipt["cache_status"] == "created"


def test_run_writes_exact_row_aligned_public_output_and_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.npz"
    source_path.write_bytes(b"frozen source fixture")
    source_sha = _sha256(source_path)
    spot_ids = np.asarray(["S1:A", "S1:B", "S2:C", "S2:D"])
    source = builder.SourceContract(
        path=source_path,
        sha256=source_sha,
        schema=builder.V1.SOURCE_SCHEMA,
        spot_ids=spot_ids,
        barcode_ids=np.asarray(["A", "B", "C", "D"]),
        section_ids=np.asarray(["S1", "S1", "S2", "S2"]),
        pixel_xy=np.asarray([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float64),
        receipt={"builder_implementation_sha256": builder.FROZEN_V1_BUILDER_SHA256},
    )
    sections = tuple(
        builder.SectionContract(
            section=name,
            row_indices=np.asarray(rows, dtype=np.int64),
            spot_ids=spot_ids[rows],
            barcodes=source.barcode_ids[rows],
            pixel_xy=source.pixel_xy[rows],
            receipt={},
        )
        for name, rows in (("S1", [0, 1]), ("S2", [2, 3]))
    )
    manifest = SimpleNamespace(feature_width=3, input_pixels=224)

    class RunEncoder:
        def encode(self, patches: np.ndarray) -> np.ndarray:
            rows = len(patches)
            return np.tile(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32), (rows, 1))

    run_encoder = RunEncoder()

    monkeypatch.setattr(builder, "_load_source", lambda *_args: source)
    monkeypatch.setattr(builder, "_section_contracts", lambda _source: sections)
    monkeypatch.setattr(builder.V1, "load_encoder_manifest", lambda _path: manifest)
    monkeypatch.setattr(builder.V1, "create_frozen_encoder", lambda *_args, **_kwargs: run_encoder)
    monkeypatch.setattr(
        builder,
        "_encoder_identity",
        lambda *_args: {
            "repository": "MahmoodLab/UNI2-h",
            "official_local_parity_claim": "none_not_assessed",
        },
    )

    def fake_section_features(*, section, **_kwargs):
        offset = int(section.row_indices[0])
        natural = np.asarray([[1.0 + offset, 2.0, 3.0], [1.5 + offset, 2.5, 3.5]], dtype=np.float16)
        centered = natural + np.float16(0.25)
        return natural, centered, {"section": section.section}

    monkeypatch.setattr(builder, "_section_features", fake_section_features)
    output = tmp_path / "uni2h.npz"
    args = builder.parse_args(
        [
            "--source",
            str(source_path),
            "--source-sha256",
            source_sha,
            "--output",
            str(output),
        ]
    )
    assert builder.run(args) == 0
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_version",
            "spot_ids",
            "image_features_112um",
            "image_features_55um",
            "blank_image_feature_vector",
            "source_sha256",
            "receipt_json",
        }
        np.testing.assert_array_equal(archive["spot_ids"].astype(str), spot_ids)
        assert archive["image_features_112um"].shape == (4, 3)
        assert archive["image_features_55um"].shape == (4, 3)
        assert archive["blank_image_feature_vector"].shape == (3,)
        assert str(archive["source_sha256"].item()) == source_sha
        receipt = json.loads(str(archive["receipt_json"].item()))
    assert receipt["encoder"]["official_local_parity_claim"] == "none_not_assessed"
    assert receipt["scientific_role"]["encoder"].startswith("secondary_sensitivity")
    assert receipt["row_alignment"]["output_spot_ids_exactly_equal_source"] is True
    assert receipt["blank_image_control"]["array_sha256"] == builder.V1._array_sha256(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float16)
    )


def test_cli_is_cuda_only_and_run_enforces_batch_one_through_eight() -> None:
    args = builder.parse_args(["--source-sha256", "a" * 64])
    assert args.device == "cuda"
    assert 1 <= args.batch_size <= builder.MAX_BATCH_SIZE
    with pytest.raises(SystemExit):
        builder.parse_args(["--source-sha256", "a" * 64, "--device", "cpu"])
    too_large = builder.parse_args(
        ["--source-sha256", "a" * 64, "--batch-size", str(builder.MAX_BATCH_SIZE + 1)]
    )
    with pytest.raises(ValueError, match="between 1 and 8"):
        builder.run(too_large)
