"""H&E-only frozen feature caches preserve the exact regional measurement axis."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from heir.features import load_encoder_manifest
from heir.personalized.region_features import (
    grid_identity,
    load_feature_cache,
    write_feature_cache,
)
from heir.personalized.regions import build_region_grid


class StubEncoder:
    def __init__(self, manifest):
        self.manifest_sha256 = manifest.sha256
        self.feature_width = manifest.feature_width
        self.batches = []
        self.callback = None

    def encode(self, patches):
        self.batches.append(patches.copy())
        if self.callback is not None:
            self.callback()
        features = np.zeros((len(patches), self.feature_width), dtype=np.float32)
        features[:, :3] = patches.mean((1, 2))
        features[:, 3:6] = patches[:, 0, 0]
        return features


@pytest.fixture
def image_case(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    image = np.full((3, 5, 3), [170, 70, 130], dtype=np.uint8)
    image[2, :, 0] = [100, 120, 140, 160, 180]
    image[:, 4, 1] = [20, 40, 60]
    source = tmp_path / "processed.png"
    image_module.fromarray(image).save(source)
    grid = build_region_grid(image, bin_size_pixels=2)
    manifest = load_encoder_manifest(
        Path(__file__).parents[1] / "manifests/encoders/hoptimus1.json"
    )
    return source, grid, manifest, hashlib.sha256(source.read_bytes()).hexdigest(), image


def _write(case, output, encoder=None, **kwargs):
    source, grid, manifest, digest, _ = case
    encoder = encoder or StubEncoder(manifest)
    metadata = write_feature_cache(
        source, grid, output, encoder, manifest, expected_image_sha256=digest, **kwargs
    )
    return encoder, metadata


def _load(case, path, **kwargs):
    _, grid, manifest, digest, _ = case
    return load_feature_cache(
        path,
        grid,
        expected_image_sha256=digest,
        expected_encoder_manifest_sha256=manifest.sha256,
        **kwargs,
    )


def test_cache_uses_exact_unpadded_unflipped_regions_and_restores_order(image_case, tmp_path):
    _, grid, manifest, digest, image = image_case
    path = tmp_path / "features.npz"
    encoder, metadata = _write(image_case, path, batch_size=2)
    features, loaded = _load(image_case, path)
    assert features.shape == (6, 1536)
    assert features.dtype == np.float32
    assert all(len(batch) <= 2 for batch in encoder.batches)
    assert {batch.shape[1:3] for batch in encoder.batches} == {(2, 2), (2, 1), (1, 2), (1, 1)}
    for row, (x, y) in enumerate(grid.bin_keys):
        crop = image[y * 2 : min(y * 2 + 2, 3), x * 2 : min(x * 2 + 2, 5)]
        np.testing.assert_allclose(features[row, :3], crop.mean((0, 1)))
        np.testing.assert_array_equal(features[row, 3:6], crop[0, 0])
    assert loaded == metadata
    assert loaded["image_sha256"] == digest
    assert loaded["grid_sha256"] == grid_identity(grid)
    assert loaded["encoder"]["checkpoint_sha256"] == manifest.checkpoint_sha256
    assert loaded["encoder"]["config_sha256"] == manifest.config_sha256
    assert loaded["encoder"]["normalization"]["mean"] == list(manifest.mean)
    assert loaded["frozen"] is True
    assert loaded["physical_scale_verified"] is False
    assert loaded["coordinate_units"] == "registered_pixel"
    assert loaded["preferred_model_mpp"] == 0.5
    assert loaded["contains_spatial_rna_derived_features"] is False
    assert loaded["contains_protein_derived_features"] is False
    # The cache remains sufficient after the source image disappears; no molecular
    # data or encoder checkpoints exist anywhere in this fixture.
    image_case[0].unlink()
    np.testing.assert_array_equal(_load(image_case, path)[0], features)


def test_grid_identity_ignores_only_replaceable_rgb_features(image_case):
    _, grid, _, _, _ = image_case
    replacement = replace(
        grid,
        image_features=np.full((len(grid.region_ids), 1536), np.nan),
        settings={
            **grid.settings,
            "encoder_name": "replacement",
            "encoder_manifest_sha256": "a" * 64,
        },
    )
    assert grid_identity(grid) == grid_identity(replacement)
    changed = replace(grid, tissue_fraction=np.full(len(grid.region_ids), 0.5))
    assert grid_identity(grid) != grid_identity(changed)
    reordered = replace(
        grid,
        region_ids=grid.region_ids[::-1],
        coordinates=grid.coordinates[::-1],
        bin_keys=grid.bin_keys[::-1],
        tissue_fraction=grid.tissue_fraction[::-1],
        component_ids=grid.component_ids[::-1],
    )
    assert grid_identity(grid) != grid_identity(reordered)


@pytest.mark.parametrize("change", ["bins", "centers", "image_size", "orientation", "nan"])
def test_invalid_grid_geometry_rejected_before_encoding(image_case, tmp_path, change):
    source, grid, manifest, digest, image = image_case
    if change == "bins":
        grid = replace(grid, settings={**grid.settings, "bin_size_pixels": 3})
    elif change == "centers":
        grid = replace(grid, coordinates=grid.coordinates + 0.1)
    elif change == "image_size":
        # A geometrically valid grid belonging to a different image still fails.
        grid = build_region_grid(np.tile(image, (2, 1, 1)), bin_size_pixels=2)
    elif change == "orientation":
        grid = replace(grid, settings={**grid.settings, "array_row_is_y": False})
    else:
        grid = replace(grid, tissue_fraction=np.full(len(grid.region_ids), np.nan))
    encoder = StubEncoder(manifest)
    with pytest.raises(ValueError, match="grid|dimensions"):
        write_feature_cache(
            source,
            grid,
            tmp_path / "invalid.npz",
            encoder,
            manifest,
            expected_image_sha256=digest,
        )
    assert not encoder.batches


def test_changed_image_before_or_during_encoding_never_publishes(image_case, tmp_path):
    source, grid, manifest, digest, _ = image_case
    encoder = StubEncoder(manifest)
    path = tmp_path / "features.npz"
    with pytest.raises(ValueError, match="checksum"):
        write_feature_cache(source, grid, path, encoder, manifest, expected_image_sha256="0" * 64)
    assert not encoder.batches
    encoder.callback = lambda: source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed during"):
        _write(image_case, path, encoder=encoder)
    assert not path.exists()


@pytest.mark.parametrize("change", ["manifest", "width", "batch_zero", "batch_bool", "values"])
def test_encoder_and_batch_contracts_rejected(image_case, tmp_path, change):
    encoder = StubEncoder(image_case[2])
    kwargs = {}
    if change == "manifest":
        encoder.manifest_sha256 = "a" * 64
    elif change == "width":
        encoder.feature_width = 42
    elif change == "batch_zero":
        kwargs["batch_size"] = 0
    elif change == "batch_bool":
        kwargs["batch_size"] = True
    else:
        encoder.encode = lambda patches: np.full((len(patches), encoder.feature_width), np.nan)
    with pytest.raises(ValueError, match="encoder|batch_size"):
        _write(image_case, tmp_path / "invalid.npz", encoder=encoder, **kwargs)
    assert not (tmp_path / "invalid.npz").exists()


@pytest.mark.parametrize("change", ["features", "dtype", "ids", "coordinates", "encoder", "crop"])
def test_cache_corruption_is_rejected(image_case, tmp_path, change):
    path = tmp_path / "features.npz"
    _write(image_case, path)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if change == "features":
        values["features"][0, 0] += 1
    elif change == "dtype":
        values["features"] = values["features"].astype(np.float64)
    elif change == "ids":
        values["region_ids"] = values["region_ids"][::-1]
    elif change == "coordinates":
        values["coordinates"][0, 0] += 1
    else:
        metadata = json.loads(values["metadata_json"].item())
        if change == "encoder":
            metadata["encoder"]["checkpoint_sha256"] = "a" * 64
        else:
            metadata["crop_definition"] = "padded"
        values["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="feature cache"):
        _load(image_case, path)


def test_wrong_grid_or_expected_identity_rejected_and_existing_cache_preserved(
    image_case, tmp_path
):
    source, grid, manifest, digest, _ = image_case
    path = tmp_path / "features.npz"
    _write(image_case, path)
    original = path.read_bytes()
    encoder = StubEncoder(manifest)
    with pytest.raises(FileExistsError):
        _write(image_case, path, encoder=encoder)
    assert path.read_bytes() == original
    assert not encoder.batches
    for altered_grid, image_hash, manifest_hash in [
        (
            replace(grid, tissue_fraction=np.full(len(grid.region_ids), 0.5)),
            digest,
            manifest.sha256,
        ),
        (grid, "a" * 64, manifest.sha256),
        (grid, digest, "a" * 64),
    ]:
        with pytest.raises(ValueError, match="identity mismatch"):
            load_feature_cache(
                path,
                altered_grid,
                expected_image_sha256=image_hash,
                expected_encoder_manifest_sha256=manifest_hash,
            )
