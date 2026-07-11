import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from heir.image import load_feature_bundle, load_nuclei
from heir.segmentation.spaceranger import (
    MORPHOLOGY_FEATURE_NAMES,
    SEGMENTATION_METHOD,
    discover_spaceranger_executable,
    export_spaceranger_artifacts,
    read_spaceranger_geojson,
    run_spaceranger_segment,
)


def _write_geojson(path: Path, duplicate: bool = False) -> Path:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [4, 0], [4, 2], [0, 2], [0, 0]]],
                },
                "properties": {"cell_id": 10, "nucleus_centroid": [2, 1]},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[10, 10], [12, 10], [12, 12], [10, 12], [10, 10]]],
                },
                "properties": {"cell_id": 10 if duplicate else 11},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[20, 20], [21, 20], [21, 21], [20, 21], [20, 20]]],
                },
                "properties": {"cell_id": 12, "nucleus_centroid": [20.5, 20.5]},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_geojson_parser_namespaces_ids_and_derives_ten_features(tmp_path):
    source = _write_geojson(tmp_path / "nucleus_segmentations.geojson")
    result = read_spaceranger_geojson(
        source,
        slide_id="B1_4",
        spaceranger_version="spaceranger 4.1.0",
        minimum_area_px2=2.0,
    )

    assert result.nucleus_ids.tolist() == ["B1_4::10", "B1_4::11"]
    assert result.source_ids.tolist() == ["10", "11"]
    assert result.morphology.shape == (2, 10)
    np.testing.assert_allclose(result.morphology[:, 0], [8.0, 4.0])
    np.testing.assert_allclose(result.centroids_px, [[2.0, 1.0], [11.0, 11.0]])
    assert result.spaceranger_version == "4.1.0"
    assert result.method == SEGMENTATION_METHOD
    assert result.skipped_features == 1
    assert result.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    table = result.to_nucleus_table()
    assert table.morphology_names == MORPHOLOGY_FEATURE_NAMES
    assert np.isnan(table.confidence).all()
    assert table.metadata["segmentation_version"].tolist() == ["4.1.0", "4.1.0"]


def test_exported_csv_and_npz_are_safe_and_ready_for_heir_loaders(tmp_path):
    segmentation = read_spaceranger_geojson(
        _write_geojson(tmp_path / "nucleus_segmentations.geojson"),
        slide_id="slide1",
        spaceranger_version="4.0.1",
        minimum_area_px2=0.0,
    )
    csv_path, npz_path = export_spaceranger_artifacts(
        segmentation,
        csv_path=tmp_path / "nested" / "nuclei.csv",
        npz_path=tmp_path / "nested" / "features.npz",
    )

    nuclei = load_nuclei(csv_path)
    assert nuclei.nucleus_ids.tolist() == segmentation.nucleus_ids.tolist()
    assert nuclei.morphology.shape == (3, 10)
    assert nuclei.morphology_names == MORPHOLOGY_FEATURE_NAMES
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["confidence"] == ""
    assert rows[0]["segmentation_method"] == SEGMENTATION_METHOD
    assert rows[0]["segmentation_source_sha256"] == segmentation.source_sha256

    bundle = load_feature_bundle(npz_path, expected_ids=segmentation.nucleus_ids)
    assert bundle.features.shape == (3, 10)
    assert np.isfinite(bundle.features).all()
    np.testing.assert_allclose(bundle.coordinates, segmentation.centroids_px)
    with np.load(npz_path, allow_pickle=False) as archive:
        assert not any(archive[name].dtype.hasobject for name in archive.files)
        assert str(archive["segmentation_version"].item()) == "4.0.1"
        assert str(archive["segmentation_source_sha256"].item()) == segmentation.source_sha256
        np.testing.assert_allclose(archive["morphology"], segmentation.morphology)

    with pytest.raises(FileExistsError):
        export_spaceranger_artifacts(
            segmentation,
            csv_path=csv_path,
            npz_path=npz_path,
        )


def test_parser_rejects_duplicate_ids_outside_centroids_and_wrong_versions(tmp_path):
    duplicate = _write_geojson(tmp_path / "duplicate.geojson", duplicate=True)
    with pytest.raises(ValueError, match="unique"):
        read_spaceranger_geojson(
            duplicate,
            slide_id="s1",
            spaceranger_version="4.1.0",
            minimum_area_px2=0,
        )
    outside = json.loads(duplicate.read_text(encoding="utf-8"))
    outside["features"][1]["properties"]["cell_id"] = 11
    outside["features"][0]["properties"]["nucleus_centroid"] = [100, 100]
    duplicate.write_text(json.dumps(outside), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        read_spaceranger_geojson(
            duplicate,
            slide_id="s1",
            spaceranger_version="4.1.0",
            minimum_area_px2=0,
        )
    with pytest.raises(ValueError, match="4.x"):
        read_spaceranger_geojson(
            duplicate,
            slide_id="s1",
            spaceranger_version="3.1.0",
        )


def test_discovery_and_runner_use_argument_list_and_preserve_auto_cuda(tmp_path, monkeypatch):
    binary = tmp_path / "spaceranger"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("SPACERANGER", str(binary))
    assert discover_spaceranger_executable() == binary.resolve()

    image = tmp_path / "slide.tif"
    image.write_bytes(b"synthetic-image")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "spaceranger 4.1.0\n", "")
        output = Path(command[command.index("--output-dir") + 1]) / "outs"
        output.mkdir(parents=True)
        (output / "nucleus_segmentations.geojson").write_text(
            '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("heir.segmentation.spaceranger.subprocess.run", fake_run)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    result = run_spaceranger_segment(
        image,
        run_id="sample_1",
        output_directory=tmp_path / "runs",
        executable=binary,
        localcores=2,
        localmem_gb=4,
    )

    segment_command, segment_kwargs = calls[-1]
    assert isinstance(segment_command, list)
    assert segment_command[1] == "segment"
    assert "--disable-ui" in segment_command
    assert "CUDA_VISIBLE_DEVICES" not in segment_kwargs["env"]
    assert result.spaceranger_version == "4.1.0"
    assert result.geojson_path.is_file()
    assert result.tissue_image_sha256 == hashlib.sha256(image.read_bytes()).hexdigest()


def test_runner_rejects_existing_output_and_explicitly_scopes_cuda(tmp_path, monkeypatch):
    binary = tmp_path / "spaceranger"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    image = tmp_path / "slide.jpg"
    image.write_bytes(b"image")

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "spaceranger 4.1.0", "")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        output = Path(command[command.index("--output-dir") + 1]) / "outs"
        output.mkdir(parents=True)
        (output / "nucleus_segmentations.geojson").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("heir.segmentation.spaceranger.subprocess.run", fake_run)
    output_root = tmp_path / "runs"
    first = run_spaceranger_segment(
        image,
        run_id="sample",
        output_directory=output_root,
        executable=binary,
        cuda_visible_devices="2",
    )
    assert first.cuda_visible_devices == "2"
    with pytest.raises(FileExistsError):
        run_spaceranger_segment(
            image,
            run_id="sample",
            output_directory=output_root,
            executable=binary,
        )
