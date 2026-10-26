"""Declared CODEX fragment replay tests; no registration is estimated."""

import hashlib
import json

import numpy as np
import pytest

from heir.personalized.codex_replay import _codex_geometry, replay_codex


def _parameters(degrees=(90, -90), shift_x=(10, 30), shift_y=(20, 40)):
    return {"degrees": degrees, "shift_x": shift_x, "shift_y": shift_y}


def _fixture_geometry():
    xy = np.array([[0, 0], [10, 0], [90, 80], [0, 20], [10, 20], [0, 30]], float)
    region = np.array([1, 1, 1, 2, 2, 2])
    counts = np.array([[2, 0], [0, 2], [0, 0], [1, 2], [2, 1], [0, 1]], float)
    original = xy * 0.386
    expected = np.empty_like(original)
    for label, origin_rows, rotation, shift in (
        (1, [0, 1], [[0, -1], [1, 0]], [10, 20]),
        (2, [3, 4, 5], [[0, 1], [-1, 0]], [30, 40]),
    ):
        origin = original[origin_rows].mean(0)
        selected = region == label
        expected[selected] = (original[selected] - origin) @ np.asarray(rotation).T + origin + shift
    return xy, region, counts, original, expected


def test_fragment_rotation_uses_pre_registration_qc_and_preserves_order():
    xy, region, counts, original, expected = _fixture_geometry()
    result = _codex_geometry(xy, region, counts, _parameters(), biopsy_id="test")
    np.testing.assert_allclose(result["original"], original)
    np.testing.assert_allclose(result["replayed"], expected)
    np.testing.assert_array_equal(result["qc"], [True, True, False, True, True, True])
    assert result["chain"]["fragments"]["1"]["rotation_population_n"] == 2
    assert not result["chain"]["physical_calibration_independently_verified"]


def test_multi_fragment_branch_ignores_author_flip_column():
    xy, region, counts, _, _ = _fixture_geometry()
    baseline = _codex_geometry(xy, region, counts, _parameters(), biopsy_id="test")
    flipped = _codex_geometry(xy, region, counts, {**_parameters(), "flip": "v"}, biopsy_id="test")
    np.testing.assert_array_equal(flipped["replayed"], baseline["replayed"])


def test_514_doubles_scale_and_removes_overlap_before_fragment_centering():
    xy = np.array([[0, 0], [1, 1], [200, 5], [1000, 5], [100, 10], [5000, 10]], float)
    region = np.array([1, 1, 2, 2, 3, 3])
    result = _codex_geometry(
        xy,
        region,
        np.ones((6, 2)),
        _parameters((90, 90, 90), (0, 0, 0), (0, 0, 0)),
        biopsy_id="514-6760",
    )
    np.testing.assert_allclose(result["original"], xy * 0.772)
    np.testing.assert_array_equal(result["pre_rotation"], [True, True, False, True, True, False])
    assert result["chain"]["fragments"]["2"]["rotation_origin"] == pytest.approx([772, 3.86])
    assert result["chain"]["fragments"]["3"]["rotation_origin"] == pytest.approx([77.2, 7.72])


def test_213_overlap_line_uses_transformed_fourth_fragment_and_only_removes_third():
    xy = np.array([[0, 0], [0, 20], [5, 1], [0, 10], [10, 0]], float)
    region = np.array([1, 3, 3, 4, 4])
    result = _codex_geometry(
        xy,
        region,
        np.ones((5, 1)),
        _parameters((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        biopsy_id="213-6752",
    )
    np.testing.assert_array_equal(
        result["retained_before_HE_mask"], [True, False, True, True, True]
    )
    assert result["chain"]["post_rotation_overlap_line_213"] == pytest.approx(
        {"slope": -1, "intercept": 3.86}
    )


@pytest.mark.parametrize("regions", [[1, 1.5], [0, 1], [1, np.nan]])
def test_invalid_fragment_labels_rejected(regions):
    with pytest.raises(ValueError, match="positive integer"):
        _codex_geometry(np.ones((2, 2)), regions, np.ones((2, 2)), _parameters(), biopsy_id="test")


def test_missing_fragment_parameter_rejected():
    with pytest.raises(ValueError, match="lacks a declared"):
        _codex_geometry(
            np.array([[0, 0], [1, 1]]), [1, 3], np.ones((2, 1)), _parameters(), biopsy_id="test"
        )


@pytest.fixture
def codex_sources(tmp_path):
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    sparse = pytest.importorskip("scipy.sparse")
    image = pytest.importorskip("PIL.Image")
    root = tmp_path
    spatial = root / "reference/HTAPP-Pipelines/HTAPP_MBC/03_spatial_analysis"
    (spatial / "00.1_processing").mkdir(parents=True)
    source = []
    for replicate in (1, 2):
        for key, value in {
            "crop_left": replicate,
            "crop_right": 0,
            "crop_top": 0,
            "crop_bottom": -replicate,
            "scale": 0.6905,
            "degrees": 90 if replicate == 1 else 95.5,
        }.items():
            source.append(f'registration.loc["HE {replicate}", "{key}"]={value}\n')
    for key, value in _parameters().items():
        source.append(f'registration.loc["codex 1", "{key}"]={value!r}\n')
    (spatial / "00.1_processing/HTAPP-313-SMP-932_processing.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": source}]})
    )
    (spatial / "spatial_utils.py").write_text("# immutable fixture source\n")
    (spatial / "00_matchSpatial.ipynb").write_text('{"cells": []}')
    entry = {
        "patient_id": "HTA1_313",
        "specimen_id": "HTA1_313_932",
        "sources": [],
        "role_sources": {},
    }

    def register(path, modality, replicate):
        record = {
            "working_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "modality": modality,
            "replicate_id": replicate,
        }
        entry["sources"].append(record)
        return record

    raw = root / "working/unregistered/fixture"
    registered = root / "working/registered/fixture"
    raw.mkdir(parents=True)
    registered.mkdir(parents=True)
    xy, region, counts, original, expected = _fixture_geometry()
    ids = [f"cell{i}" for i in range(len(xy))]
    native = ad.AnnData(
        X=sparse.csc_matrix(counts),
        obs=pd.DataFrame({"xcoord": xy[:, 0], "ycoord": xy[:, 1], "region": region}, index=ids),
    )
    native_path = raw / "native_codex.h5ad"
    native.write_h5ad(native_path)
    register(native_path, "codex", "1")
    kept = np.array([4, 0, 5, 1, 3])  # Deliberately different from raw row order.
    stored = ad.AnnData(
        X=np.ones((len(kept), 1)),
        obs=pd.DataFrame(
            {
                "x": expected[kept, 0],
                "y": expected[kept, 1],
                "x_orig": original[kept, 0],
                "y_orig": original[kept, 1],
                "region": region[kept],
                "replicate": ["1"] * len(kept),
            },
            index=[ids[i] + "-1" for i in kept],
        ),
    )
    stored_path = registered / "registered_codex.h5ad"
    stored.write_h5ad(stored_path)
    entry["role_sources"]["protein"] = register(stored_path, "codex", "1")
    for replicate in ("1", "2"):
        for directory, name, shape in (
            (raw, f"{replicate}_HE_highres.jpg", (10, 12)),
            (registered, f"HE_{replicate}_processed.jpg", (5, 6)),
        ):
            path = directory / name
            image.fromarray(np.full((*shape, 3), 120, dtype=np.uint8)).save(path)
            register(path, "HE", replicate)
    return root, entry, kept


def test_full_api_exact_id_join_provenance_and_image_frame(codex_sources):
    root, entry, kept = codex_sources
    reports, chains, arrays = replay_codex(root, "313-932", entry)
    report = reports[0]
    assert report["n_native"] == 6
    assert report["post_qc_n"] == 5
    assert report["n_registered"] == report["n_matched"] == 5
    assert report["coordinate_replay_passed"]
    assert report["registered_selection_exact_before_HE_mask"]
    assert not report["biological_alignment_validated"]
    assert not report["independent_landmark_residuals_available"]
    assert report["max_replay_error_pixels"] < 1e-12
    np.testing.assert_array_equal(arrays["313-932__codex__1__registered_to_native"], kept)
    for image in chains[0]["image_chains"]:
        assert image["images"]["native_highres"]["dimensions_width_height"] == [12, 10]
        assert image["images"]["processed"]["dimensions_width_height"] == [6, 5]
        assert image["array_index_convention"] == "image[y,x]; first_stored_row_is_y0"
        assert not image["raster_replay_performed"]
        assert not image["physical_calibration_independently_verified"]


def test_full_api_verifies_input_bytes_before_replay(codex_sources):
    root, entry, _ = codex_sources
    source = entry["role_sources"]["protein"]
    source["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum differs"):
        replay_codex(root, "313-932", entry)
