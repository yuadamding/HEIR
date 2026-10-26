"""H&E-only region geometry, orientation, and fixed image feature contracts."""

import json

import numpy as np
import pytest

from heir.features.handcrafted import HandcraftedPatchEncoder
from heir.personalized.regions import build_region_grid, point_to_grid


def test_lower_origin_uses_array_row_as_y_without_flip():
    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    image[:2, :2] = [180, 60, 120]
    image[2:, 2:] = [120, 80, 180]
    grid = build_region_grid(image, bin_size_pixels=2)
    assert grid.region_ids == ("he_region:0:0", "he_region:1:1")
    np.testing.assert_array_equal(grid.bin_keys, [[0, 0], [1, 1]])
    np.testing.assert_allclose(grid.coordinates, [[1, 1], [3, 3]])
    np.testing.assert_allclose(
        grid.image_features[:, :3], np.array([[180, 60, 120], [120, 80, 180]]) / 255
    )
    np.testing.assert_array_equal(point_to_grid(grid, [[0, 0], [3, 3], [0, 3]]), [0, 1, -1])
    assert grid.settings["coordinate_units"] == "registered_pixel"
    assert grid.settings["array_row_is_y"] is True
    json.dumps(grid.settings)  # Persistable, target-independent preprocessing contract.


def test_fixed_tissue_thresholds_exclude_blank_and_dark_regions():
    image = np.full((2, 6, 3), 255, dtype=np.uint8)
    image[:, 2:4] = [0, 0, 0]
    image[0, 4] = [180, 60, 120]
    grid = build_region_grid(image, bin_size_pixels=2, min_tissue_fraction=0.25)
    np.testing.assert_array_equal(grid.bin_keys, [[2, 0]])
    np.testing.assert_allclose(grid.tissue_fraction, [0.25])
    with pytest.raises(ValueError, match="no regions"):
        build_region_grid(image, bin_size_pixels=2, min_tissue_fraction=0.26)


def test_partial_bins_use_actual_area_features_and_clipped_bounds():
    image = np.empty((3, 5, 3), dtype=np.uint8)
    image[:] = [170, 90, 130]
    image[2, 4] = [100, 40, 160]
    grid = build_region_grid(image, bin_size_pixels=2)
    np.testing.assert_array_equal(grid.bin_keys[-1], [2, 1])
    np.testing.assert_allclose(grid.coordinates[-1], [4.5, 2.5])
    np.testing.assert_allclose(grid.tissue_fraction, 1)
    expected = HandcraftedPatchEncoder().encode(image[2:3, 4:5][None])[0]
    np.testing.assert_allclose(grid.image_features[-1], expected)
    points = [[4, 2], [4.999, 2.999], [5, 2], [4, 3], [-0.01, 0], [2, 0]]
    np.testing.assert_array_equal(point_to_grid(grid, points), [5, 5, -1, -1, -1, 1])


def test_components_separate_gaps_and_diagonal_contact():
    image = np.full((6, 8, 3), 255, dtype=np.uint8)
    for x, y in [(0, 0), (1, 0), (3, 0), (2, 1), (2, 2)]:
        image[y * 2 : (y + 1) * 2, x * 2 : (x + 1) * 2] = [180, 60, 120]
    grid = build_region_grid(image, bin_size_pixels=2)
    assert grid.component_ids == (
        "he_component:0",
        "he_component:0",
        "he_component:1",
        "he_component:2",
        "he_component:2",
    )


def test_point_coverage_never_changes_the_grid_and_replicates_remain_separate():
    image = np.full((2, 4, 3), 255, dtype=np.uint8)
    image[:, :2] = [180, 60, 120]
    first = build_region_grid(image, bin_size_pixels=2)
    np.testing.assert_array_equal(point_to_grid(first, [[0.5, 0.5], [0.5, 0.5]]), [0, 0])
    np.testing.assert_array_equal(point_to_grid(first, [[3, 0.5]]), [-1])
    second = build_region_grid(image, bin_size_pixels=2)
    assert first.region_ids == second.region_ids
    np.testing.assert_array_equal(first.image_features, second.image_features)
    np.testing.assert_array_equal(first.bin_keys, second.bin_keys)
    assert not first.settings["contains_spatial_rna_derived_features"]
    assert not first.settings["contains_protein_derived_features"]


def test_image_path_and_array_give_same_regions(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    image = np.full((3, 4, 3), [180, 60, 120], dtype=np.uint8)
    path = tmp_path / "processed.png"
    image_module.fromarray(image).save(path)
    from_path = build_region_grid(path, bin_size_pixels=2)
    from_array = build_region_grid(image, bin_size_pixels=2)
    np.testing.assert_array_equal(from_path.coordinates, from_array.coordinates)
    np.testing.assert_array_equal(from_path.image_features, from_array.image_features)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_bin_size_rejected(size):
    with pytest.raises(ValueError, match="positive integer"):
        build_region_grid(np.zeros((2, 2, 3), dtype=np.uint8), bin_size_pixels=size)


def test_invalid_images_thresholds_and_points_rejected():
    with pytest.raises(ValueError, match="uint8 RGB"):
        build_region_grid(np.zeros((2, 2, 3), dtype=float))
    image = np.full((2, 2, 3), [180, 60, 120], dtype=np.uint8)
    for kwargs in [
        {"min_tissue_fraction": 0},
        {"min_saturation": np.nan},
        {"min_intensity": 0.9, "max_intensity": 0.2},
    ]:
        with pytest.raises(ValueError):
            build_region_grid(image, **kwargs)
    grid = build_region_grid(image)
    with pytest.raises(ValueError, match="finite N-by-2"):
        point_to_grid(grid, [[np.nan, 0]])
    assert point_to_grid(grid, np.empty((0, 2))).shape == (0,)
