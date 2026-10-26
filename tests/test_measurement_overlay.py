"""Measurement-only panels preserve the shared lower-origin image/point frame."""

import json

import numpy as np
import pytest
from PIL import Image

from heir.personalized.inspection import write_measurement_overlay
from heir.personalized.regions import build_region_grid


def _source(tmp_path):
    pixels = np.empty((20, 20, 3), dtype=np.uint8)
    pixels[:10] = [180, 40, 100]
    pixels[10:] = [80, 40, 180]
    path = tmp_path / "he.png"
    Image.fromarray(pixels).save(path)
    return path, build_region_grid(pixels, bin_size_pixels=10)


def test_measurement_overlay_geometry_only_and_lower_origin(tmp_path):
    source, grid = _source(tmp_path)
    xy = np.array([[2.0, 2.0], [15.0, 15.0], [-1.0, 0.0], [2.0, 3.0]])
    original = xy.copy()
    output = tmp_path / "overlay.png"
    summary = write_measurement_overlay(
        output, source, grid, xy, ["A", "A", "A", "B"], np.array([0])
    )
    first, second = summary["panels"]
    assert first["source_beads"] == 3 and first["outside_image"] == 1
    assert first["displayed_beads"] == 2 and second["displayed_beads"] == 1
    assert first["in_common_fine_regions"] == second["in_common_fine_regions"] == 1
    assert summary["common_fine_region_ids"] == [grid.region_ids[0]]
    assert not summary["registration_qualified"] and not summary["physical_scale_verified"]
    assert summary["section_linkage"] == "unknown"
    with Image.open(output) as result:
        pixels = np.asarray(result)
    left, top, width, height = first["image_rectangle_pixels"]
    # Unmarked image positions show row zero at the bottom; beads use the same transform.
    np.testing.assert_array_equal(pixels[top + 2, left + 8], [80, 40, 180])
    np.testing.assert_array_equal(pixels[top + height - 3, left + 8], [180, 40, 100])
    assert not np.array_equal(pixels[top + height - 2, left + 2], [180, 40, 100])
    np.testing.assert_array_equal(xy, original)
    json.dumps(summary, allow_nan=False)
    with pytest.raises(FileExistsError):
        write_measurement_overlay(output, source, grid, xy, ["A", "A", "A", "B"], np.array([0]))


@pytest.mark.parametrize(
    "common", [np.array([0, 0]), np.array([4]), np.array([True]), np.array([0.5])]
)
def test_measurement_overlay_rejects_invalid_common_region_axis(tmp_path, common):
    source, grid = _source(tmp_path)
    with pytest.raises(ValueError, match="common fine-region"):
        write_measurement_overlay(tmp_path / "bad.png", source, grid, [[1, 1]], ["A"], common)


def test_measurement_overlay_empty_support_and_nonfinite_coordinates(tmp_path):
    source, grid = _source(tmp_path)
    summary = write_measurement_overlay(
        tmp_path / "empty.png", source, grid, [[-1, -1]], ["A"], np.array([], dtype=int)
    )
    assert summary["common_fine_region_ids"] == [] and summary["panels"][0]["displayed_beads"] == 0
    with pytest.raises(ValueError, match="finite"):
        write_measurement_overlay(
            tmp_path / "bad.png", source, grid, [[np.nan, 1]], ["A"], np.array([], dtype=int)
        )
