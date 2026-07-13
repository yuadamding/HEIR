"""Tests for specimen-covering histology-bag capping."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "split_histology_bag", ROOT / "scripts" / "split_histology_bag.py"
)
assert SPEC is not None and SPEC.loader is not None
SPLITTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPLITTER)
_spatially_stratified_cap = SPLITTER._spatially_stratified_cap


def test_spatially_stratified_cap_is_deterministic_and_covers_quadrants() -> None:
    coordinates = np.asarray(
        [
            [x_offset + x, y_offset + y]
            for x_offset, y_offset in ((0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (100.0, 100.0))
            for x in range(4)
            for y in range(4)
        ],
        dtype=np.float64,
    )
    mask = np.ones(len(coordinates), dtype=np.bool_)

    first = _spatially_stratified_cap(coordinates, mask, maximum=8)
    second = _spatially_stratified_cap(coordinates, mask, maximum=8)

    np.testing.assert_array_equal(first, second)
    assert int(first.sum()) == 8
    retained = coordinates[first]
    quadrants = set(zip(retained[:, 0] >= 50.0, retained[:, 1] >= 50.0))
    assert quadrants == {(False, False), (False, True), (True, False), (True, True)}


def test_spatially_stratified_cap_preserves_an_uncapped_mask() -> None:
    coordinates = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    mask = np.asarray([True, False, True])

    result = _spatially_stratified_cap(coordinates, mask, maximum=2)

    np.testing.assert_array_equal(result, mask)
    assert result is not mask
