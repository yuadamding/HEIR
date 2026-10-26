"""H&E-only tissue regions in the authors' registered image-pixel frame.

The processed image is indexed as ``image[y, x]`` without a vertical flip: the
authors display these arrays with ``origin='lower'``. Pixels are grouped into
fixed, half-open grid bins starting at (0, 0), clipped at the image boundary.
These regions make no nucleus or same-cell correspondence claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from heir.features.handcrafted import HandcraftedPatchEncoder


@dataclass(frozen=True)
class RegionGrid:
    """Selected H&E regions; all coordinates and bin widths are registered pixels."""

    region_ids: tuple[str, ...]
    coordinates: np.ndarray
    image_features: np.ndarray
    bin_keys: np.ndarray
    tissue_fraction: np.ndarray
    component_ids: tuple[str, ...]
    settings: dict
    tissue_pixel_counts: np.ndarray | None = None


def _rgb_image(image):
    if isinstance(image, (str, Path)):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("reading H&E image paths requires Pillow") from exc
        with Image.open(image) as source:
            values = np.asarray(source.convert("RGB"))
    else:
        values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError("H&E image must be an H-by-W-by-3 uint8 RGB array")
    if min(values.shape[:2]) == 0:
        raise ValueError("H&E image must have nonempty dimensions")
    return values


def _components(bin_keys):
    """Deterministic four-neighbor components of selected bins, independent of assays."""
    lookup = {tuple(key): i for i, key in enumerate(bin_keys)}
    result = [None] * len(bin_keys)
    component = 0
    for start in range(len(bin_keys)):
        if result[start] is not None:
            continue
        name = f"he_component:{component}"
        component += 1
        result[start] = name
        pending = [start]
        while pending:
            index = pending.pop()
            x, y = bin_keys[index]
            for key in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                adjacent = lookup.get(key)
                if adjacent is not None and result[adjacent] is None:
                    result[adjacent] = name
                    pending.append(adjacent)
    return tuple(result)


def build_region_grid(
    image,
    *,
    bin_size_pixels=100,
    min_tissue_fraction=0.1,
    min_saturation=0.05,
    min_intensity=0.05,
    max_intensity=0.95,
    compute_features=True,
) -> RegionGrid:
    """Select bins from RGB tissue fraction and encode their actual image patches.

    A tissue pixel has HSV saturation ``(max(RGB)-min(RGB))/max(RGB)`` at
    least ``min_saturation`` and mean RGB intensity, divided by 255, within
    ``[min_intensity, max_intensity]``. Defaults exclude white background and
    nearly black pixels. A bin is selected when its tissue fraction is at least
    ``min_tissue_fraction``; the denominator is its actual clipped pixel area.

    Features are the existing 12-dimensional HandcraftedPatchEncoder negative
    control, using all pixels of each selected patch without padding. This is
    deterministic image processing, not a trained histology encoder. Components
    describe four-neighbor connectivity of selected bins, not tissue topology
    within individual bins. No protein or spatial-RNA input is accepted.
    """
    if (
        isinstance(bin_size_pixels, (bool, np.bool_))
        or not isinstance(bin_size_pixels, (int, np.integer))
        or bin_size_pixels <= 0
    ):
        raise ValueError("bin_size_pixels must be a positive integer")
    thresholds = np.asarray(
        [min_tissue_fraction, min_saturation, min_intensity, max_intensity], dtype=float
    )
    if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("tissue thresholds must be finite values in [0, 1]")
    if min_tissue_fraction <= 0 or min_intensity >= max_intensity:
        raise ValueError("tissue fraction must be positive and intensity bounds increasing")
    if not isinstance(compute_features, (bool, np.bool_)):
        raise ValueError("compute_features must be boolean")
    values = _rgb_image(image)
    height, width = values.shape[:2]
    encoder = HandcraftedPatchEncoder() if compute_features else None
    keys, coordinates, features, fractions, tissue_counts = [], [], [], [], []
    # Work one patch at a time: no full-image float raster or padding allocation.
    for y in range(0, height, bin_size_pixels):
        for x in range(0, width, bin_size_pixels):
            right, top = min(x + bin_size_pixels, width), min(y + bin_size_pixels, height)
            patch = values[y:top, x:right]
            maximum = patch.max(axis=2).astype(np.float32)
            saturation = (maximum - patch.min(axis=2)) / np.maximum(maximum, 1)
            intensity = patch.mean(axis=2) / 255.0
            tissue = (
                (saturation >= min_saturation)
                & (intensity >= min_intensity)
                & (intensity <= max_intensity)
            )
            fraction = float(tissue.mean())
            if fraction < min_tissue_fraction:
                continue
            keys.append((x // bin_size_pixels, y // bin_size_pixels))
            coordinates.append(((x + right) / 2, (y + top) / 2))
            if encoder is not None:
                features.append(encoder.encode(patch[None])[0])
            fractions.append(fraction)
            tissue_counts.append(int(tissue.sum()))
    if not keys:
        raise ValueError("H&E image contains no regions meeting the fixed tissue thresholds")
    keys = np.asarray(keys, dtype=np.int64)
    settings = {
        "schema": "heir.he_region_grid.v1",
        "prediction_unit": "tissue_region",
        "coordinate_units": "registered_pixel",
        "image_shape": [height, width],
        "bin_size_pixels": int(bin_size_pixels),
        "origin_pixels": [0, 0],
        "array_row_is_y": True,
        "display_origin": "lower",
        "edge_bins": "clip_to_image_no_padding",
        "min_tissue_fraction": float(min_tissue_fraction),
        "min_saturation": float(min_saturation),
        "min_intensity": float(min_intensity),
        "max_intensity": float(max_intensity),
        "component_connectivity": "four_neighbor_selected_bins",
        "encoder_name": "handcrafted_rgb_moments_negative_control"
        if encoder
        else "none_measurement_only",
        "encoder_manifest_sha256": encoder.manifest_sha256 if encoder else None,
        "contains_spatial_rna_derived_features": False,
        "contains_protein_derived_features": False,
    }
    return RegionGrid(
        region_ids=tuple(f"he_region:{x}:{y}" for x, y in keys),
        coordinates=np.asarray(coordinates, dtype=np.float32),
        image_features=np.asarray(features, dtype=np.float32)
        if encoder
        else np.empty((len(keys), 0), dtype=np.float32),
        bin_keys=keys,
        tissue_fraction=np.asarray(fractions, dtype=np.float32),
        component_ids=_components(keys),
        settings=settings,
        tissue_pixel_counts=np.asarray(tissue_counts, dtype=np.int64),
    )


def point_to_grid(grid: RegionGrid, coordinates) -> np.ndarray:
    """Map each assay point to an existing H&E region, or -1 outside selected coverage.

    Mapping does not modify the grid, select additional regions, or merge assay
    replicates. Nonfinite points are rejected rather than silently discarded.
    """
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("point coordinates must be finite N-by-2 values")
    height, width = grid.settings["image_shape"]
    size = grid.settings["bin_size_pixels"]
    inside = (
        (values[:, 0] >= 0) & (values[:, 1] >= 0) & (values[:, 0] < width) & (values[:, 1] < height)
    )
    result = np.full(len(values), -1, dtype=np.int64)
    lookup = {tuple(key): i for i, key in enumerate(grid.bin_keys)}
    for row in np.flatnonzero(inside):
        key = tuple(np.floor(values[row] / size).astype(np.int64))
        result[row] = lookup.get(key, -1)
    return result


@dataclass(frozen=True)
class NestedRegionGrid:
    """Explicit fine-to-parent membership; gaps never become inferred tissue."""

    region_ids: tuple[str, ...]
    coordinates: np.ndarray
    bin_keys: np.ndarray
    fine_to_parent: np.ndarray
    parent_fine_indices: tuple[tuple[int, ...], ...]
    parent_fine_region_ids: tuple[tuple[str, ...], ...]
    bin_area_pixels: np.ndarray
    support_area_pixels: np.ndarray
    tissue_area_pixels: np.ndarray
    scale_pixels: int
    base_region_ids: tuple[str, ...]


def fine_region_areas(grid):
    """Return clipped rectangular support and H&E tissue areas in image pixels."""
    size = grid.settings["bin_size_pixels"]
    extent = np.asarray(grid.settings["image_shape"][::-1])
    lower = grid.bin_keys * size
    area = np.prod(np.minimum(lower + size, extent) - lower, axis=1)
    tissue = (
        np.asarray(grid.tissue_pixel_counts).copy()
        if grid.tissue_pixel_counts is not None
        else area * grid.tissue_fraction
    )
    return area, tissue


def nested_grid_membership(grid, scale_pixels, *, fine_indices=None):
    """Nest selected base bins at origin zero without redefining the footprint.

    A coarse rectangle can contain excluded/unselected gaps. ``support_area_pixels``
    counts only its declared fine-bin union; tissue area sums H&E tissue pixels
    within that union. Empty support produces an explicitly empty parent axis.
    """
    base = grid.settings["bin_size_pixels"]
    if (
        not isinstance(scale_pixels, (int, np.integer))
        or isinstance(scale_pixels, (bool, np.bool_))
        or scale_pixels < base
        or scale_pixels % base
    ):
        raise ValueError("nested scale must be an integer multiple of the base grid size")
    if grid.settings["origin_pixels"] != [0, 0]:
        raise ValueError("nested measurement grids require origin zero")
    selected = np.arange(len(grid.region_ids)) if fine_indices is None else np.asarray(fine_indices)
    if selected.size == 0:
        selected = selected.astype(np.int64)
    if (
        selected.ndim != 1
        or selected.dtype.kind not in "iu"
        or len(set(selected.tolist())) != len(selected)
        or np.any(selected < 0)
        or np.any(selected >= len(grid.region_ids))
    ):
        raise ValueError("fine support requires unique valid integer region indices")
    selected = np.sort(selected)
    parent_keys = grid.bin_keys[selected] // (scale_pixels // base)
    keys = sorted(set(map(tuple, parent_keys)), key=lambda key: (key[1], key[0]))
    key_array = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
    lookup = {key: i for i, key in enumerate(keys)}
    mapping = np.full(len(grid.region_ids), -1, dtype=np.int64)
    for index, key in zip(selected, parent_keys):
        mapping[index] = lookup[tuple(key)]
    members = tuple(tuple(np.flatnonzero(mapping == i).tolist()) for i in range(len(keys)))
    extent = np.asarray(grid.settings["image_shape"][::-1])
    lower = key_array * scale_pixels
    upper = np.minimum(lower + scale_pixels, extent)
    area, tissue = fine_region_areas(grid)
    return NestedRegionGrid(
        region_ids=tuple(
            f"he_region:{x}:{y}" if scale_pixels == base else f"he_region_{scale_pixels}:{x}:{y}"
            for x, y in keys
        ),
        coordinates=((lower + upper) / 2).astype(np.float32),
        bin_keys=key_array,
        fine_to_parent=mapping,
        parent_fine_indices=members,
        parent_fine_region_ids=tuple(tuple(grid.region_ids[i] for i in group) for group in members),
        bin_area_pixels=np.prod(upper - lower, axis=1),
        support_area_pixels=np.asarray([area[list(group)].sum() for group in members]),
        tissue_area_pixels=np.asarray([tissue[list(group)].sum() for group in members]),
        scale_pixels=int(scale_pixels),
        base_region_ids=grid.region_ids,
    )
