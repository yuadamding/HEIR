"""Physical-coordinate transforms and shared protein observation regions.

Serial sections are never assigned by nearest-cell identity. Supplied affine
registration is reused; registration estimation is deliberately outside this adapter.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np


def coordinates_um(coordinates, *, units: str, micrometres_per_pixel=None):
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("coordinates must be finite N-by-2 values")
    if units == "um":
        return values
    if units != "pixel":
        raise ValueError("coordinate units must be um or pixel")
    scale = np.asarray(micrometres_per_pixel, dtype=float)
    if scale.size not in (1, 2) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("pixel coordinates require positive physical pixel scales")
    return values * scale


def prepare_geometry(coordinates, registration: dict, *, source_frame="protein"):
    """Map source physical coordinates into the H&E micrometre frame.

    Matrix convention is column-vector source-to-target: ``target = M @
    [x_um, y_um, 1]``. Pixel conversion precedes the physical affine transform.
    """
    if registration.get("source_frame") != source_frame:
        raise ValueError("registration source_frame differs from requested coordinates")
    if registration.get("target_frame") != "he" or registration.get("target_units") != "um":
        raise ValueError("registration must target the H&E micrometre frame")
    if registration.get("direction") != "source_to_target":
        raise ValueError("registration direction must explicitly be source_to_target")
    matrix = np.asarray(registration.get("affine"), dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("registration affine must be a finite 3-by-3 matrix")
    if not np.allclose(matrix[2], [0, 0, 1]) or abs(np.linalg.det(matrix[:2, :2])) < 1e-12:
        raise ValueError("registration must be an invertible affine transform")
    uncertainty = registration.get("uncertainty_um")
    if uncertainty is None or not np.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError("registration requires nonnegative uncertainty_um")
    physical = coordinates_um(
        coordinates,
        units=registration.get("source_units"),
        micrometres_per_pixel=registration.get("micrometres_per_pixel"),
    )
    homogeneous = np.column_stack((physical, np.ones(len(physical))))
    return (homogeneous @ matrix.T)[:, :2]


def region_bins(coordinates, *, bin_size_um=100.0, origin_um=(0.0, 0.0)):
    """Return reproducible global bin IDs, never an inferred cell correspondence."""
    coordinates = coordinates_um(coordinates, units="um")
    if not np.isfinite(bin_size_um) or bin_size_um <= 0:
        raise ValueError("bin_size_um must be positive")
    origin = np.asarray(origin_um, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("bin origin must be a finite physical coordinate")
    indices = np.floor((coordinates - origin) / bin_size_um).astype(np.int64)
    return tuple(f"bin:{i}:{j}" for i, j in indices)


def tissue_mask_edge_filter(coordinates, mask, *, origin_um, pixel_size_um):
    """Return an edge predicate that excludes segments crossing a tissue-mask gap.

    The Boolean raster is indexed ``mask[y, x]``. Every interval between pixel
    boundary crossings is checked, including endpoints; a coarse sampling stride
    cannot jump over a thin gap. Outside-mask space is unsupported.
    """
    coordinates = coordinates_um(coordinates, units="um")
    mask = np.asarray(mask)
    origin, scale = np.asarray(origin_um, dtype=float), np.asarray(pixel_size_um, dtype=float)
    if mask.ndim != 2 or min(mask.shape) == 0 or mask.dtype != bool:
        raise ValueError("tissue mask must be a nonempty two-dimensional Boolean raster")
    if origin.shape != (2,) or scale.shape != (2,):
        raise ValueError("mask origin_um and pixel_size_um require x/y pairs")
    if not np.isfinite(origin).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("mask coordinates require finite origin and positive physical pixel size")
    pixels = (coordinates - origin) / scale
    cache = {}

    def allowed(sender, receiver):
        key = tuple(sorted((int(sender), int(receiver))))
        if key in cache:
            return cache[key]
        start, end = pixels[sender], pixels[receiver]
        delta = end - start
        crossings = [0.0, 1.0]
        for axis in range(2):
            if delta[axis] != 0:
                lower, upper = sorted((start[axis], end[axis]))
                boundaries = np.arange(np.floor(lower) + 1, np.ceil(upper))
                crossings.extend(((boundaries - start[axis]) / delta[axis]).tolist())
        crossings = np.unique(crossings)
        locations = np.concatenate(([0.0, 1.0], (crossings[:-1] + crossings[1:]) / 2))
        visited = np.floor(start + locations[:, None] * delta).astype(np.int64)
        valid = (
            (visited[:, 0] >= 0)
            & (visited[:, 0] < mask.shape[1])
            & (visited[:, 1] >= 0)
            & (visited[:, 1] < mask.shape[0])
        )
        result = bool(valid.all() and mask[visited[:, 1], visited[:, 0]].all())
        cache[key] = result
        return result

    return allowed


def mask_component_ids(coordinates, component_ids, mask, *, origin_um, pixel_size_um):
    """Split declared components by connected mask islands; reject unsupported points."""
    from scipy.ndimage import label

    coordinates = coordinates_um(coordinates, units="um")
    # Reuse the raster contract, including outside-mask rejection at each point.
    supported = tissue_mask_edge_filter(
        coordinates, mask, origin_um=origin_um, pixel_size_um=pixel_size_um
    )
    labels, _ = label(mask)
    pixels = np.floor((coordinates - np.asarray(origin_um)) / np.asarray(pixel_size_um)).astype(
        np.int64
    )
    if len(component_ids) != len(coordinates):
        raise ValueError("tissue component axes differ")
    if any(not supported(i, i) for i in range(len(coordinates))):
        raise ValueError("protein/H&E coordinates outside the declared tissue mask")
    return tuple(
        (str(component), int(labels[y, x])) for component, (x, y) in zip(component_ids, pixels)
    )


def regional_protein_evidence(
    cell_ids,
    he_coordinates_um,
    protein_coordinates_um,
    values,
    observed,
    *,
    bin_size_um=100.0,
    origin_um=(0.0, 0.0),
    quality=None,
    he_component_ids,
    protein_component_ids,
):
    """Build a shared, component-specific geometry operator and uniform protein means.

    Both assays use the same fixed physical bins, independently of the model's
    communication graph. A region-marker is observed only if every source
    measurement in that region has positive-confidence evidence. Missingness
    therefore never changes a marker's geometric averaging kernel. Confidence
    is averaged separately from intensity and remains an absolute loss weight.
    """
    import json

    from .schema import ProteinObservationOperator

    he = coordinates_um(he_coordinates_um, units="um")
    protein = coordinates_um(protein_coordinates_um, units="um")
    values, observed = np.asarray(values, dtype=float), np.asarray(observed)
    if values.ndim != 2 or values.shape[0] != len(protein) or observed.shape != values.shape:
        raise ValueError("protein coordinate, value, and mask axes differ")
    if observed.dtype != bool:
        raise ValueError("protein observed mask must be boolean")
    if not np.isfinite(values[observed]).all() or (values[observed] < 0).any():
        raise ValueError("observed protein intensities must be finite and nonnegative")
    quality = np.ones(values.shape) if quality is None else np.asarray(quality, dtype=float)
    if quality.shape != values.shape or not np.isfinite(quality[observed]).all():
        raise ValueError("protein quality must be finite at observed measurements")
    if ((quality[observed] < 0) | (quality[observed] > 1)).any():
        raise ValueError("protein quality must lie in [0, 1]")
    if len(cell_ids) != len(he):
        raise ValueError("H&E cell ID and coordinate axes differ")
    if len(he_component_ids) != len(he) or len(protein_component_ids) != len(protein):
        raise ValueError("tissue component axes differ")

    def groups(points, components):
        result = {}
        bins = region_bins(points, bin_size_um=bin_size_um, origin_um=origin_um)
        for i, (component, region) in enumerate(zip(components, bins)):
            # JSON encoding avoids collisions between arbitrary component names.
            key = json.dumps([component, region], separators=(",", ":"))
            result.setdefault(key, []).append(i)
        return result

    he_groups, protein_groups = groups(he, he_component_ids), groups(protein, protein_component_ids)
    regions = tuple(sorted(he_groups.keys() & protein_groups.keys()))
    if not regions:
        raise ValueError("protein and H&E have no shared tissue-region coverage")
    result = np.zeros((len(regions), values.shape[1]), dtype=np.float32)
    mask = np.zeros(result.shape, dtype=bool)
    confidence = np.zeros(result.shape, dtype=np.float32)
    rows, columns, weights = [], [], []
    for row, region in enumerate(regions):
        cells, sources = he_groups[region], protein_groups[region]
        rows.extend([row] * len(cells))
        columns.extend(cells)
        weights.extend([1.0 / len(cells)] * len(cells))
        present = (observed[sources] & (quality[sources] > 0)).all(axis=0)
        mask[row] = present
        result[row, present] = values[np.ix_(sources, present)].mean(axis=0)
        confidence[row, present] = quality[np.ix_(sources, present)].mean(axis=0)
    operator = ProteinObservationOperator(
        cell_ids=tuple(cell_ids),
        observation_ids=regions,
        rows=np.asarray(rows, dtype=np.int64),
        columns=np.asarray(columns, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float32),
    )
    coverage = {
        "shared_regions": len(regions),
        "excluded_he_regions": len(he_groups.keys() - protein_groups.keys()),
        "excluded_protein_regions": len(protein_groups.keys() - he_groups.keys()),
        "unobserved_region_markers": int((~mask).sum()),
    }
    return result, mask, confidence, operator, coverage


def write_overlay(
    path,
    he_coordinates_um,
    *,
    protein_coordinates_um=None,
    spatial_coordinates_um=None,
    bin_size_um=100.0,
    he_image_href=None,
    image_bounds_um=None,
    title="H&E nuclei, protein and spatial RNA coverage",
):
    """Write an inspectable SVG in physical units, with an optional H&E raster.

    Spatial RNA coordinates are visualization-only arguments; prediction never
    invokes this function with targets. Without a raster this is a geometry
    plot, not evidence that image registration has been visually qualified.
    """
    he = coordinates_um(he_coordinates_um, units="um")
    layers = [(he, "#333333", "H&E nuclei")]
    for values, color, label in (
        (protein_coordinates_um, "#1976d2", "Protein measurements"),
        (spatial_coordinates_um, "#d84315", "Spatial RNA observations"),
    ):
        if values is not None:
            layers.append((coordinates_um(values, units="um"), color, label))
    nonempty = [values for values, _, _ in layers if len(values)]
    if not nonempty:
        raise ValueError("overlay requires coordinates")
    if not np.isfinite(bin_size_um) or bin_size_um <= 0:
        raise ValueError("bin_size_um must be positive")
    points = np.concatenate(nonempty)
    lower = np.floor(points.min(0) / bin_size_um) * bin_size_um
    upper = (np.floor(points.max(0) / bin_size_um) + 1) * bin_size_um
    width, height = upper - lower
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{lower[0]} {lower[1]} '
        f'{width} {height}" width="1000" role="img">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(', '.join(label for _, _, label in layers))}; "
        f"grid {bin_size_um:g} micrometres. Registration requires visual review.</desc>",
        f'<rect x="{lower[0]}" y="{lower[1]}" width="{width}" height="{height}" fill="white"/>',
    ]
    if he_image_href is not None:
        bounds = np.asarray(image_bounds_um, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or (bounds[2:] <= 0).any():
            raise ValueError("H&E raster requires physical x, y, width, height bounds")
        parts.append(
            f'<image href="{html.escape(str(he_image_href), quote=True)}" x="{bounds[0]}" '
            f'y="{bounds[1]}" width="{bounds[2]}" height="{bounds[3]}" opacity="0.6"/>'
        )
    for x in np.arange(lower[0], upper[0] + bin_size_um / 2, bin_size_um):
        parts.append(f'<path d="M {x} {lower[1]} V {upper[1]}" stroke="#bbb" stroke-width="0.5"/>')
    for y in np.arange(lower[1], upper[1] + bin_size_um / 2, bin_size_um):
        parts.append(f'<path d="M {lower[0]} {y} H {upper[0]}" stroke="#bbb" stroke-width="0.5"/>')
    for values, color, label in layers:
        parts.append(f'<g fill="{color}" opacity="0.65"><title>{label}</title>')
        parts.extend(f'<circle cx="{x}" cy="{y}" r="2"/>' for x, y in values)
        parts.append("</g>")
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")
