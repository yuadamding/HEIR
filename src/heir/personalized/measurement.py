"""Raw measurement substrate and nested, geometry-defined regional aggregation.

Counts are summed before normalization. Common support means the same union of
bead-occupied base H&E bins, not verified identical tissue within each bin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regions import (
    NestedRegionGrid,
    RegionGrid,
    fine_region_areas,
    nested_grid_membership,
    point_to_grid,
)
from .schema import identifiers


@dataclass(frozen=True)
class MeasurementSubstrate:
    biopsy_id: str
    patient_id: str
    specimen_id: str
    grid: RegionGrid
    gene_ids: tuple[str, ...]
    independent_rna_gene_ids: tuple[str, ...]
    bead_ids: tuple[str, ...]
    coordinates: np.ndarray
    replicates: tuple[str, ...]
    counts: object  # scipy CSR, with every retained measured gene column
    base_region_indices: np.ndarray
    provenance: dict

    def __post_init__(self):
        from scipy.sparse import isspmatrix_csr

        from .reference import _validate_counts

        identifiers(self.bead_ids, "measurement bead IDs")
        identifiers(self.gene_ids, "measured spatial gene IDs")
        identifiers(self.independent_rna_gene_ids, "measured independent RNA gene IDs")
        n = len(self.bead_ids)
        if (
            not isspmatrix_csr(self.counts)
            or self.counts.shape != (n, len(self.gene_ids))
            or len(self.replicates) != n
            or any(not str(r).strip() for r in self.replicates)
        ):
            raise ValueError(
                "measurement counts and replicate labels must match the named bead axis"
            )
        _validate_counts(self.counts)
        mapped = point_to_grid(self.grid, self.coordinates)
        if mapped.shape != (n,) or not np.array_equal(mapped, self.base_region_indices):
            raise ValueError("base bead assignments differ from the declared H&E geometry")


@dataclass(frozen=True)
class RegionalMeasurements:
    grid: NestedRegionGrid
    gene_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    observation_replicates: tuple[str, ...]
    region_indices: np.ndarray
    counts: object  # CSR: only observed region-replicate rows; absent rows are not zeros
    bead_counts: np.ndarray
    occupied_fine_indices: tuple[tuple[int, ...], ...]
    occupied_area_pixels: np.ndarray
    tissue_area_pixels: np.ndarray
    bead_base_region_indices: np.ndarray
    bead_parent_indices: np.ndarray
    bead_observation_indices: np.ndarray
    conservation: dict

    @property
    def region_ids(self):
        return self.grid.region_ids

    @property
    def coordinates(self):
        return self.grid.coordinates


def _replicates(substrate, replicate_ids):
    actual = tuple(sorted(set(substrate.replicates)))
    selected = actual if replicate_ids is None else tuple(map(str, replicate_ids))
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(actual):
        raise ValueError("requested replicates must be unique measured replicate IDs")
    return tuple(sorted(selected))


def mapped_base_regions(substrate, displacement_by_replicate=None):
    """Remap original bead positions after fixed registered-pixel translations.

    The H&E grid stays fixed. Beads can leave or enter selected bins after a
    shift; no expression value is inspected to establish their geometry.
    """
    shifts = displacement_by_replicate or {}
    if not set(shifts) <= set(substrate.replicates):
        raise ValueError("displacement names an unmeasured replicate")
    if not shifts:
        return substrate.base_region_indices.copy()
    xy = np.array(substrate.coordinates, dtype=float, copy=True)
    replicates = np.asarray(substrate.replicates)
    for replicate, shift in shifts.items():
        delta = np.asarray(shift, dtype=float)
        if delta.shape != (2,) or not np.isfinite(delta).all():
            raise ValueError("displacement must contain finite x/y registered pixels")
        xy[replicates == replicate] += delta
    return point_to_grid(substrate.grid, xy)


def common_geometry_support(substrate, replicate_ids, *, displacement_by_replicate=None):
    """Return fine-bin indices occupied by at least one bead in every replicate."""
    selected = _replicates(substrate, replicate_ids)
    mapping = mapped_base_regions(substrate, displacement_by_replicate)
    labels = np.asarray(substrate.replicates)
    support = set(range(len(substrate.grid.region_ids)))
    for replicate in selected:
        support &= set(mapping[(labels == replicate) & (mapping >= 0)].tolist())
    return np.asarray(sorted(support), dtype=np.int64)


def aggregate_measurements(
    substrate,
    *,
    scale_pixels=100,
    replicate_ids=None,
    support_fine_indices=None,
    displacement_by_replicate=None,
):
    """Aggregate raw beads to fine bins and then parents; verify direct equivalence.

    The full parent axis describes the declared fine-bin support even where an
    assay has no beads. Sparse observation rows describe only measured parents;
    all-zero measured parents are retained. Every original bead's base, parent
    and observation assignments are returned (outside support is -1).
    """
    from scipy.sparse import csr_matrix

    selected = _replicates(substrate, replicate_ids)
    parent = nested_grid_membership(substrate.grid, scale_pixels, fine_indices=support_fine_indices)
    base = mapped_base_regions(substrate, displacement_by_replicate)
    labels = np.asarray(substrate.replicates)
    parents = np.full(len(base), -1, dtype=np.int64)
    inside = (base >= 0) & np.isin(labels, selected)
    parents[inside] = parent.fine_to_parent[base[inside]]
    included = np.flatnonzero(parents >= 0)
    fine_keys = sorted({(str(labels[i]), int(base[i])) for i in included})
    coarse_keys = sorted({(str(labels[i]), int(parents[i])) for i in included})
    fine_lookup = {key: i for i, key in enumerate(fine_keys)}
    coarse_lookup = {key: i for i, key in enumerate(coarse_keys)}
    fine_rows = np.asarray(
        [fine_lookup[(str(labels[i]), int(base[i]))] for i in included], dtype=np.int64
    )
    coarse_rows = np.asarray(
        [coarse_lookup[(str(labels[i]), int(parents[i]))] for i in included], dtype=np.int64
    )
    fine_pool = csr_matrix(
        (np.ones(len(included), dtype=np.int64), (fine_rows, included)),
        shape=(len(fine_keys), len(base)),
    )
    fine_counts = fine_pool @ substrate.counts
    to_coarse = np.asarray(
        [coarse_lookup[(rep, int(parent.fine_to_parent[index]))] for rep, index in fine_keys],
        dtype=np.int64,
    )
    coarse_pool = csr_matrix(
        (np.ones(len(fine_keys), dtype=np.int64), (to_coarse, np.arange(len(fine_keys)))),
        shape=(len(coarse_keys), len(fine_keys)),
    )
    counts = (coarse_pool @ fine_counts).tocsr()
    direct_pool = csr_matrix(
        (np.ones(len(included), dtype=np.int64), (coarse_rows, included)),
        shape=(len(coarse_keys), len(base)),
    )
    direct = direct_pool @ substrate.counts
    if (counts - direct).nnz or (counts.data < 0).any():
        raise ValueError(
            "nested measurement aggregation differs from raw beads or count overflowed"
        )
    assigned = np.full(len(base), -1, dtype=np.int64)
    assigned[included] = coarse_rows
    bead_counts = np.bincount(coarse_rows, minlength=len(coarse_keys))
    occupied = tuple(
        tuple(index for j, (_, index) in enumerate(fine_keys) if to_coarse[j] == row)
        for row in range(len(coarse_keys))
    )
    fine_area, fine_tissue = fine_region_areas(substrate.grid)
    checks = {}
    for rep in selected:
        source = included[labels[included] == rep]
        rows = [i for i, (label, _) in enumerate(coarse_keys) if label == rep]
        expected, actual = int(substrate.counts[source].sum()), int(counts[rows].sum())
        if expected != actual:
            raise ValueError("measurement counts are not conserved within replicate")
        checks[rep] = {
            "n_source_beads": int((labels == rep).sum()),
            "n_included_beads": len(source),
            "included_counts_total": expected,
            "aggregated_counts_total": actual,
        }
    return RegionalMeasurements(
        parent,
        substrate.gene_ids,
        tuple(f"slide_seq:{rep}:{parent.region_ids[index]}" for rep, index in coarse_keys),
        tuple(rep for rep, _ in coarse_keys),
        np.asarray([index for _, index in coarse_keys], dtype=np.int64),
        counts,
        bead_counts,
        occupied,
        np.asarray([fine_area[list(group)].sum() for group in occupied]),
        np.asarray([fine_tissue[list(group)].sum() for group in occupied]),
        base,
        parents,
        assigned,
        {
            "direct_bead_aggregation_equal": True,
            "by_replicate": checks,
            "aggregation": "raw_beads_to_fine_then_parent_before_normalization",
            "support_definition": (
                "union_of_declared_bead_occupied_base_bins; intrabin_overlap_unknown"
            ),
        },
    )


def measurement_values(measurements, *, outcome_gene_ids, normalization_gene_ids):
    """Select outcomes while retaining a separately declared normalization panel."""
    outcomes, normal = tuple(outcome_gene_ids), tuple(normalization_gene_ids)
    identifiers(outcomes, "measurement outcome genes")
    identifiers(normal, "measurement normalization panel")
    if len(normal) < 2:
        raise ValueError("measurement normalization requires at least two fixed genes")
    if not (set(outcomes) | set(normal)) <= set(measurements.gene_ids):
        raise ValueError(
            "requested genes are absent from the measured gene axis; missing is not zero"
        )
    lookup = {gene: i for i, gene in enumerate(measurements.gene_ids)}
    values = measurements.counts[:, [lookup[g] for g in outcomes]].toarray()
    exposure = np.asarray(measurements.counts[:, [lookup[g] for g in normal]].sum(1)).ravel()
    total = np.asarray(measurements.counts.sum(1)).ravel()
    fraction = np.divide(exposure, total, out=np.full(total.shape, np.nan), where=total > 0)
    return {
        "outcome_gene_ids": outcomes,
        "normalization_gene_ids": normal,
        "outcome_counts": values,
        "normalization_exposure": exposure,
        "total_exposure": total,
        "panel_fraction": fraction,
        "eligible": exposure > 0,
        "observed": np.ones(values.shape, dtype=bool),
    }
