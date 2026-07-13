#!/usr/bin/env python3
"""Create deterministic, disjoint spatial-block HEIR train/validation bags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from heir.data import HistologyBag
from heir.training import spatial_block_split_masks, subset_histology_bag
from heir.utils import atomic_json_dump


def _spatially_stratified_cap(
    coordinates_um: np.ndarray,
    mask: np.ndarray,
    maximum: int,
) -> np.ndarray:
    """Cap a partition while retaining spatial coverage and local coherence.

    Cells are allocated proportionally across a prespecified quantile grid.  Within
    each occupied stratum, the cells nearest that stratum's median are retained.
    This preserves local graph neighborhoods without collapsing every capped bag
    onto the specimen-wide median coordinate.
    """

    indices = np.flatnonzero(mask)
    if len(indices) <= maximum:
        return np.array(mask, copy=True)
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape != (len(mask), 2):
        raise ValueError("coordinates_um must have cells-by-two shape")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates_um must be finite")

    # At most 8x8 strata keeps each retained region large enough to preserve
    # useful local edges, while quantile boundaries avoid empty density tails.
    bins_per_axis = min(8, max(2, int(np.sqrt(maximum / 128.0))))
    selected_coordinates = coordinates[indices]
    quantiles = np.linspace(0.0, 1.0, bins_per_axis + 1)[1:-1]
    x_boundaries = np.unique(np.quantile(selected_coordinates[:, 0], quantiles))
    y_boundaries = np.unique(np.quantile(selected_coordinates[:, 1], quantiles))
    x_bins = np.searchsorted(x_boundaries, selected_coordinates[:, 0], side="right")
    y_bins = np.searchsorted(y_boundaries, selected_coordinates[:, 1], side="right")
    stratum_ids = x_bins * (len(y_boundaries) + 1) + y_bins
    occupied, counts = np.unique(stratum_ids, return_counts=True)

    exact_allocation = maximum * counts.astype(np.float64) / len(indices)
    allocation = np.floor(exact_allocation).astype(np.int64)
    allocation = np.minimum(allocation, counts)
    remainder = maximum - int(allocation.sum())
    fractional = exact_allocation - allocation
    order = np.lexsort((occupied, -counts, -fractional))
    while remainder > 0:
        progressed = False
        for position in order:
            if allocation[position] < counts[position]:
                allocation[position] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
        if not progressed:
            raise RuntimeError("unable to allocate the spatially stratified cap")

    capped = np.zeros_like(mask, dtype=np.bool_)
    for stratum, retain in zip(occupied, allocation):
        local = np.flatnonzero(stratum_ids == stratum)
        local_indices = indices[local]
        center = np.median(coordinates[local_indices], axis=0)
        squared_distance = np.square(coordinates[local_indices] - center).sum(axis=1)
        local_order = np.lexsort((local_indices, squared_distance))
        capped[local_indices[local_order[: int(retain)]]] = True
    if int(capped.sum()) != maximum:
        raise RuntimeError("spatially stratified cap selected an unexpected number of cells")
    return capped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--block-size-um", type=float, default=512.0)
    parser.add_argument("--maximum-train-cells", type=int, default=15000)
    parser.add_argument("--maximum-validation-cells", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    bag = HistologyBag.load_npz(args.input.expanduser().resolve())
    training_mask, validation_mask = spatial_block_split_masks(
        bag.coordinates_um,
        validation_fraction=args.validation_fraction,
        block_size_um=args.block_size_um,
        seed=args.seed,
    )
    for name, maximum in (
        ("maximum-train-cells", args.maximum_train_cells),
        ("maximum-validation-cells", args.maximum_validation_cells),
    ):
        if maximum <= 0:
            raise ValueError("%s must be positive" % name)

    training_mask = _spatially_stratified_cap(
        bag.coordinates_um, training_mask, args.maximum_train_cells
    )
    validation_mask = _spatially_stratified_cap(
        bag.coordinates_um, validation_mask, args.maximum_validation_cells
    )
    train_bag = subset_histology_bag(bag, training_mask)
    validation_bag = subset_histology_bag(bag, validation_mask)
    train_bag.save_npz(args.train_output.expanduser().resolve())
    validation_bag.save_npz(args.validation_output.expanduser().resolve())
    report = {
        "input": str(args.input.expanduser().resolve()),
        "train_output": str(args.train_output.expanduser().resolve()),
        "validation_output": str(args.validation_output.expanduser().resolve()),
        "total_nuclei": bag.n_nuclei,
        "train_nuclei": train_bag.n_nuclei,
        "validation_nuclei": validation_bag.n_nuclei,
        "train_edges": int(train_bag.edge_index.shape[1]),
        "validation_edges": int(validation_bag.edge_index.shape[1]),
        "validation_fraction": args.validation_fraction,
        "block_size_um": args.block_size_um,
        "maximum_train_cells": args.maximum_train_cells,
        "maximum_validation_cells": args.maximum_validation_cells,
        "cap_strategy": "proportional_quantile_grid_local_median_v1",
        "seed": args.seed,
        "nucleus_overlap": int(
            len(set(train_bag.nucleus_ids.tolist()) & set(validation_bag.nucleus_ids.tolist()))
        ),
    }
    atomic_json_dump(report, args.summary.expanduser().resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
