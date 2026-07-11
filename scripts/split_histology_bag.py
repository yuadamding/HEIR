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

    def coherent_cap(mask: np.ndarray, maximum: int) -> np.ndarray:
        indices = np.flatnonzero(mask)
        if len(indices) <= maximum:
            return mask
        center = np.median(bag.coordinates_um[indices], axis=0)
        distance = np.square(bag.coordinates_um[indices] - center).sum(axis=1)
        order = np.lexsort((indices, distance))[:maximum]
        capped = np.zeros_like(mask)
        capped[indices[order]] = True
        return capped

    training_mask = coherent_cap(training_mask, args.maximum_train_cells)
    validation_mask = coherent_cap(validation_mask, args.maximum_validation_cells)
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
        "seed": args.seed,
        "nucleus_overlap": int(
            len(set(train_bag.nucleus_ids.tolist()) & set(validation_bag.nucleus_ids.tolist()))
        ),
    }
    atomic_json_dump(report, args.summary.expanduser().resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
