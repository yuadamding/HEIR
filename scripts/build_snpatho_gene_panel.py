#!/usr/bin/env python3
"""Build the frozen 500-gene snPATHO panel without reading target expression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from heir.data.gene_panel import build_snpatho_panel

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_AVAILABILITY = tuple(
    REPOSITORY / "artifacts" / "snpatho" / case / "reference.h5ad"
    for case in ("4066", "4399", "4411")
)
DEFAULT_EVALUATION = tuple(
    REPOSITORY / "artifacts" / "snpatho" / case / "visium_truth.h5ad"
    for case in ("4066", "4399", "4411")
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY / "manifests" / "natcommun.tsv",
        help="Audited manifest containing the NatCommun B1_4 development record.",
    )
    parser.add_argument(
        "--curated",
        type=Path,
        default=REPOSITORY / "manifests" / "gene_panel_example.tsv",
        help="The frozen 70-gene curated seed list; this file is never modified.",
    )
    parser.add_argument(
        "--availability-h5ad",
        type=Path,
        action="append",
        default=None,
        help=(
            "snPATHO reference H5AD used only for var metadata. Repeat for multiple files; "
            "defaults to all three converted references."
        ),
    )
    parser.add_argument(
        "--availability-10x",
        type=Path,
        action="append",
        default=None,
        help="Optional filtered 10x HDF5 used only for matrix/features/name metadata.",
    )
    parser.add_argument(
        "--visium-gene-h5ad",
        type=Path,
        action="append",
        default=None,
        help=(
            "QC Visium derivative used only for var/feature_name. Repeat exactly three times; "
            "defaults to the 4066, 4399, and 4411 derivatives."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "manifests" / "gene_panel_snpatho_500.tsv",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=REPOSITORY / "manifests" / "gene_panel_snpatho_500.provenance.json",
    )
    parser.add_argument("--panel-size", type=int, default=500)
    parser.add_argument("--markers-per-type", type=int, default=40)
    parser.add_argument("--minimum-detection", type=float, default=0.01)
    parser.add_argument("--minimum-type-detection", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    availability_h5ads = (
        tuple(args.availability_h5ad)
        if args.availability_h5ad is not None
        else DEFAULT_AVAILABILITY
    )
    evaluation_h5ads = (
        tuple(args.visium_gene_h5ad) if args.visium_gene_h5ad is not None else DEFAULT_EVALUATION
    )
    payload = build_snpatho_panel(
        manifest_path=args.manifest,
        curated_path=args.curated,
        availability_h5ads=availability_h5ads,
        evaluation_h5ads=evaluation_h5ads,
        availability_10x=tuple(args.availability_10x or ()),
        output_path=args.output,
        provenance_path=args.provenance,
        panel_size=args.panel_size,
        markers_per_type=args.markers_per_type,
        minimum_detection=args.minimum_detection,
        minimum_type_detection=args.minimum_type_detection,
        chunk_size=args.chunk_size,
    )
    summary = {
        "panel": payload["panel"],
        "counts": payload["counts"],
        "balanced_marker_counts": payload["balanced_marker_counts"],
        "provenance": str(args.provenance.expanduser().resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
