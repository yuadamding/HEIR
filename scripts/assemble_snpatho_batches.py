#!/usr/bin/env python3
"""Atomically invoke HEIR assembly for disjoint snPATHO train/validation bags."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--donor-id", required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--analysis-role", required=True)
    parser.add_argument("--train-histology", required=True)
    parser.add_argument("--validation-histology", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--ood-artifact", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--artifact-threshold", type=float, default=0.5)
    args = parser.parse_args()

    outputs = tuple(
        Path(value).expanduser().resolve() for value in (args.train_output, args.validation_output)
    )
    if outputs[0] == outputs[1]:
        raise ValueError("train and validation outputs must differ")
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            "batch output already exists; orchestration validates before skipping"
        )
    common = (
        "--prototypes",
        args.prototypes,
        "--reference",
        args.reference,
        "--sample-id",
        args.sample_id,
        "--donor-id",
        args.donor_id,
        "--block-id",
        args.block_id,
        "--analysis-role",
        args.analysis_role,
        "--ood-artifact",
        args.ood_artifact,
        "--artifact-threshold",
        str(args.artifact_threshold),
    )
    for role, histology, output in (
        ("train", args.train_histology, args.train_output),
        ("validation", args.validation_histology, args.validation_output),
    ):
        command = (
            sys.executable,
            "-m",
            "heir",
            "assemble-batch",
            "--histology",
            histology,
            *common,
            "--bag-id",
            "%s_%s" % (args.sample_id, role),
            "--output",
            output,
        )
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError("%s batch assembly failed" % role)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
