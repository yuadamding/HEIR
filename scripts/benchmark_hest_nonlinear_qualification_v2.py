#!/usr/bin/env python3
"""Run outcome-locked HEST nonlinear qualification v2 phases."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from heir.evaluation.nonlinear_qualification_v2_execution import (
    PRODUCTION_TRIALS,
    atomic_json,
    calibrate,
    preflight,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/mnt/seagate/HEIR_runs/hest_hoptimus1_qualification/source.npz")
DEFAULT_BLANK = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/blank_patch.npz")
DEFAULT_BLANK_RECEIPT = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/blank_patch_receipt.json")
DEFAULT_TOPOLOGY = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/registration_topology.json")
DEFAULT_PROTOCOL = ROOT / "configs/hest_nonlinear_qualification_v2.json"
DEFAULT_MANIFEST = ROOT / "manifests/studies/hest_nonlinear_qualification_v2.retrospective.json"
DEFAULT_PREFLIGHT = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/preflight.json")
DEFAULT_CALIBRATION = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/calibration.json")
DEFAULT_CHECKPOINT = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/calibration.checkpoint.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "calibrate", "full"), required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--blank-supplement", type=Path, default=DEFAULT_BLANK)
    parser.add_argument("--blank-receipt", type=Path, default=DEFAULT_BLANK_RECEIPT)
    parser.add_argument("--registration-topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight-output", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--calibration-output", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--calibration-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trials-per-condition", type=int)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    receipt = dict(
        preflight(
            source=args.source,
            blank_supplement=args.blank_supplement,
            blank_receipt=args.blank_receipt,
            registration_topology=args.registration_topology,
            protocol_path=args.protocol,
            manifest_path=args.manifest,
            repository_root=ROOT,
        )
    )
    atomic_json(args.preflight_output, receipt)
    if args.phase == "preflight":
        return receipt
    if args.phase == "full":
        raise RuntimeError(
            "full phase remains locked until the observed B0-N7 runner and a passing "
            "production calibration receipt are present"
        )
    trials = args.trials_per_condition
    if trials is None:
        trials = 1 if args.smoke else PRODUCTION_TRIALS
    calibration = dict(
        calibrate(
            source=args.source,
            preflight_receipt=receipt,
            checkpoint_path=args.calibration_checkpoint,
            trials_per_condition=trials,
            smoke=args.smoke,
        )
    )
    atomic_json(args.calibration_output, calibration)
    return calibration


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args)
    print(
        f"phase={args.phase} molecular_outcomes_read={result['molecular_outcomes_read']} "
        f"biological_execution_authorized="
        f"{result.get('HEST_molecular_execution_authorized', False)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
