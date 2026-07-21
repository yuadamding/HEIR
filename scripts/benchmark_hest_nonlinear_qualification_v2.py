#!/usr/bin/env python3
# ruff: noqa: E402
"""Run outcome-locked HEST nonlinear qualification v2 phases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        _thread_value = int(os.environ.get(_thread_variable, "4"))
    except ValueError:
        _thread_value = 4
    os.environ[_thread_variable] = str(min(max(_thread_value, 1), 4))

from heir.evaluation.nonlinear_qualification_v2_data import load_hest_biological_data
from heir.evaluation.nonlinear_qualification_v2_execution import (
    PRODUCTION_TRIALS,
    atomic_json,
    authorize_biological_execution,
    calibrate,
    implementation_smoke,
    preflight,
)
from heir.evaluation.nonlinear_qualification_v2_full import run_full_qualification

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
DEFAULT_CALIBRATION_SMOKE = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/calibration.smoke.json")
DEFAULT_CHECKPOINT_SMOKE = Path(
    "/mnt/seagate/HEIR_runs/hest_nonlinear_v2/calibration.smoke.checkpoint.json"
)
DEFAULT_SMOKE = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/implementation_smoke.json")
DEFAULT_SMOKE_CHECKPOINTS = Path(
    "/mnt/seagate/HEIR_runs/hest_nonlinear_v2/implementation_smoke_checkpoints"
)
DEFAULT_WORK_DIR = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/full")
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/report.json")
DEFAULT_MARKDOWN = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/report.md")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("preflight", "smoke", "calibrate", "full"), required=True
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--blank-supplement", type=Path, default=DEFAULT_BLANK)
    parser.add_argument("--blank-receipt", type=Path, default=DEFAULT_BLANK_RECEIPT)
    parser.add_argument("--registration-topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight-output", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--calibration-checkpoint", type=Path)
    parser.add_argument("--implementation-smoke-output", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument(
        "--implementation-smoke-checkpoints", type=Path, default=DEFAULT_SMOKE_CHECKPOINTS
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trials-per-condition", type=int)
    return parser


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not a readable JSON receipt") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON receipt")
    return value


def _calibration_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    smoke = bool(args.phase == "calibrate" and args.smoke)
    output = args.calibration_output or (
        DEFAULT_CALIBRATION_SMOKE if smoke else DEFAULT_CALIBRATION
    )
    checkpoint = args.calibration_checkpoint or (
        DEFAULT_CHECKPOINT_SMOKE if smoke else DEFAULT_CHECKPOINT
    )
    if output.expanduser().resolve() == checkpoint.expanduser().resolve():
        raise ValueError("calibration receipt and checkpoint paths must differ")
    return output, checkpoint


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.phase != "calibrate" and (args.smoke or args.trials_per_condition is not None):
        raise ValueError("--smoke and --trials-per-condition apply only to calibration")
    calibration_output, calibration_checkpoint = _calibration_paths(args)
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
    if args.phase == "smoke":
        smoke_receipt = dict(
            implementation_smoke(
                preflight_receipt=receipt,
                checkpoint_dir=args.implementation_smoke_checkpoints,
                device=args.device,
            )
        )
        atomic_json(args.implementation_smoke_output, smoke_receipt)
        return smoke_receipt
    if args.phase == "full":
        smoke_receipt = _json(args.implementation_smoke_output)
        calibration_receipt = _json(calibration_output)
        authorization = authorize_biological_execution(
            source=args.source,
            preflight_receipt=receipt,
            implementation_smoke_receipt=smoke_receipt,
            calibration_receipt=calibration_receipt,
            device=args.device,
        )
        data = load_hest_biological_data(
            args.source,
            args.blank_supplement,
            authorization=authorization,
        )
        return dict(
            run_full_qualification(
                data,
                authorization,
                work_dir=args.work_dir,
                report_path=args.output,
                markdown_path=args.markdown_output,
                device=args.device,
            )
        )
    trials = args.trials_per_condition
    if trials is None:
        trials = 1 if args.smoke else PRODUCTION_TRIALS
    calibration = dict(
        calibrate(
            source=args.source,
            preflight_receipt=receipt,
            checkpoint_path=calibration_checkpoint,
            trials_per_condition=trials,
            smoke=args.smoke,
        )
    )
    atomic_json(calibration_output, calibration)
    return calibration


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args)
    print(
        f"phase={args.phase} molecular_outcomes_read={result['molecular_outcomes_read']} "
        f"biological_execution_authorized="
        f"{result.get('HEST_molecular_execution_authorized', False)} "
        f"biological_experiment_run={result.get('biological_experiment_run', False)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
