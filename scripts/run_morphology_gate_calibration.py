#!/usr/bin/env python3
"""Run resumable exact morphology-gate calibration on synthetic data only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument(
        "--execution-report-output",
        type=Path,
        default=None,
        help="optional explicitly non-authorizing execution metadata",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="resumable per-trial report-hash checkpoint",
    )
    parser.add_argument(
        "--trials-per-condition",
        type=int,
        default=None,
        help="defaults to 1000 in production or 1 with --smoke-test",
    )
    parser.add_argument("--base-seed", type=int, default=1729)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=1,
        help="hard limit requested from BLAS/OpenMP/PyTorch CPU pools (default: 1)",
    )
    parser.add_argument(
        "--maximum-process-rss-gib",
        type=float,
        default=16.0,
        help="checkpoint and stop if process RSS exceeds this value (default: 16)",
    )
    parser.add_argument(
        "--maximum-address-space-gib",
        type=float,
        default=64.0,
        help="separate RLIMIT_AS soft ceiling; keep above CUDA virtual mappings (default: 64)",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "allow fewer than 1000 trials per condition; still executes 999 permutations "
            "and can never authorize final inference"
        ),
    )
    args = parser.parse_args(argv)

    # Set pool variables before importing NumPy/PyTorch through the calibration runner.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = str(args.max_cpu_threads)
    os.environ["HEIR_CALIBRATION_DEDICATED_PROCESS"] = "1"
    from heir.evaluation.morphology_calibration_runner import (
        PRODUCTION_TRIALS_PER_CONDITION,
        CalibrationRunPlan,
        load_calibration_run_config,
        run_actual_gate_calibration,
    )
    from heir.utils import atomic_json_dump, reject_output_input_collisions, sha256_file

    config_path = args.config.expanduser().resolve()
    evidence_path = args.evidence_output.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    execution_path = (
        args.execution_report_output.expanduser().resolve()
        if args.execution_report_output is not None
        else None
    )
    if args.smoke_test and execution_path is None:
        raise ValueError(
            "--smoke-test requires --execution-report-output to persist non-authorization"
        )
    if evidence_path.exists():
        raise ValueError("calibration evidence output already exists and is immutable")
    if execution_path is not None and execution_path.exists():
        raise ValueError("calibration execution report already exists and is immutable")
    outputs = (evidence_path, checkpoint_path) + (
        (execution_path,) if execution_path is not None else ()
    )
    reject_output_input_collisions(outputs, (config_path,), label="morphology calibration run")

    settings = load_calibration_run_config(config_path)
    trials = args.trials_per_condition
    if trials is None:
        trials = 1 if args.smoke_test else PRODUCTION_TRIALS_PER_CONDITION
    plan = CalibrationRunPlan(
        exact_gate_settings=settings,
        trials_per_condition=trials,
        base_seed=args.base_seed,
        device=args.device,
        smoke_test=args.smoke_test,
        checkpoint_every=args.checkpoint_every,
        max_cpu_threads=args.max_cpu_threads,
        maximum_process_rss_gib=args.maximum_process_rss_gib,
        maximum_address_space_gib=args.maximum_address_space_gib,
    )
    execution = run_actual_gate_calibration(
        plan,
        checkpoint_path=checkpoint_path,
        resume=not args.no_resume,
    )
    evidence = execution["evidence"]
    if not isinstance(evidence, dict):
        raise AssertionError("calibration runner returned malformed evidence")
    atomic_json_dump(evidence, evidence_path)
    if execution_path is not None:
        atomic_json_dump(dict(execution), execution_path)
    print(
        json.dumps(
            {
                "schema": execution["schema"],
                "production_contract_satisfied": execution["production_contract_satisfied"],
                "smoke_test": execution["smoke_test"],
                "authorizes_scientific_claims": False,
                "authorizes_final_inference": False,
                "actual_gate_entrypoint": execution["actual_gate_entrypoint"],
                "evidence": str(evidence_path),
                "evidence_sha256": sha256_file(evidence_path),
                "evidence_content_sha256": execution["evidence_content_sha256"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "execution_report": (str(execution_path) if execution_path is not None else None),
                "non_authorizing_reason": execution["non_authorizing_reason"],
                "resource_limits": execution["resource_limits"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
