#!/usr/bin/env python3
"""Compile completed actual-gate simulations into a v3 calibration receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from heir.evaluation.morphology_calibration import (
    CalibrationFailure,
    compile_actual_gate_calibration_receipt,
)
from heir.utils import atomic_json_dump, reject_output_input_collisions, sha256_file


def _load_json_object(path: Path, name: str) -> Mapping[str, object]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("%s is not valid JSON" % name) from error
    if not isinstance(content, Mapping):
        raise ValueError("%s must be a JSON object" % name)
    return content


def _load_configuration(
    path: Path,
) -> tuple[Mapping[str, object], Mapping[str, object], float]:
    content = _load_json_object(path, "calibration configuration")
    if set(content) != {
        "schema",
        "exact_gate_settings",
        "thresholds",
        "confidence_level",
    }:
        raise ValueError("calibration configuration is incomplete or contains extras")
    if content["schema"] != "heir.morphology_gate_calibration_config.v2":
        raise ValueError("calibration configuration schema is unsupported")
    settings = content["exact_gate_settings"]
    thresholds = content["thresholds"]
    if not isinstance(settings, Mapping) or not isinstance(thresholds, Mapping):
        raise ValueError("calibration configuration settings or thresholds are malformed")
    return settings, thresholds, float(content["confidence_level"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help=("aggregate evidence from repeated production-gate calls on synthetic artifacts"),
    )
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, default=None)
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    evidence_path = args.evidence.expanduser().resolve()
    output_path = args.report_output.expanduser().resolve()
    diagnostic_path = (
        args.diagnostic_output.expanduser().resolve()
        if args.diagnostic_output is not None
        else None
    )
    if not config_path.is_file() or not evidence_path.is_file():
        raise ValueError("calibration config and actual-gate evidence must exist")
    if output_path.exists():
        raise ValueError("calibration receipt output already exists and is immutable")
    if diagnostic_path is not None and diagnostic_path.exists():
        raise ValueError("calibration diagnostic output already exists and is immutable")
    outputs = (output_path,) + ((diagnostic_path,) if diagnostic_path is not None else ())
    reject_output_input_collisions(
        outputs,
        (config_path, evidence_path),
        label="morphology calibration",
    )
    settings, thresholds, confidence_level = _load_configuration(config_path)
    evidence = _load_json_object(evidence_path, "actual-gate calibration evidence")
    try:
        receipt = compile_actual_gate_calibration_receipt(
            settings,
            thresholds,
            evidence,
            confidence_level=confidence_level,
        )
    except CalibrationFailure as error:
        if diagnostic_path is not None:
            atomic_json_dump(dict(error.diagnostic), diagnostic_path)
        print(
            json.dumps(
                {
                    "calibrated": False,
                    "error": str(error),
                    "false_pass_upper_confidence_bound": error.diagnostic[
                        "maximum_complete_gate_false_pass_upper_confidence_bound"
                    ],
                    "hypothesis_false_pass_upper_confidence_bound": error.diagnostic[
                        "maximum_hypothesis_decision_false_pass_upper_confidence_bound"
                    ],
                    "power_lower_confidence_bound": error.diagnostic[
                        "minimum_power_lower_confidence_bound"
                    ],
                    "hypothesis_power_lower_confidence_bound": error.diagnostic[
                        "minimum_hypothesis_decision_power_lower_confidence_bound"
                    ],
                    "diagnostic": (str(diagnostic_path) if diagnostic_path is not None else None),
                    "diagnostic_sha256": (
                        sha256_file(diagnostic_path) if diagnostic_path is not None else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    atomic_json_dump(dict(receipt), output_path)
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "pass": receipt["pass"],
                "actual_gate_entrypoint": receipt["actual_gate_entrypoint"],
                "report": str(output_path),
                "report_sha256": sha256_file(output_path),
                "confirmatory_scientific_settings_sha256": receipt[
                    "confirmatory_scientific_settings_sha256"
                ],
                "false_pass_upper_confidence_bound": receipt[
                    "maximum_complete_gate_false_pass_upper_confidence_bound"
                ],
                "hypothesis_false_pass_upper_confidence_bound": receipt[
                    "maximum_hypothesis_decision_false_pass_upper_confidence_bound"
                ],
                "power_lower_confidence_bound": receipt["minimum_power_lower_confidence_bound"],
                "hypothesis_power_lower_confidence_bound": receipt[
                    "minimum_hypothesis_decision_power_lower_confidence_bound"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
