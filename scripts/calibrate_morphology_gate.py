#!/usr/bin/env python3
"""Issue a synthetic-only pre-lock morphology-gate calibration receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from heir.evaluation.morphology_calibration import (
    CalibrationFailure,
    calibrate_morphology_gate,
)
from heir.utils import atomic_json_dump, reject_output_input_collisions, sha256_file


def _load_configuration(path: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("calibration configuration is not valid JSON") from error
    if not isinstance(content, Mapping):
        raise ValueError("calibration configuration must be a JSON object")
    allowed = {"schema", "scenario_config", "thresholds"}
    if set(str(value) for value in content) - allowed:
        raise ValueError("calibration configuration contains unsupported inputs")
    schema = content.get("schema")
    if schema is not None and schema != "heir.morphology_gate_calibration_config.v1":
        raise ValueError("calibration configuration schema is unsupported")
    scenario_config = content.get("scenario_config")
    thresholds = content.get("thresholds")
    if not isinstance(scenario_config, Mapping) or not isinstance(thresholds, Mapping):
        raise ValueError("calibration configuration lacks scenario_config or thresholds")
    return scenario_config, thresholds


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, default=None)
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    output_path = args.report_output.expanduser().resolve()
    diagnostic_path = (
        args.diagnostic_output.expanduser().resolve()
        if args.diagnostic_output is not None
        else None
    )
    if not config_path.is_file():
        raise ValueError("calibration configuration must be an existing file")
    if output_path.exists():
        raise ValueError("calibration receipt output already exists and is immutable")
    if diagnostic_path is not None and diagnostic_path.exists():
        raise ValueError("calibration diagnostic output already exists and is immutable")
    outputs = (output_path,) + (
        (diagnostic_path,) if diagnostic_path is not None else ()
    )
    reject_output_input_collisions(
        outputs, (config_path,), label="morphology calibration"
    )
    scenario_config, thresholds = _load_configuration(config_path)
    try:
        receipt = calibrate_morphology_gate(scenario_config, thresholds)
    except CalibrationFailure as error:
        if diagnostic_path is not None:
            atomic_json_dump(dict(error.diagnostic), diagnostic_path)
        print(
            json.dumps(
                {
                    "calibrated": False,
                    "error": str(error),
                    "maximum_complete_gate_false_pass_probability": error.diagnostic[
                        "maximum_complete_gate_false_pass_probability"
                    ],
                    "power_at_minimum_meaningful_effect": error.diagnostic[
                        "power_at_minimum_meaningful_effect"
                    ],
                    "diagnostic": (
                        str(diagnostic_path) if diagnostic_path is not None else None
                    ),
                    "diagnostic_sha256": (
                        sha256_file(diagnostic_path)
                        if diagnostic_path is not None
                        else None
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
                "report": str(output_path),
                "report_sha256": sha256_file(output_path),
                "scenario_config_sha256": receipt["scenario_config_sha256"],
                "thresholds_sha256": receipt["thresholds_sha256"],
                "maximum_complete_gate_false_pass_probability": receipt[
                    "maximum_complete_gate_false_pass_probability"
                ],
                "power_at_minimum_meaningful_effect": receipt[
                    "power_at_minimum_meaningful_effect"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
