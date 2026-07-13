from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import pytest

from heir.evaluation.morphology_calibration import (
    REQUIRED_SCENARIO_FAMILIES,
    CalibrationFailure,
    calibrate_morphology_gate,
)
from heir.evaluation.power import (
    REQUIRED_COMPLETE_GATE_CHECKS,
    validate_calibration_receipt,
)


def test_complete_calibration_executes_every_null_and_minimum_effect_scenario(
    calibration_receipt: Mapping[str, object],
) -> None:
    assert calibration_receipt["schema"] == "heir.morphology_gate_calibration.v1"
    assert calibration_receipt["locked_outcomes_used"] is False
    assert calibration_receipt["synthetic_data_only"] is True
    assert calibration_receipt["complete_gate_executed"] is True
    assert set(calibration_receipt["complete_gate_check_ids"]) == set(
        REQUIRED_COMPLETE_GATE_CHECKS
    )
    assert set(calibration_receipt["scenario_results"]) == set(
        REQUIRED_SCENARIO_FAMILIES
    )
    for result in calibration_receipt["scenario_results"].values():
        assert result["null"]["trials"] == 10
        assert result["minimum_meaningful_effect"]["trials"] == 10
        assert "complete_gate_pass_fraction" in result["null"]
        assert "complete_gate_pass_fraction" in result["minimum_meaningful_effect"]
        assert result["null"]["permutation_nulls"]["local_roi_permutations"] == 19
        assert (
            result["null"]["permutation_nulls"]["spatial_block_permutations"]
            == 19
        )
    validated = validate_calibration_receipt(calibration_receipt, required=True)
    assert validated["maximum_complete_gate_false_pass_probability"] <= 0.05
    assert validated["power_at_minimum_meaningful_effect"] >= 0.80


def test_calibration_is_deterministic_and_hash_bound(
    calibration_receipt: Mapping[str, object],
    calibration_scenario_config: Mapping[str, object],
    calibration_thresholds: Mapping[str, object],
) -> None:
    repeated = calibrate_morphology_gate(
        calibration_scenario_config, calibration_thresholds
    )
    assert repeated["simulation_sha256"] == calibration_receipt["simulation_sha256"]
    assert repeated["receipt_content_sha256"] == calibration_receipt[
        "receipt_content_sha256"
    ]
    tampered = copy.deepcopy(calibration_receipt)
    first = REQUIRED_SCENARIO_FAMILIES[0]
    tampered["scenario_results"][first]["null"][
        "complete_gate_pass_fraction"
    ] = 0.5
    with pytest.raises(ValueError, match="pass counts are inconsistent"):
        validate_calibration_receipt(tampered, required=True)


def test_underpowered_simulation_emits_only_a_diagnostic(
    calibration_scenario_config: Mapping[str, object],
    calibration_thresholds: Mapping[str, object],
) -> None:
    underpowered = {**calibration_scenario_config, "minimum_effect_loading": 0.01}
    with pytest.raises(CalibrationFailure) as error:
        calibrate_morphology_gate(underpowered, calibration_thresholds)
    assert error.value.diagnostic["schema"].endswith("_diagnostic.v1")
    assert error.value.diagnostic["calibrated"] is False
    assert error.value.diagnostic["power_at_minimum_meaningful_effect"] < 0.80


def test_calibration_cli_atomically_separates_receipts_and_failed_diagnostics(
    tmp_path: Path,
    calibration_scenario_config: Mapping[str, object],
    calibration_thresholds: Mapping[str, object],
) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "calibrate_morphology_gate.py"
    config = tmp_path / "calibration.json"
    report = tmp_path / "receipt.json"
    diagnostic = tmp_path / "diagnostic.json"
    config.write_text(
        json.dumps(
            {
                "scenario_config": calibration_scenario_config,
                "thresholds": calibration_thresholds,
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--report-output",
            str(report),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert report.is_file()
    validate_calibration_receipt(json.loads(report.read_text()), required=True)

    failed_config = tmp_path / "failed.json"
    failed_report = tmp_path / "must-not-exist.json"
    failed_config.write_text(
        json.dumps(
            {
                "scenario_config": {
                    **calibration_scenario_config,
                    "minimum_effect_loading": 0.01,
                },
                "thresholds": calibration_thresholds,
            }
        ),
        encoding="utf-8",
    )
    failed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(failed_config),
            "--report-output",
            str(failed_report),
            "--diagnostic-output",
            str(diagnostic),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 2
    assert not failed_report.exists()
    assert json.loads(diagnostic.read_text())["schema"].endswith("_diagnostic.v1")
