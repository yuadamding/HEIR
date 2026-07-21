from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

from heir.evaluation import nonlinear_qualification_v2_execution as execution
from heir.evaluation.nonlinear_qualification_v2_contract import CALIBRATION_EXPECTATIONS
from heir.evaluation.registration_topology import TopologyArrays


def _topology() -> TopologyArrays:
    donors = np.repeat([f"D{index:02d}" for index in range(12)], 40)
    rows = len(donors)
    within_donor = np.tile(np.arange(40), 12)
    return TopologyArrays(
        observation_ids=np.asarray([f"cell-{index}" for index in range(rows)]),
        donor_ids=donors,
        section_ids=np.asarray([f"{donor}-S" for donor in donors]),
        type_labels=np.tile(np.repeat([0, 1], 20), 12),
        pool_roles=np.full(rows, "evaluation", dtype="<U16"),
        registration_quality_scores=np.where(within_donor < 20, 0.2, 1.2),
        registration_quality_strata=np.where(within_donor < 20, "best", "failed"),
    )


def _result(condition: str, trial_index: int) -> Mapping[str, object]:
    expected = CALIBRATION_EXPECTATIONS[condition]
    return {
        "condition": condition,
        "trial_index": trial_index,
        "D1": expected["D1"],
        "D2": expected["D2"],
        "scored": {
            "decision_metrics": {
                "D1": {"condition": condition, "trial_index": trial_index},
                "D2": {"condition": condition, "trial_index": trial_index},
            }
        },
    }


def test_calibration_resumes_at_the_first_uncommitted_trial(tmp_path: Path) -> None:
    checkpoint = tmp_path / "calibration.checkpoint.json"
    calls: list[tuple[str, int]] = []

    def interrupted(inputs, *, condition: str, trial_index: int, **kwargs):
        del inputs, kwargs
        calls.append((condition, trial_index))
        if len(calls) == 6:
            raise RuntimeError("simulated interruption")
        return _result(condition, trial_index)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        execution.run_calibration_trials(
            _topology(),
            binding={"protocol_sha256": "a" * 64},
            checkpoint_path=checkpoint,
            trials_per_condition=2,
            smoke=True,
            trial_runner=interrupted,
        )
    assert len(calls) == 6

    resumed_calls: list[tuple[str, int]] = []

    def resumed(inputs, *, condition: str, trial_index: int, **kwargs):
        del inputs, kwargs
        resumed_calls.append((condition, trial_index))
        return _result(condition, trial_index)

    receipt = execution.run_calibration_trials(
        _topology(),
        binding={"protocol_sha256": "a" * 64},
        checkpoint_path=checkpoint,
        trials_per_condition=2,
        smoke=True,
        trial_runner=resumed,
    )
    assert len(resumed_calls) == 15
    assert receipt["completed_trials"] == 20
    assert receipt["calibration"]["all_trials_match_expectation"] is True
    assert receipt["HEST_molecular_execution_authorized"] is False

    execution.run_calibration_trials(
        _topology(),
        binding={"protocol_sha256": "a" * 64},
        checkpoint_path=checkpoint,
        trials_per_condition=2,
        smoke=True,
        trial_runner=resumed,
    )
    assert len(resumed_calls) == 15


def test_calibration_rejects_checkpoint_tampering(tmp_path: Path) -> None:
    checkpoint = tmp_path / "calibration.checkpoint.json"
    execution.run_calibration_trials(
        _topology(),
        binding={"protocol_sha256": "b" * 64},
        checkpoint_path=checkpoint,
        trials_per_condition=1,
        smoke=True,
        trial_runner=lambda inputs, *, condition, trial_index, **kwargs: _result(
            condition, trial_index
        ),
    )
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    value["records"]["global_null"][0]["D1"] = True
    checkpoint.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint identity"):
        execution.run_calibration_trials(
            _topology(),
            binding={"protocol_sha256": "b" * 64},
            checkpoint_path=checkpoint,
            trials_per_condition=1,
            smoke=True,
            trial_runner=lambda *args, **kwargs: {},
        )


def test_calibration_rejects_structurally_invalid_rehashed_record(tmp_path: Path) -> None:
    checkpoint = tmp_path / "calibration.checkpoint.json"
    execution.run_calibration_trials(
        _topology(),
        binding={"protocol_sha256": "c" * 64},
        checkpoint_path=checkpoint,
        trials_per_condition=1,
        smoke=True,
        trial_runner=lambda inputs, *, condition, trial_index, **kwargs: _result(
            condition, trial_index
        ),
    )
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    value["records"]["global_null"] = ["invalid"]
    value["sha256"] = execution._checkpoint_hash(value)
    checkpoint.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="trial record"):
        execution.run_calibration_trials(
            _topology(),
            binding={"protocol_sha256": "c" * 64},
            checkpoint_path=checkpoint,
            trials_per_condition=1,
            smoke=True,
            trial_runner=lambda *args, **kwargs: {},
        )
