from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

from heir.evaluation import nonlinear_qualification_v2_execution as execution
from heir.evaluation.nonlinear_qualification_v2_contract import (
    CALIBRATION_EXPECTATIONS,
    SOURCE_SHA256,
    TOPOLOGY_RECEIPT_SHA256,
    evaluate_synthetic_calibration,
)
from heir.evaluation.nonlinear_qualification_v2_smoke import COMPLETE_REFIT_STEPS
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


def _smoke_receipt(preflight: Mapping[str, object]) -> dict[str, object]:
    binding = {
        "protocol_sha256": preflight["protocol_sha256"],
        "manifest_sha256": preflight["manifest_sha256"],
        "registered_source_sha256": SOURCE_SHA256,
        "preflight_sha256": execution.canonical_sha256(preflight),
        "implementation": preflight["implementation"],
        "git_commit": preflight["runtime"]["git_commit"],
        "device": "cpu",
        "permutations_per_family": 20,
    }
    null = {
        "permutations": 20,
        "empirical_p": 1 / 21,
        "mapping_set_sha256": "1" * 64,
        "refit_receipts_sha256": "2" * 64,
        "completed_steps": list(COMPLETE_REFIT_STEPS),
        "model_refit_for_every_permutation": True,
    }
    return {
        "schema": execution.IMPLEMENTATION_SMOKE_SCHEMA,
        "binding": binding,
        "binding_sha256": execution.canonical_sha256(binding),
        "candidate_ids": ["tiny-rank-2", "small-rank-4"],
        "rank_candidates": [2, 4],
        "selected_candidate_id": "tiny-rank-2",
        "selected_epoch": 3,
        "checkpoint_receipt": {
            "schema": "heir.neural_probe_checkpoint_receipt.v1",
            "state_sha256": "3" * 64,
            "array_registry_sha256": "4" * 64,
            "metadata_sha256": "5" * 64,
        },
        "checkpoint_replay_identical": True,
        "permutations_per_family": 20,
        "complete_refit_steps": list(COMPLETE_REFIT_STEPS),
        "nulls": {
            "within_section_type_derangement": dict(null),
            "different_spatial_block_reassignment": dict(null),
        },
        "smoke_passed": True,
        "molecular_outcomes_read": False,
        "biological_experiment_run": False,
        "engineering_decision_available": False,
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_h_ref": False,
        "authorizes_full_heir": False,
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


def test_biological_authorization_replays_the_production_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    topology_receipt = {
        "source_members_opened": ["outcome-blind"],
        "ordered_block_ids_sha256": "d" * 64,
        "molecular_outcomes_read": False,
    }
    monkeypatch.setattr(
        execution,
        "load_calibration_topology",
        lambda source: (None, topology_receipt),
    )
    preflight = {
        "schema": execution.PREFLIGHT_SCHEMA,
        "preflight_passed": True,
        "protocol_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "implementation": {"runner.py": "c" * 64},
        "runtime": {
            "git_commit": "e" * 40,
            "git_worktree_dirty": False,
        },
        "molecular_outcomes_read": False,
    }
    base_binding = {
        "protocol_sha256": preflight["protocol_sha256"],
        "manifest_sha256": preflight["manifest_sha256"],
        "registered_source_sha256": SOURCE_SHA256,
        "registration_topology_sha256": TOPOLOGY_RECEIPT_SHA256,
        "preflight_sha256": execution.canonical_sha256(preflight),
        "implementation": preflight["implementation"],
        "git_commit": preflight["runtime"]["git_commit"],
        "topology_access": topology_receipt,
    }
    binding = {
        **base_binding,
        "trials_per_condition": 200,
        "smoke": False,
        "registration_bootstrap_resamples": 20_000,
    }
    binding_sha = execution.canonical_sha256(binding)
    records = {
        condition: [
            {
                "condition": condition,
                "trial_index": index,
                "D1": expected["D1"],
                "D2": expected["D2"],
                "expected": expected,
                "matches_expectation": True,
                "decision_metrics_sha256": "f" * 64,
                "molecular_outcomes_read": False,
            }
            for index in range(200)
        ]
        for condition, expected in CALIBRATION_EXPECTATIONS.items()
    }
    checkpoint = {
        "schema": execution.CHECKPOINT_SCHEMA,
        "binding": binding,
        "binding_sha256": binding_sha,
        "records": records,
        "molecular_outcomes_read": False,
        "biological_experiment_run": False,
    }
    checkpoint["sha256"] = execution._checkpoint_hash(checkpoint)
    checkpoint_path = tmp_path / "production.checkpoint.json"
    execution.atomic_json(checkpoint_path, checkpoint)
    pass_counts = {
        condition: {
            decision: sum(record[decision] for record in local) for decision in ("D1", "D2")
        }
        for condition, local in records.items()
    }
    calibration = evaluate_synthetic_calibration(pass_counts)
    receipt = {
        "schema": execution.CALIBRATION_SCHEMA,
        "binding": binding,
        "binding_sha256": binding_sha,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint["sha256"],
        "production_calibration": True,
        "trials_per_condition": 200,
        "completed_trials": 2_000,
        "pass_counts": pass_counts,
        "calibration": calibration,
        "calibration_passed": True,
        "HEST_molecular_execution_authorized": True,
        "molecular_outcomes_read": False,
        "biological_experiment_run": False,
    }
    smoke_receipt = _smoke_receipt(preflight)

    authorization = execution.authorize_biological_execution(
        source=tmp_path / "source.npz",
        preflight_receipt=preflight,
        implementation_smoke_receipt=smoke_receipt,
        calibration_receipt=receipt,
        device="cpu",
    )
    assert authorization["biological_execution_authorized"] is True
    assert authorization["device"] == "cpu"
    assert authorization["molecular_outcomes_read"] is False
    assert authorization["authorizes_h_cell"] is False


def test_biological_authorization_rejects_smoke_calibration(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        execution,
        "load_calibration_topology",
        lambda source: (None, {}),
    )
    preflight = {
        "schema": execution.PREFLIGHT_SCHEMA,
        "preflight_passed": True,
        "protocol_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "implementation": {},
        "runtime": {"git_commit": "c" * 40, "git_worktree_dirty": False},
        "molecular_outcomes_read": False,
    }
    smoke_receipt = _smoke_receipt(preflight)
    with pytest.raises(ValueError, match="production-valid"):
        execution.authorize_biological_execution(
            source=tmp_path / "source.npz",
            preflight_receipt=preflight,
            implementation_smoke_receipt=smoke_receipt,
            calibration_receipt={"production_calibration": False},
            device="cpu",
        )


def test_biological_authorization_rejects_incomplete_refit_smoke(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(execution, "load_calibration_topology", lambda source: (None, {}))
    preflight = {
        "schema": execution.PREFLIGHT_SCHEMA,
        "preflight_passed": True,
        "protocol_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "implementation": {},
        "runtime": {"git_commit": "c" * 40, "git_worktree_dirty": False},
        "molecular_outcomes_read": False,
    }
    smoke_receipt = _smoke_receipt(preflight)
    smoke_receipt["nulls"]["within_section_type_derangement"]["permutations"] = 19

    with pytest.raises(ValueError, match="null receipt"):
        execution.authorize_biological_execution(
            source=tmp_path / "source.npz",
            preflight_receipt=preflight,
            implementation_smoke_receipt=smoke_receipt,
            calibration_receipt={},
            device="cpu",
        )
