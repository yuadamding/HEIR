"""Fail-closed development refinement orchestration tests."""

import json
import subprocess
import sys
from pathlib import Path


def _plan(role: str = "development") -> dict:
    return {
        "schema": "heir.refinement_development_plan.v1",
        "analysis_role": role,
        "refinement_rounds": 1,
        "stages": [
            {"name": "train", "command": ["true"], "outputs": ["outputs/train.pt"]},
            {
                "name": "predict_round_0",
                "command": ["true"],
                "outputs": ["outputs/round0.npz"],
            },
            {
                "name": "refine_round_1",
                "command": ["true"],
                "outputs": ["outputs/refined.pt"],
            },
            {
                "name": "predict_round_1",
                "command": ["true"],
                "outputs": ["outputs/round1.npz"],
            },
            {
                "name": "development_spatial_evaluation",
                "command": ["true"],
                "outputs": ["outputs/evaluation.json"],
            },
        ],
    }


def test_development_orchestrator_prescribes_round_sequence(tmp_path) -> None:
    repository = Path(__file__).resolve().parents[1]
    plan = tmp_path / "plan.json"
    status = tmp_path / "status.json"
    plan.write_text(json.dumps(_plan()), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_refinement_development.py"),
            "--plan",
            str(plan),
            "--status",
            str(status),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(status.read_text())
    assert [item["name"] for item in payload["stages"]] == [
        "train",
        "predict_round_0",
        "refine_round_1",
        "predict_round_1",
        "development_spatial_evaluation",
    ]
    assert {item["state"] for item in payload["stages"]} == {"planned"}


def test_development_orchestrator_rejects_locked_role(tmp_path) -> None:
    repository = Path(__file__).resolve().parents[1]
    plan = tmp_path / "locked.json"
    plan.write_text(json.dumps(_plan("locked_validation")), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_refinement_development.py"),
            "--plan",
            str(plan),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "restricted to development roles" in result.stderr
