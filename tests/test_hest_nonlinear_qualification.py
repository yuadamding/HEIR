from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _runner():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_hest_nonlinear_qualification.py"
    spec = importlib.util.spec_from_file_location("benchmark_hest_nonlinear", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _runner()


def test_sparse_source_labels_receive_one_frozen_contiguous_mapping() -> None:
    raw = np.asarray([0, 2, 5, 2, 5, 0])
    names = np.asarray(["A", "unused-1", "B", "unused-3", "unused-4", "C"])
    mapping = RUNNER._type_mapping(raw, names)
    assert mapping["rows"] == [
        {"raw_label": 0, "contiguous_label": 0, "type_name": "A"},
        {"raw_label": 2, "contiguous_label": 1, "type_name": "B"},
        {"raw_label": 5, "contiguous_label": 2, "type_name": "C"},
    ]
    assert len(mapping["sha256"]) == 64


def test_blocked_full_phase_writes_non_authorizing_report(tmp_path: Path, monkeypatch) -> None:
    blockers = [
        "registered_source_missing_receipt_bound_blank_patch_embedding",
        "best_registration_subset_has_zero_donor_type_strata_at_primary_support_20",
    ]

    def preflight(*args, **kwargs):
        return {
            "schema": "heir.hest_nonlinear_qualification_preflight.v1",
            "analysis_status": RUNNER.ANALYSIS_STATUS,
            "execution_authorized": False,
            "blockers": blockers,
        }

    monkeypatch.setattr(RUNNER, "preflight", preflight)
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    report = RUNNER.run(
        source=tmp_path / "unused.npz",
        protocol=tmp_path / "unused.json",
        manifest=tmp_path / "unused-manifest.json",
        output=output,
        markdown_output=markdown,
        checkpoint_dir=tmp_path / "checkpoints",
        phase="full",
        device="cpu",
        verify_source_hash=False,
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert report == persisted
    assert report["biological_experiment_run"] is False
    assert report["engineering_decision_available"] is False
    assert report["supports_new_prospective_estimator_protocol"] is False
    assert report["authorizes_h_cell"] is False
    assert report["authorizes_h_intrinsic"] is False
    assert report["authorizes_h_ref"] is False
    assert report["authorizes_full_heir"] is False
    assert "full exposed-HEST qualification was not run" in markdown.read_text()


def test_synthetic_smoke_executes_exact_registered_null_counts(tmp_path: Path) -> None:
    result = RUNNER._synthetic_smoke(tmp_path / "checkpoints", device="cpu")
    assert result["smoke_pass"] is True
    assert result["biological_data_fit"] is False
    assert result["checkpoint_replay_identical"] is True
    assert set(result["nulls"]) == {
        "within_section_type_derangement",
        "different_spatial_block_reassignment",
    }
    assert all(null["permutations"] == 20 for null in result["nulls"].values())
    assert all(
        null["model_refit_for_every_permutation"] is True for null in result["nulls"].values()
    )
    assert result["complete_refit_steps_exercised"] == [
        "preprocessing",
        "target_fitting",
        "hyperparameter_selection",
        "training",
        "checkpoint_selection",
        "prediction",
        "scoring",
    ]
    for null in result["nulls"].values():
        assert len(null["refit_receipts"]) == 20
        assert all(len(row["target_fit_sha256"]) == 64 for row in null["refit_receipts"])
        assert all(
            row["inner_donors"] == ["D1", "D2", "D3", "D4"]
            for row in null["refit_receipts"]
        )
        assert all(row["refit_checkpoint_sha256"] for row in null["refit_receipts"])
