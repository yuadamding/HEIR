from __future__ import annotations

import json
from pathlib import Path

import pytest

from heir.data import (
    STUDY_MANIFEST_SCHEMA,
    StudyManifest,
    freeze_manifest_content,
    open_manifest_content,
)
from heir.utils import atomic_json_dump


def _draft() -> dict[str, object]:
    return {
        "schema": STUDY_MANIFEST_SCHEMA,
        "study_id": "hest_cell_uni2h_v1",
        "status": "draft",
        "hypothesis_ids": ["H-MEAS", "H-CELL", "H-INTRINSIC"],
        "git_commit": "",
        "analysis_plan_sha256": "1" * 64,
        "container_digest": "",
        "dataset": {
            "repository": "MahmoodLab/hest",
            "revision": "revision",
            "source_study": "GSE250346",
            "source_manifest_sha256": "2" * 64,
        },
        "partitions": {
            "development_donors": ["d1", "d2", "d3"],
            "locked_test_donors": ["d4", "d5", "d6"],
            "external_test_donors": [],
            "split_manifest_sha256": "3" * 64,
        },
        "observations": {
            "level": "cell",
            "registration_method": "native_xenium_cell_id_join",
            "target_variants": [
                "nucleus_overlapping_transcripts",
                "whole_cell_assigned_transcripts",
            ],
            "broad_type_field": "final_lineage",
            "fine_type_field": "final_CT",
        },
        "encoder": {
            "manifest_sha256": "4" * 64,
            "feature_space_id": "uni2h_1536",
            "checkpoint_sha256": "5" * 64,
        },
        "crop_protocols": ["6" * 64],
        "target_gene_panel_sha256": "7" * 64,
        "type_marker_panel_sha256": "8" * 64,
        "technical_covariates": ["log1p_library_size"],
        "controls": ["coordinate_only", "stain_only", "context_only"],
        "hyperparameter_grid": {"rank": [2, 4], "ridge": [0.1, 1.0]},
        "randomization": {"permutations": 999, "nulls": ["local", "regional"]},
        "primary_endpoint": {"metric": "equal_donor_equal_fine_type_r2"},
        "secondary_endpoints": [],
        "coverage_requirements": {"minimum_fraction": 0.8},
        "decision_thresholds": {"minimum_r2": 0.05},
    }


def _write(path: Path, payload: dict[str, object]) -> StudyManifest:
    atomic_json_dump(payload, path)
    return StudyManifest.load(path)


def test_locked_manifest_prohibits_scientific_cli_overrides(tmp_path: Path) -> None:
    locked = freeze_manifest_content(
        _draft(),
        git_commit="a" * 40,
        container_digest="sha256:" + "b" * 64,
        locked_at="2026-01-01T00:00:00+00:00",
    )
    manifest = _write(tmp_path / "locked.json", dict(locked))
    assert manifest.status == "locked"
    with pytest.raises(ValueError, match="prohibits CLI scientific overrides"):
        manifest.reject_cli_overrides({"rank": 8})
    manifest.reject_cli_overrides({"rank": None})


def test_opened_manifest_preserves_locked_receipt_and_is_one_way(tmp_path: Path) -> None:
    locked_content = freeze_manifest_content(
        _draft(),
        git_commit="a" * 40,
        container_digest="sha256:" + "b" * 64,
        locked_at="2026-01-01T00:00:00+00:00",
    )
    locked = _write(tmp_path / "locked.json", dict(locked_content))
    opened_content = open_manifest_content(
        locked,
        opened_by_commit="c" * 40,
        permitted_claims=("H-CELL",),
        opened_at="2026-02-01T00:00:00+00:00",
    )
    opened = _write(tmp_path / "opened.json", dict(opened_content))
    assert opened.status == "opened"
    assert opened.content["opening"]["locked_manifest_sha256"] == locked.sha256
    assert opened.content["opening"]["adoption_for_future_models"] is False
    with pytest.raises(ValueError, match="only a locked study"):
        open_manifest_content(
            opened,
            opened_by_commit="d" * 40,
            permitted_claims=(),
        )


def test_study_manifest_rejects_donor_overlap(tmp_path: Path) -> None:
    draft = _draft()
    draft["partitions"]["locked_test_donors"] = ["d3", "d4", "d5"]
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(draft), encoding="utf-8")
    with pytest.raises(ValueError, match="partitions overlap"):
        StudyManifest.load(path)
