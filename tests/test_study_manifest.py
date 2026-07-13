from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from heir.data import (
    STUDY_MANIFEST_SCHEMA,
    StudyManifest,
    freeze_manifest_content,
    open_manifest_content,
    ordered_ids_sha256,
)
from heir.utils import atomic_json_dump


def _draft() -> dict[str, object]:
    return {
        "schema": STUDY_MANIFEST_SCHEMA,
        "study_id": "hest_cell_uni2h_v1",
        "study_stage": "confirmatory_morphology",
        "status": "draft",
        "hypothesis_ids": ["H-CELL", "H-INTRINSIC"],
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
            "supported_fine_type_ids": ["ft1", "ft2"],
            "supported_fine_type_ids_sha256": ordered_ids_sha256(["ft1", "ft2"]),
        },
        "encoder": {
            "manifest_sha256": "4" * 64,
            "feature_space_id": "uni2h_1536",
            "checkpoint_sha256": "5" * 64,
        },
        "crop_protocols": ["6" * 64],
        "reference_splits": {
            "primary_split_id": "primary",
            "split_ids": ["primary", "reference_hash_fold_0", "reference_hash_fold_1"],
        },
        "candidate_target_gene_panel_sha256": "7" * 64,
        "target_gene_panel_sha256": "9" * 64,
        "type_marker_panel_sha256": "8" * 64,
        "prerequisites": {
            "measurement_report_sha256": "a" * 64,
            "measurement_study_manifest_sha256": "b" * 64,
            "measurement_source_sha256": "c" * 64,
        },
        "lock_protection": {
            "reserved_exclusively_for": "H-CELL",
            "reserved_donor_ids": ["d4", "d5", "d6"],
            "prior_outcome_access_confirmed_false": True,
            "hescape_analysis_scope": "development_donors_only_hest_lock_unopened",
            "hescape_allowed_donor_ids": ["d1", "d2", "d3"],
            "forbidden_prior_outcome_uses": ["HESCAPE_locked_regional_outcomes"],
        },
        "label_target_independence": {
            "strategy": "exact_gene_disjoint_marker_panel",
            "marker_panel_sha256": "8" * 64,
            "candidate_marker_target_overlap_count": 0,
            "establishes_full_target_independence": True,
            "limitation": "No known residual marker overlap.",
        },
        "technical_covariates": ["log1p_library_size"],
        "controls": ["coordinate_only", "stain_only", "context_only"],
        "hyperparameter_grid": {"rank": [2, 4], "ridge": [0.1, 1.0]},
        "randomization": {"permutations": 999, "nulls": ["local", "regional"]},
        "primary_endpoint": {"metric": "equal_donor_equal_fine_type_r2"},
        "secondary_endpoints": [],
        "coverage_requirements": {
            "minimum_fraction": 0.8,
            "maximum_reference_evaluation_absolute_smd": 0.25,
            "maximum_reference_evaluation_categorical_total_variation": 0.25,
        },
        "decision_thresholds": {"minimum_r2": 0.05},
        "morphology_gate": {
            "experiment_role": "primary_hest_uni2h",
            "scientific_scope": "registered_cell_local_context_association",
            "final_inference": True,
            "calibration_receipt_sha256": "d" * 64,
            "minimum_final_permutations": 999,
            "minimum_coordinate_delta": 0.01,
            "minimum_stain_delta": 0.01,
            "minimum_null_shuffled_fraction": 0.95,
            "minimum_strata_coverage": 0.8,
            "minimum_expression_error_reduction": 0.05,
            "minimum_basis_ceiling_r2": 0.3,
            "maximum_direct_contrast_p": 0.05,
            "minimum_mask_implementation_pass_fraction": 1.0,
            "donor_bootstrap_iterations": 2000,
            "donor_bootstrap_seed": 17,
            "prespecified_fixed_hyperparameters": True,
        },
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


def test_confirmatory_draft_cannot_lock_before_measurement_and_independence(tmp_path: Path) -> None:
    draft = _draft()
    draft["target_gene_panel_sha256"] = None
    draft["prerequisites"] = {
        "measurement_report_sha256": None,
        "measurement_study_manifest_sha256": None,
        "measurement_source_sha256": None,
    }
    draft["observations"]["supported_fine_type_ids"] = []
    draft["observations"]["supported_fine_type_ids_sha256"] = None
    draft["label_target_independence"]["establishes_full_target_independence"] = False
    draft["morphology_gate"]["calibration_receipt_sha256"] = None
    manifest = _write(tmp_path / "pending.json", draft)
    assert manifest.status == "draft"
    with pytest.raises(ValueError, match="prerequisites"):
        freeze_manifest_content(
            draft,
            git_commit="a" * 40,
            container_digest="sha256:" + "b" * 64,
        )


def test_measurement_plan_is_separate_and_has_no_morphology_fields(tmp_path: Path) -> None:
    draft = _draft()
    draft["study_id"] = "hest_measurement_v1"
    draft["study_stage"] = "measurement_development"
    draft["hypothesis_ids"] = ["H-MEAS"]
    for name in (
        "encoder",
        "crop_protocols",
        "reference_splits",
        "target_gene_panel_sha256",
        "technical_covariates",
        "controls",
        "hyperparameter_grid",
        "prerequisites",
        "morphology_gate",
    ):
        draft.pop(name)
    draft["observations"].pop("supported_fine_type_ids")
    draft["observations"].pop("supported_fine_type_ids_sha256")
    draft["randomization"] = {
        "transcript_split_salt": "measurement-test-salt",
        "donor_cross_fit_seed": 17,
        "selection_partition": "development_only",
    }
    draft["coverage_requirements"] = {
        "minimum_coverage_fraction": 0.8,
        "minimum_reference_cells_per_stratum": 4,
        "minimum_evaluation_cells_per_stratum": 4,
        "minimum_development_donors_per_fine_type": 2,
        "minimum_locked_donors_per_fine_type": 0,
        "maximum_reference_evaluation_row_overlap": 0,
        "maximum_reference_evaluation_block_overlap": 0,
        "maximum_reference_evaluation_source_file_overlap": 0,
    }
    manifest = _write(tmp_path / "measurement.json", draft)
    assert manifest.study_stage == "measurement_development"
    assert manifest.hypothesis_ids == ("H-MEAS",)
    assert draft["coverage_requirements"]["minimum_locked_donors_per_fine_type"] == 0

    invalid = copy.deepcopy(draft)
    invalid["coverage_requirements"]["minimum_locked_donors_per_fine_type"] = 1
    with pytest.raises(ValueError, match="locked donor coverage minimum must be zero"):
        _write(tmp_path / "invalid-measurement.json", invalid)
