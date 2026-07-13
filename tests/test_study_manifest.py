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
        "target_gene_panel_sha256": ordered_ids_sha256(["TARGET_A", "TARGET_B"]),
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
            "strategy": "development-donor cross-fitted gene-disjoint annotation",
            "evidence_kind": "development_donor_cross_fitted_gene_disjoint_annotation",
            "annotation_receipt_sha256": "e" * 64,
            "ordered_annotation_feature_ids": ["ANNOTATION_A", "ANNOTATION_B"],
            "ordered_annotation_feature_ids_sha256": ordered_ids_sha256(
                ["ANNOTATION_A", "ANNOTATION_B"]
            ),
            "ordered_target_gene_ids": ["TARGET_A", "TARGET_B"],
            "ordered_target_gene_ids_sha256": ordered_ids_sha256(["TARGET_A", "TARGET_B"]),
            "annotation_target_overlap_count": 0,
            "annotation_training_scope": "development_donors_only",
            "annotation_training_donor_ids": ["d1", "d2", "d3"],
            "annotation_training_donor_ids_sha256": ordered_ids_sha256(["d1", "d2", "d3"]),
            "locked_donors_used_for_training": False,
            "same_cohort_annotation": True,
            "cross_fitting_method": "leave_one_donor_out",
            "cross_fitting_receipt_sha256": "f" * 64,
            "establishes_full_target_independence": True,
            "limitation": "Within-source validation only.",
        },
        "technical_covariates": ["log1p_library_size"],
        "controls": ["coordinate_only", "stain_only", "context_only"],
        "hyperparameter_grid": {"rank": [2, 4], "ridge": [0.1, 1.0]},
        "randomization": {"permutations": 999, "nulls": ["local", "regional"]},
        "primary_endpoint": {"metric": "equal_donor_equal_fine_type_r2"},
        "secondary_endpoints": [],
        "coverage_requirements": {
            "minimum_fraction": 0.8,
            "minimum_reference_cells_per_donor_section_type": 20,
            "minimum_evaluation_cells_per_donor_section_type": 20,
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
        "locked_measurement_audit": {
            "audit_timing": "after_confirmatory_lock_before_morphology_inference",
            "selection_changes_forbidden": True,
            "coverage_denominator": "all_h_meas_supported_fine_types_and_locked_donors",
            "maximum_annotation_nucleus_p95_um": 8.0,
            "maximum_annotation_cell_p95_um": 12.0,
            "maximum_cell_nucleus_p95_um": 8.0,
            "maximum_registration_nucleus_diameter_ratio_p95": 0.5,
            "maximum_registration_nearest_neighbor_ratio_p95": 0.5,
            "maximum_registration_outlier_fraction": 0.05,
            "maximum_nucleus_outside_cell_fraction": 0.01,
            "minimum_nucleus_cell_area_ratio": 0.05,
            "maximum_nucleus_cell_area_ratio": 0.95,
            "maximum_segmentation_outlier_fraction": 0.05,
            "maximum_crop_padding_p95": 0.25,
            "mostly_padded_cutoff": 0.5,
            "maximum_mostly_padded_fraction": 0.01,
            "minimum_within_fine_type_reliability": 0.4,
            "minimum_reliability_rows": 40,
            "minimum_locked_donor_type_reliability_fraction": 0.8,
        },
    }


def _write(path: Path, payload: dict[str, object]) -> StudyManifest:
    atomic_json_dump(payload, path)
    return StudyManifest.load(path)


def _pending_independence() -> dict[str, object]:
    return {
        "strategy": "pending exact annotation provenance",
        "evidence_kind": "pending",
        "annotation_receipt_sha256": None,
        "ordered_annotation_feature_ids": [],
        "ordered_annotation_feature_ids_sha256": None,
        "ordered_target_gene_ids": [],
        "ordered_target_gene_ids_sha256": None,
        "annotation_target_overlap_count": None,
        "annotation_training_scope": "unknown_pending_provenance",
        "annotation_training_donor_ids": [],
        "annotation_training_donor_ids_sha256": None,
        "locked_donors_used_for_training": None,
        "same_cohort_annotation": True,
        "cross_fitting_method": "pending",
        "cross_fitting_receipt_sha256": None,
        "establishes_full_target_independence": False,
        "limitation": "Exact annotation features and training provenance are unavailable.",
    }


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
    assert (
        opened.content["opening"]["locked_content_sha256"]
        == locked.content["locked_content_sha256"]
    )
    assert len(opened.content["opening"]["opening_receipt_sha256"]) == 64
    assert opened.content["opening"]["adoption_for_future_models"] is False
    with pytest.raises(ValueError, match="only a locked study"):
        open_manifest_content(
            opened,
            opened_by_commit="d" * 40,
            permitted_claims=(),
        )


def test_opening_claims_must_be_frozen_hypotheses(tmp_path: Path) -> None:
    locked_content = freeze_manifest_content(
        _draft(),
        git_commit="a" * 40,
        container_digest="sha256:" + "b" * 64,
        locked_at="2026-01-01T00:00:00+00:00",
    )
    locked = _write(tmp_path / "locked.json", dict(locked_content))

    with pytest.raises(ValueError, match="subset of the frozen hypotheses"):
        open_manifest_content(
            locked,
            opened_by_commit="c" * 40,
            permitted_claims=("H-CELL", "H-EXT"),
        )


def test_full_runtime_verification_requires_clean_matching_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locked_content = freeze_manifest_content(
        _draft(),
        git_commit="a" * 40,
        container_digest="sha256:" + "b" * 64,
        locked_at="2026-01-01T00:00:00+00:00",
    )
    path = tmp_path / "locked.json"
    atomic_json_dump(dict(locked_content), path)
    clean_checks: list[Path] = []
    monkeypatch.setattr("heir.data.study_manifest.current_git_commit", lambda root: "a" * 40)
    monkeypatch.setattr(
        "heir.data.study_manifest.require_clean_worktree",
        lambda root: clean_checks.append(Path(root)),
    )
    monkeypatch.delenv("HEIR_CONTAINER_DIGEST", raising=False)

    with pytest.raises(ValueError, match="HEIR_CONTAINER_DIGEST is required"):
        StudyManifest.load(
            path,
            require_status="locked",
            verify_runtime=True,
            require_clean_runtime=True,
            verify_container_digest=True,
            repository_root=tmp_path,
        )
    monkeypatch.setenv("HEIR_CONTAINER_DIGEST", "sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="container digest differs"):
        StudyManifest.load(
            path,
            require_status="locked",
            verify_runtime=True,
            require_clean_runtime=True,
            verify_container_digest=True,
            repository_root=tmp_path,
        )
    monkeypatch.setenv("HEIR_CONTAINER_DIGEST", "sha256:" + "b" * 64)
    verified = StudyManifest.load(
        path,
        require_status="locked",
        verify_runtime=True,
        require_clean_runtime=True,
        verify_container_digest=True,
        repository_root=tmp_path,
    )
    assert verified.status == "locked"
    assert clean_checks == [tmp_path, tmp_path, tmp_path]


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("locked_manifest_sha256", "0" * 64),
        ("opened_by_commit", "d" * 40),
        ("opened_at", "2026-03-01T00:00:00+00:00"),
        ("permitted_claims", []),
    ),
)
def test_opened_manifest_rejects_modified_opening_receipt(
    tmp_path: Path, field: str, replacement: object
) -> None:
    locked_content = freeze_manifest_content(
        _draft(),
        git_commit="a" * 40,
        container_digest="sha256:" + "b" * 64,
        locked_at="2026-01-01T00:00:00+00:00",
    )
    locked = _write(tmp_path / "locked.json", dict(locked_content))
    opened = dict(
        open_manifest_content(
            locked,
            opened_by_commit="c" * 40,
            permitted_claims=("H-CELL",),
            opened_at="2026-02-01T00:00:00+00:00",
        )
    )
    opened["opening"][field] = replacement

    with pytest.raises(ValueError, match="opening receipt was modified"):
        _write(tmp_path / (field + ".json"), opened)


def test_opened_manifest_rejects_locked_content_linkage_drift(tmp_path: Path) -> None:
    locked_content = freeze_manifest_content(
        _draft(),
        git_commit="a" * 40,
        container_digest="sha256:" + "b" * 64,
        locked_at="2026-01-01T00:00:00+00:00",
    )
    locked = _write(tmp_path / "locked.json", dict(locked_content))
    opened = dict(
        open_manifest_content(
            locked,
            opened_by_commit="c" * 40,
            permitted_claims=("H-CELL",),
            opened_at="2026-02-01T00:00:00+00:00",
        )
    )
    opened["opening"]["locked_content_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="does not bind the locked scientific content"):
        _write(tmp_path / "linkage-drift.json", opened)


def test_opened_manifest_requires_complete_opening_receipt(tmp_path: Path) -> None:
    locked_content = freeze_manifest_content(
        _draft(),
        git_commit="a" * 40,
        container_digest="sha256:" + "b" * 64,
        locked_at="2026-01-01T00:00:00+00:00",
    )
    locked = _write(tmp_path / "locked.json", dict(locked_content))
    opened = dict(
        open_manifest_content(
            locked,
            opened_by_commit="c" * 40,
            permitted_claims=("H-CELL",),
            opened_at="2026-02-01T00:00:00+00:00",
        )
    )
    del opened["opening"]["opening_receipt_sha256"]

    with pytest.raises(ValueError, match="opening is incomplete"):
        _write(tmp_path / "missing-opening-receipt.json", opened)


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
    draft["label_target_independence"] = _pending_independence()
    draft["morphology_gate"]["calibration_receipt_sha256"] = None
    manifest = _write(tmp_path / "pending.json", draft)
    assert manifest.status == "draft"
    with pytest.raises(ValueError, match="label-target independence"):
        freeze_manifest_content(
            draft,
            git_commit="a" * 40,
            container_digest="sha256:" + "b" * 64,
        )


def test_confirmatory_lock_cannot_be_authorized_by_boolean_only(tmp_path: Path) -> None:
    draft = _draft()
    draft["label_target_independence"] = _pending_independence()
    draft["label_target_independence"]["establishes_full_target_independence"] = True
    with pytest.raises(ValueError, match="pending.*must not claim"):
        _write(tmp_path / "boolean-only.json", draft)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("annotation_hash", "annotation feature IDs differ"),
        ("target_overlap", "overlap the frozen"),
        ("locked_training", "locked donors cannot train"),
        ("locked_training_id", "include locked donors"),
        ("missing_cross_fit", "requires development-only donor cross-fitting"),
        ("missing_receipt", "requires a receipt"),
        ("cohort_kind_conflict", "conflicts with same-cohort annotation scope"),
    ),
)
def test_confirmatory_independence_evidence_fails_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    draft = _draft()
    contract = draft["label_target_independence"]
    if mutation == "annotation_hash":
        contract["ordered_annotation_feature_ids_sha256"] = "0" * 64
    elif mutation == "target_overlap":
        contract["ordered_annotation_feature_ids"] = ["ANNOTATION_A", "TARGET_A"]
        contract["ordered_annotation_feature_ids_sha256"] = ordered_ids_sha256(
            contract["ordered_annotation_feature_ids"]
        )
    elif mutation == "locked_training":
        contract["locked_donors_used_for_training"] = True
    elif mutation == "locked_training_id":
        contract["annotation_training_donor_ids"] = ["d1", "d2", "d4"]
        contract["annotation_training_donor_ids_sha256"] = ordered_ids_sha256(
            contract["annotation_training_donor_ids"]
        )
    elif mutation == "missing_cross_fit":
        contract["cross_fitting_method"] = "pending"
        contract["cross_fitting_receipt_sha256"] = None
    elif mutation == "missing_receipt":
        contract["annotation_receipt_sha256"] = None
    elif mutation == "cohort_kind_conflict":
        contract["same_cohort_annotation"] = False
        contract["annotation_training_scope"] = "orthogonal_no_rna_training"
        contract["annotation_training_donor_ids"] = []
        contract["annotation_training_donor_ids_sha256"] = None
        contract["cross_fitting_method"] = "not_applicable"
        contract["cross_fitting_receipt_sha256"] = None
    with pytest.raises(ValueError, match=message):
        _write(tmp_path / (mutation + ".json"), draft)


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
        "locked_measurement_audit",
    ):
        draft.pop(name)
    draft["observations"].pop("supported_fine_type_ids")
    draft["observations"].pop("supported_fine_type_ids_sha256")
    draft["label_target_independence"] = _pending_independence()
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
        "same_section_source_overlap_allowed": True,
    }
    draft["decision_thresholds"] = {"required_opposite_pool_guard_um": 20.0}
    manifest = _write(tmp_path / "measurement.json", draft)
    assert manifest.study_stage == "measurement_development"
    assert manifest.hypothesis_ids == ("H-MEAS",)
    assert draft["coverage_requirements"]["minimum_locked_donors_per_fine_type"] == 0

    invalid = copy.deepcopy(draft)
    invalid["coverage_requirements"]["minimum_locked_donors_per_fine_type"] = 1
    with pytest.raises(ValueError, match="locked donor coverage minimum must be zero"):
        _write(tmp_path / "invalid-measurement.json", invalid)

    invalid_guard = copy.deepcopy(draft)
    invalid_guard["decision_thresholds"]["required_opposite_pool_guard_um"] = 0.0
    with pytest.raises(ValueError, match="opposite-pool guard must be finite and positive"):
        _write(tmp_path / "invalid-measurement-guard.json", invalid_guard)
