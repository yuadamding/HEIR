from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from heir.evaluation.nonlinear_qualification_contract import (
    ANALYSIS_STATUS,
    ARM_IDS,
    AUTHORIZATION_FIELDS,
    FROZEN_PREDECESSOR_SHA256,
    REPORT_SCHEMA,
    SUPPORT_THRESHOLDS,
    evaluate_engineering_support,
    file_sha256,
    non_authorizing_report_fields,
    validate_protocol,
    validate_report_authorization,
    validate_retrospective_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/hest_nonlinear_qualification_v1.json"
MANIFEST_PATH = ROOT / "manifests/studies/hest_nonlinear_qualification.retrospective.json"
PROTOCOL_SHA256 = "656419fcb75919fb938ba5a351014b7b72ba54b077da4c557103796f269d044f"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _passing_metrics() -> dict[str, float]:
    return {
        "donor_type_macro_residual_coordinate_r2": 0.05,
        "donor_section_type_macro_residual_coordinate_r2": 0.05,
        "improvement_over_b2_r2": 0.01,
        "improvement_over_n0_r2": 0.03,
        "molecular_error_reduction_over_reference_mean": 0.05,
        "positive_supported_donor_type_strata_fraction": 0.80,
        "positive_donors_versus_n0_fraction": 0.80,
        "maximum_single_donor_gain_fraction": 0.49,
        "molecular_variance_ratio": 0.50,
        "median_type_coverage": 0.50,
        "abstention_rate": 0.50,
        "rare_state_recall_drop": 0.20,
        "within_section_type_refitted_null_empirical_p": 1.0 / 101.0,
        "different_spatial_block_refitted_null_empirical_p": 1.0 / 101.0,
        "intrinsic_increment_over_target_removed_r2": 0.01,
        "best_registration_minus_all_rows_r2": -0.01,
    }


def test_protocol_bytes_matrix_and_status_are_frozen() -> None:
    protocol = _load(PROTOCOL_PATH)

    assert file_sha256(PROTOCOL_PATH) == PROTOCOL_SHA256
    assert validate_protocol(protocol) == protocol
    assert [row["id"] for row in protocol["experiment_matrix"]] == list(ARM_IDS)
    assert protocol["analysis_status"] == ANALYSIS_STATUS
    assert protocol["encoder_scope"]["primary"] == "bioptimus/H-optimus-1"
    assert protocol["encoder_scope"]["UNI2_h"].endswith("not_executed_in_this_protocol")
    assert protocol["archived_historical_anchor"]["relationship_to_B1"].startswith(
        "frozen_noncomparable_anchor"
    )
    assert protocol["molecular_target"]["rank_candidates"] == [2, 4, 6]
    assert protocol["molecular_target"]["minimum_basis_ceiling_r2"] == 0.3
    assert protocol["architecture_matched_controls"]["combined_nuisance_representation"] == {
        "source_key": "all_controls",
        "dimensions": 158,
        "definition": "frozen_deduplicated_existing_combined_nonimage_controls",
        "full_nuisance_covariates_31d_is_not_a_substitute": True,
    }
    assert protocol["architecture_matched_controls"]["blank_patch_input"]["status"] == (
        "blocked_missing_feature_supplement"
    )
    assert (
        protocol["validation_design"][
            "best_registration_supported_donor_type_strata_at_primary_support"
        ]
        == 0
    )
    assert protocol["execution_state"]["execution_authorized"] is False
    assert protocol["execution_state"]["neural_probe_implemented"] is True
    assert protocol["execution_state"]["synthetic_smoke_complete"] is True
    assert protocol["execution_state"]["biological_experiment_run"] is False
    assert all(protocol[field] is False for field in AUTHORIZATION_FIELDS)


def test_frozen_predecessor_files_still_have_registered_bytes() -> None:
    protocol = _load(PROTOCOL_PATH)

    assert protocol["frozen_predecessor_evidence"] == FROZEN_PREDECESSOR_SHA256
    for relative_path, expected_sha256 in FROZEN_PREDECESSOR_SHA256.items():
        assert file_sha256(ROOT / relative_path) == expected_sha256


def test_protocol_rejects_arm_threshold_encoder_and_authorization_drift() -> None:
    protocol = _load(PROTOCOL_PATH)

    changed_arm = copy.deepcopy(protocol)
    changed_arm["experiment_matrix"][1]["estimator"] = "mlp"
    with pytest.raises(ValueError, match="arm order or identity"):
        validate_protocol(changed_arm)

    changed_threshold = copy.deepcopy(protocol)
    changed_threshold["engineering_support_rule"]["improvement_over_b2_r2"]["threshold"] = 0.0
    with pytest.raises(ValueError, match="thresholds differ"):
        validate_protocol(changed_threshold)

    changed_encoder = copy.deepcopy(protocol)
    changed_encoder["encoder_scope"]["UNI2_h"] = "run"
    with pytest.raises(ValueError, match="may not execute"):
        validate_protocol(changed_encoder)

    changed_authorization = copy.deepcopy(protocol)
    changed_authorization["authorizes_h_cell"] = True
    with pytest.raises(ValueError, match="authorizes_h_cell must be false"):
        validate_protocol(changed_authorization)


def test_retrospective_manifest_binds_protocol_and_cannot_regain_a_lock() -> None:
    protocol = _load(PROTOCOL_PATH)
    manifest = _load(MANIFEST_PATH)

    assert (
        validate_retrospective_manifest(
            manifest,
            protocol,
            protocol_file_sha256=file_sha256(PROTOCOL_PATH),
        )
        == manifest
    )
    assert manifest["all_molecular_outcomes_previously_exposed"] is True
    assert manifest["prospective_lock_eligible"] is False
    assert manifest["registered_source"]["source_qc_pass"] is False
    assert manifest["registered_source"]["source_qc_failure_may_be_overridden"] is False
    assert manifest["execution"]["implementation_available"] is True
    assert manifest["execution"]["full_biological_runner_available"] is False
    assert manifest["execution"]["smoke_run_complete"] is True
    assert manifest["execution"]["biological_experiment_run"] is False
    assert manifest["execution"]["authorized"] is False
    assert manifest["registered_source"]["blank_patch_embedding_available"] is False
    assert all(manifest[field] is False for field in AUTHORIZATION_FIELDS)

    changed = copy.deepcopy(manifest)
    changed["prospective_lock_eligible"] = True
    with pytest.raises(ValueError, match="cannot regain prospective eligibility"):
        validate_retrospective_manifest(
            changed,
            protocol,
            protocol_file_sha256=file_sha256(PROTOCOL_PATH),
        )

    changed = copy.deepcopy(manifest)
    changed["protocol_binding"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="actual file bytes"):
        validate_retrospective_manifest(
            changed,
            protocol,
            protocol_file_sha256=file_sha256(PROTOCOL_PATH),
        )


def test_complete_threshold_pass_supports_only_prospective_protocol_design() -> None:
    report = evaluate_engineering_support(_passing_metrics())

    assert report["schema"] == REPORT_SCHEMA
    assert report["analysis_status"] == ANALYSIS_STATUS
    assert len(report["criteria"]) == len(SUPPORT_THRESHOLDS)
    assert all(row["pass"] for row in report["criteria"])
    assert report["supports_new_prospective_estimator_protocol"] is True
    assert all(report[field] is False for field in AUTHORIZATION_FIELDS)
    assert validate_report_authorization(report) == report


def test_complexity_tax_failure_stops_engineering_support() -> None:
    metrics = _passing_metrics()
    metrics["improvement_over_b2_r2"] = 0.009

    report = evaluate_engineering_support(metrics)

    assert report["supports_new_prospective_estimator_protocol"] is False
    failed = [row["metric"] for row in report["criteria"] if not row["pass"]]
    assert failed == ["improvement_over_b2_r2"]
    assert all(report[field] is False for field in AUTHORIZATION_FIELDS)


def test_support_metrics_are_exact_finite_and_bounded() -> None:
    missing = _passing_metrics()
    missing.pop("median_type_coverage")
    with pytest.raises(ValueError, match="exactly"):
        evaluate_engineering_support(missing)

    invalid = _passing_metrics()
    invalid["abstention_rate"] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_engineering_support(invalid)

    invalid = _passing_metrics()
    invalid["molecular_variance_ratio"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluate_engineering_support(invalid)

    improved_rare_state = _passing_metrics()
    improved_rare_state["rare_state_recall_drop"] = -0.1
    assert (
        evaluate_engineering_support(improved_rare_state)[
            "supports_new_prospective_estimator_protocol"
        ]
        is True
    )


def test_report_is_permanently_non_authorizing_even_after_engineering_pass() -> None:
    fields = non_authorizing_report_fields(True)
    attempted = {
        "schema": REPORT_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        **fields,
        "authorizes_h_intrinsic": True,
    }

    with pytest.raises(ValueError, match="authorizes_h_intrinsic must be false"):
        validate_report_authorization(attempted)


def test_protocol_documents_distinguish_smoke_from_biological_execution() -> None:
    protocol_report = (ROOT / "reports/hest_nonlinear_qualification_protocol.md").read_text(
        encoding="utf-8"
    )
    versions = (ROOT / "docs/estimator_versions.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/schaf_informed_roadmap.md").read_text(encoding="utf-8")

    assert "not a biological-results report" in protocol_report
    assert "used no HEST molecular rows" in " ".join(protocol_report.split())
    assert "`retrospective_exposed_non_authorizing`" in protocol_report
    assert "`nonlinear_qualification_v1`" in versions
    assert "Engineering qualification only" in versions
    assert "pristine H-CELL" in roadmap
    assert "Estimator work cannot substitute" in roadmap
