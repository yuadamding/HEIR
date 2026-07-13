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
    compile_actual_gate_calibration_receipt,
)
from heir.evaluation.power import (
    BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS,
    CALIBRATION_RECEIPT_SCHEMA,
    GLOBAL_NULL_CONDITION,
    REQUIRED_COMPLETE_GATE_CHECKS,
    REQUIRED_CROP_FAMILY_IDS,
    REQUIRED_HYPOTHESIS_DECISIONS,
    binomial_upper_confidence_bound,
    build_confirmatory_design_binding,
    canonical_sha256,
    confirmatory_scientific_manifest_projection,
    required_simultaneous_confidence_level,
    validate_calibration_receipt,
    validate_exact_gate_settings,
)


def test_exact_gate_receipt_binds_full_confirmatory_design(
    calibration_receipt: Mapping[str, object],
) -> None:
    assert calibration_receipt["schema"] == CALIBRATION_RECEIPT_SCHEMA
    assert calibration_receipt["surrogate"] is False
    assert calibration_receipt["exact_gate_executed"] is True
    assert calibration_receipt["locked_outcomes_used"] is False
    settings = calibration_receipt["exact_gate_settings"]
    assert settings["evaluation_donors"] == 5
    assert set(settings["crop_family_ids"]) == set(REQUIRED_CROP_FAMILY_IDS)
    assert settings["target_rank_grid"] == [2, 4, 6]
    assert settings["ridge_penalty_grid"] == [0.1, 1.0, 10.0, 100.0]
    assert len(settings["reference_split_ids"]) == 3
    assert settings["permutations_per_null"] == 999
    assert set(calibration_receipt["complete_gate_check_ids"]) == set(REQUIRED_COMPLETE_GATE_CHECKS)
    assert set(calibration_receipt["hypothesis_decision_ids"]) == set(REQUIRED_HYPOTHESIS_DECISIONS)
    assert set(calibration_receipt["scenario_results"]) == set(REQUIRED_SCENARIO_FAMILIES)
    for result in calibration_receipt["scenario_results"].values():
        assert result[GLOBAL_NULL_CONDITION]["trials"] == 1000
        for condition_id in BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS.values():
            assert result[condition_id]["trials"] == 1000
        assert set(result[GLOBAL_NULL_CONDITION]["hypothesis_decision_passes"]) == set(
            REQUIRED_HYPOTHESIS_DECISIONS
        )
        assert result[GLOBAL_NULL_CONDITION]["permutation_nulls"] == {
            "local_roi_permutations": 999,
            "spatial_block_permutations": 999,
            "local_roi_seed_counts": {"17": 333, "29": 333, "41": 333},
            "spatial_block_seed_counts": {"17": 333, "29": 333, "41": 333},
        }
    validated = validate_calibration_receipt(calibration_receipt, required=True)
    assert validated["maximum_complete_gate_false_pass_upper_confidence_bound"] <= 0.05
    assert validated["minimum_power_lower_confidence_bound"] >= 0.80
    assert validated["maximum_hypothesis_decision_false_pass_upper_confidence_bound"] <= 0.05
    assert validated["minimum_hypothesis_decision_power_lower_confidence_bound"] >= 0.80


def test_receipt_is_content_hash_bound_and_rejects_surrogate_v1(
    calibration_receipt: Mapping[str, object],
) -> None:
    tampered = copy.deepcopy(calibration_receipt)
    first = REQUIRED_SCENARIO_FAMILIES[0]
    tampered["scenario_results"][first][GLOBAL_NULL_CONDITION]["complete_gate_passes"] = 1
    with pytest.raises(ValueError, match="pass counts are inconsistent"):
        validate_calibration_receipt(tampered, required=True)
    with pytest.raises(ValueError, match="surrogate.*cannot authorize"):
        validate_calibration_receipt(
            {**calibration_receipt, "schema": "heir.morphology_gate_calibration.v1"},
            required=True,
        )


def test_receipt_rejects_extras_provenance_drift_and_seed_redistribution(
    calibration_receipt: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError, match="contains extras"):
        validate_calibration_receipt(
            {**calibration_receipt, "unregistered_field": True},
            required=True,
        )

    changed_source = copy.deepcopy(calibration_receipt)
    changed_source["run_contract"]["generator_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="executable provenance changed"):
        validate_calibration_receipt(changed_source, required=True)

    redistributed = copy.deepcopy(calibration_receipt)
    condition = redistributed["scenario_results"][REQUIRED_SCENARIO_FAMILIES[0]][
        GLOBAL_NULL_CONDITION
    ]
    condition["permutation_nulls"]["local_roi_seed_counts"] = {
        "17": 332,
        "29": 333,
        "41": 334,
    }
    with pytest.raises(ValueError, match="seed counts differ"):
        validate_calibration_receipt(redistributed, required=True)


def test_global_null_source_conclusion_is_calibrated_not_merely_reported(
    calibration_receipt: Mapping[str, object],
) -> None:
    tampered = copy.deepcopy(calibration_receipt)
    condition = tampered["scenario_results"][REQUIRED_SCENARIO_FAMILIES[0]][GLOBAL_NULL_CONDITION]
    condition["morphology_source_conclusion_counts"]["no_morphology_specific_information"] = 999
    condition["morphology_source_conclusion_counts"]["nucleus_dominant"] = 1
    with pytest.raises(ValueError, match="source-conclusion correct count differs"):
        validate_calibration_receipt(tampered, required=True)


def test_partial_null_false_pass_fails_truth_matrix_calibration(
    exact_gate_settings: Mapping[str, object],
    calibration_thresholds: Mapping[str, object],
    calibration_evidence: Mapping[str, object],
) -> None:
    tampered = copy.deepcopy(calibration_evidence)
    context_condition = BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS["G3_context_only"]
    condition = tampered["scenario_results"][REQUIRED_SCENARIO_FAMILIES[0]][context_condition]
    condition["hypothesis_decision_passes"]["G3_nucleus_intrinsic"] = 900

    with pytest.raises(CalibrationFailure, match="failed exact false-pass"):
        compile_actual_gate_calibration_receipt(
            exact_gate_settings,
            calibration_thresholds,
            tampered,
        )


def test_authorizing_boundary_hash_must_match_quantitative_condition(
    exact_gate_settings: Mapping[str, object],
    calibration_thresholds: Mapping[str, object],
    calibration_evidence: Mapping[str, object],
) -> None:
    tampered = copy.deepcopy(calibration_evidence)
    run_contract = tampered["run_contract"]
    nucleus_condition = BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS["G3_nucleus_intrinsic"]
    run_contract["dgp_effect_spec"]["effect_definition"]["condition_definitions"][
        nucleus_condition
    ]["minimum_effect"] = 0.08
    run_contract["dgp_effect_spec_sha256"] = canonical_sha256(run_contract["dgp_effect_spec"])
    tampered["run_contract_sha256"] = canonical_sha256(run_contract)

    with pytest.raises(ValueError, match="boundary hash differs"):
        compile_actual_gate_calibration_receipt(
            exact_gate_settings,
            calibration_thresholds,
            tampered,
        )


def test_checked_in_pre_h_meas_configuration_is_deliberately_non_executable(
    calibration_configuration: Mapping[str, object],
) -> None:
    pending = calibration_configuration["exact_gate_settings"]["confirmatory_design_binding"]
    assert pending["status"] == "pending_pre_h_meas"
    with pytest.raises(ValueError, match="pending pre-H-MEAS"):
        validate_exact_gate_settings(calibration_configuration["exact_gate_settings"])
    assert required_simultaneous_confidence_level() == pytest.approx(1.0 - 0.05 / 420.0)


def test_design_projection_excludes_only_lifecycle_and_receipt_cycle() -> None:
    root = Path(__file__).resolve().parents[1]
    content = json.loads(
        (root / "manifests/studies/hest_lung_cell_association.draft.json").read_text(
            encoding="utf-8"
        )
    )
    first = confirmatory_scientific_manifest_projection(content)
    changed = copy.deepcopy(content)
    changed["study_stage"] = "locked_confirmatory"
    changed["status"] = "opened"
    changed["morphology_gate"]["calibration_receipt_sha256"] = "a" * 64
    second = confirmatory_scientific_manifest_projection(changed)
    assert canonical_sha256(first) == canonical_sha256(second)

    changed["decision_thresholds"]["minimum_macro_r2"] = 0.051
    third = confirmatory_scientific_manifest_projection(changed)
    assert canonical_sha256(first) != canonical_sha256(third)


def test_completed_design_binding_requires_outcome_free_exact_stratum_topology() -> None:
    root = Path(__file__).resolve().parents[1]
    content = json.loads(
        (root / "manifests/studies/hest_lung_cell_association.draft.json").read_text(
            encoding="utf-8"
        )
    )
    genes = tuple("G%d" % index for index in range(8))
    fine_types = ("epithelial", "immune")
    measurement_sha = "a" * 64
    content["target_gene_panel_sha256"] = canonical_sha256(list(genes))
    content["observations"]["supported_fine_type_ids"] = list(fine_types)
    content["observations"]["supported_fine_type_ids_sha256"] = canonical_sha256(list(fine_types))
    content["prerequisites"]["measurement_report_sha256"] = measurement_sha
    donors = tuple(content["partitions"]["development_donors"]) + tuple(
        content["partitions"]["locked_test_donors"]
    )
    planned = tuple(
        "%s|%s_section_0|%s" % (donor, donor, fine_type)
        for donor in donors
        for fine_type in fine_types
    )

    pending = build_confirmatory_design_binding(
        content,
        measurement_receipt_sha256=measurement_sha,
        ordered_target_gene_ids=genes,
        supported_fine_type_ids=fine_types,
    )
    assert pending["planned_stratum_topology_status"] == "pending_h_meas_stratum_topology"

    completed = build_confirmatory_design_binding(
        content,
        measurement_receipt_sha256=measurement_sha,
        ordered_target_gene_ids=genes,
        supported_fine_type_ids=fine_types,
        ordered_planned_stratum_ids=planned,
        planned_stratum_minimum_evaluation_cells=(20,) * len(planned),
    )
    assert completed["planned_stratum_topology_status"] == "complete"
    assert completed["planned_stratum_manifest_sha256"] == canonical_sha256(list(planned))

    with pytest.raises(ValueError, match="frozen H-CELL minimum"):
        build_confirmatory_design_binding(
            content,
            measurement_receipt_sha256=measurement_sha,
            ordered_target_gene_ids=genes,
            supported_fine_type_ids=fine_types,
            ordered_planned_stratum_ids=planned,
            planned_stratum_minimum_evaluation_cells=(19,) * len(planned),
        )


def test_legacy_calibration_api_is_explicitly_non_authorizing() -> None:
    diagnostic = calibrate_morphology_gate(
        {"replicates_per_condition": 24, "evaluation_donors": 8},
        {"maximum_complete_gate_false_pass_probability": 0.05},
    )
    assert diagnostic["surrogate"] is True
    assert diagnostic["exact_gate_executed"] is False
    assert diagnostic["authorizes_final_inference"] is False
    with pytest.raises(ValueError, match="surrogate.*cannot authorize"):
        validate_calibration_receipt(diagnostic, required=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("evaluation_donors", 8, "exactly five"),
        ("development_donors", 5, "all ten"),
        ("crop_family_ids", list(REQUIRED_CROP_FAMILY_IDS[:4]), "crop_family_ids"),
        ("target_rank_grid", [2], "rank/ridge grid"),
        ("permutations_per_null", 19, "exactly 333 permutations"),
        ("reference_split_ids", ["primary"], "at least two reference splits"),
    ),
)
def test_exact_gate_compiler_rejects_surrogate_dimensions(
    exact_gate_settings: Mapping[str, object],
    calibration_thresholds: Mapping[str, object],
    calibration_evidence: Mapping[str, object],
    field: str,
    value: object,
    message: str,
) -> None:
    settings = {**exact_gate_settings, field: value}
    with pytest.raises(ValueError, match=message):
        compile_actual_gate_calibration_receipt(
            settings,
            calibration_thresholds,
            calibration_evidence,
        )


def test_at_least_one_thousand_actual_gate_runs_are_required_per_condition(
    exact_gate_settings: Mapping[str, object],
    calibration_thresholds: Mapping[str, object],
    mutable_calibration_evidence: Mapping[str, object],
) -> None:
    first = REQUIRED_SCENARIO_FAMILIES[0]
    condition = mutable_calibration_evidence["scenario_results"][first][GLOBAL_NULL_CONDITION]
    condition["trials"] = 999
    condition["actual_gate_executions"] = 999
    with pytest.raises(ValueError, match=">=1000 complete executions"):
        compile_actual_gate_calibration_receipt(
            exact_gate_settings,
            calibration_thresholds,
            mutable_calibration_evidence,
        )


def test_calibration_binds_live_decision_parameters_and_all_simultaneous_bounds(
    exact_gate_settings: Mapping[str, object],
    calibration_thresholds: Mapping[str, object],
    calibration_evidence: Mapping[str, object],
) -> None:
    altered_parameters = dict(exact_gate_settings["gate_parameters"])
    altered_parameters["minimum_support"] = 19
    with pytest.raises(ValueError, match="decision parameters"):
        compile_actual_gate_calibration_receipt(
            {**exact_gate_settings, "gate_parameters": altered_parameters},
            calibration_thresholds,
            calibration_evidence,
        )
    with pytest.raises(ValueError, match="Bonferroni-adjusted"):
        compile_actual_gate_calibration_receipt(
            exact_gate_settings,
            calibration_thresholds,
            calibration_evidence,
            confidence_level=0.9991666666666666,
        )


def test_false_pass_uses_upper_confidence_bound_not_point_estimate(
    exact_gate_settings: Mapping[str, object],
    calibration_thresholds: Mapping[str, object],
    mutable_calibration_evidence: Mapping[str, object],
) -> None:
    assert binomial_upper_confidence_bound(0, 1000) == pytest.approx(1.0 - 0.05 ** (1.0 / 1000.0))
    for result in mutable_calibration_evidence["scenario_results"].values():
        result[GLOBAL_NULL_CONDITION]["complete_gate_passes"] = 40
    with pytest.raises(CalibrationFailure) as error:
        compile_actual_gate_calibration_receipt(
            exact_gate_settings,
            calibration_thresholds,
            mutable_calibration_evidence,
        )
    assert error.value.diagnostic["maximum_complete_gate_false_pass_probability"] == 0.04
    assert error.value.diagnostic["maximum_complete_gate_false_pass_upper_confidence_bound"] > 0.05
    assert error.value.diagnostic["schema"].endswith("_diagnostic.v3")


def test_receipt_can_be_checked_against_expected_live_settings(
    calibration_receipt: Mapping[str, object],
    exact_gate_settings: Mapping[str, object],
) -> None:
    validated = validate_calibration_receipt(
        calibration_receipt,
        required=True,
        expected_settings=exact_gate_settings,
    )
    assert validated["exact_gate_settings_sha256"] == canonical_sha256(exact_gate_settings)
    with pytest.raises(ValueError, match="live confirmatory settings"):
        validate_calibration_receipt(
            calibration_receipt,
            required=True,
            expected_settings={
                **exact_gate_settings,
                "confirmatory_analysis_plan_sha256": "f" * 64,
            },
        )


def test_cli_requires_external_actual_gate_evidence_and_separates_diagnostics(
    tmp_path: Path,
    calibration_configuration: Mapping[str, object],
    exact_gate_settings: Mapping[str, object],
    calibration_evidence: Mapping[str, object],
) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "calibrate_morphology_gate.py"
    config = tmp_path / "calibration.json"
    evidence = tmp_path / "evidence.json"
    report = tmp_path / "receipt.json"
    diagnostic = tmp_path / "diagnostic.json"
    completed_configuration = {
        **calibration_configuration,
        "exact_gate_settings": exact_gate_settings,
    }
    config.write_text(json.dumps(completed_configuration), encoding="utf-8")
    evidence.write_text(json.dumps(calibration_evidence), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--evidence",
            str(evidence),
            "--report-output",
            str(report),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    validate_calibration_receipt(json.loads(report.read_text()), required=True)

    failed_evidence = copy.deepcopy(calibration_evidence)
    for result in failed_evidence["scenario_results"].values():
        result[GLOBAL_NULL_CONDITION]["complete_gate_passes"] = 40
    failed_path = tmp_path / "failed_evidence.json"
    failed_report = tmp_path / "must-not-exist.json"
    failed_path.write_text(json.dumps(failed_evidence), encoding="utf-8")
    failed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--evidence",
            str(failed_path),
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
    assert json.loads(diagnostic.read_text())["schema"].endswith("_diagnostic.v3")
