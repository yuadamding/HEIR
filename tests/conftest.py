from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Mapping
from unittest.mock import patch

import pytest

from heir.evaluation.morphology_calibration import (
    REQUIRED_SCENARIO_FAMILIES,
    compile_actual_gate_calibration_receipt,
)
from heir.evaluation.morphology_calibration_runner import (
    AUTHORIZING_DGP_EFFECT_SPEC,
    synthetic_completed_confirmatory_design_binding,
)
from heir.evaluation.power import (
    ACTUAL_GATE_ENTRYPOINT,
    ACTUAL_GATE_REPORT_SCHEMA,
    BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS,
    CALIBRATION_ENGINE,
    CALIBRATION_EVIDENCE_SCHEMA,
    CALIBRATION_GENERATOR_VERSION,
    CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES,
    CALIBRATION_RUN_CONTRACT_SCHEMA,
    CALIBRATION_TRIAL_REPORT_MANIFEST_SCHEMA,
    CALIBRATION_TRIAL_REPORT_STORAGE_LAYOUT,
    GLOBAL_NULL_CONDITION,
    REQUIRED_CALIBRATION_SCENARIOS,
    REQUIRED_COMPLETE_GATE_CHECKS,
    REQUIRED_HYPOTHESIS_DECISIONS,
    calibration_trial_seed,
    canonical_sha256,
    current_calibration_executable_provenance,
)


@pytest.fixture(scope="session")
def calibration_configuration() -> Mapping[str, object]:
    path = Path(__file__).resolve().parents[1] / "configs" / "morphology_gate_calibration.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def exact_gate_settings(
    calibration_configuration: Mapping[str, object],
) -> Mapping[str, object]:
    settings = copy.deepcopy(calibration_configuration["exact_gate_settings"])
    settings["complete_gate_check_ids"] = list(REQUIRED_COMPLETE_GATE_CHECKS)
    settings["confirmatory_design_binding"] = synthetic_completed_confirmatory_design_binding()
    return settings


@pytest.fixture(scope="session")
def calibration_thresholds(
    calibration_configuration: Mapping[str, object],
) -> Mapping[str, object]:
    return calibration_configuration["thresholds"]


@pytest.fixture(scope="session")
def calibration_evidence(
    exact_gate_settings: Mapping[str, object],
) -> Mapping[str, object]:
    condition_ids = [
        GLOBAL_NULL_CONDITION,
        *(
            BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS[decision_id]
            for decision_id in REQUIRED_HYPOTHESIS_DECISIONS
        ),
    ]
    dgp = copy.deepcopy(AUTHORIZING_DGP_EFFECT_SPEC)
    decision_truth = dgp["decision_truth_by_condition"]
    expected_conclusions = dgp["expected_source_conclusion_by_condition"]
    run_contract = {
        "schema": CALIBRATION_RUN_CONTRACT_SCHEMA,
        "generator_version": CALIBRATION_GENERATOR_VERSION,
        **current_calibration_executable_provenance(),
        "dgp_effect_spec": dgp,
        "dgp_effect_spec_sha256": canonical_sha256(dgp),
        "actual_gate_entrypoint": ACTUAL_GATE_ENTRYPOINT,
        "exact_gate_settings": dict(exact_gate_settings),
        "exact_gate_settings_sha256": canonical_sha256(exact_gate_settings),
        "permutations_per_null": 999,
        "permutation_seeds": [17, 29, 41],
        "permutations_per_seed": 333,
        "scenario_families": list(REQUIRED_CALIBRATION_SCENARIOS),
        "conditions": condition_ids,
        "trials_per_condition": 1000,
        "base_seed": 1729,
        "device": "cpu",
        "smoke_test": False,
        "process_isolation": "dedicated_cli_process",
        "max_cpu_threads": 1,
        "maximum_process_rss_gib": 16.0,
        "maximum_address_space_gib": 64.0,
        "trial_report_manifest_schema": CALIBRATION_TRIAL_REPORT_MANIFEST_SCHEMA,
        "trial_report_storage_layout": CALIBRATION_TRIAL_REPORT_STORAGE_LAYOUT,
    }
    seed_rows = [
        {
            "seed": seed,
            "required_unique_permutations": 333,
            "generated_unique_permutations": 333,
        }
        for seed in (17, 29, 41)
    ]
    settings_sha256 = canonical_sha256(exact_gate_settings)
    run_contract_sha256 = canonical_sha256(run_contract)
    report_templates_by_id = {}
    template_ids_by_condition = {}
    for condition in condition_ids:
        truth = decision_truth[condition]
        expected_conclusion = expected_conclusions[condition]

        def report(*, success: bool) -> Mapping[str, object]:
            component_pass = bool(success and condition != GLOBAL_NULL_CONDITION)
            return {
                "schema_version": ACTUAL_GATE_REPORT_SCHEMA,
                "component_pass": component_pass,
                "final_inference": True,
                "synthetic_calibration_execution": True,
                "scientific_authorization_suppressed": True,
                "calibration_exact_gate_settings_sha256": settings_sha256,
                "calibration_exact_gate_settings": exact_gate_settings,
                "checks": {name: component_pass for name in REQUIRED_COMPLETE_GATE_CHECKS},
                "hypothesis_decisions": {
                    name: {
                        "tested": True,
                        "pass": bool(success and truth[name]),
                        **(
                            {
                                "registration_quality_sensitivity_pass": bool(
                                    success and truth[name]
                                ),
                                "registration_quality_sensitivity": {"synthetic_fixture": True},
                            }
                            if name in {"G3_nucleus_intrinsic", "G3_cell_intrinsic"}
                            else (
                                {
                                    (
                                        "incremental_intrinsic_registration_quality_"
                                        "sensitivity_pass"
                                    ): bool(success and truth[name]),
                                    "incremental_intrinsic_registration_quality_sensitivity": {
                                        "synthetic_fixture": True
                                    },
                                }
                                if name == "G3_mixed_intrinsic_context"
                                else {}
                            )
                        ),
                    }
                    for name in REQUIRED_HYPOTHESIS_DECISIONS
                },
                "morphology_source_conclusion": (
                    expected_conclusion
                    if success or condition == GLOBAL_NULL_CONDITION
                    else "inconclusive"
                ),
                "permutation_control": {
                    "total_permutations": 999,
                    "seeds": seed_rows,
                },
                "spatial_block_permutation_control": {
                    "total_permutations": 999,
                    "seeds": seed_rows,
                },
                "authorizes_full_heir": False,
                "authorizes_population_inference": False,
                "authorizes_external_generalization": False,
                "authorizes_validated_regional_association": False,
                "authorizes_nucleus_intrinsic_claim": False,
                "authorizes_cell_intrinsic_claim": False,
                "synthetic_fixture_condition": condition,
                "synthetic_fixture_success": success,
            }

        successful = report(success=True)
        failed = report(success=False)
        successful_template_id = canonical_sha256(
            {"condition": condition, "outcome": "success", "report": successful}
        )
        failed_template_id = canonical_sha256(
            {"condition": condition, "outcome": "failure", "report": failed}
        )
        report_templates_by_id[successful_template_id] = successful
        if condition != GLOBAL_NULL_CONDITION:
            report_templates_by_id[failed_template_id] = failed
        template_ids_by_condition[condition] = (
            [successful_template_id]
            if condition == GLOBAL_NULL_CONDITION
            else [successful_template_id, failed_template_id]
        )

    scenario_results = {}
    ordered_report_hashes = {}
    template_runs = {}
    for scenario in REQUIRED_SCENARIO_FAMILIES:
        scenario_results[scenario] = {}
        ordered_report_hashes[scenario] = {}
        template_runs[scenario] = {}
        for condition in condition_ids:
            passes = 0 if condition == GLOBAL_NULL_CONDITION else 900
            expected_conclusion = expected_conclusions[condition]
            template_ids = template_ids_by_condition[condition]
            template_id_by_trial = (
                [template_ids[0]] * 1000
                if condition == GLOBAL_NULL_CONDITION
                else [template_ids[0]] * 900 + [template_ids[1]] * 100
            )
            template_runs[scenario][condition] = (
                [{"start": 0, "stop": 1000, "template_id": template_ids[0]}]
                if condition == GLOBAL_NULL_CONDITION
                else [
                    {"start": 0, "stop": 900, "template_id": template_ids[0]},
                    {"start": 900, "stop": 1000, "template_id": template_ids[1]},
                ]
            )
            report_hashes = []
            for trial_index, template_id in enumerate(template_id_by_trial):
                identity = {
                    "scenario": scenario,
                    "condition": condition,
                    "trial_index": trial_index,
                    "trial_seed": calibration_trial_seed(
                        1729,
                        scenario,
                        condition,
                        trial_index,
                        ordered_conditions=condition_ids,
                    ),
                }
                report_hashes.append(
                    canonical_sha256(
                        {
                            **report_templates_by_id[template_id],
                            "calibration_trial_identity": identity,
                            "calibration_run_contract_sha256": run_contract_sha256,
                        }
                    )
                )
            ordered_report_hashes[scenario][condition] = report_hashes
            conclusion_counts = {name: 0 for name in CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES}
            conclusion_counts[expected_conclusion] = 1000 if passes == 0 else 900
            if passes:
                conclusion_counts["inconclusive"] = 100
            scenario_results[scenario][condition] = {
                "trials": 1000,
                "complete_gate_passes": passes,
                "hypothesis_decision_passes": {
                    decision_id: (900 if decision_truth[condition][decision_id] else 0)
                    for decision_id in REQUIRED_HYPOTHESIS_DECISIONS
                },
                "any_false_hypothesis_decision_passes": 0,
                "morphology_source_conclusion_counts": conclusion_counts,
                "actual_gate_executions": 1000,
                "trial_report_set_sha256": canonical_sha256(
                    {"ordered_actual_gate_report_sha256": report_hashes}
                ),
                "trial_realization_set_sha256": canonical_sha256(
                    {"ordered_trial_realization_sha256": report_hashes}
                ),
                "all_trial_reports_use_exact_settings": True,
                "all_trial_reports_include_required_checks": True,
                "permutation_nulls": {
                    "local_roi_permutations": 999,
                    "spatial_block_permutations": 999,
                    "local_roi_seed_counts": {"17": 333, "29": 333, "41": 333},
                    "spatial_block_seed_counts": {"17": 333, "29": 333, "41": 333},
                },
            }
    manifest_core = {
        "schema": CALIBRATION_TRIAL_REPORT_MANIFEST_SCHEMA,
        "storage": {
            "kind": "non_authorizing_test_fixture_templates",
            "layout": CALIBRATION_TRIAL_REPORT_STORAGE_LAYOUT,
            "report_templates_by_id": report_templates_by_id,
            "template_runs_by_scenario_condition": template_runs,
        },
        "ordered_report_sha256s_by_scenario_condition": ordered_report_hashes,
        "report_reference_count": (len(REQUIRED_SCENARIO_FAMILIES) * len(condition_ids) * 1000),
        "unique_report_count": (len(REQUIRED_SCENARIO_FAMILIES) * len(condition_ids) * 1000),
    }
    trial_report_manifest = {
        **manifest_core,
        "manifest_content_sha256": canonical_sha256(manifest_core),
    }
    return {
        "schema": CALIBRATION_EVIDENCE_SCHEMA,
        "engine": CALIBRATION_ENGINE,
        "actual_gate_entrypoint": ACTUAL_GATE_ENTRYPOINT,
        "exact_gate_settings_sha256": canonical_sha256(exact_gate_settings),
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha256,
        "trial_report_manifest": trial_report_manifest,
        "scenario_results": scenario_results,
    }


@pytest.fixture(scope="session")
def calibration_receipt(
    exact_gate_settings: Mapping[str, object],
    calibration_thresholds: Mapping[str, object],
    calibration_evidence: Mapping[str, object],
) -> Mapping[str, object]:
    # Aggregate calibration-math fixture only.  Production compilation rejects
    # its compact templates; preserved file-backed actual-gate attestation is
    # exercised separately by the runner tests.
    with patch(
        "heir.evaluation.morphology_calibration._recompute_evidence_from_trial_manifest",
        return_value=calibration_evidence["scenario_results"],
    ):
        return compile_actual_gate_calibration_receipt(
            exact_gate_settings,
            calibration_thresholds,
            calibration_evidence,
        )


@pytest.fixture
def mutable_calibration_evidence(
    calibration_evidence: Mapping[str, object],
) -> Mapping[str, object]:
    return copy.deepcopy(calibration_evidence)
