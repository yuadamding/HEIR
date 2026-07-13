from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Mapping

import pytest

from heir.evaluation.morphology_calibration import (
    REQUIRED_SCENARIO_FAMILIES,
    compile_actual_gate_calibration_receipt,
)
from heir.evaluation.morphology_calibration_runner import (
    synthetic_completed_confirmatory_design_binding,
)
from heir.evaluation.power import (
    ACTUAL_GATE_ENTRYPOINT,
    BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS,
    BOUNDARY_EXPECTED_SOURCE_CONCLUSION,
    CALIBRATION_DGP_SPEC_SCHEMA,
    CALIBRATION_ENGINE,
    CALIBRATION_EVIDENCE_SCHEMA,
    CALIBRATION_GENERATOR_VERSION,
    CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES,
    CALIBRATION_RUN_CONTRACT_SCHEMA,
    GLOBAL_NULL_CONDITION,
    REQUIRED_CALIBRATION_SCENARIOS,
    REQUIRED_HYPOTHESIS_DECISIONS,
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
    condition_definitions = {
        GLOBAL_NULL_CONDITION: {
            "construction": "synthetic_test_global_null",
            "minimum_effect": 0.0,
        },
        **{
            condition_id: {
                "decision_id": decision_id,
                "construction": "synthetic_test_quantitative_boundary",
                "minimum_effect": 0.05,
            }
            for decision_id, condition_id in BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS.items()
        },
    }
    condition_ids = [
        GLOBAL_NULL_CONDITION,
        *(
            BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS[decision_id]
            for decision_id in REQUIRED_HYPOTHESIS_DECISIONS
        ),
    ]
    decision_truth = {
        GLOBAL_NULL_CONDITION: {decision_id: False for decision_id in REQUIRED_HYPOTHESIS_DECISIONS}
    }
    for decision_id, condition_id in BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS.items():
        decision_truth[condition_id] = {
            candidate: candidate in {"G2_local_context", decision_id}
            for candidate in REQUIRED_HYPOTHESIS_DECISIONS
        }
    expected_conclusions = {
        GLOBAL_NULL_CONDITION: "no_morphology_specific_information",
        **{
            condition_id: BOUNDARY_EXPECTED_SOURCE_CONCLUSION[decision_id]
            for decision_id, condition_id in BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS.items()
        },
    }
    dgp = {
        "schema": CALIBRATION_DGP_SPEC_SCHEMA,
        "authorizing_boundary_calibration": True,
        "null_condition_id": GLOBAL_NULL_CONDITION,
        "alternative_condition_id": None,
        "boundary_condition_ids_by_hypothesis": dict(BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS),
        "decision_truth_by_condition": decision_truth,
        "effect_definition": {
            "schema": "heir.synthetic_test_boundary.v1",
            "condition_definitions": condition_definitions,
        },
        "expected_source_conclusion_by_condition": expected_conclusions,
        "hypothesis_specific_boundary_sha256": {
            decision_id: canonical_sha256(condition_definitions[condition_id])
            for decision_id, condition_id in BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS.items()
        },
    }
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
    }
    scenario_results = {}
    for scenario in REQUIRED_SCENARIO_FAMILIES:
        scenario_results[scenario] = {}
        for condition in condition_ids:
            passes = 0 if condition == GLOBAL_NULL_CONDITION else 900
            expected_conclusion = expected_conclusions[condition]
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
                "morphology_source_conclusion_counts": conclusion_counts,
                "actual_gate_executions": 1000,
                "trial_report_set_sha256": canonical_sha256(
                    {
                        "scenario": scenario,
                        "condition": condition,
                        "synthetic_fixture": True,
                    }
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
    return {
        "schema": CALIBRATION_EVIDENCE_SCHEMA,
        "engine": CALIBRATION_ENGINE,
        "actual_gate_entrypoint": ACTUAL_GATE_ENTRYPOINT,
        "exact_gate_settings_sha256": canonical_sha256(exact_gate_settings),
        "run_contract": run_contract,
        "run_contract_sha256": canonical_sha256(run_contract),
        "scenario_results": scenario_results,
    }


@pytest.fixture(scope="session")
def calibration_receipt(
    exact_gate_settings: Mapping[str, object],
    calibration_thresholds: Mapping[str, object],
    calibration_evidence: Mapping[str, object],
) -> Mapping[str, object]:
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
