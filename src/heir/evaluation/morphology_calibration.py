"""Compile exact morphology-gate simulation evidence into a locked receipt.

This module deliberately does not contain a second, simplified morphology
model.  Calibration evidence must come from repeated calls to the production
``evaluate_morphology_ridge_gate`` entrypoint on synthetic artifacts.  The
compiler verifies that evidence and issues a receipt only when the exact
one-sided error and power confidence bounds pass.
"""

from __future__ import annotations

from typing import Mapping

from .power import (
    ACTUAL_GATE_ENTRYPOINT,
    CALIBRATION_ENGINE,
    CALIBRATION_EVIDENCE_SCHEMA,
    CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES,
    CALIBRATION_RECEIPT_SCHEMA,
    REQUIRED_CALIBRATION_SCENARIOS,
    REQUIRED_COMPLETE_GATE_CHECKS,
    REQUIRED_HYPOTHESIS_DECISIONS,
    REQUIRED_MORPHOLOGY_SOURCE_CONCLUSIONS,
    binomial_lower_confidence_bound,
    binomial_upper_confidence_bound,
    canonical_sha256,
    required_simultaneous_confidence_level,
    validate_calibration_receipt,
    validate_calibration_run_contract,
    validate_exact_gate_settings,
)

REQUIRED_SCENARIO_FAMILIES = REQUIRED_CALIBRATION_SCENARIOS
LEGACY_SURROGATE_ENGINE = "heir.legacy_surrogate_morphology_calibration.v1"


class CalibrationFailure(ValueError):
    """Raised when exact-gate evidence cannot issue an authorizing receipt."""

    def __init__(self, message: str, diagnostic: Mapping[str, object]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


def legacy_surrogate_diagnostic(
    scenario_config: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> Mapping[str, object]:
    """Describe legacy surrogate inputs without executing or authorizing them."""

    return {
        "schema": "heir.morphology_gate_surrogate_diagnostic.v2",
        "engine": LEGACY_SURROGATE_ENGINE,
        "pass": False,
        "calibrated": False,
        "authorizes_final_inference": False,
        "exact_gate_executed": False,
        "surrogate": True,
        "locked_outcomes_used": False,
        "synthetic_data_only": True,
        "reason": (
            "The legacy reduced simulator does not execute the production morphology gate "
            "and cannot issue an H-CELL calibration receipt."
        ),
        "legacy_scenario_config_sha256": canonical_sha256(scenario_config),
        "legacy_thresholds_sha256": canonical_sha256(thresholds),
    }


def calibrate_morphology_gate(
    scenario_config: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> Mapping[str, object]:
    """Return a non-authorizing diagnostic for the removed v1 surrogate API.

    Kept as an explicit compatibility trap: old scripts cannot silently create
    a receipt after the scientific contract changed to exact-gate calibration.
    """

    return legacy_surrogate_diagnostic(scenario_config, thresholds)


def _raw_condition(
    value: object,
    *,
    name: str,
    confidence_level: float,
    complete_gate_expected_pass: bool,
    decision_truth: Mapping[str, bool],
    permutations_per_null: int,
    permutation_seeds: tuple[int, ...],
    permutations_per_seed: int,
    expected_trials: int,
    expected_source_conclusion: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("actual-gate calibration condition %s is malformed" % name)
    required = {
        "trials",
        "complete_gate_passes",
        "hypothesis_decision_passes",
        "morphology_source_conclusion_counts",
        "actual_gate_executions",
        "trial_report_set_sha256",
        "all_trial_reports_use_exact_settings",
        "all_trial_reports_include_required_checks",
        "permutation_nulls",
    }
    if set(str(field) for field in value) != required:
        raise ValueError("actual-gate calibration condition %s is incomplete" % name)
    if (
        not isinstance(decision_truth, Mapping)
        or set(decision_truth) != set(REQUIRED_HYPOTHESIS_DECISIONS)
        or any(not isinstance(flag, bool) for flag in decision_truth.values())
    ):
        raise ValueError("actual-gate calibration condition has invalid decision truth")
    trials = value["trials"]
    passes = value["complete_gate_passes"]
    executions = value["actual_gate_executions"]
    if (
        isinstance(trials, bool)
        or not isinstance(trials, (int, float))
        or int(trials) != trials
        or int(trials) < 1000
        or isinstance(passes, bool)
        or not isinstance(passes, (int, float))
        or int(passes) != passes
        or not 0 <= int(passes) <= int(trials)
        or isinstance(executions, bool)
        or not isinstance(executions, (int, float))
        or int(executions) != int(trials)
    ):
        raise ValueError(
            "actual-gate calibration requires >=1000 complete executions per condition"
        )
    report_hash = str(value["trial_report_set_sha256"])
    if len(report_hash) != 64 or any(
        character not in "0123456789abcdef" for character in report_hash
    ):
        raise ValueError("actual-gate trial report-set hash is malformed")
    if (
        value["all_trial_reports_use_exact_settings"] is not True
        or value["all_trial_reports_include_required_checks"] is not True
    ):
        raise ValueError("actual-gate trial reports do not reproduce the frozen gate")
    permutation_nulls = value["permutation_nulls"]
    if not isinstance(permutation_nulls, Mapping) or set(permutation_nulls) != {
        "local_roi_permutations",
        "spatial_block_permutations",
        "local_roi_seed_counts",
        "spatial_block_seed_counts",
    }:
        raise ValueError("actual-gate calibration permutation counts are malformed")
    for count in (
        permutation_nulls["local_roi_permutations"],
        permutation_nulls["spatial_block_permutations"],
    ):
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, float))
            or int(count) != count
            or int(count) != permutations_per_null
        ):
            raise ValueError("actual-gate calibration differs from the exact permutation total")
    expected_seed_counts = {str(seed): permutations_per_seed for seed in permutation_seeds}
    if (
        permutation_nulls["local_roi_seed_counts"] != expected_seed_counts
        or permutation_nulls["spatial_block_seed_counts"] != expected_seed_counts
    ):
        raise ValueError("actual-gate calibration seed streams differ from frozen counts")
    trials = int(trials)
    passes = int(passes)
    if trials != expected_trials:
        raise ValueError("actual-gate calibration trial count differs from its run contract")
    raw_decision_passes = value["hypothesis_decision_passes"]
    if not isinstance(raw_decision_passes, Mapping) or set(raw_decision_passes) != set(
        REQUIRED_HYPOTHESIS_DECISIONS
    ):
        raise ValueError("actual-gate calibration lacks decision-specific pass counts")
    decision_passes = {}
    for decision_id in REQUIRED_HYPOTHESIS_DECISIONS:
        count = raw_decision_passes[decision_id]
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, float))
            or int(count) != count
            or not 0 <= int(count) <= trials
        ):
            raise ValueError("actual-gate calibration decision pass count is invalid")
        decision_passes[decision_id] = int(count)
    raw_conclusions = value["morphology_source_conclusion_counts"]
    if not isinstance(raw_conclusions, Mapping) or set(raw_conclusions) != set(
        CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES
    ):
        raise ValueError("actual-gate calibration lacks source-conclusion counts")
    conclusion_counts = {}
    for conclusion in CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES:
        count = raw_conclusions[conclusion]
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, float))
            or int(count) != count
            or int(count) < 0
        ):
            raise ValueError("actual-gate source-conclusion count is invalid")
        conclusion_counts[conclusion] = int(count)
    if sum(conclusion_counts.values()) != trials:
        raise ValueError("actual-gate source-conclusion counts do not sum to trials")
    if expected_source_conclusion not in REQUIRED_MORPHOLOGY_SOURCE_CONCLUSIONS:
        raise ValueError("calibration expected source conclusion is unsupported")
    correct_conclusions = conclusion_counts[expected_source_conclusion]
    conclusion_lower_bound = binomial_lower_confidence_bound(
        correct_conclusions,
        trials,
        confidence_level=confidence_level,
    )

    def confidence_bound(count: int, expected_pass: bool) -> float:
        return (
            binomial_lower_confidence_bound(count, trials, confidence_level=confidence_level)
            if expected_pass
            else binomial_upper_confidence_bound(count, trials, confidence_level=confidence_level)
        )

    bound = (
        binomial_lower_confidence_bound(passes, trials, confidence_level=confidence_level)
        if complete_gate_expected_pass
        else binomial_upper_confidence_bound(passes, trials, confidence_level=confidence_level)
    )
    return {
        "trials": trials,
        "complete_gate_passes": passes,
        "complete_gate_pass_fraction": passes / trials,
        "complete_gate_pass_confidence_bound": bound,
        "hypothesis_decision_passes": decision_passes,
        "hypothesis_decision_pass_fractions": {
            decision_id: count / trials for decision_id, count in decision_passes.items()
        },
        "hypothesis_decision_pass_confidence_bounds": {
            decision_id: confidence_bound(count, decision_truth[decision_id])
            for decision_id, count in decision_passes.items()
        },
        "morphology_source_conclusion_counts": conclusion_counts,
        "expected_morphology_source_conclusion": expected_source_conclusion,
        "morphology_source_conclusion_correct_count": correct_conclusions,
        "morphology_source_conclusion_correct_fraction": correct_conclusions / trials,
        "morphology_source_conclusion_correct_lower_confidence_bound": (conclusion_lower_bound),
        "actual_gate_executions": trials,
        "trial_report_set_sha256": report_hash,
        "all_trial_reports_use_exact_settings": True,
        "all_trial_reports_include_required_checks": True,
        "permutation_nulls": {
            "local_roi_permutations": int(permutation_nulls["local_roi_permutations"]),
            "spatial_block_permutations": int(permutation_nulls["spatial_block_permutations"]),
            "local_roi_seed_counts": dict(permutation_nulls["local_roi_seed_counts"]),
            "spatial_block_seed_counts": dict(permutation_nulls["spatial_block_seed_counts"]),
        },
    }


def compile_actual_gate_calibration_receipt(
    exact_gate_settings: Mapping[str, object],
    thresholds: Mapping[str, object],
    evidence: Mapping[str, object],
    *,
    confidence_level: float = 0.9998809523809524,
) -> Mapping[str, object]:
    """Issue a v3 receipt from completed production-gate simulation evidence."""

    settings = validate_exact_gate_settings(exact_gate_settings)
    if set(thresholds) != {
        "maximum_false_pass_upper_confidence_bound",
        "minimum_power_lower_confidence_bound",
    }:
        raise ValueError("exact-gate calibration thresholds are incomplete")
    maximum_false_pass = float(thresholds["maximum_false_pass_upper_confidence_bound"])
    minimum_power = float(thresholds["minimum_power_lower_confidence_bound"])
    if not 0.0 <= maximum_false_pass <= 0.05 or not 0.80 <= minimum_power <= 1.0:
        raise ValueError("exact-gate calibration thresholds are weaker than required")
    minimum_simultaneous_level = required_simultaneous_confidence_level()
    if not minimum_simultaneous_level <= float(confidence_level) < 1.0:
        raise ValueError(
            "exact-gate calibration requires Bonferroni-adjusted 95% simultaneous bounds"
        )
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "schema",
        "engine",
        "actual_gate_entrypoint",
        "exact_gate_settings_sha256",
        "run_contract",
        "run_contract_sha256",
        "scenario_results",
    }:
        raise ValueError("actual-gate calibration evidence is incomplete")
    settings_sha256 = canonical_sha256(settings)
    if (
        evidence["schema"] != CALIBRATION_EVIDENCE_SCHEMA
        or evidence["engine"] != CALIBRATION_ENGINE
        or evidence["actual_gate_entrypoint"] != ACTUAL_GATE_ENTRYPOINT
        or evidence["exact_gate_settings_sha256"] != settings_sha256
    ):
        raise ValueError("calibration evidence is not bound to the exact production gate")
    run_contract = validate_calibration_run_contract(
        evidence["run_contract"],
        expected_settings_sha256=settings_sha256,
        require_authorizing_boundary=True,
    )
    if evidence["run_contract_sha256"] != canonical_sha256(run_contract):
        raise ValueError("calibration evidence run-contract hash differs")
    dgp = run_contract["dgp_effect_spec"]
    null_condition_id = str(dgp["null_condition_id"])
    condition_ids = tuple(str(value) for value in run_contract["conditions"])
    boundary_condition_ids = condition_ids[1:]
    decision_truth = dgp["decision_truth_by_condition"]
    expected_conclusions = dgp["expected_source_conclusion_by_condition"]
    expected_trials = int(run_contract["trials_per_condition"])
    raw_results = evidence["scenario_results"]
    if not isinstance(raw_results, Mapping) or set(raw_results) != set(REQUIRED_SCENARIO_FAMILIES):
        raise ValueError("actual-gate evidence lacks required stress families")
    permutations = int(settings["permutations_per_null"])
    scenario_results = {}
    for scenario in REQUIRED_SCENARIO_FAMILIES:
        raw_result = raw_results[scenario]
        if not isinstance(raw_result, Mapping) or set(raw_result) != set(condition_ids):
            raise ValueError("actual-gate scenario lacks the frozen truth-matrix conditions")
        scenario_results[scenario] = {
            condition_id: _raw_condition(
                raw_result[condition_id],
                name="%s.%s" % (scenario, condition_id),
                confidence_level=float(confidence_level),
                complete_gate_expected_pass=condition_id != null_condition_id,
                decision_truth=decision_truth[condition_id],
                permutations_per_null=permutations,
                permutation_seeds=tuple(int(value) for value in settings["permutation_seeds"]),
                permutations_per_seed=int(settings["permutations_per_seed"]),
                expected_trials=expected_trials,
                expected_source_conclusion=str(expected_conclusions[condition_id]),
            )
            for condition_id in condition_ids
        }
    null_rates = [
        float(result[null_condition_id]["complete_gate_pass_fraction"])
        for result in scenario_results.values()
    ]
    null_upper_bounds = [
        float(result[null_condition_id]["complete_gate_pass_confidence_bound"])
        for result in scenario_results.values()
    ]
    effect_rates = [
        float(result[condition_id]["complete_gate_pass_fraction"])
        for result in scenario_results.values()
        for condition_id in boundary_condition_ids
    ]
    effect_lower_bounds = [
        float(result[condition_id]["complete_gate_pass_confidence_bound"])
        for result in scenario_results.values()
        for condition_id in boundary_condition_ids
    ]
    null_decision_rates = [
        float(result[condition_id]["hypothesis_decision_pass_fractions"][decision_id])
        for result in scenario_results.values()
        for condition_id in condition_ids
        for decision_id in REQUIRED_HYPOTHESIS_DECISIONS
        if decision_truth[condition_id][decision_id] is False
    ]
    null_decision_upper_bounds = [
        float(result[condition_id]["hypothesis_decision_pass_confidence_bounds"][decision_id])
        for result in scenario_results.values()
        for condition_id in condition_ids
        for decision_id in REQUIRED_HYPOTHESIS_DECISIONS
        if decision_truth[condition_id][decision_id] is False
    ]
    effect_decision_rates = [
        float(result[condition_id]["hypothesis_decision_pass_fractions"][decision_id])
        for result in scenario_results.values()
        for condition_id in boundary_condition_ids
        for decision_id in REQUIRED_HYPOTHESIS_DECISIONS
        if decision_truth[condition_id][decision_id] is True
    ]
    effect_decision_lower_bounds = [
        float(result[condition_id]["hypothesis_decision_pass_confidence_bounds"][decision_id])
        for result in scenario_results.values()
        for condition_id in boundary_condition_ids
        for decision_id in REQUIRED_HYPOTHESIS_DECISIONS
        if decision_truth[condition_id][decision_id] is True
    ]
    null_conclusion_correct_lower_bounds = [
        float(
            result[null_condition_id]["morphology_source_conclusion_correct_lower_confidence_bound"]
        )
        for result in scenario_results.values()
    ]
    alternative_conclusion_correct_lower_bounds = [
        float(result[condition_id]["morphology_source_conclusion_correct_lower_confidence_bound"])
        for result in scenario_results.values()
        for condition_id in boundary_condition_ids
    ]
    calibrated = bool(
        max(null_upper_bounds) <= maximum_false_pass
        and max(null_decision_upper_bounds) <= maximum_false_pass
        and min(effect_lower_bounds) >= minimum_power
        and min(effect_decision_lower_bounds) >= minimum_power
        and min(null_conclusion_correct_lower_bounds) >= 1.0 - maximum_false_pass
        and min(alternative_conclusion_correct_lower_bounds) >= minimum_power
    )
    thresholds = {
        "maximum_false_pass_upper_confidence_bound": maximum_false_pass,
        "minimum_power_lower_confidence_bound": minimum_power,
    }
    thresholds_sha256 = canonical_sha256(thresholds)
    simulation_core = {
        "engine": CALIBRATION_ENGINE,
        "exact_gate_settings_sha256": settings_sha256,
        "thresholds_sha256": thresholds_sha256,
        "run_contract_sha256": evidence["run_contract_sha256"],
        "scenario_results": scenario_results,
    }
    diagnostic = {
        "schema": "heir.morphology_gate_calibration_diagnostic.v3",
        "engine": CALIBRATION_ENGINE,
        "actual_gate_entrypoint": ACTUAL_GATE_ENTRYPOINT,
        "exact_gate_executed": True,
        "surrogate": False,
        "synthetic_data_only": True,
        "locked_outcomes_used": False,
        "confirmatory_scientific_settings_sha256": settings_sha256,
        "exact_gate_settings": dict(settings),
        "exact_gate_settings_sha256": settings_sha256,
        "run_contract": dict(run_contract),
        "run_contract_sha256": evidence["run_contract_sha256"],
        "generator_version": run_contract["generator_version"],
        "generator_source_sha256": run_contract["generator_source_sha256"],
        "gate_source_sha256": run_contract["gate_source_sha256"],
        "compiler_source_sha256": run_contract["compiler_source_sha256"],
        "contract_source_sha256": run_contract["contract_source_sha256"],
        "scientific_source_tree_sha256": run_contract["scientific_source_tree_sha256"],
        "dependency_lock_sha256": run_contract["dependency_lock_sha256"],
        "dgp_effect_spec": dict(dgp),
        "dgp_effect_spec_sha256": run_contract["dgp_effect_spec_sha256"],
        "thresholds": thresholds,
        "thresholds_sha256": thresholds_sha256,
        "scenario_families": list(REQUIRED_SCENARIO_FAMILIES),
        "scenario_results": scenario_results,
        "complete_gate_check_ids": list(REQUIRED_COMPLETE_GATE_CHECKS),
        "hypothesis_decision_ids": list(REQUIRED_HYPOTHESIS_DECISIONS),
        "confidence_level": float(confidence_level),
        "maximum_complete_gate_false_pass_probability": max(null_rates),
        "maximum_complete_gate_false_pass_upper_confidence_bound": max(null_upper_bounds),
        "maximum_hypothesis_decision_false_pass_probability": max(null_decision_rates),
        "maximum_hypothesis_decision_false_pass_upper_confidence_bound": max(
            null_decision_upper_bounds
        ),
        "power_at_quantitatively_frozen_boundary": min(effect_rates),
        "minimum_power_lower_confidence_bound": min(effect_lower_bounds),
        "minimum_hypothesis_decision_power_at_quantitatively_frozen_boundary": min(
            effect_decision_rates
        ),
        "minimum_hypothesis_decision_power_lower_confidence_bound": min(
            effect_decision_lower_bounds
        ),
        "minimum_global_null_source_conclusion_correct_lower_confidence_bound": min(
            null_conclusion_correct_lower_bounds
        ),
        "minimum_boundary_source_conclusion_correct_lower_confidence_bound": min(
            alternative_conclusion_correct_lower_bounds
        ),
        "simulation_sha256": canonical_sha256(simulation_core),
        "calibrated": calibrated,
    }
    if not calibrated:
        raise CalibrationFailure(
            "actual morphology gate failed exact false-pass or power calibration",
            diagnostic,
        )
    receipt_core = {
        **diagnostic,
        "schema": CALIBRATION_RECEIPT_SCHEMA,
        "pass": True,
    }
    receipt_core.pop("calibrated")
    receipt = {
        **receipt_core,
        "receipt_content_sha256": canonical_sha256(receipt_core),
    }
    validate_calibration_receipt(receipt, required=True)
    return receipt


__all__ = [
    "CALIBRATION_ENGINE",
    "CALIBRATION_RECEIPT_SCHEMA",
    "LEGACY_SURROGATE_ENGINE",
    "REQUIRED_SCENARIO_FAMILIES",
    "CalibrationFailure",
    "calibrate_morphology_gate",
    "compile_actual_gate_calibration_receipt",
    "legacy_surrogate_diagnostic",
]
