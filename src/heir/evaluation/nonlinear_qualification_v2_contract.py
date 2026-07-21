"""Frozen protocol and decision contract for exposed HEST qualification v2."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from heir.evaluation.nonlinear_qualification_contract import (
    ANALYSIS_STATUS,
    file_sha256,
)
from heir.evaluation.nonlinear_qualification_contract import (
    validate_protocol as validate_v1_protocol,
)

PROTOCOL_SCHEMA = "heir.hest_nonlinear_qualification_protocol.v2"
MANIFEST_SCHEMA = "heir.hest_nonlinear_qualification_manifest.v2"
REPORT_SCHEMA = "heir.hest_nonlinear_qualification_report.v2"
BASE_PROTOCOL_SHA256 = "656419fcb75919fb938ba5a351014b7b72ba54b077da4c557103796f269d044f"
V2_PROTOCOL_SHA256 = "6e27f6bd1ea6a36d52ad23e8a34d603e917fa5c4e92123cb8f77fa1d64d9cb50"
SOURCE_SHA256 = "f7e7d4e97727cc17e71a81a252ab35fd2ca1c0e70054cba3ed38c2f7b7f65636"
BLANK_SUPPLEMENT_SHA256 = "a7495d795c6a73e9a8bfc172448d0d161a99402b0bc63cd1ae6966e1c3968219"
BLANK_RECEIPT_SHA256 = "41917b39c0e50f27e252d76a4507e0a7599591e00c09b25074f49b9413bbd962"
BLANK_FEATURE_SHA256 = "d81926fd76fef69e9e59b8777e068a008716eedbf939b0b85ef1e4018b57df6f"
TOPOLOGY_RECEIPT_SHA256 = "8f619537486e7e5814797551c5e8f22681118f4cfef49a98bf343f6aea170742"
INPUT_BUILD_GIT_COMMIT = "14169b0478d38d9657722d5ba3887aa3d768dd8d"
SELECTED_DONORS = (
    "THD0008",
    "THD0011",
    "TILD117",
    "TILD175",
    "VUHD069",
    "VUHD116",
    "VUILD102",
    "VUILD105",
    "VUILD106",
    "VUILD107",
    "VUILD115",
    "VUILD78",
    "VUILD91",
    "VUILD96",
)
CALIBRATION_EXPECTATIONS = {
    "global_null": {"D1": False, "D2": False},
    "nuisance_only": {"D1": False, "D2": False},
    "linear_image": {"D1": False, "D2": False},
    "nonlinear_image": {"D1": True, "D2": False},
    "context_only": {"D1": True, "D2": False},
    "nucleus_only": {"D1": True, "D2": True},
    "cell_only": {"D1": True, "D2": True},
    "donor_shortcut": {"D1": False, "D2": False},
    "variance_collapse": {"D1": False, "D2": False},
    "signal_confined_to_poor_registration": {"D1": False, "D2": False},
}
D1_THRESHOLDS = {
    "donor_type_macro_residual_coordinate_R2": (">=", 0.05),
    "donor_section_type_macro_residual_coordinate_R2": (">=", 0.05),
    "A_star_minus_B2_R2": (">=", 0.01),
    "A_star_minus_N0_R2": (">=", 0.03),
    "A_star_minus_C_blank_R2": (">=", 0.03),
    "relative_molecular_error_reduction_over_reference_mean": (">=", 0.05),
    "positive_donor_type_strata_fraction": (">=", 0.80),
    "positive_donors_versus_N0_fraction": (">=", 0.80),
    "maximum_single_donor_gain_fraction": ("<", 0.50),
    "molecular_variance_ratio": (">=", 0.50),
    "median_type_coverage": (">=", 0.50),
    "abstention_rate": ("<=", 0.50),
    "rare_state_recall_drop_vs_B2": ("<=", 0.20),
    "A_star_vs_B2_familywise_adjusted_p": ("<=", 0.05),
    "A_star_vs_N0_familywise_adjusted_p": ("<=", 0.05),
    "A_star_vs_C_blank_familywise_adjusted_p": ("<=", 0.05),
    "within_section_type_refitted_null_empirical_p": ("<=", 0.01),
    "different_spatial_block_refitted_null_empirical_p": ("<=", 0.01),
    "registration_quality_point_difference": (">=", -0.01),
    "registration_quality_one_sided_lower_bound": (">=", -0.01),
}
D2_THRESHOLDS = {
    "increment_over_N5_R2": (">=", 0.01),
    "localization_familywise_adjusted_p": ("<=", 0.05),
    "registration_quality_one_sided_lower_bound": (">=", -0.01),
    "absolute_donor_type_macro_R2": (">", 0.0),
    "increment_over_N0_R2": (">", 0.0),
    "increment_over_C_blank_R2": (">", 0.0),
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _non_authorizing(value: Mapping[str, Any], name: str) -> None:
    if value.get("analysis_status") != ANALYSIS_STATUS:
        raise ValueError(f"{name} must remain {ANALYSIS_STATUS}")
    for field in (
        "authorizes_h_cell",
        "authorizes_h_intrinsic",
        "authorizes_h_ref",
        "authorizes_full_heir",
    ):
        if value.get(field) is not False:
            raise ValueError(f"{name}.{field} must be false")


def _normalized_rules(value: Mapping[str, Any]) -> Mapping[str, tuple[str, float]]:
    result = {}
    for name, raw in value.items():
        rule = _mapping(raw, name)
        if set(rule) != {"operator", "threshold"} or rule["operator"] not in {
            ">=",
            "<=",
            "<",
            ">",
        }:
            raise ValueError(f"decision rule {name} is malformed")
        threshold = rule["threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"decision threshold {name} must be numeric")
        result[str(name)] = (str(rule["operator"]), float(threshold))
    return result


def validate_protocol(protocol: Mapping[str, Any], *, repository_root: Path) -> dict[str, Any]:
    """Validate v2 as a compact, explicit delta over immutable v1 bytes."""

    value = dict(_mapping(protocol, "v2 protocol"))
    if value.get("schema") != PROTOCOL_SCHEMA or value.get("protocol_id") != (
        "hest_nonlinear_qualification_v2"
    ):
        raise ValueError("unsupported HEST nonlinear qualification v2 protocol")
    _non_authorizing(value, "v2 protocol")
    if (
        value.get("study_role") != "retrospective_estimator_qualification_only"
        or value.get("outcome_exposure") != "all_hest_outcomes_previously_exposed"
        or value.get("molecular_outcomes_read_during_v2_registration") is not False
        or value.get("supports_prospective_local_context_estimator_design") is not False
        or value.get("supports_intrinsic_localization_followup") is not False
    ):
        raise ValueError("v2 registration must remain exposed, unexecuted, and nonauthorizing")

    base = _mapping(value.get("base_protocol"), "base_protocol")
    base_path = repository_root / str(base.get("path"))
    if (
        base.get("sha256") != BASE_PROTOCOL_SHA256
        or file_sha256(base_path) != BASE_PROTOCOL_SHA256
        or base.get("relationship") != "immutable_v1_inherited_except_explicit_v2_changes"
    ):
        raise ValueError("v2 does not bind the immutable v1 protocol")
    validate_v1_protocol(_json(base_path))
    inherited = set(base.get("inherited_sections", []))
    required_inheritance = {
        "experiment_matrix_B0_through_N7",
        "molecular_target_and_rank_grid",
        "outer_and_inner_donor_splits",
        "primary_support_20_hierarchical_R2_endpoints",
        "training_and_selection",
        "complete_refitted_null_families_and_counts",
        "all_v1_effect_variance_coverage_rare_state_and_null_thresholds",
    }
    if not required_inheritance <= inherited:
        raise ValueError("v2 fails to inherit a required v1 scientific contract")

    source = _mapping(value.get("registered_source"), "registered_source")
    if (
        source.get("sha256") != SOURCE_SHA256
        or source.get("rows") != 36121
        or source.get("reference_rows") != 18059
        or source.get("evaluation_rows") != 18062
        or source.get("biological_donors") != 15
        or source.get("rna_derived_types_cannot_authorize_h_cell") is not True
    ):
        raise ValueError("v2 registered source differs from exposed HEST")
    inputs = _mapping(value.get("outcome_blind_inputs"), "outcome_blind_inputs")
    blank = _mapping(inputs.get("blank_supplement"), "blank_supplement")
    if (
        blank.get("sha256") != BLANK_SUPPLEMENT_SHA256
        or blank.get("receipt_sha256") != BLANK_RECEIPT_SHA256
        or blank.get("feature_sha256") != BLANK_FEATURE_SHA256
        or blank.get("build_git_commit") != INPUT_BUILD_GIT_COMMIT
        or blank.get("feature_shape") != [1, 1536]
        or blank.get("broadcast_rows") != 36121
        or blank.get("independent_pixel_recomputation_passed") is not True
        or blank.get("molecular_outcomes_read") is not False
    ):
        raise ValueError("v2 blank supplement identity is invalid")
    topology = _mapping(inputs.get("registration_topology"), "registration_topology")
    if (
        topology.get("sha256") != TOPOLOGY_RECEIPT_SHA256
        or topology.get("audit_git_commit") != INPUT_BUILD_GIT_COMMIT
        or topology.get("selection_status") != "passed"
        or topology.get("molecular_outcomes_read") is not False
    ):
        raise ValueError("v2 registration topology identity is invalid")

    band = _mapping(value.get("registration_band_selection"), "registration_band_selection")
    if (
        band.get("candidate_order") != ["Q1", "Q2", "Q3"]
        or band.get("selected_quality_band_id") != "Q3"
        or band.get("selected_source_strata") != ["best", "intermediate", "near_threshold"]
        or band.get("selected_rows") != 3991
        or tuple(band.get("included_donor_ids", [])) != SELECTED_DONORS
        or band.get("largest_donor_fraction") != 0.19218241042345277
        or band.get("selection_used_molecular_outcomes") is not False
        or band.get("no_new_numeric_registration_cutoff") is not True
    ):
        raise ValueError("v2 selected registration band differs from the topology receipt")
    if band.get("rule") != {
        "minimum_donors": 12,
        "minimum_evaluation_rows": 500,
        "minimum_evaluation_rows_per_included_donor": 20,
        "minimum_represented_types_per_included_donor": 2,
        "maximum_donor_fraction": 0.2,
    }:
        raise ValueError("v2 topology selection rule changed")

    algorithms = _mapping(value.get("algorithm_selection"), "algorithm_selection")
    if (
        algorithms.get("primary_image_candidates") != ["N1", "N2", "N6", "N7"]
        or algorithms.get("single_view_candidates") != ["N1", "N2"]
        or algorithms.get("single_view_reused_without_retuning_for")
        != ["N3", "N4", "N5", "C_blank_single_view"]
        or algorithms.get("outer_evaluation_unit")
        != "entire_selection_algorithm_not_post_hoc_winning_arm"
    ):
        raise ValueError("v2 algorithm-level selection changed")
    guards = _mapping(value.get("training_inheritance_guards"), "training guards")
    if guards != {
        "v1_rank_optimizer_and_training_grid": "inherited_exactly",
        "batch_normalization": "prohibited",
        "encoder_fine_tuning": "prohibited",
        "row_level_random_split": "prohibited",
        "heldout_outer_donor_used_before_evaluation": False,
        "shared_local_field_blocks_kept_intact": True,
    }:
        raise ValueError("v2 training inheritance guards changed")
    controls = _mapping(value.get("architecture_matched_controls"), "controls")
    if set(controls) != {"C_blank", "C_target_removed"}:
        raise ValueError("v2 architecture-matched controls are incomplete")
    if _mapping(controls["C_blank"], "C_blank").get("image_input") != (
        "real_receipt_bound_H_optimus_blank_embedding"
    ):
        raise ValueError("v2 blank control is not the real encoded image")

    registration = _mapping(value.get("registration_sensitivity"), "registration_sensitivity")
    if (
        registration.get("minimum_donor_type_support") != 5
        or registration.get("minimum_formal_donors") != 12
        or registration.get("all_rows_composition_rule")
        != "reuse_selected_quality_supported_donor_type_strata"
        or registration.get("comparisons") != ["A_star_vs_N0", "N1_vs_N5", "N3_vs_N5", "N4_vs_N5"]
        or registration.get("noninferiority_margin") != -0.01
        or registration.get("models_retrained_on_quality_subset") is not False
        or registration.get("primary_support_20_R2_threshold_changed") is not False
        or registration.get("uncertainty")
        != {
            "method": "equal_donor_nonparametric_bootstrap",
            "resamples": 20000,
            "seed": 20260721,
            "one_sided_alpha": 0.05,
        }
    ):
        raise ValueError("v2 registration sensitivity differs from the frozen estimand")
    multiplicity = _mapping(value.get("multiplicity"), "multiplicity")
    if (
        multiplicity.get("method") != "exact_donor_sign_flip_max_statistic"
        or multiplicity.get("primary_direct_family")
        != ["A_star_vs_B2", "A_star_vs_N0", "A_star_vs_C_blank"]
        or multiplicity.get("localization_family") != ["N1_vs_N5", "N3_vs_N5", "N4_vs_N5"]
        or multiplicity.get("familywise_adjusted_p_maximum") != 0.05
    ):
        raise ValueError("v2 multiplicity families changed")
    nulls = _mapping(value.get("refitted_null_details"), "refitted null details")
    if (
        nulls.get("stratification") != ["donor", "section", "fine_type", "pool_role"]
        or nulls.get("multiview_mapping") != "one_synchronous_row_mapping_for_all_image_branches"
        or nulls.get("smoke_permutations_per_family") != 20
        or nulls.get("final_permutations_per_family") != 100
        or nulls.get("post_fit_feature_shuffle") != "prohibited"
        or nulls.get("complete_refit_steps")
        != [
            "feature_preprocessing",
            "technical_correction",
            "molecular_target_fitting",
            "rank_selection",
            "neural_architecture_selection",
            "training",
            "checkpoint_selection",
            "prediction",
            "scoring",
        ]
    ):
        raise ValueError("v2 refitted-null construction changed")
    decisions = _mapping(value.get("decision_rules"), "decision_rules")
    if (
        _normalized_rules(
            _mapping(
                decisions.get("D1_supports_prospective_local_context_estimator_design"),
                "D1 rules",
            )
        )
        != D1_THRESHOLDS
    ):
        raise ValueError("v2 D1 thresholds changed")
    d2 = _mapping(decisions.get("D2_supports_intrinsic_localization_followup"), "D2 rules")
    if (
        d2.get("eligible_arms") != ["N3", "N4"]
        or d2.get("D1_must_also_pass") is not True
        or _normalized_rules(_mapping(d2.get("per_arm_rules"), "D2 arm rules")) != D2_THRESHOLDS
    ):
        raise ValueError("v2 D2 thresholds changed")
    calibration = _mapping(value.get("synthetic_calibration"), "synthetic_calibration")
    generator = _mapping(calibration.get("generator"), "synthetic generator")
    if (
        calibration.get("trials_per_condition") != 200
        or calibration.get("conditions") != CALIBRATION_EXPECTATIONS
        or calibration.get("simultaneous_alpha") != 0.05
        or calibration.get("false_positive_upper_bound_must_be_below") != 0.05
        or calibration.get("true_positive_lower_bound_must_exceed") != 0.8
        or calibration.get("HEST_molecular_execution_allowed_after_calibration_failure")
        is not False
        or generator.get("registered_molecular_or_image_features_read") is not False
        or generator.get("coordinate_rank") != 4
        or generator.get("nuisance_width") != 158
        or generator.get("image_view_width") != 1536
        or generator.get("signal_amplitude") != 2.0
        or generator.get("coordinate_noise_standard_deviation") != 0.5
        or generator.get(
            "same_scoring_multiplicity_registration_and_decision_code_as_biological_run"
        )
        is not True
        or generator.get("nested_training_and_refit_path_checked_by_separate_implementation_smoke")
        is not True
    ):
        raise ValueError("v2 synthetic calibration contract changed")
    report = _mapping(value.get("report_contract"), "report_contract")
    if (
        report.get("biological_experiment_run_before_full_phase") is not False
        or report.get("engineering_decision_available_before_full_phase") is not False
        or report.get("all_results_remain_retrospective_exposed_non_authorizing") is not True
        or report.get("intermediate_HEST_results_may_change_frozen_choices") is not False
    ):
        raise ValueError("v2 report contract is not fail closed")
    return value


def validate_manifest(
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    protocol_file_sha256: str,
) -> dict[str, Any]:
    """Validate the static, pre-execution v2 manifest and all outcome-blind bindings."""

    validate_protocol(protocol, repository_root=repository_root)
    value = dict(_mapping(manifest, "v2 manifest"))
    if value.get("schema") != MANIFEST_SCHEMA or value.get("study_id") != (
        "hest_nonlinear_qualification_v2"
    ):
        raise ValueError("unsupported HEST nonlinear qualification v2 manifest")
    _non_authorizing(value, "v2 manifest")
    if (
        value.get("status") != "registered_pending_runner_and_synthetic_calibration"
        or value.get("study_role") != "retrospective_estimator_qualification_only"
        or value.get("all_molecular_outcomes_previously_exposed") is not True
        or value.get("prospective_lock_eligible") is not False
        or value.get("molecular_outcomes_opened_under_v2") is not False
        or value.get("supports_prospective_local_context_estimator_design") is not False
        or value.get("supports_intrinsic_localization_followup") is not False
    ):
        raise ValueError("v2 manifest must remain registered, unexecuted, and exposed")
    binding = _mapping(value.get("protocol_binding"), "protocol binding")
    if (
        binding
        != {
            "path": "configs/hest_nonlinear_qualification_v2.json",
            "sha256": V2_PROTOCOL_SHA256,
            "frozen_before_molecular_execution": True,
            "intermediate_results_may_modify_protocol": False,
        }
        or protocol_file_sha256 != V2_PROTOCOL_SHA256
    ):
        raise ValueError("v2 manifest protocol binding differs from actual frozen bytes")
    source = _mapping(value.get("registered_source"), "manifest registered source")
    if source != {
        "path": "/mnt/seagate/HEIR_runs/hest_hoptimus1_qualification/source.npz",
        "sha256": SOURCE_SHA256,
        "rows": 36121,
        "outcomes_previously_exposed": True,
    }:
        raise ValueError("v2 manifest source binding changed")
    artifacts = _mapping(value.get("outcome_blind_artifacts"), "manifest artifacts")
    if artifacts != {
        "blank_supplement_sha256": BLANK_SUPPLEMENT_SHA256,
        "blank_receipt_sha256": BLANK_RECEIPT_SHA256,
        "blank_feature_sha256": BLANK_FEATURE_SHA256,
        "registration_topology_sha256": TOPOLOGY_RECEIPT_SHA256,
        "selected_quality_band_id": "Q3",
        "selected_rows": 3991,
        "included_donor_ids": list(SELECTED_DONORS),
        "molecular_outcomes_read": False,
    }:
        raise ValueError("v2 manifest outcome-blind artifact binding changed")
    if value.get("experiment_ids") != [
        "B0",
        "B1",
        "B2",
        "B3",
        "N0",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
        "N7",
        "C_blank",
        "C_target_removed",
    ]:
        raise ValueError("v2 manifest experiment IDs changed")
    execution = _mapping(value.get("execution"), "manifest execution")
    if execution != {
        "preflight_authorized": True,
        "synthetic_calibration_authorized": True,
        "full_biological_execution_authorized_at_registration": False,
        "full_biological_execution_requires": [
            "clean_committed_full_runner",
            "all_preflight_hash_and_contract_checks_pass",
            "passing_200_trial_per_condition_synthetic_calibration_receipt",
            "calibration_receipt_records_molecular_outcomes_read_false",
        ],
        "biological_experiment_run": False,
        "engineering_decision_available": False,
        "final_report_available": False,
    }:
        raise ValueError("v2 manifest execution boundary changed")
    return value


def _passes(operator: str, value: float, threshold: float) -> bool:
    return {
        ">=": value >= threshold,
        "<=": value <= threshold,
        "<": value < threshold,
        ">": value > threshold,
    }[operator]


def _criteria(
    metrics: Mapping[str, Any], thresholds: Mapping[str, tuple[str, float]]
) -> list[Mapping[str, object]]:
    if set(metrics) != set(thresholds):
        raise ValueError("decision metrics differ from the frozen rule")
    result = []
    for name, (operator, threshold) in thresholds.items():
        raw = metrics[name]
        if raw is None:
            result.append(
                {
                    "metric": name,
                    "operator": operator,
                    "threshold": threshold,
                    "observed": None,
                    "passed": False,
                }
            )
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"decision metric {name} must be numeric")
        observed = float(raw)
        if not math.isfinite(observed):
            raise ValueError(f"decision metric {name} must be finite")
        result.append(
            {
                "metric": name,
                "operator": operator,
                "threshold": threshold,
                "observed": observed,
                "passed": _passes(operator, observed, threshold),
            }
        )
    return result


def evaluate_decisions(
    d1_metrics: Mapping[str, Any], d2_metrics: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, object]:
    """Evaluate D1 and D2 separately while preserving all biological claim boundaries."""

    d1 = _criteria(d1_metrics, D1_THRESHOLDS)
    d1_pass = all(bool(row["passed"]) for row in d1)
    if set(d2_metrics) != {"N3", "N4"}:
        raise ValueError("D2 requires exact N3 and N4 metric blocks")
    arm_results = {}
    for arm in ("N3", "N4"):
        criteria = _criteria(d2_metrics[arm], D2_THRESHOLDS)
        arm_results[arm] = {
            "criteria": criteria,
            "passed_all_arm_rules": all(bool(row["passed"]) for row in criteria),
        }
    d2_pass = d1_pass and any(bool(value["passed_all_arm_rules"]) for value in arm_results.values())
    return {
        "schema": REPORT_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "D1": {"criteria": d1, "passed": d1_pass},
        "D2": {"arms": arm_results, "D1_required": True, "passed": d2_pass},
        "supports_prospective_local_context_estimator_design": d1_pass,
        "supports_intrinsic_localization_followup": d2_pass,
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_h_ref": False,
        "authorizes_full_heir": False,
    }


def exact_sign_flip_max_test(
    donor_effects: Mapping[str, Mapping[str, float]],
) -> Mapping[str, object]:
    """Compute exact one-sided max-statistic adjusted p-values across one family."""

    if not donor_effects:
        raise ValueError("sign-flip family cannot be empty")
    comparison_ids = tuple(donor_effects)
    donor_sets = [set(values) for values in donor_effects.values()]
    if not donor_sets[0] or any(donors != donor_sets[0] for donors in donor_sets[1:]):
        raise ValueError("sign-flip family comparisons require identical donor IDs")
    donors = tuple(sorted(donor_sets[0]))
    if not 2 <= len(donors) <= 20:
        raise ValueError("exact sign-flip family requires between 2 and 20 donors")
    matrix = np.asarray(
        [[float(donor_effects[name][donor]) for donor in donors] for name in comparison_ids],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("sign-flip donor effects must be finite")
    observed = matrix.mean(axis=1)
    total = 1 << len(donors)
    exceed = np.zeros(len(comparison_ids), dtype=np.int64)
    chunk = 4096
    shifts = np.arange(len(donors), dtype=np.uint64)
    for start in range(0, total, chunk):
        integers = np.arange(start, min(start + chunk, total), dtype=np.uint64)
        bits = ((integers[:, None] >> shifts[None, :]) & 1).astype(np.float64)
        signs = bits * 2.0 - 1.0
        statistics = signs @ matrix.T / len(donors)
        maximum = statistics.max(axis=1)
        exceed += np.asarray(
            [np.count_nonzero(maximum >= value - 1.0e-15) for value in observed],
            dtype=np.int64,
        )
    adjusted = {name: float(exceed[index] / total) for index, name in enumerate(comparison_ids)}
    return {
        "method": "exact_donor_sign_flip_max_statistic",
        "donor_ids": list(donors),
        "sign_patterns": total,
        "observed_mean_effects": {
            name: float(observed[index]) for index, name in enumerate(comparison_ids)
        },
        "familywise_adjusted_p": adjusted,
    }


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if k >= n else 0.0
    return float(
        sum(
            math.comb(n, index) * probability**index * (1.0 - probability) ** (n - index)
            for index in range(k + 1)
        )
    )


def _clopper_pearson_one_sided(successes: int, trials: int, *, alpha: float) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("binomial confidence inputs are invalid")
    if successes == 0:
        lower = 0.0
    else:
        low, high = 0.0, 1.0
        for _ in range(80):
            middle = (low + high) / 2.0
            survival = 1.0 - _binomial_cdf(successes - 1, trials, middle)
            if survival > alpha:
                high = middle
            else:
                low = middle
        lower = (low + high) / 2.0
    if successes == trials:
        upper = 1.0
    else:
        low, high = 0.0, 1.0
        for _ in range(80):
            middle = (low + high) / 2.0
            cdf = _binomial_cdf(successes, trials, middle)
            if cdf > alpha:
                low = middle
            else:
                high = middle
        upper = (low + high) / 2.0
    return lower, upper


def evaluate_synthetic_calibration(
    pass_counts: Mapping[str, Mapping[str, int]],
    *,
    trials_per_condition: int = 200,
) -> Mapping[str, object]:
    """Apply simultaneous exact binomial bounds to all frozen D1/D2 expectations."""

    if set(pass_counts) != set(CALIBRATION_EXPECTATIONS) or trials_per_condition != 200:
        raise ValueError("synthetic calibration conditions or trial count changed")
    assertions = len(CALIBRATION_EXPECTATIONS) * 2
    alpha = 0.05 / assertions
    rows = []
    for condition, expectations in CALIBRATION_EXPECTATIONS.items():
        observed = _mapping(pass_counts[condition], condition)
        if set(observed) != {"D1", "D2"}:
            raise ValueError("each calibration condition requires D1 and D2 pass counts")
        for decision in ("D1", "D2"):
            count = observed[decision]
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError("calibration pass counts must be integers")
            lower, upper = _clopper_pearson_one_sided(count, trials_per_condition, alpha=alpha)
            expected_pass = expectations[decision]
            passed = lower > 0.80 if expected_pass else upper < 0.05
            rows.append(
                {
                    "condition": condition,
                    "decision": decision,
                    "expected_pass": expected_pass,
                    "passes": count,
                    "trials": trials_per_condition,
                    "simultaneous_one_sided_alpha": alpha,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "passed": passed,
                }
            )
    calibration_passed = all(bool(row["passed"]) for row in rows)
    return {
        "schema": "heir.hest_nonlinear_v2_synthetic_calibration.v1",
        "trials_per_condition": trials_per_condition,
        "simultaneous_family_assertions": assertions,
        "criteria": rows,
        "calibration_passed": calibration_passed,
        "HEST_molecular_execution_authorized": calibration_passed,
        "molecular_outcomes_read": False,
    }


__all__: Sequence[str] = (
    "BASE_PROTOCOL_SHA256",
    "BLANK_FEATURE_SHA256",
    "BLANK_RECEIPT_SHA256",
    "BLANK_SUPPLEMENT_SHA256",
    "CALIBRATION_EXPECTATIONS",
    "D1_THRESHOLDS",
    "D2_THRESHOLDS",
    "MANIFEST_SCHEMA",
    "PROTOCOL_SCHEMA",
    "REPORT_SCHEMA",
    "SELECTED_DONORS",
    "SOURCE_SHA256",
    "TOPOLOGY_RECEIPT_SHA256",
    "V2_PROTOCOL_SHA256",
    "evaluate_decisions",
    "evaluate_synthetic_calibration",
    "exact_sign_flip_max_test",
    "validate_manifest",
    "validate_protocol",
)
