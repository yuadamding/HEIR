"""Frozen contract for the exposed HEST nonlinear estimator qualification.

This module does not fit a model.  It validates the protocol/manifest pair and
turns prespecified engineering metrics into a bounded go/no-go decision.  A
positive decision may support designing a *new* prospective estimator
protocol, but the report is permanently unable to authorize a biological
hypothesis.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

PROTOCOL_SCHEMA = "heir.hest_nonlinear_qualification_protocol.v1"
MANIFEST_SCHEMA = "heir.hest_nonlinear_qualification_manifest.v1"
REPORT_SCHEMA = "heir.hest_nonlinear_qualification_report.v1"
ANALYSIS_STATUS = "retrospective_exposed_non_authorizing"

AUTHORIZATION_FIELDS = (
    "authorizes_h_cell",
    "authorizes_h_intrinsic",
    "authorizes_h_ref",
    "authorizes_full_heir",
)

NEURAL_CONTROL_FAMILIES = (
    "neural_reference_mean_only",
    "neural_combined_nuisance_only",
    "neural_image_only",
    "neural_combined_nuisance_plus_image",
    "neural_blank_patch",
    "neural_target_removed",
)

ARM_IDS = (
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
)

EXPECTED_ARMS = (
    ("B0", "none", "best_existing_non_image_ridge"),
    ("B1", "historical_form_96d_H_optimus_projection", "per_type_ridge"),
    ("B2", "full_1536d_feature", "per_type_ridge"),
    ("B3", "full_1536d_feature", "shared_linear_type_adapter"),
    ("N0", "combined_nuisance_features", "small_mlp"),
    ("N1", "natural_112um_crop", "small_mlp"),
    ("N2", "natural_112um_crop", "small_mlp_fine_type_adapter"),
    ("N3", "nucleus_mask_crop", "same_selected_mlp_family"),
    ("N4", "cell_mask_crop", "same_selected_mlp_family"),
    ("N5", "target_cell_removed_crop", "same_selected_mlp_family"),
    ("N6", "full_nucleus_cell_views", "late_fusion_mlp"),
    ("N7", "combined_nuisance_plus_n6_views", "late_fusion_mlp"),
)

SUPPORT_THRESHOLDS = {
    "donor_type_macro_residual_coordinate_r2": (">=", 0.05),
    "donor_section_type_macro_residual_coordinate_r2": (">=", 0.05),
    "improvement_over_b2_r2": (">=", 0.01),
    "improvement_over_n0_r2": (">=", 0.03),
    "molecular_error_reduction_over_reference_mean": (">=", 0.05),
    "positive_supported_donor_type_strata_fraction": (">=", 0.80),
    "positive_donors_versus_n0_fraction": (">=", 0.80),
    "maximum_single_donor_gain_fraction": ("<", 0.50),
    "molecular_variance_ratio": (">=", 0.50),
    "median_type_coverage": (">=", 0.50),
    "abstention_rate": ("<=", 0.50),
    "rare_state_recall_drop": ("<=", 0.20),
    "within_section_type_refitted_null_empirical_p": ("<=", 0.01),
    "different_spatial_block_refitted_null_empirical_p": ("<=", 0.01),
    "intrinsic_increment_over_target_removed_r2": (">=", 0.01),
    "best_registration_minus_all_rows_r2": (">=", -0.01),
}

FROZEN_PREDECESSOR_SHA256 = {
    "scripts/benchmark_hest_retrospective.py": (
        "853a2801e94e3c727631d4e3eb8e0a0ba766bd3768b3302b4219ce14a61c9878"
    ),
    "scripts/benchmark_morphology_state_gate.py": (
        "5975f2cc0ff586a89f39f78bbf989f1ecd1e03347ea9fba9b74673b316385919"
    ),
    "src/heir/evaluation/morphology_gate.py": (
        "4c92b3c853203cf9e580db311536cf428283e41fd473436b8a6f3d4be80b3cad"
    ),
    "src/heir/evaluation/reference_fusion_v2.py": (
        "894ad3de42cc3d253aee400cb17cd7ff2257c2aca016601d7eb0f2bc2136bf3a"
    ),
    "reports/hest_retrospective.md": (
        "1e833867cef32504f1f0f2d4d52471e9ff17ee992cfd8c4e914e347495a95027"
    ),
    "reports/hest_scientific_reanalysis.md": (
        "2c571cc3e611466dd0c0f957cb738aa9ffa319a93562c4519bdfe02b412c733d"
    ),
    "reports/natcommun_hoptimus_primary_results_v2.md": (
        "b4771b6d10dfbbe22cc4ecf873435cf370fb117a608ab9e6982634bde7374af7"
    ),
}

PROPORTION_METRICS = frozenset(
    {
        "positive_supported_donor_type_strata_fraction",
        "positive_donors_versus_n0_fraction",
        "maximum_single_donor_gain_fraction",
        "median_type_coverage",
        "abstention_rate",
        "within_section_type_refitted_null_empirical_p",
        "different_spatial_block_refitted_null_empirical_p",
    }
)
SIGNED_UNIT_METRICS = frozenset({"rare_state_recall_drop"})


def canonical_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 for a JSON-compatible value."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % name)
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], name: str) -> None:
    if set(value) != set(expected):
        raise ValueError("%s fields differ from the frozen contract" % name)


def _require_non_authorizing(value: Mapping[str, Any], name: str) -> None:
    if value.get("analysis_status") != ANALYSIS_STATUS:
        raise ValueError("%s must remain %s" % (name, ANALYSIS_STATUS))
    for field in AUTHORIZATION_FIELDS:
        if value.get(field) is not False:
            raise ValueError("%s.%s must be false" % (name, field))


def _normalized_thresholds(value: Mapping[str, Any]) -> Dict[str, tuple]:
    normalized: Dict[str, tuple] = {}
    for metric, rule in value.items():
        rule_mapping = _mapping(rule, "engineering_support_rule.%s" % metric)
        _exact_keys(rule_mapping, ("operator", "threshold"), metric)
        operator = rule_mapping["operator"]
        threshold = rule_mapping["threshold"]
        if operator not in {">=", "<=", "<"}:
            raise ValueError("unsupported threshold operator for %s" % metric)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold for %s must be numeric" % metric)
        if not math.isfinite(float(threshold)):
            raise ValueError("threshold for %s must be finite" % metric)
        normalized[str(metric)] = (str(operator), float(threshold))
    return normalized


def validate_protocol(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the immutable scientific fields of qualification protocol v1."""

    value = dict(_mapping(protocol, "protocol"))
    if value.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported nonlinear qualification protocol schema")
    _require_non_authorizing(value, "protocol")

    if value.get("protocol_id") != "hest_nonlinear_qualification_v1":
        raise ValueError("protocol_id differs from the frozen qualification")
    if value.get("study_role") != "retrospective_estimator_qualification_only":
        raise ValueError("study_role may not be prospective or confirmatory")
    if value.get("outcome_exposure") != "all_hest_outcomes_previously_exposed":
        raise ValueError("protocol must disclose complete prior outcome exposure")
    if value.get("supports_new_prospective_estimator_protocol") is not False:
        raise ValueError("an unexecuted protocol cannot claim engineering support")

    predecessors = _mapping(value.get("frozen_predecessor_evidence"), "frozen_predecessor_evidence")
    if dict(predecessors) != FROZEN_PREDECESSOR_SHA256:
        raise ValueError("frozen predecessor identities differ from the registered evidence")

    encoder = _mapping(value.get("encoder_scope"), "encoder_scope")
    if encoder.get("primary") != "bioptimus/H-optimus-1":
        raise ValueError("H-optimus-1 must remain the only executed image encoder")
    if encoder.get("frozen_no_fine_tuning") is not True:
        raise ValueError("encoder fine-tuning is outside nonlinear qualification v1")
    if encoder.get("UNI2_h") != ("historical_frozen_evidence_only_not_executed_in_this_protocol"):
        raise ValueError("UNI2-h may not execute under this protocol")

    matrix = value.get("experiment_matrix")
    if not isinstance(matrix, list) or len(matrix) != len(EXPECTED_ARMS):
        raise ValueError("experiment_matrix differs from the frozen 12-arm ladder")
    observed = []
    for index, arm in enumerate(matrix):
        row = _mapping(arm, "experiment_matrix[%d]" % index)
        _exact_keys(row, ("id", "image_representation", "estimator", "question"), "arm")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("every experiment arm requires a non-empty question")
        observed.append((row["id"], row["image_representation"], row["estimator"]))
    if tuple(observed) != EXPECTED_ARMS:
        raise ValueError("experiment arm order or identity differs from the frozen matrix")

    thresholds = _normalized_thresholds(
        _mapping(value.get("engineering_support_rule"), "engineering_support_rule")
    )
    if thresholds != SUPPORT_THRESHOLDS:
        raise ValueError("engineering support thresholds differ from the frozen contract")

    nulls = _mapping(value.get("refitted_nulls"), "refitted_nulls")
    if nulls.get("families") != [
        "within_section_type_image_derangement",
        "different_spatial_block_reassignment",
    ]:
        raise ValueError("both frozen refitted-null families are required")
    if nulls.get("smoke_permutations_per_family") != 20:
        raise ValueError("smoke qualification requires exactly 20 permutations")
    if nulls.get("final_permutations_per_family") != 100:
        raise ValueError("final qualification requires exactly 100 permutations")
    if nulls.get("complete_refit_required") is not True:
        raise ValueError("nulls must repeat selection, fitting, and scoring")

    target = _mapping(value.get("molecular_target"), "molecular_target")
    required_target_steps = [
        "same_section_spatially_independent_donor_section_type_reference_mean",
        "development_fold_only_technical_correction",
        "development_fold_only_type_specific_low_rank_molecular_basis",
        "predict_residual_coordinates",
        "reconstruct_measured_target_genes",
    ]
    if target.get("ordered_steps") != required_target_steps:
        raise ValueError("molecular target differs from the existing HEIR residual construction")
    if target.get("raw_whole_transcriptome_primary_target") is not False:
        raise ValueError("raw whole-transcriptome prediction is prohibited in v1")
    if target.get("rank_candidates") != [2, 4, 6]:
        raise ValueError("molecular rank grid must remain [2, 4, 6]")
    if target.get("minimum_basis_ceiling_r2") != 0.3:
        raise ValueError("molecular basis ceiling must remain at least 0.30")
    if target.get("coordinate_standardization") != (
        "donor_type_weighted_inner_or_outer_training_mean_and_variance_then_inverse_"
        "transform_before_scoring"
    ):
        raise ValueError("neural coordinate standardization differs from the frozen protocol")
    if target.get("inner_selection_target_fit_scope") != (
        "each_inner_training_donor_subset_only_never_inner_validation_or_outer_heldout_rows"
    ):
        raise ValueError("inner-fold target fitting must remain inner-training-only")
    if target.get("outer_refit_target_scope") != (
        "all_outer_training_donors_after_inner_rank_selection"
    ):
        raise ValueError("outer target refit must remain outer-training-only")

    validation = _mapping(value.get("validation_design"), "validation_design")
    if validation.get("outer_split") != "leave_one_biological_donor_out":
        raise ValueError("outer evaluation must leave one biological donor out")
    if validation.get("inner_split") != "leave_one_training_donor_out":
        raise ValueError("model selection must be donor-held-out")
    if validation.get("outer_donor_may_affect_training_or_selection") is not False:
        raise ValueError("outer donor leakage is prohibited")
    if validation.get("primary_minimum_donor_section_type_support") != 20:
        raise ValueError("primary hierarchical support must remain 20 cells")
    if (
        validation.get("best_registration_primary_minimum_support") != 20
        or validation.get("best_registration_supported_donor_type_strata_at_primary_support") != 0
        or validation.get("best_registration_current_evaluability") != "unavailable_fail_closed"
    ):
        raise ValueError("best-registration sensitivity must remain unavailable and fail closed")

    controls = _mapping(value.get("architecture_matched_controls"), "architecture_matched_controls")
    if controls.get("required_families") != list(NEURAL_CONTROL_FAMILIES):
        raise ValueError("architecture-matched neural control families differ from the registry")
    nuisance = _mapping(
        controls.get("combined_nuisance_representation"),
        "combined_nuisance_representation",
    )
    if (
        nuisance.get("source_key") != "all_controls"
        or nuisance.get("dimensions") != 158
        or nuisance.get("full_nuisance_covariates_31d_is_not_a_substitute") is not True
    ):
        raise ValueError("combined nuisance must be the frozen deduplicated 158D all_controls")
    blank = _mapping(controls.get("blank_patch_input"), "blank_patch_input")
    if (
        blank.get("status") != "blocked_missing_feature_supplement"
        or blank.get("source_availability") != "unavailable_in_registered_source"
        or blank.get("may_synthesize_or_reuse_zero_vector") is not False
    ):
        raise ValueError("missing blank-patch features must block execution without fabrication")

    baselines = _mapping(value.get("linear_baselines"), "linear_baselines")
    if baselines.get("B0_feature_candidates") != [
        "technical",
        "spatial",
        "stain_qc",
        "morphometry_density",
        "combined_nonimage",
    ]:
        raise ValueError("B0 control-family candidates differ from the frozen registry")
    projection = _mapping(baselines.get("B1_projection"), "B1_projection")
    if projection != {
        "input_dimensions": 1536,
        "output_dimensions": 96,
        "distribution": "Rademacher_plus_or_minus_one_over_sqrt_96",
        "seed": 20260720,
        "outcome_independent_and_shared_across_outer_folds": True,
    }:
        raise ValueError("B1 historical-form projection differs from the frozen construction")
    if baselines.get("ridge_alpha_candidates") != [0.0001, 0.01, 1.0, 100.0]:
        raise ValueError("ridge alpha grid differs from the frozen protocol")

    architectures = _mapping(value.get("architectures"), "architectures")
    if set(architectures) != {
        "input_width_policy",
        "shared_linear_type_adapter",
        "mlp_tiny",
        "mlp_small",
        "fine_type_adapter",
        "late_fusion",
    }:
        raise ValueError("architecture ladder differs from the frozen protocol")
    if architectures.get("input_width_policy") != (
        "use_the_registered_arm_width_image_1536_nuisance_158_or_concatenated_width"
    ):
        raise ValueError("single-view input-width policy differs from the frozen protocol")
    linear_adapter = _mapping(
        architectures["shared_linear_type_adapter"], "shared_linear_type_adapter"
    )
    if dict(linear_adapter) != {
        "shared_component": "Linear(input_width,rank)",
        "type_specific_component": "rank_8_low_rank_linear_adapter",
        "separate_network_per_type": False,
    }:
        raise ValueError("shared linear type adapter differs from the frozen architecture")
    tiny = _mapping(architectures["mlp_tiny"], "mlp_tiny")
    small = _mapping(architectures["mlp_small"], "mlp_small")
    adapter = _mapping(architectures["fine_type_adapter"], "fine_type_adapter")
    fusion = _mapping(architectures["late_fusion"], "late_fusion")
    if tiny.get("layers") != [
        "LayerNorm(input_width)",
        "Linear(input_width,64)",
        "GELU",
        "Dropout(0.0)",
        "Linear(64,rank)",
    ]:
        raise ValueError("mlp_tiny differs from the frozen architecture")
    if small.get("layers") != [
        "LayerNorm(input_width)",
        "Linear(input_width,256)",
        "GELU",
        "Dropout(0.2)",
        "Linear(256,64)",
        "GELU",
        "Linear(64,rank)",
    ]:
        raise ValueError("mlp_small differs from the frozen architecture")
    if (
        adapter.get("embedding_width") != 16
        or adapter.get("adapter_rank") != 8
        or adapter.get("conditioning") != "FiLM_scale_and_shift_on_first_hidden_layer"
    ):
        raise ValueError("fine-type adapter differs from the frozen architecture")
    if (
        fusion.get("per_view")
        != ["LayerNorm(view_width)", "Linear(view_width,64)", "GELU"]
        or fusion.get("N6_input_dimensions") != [1536, 1536, 1536]
        or fusion.get("N6_input_views")
        != [
            "crop_112um",
            "nucleus_mask_only",
            "cell_mask_only",
        ]
        or fusion.get("N7_input_views")
        != [
            "deduplicated_all_controls_158d",
            "crop_112um",
            "nucleus_mask_only",
            "cell_mask_only",
        ]
        or fusion.get("N7_input_dimensions") != [158, 1536, 1536, 1536]
        or fusion.get("target_cell_removed_is_focal_input") is not False
    ):
        raise ValueError("late-fusion inputs differ from the frozen architecture")

    training = _mapping(value.get("training_and_selection"), "training_and_selection")
    expected_training = {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay_candidates": [0.0001, 0.01],
        "maximum_epochs": 100,
        "batch_size": 256,
        "inner_fold_early_stopping_patience": 10,
        "fixed_seeds": [17, 29, 41],
        "gradient_clipping_norm": 1.0,
        "loss": "donor_type_weighted_latent_coordinate_MSE",
        "selection_metric": "donor_balanced_validation_R2_not_cell_weighted_validation_loss",
        "inner_procedure": (
            "leave_one_outer_training_donor_out_fit_normalization_and_target_on_inner_training_"
            "then_select_checkpoint_and_configuration_on_the_inner_validation_donor_never_the_"
            "outer_donor"
        ),
        "outer_refit_epochs": "median_selected_epoch_count_across_successful_inner_folds",
        "outer_refit_scope": "all_outer_training_donors_only",
        "seed_aggregation": "average_three_seed_predictions",
        "per_seed_reporting": True,
        "configuration_failure_action": "reject_if_any_inner_donor_fold_fails",
    }
    if dict(training) != expected_training:
        raise ValueError("training and selection settings differ from the frozen protocol")
    diagnostics = _mapping(value.get("prediction_diagnostics"), "prediction_diagnostics")
    if diagnostics.get("coverage_policy") != "dense_prediction_no_post_hoc_OOD_abstention_v1":
        raise ValueError("prediction coverage policy differs from the frozen protocol")
    if diagnostics.get("rare_state_recall_drop") != "B2_recall_minus_candidate_recall":
        raise ValueError("rare-state comparator differs from the frozen protocol")
    intrinsic = _mapping(value.get("intrinsic_increment_definition"), "intrinsic_increment")
    if intrinsic.get("registered_statistic") != (
        "minimum_of_N1_N3_N4_paired_donor_type_macro_r2_increments_over_N5"
    ):
        raise ValueError("target-removed engineering contrast differs from the frozen protocol")
    if value.get("model_selection_tie_breaking") != [
        "reject_configurations_that_fail_any_inner_donor_fold",
        "reject_configurations_with_minimum_inner_fold_basis_ceiling_below_0.3",
        "reject_configurations_with_median_variance_ratio_below_0.5",
        "maximize_donor_section_type_macro_R2",
        "then_maximize_donor_type_macro_R2",
        "then_choose_fewer_parameters",
        "then_choose_larger_weight_decay",
        "then_choose_lexicographically_smaller_model_id",
    ]:
        raise ValueError("model-selection tie breaking differs from the frozen protocol")

    execution = _mapping(value.get("execution_state"), "execution_state")
    if execution.get("execution_authorized") is not False or execution.get("blockers") != [
        "registered_source_has_no_blank_patch_embedding",
        "best_registration_subset_has_zero_supported_donor_type_strata_at_primary_support_20",
    ]:
        raise ValueError("currently unavailable inputs must block v1 execution")
    expected_execution_status = {
        "neural_probe_implemented": True,
        "architecture_matched_controls_and_nulls_implemented": True,
        "preflight_and_synthetic_smoke_runner_implemented": True,
        "full_biological_runner_implemented": False,
        "synthetic_smoke_complete": True,
        "biological_experiment_run": False,
        "engineering_decision_available": False,
    }
    if any(
        execution.get(field) is not expected
        for field, expected in expected_execution_status.items()
    ):
        raise ValueError("implementation status differs from the blocked synthetic-only state")

    return value


def validate_retrospective_manifest(
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    protocol_file_sha256: str,
) -> Dict[str, Any]:
    """Validate that the exposed-study manifest binds the exact protocol bytes."""

    validate_protocol(protocol)
    value = dict(_mapping(manifest, "manifest"))
    if value.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported nonlinear qualification manifest schema")
    _require_non_authorizing(value, "manifest")
    if value.get("study_id") != "hest_nonlinear_qualification_v1":
        raise ValueError("manifest study_id differs from the qualification")
    if value.get("status") != ("blocked_missing_feature_supplement_and_registration_support"):
        raise ValueError("manifest must remain blocked and must not claim biological execution")
    if value.get("study_role") != "retrospective_estimator_qualification_only":
        raise ValueError("manifest cannot assign a confirmatory role")
    if value.get("all_molecular_outcomes_previously_exposed") is not True:
        raise ValueError("manifest must record prior access to all molecular outcomes")
    if value.get("prospective_lock_eligible") is not False:
        raise ValueError("exposed HEST data cannot regain prospective eligibility")
    if value.get("supports_new_prospective_estimator_protocol") is not False:
        raise ValueError("an unexecuted manifest cannot claim engineering support")

    binding = _mapping(value.get("protocol_binding"), "protocol_binding")
    if binding.get("path") != "configs/hest_nonlinear_qualification_v1.json":
        raise ValueError("manifest points to the wrong protocol")
    if binding.get("sha256") != protocol_file_sha256:
        raise ValueError("manifest protocol hash differs from the actual file bytes")
    if binding.get("frozen_before_execution") is not True:
        raise ValueError("protocol must be frozen before qualification execution")
    if value.get("experiment_arm_ids") != list(ARM_IDS):
        raise ValueError("manifest arm IDs differ from the protocol")
    source = _mapping(value.get("registered_source"), "registered_source")
    if (
        source.get("blank_patch_embedding_available") is not False
        or source.get("blank_patch_policy") != "blocked_missing_feature_supplement_never_fabricate"
    ):
        raise ValueError("manifest must fail closed on the missing blank-patch embedding")
    execution = _mapping(value.get("execution"), "execution")
    if execution.get("authorized") is not False or execution.get("blockers") != [
        "registered_source_missing_receipt_bound_blank_patch_embedding",
        "best_registration_subset_has_zero_donor_type_strata_at_primary_support_20",
    ]:
        raise ValueError("manifest execution must remain blocked on registered inputs")
    if (
        execution.get("implementation_available") is not True
        or execution.get("full_biological_runner_available") is not False
        or execution.get("smoke_run_complete") is not True
        or execution.get("final_run_complete") is not False
        or execution.get("biological_experiment_run") is not False
        or execution.get("engineering_decision_available") is not False
        or execution.get("report_available") is not True
        or execution.get("report_scope")
        != "synthetic_implementation_smoke_no_biological_rows"
    ):
        raise ValueError("manifest must distinguish the synthetic smoke from biological execution")
    return value


def _passes(operator: str, value: float, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == "<=":
        return value <= threshold
    if operator == "<":
        return value < threshold
    raise ValueError("unsupported operator: %s" % operator)


def non_authorizing_report_fields(
    supports_new_prospective_estimator_protocol: bool,
) -> Dict[str, bool]:
    """Build the only authorization block permitted for this qualification."""

    if not isinstance(supports_new_prospective_estimator_protocol, bool):
        raise ValueError("engineering support decision must be boolean")
    return {
        "supports_new_prospective_estimator_protocol": (
            supports_new_prospective_estimator_protocol
        ),
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_h_ref": False,
        "authorizes_full_heir": False,
    }


def evaluate_engineering_support(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate every frozen engineering threshold without biological promotion."""

    values = _mapping(metrics, "metrics")
    if set(values) != set(SUPPORT_THRESHOLDS):
        raise ValueError("metrics must contain exactly the frozen support-rule fields")
    criteria = []
    for metric, (operator, threshold) in SUPPORT_THRESHOLDS.items():
        raw = values[metric]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("%s must be numeric" % metric)
        observed = float(raw)
        if not math.isfinite(observed):
            raise ValueError("%s must be finite" % metric)
        if metric in PROPORTION_METRICS and not 0.0 <= observed <= 1.0:
            raise ValueError("%s must lie in [0, 1]" % metric)
        if metric in SIGNED_UNIT_METRICS and not -1.0 <= observed <= 1.0:
            raise ValueError("%s must lie in [-1, 1]" % metric)
        criteria.append(
            {
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "observed": observed,
                "pass": _passes(operator, observed, threshold),
            }
        )
    supported = all(bool(row["pass"]) for row in criteria)
    return {
        "schema": REPORT_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "criteria": criteria,
        **non_authorizing_report_fields(supported),
    }


def validate_report_authorization(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Reject any attempt to turn an exposed qualification into biological evidence."""

    value = dict(_mapping(report, "report"))
    if value.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported nonlinear qualification report schema")
    _require_non_authorizing(value, "report")
    if not isinstance(value.get("supports_new_prospective_estimator_protocol"), bool):
        raise ValueError("report engineering-support decision must be boolean")
    return value


__all__ = [
    "ANALYSIS_STATUS",
    "ARM_IDS",
    "AUTHORIZATION_FIELDS",
    "FROZEN_PREDECESSOR_SHA256",
    "MANIFEST_SCHEMA",
    "PROTOCOL_SCHEMA",
    "REPORT_SCHEMA",
    "SUPPORT_THRESHOLDS",
    "canonical_sha256",
    "evaluate_engineering_support",
    "file_sha256",
    "non_authorizing_report_fields",
    "validate_protocol",
    "validate_report_authorization",
    "validate_retrospective_manifest",
]
