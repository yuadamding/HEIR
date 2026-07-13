"""Fail-closed authorization for exact morphology-gate calibration evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

CALIBRATION_RECEIPT_SCHEMA = "heir.morphology_gate_calibration.v3"
CALIBRATION_ENGINE = "heir.actual_morphology_gate.v3"
ACTUAL_GATE_ENTRYPOINT = "heir.evaluation.morphology_gate.evaluate_morphology_ridge_gate"
ACTUAL_GATE_REPORT_SCHEMA = "heir.morphology_ridge_evaluation.v5"
CALIBRATION_EVIDENCE_SCHEMA = "heir.actual_morphology_gate_calibration_evidence.v3"
CALIBRATION_RUN_CONTRACT_SCHEMA = "heir.morphology_gate_calibration_run_contract.v3"
CALIBRATION_DGP_SPEC_SCHEMA = "heir.morphology_gate_calibration_dgp.v3"
CALIBRATION_GENERATOR_VERSION = "heir.synthetic_morphology_gate_generator.v3"
GLOBAL_NULL_CONDITION = "global_null"
PRELIMINARY_ALTERNATIVE_CONDITION = "preliminary_full_shared_latent"
PENDING_DESIGN_BINDING_STATUS = "pending_pre_h_meas"
COMPLETE_DESIGN_BINDING_STATUS = "complete"

REQUIRED_CALIBRATION_SCENARIOS = (
    "spatial_autocorrelation",
    "disease_imbalance",
    "section_effects",
    "missing_fine_types",
    "variable_transcript_reliability",
    "unbalanced_donor_cell_counts",
    "inactive_permutation_strata",
    "nuisance_selection",
    "target_panel_selection",
    "crop_family_multiplicity",
)

# These are the confirmatory H-CELL checks returned by the actual gate.  G3
# decisions are bound separately because they are hypothesis decisions rather
# than members of ``report["checks"]``.
REQUIRED_COMPLETE_GATE_CHECKS = (
    "primary_claim_is_explicit_local_context",
    "matched_macro_r2",
    "macro_donor_type_r2",
    "local_roi_null_separates",
    "spatial_block_null_separates",
    "every_required_null_separates",
    "permutations_change_training_rows",
    "supported_donor_type_coverage",
    "positive_supported_strata",
    "donor_consistency",
    "not_single_donor_driven",
    "beats_coordinate_only",
    "paired_coordinate_effect_ci_positive",
    "beats_best_independently_tuned_nuisance",
    "paired_best_nuisance_effect_ci_positive",
    "exact_donor_paired_main_effect",
    "matched_donor_bootstrap_ci_positive",
    "expression_relevance",
    "adequate_basis_ceiling",
    "rank_direction_stable",
    "reference_split_direction_stable",
    "planned_coverage_retained",
    "disease_inclusive_endpoint_reported",
    "disease_adjusted_or_single_disease_endpoint_reported",
    "source_coverage_audit_available",
    "reference_evaluation_balance_passes",
    "locked_measurement_audit_passes",
)

REQUIRED_HYPOTHESIS_DECISIONS = (
    "G2_local_context",
    "G3_nucleus_intrinsic",
    "G3_cell_intrinsic",
    "G3_context_only",
    "G3_mixed_intrinsic_context",
)
BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS = {
    decision_id: "quantitative_boundary::%s" % decision_id
    for decision_id in REQUIRED_HYPOTHESIS_DECISIONS
}
BOUNDARY_EXPECTED_SOURCE_CONCLUSION = {
    "G2_local_context": "multiple_sources_without_incremental_combination",
    "G3_nucleus_intrinsic": "nucleus_dominant",
    "G3_cell_intrinsic": "cell_dominant",
    "G3_context_only": "context_dominant",
    "G3_mixed_intrinsic_context": "mixed_intrinsic_and_contextual_information",
}

REQUIRED_CROP_FAMILY_IDS = (
    "crop_112um",
    "nucleus_mask_only",
    "nucleus_mask_mean_fill_112um",
    "nucleus_mask_blurred_112um",
    "nucleus_shape_random_location_mean_fill_112um",
    "cell_mask_only",
    "cell_mask_mean_fill_112um",
    "cell_mask_blurred_112um",
    "cell_shape_random_location_mean_fill_112um",
    "context_ring_32_to_112um",
    "context_ring_64_to_112um",
    "target_cell_removed_112um",
    "target_cell_removed_mean_fill_112um",
    "target_cell_removed_blurred_112um",
    "random_location_cell_removed_mean_fill_112um",
    "crop_32um",
    "crop_64um",
    "blank_patch",
)

REQUIRED_NUISANCE_FAMILIES = (
    "reference_mean_only",
    "technical_only",
    "coordinate_only",
    "spatial_only",
    "local_density_only",
    "boundary_only",
    "stain_only",
    "nuclear_morphometrics_only",
    "cell_morphometrics_only",
    "cellvit_context_only",
    "disease_site_batch_only",
    "disease_site_batch_section_only",
    "combined_nuisance_only",
)

REQUIRED_PERMUTATION_TRANSFORMS = (
    "development_donor_type_roi_permutation_with_selection_and_refit",
    "development_donor_type_spatial_block_permutation_with_selection_and_refit",
)

G2_MULTIPLICITY_METHOD = "intersection_union_closed_testing_all_frozen_nuisance_families"
G3_MULTIPLICITY_METHOD = "exact_donor_sign_flip_max_statistic"
REQUIRED_G3_CONTRAST_PAIRS = {
    "nucleus_white_vs_random_shape": [
        "nucleus_mask_image",
        "crop_image::nucleus_shape_random_location_mean_fill_112um",
    ],
    "nucleus_white_vs_blurred_fill": [
        "nucleus_mask_image",
        "crop_image::nucleus_mask_blurred_112um",
    ],
    "nucleus_mean_vs_random_shape": [
        "crop_image::nucleus_mask_mean_fill_112um",
        "crop_image::nucleus_shape_random_location_mean_fill_112um",
    ],
    "nucleus_mean_vs_blurred_fill": [
        "crop_image::nucleus_mask_mean_fill_112um",
        "crop_image::nucleus_mask_blurred_112um",
    ],
    "cell_white_vs_random_shape": [
        "cell_mask_image",
        "crop_image::cell_shape_random_location_mean_fill_112um",
    ],
    "cell_white_vs_blurred_fill": [
        "cell_mask_image",
        "crop_image::cell_mask_blurred_112um",
    ],
    "cell_mean_vs_random_shape": [
        "crop_image::cell_mask_mean_fill_112um",
        "crop_image::cell_shape_random_location_mean_fill_112um",
    ],
    "cell_mean_vs_blurred_fill": [
        "crop_image::cell_mask_mean_fill_112um",
        "crop_image::cell_mask_blurred_112um",
    ],
    "context_white_vs_random_location": [
        "target_cell_removed_context_image",
        "crop_image::random_location_cell_removed_mean_fill_112um",
    ],
    "context_white_vs_blurred_fill": [
        "target_cell_removed_context_image",
        "crop_image::target_cell_removed_blurred_112um",
    ],
    "context_mean_vs_random_location": [
        "crop_image::target_cell_removed_mean_fill_112um",
        "crop_image::random_location_cell_removed_mean_fill_112um",
    ],
    "context_mean_vs_blurred_fill": [
        "crop_image::target_cell_removed_mean_fill_112um",
        "crop_image::target_cell_removed_blurred_112um",
    ],
    "full_context_vs_target_removed_white": [
        "primary_local_context_image",
        "target_cell_removed_context_image",
    ],
    "full_context_vs_target_removed_mean": [
        "primary_local_context_image",
        "crop_image::target_cell_removed_mean_fill_112um",
    ],
    "full_context_vs_nucleus_white": [
        "primary_local_context_image",
        "nucleus_mask_image",
    ],
    "full_context_vs_nucleus_mean": [
        "primary_local_context_image",
        "crop_image::nucleus_mask_mean_fill_112um",
    ],
    "full_context_vs_cell_white": [
        "primary_local_context_image",
        "cell_mask_image",
    ],
    "full_context_vs_cell_mean": [
        "primary_local_context_image",
        "crop_image::cell_mask_mean_fill_112um",
    ],
}
REQUIRED_MORPHOLOGY_SOURCE_CONCLUSIONS = (
    "nucleus_dominant",
    "cell_dominant",
    "context_dominant",
    "mixed_intrinsic_and_contextual_information",
    "multiple_sources_without_incremental_combination",
    "no_morphology_specific_information",
)
# The production gate can fail closed before it can assign one of the six
# scientific conclusions.  Calibration must count those outcomes as errors,
# not abort and silently discard the trial.
CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES = (
    *REQUIRED_MORPHOLOGY_SOURCE_CONCLUSIONS,
    "inconclusive",
    "not_tested",
)
REQUIRED_GATE_PARAMETERS = {
    "minimum_final_permutations": 999,
    "minimum_support": 20,
    "minimum_development_donors": 3,
    "minimum_locked_donors": 3,
    "minimum_macro_r2": 0.05,
    "minimum_shuffle_delta": 0.03,
    "minimum_coordinate_delta": 0.01,
    "minimum_stain_delta": 0.01,
    "maximum_direct_contrast_p": 0.05,
    "minimum_mask_implementation_pass_fraction": 1.0,
    "minimum_null_shuffled_fraction": 0.95,
    "minimum_strata_coverage": 0.8,
    "maximum_permutation_p": 0.01,
    "minimum_positive_strata_fraction": 0.8,
    "minimum_expression_error_reduction": 0.05,
    "minimum_basis_ceiling_r2": 0.3,
    "donor_bootstrap_iterations": 2000,
    "donor_bootstrap_seed": 17,
    "prespecified_fixed_hyperparameters": False,
}


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible scientific contract deterministically."""

    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def current_calibration_executable_provenance() -> Mapping[str, str]:
    """Hash the complete HEIR package and dependency lock used by calibration."""

    directory = Path(__file__).resolve().parent
    paths = {
        "gate_source_sha256": directory / "morphology_gate.py",
        "compiler_source_sha256": directory / "morphology_calibration.py",
        "generator_source_sha256": directory / "morphology_calibration_runner.py",
        "contract_source_sha256": directory / "power.py",
    }
    package_root = directory.parent
    source_tree = {
        str(path.relative_to(package_root.parent)): _source_file_sha256(path)
        for path in sorted(package_root.rglob("*.py"))
    }
    repository_root = package_root.parents[1]
    lock_files = {
        name: _source_file_sha256(repository_root / name) for name in ("pyproject.toml", "uv.lock")
    }
    return {
        **{name: _source_file_sha256(path) for name, path in paths.items()},
        "scientific_source_tree_sha256": canonical_sha256(source_tree),
        "dependency_lock_sha256": canonical_sha256(lock_files),
    }


_SCIENTIFIC_MANIFEST_PROJECTION_FIELDS = (
    "schema",
    "study_id",
    "hypothesis_ids",
    "analysis_plan_sha256",
    "dataset",
    "partitions",
    "observations",
    "encoder",
    "crop_protocols",
    "reference_splits",
    "candidate_target_gene_panel_sha256",
    "target_gene_panel_sha256",
    "type_marker_panel_sha256",
    "prerequisites",
    "lock_protection",
    "label_target_independence",
    "technical_covariates",
    "controls",
    "hyperparameter_grid",
    "randomization",
    "primary_endpoint",
    "secondary_endpoints",
    "coverage_requirements",
    "decision_thresholds",
    "morphology_gate",
    "locked_measurement_audit",
)


def confirmatory_scientific_manifest_projection(
    content: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the outcome-free H-CELL settings projection bound by calibration.

    Mutable lifecycle fields and the calibration receipt hash are deliberately
    excluded, avoiding a circular hash while retaining every scientific choice.
    """

    missing = [name for name in _SCIENTIFIC_MANIFEST_PROJECTION_FIELDS if name not in content]
    if missing:
        raise ValueError(
            "confirmatory manifest lacks scientific projection fields: %s" % ", ".join(missing)
        )
    projection = {name: content[name] for name in _SCIENTIFIC_MANIFEST_PROJECTION_FIELDS}
    gate = projection["morphology_gate"]
    if not isinstance(gate, Mapping):
        raise ValueError("confirmatory manifest morphology_gate is malformed")
    projection = dict(projection)
    projection["morphology_gate"] = {
        str(name): value for name, value in gate.items() if name != "calibration_receipt_sha256"
    }
    return projection


def planned_donor_type_support_pattern_sha256(
    development_donor_ids: Sequence[str],
    locked_test_donor_ids: Sequence[str],
    supported_fine_type_ids: Sequence[str],
) -> str:
    """Hash the complete pre-outcome donor-by-type intention-to-analyze pattern."""

    rows = [
        {"role": role, "donor_id": donor, "fine_type_id": fine_type}
        for role, donors in (
            ("development", development_donor_ids),
            ("locked_test", locked_test_donor_ids),
        )
        for donor in donors
        for fine_type in supported_fine_type_ids
    ]
    return canonical_sha256(rows)


def pending_confirmatory_design_binding() -> Mapping[str, object]:
    """Return the only pending binding accepted in a checked-in pre-H-MEAS config."""

    return {
        "status": PENDING_DESIGN_BINDING_STATUS,
        "reason": (
            "H-MEAS has not yet issued the development-only target-panel and supported-type "
            "receipt; authorizing calibration is forbidden"
        ),
    }


def validate_confirmatory_design_binding(
    value: object,
    *,
    allow_pending: bool = False,
) -> Mapping[str, object]:
    """Validate the completed, pre-outcome scientific design identity."""

    if not isinstance(value, Mapping):
        raise ValueError("calibration confirmatory_design_binding must be an object")
    if value.get("status") == PENDING_DESIGN_BINDING_STATUS:
        if set(value) != {"status", "reason"} or not str(value.get("reason", "")).strip():
            raise ValueError("pending confirmatory design binding is malformed")
        if not allow_pending:
            raise ValueError("calibration is pending pre-H-MEAS and cannot run or issue a receipt")
        return value
    required = {
        "status",
        "scientific_manifest_projection_sha256",
        "measurement_receipt_sha256",
        "ordered_target_gene_ids",
        "target_panel_sha256",
        "target_gene_count",
        "ordered_supported_fine_type_ids",
        "supported_fine_type_ids_sha256",
        "development_donor_ids",
        "locked_test_donor_ids",
        "encoder_manifest_sha256",
        "crop_manifest_sha256s",
        "planned_donor_type_support_pattern_sha256",
        "planned_stratum_topology_status",
        "ordered_planned_stratum_ids",
        "planned_stratum_manifest_sha256",
        "planned_stratum_minimum_evaluation_cells",
        "planned_stratum_support_pattern_sha256",
    }
    if set(value) != required or value.get("status") != COMPLETE_DESIGN_BINDING_STATUS:
        raise ValueError("completed confirmatory design binding is incomplete or contains extras")
    for name in (
        "scientific_manifest_projection_sha256",
        "measurement_receipt_sha256",
        "target_panel_sha256",
        "supported_fine_type_ids_sha256",
        "encoder_manifest_sha256",
        "planned_donor_type_support_pattern_sha256",
    ):
        _sha256(value[name], "confirmatory_design_binding.%s" % name)
    target_gene_count = _integer(value["target_gene_count"], "target_gene_count", minimum=1)
    target_genes = _unique_strings(value["ordered_target_gene_ids"], "ordered_target_gene_ids")
    if target_gene_count < 6:
        raise ValueError("confirmatory target panel cannot support the frozen rank-six grid")
    if (
        target_gene_count != len(target_genes)
        or canonical_sha256(list(target_genes)) != value["target_panel_sha256"]
    ):
        raise ValueError("ordered target genes differ from their completed binding")
    fine_types = _unique_strings(
        value["ordered_supported_fine_type_ids"],
        "ordered_supported_fine_type_ids",
    )
    if canonical_sha256(list(fine_types)) != value["supported_fine_type_ids_sha256"]:
        raise ValueError("supported fine-type IDs differ from their binding hash")
    development = _unique_strings(value["development_donor_ids"], "development_donor_ids")
    locked = _unique_strings(value["locked_test_donor_ids"], "locked_test_donor_ids")
    if len(development) != 10 or len(locked) != 5 or set(development) & set(locked):
        raise ValueError(
            "confirmatory design must bind ten development and five disjoint locked donors"
        )
    crop_hashes = _unique_strings(value["crop_manifest_sha256s"], "crop_manifest_sha256s")
    for index, digest in enumerate(crop_hashes):
        _sha256(digest, "crop_manifest_sha256s[%d]" % index)
    expected_pattern = planned_donor_type_support_pattern_sha256(development, locked, fine_types)
    if expected_pattern != value["planned_donor_type_support_pattern_sha256"]:
        raise ValueError("planned donor-by-type support pattern differs from the completed binding")
    topology_status = value["planned_stratum_topology_status"]
    stratum_ids = (
        _unique_strings(
            value["ordered_planned_stratum_ids"],
            "ordered_planned_stratum_ids",
        )
        if value["ordered_planned_stratum_ids"]
        else ()
    )
    support_counts_value = value["planned_stratum_minimum_evaluation_cells"]
    if not isinstance(support_counts_value, list):
        raise ValueError("planned stratum support counts must be a list")
    if topology_status == "pending_h_meas_stratum_topology":
        if (
            stratum_ids
            or support_counts_value
            or value["planned_stratum_manifest_sha256"] is not None
            or value["planned_stratum_support_pattern_sha256"] is not None
        ):
            raise ValueError("pending calibration topology must not contain planned strata")
    elif topology_status == "complete":
        support_counts = tuple(
            _integer(count, "planned stratum minimum evaluation cells", minimum=1)
            for count in support_counts_value
        )
        parsed = tuple(tuple(stratum.split("|")) for stratum in stratum_ids)
        if (
            len(support_counts) != len(stratum_ids)
            or any(len(parts) != 3 or any(not part for part in parts) for parts in parsed)
            or {parts[0] for parts in parsed} != set(development) | set(locked)
            or {parts[2] for parts in parsed} != set(fine_types)
            or {(parts[0], parts[2]) for parts in parsed}
            != {(donor, fine_type) for donor in development + locked for fine_type in fine_types}
        ):
            raise ValueError("completed calibration topology differs from donor/section/type scope")
        if value["planned_stratum_manifest_sha256"] != canonical_sha256(list(stratum_ids)):
            raise ValueError("planned calibration strata differ from their manifest hash")
        support_pattern = [
            {"stratum_id": stratum_id, "minimum_evaluation_cells": count}
            for stratum_id, count in zip(stratum_ids, support_counts)
        ]
        if value["planned_stratum_support_pattern_sha256"] != canonical_sha256(support_pattern):
            raise ValueError("planned calibration stratum support differs from its hash")
    else:
        raise ValueError("calibration topology status is unsupported")
    return value


def build_confirmatory_design_binding(
    manifest_content: Mapping[str, object],
    *,
    measurement_receipt_sha256: str,
    ordered_target_gene_ids: Sequence[str],
    supported_fine_type_ids: Sequence[str],
    ordered_planned_stratum_ids: Optional[Sequence[str]] = None,
    planned_stratum_minimum_evaluation_cells: Optional[Sequence[int]] = None,
) -> Mapping[str, object]:
    """Build the completed design binding after H-MEAS and before locked outcomes."""

    receipt_sha = _sha256(measurement_receipt_sha256, "measurement_receipt_sha256")
    genes = tuple(str(value) for value in ordered_target_gene_ids)
    fine_types = tuple(str(value) for value in supported_fine_type_ids)
    if (
        not genes
        or len(set(genes)) != len(genes)
        or not fine_types
        or len(set(fine_types)) != len(fine_types)
    ):
        raise ValueError("H-MEAS target genes and supported fine types must be unique and nonempty")
    partitions = manifest_content.get("partitions")
    observations = manifest_content.get("observations")
    encoder = manifest_content.get("encoder")
    prerequisites = manifest_content.get("prerequisites")
    if not all(
        isinstance(value, Mapping) for value in (partitions, observations, encoder, prerequisites)
    ):
        raise ValueError("confirmatory manifest design bindings are malformed")
    development = tuple(str(value) for value in partitions.get("development_donors", ()))
    locked = tuple(str(value) for value in partitions.get("locked_test_donors", ()))
    target_sha = canonical_sha256(list(genes))
    type_sha = canonical_sha256(list(fine_types))
    if (
        manifest_content.get("target_gene_panel_sha256") != target_sha
        or observations.get("supported_fine_type_ids") != list(fine_types)
        or observations.get("supported_fine_type_ids_sha256") != type_sha
        or prerequisites.get("measurement_report_sha256") != receipt_sha
    ):
        raise ValueError("completed H-CELL manifest differs from the H-MEAS selection receipt")
    crop_hashes = tuple(str(value) for value in manifest_content.get("crop_protocols", ()))
    topology_supplied = ordered_planned_stratum_ids is not None
    if topology_supplied != (planned_stratum_minimum_evaluation_cells is not None):
        raise ValueError("planned stratum IDs and support counts must be supplied together")
    planned_strata = (
        _unique_strings(list(ordered_planned_stratum_ids), "ordered_planned_stratum_ids")
        if ordered_planned_stratum_ids is not None
        else ()
    )
    planned_support = (
        tuple(
            _integer(value, "planned stratum minimum evaluation cells", minimum=1)
            for value in planned_stratum_minimum_evaluation_cells
        )
        if planned_stratum_minimum_evaluation_cells is not None
        else ()
    )
    coverage = manifest_content.get("coverage_requirements")
    if not isinstance(coverage, Mapping):
        raise ValueError("confirmatory manifest coverage requirements are malformed")
    frozen_minimum = _integer(
        coverage.get("minimum_evaluation_cells_per_donor_section_type"),
        "minimum evaluation cells per donor/section/type",
        minimum=1,
    )
    if planned_support and any(value != frozen_minimum for value in planned_support):
        raise ValueError("planned stratum support differs from the frozen H-CELL minimum")
    support_pattern = [
        {"stratum_id": stratum_id, "minimum_evaluation_cells": count}
        for stratum_id, count in zip(planned_strata, planned_support)
    ]
    binding = {
        "status": COMPLETE_DESIGN_BINDING_STATUS,
        "scientific_manifest_projection_sha256": canonical_sha256(
            confirmatory_scientific_manifest_projection(manifest_content)
        ),
        "measurement_receipt_sha256": receipt_sha,
        "ordered_target_gene_ids": list(genes),
        "target_panel_sha256": target_sha,
        "target_gene_count": len(genes),
        "ordered_supported_fine_type_ids": list(fine_types),
        "supported_fine_type_ids_sha256": type_sha,
        "development_donor_ids": list(development),
        "locked_test_donor_ids": list(locked),
        "encoder_manifest_sha256": encoder.get("manifest_sha256"),
        "crop_manifest_sha256s": list(crop_hashes),
        "planned_donor_type_support_pattern_sha256": planned_donor_type_support_pattern_sha256(
            development,
            locked,
            fine_types,
        ),
        "planned_stratum_topology_status": (
            "complete" if topology_supplied else "pending_h_meas_stratum_topology"
        ),
        "ordered_planned_stratum_ids": list(planned_strata),
        "planned_stratum_manifest_sha256": (
            canonical_sha256(list(planned_strata)) if topology_supplied else None
        ),
        "planned_stratum_minimum_evaluation_cells": list(planned_support),
        "planned_stratum_support_pattern_sha256": (
            canonical_sha256(support_pattern) if topology_supplied else None
        ),
    }
    return validate_confirmatory_design_binding(binding)


def _sha256(value: object, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("calibration receipt %s must be a lowercase SHA-256" % name)
    return digest


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or int(value) != value
        or int(value) < minimum
    ):
        raise ValueError("calibration receipt %s must be an integer >= %d" % (name, minimum))
    return int(value)


def _finite_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("calibration receipt %s must be a probability" % name)
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("calibration receipt %s must be in [0, 1]" % name)
    return result


def _unique_strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("calibration receipt %s must be a list" % name)
    result = tuple(str(item) for item in value)
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError("calibration receipt %s contains empty or duplicate values" % name)
    return result


def _log_binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Return log(P[X <= successes]) for X ~ Binomial(trials, probability)."""

    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 0.0 if successes >= trials else -math.inf
    logs = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        + index * math.log(probability)
        + (trials - index) * math.log1p(-probability)
        for index in range(successes + 1)
    ]
    maximum = max(logs)
    return maximum + math.log(sum(math.exp(value - maximum) for value in logs))


def binomial_upper_confidence_bound(
    passes: int, trials: int, *, confidence_level: float = 0.95
) -> float:
    """Exact one-sided Clopper-Pearson upper confidence bound."""

    successes = _integer(passes, "passes")
    count = _integer(trials, "trials", minimum=1)
    confidence = _finite_probability(confidence_level, "confidence_level")
    if successes > count or not 0.0 < confidence < 1.0:
        raise ValueError("binomial confidence-bound inputs are invalid")
    if successes == count:
        return 1.0
    target = math.log1p(-confidence)
    lower = successes / count
    upper = 1.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if _log_binomial_cdf(successes, count, midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def binomial_lower_confidence_bound(
    passes: int, trials: int, *, confidence_level: float = 0.95
) -> float:
    """Exact one-sided Clopper-Pearson lower confidence bound."""

    successes = _integer(passes, "passes")
    count = _integer(trials, "trials", minimum=1)
    if successes > count:
        raise ValueError("binomial confidence-bound inputs are invalid")
    return 1.0 - binomial_upper_confidence_bound(
        count - successes,
        count,
        confidence_level=confidence_level,
    )


def _same_set(value: object, expected: Sequence[str], name: str) -> tuple[str, ...]:
    observed = _unique_strings(value, name)
    if set(observed) != set(expected) or len(observed) != len(expected):
        raise ValueError("calibration receipt %s differs from the frozen contract" % name)
    return observed


def validate_calibration_run_contract(
    value: object,
    *,
    expected_settings_sha256: str,
    require_authorizing_boundary: bool,
) -> Mapping[str, object]:
    """Validate reproducible executable and data-generating-process provenance."""

    if not isinstance(value, Mapping):
        raise ValueError("calibration run_contract must be an object")
    required = {
        "schema",
        "generator_version",
        "generator_source_sha256",
        "gate_source_sha256",
        "compiler_source_sha256",
        "contract_source_sha256",
        "scientific_source_tree_sha256",
        "dependency_lock_sha256",
        "dgp_effect_spec",
        "dgp_effect_spec_sha256",
        "actual_gate_entrypoint",
        "exact_gate_settings",
        "exact_gate_settings_sha256",
        "permutations_per_null",
        "permutation_seeds",
        "permutations_per_seed",
        "scenario_families",
        "conditions",
        "trials_per_condition",
        "base_seed",
        "device",
        "smoke_test",
        "process_isolation",
        "max_cpu_threads",
        "maximum_process_rss_gib",
        "maximum_address_space_gib",
    }
    if set(value) != required or value.get("schema") != CALIBRATION_RUN_CONTRACT_SCHEMA:
        raise ValueError("calibration run_contract is incomplete or contains extras")
    if value["actual_gate_entrypoint"] != ACTUAL_GATE_ENTRYPOINT:
        raise ValueError("calibration run_contract does not use the actual gate entrypoint")
    if _sha256(value["exact_gate_settings_sha256"], "run exact settings") != _sha256(
        expected_settings_sha256,
        "expected exact settings",
    ):
        raise ValueError("calibration run_contract differs from exact gate settings")
    run_settings = validate_exact_gate_settings(value["exact_gate_settings"])
    if canonical_sha256(run_settings) != expected_settings_sha256:
        raise ValueError("calibration run_contract embeds different exact gate settings")
    provenance = current_calibration_executable_provenance()
    for name, expected in provenance.items():
        if _sha256(value[name], "run_contract.%s" % name) != expected:
            raise ValueError("calibration executable provenance changed: %s" % name)
    if value["generator_version"] != CALIBRATION_GENERATOR_VERSION:
        raise ValueError("calibration run contract uses an unsupported generator version")
    if not str(value["device"]).strip():
        raise ValueError("calibration device must be explicit")
    if isinstance(value["base_seed"], bool) or not isinstance(value["base_seed"], int):
        raise ValueError("calibration base seed must be an integer")
    if value["scenario_families"] != list(REQUIRED_CALIBRATION_SCENARIOS):
        raise ValueError("calibration run scenario order differs from the frozen contract")
    dgp = value["dgp_effect_spec"]
    if not isinstance(dgp, Mapping):
        raise ValueError("calibration DGP/effect specification is malformed")
    dgp_required = {
        "schema",
        "authorizing_boundary_calibration",
        "null_condition_id",
        "alternative_condition_id",
        "boundary_condition_ids_by_hypothesis",
        "decision_truth_by_condition",
        "effect_definition",
        "expected_source_conclusion_by_condition",
        "hypothesis_specific_boundary_sha256",
    }
    if set(dgp) != dgp_required or dgp.get("schema") != CALIBRATION_DGP_SPEC_SCHEMA:
        raise ValueError("calibration DGP/effect specification is incomplete")
    if not isinstance(dgp["authorizing_boundary_calibration"], bool):
        raise ValueError("calibration DGP authorization flag must be boolean")
    if not isinstance(dgp["effect_definition"], Mapping) or not dgp["effect_definition"]:
        raise ValueError("calibration DGP effect definition must be explicit")
    dgp_sha = canonical_sha256(dgp)
    if _sha256(value["dgp_effect_spec_sha256"], "dgp_effect_spec_sha256") != dgp_sha:
        raise ValueError("calibration DGP/effect specification hash differs")
    null_id = str(dgp["null_condition_id"])
    if null_id != GLOBAL_NULL_CONDITION:
        raise ValueError("calibration DGP global-null identity is invalid")
    boundary_ids = dgp["boundary_condition_ids_by_hypothesis"]
    decision_truth = dgp["decision_truth_by_condition"]
    if not isinstance(boundary_ids, Mapping) or not isinstance(decision_truth, Mapping):
        raise ValueError("calibration DGP boundary identities or decision truth are malformed")
    authorizing = dgp["authorizing_boundary_calibration"] is True
    if authorizing:
        design_binding = run_settings["confirmatory_design_binding"]
        if design_binding["planned_stratum_topology_status"] != "complete":
            raise ValueError(
                "authorizing calibration requires a complete donor/section/type topology"
            )
        if (
            dgp["alternative_condition_id"] is not None
            or dict(boundary_ids) != BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS
        ):
            raise ValueError("authorizing calibration lacks the frozen partial-null truth matrix")
        condition_ids = [
            null_id,
            *(boundary_ids[name] for name in REQUIRED_HYPOTHESIS_DECISIONS),
        ]
    else:
        alternative_id = str(dgp["alternative_condition_id"])
        if boundary_ids or alternative_id != PRELIMINARY_ALTERNATIVE_CONDITION:
            raise ValueError("preliminary calibration uses an unsupported diagnostic alternative")
        condition_ids = [null_id, alternative_id]
    if value["conditions"] != condition_ids or len(set(condition_ids)) != len(condition_ids):
        raise ValueError("calibration run conditions differ from the DGP specification")
    if set(decision_truth) != set(condition_ids):
        raise ValueError("calibration DGP decision-truth conditions are incomplete")
    normalized_truth = {}
    for condition_id in condition_ids:
        truth = decision_truth[condition_id]
        if not isinstance(truth, Mapping) or set(truth) != set(REQUIRED_HYPOTHESIS_DECISIONS):
            raise ValueError("calibration DGP decision-truth row is incomplete")
        if any(not isinstance(flag, bool) for flag in truth.values()):
            raise ValueError("calibration DGP decision truth must be boolean")
        normalized_truth[condition_id] = dict(truth)
    if any(normalized_truth[null_id].values()):
        raise ValueError("global-null calibration cannot declare a true hypothesis")
    if authorizing:
        for decision_id in REQUIRED_HYPOTHESIS_DECISIONS:
            condition_id = str(boundary_ids[decision_id])
            expected_true = {"G2_local_context", decision_id}
            observed_true = {name for name, flag in normalized_truth[condition_id].items() if flag}
            if observed_true != expected_true:
                raise ValueError("authorizing boundary truth must isolate one G3 decision over G2")
    elif not normalized_truth[PRELIMINARY_ALTERNATIVE_CONDITION]["G2_local_context"]:
        raise ValueError("preliminary shared-latent alternative must contain a G2 signal")
    expected_conclusions = dgp["expected_source_conclusion_by_condition"]
    if not isinstance(expected_conclusions, Mapping) or set(expected_conclusions) != set(
        condition_ids
    ):
        raise ValueError("calibration DGP lacks expected morphology-source conclusions")
    for conclusion in expected_conclusions.values():
        if conclusion not in REQUIRED_MORPHOLOGY_SOURCE_CONCLUSIONS:
            raise ValueError("calibration DGP contains an unsupported source conclusion")
    if expected_conclusions[null_id] != "no_morphology_specific_information":
        raise ValueError("global-null calibration must expect no morphology-specific information")
    if authorizing:
        for decision_id, condition_id in boundary_ids.items():
            if (
                expected_conclusions[condition_id]
                != BOUNDARY_EXPECTED_SOURCE_CONCLUSION[decision_id]
            ):
                raise ValueError("authorizing boundary has the wrong source conclusion")
    boundary_hashes = dgp["hypothesis_specific_boundary_sha256"]
    if not isinstance(boundary_hashes, Mapping):
        raise ValueError("calibration hypothesis-specific boundary hashes are malformed")
    condition_definitions = dgp["effect_definition"].get("condition_definitions")
    if not isinstance(condition_definitions, Mapping) or set(condition_definitions) != set(
        condition_ids
    ):
        raise ValueError("calibration DGP lacks quantitative definitions for every condition")
    if any(
        not isinstance(definition, Mapping) or not definition
        for definition in condition_definitions.values()
    ):
        raise ValueError("calibration DGP condition definition is empty")
    if authorizing:
        if set(boundary_hashes) != set(REQUIRED_HYPOTHESIS_DECISIONS):
            raise ValueError("authorizing calibration lacks every frozen hypothesis boundary")
        for name, digest in boundary_hashes.items():
            if _sha256(digest, "hypothesis boundary %s" % name) != canonical_sha256(
                condition_definitions[boundary_ids[name]]
            ):
                raise ValueError("hypothesis boundary hash differs from its quantitative DGP")
    elif boundary_hashes:
        raise ValueError("preliminary calibration cannot claim hypothesis boundary hashes")
    elif alternative_id != PRELIMINARY_ALTERNATIVE_CONDITION:
        raise ValueError("preliminary calibration uses an unsupported diagnostic alternative")
    if require_authorizing_boundary and not authorizing:
        raise ValueError(
            "preliminary full-shared-latent calibration cannot issue an authorizing receipt"
        )
    trials = _integer(value["trials_per_condition"], "trials_per_condition", minimum=1)
    if require_authorizing_boundary and (value["smoke_test"] is not False or trials < 1000):
        raise ValueError("authorizing calibration requires >=1000 non-smoke trials per condition")
    if not isinstance(value["smoke_test"], bool):
        raise ValueError("calibration smoke_test flag must be boolean")
    if value["process_isolation"] not in {"dedicated_cli_process", "in_process_smoke"}:
        raise ValueError("calibration process isolation is unsupported")
    if value["smoke_test"] is False and value["process_isolation"] != "dedicated_cli_process":
        raise ValueError("non-smoke calibration requires a dedicated CLI process")
    if _integer(value["permutations_per_null"], "run permutations_per_null", minimum=1) < 999:
        raise ValueError("calibration run contract uses too few permutations")
    seeds = value["permutation_seeds"]
    if seeds != [17, 29, 41]:
        raise ValueError("calibration run contract uses different permutation seed streams")
    per_seed = _integer(value["permutations_per_seed"], "run permutations_per_seed", minimum=1)
    if per_seed != 333 or int(value["permutations_per_null"]) != per_seed * len(seeds):
        raise ValueError("calibration run contract lacks exactly 333 permutations per stream")
    max_threads = _integer(value["max_cpu_threads"], "run max_cpu_threads", minimum=1)
    if max_threads > max(int(os.cpu_count() or 1), 1):
        raise ValueError("calibration run contract requests unavailable CPU threads")
    maximum_rss = value["maximum_process_rss_gib"]
    if (
        isinstance(maximum_rss, bool)
        or not isinstance(maximum_rss, (int, float))
        or not math.isfinite(float(maximum_rss))
        or float(maximum_rss) <= 0.0
    ):
        raise ValueError("calibration run contract has an invalid RSS limit")
    maximum_address_space = value["maximum_address_space_gib"]
    if (
        isinstance(maximum_address_space, bool)
        or not isinstance(maximum_address_space, (int, float))
        or not math.isfinite(float(maximum_address_space))
        or float(maximum_address_space) <= 0.0
    ):
        raise ValueError("calibration run contract has an invalid address-space limit")
    return value


def required_simultaneous_confidence_level() -> float:
    """Bonferroni level for complete, decision, and source-classification bounds."""

    conditions = 1 + len(REQUIRED_HYPOTHESIS_DECISIONS)
    tested_rates = (
        len(REQUIRED_CALIBRATION_SCENARIOS) * conditions * (2 + len(REQUIRED_HYPOTHESIS_DECISIONS))
    )
    return 1.0 - (0.05 / tested_rates)


def validate_exact_gate_settings(
    settings: object,
    *,
    allow_pending_design: bool = False,
) -> Mapping[str, object]:
    """Validate the scientific settings reproduced by every synthetic trial."""

    if not isinstance(settings, Mapping):
        raise ValueError("calibration exact_gate_settings must be an object")
    required = {
        "actual_gate_entrypoint",
        "actual_gate_report_schema",
        "confirmatory_analysis_plan_sha256",
        "confirmatory_design_binding",
        "development_donors",
        "evaluation_donors",
        "crop_family_ids",
        "nuisance_families",
        "g2_multiplicity_method",
        "g2_nuisance_family_ids",
        "g3_multiplicity_method",
        "g3_contrast_pairs",
        "allowed_morphology_source_conclusions",
        "target_rank_grid",
        "ridge_penalty_grid",
        "reference_split_ids",
        "permutation_transforms",
        "permutation_seeds",
        "permutations_per_seed",
        "permutations_per_null",
        "maximum_permutation_p",
        "gate_parameters",
        "complete_gate_check_ids",
        "hypothesis_decision_ids",
        "final_inference",
    }
    if set(str(name) for name in settings) != required:
        raise ValueError("calibration exact_gate_settings are incomplete or contain extras")
    if settings["actual_gate_entrypoint"] != ACTUAL_GATE_ENTRYPOINT:
        raise ValueError("calibration did not execute the actual morphology gate")
    if settings["actual_gate_report_schema"] != ACTUAL_GATE_REPORT_SCHEMA:
        raise ValueError("calibration morphology report schema differs from the frozen gate")
    _sha256(
        settings["confirmatory_analysis_plan_sha256"],
        "confirmatory_analysis_plan_sha256",
    )
    validate_confirmatory_design_binding(
        settings["confirmatory_design_binding"],
        allow_pending=allow_pending_design,
    )
    if _integer(settings["development_donors"], "development_donors", minimum=1) != 10:
        raise ValueError("calibration must reproduce all ten development donors")
    if _integer(settings["evaluation_donors"], "evaluation_donors", minimum=1) != 5:
        raise ValueError("calibration must use exactly five locked/evaluation donors")
    _same_set(settings["crop_family_ids"], REQUIRED_CROP_FAMILY_IDS, "crop_family_ids")
    _same_set(settings["nuisance_families"], REQUIRED_NUISANCE_FAMILIES, "nuisance_families")
    if settings["g2_multiplicity_method"] != G2_MULTIPLICITY_METHOD:
        raise ValueError("calibration G2 multiplicity differs from the frozen gate")
    _same_set(
        settings["g2_nuisance_family_ids"],
        REQUIRED_NUISANCE_FAMILIES,
        "g2_nuisance_family_ids",
    )
    if settings["g3_multiplicity_method"] != G3_MULTIPLICITY_METHOD:
        raise ValueError("calibration G3 multiplicity differs from the frozen gate")
    observed_pairs = settings["g3_contrast_pairs"]
    if (
        not isinstance(observed_pairs, Mapping)
        or dict(observed_pairs) != REQUIRED_G3_CONTRAST_PAIRS
    ):
        raise ValueError("calibration G3 contrast pairs differ from the frozen gate")
    _same_set(
        settings["allowed_morphology_source_conclusions"],
        REQUIRED_MORPHOLOGY_SOURCE_CONCLUSIONS,
        "allowed_morphology_source_conclusions",
    )
    ranks = settings["target_rank_grid"]
    penalties = settings["ridge_penalty_grid"]
    if ranks != [2, 4, 6] or penalties != [0.1, 1.0, 10.0, 100.0]:
        raise ValueError("calibration rank/ridge grid differs from the frozen H-CELL grid")
    splits = _unique_strings(settings["reference_split_ids"], "reference_split_ids")
    if len(splits) < 3 or splits[0] != "primary":
        raise ValueError("calibration requires the primary and at least two reference splits")
    _same_set(
        settings["permutation_transforms"],
        REQUIRED_PERMUTATION_TRANSFORMS,
        "permutation_transforms",
    )
    if settings["permutation_seeds"] != [17, 29, 41]:
        raise ValueError("calibration permutation seed streams differ from H-CELL")
    if _integer(settings["permutations_per_seed"], "permutations_per_seed") != 333:
        raise ValueError("calibration permutations per seed differ from H-CELL")
    per_seed = _integer(settings["permutations_per_seed"], "permutations_per_seed")
    per_null = _integer(settings["permutations_per_null"], "permutations_per_null")
    if per_null != per_seed * len(settings["permutation_seeds"]) or per_null != 999:
        raise ValueError("calibration requires exactly 333 permutations from each seed stream")
    if _finite_probability(settings["maximum_permutation_p"], "maximum_permutation_p") != 0.01:
        raise ValueError("calibration must reproduce the p <= 0.01 permutation threshold")
    gate_parameters = settings["gate_parameters"]
    if (
        not isinstance(gate_parameters, Mapping)
        or dict(gate_parameters) != REQUIRED_GATE_PARAMETERS
    ):
        raise ValueError("calibration decision parameters differ from the frozen H-CELL gate")
    if gate_parameters["maximum_permutation_p"] != settings["maximum_permutation_p"]:
        raise ValueError("calibration permutation thresholds disagree")
    _same_set(
        settings["complete_gate_check_ids"],
        REQUIRED_COMPLETE_GATE_CHECKS,
        "complete_gate_check_ids",
    )
    _same_set(
        settings["hypothesis_decision_ids"],
        REQUIRED_HYPOTHESIS_DECISIONS,
        "hypothesis_decision_ids",
    )
    if settings["final_inference"] is not True:
        raise ValueError("calibration trials must execute the final-inference code path")
    return settings


def exact_gate_settings_fingerprint(settings: object) -> str:
    """Return the validated scientific-settings fingerprint used by a receipt."""

    return canonical_sha256(validate_exact_gate_settings(settings))


def _validate_condition(
    value: object,
    *,
    name: str,
    confidence_level: float,
    permutations_per_null: int,
    permutation_seeds: tuple[int, ...],
    permutations_per_seed: int,
    expected_trials: int,
    complete_gate_expected_pass: bool,
    decision_truth: Mapping[str, bool],
    expected_source_conclusion: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("calibration scenario %s is malformed" % name)
    required = {
        "trials",
        "complete_gate_passes",
        "complete_gate_pass_fraction",
        "complete_gate_pass_confidence_bound",
        "hypothesis_decision_passes",
        "hypothesis_decision_pass_fractions",
        "hypothesis_decision_pass_confidence_bounds",
        "morphology_source_conclusion_counts",
        "expected_morphology_source_conclusion",
        "morphology_source_conclusion_correct_count",
        "morphology_source_conclusion_correct_fraction",
        "morphology_source_conclusion_correct_lower_confidence_bound",
        "actual_gate_executions",
        "trial_report_set_sha256",
        "all_trial_reports_use_exact_settings",
        "all_trial_reports_include_required_checks",
        "permutation_nulls",
    }
    if set(str(field) for field in value) != required:
        raise ValueError("calibration scenario %s has incomplete evidence" % name)
    if (
        not isinstance(decision_truth, Mapping)
        or set(decision_truth) != set(REQUIRED_HYPOTHESIS_DECISIONS)
        or any(not isinstance(flag, bool) for flag in decision_truth.values())
    ):
        raise ValueError("calibration scenario has invalid decision truth")
    trials = _integer(value["trials"], "%s.trials" % name, minimum=1000)
    if trials != expected_trials:
        raise ValueError("calibration scenario trial count differs from its run contract")
    passes = _integer(value["complete_gate_passes"], "%s.passes" % name)
    if passes > trials:
        raise ValueError("calibration scenario %s has impossible pass counts" % name)
    fraction = _finite_probability(value["complete_gate_pass_fraction"], "%s.pass_fraction" % name)
    if abs(fraction - passes / trials) > 1.0e-12:
        raise ValueError("calibration empirical pass counts are inconsistent")
    expected_bound = (
        binomial_lower_confidence_bound(passes, trials, confidence_level=confidence_level)
        if complete_gate_expected_pass
        else binomial_upper_confidence_bound(passes, trials, confidence_level=confidence_level)
    )
    observed_bound = _finite_probability(
        value["complete_gate_pass_confidence_bound"], "%s.confidence_bound" % name
    )
    if abs(expected_bound - observed_bound) > 1.0e-12:
        raise ValueError("calibration confidence bound differs from exact binomial evidence")
    decision_passes = value["hypothesis_decision_passes"]
    decision_fractions = value["hypothesis_decision_pass_fractions"]
    decision_bounds = value["hypothesis_decision_pass_confidence_bounds"]
    if not all(
        isinstance(mapping, Mapping) and set(mapping) == set(REQUIRED_HYPOTHESIS_DECISIONS)
        for mapping in (decision_passes, decision_fractions, decision_bounds)
    ):
        raise ValueError("calibration decision-specific evidence is incomplete")
    for decision_id in REQUIRED_HYPOTHESIS_DECISIONS:
        count = _integer(decision_passes[decision_id], "%s.%s.passes" % (name, decision_id))
        if count > trials:
            raise ValueError("calibration decision pass count exceeds its trials")
        decision_fraction = _finite_probability(
            decision_fractions[decision_id], "%s.%s.fraction" % (name, decision_id)
        )
        if abs(decision_fraction - count / trials) > 1.0e-12:
            raise ValueError("calibration decision pass counts are inconsistent")
        decision_expected_bound = (
            binomial_lower_confidence_bound(count, trials, confidence_level=confidence_level)
            if decision_truth[decision_id]
            else binomial_upper_confidence_bound(count, trials, confidence_level=confidence_level)
        )
        decision_observed_bound = _finite_probability(
            decision_bounds[decision_id], "%s.%s.bound" % (name, decision_id)
        )
        if abs(decision_expected_bound - decision_observed_bound) > 1.0e-12:
            raise ValueError("calibration decision confidence bound differs")
    conclusion_counts = value["morphology_source_conclusion_counts"]
    if not isinstance(conclusion_counts, Mapping) or set(conclusion_counts) != set(
        CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES
    ):
        raise ValueError("calibration source-conclusion evidence is incomplete")
    normalized_conclusions = {
        conclusion: _integer(count, "%s.%s.count" % (name, conclusion))
        for conclusion, count in conclusion_counts.items()
    }
    if sum(normalized_conclusions.values()) != trials:
        raise ValueError("calibration source-conclusion counts do not sum to trials")
    if (
        value["expected_morphology_source_conclusion"] != expected_source_conclusion
        or expected_source_conclusion not in REQUIRED_MORPHOLOGY_SOURCE_CONCLUSIONS
    ):
        raise ValueError("calibration expected source conclusion differs from the DGP")
    correct_count = normalized_conclusions[expected_source_conclusion]
    if (
        _integer(
            value["morphology_source_conclusion_correct_count"],
            "%s.source_conclusion_correct_count" % name,
        )
        != correct_count
    ):
        raise ValueError("calibration source-conclusion correct count differs")
    correct_fraction = _finite_probability(
        value["morphology_source_conclusion_correct_fraction"],
        "%s.source_conclusion_correct_fraction" % name,
    )
    if abs(correct_fraction - correct_count / trials) > 1.0e-12:
        raise ValueError("calibration source-conclusion fraction differs")
    expected_conclusion_bound = binomial_lower_confidence_bound(
        correct_count,
        trials,
        confidence_level=confidence_level,
    )
    observed_conclusion_bound = _finite_probability(
        value["morphology_source_conclusion_correct_lower_confidence_bound"],
        "%s.source_conclusion_correct_lower_confidence_bound" % name,
    )
    if abs(expected_conclusion_bound - observed_conclusion_bound) > 1.0e-12:
        raise ValueError("calibration source-conclusion confidence bound differs")
    if _integer(value["actual_gate_executions"], "%s.actual_gate_executions" % name) != trials:
        raise ValueError("calibration evidence did not execute the actual gate for every trial")
    _sha256(value["trial_report_set_sha256"], "%s.trial_report_set_sha256" % name)
    if (
        value["all_trial_reports_use_exact_settings"] is not True
        or value["all_trial_reports_include_required_checks"] is not True
    ):
        raise ValueError("calibration evidence contains incomplete or mismatched gate reports")
    nulls = value["permutation_nulls"]
    if not isinstance(nulls, Mapping) or set(nulls) != {
        "local_roi_permutations",
        "spatial_block_permutations",
        "local_roi_seed_counts",
        "spatial_block_seed_counts",
    }:
        raise ValueError("calibration permutation evidence is malformed")
    if (
        _integer(nulls["local_roi_permutations"], "local_roi_permutations") != permutations_per_null
        or _integer(nulls["spatial_block_permutations"], "spatial_block_permutations")
        != permutations_per_null
    ):
        raise ValueError("calibration evidence differs from the exact permutation total")
    expected_seed_counts = {str(seed): permutations_per_seed for seed in permutation_seeds}
    if (
        nulls["local_roi_seed_counts"] != expected_seed_counts
        or nulls["spatial_block_seed_counts"] != expected_seed_counts
    ):
        raise ValueError("calibration evidence seed counts differ from frozen streams")
    return value


def validate_calibration_receipt(
    receipt: Optional[Mapping[str, object]],
    *,
    required: bool,
    expected_settings: Optional[Mapping[str, object]] = None,
) -> Mapping[str, object]:
    """Validate exact-gate pre-lock calibration without opening biological outcomes."""

    if receipt is None:
        if required:
            raise ValueError("final morphology inference requires a calibration receipt")
        return {"available": False, "required": False}
    if receipt.get("schema") != CALIBRATION_RECEIPT_SCHEMA:
        raise ValueError(
            "legacy or surrogate morphology calibration cannot authorize final inference"
        )
    required_fields = {
        "schema",
        "pass",
        "engine",
        "actual_gate_entrypoint",
        "exact_gate_executed",
        "surrogate",
        "synthetic_data_only",
        "locked_outcomes_used",
        "confirmatory_scientific_settings_sha256",
        "exact_gate_settings",
        "exact_gate_settings_sha256",
        "run_contract",
        "run_contract_sha256",
        "generator_version",
        "generator_source_sha256",
        "gate_source_sha256",
        "compiler_source_sha256",
        "contract_source_sha256",
        "scientific_source_tree_sha256",
        "dependency_lock_sha256",
        "dgp_effect_spec",
        "dgp_effect_spec_sha256",
        "thresholds",
        "thresholds_sha256",
        "scenario_families",
        "scenario_results",
        "complete_gate_check_ids",
        "hypothesis_decision_ids",
        "confidence_level",
        "maximum_complete_gate_false_pass_probability",
        "maximum_complete_gate_false_pass_upper_confidence_bound",
        "maximum_hypothesis_decision_false_pass_probability",
        "maximum_hypothesis_decision_false_pass_upper_confidence_bound",
        "power_at_quantitatively_frozen_boundary",
        "minimum_power_lower_confidence_bound",
        "minimum_hypothesis_decision_power_at_quantitatively_frozen_boundary",
        "minimum_hypothesis_decision_power_lower_confidence_bound",
        "minimum_global_null_source_conclusion_correct_lower_confidence_bound",
        "minimum_boundary_source_conclusion_correct_lower_confidence_bound",
        "simulation_sha256",
        "receipt_content_sha256",
    }
    if set(receipt) != required_fields:
        raise ValueError("morphology gate calibration receipt is incomplete or contains extras")
    if (
        receipt["engine"] != CALIBRATION_ENGINE
        or receipt["actual_gate_entrypoint"] != ACTUAL_GATE_ENTRYPOINT
    ):
        raise ValueError("calibration receipt was not produced by the actual morphology gate")
    settings = validate_exact_gate_settings(receipt["exact_gate_settings"])
    settings_sha256 = canonical_sha256(settings)
    if (
        _sha256(receipt["exact_gate_settings_sha256"], "exact_gate_settings_sha256")
        != settings_sha256
    ):
        raise ValueError("calibration exact-gate settings hash differs")
    if (
        _sha256(
            receipt["confirmatory_scientific_settings_sha256"],
            "confirmatory_scientific_settings_sha256",
        )
        != settings_sha256
    ):
        raise ValueError("calibration is not bound to the confirmatory scientific settings")
    if (
        expected_settings is not None
        and exact_gate_settings_fingerprint(expected_settings) != settings_sha256
    ):
        raise ValueError("calibration receipt differs from the live confirmatory settings")
    run_contract = validate_calibration_run_contract(
        receipt["run_contract"],
        expected_settings_sha256=settings_sha256,
        require_authorizing_boundary=True,
    )
    run_contract_sha = canonical_sha256(run_contract)
    if _sha256(receipt["run_contract_sha256"], "run_contract_sha256") != run_contract_sha:
        raise ValueError("calibration receipt run-contract hash differs")
    for name in (
        "generator_version",
        "generator_source_sha256",
        "gate_source_sha256",
        "compiler_source_sha256",
        "contract_source_sha256",
        "scientific_source_tree_sha256",
        "dependency_lock_sha256",
        "dgp_effect_spec",
        "dgp_effect_spec_sha256",
    ):
        if receipt[name] != run_contract[name]:
            raise ValueError("calibration receipt provenance differs: %s" % name)
    dgp = run_contract["dgp_effect_spec"]
    null_condition_id = str(dgp["null_condition_id"])
    condition_ids = tuple(str(value) for value in run_contract["conditions"])
    boundary_condition_ids = condition_ids[1:]
    decision_truth = dgp["decision_truth_by_condition"]
    expected_conclusions = dgp["expected_source_conclusion_by_condition"]
    expected_trials = int(run_contract["trials_per_condition"])
    thresholds = receipt["thresholds"]
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
        "maximum_false_pass_upper_confidence_bound",
        "minimum_power_lower_confidence_bound",
    }:
        raise ValueError("calibration decision thresholds are malformed")
    maximum_false_pass = _finite_probability(
        thresholds["maximum_false_pass_upper_confidence_bound"],
        "maximum_false_pass_upper_confidence_bound",
    )
    minimum_power = _finite_probability(
        thresholds["minimum_power_lower_confidence_bound"],
        "minimum_power_lower_confidence_bound",
    )
    if maximum_false_pass > 0.05 or minimum_power < 0.80:
        raise ValueError("calibration thresholds are weaker than the frozen requirements")
    if _sha256(receipt["thresholds_sha256"], "thresholds_sha256") != canonical_sha256(thresholds):
        raise ValueError("calibration threshold hash differs")
    scenarios = _same_set(
        receipt["scenario_families"],
        REQUIRED_CALIBRATION_SCENARIOS,
        "scenario_families",
    )
    _same_set(
        receipt["complete_gate_check_ids"],
        REQUIRED_COMPLETE_GATE_CHECKS,
        "complete_gate_check_ids",
    )
    _same_set(
        receipt["hypothesis_decision_ids"],
        REQUIRED_HYPOTHESIS_DECISIONS,
        "hypothesis_decision_ids",
    )
    results = receipt["scenario_results"]
    if not isinstance(results, Mapping) or set(str(name) for name in results) != set(scenarios):
        raise ValueError("calibration results differ from the required stress families")
    confidence = _finite_probability(receipt["confidence_level"], "confidence_level")
    minimum_simultaneous_level = required_simultaneous_confidence_level()
    if confidence < minimum_simultaneous_level or confidence >= 1.0:
        raise ValueError("calibration requires Bonferroni-adjusted 95% simultaneous bounds")
    permutations = int(settings["permutations_per_null"])
    null_rates = []
    null_upper_bounds = []
    effect_rates = []
    effect_lower_bounds = []
    null_decision_rates = []
    null_decision_upper_bounds = []
    effect_decision_rates = []
    effect_decision_lower_bounds = []
    null_conclusion_correct_lower_bounds = []
    alternative_conclusion_correct_lower_bounds = []
    for scenario in scenarios:
        result = results[scenario]
        if not isinstance(result, Mapping) or set(result) != set(condition_ids):
            raise ValueError("calibration scenario lacks the frozen truth-matrix conditions")
        validated = {
            condition_id: _validate_condition(
                result[condition_id],
                name="%s.%s" % (scenario, condition_id),
                confidence_level=confidence,
                permutations_per_null=permutations,
                permutation_seeds=tuple(int(value) for value in settings["permutation_seeds"]),
                permutations_per_seed=int(settings["permutations_per_seed"]),
                expected_trials=expected_trials,
                complete_gate_expected_pass=condition_id != null_condition_id,
                decision_truth=decision_truth[condition_id],
                expected_source_conclusion=str(expected_conclusions[condition_id]),
            )
            for condition_id in condition_ids
        }
        null = validated[null_condition_id]
        null_rates.append(float(null["complete_gate_pass_fraction"]))
        null_upper_bounds.append(float(null["complete_gate_pass_confidence_bound"]))
        for condition_id in boundary_condition_ids:
            effect_rates.append(float(validated[condition_id]["complete_gate_pass_fraction"]))
            effect_lower_bounds.append(
                float(validated[condition_id]["complete_gate_pass_confidence_bound"])
            )
        for condition_id in condition_ids:
            condition = validated[condition_id]
            for decision_id in REQUIRED_HYPOTHESIS_DECISIONS:
                rate = float(condition["hypothesis_decision_pass_fractions"][decision_id])
                bound = float(condition["hypothesis_decision_pass_confidence_bounds"][decision_id])
                if decision_truth[condition_id][decision_id]:
                    effect_decision_rates.append(rate)
                    effect_decision_lower_bounds.append(bound)
                else:
                    null_decision_rates.append(rate)
                    null_decision_upper_bounds.append(bound)
        null_conclusion_correct_lower_bounds.append(
            float(null["morphology_source_conclusion_correct_lower_confidence_bound"])
        )
        alternative_conclusion_correct_lower_bounds.extend(
            float(
                validated[condition_id][
                    "morphology_source_conclusion_correct_lower_confidence_bound"
                ]
            )
            for condition_id in boundary_condition_ids
        )
    aggregates = (
        ("maximum_complete_gate_false_pass_probability", max(null_rates)),
        (
            "maximum_complete_gate_false_pass_upper_confidence_bound",
            max(null_upper_bounds),
        ),
        (
            "maximum_hypothesis_decision_false_pass_probability",
            max(null_decision_rates),
        ),
        (
            "maximum_hypothesis_decision_false_pass_upper_confidence_bound",
            max(null_decision_upper_bounds),
        ),
        ("power_at_quantitatively_frozen_boundary", min(effect_rates)),
        ("minimum_power_lower_confidence_bound", min(effect_lower_bounds)),
        (
            "minimum_hypothesis_decision_power_at_quantitatively_frozen_boundary",
            min(effect_decision_rates),
        ),
        (
            "minimum_hypothesis_decision_power_lower_confidence_bound",
            min(effect_decision_lower_bounds),
        ),
        (
            "minimum_global_null_source_conclusion_correct_lower_confidence_bound",
            min(null_conclusion_correct_lower_bounds),
        ),
        (
            "minimum_boundary_source_conclusion_correct_lower_confidence_bound",
            min(alternative_conclusion_correct_lower_bounds),
        ),
    )
    for name, expected in aggregates:
        observed = _finite_probability(receipt[name], name)
        if abs(observed - expected) > 1.0e-12:
            raise ValueError("calibration aggregate error or power differs from scenarios")
    simulation_core = {
        "engine": receipt["engine"],
        "exact_gate_settings_sha256": settings_sha256,
        "thresholds_sha256": receipt["thresholds_sha256"],
        "run_contract_sha256": run_contract_sha,
        "scenario_results": results,
    }
    if _sha256(receipt["simulation_sha256"], "simulation_sha256") != canonical_sha256(
        simulation_core
    ):
        raise ValueError("calibration simulation hash differs")
    receipt_core = {
        str(name): value for name, value in receipt.items() if name != "receipt_content_sha256"
    }
    if _sha256(receipt["receipt_content_sha256"], "receipt_content_sha256") != canonical_sha256(
        receipt_core
    ):
        raise ValueError("calibration receipt content hash differs")
    if (
        receipt["pass"] is not True
        or receipt["exact_gate_executed"] is not True
        or receipt["surrogate"] is not False
        or receipt["synthetic_data_only"] is not True
        or receipt["locked_outcomes_used"] is not False
        or max(null_upper_bounds) > maximum_false_pass
        or max(null_decision_upper_bounds) > maximum_false_pass
        or min(effect_lower_bounds) < minimum_power
        or min(effect_decision_lower_bounds) < minimum_power
        or min(null_conclusion_correct_lower_bounds) < 1.0 - maximum_false_pass
        or min(alternative_conclusion_correct_lower_bounds) < minimum_power
    ):
        raise ValueError("calibration receipt does not satisfy frozen error and power requirements")
    return {
        "available": True,
        "required": required,
        "schema": CALIBRATION_RECEIPT_SCHEMA,
        "engine": CALIBRATION_ENGINE,
        "actual_gate_entrypoint": ACTUAL_GATE_ENTRYPOINT,
        "confirmatory_scientific_settings_sha256": settings_sha256,
        "exact_gate_settings_sha256": settings_sha256,
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
        "run_contract_sha256": run_contract_sha,
        "generator_version": run_contract["generator_version"],
        "dgp_effect_spec_sha256": run_contract["dgp_effect_spec_sha256"],
        "locked_outcomes_used": False,
        "synthetic_data_only": True,
        "exact_gate_executed": True,
        "scenario_families": sorted(scenarios),
    }


__all__ = [
    "ACTUAL_GATE_ENTRYPOINT",
    "ACTUAL_GATE_REPORT_SCHEMA",
    "BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS",
    "BOUNDARY_EXPECTED_SOURCE_CONCLUSION",
    "CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES",
    "CALIBRATION_DGP_SPEC_SCHEMA",
    "CALIBRATION_ENGINE",
    "CALIBRATION_EVIDENCE_SCHEMA",
    "CALIBRATION_GENERATOR_VERSION",
    "CALIBRATION_RECEIPT_SCHEMA",
    "CALIBRATION_RUN_CONTRACT_SCHEMA",
    "COMPLETE_DESIGN_BINDING_STATUS",
    "G2_MULTIPLICITY_METHOD",
    "G3_MULTIPLICITY_METHOD",
    "GLOBAL_NULL_CONDITION",
    "PENDING_DESIGN_BINDING_STATUS",
    "PRELIMINARY_ALTERNATIVE_CONDITION",
    "REQUIRED_CALIBRATION_SCENARIOS",
    "REQUIRED_COMPLETE_GATE_CHECKS",
    "REQUIRED_CROP_FAMILY_IDS",
    "REQUIRED_HYPOTHESIS_DECISIONS",
    "REQUIRED_G3_CONTRAST_PAIRS",
    "REQUIRED_GATE_PARAMETERS",
    "REQUIRED_MORPHOLOGY_SOURCE_CONCLUSIONS",
    "REQUIRED_NUISANCE_FAMILIES",
    "REQUIRED_PERMUTATION_TRANSFORMS",
    "binomial_lower_confidence_bound",
    "binomial_upper_confidence_bound",
    "build_confirmatory_design_binding",
    "canonical_sha256",
    "confirmatory_scientific_manifest_projection",
    "current_calibration_executable_provenance",
    "exact_gate_settings_fingerprint",
    "pending_confirmatory_design_binding",
    "planned_donor_type_support_pattern_sha256",
    "required_simultaneous_confidence_level",
    "validate_calibration_receipt",
    "validate_calibration_run_contract",
    "validate_confirmatory_design_binding",
    "validate_exact_gate_settings",
]
