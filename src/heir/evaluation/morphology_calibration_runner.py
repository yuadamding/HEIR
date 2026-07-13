"""Run exact-gate calibration trials on deterministic synthetic artifacts.

The calibration compiler accepts only aggregate evidence from the production
``evaluate_morphology_ridge_gate`` entrypoint.  This module creates those
inputs, executes every frozen stress family, and checkpoints report hashes so
long production runs can resume without opening any biological artifact.

Reduced trial counts are useful for exercising the runner, but they are
explicitly non-authorizing and cannot satisfy the receipt compiler's minimum
of 1,000 trials per condition.  The runner itself never authorizes a
scientific claim; only the separately audited receipt compiler may authorize
the final-inference code path.
"""

from __future__ import annotations

import json
import math
import os
import resource
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.data import MorphologyRidgeDatasetArtifact
from heir.utils import atomic_json_dump

from .control_models import HEST_CROP_CONTRACT
from .morphology_artifact_qc import (
    locked_measurement_audit_report,
    reference_evaluation_balance_report,
)
from .morphology_gate import (
    evaluate_morphology_ridge_gate,
    morphology_artifact_content_sha256,
)
from .power import (
    ACTUAL_GATE_ENTRYPOINT,
    AUTHORITATIVE_BOUNDARY_COMPONENT_R2,
    AUTHORITATIVE_MIXED_TOTAL_R2,
    BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS,
    BOUNDARY_EXPECTED_SOURCE_CONCLUSION,
    CALIBRATION_DGP_SPEC_SCHEMA,
    CALIBRATION_ENGINE,
    CALIBRATION_EVIDENCE_SCHEMA,
    CALIBRATION_GENERATOR_VERSION,
    CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES,
    CALIBRATION_RUN_CONTRACT_SCHEMA,
    CALIBRATION_TRIAL_REPORT_MANIFEST_SCHEMA,
    CALIBRATION_TRIAL_REPORT_STORAGE_LAYOUT,
    GLOBAL_NULL_CONDITION,
    PRELIMINARY_ALTERNATIVE_CONDITION,
    REQUIRED_CALIBRATION_SCENARIOS,
    REQUIRED_HYPOTHESIS_DECISIONS,
    actual_gate_trial_outcome,
    build_confirmatory_design_binding,
    calibration_trial_seed,
    canonical_sha256,
    current_calibration_executable_provenance,
    validate_calibration_run_contract,
    validate_confirmatory_design_binding,
    validate_exact_gate_settings,
)

PRODUCTION_TRIALS_PER_CONDITION = 1000
PRELIMINARY_CALIBRATION_CONDITIONS = (
    GLOBAL_NULL_CONDITION,
    PRELIMINARY_ALTERNATIVE_CONDITION,
)
AUTHORIZING_CALIBRATION_CONDITIONS = (
    GLOBAL_NULL_CONDITION,
    *(BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS[name] for name in REQUIRED_HYPOTHESIS_DECISIONS),
)
# Backwards-compatible public name for the explicitly diagnostic smoke mode.
CALIBRATION_CONDITIONS = PRELIMINARY_CALIBRATION_CONDITIONS
CALIBRATION_CHECKPOINT_SCHEMA = "heir.morphology_gate_calibration_checkpoint.v3"
CALIBRATION_EXECUTION_SCHEMA = "heir.morphology_gate_calibration_execution.v4"
DEDICATED_PROCESS_ENV = "HEIR_CALIBRATION_DEDICATED_PROCESS"

PRELIMINARY_CROP_SIGNAL_SCALES = {
    "crop_112um": 1.00,
    "nucleus_mask_only": 0.55,
    "nucleus_mask_mean_fill_112um": 0.52,
    "nucleus_mask_blurred_112um": 0.08,
    "nucleus_shape_random_location_mean_fill_112um": 0.04,
    "cell_mask_only": 0.65,
    "cell_mask_mean_fill_112um": 0.62,
    "cell_mask_blurred_112um": 0.10,
    "cell_shape_random_location_mean_fill_112um": 0.05,
    "context_ring_32_to_112um": 0.42,
    "context_ring_64_to_112um": 0.48,
    "target_cell_removed_112um": 0.50,
    "target_cell_removed_mean_fill_112um": 0.48,
    "target_cell_removed_blurred_112um": 0.08,
    "random_location_cell_removed_mean_fill_112um": 0.04,
    "crop_32um": 0.74,
    "crop_64um": 0.88,
    "blank_patch": 0.00,
}

_NONBLANK_CROPS = frozenset(HEST_CROP_CONTRACT) - {"blank_patch"}
_NUCLEUS_SIGNAL_CROPS = frozenset(
    {
        "crop_112um",
        "crop_32um",
        "crop_64um",
        "nucleus_mask_only",
        "nucleus_mask_mean_fill_112um",
    }
)
_CELL_SIGNAL_CROPS = frozenset(
    {
        "crop_112um",
        "crop_32um",
        "crop_64um",
        "cell_mask_only",
        "cell_mask_mean_fill_112um",
    }
)
_CONTEXT_SIGNAL_CROPS = frozenset(
    {
        "crop_112um",
        "crop_64um",
        "context_ring_32_to_112um",
        "context_ring_64_to_112um",
        "target_cell_removed_112um",
        "target_cell_removed_mean_fill_112um",
    }
)
PRELIMINARY_DGP_EFFECT_SPEC = {
    "schema": CALIBRATION_DGP_SPEC_SCHEMA,
    "authorizing_boundary_calibration": False,
    "null_condition_id": GLOBAL_NULL_CONDITION,
    "alternative_condition_id": PRELIMINARY_ALTERNATIVE_CONDITION,
    "boundary_condition_ids_by_hypothesis": {},
    "decision_truth_by_condition": {
        GLOBAL_NULL_CONDITION: {
            decision_id: False for decision_id in REQUIRED_HYPOTHESIS_DECISIONS
        },
        PRELIMINARY_ALTERNATIVE_CONDITION: {
            decision_id: decision_id in {"G2_local_context", "G3_mixed_intrinsic_context"}
            for decision_id in REQUIRED_HYPOTHESIS_DECISIONS
        },
    },
    "effect_definition": {
        "molecular_null": "independent_standard_normal_latent_times_shared_target_weights",
        "molecular_alternative": "unit_strength_shared_latent_times_shared_target_weights",
        "morphology_noise_sd": 0.08,
        "default_molecular_noise_sd": 0.18,
        "crop_signal_scales": PRELIMINARY_CROP_SIGNAL_SCALES,
        "scientific_interpretation": (
            "arbitrary full shared-latent stress alternative; not a minimum meaningful effect"
        ),
        "condition_definitions": {
            GLOBAL_NULL_CONDITION: {
                "molecular_signal": "independent_standard_normal_latent",
                "effect_scale": 0.0,
            },
            PRELIMINARY_ALTERNATIVE_CONDITION: {
                "molecular_signal": "unit_strength_shared_latent",
                "effect_scale": 1.0,
            },
        },
    },
    "expected_source_conclusion_by_condition": {
        GLOBAL_NULL_CONDITION: "no_morphology_specific_information",
        PRELIMINARY_ALTERNATIVE_CONDITION: "mixed_intrinsic_and_contextual_information",
    },
    "hypothesis_specific_boundary_sha256": {},
}


def _authorizing_dgp_effect_spec() -> Mapping[str, object]:
    """Return the frozen six-condition quantitative boundary truth matrix."""

    single_coefficient = math.sqrt(
        AUTHORITATIVE_BOUNDARY_COMPONENT_R2 / (1.0 - AUTHORITATIVE_BOUNDARY_COMPONENT_R2)
    )
    mixed_coefficient = math.sqrt(
        AUTHORITATIVE_BOUNDARY_COMPONENT_R2 / (1.0 - AUTHORITATIVE_MIXED_TOTAL_R2)
    )
    condition_definitions: dict[str, Mapping[str, object]] = {
        GLOBAL_NULL_CONDITION: {
            "boundary_kind": "global_null",
            "target_component_population_r2": {},
            "target_component_coefficients": {},
            "total_morphology_population_r2": 0.0,
        },
        BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS["G2_local_context"]: {
            "boundary_kind": "minimum_meaningful_local_context",
            "target_component_population_r2": {
                "shared_local_context": AUTHORITATIVE_BOUNDARY_COMPONENT_R2,
            },
            "target_component_coefficients": {
                "shared_local_context": single_coefficient,
            },
            "total_morphology_population_r2": AUTHORITATIVE_BOUNDARY_COMPONENT_R2,
            "source_identification_truth": "none_all_nonblank_arms_share_signal",
        },
        BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS["G3_nucleus_intrinsic"]: {
            "boundary_kind": "minimum_meaningful_nucleus_only",
            "target_component_population_r2": {
                "nucleus_intrinsic": AUTHORITATIVE_BOUNDARY_COMPONENT_R2,
            },
            "target_component_coefficients": {"nucleus_intrinsic": single_coefficient},
            "total_morphology_population_r2": AUTHORITATIVE_BOUNDARY_COMPONENT_R2,
        },
        BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS["G3_cell_intrinsic"]: {
            "boundary_kind": "minimum_meaningful_cell_only",
            "target_component_population_r2": {
                "cell_intrinsic": AUTHORITATIVE_BOUNDARY_COMPONENT_R2,
            },
            "target_component_coefficients": {"cell_intrinsic": single_coefficient},
            "total_morphology_population_r2": AUTHORITATIVE_BOUNDARY_COMPONENT_R2,
        },
        BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS["G3_context_only"]: {
            "boundary_kind": "minimum_meaningful_context_only",
            "target_component_population_r2": {
                "extrinsic_context": AUTHORITATIVE_BOUNDARY_COMPONENT_R2,
            },
            "target_component_coefficients": {"extrinsic_context": single_coefficient},
            "total_morphology_population_r2": AUTHORITATIVE_BOUNDARY_COMPONENT_R2,
        },
        BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS["G3_mixed_intrinsic_context"]: {
            "boundary_kind": "minimum_meaningful_mixed_nucleus_context",
            "target_component_population_r2": {
                "nucleus_intrinsic": AUTHORITATIVE_BOUNDARY_COMPONENT_R2,
                "extrinsic_context": AUTHORITATIVE_BOUNDARY_COMPONENT_R2,
            },
            "target_component_coefficients": {
                "nucleus_intrinsic": mixed_coefficient,
                "extrinsic_context": mixed_coefficient,
            },
            "total_morphology_population_r2": AUTHORITATIVE_MIXED_TOTAL_R2,
            "incremental_population_r2_per_source": AUTHORITATIVE_BOUNDARY_COMPONENT_R2,
        },
    }
    decision_truth = {
        GLOBAL_NULL_CONDITION: {decision_id: False for decision_id in REQUIRED_HYPOTHESIS_DECISIONS}
    }
    for decision_id, condition_id in BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS.items():
        true_decisions = {"G2_local_context", decision_id}
        if decision_id == "G3_mixed_intrinsic_context":
            true_decisions.update({"G3_nucleus_intrinsic", "G3_context_only"})
        decision_truth[condition_id] = {
            candidate: candidate in true_decisions for candidate in REQUIRED_HYPOTHESIS_DECISIONS
        }
    expected_conclusions = {
        GLOBAL_NULL_CONDITION: "no_morphology_specific_information",
        **{
            condition_id: BOUNDARY_EXPECTED_SOURCE_CONCLUSION[decision_id]
            for decision_id, condition_id in BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS.items()
        },
    }
    spec = {
        "schema": CALIBRATION_DGP_SPEC_SCHEMA,
        "authorizing_boundary_calibration": True,
        "null_condition_id": GLOBAL_NULL_CONDITION,
        "alternative_condition_id": None,
        "boundary_condition_ids_by_hypothesis": dict(BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS),
        "decision_truth_by_condition": decision_truth,
        "effect_definition": {
            "schema": "heir.quantitative_morphology_boundary.v1",
            "independent_standard_normal_components": [
                "shared_local_context",
                "nucleus_intrinsic",
                "cell_intrinsic",
                "extrinsic_context",
            ],
            "unit_variance_molecular_residual": True,
            "crop_feature_noise_sd": 0.08,
            "crop_family_multiplicity_signal_scale_range": [0.94, 1.06],
            "population_boundary_parameterization": (
                "coefficient=sqrt(component_r2/(1-total_morphology_r2))"
            ),
            "condition_definitions": condition_definitions,
        },
        "expected_source_conclusion_by_condition": expected_conclusions,
        "hypothesis_specific_boundary_sha256": {
            decision_id: canonical_sha256(condition_definitions[condition_id])
            for decision_id, condition_id in BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS.items()
        },
    }
    return spec


AUTHORIZING_DGP_EFFECT_SPEC = _authorizing_dgp_effect_spec()


def synthetic_completed_confirmatory_design_binding() -> Mapping[str, object]:
    """Return the outcome-free completed design used only by synthetic trials/tests."""

    development = tuple("development_%d" % index for index in range(10))
    locked = tuple("locked_%d" % index for index in range(5))
    fine_types = ("epithelial", "immune")
    genes = tuple("SYNTHETIC_GENE_%d" % index for index in range(8))
    planned_strata = tuple(
        "%s|%s_section_%d|%s" % (donor, donor, section_index, fine_type)
        for donor in development + locked
        for section_index in range(2)
        for fine_type in fine_types
    )
    root = Path(__file__).resolve().parents[3]
    try:
        manifest = json.loads(
            (root / "manifests/studies/hest_lung_cell_association.draft.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "synthetic calibration requires the frozen HEST scientific manifest"
        ) from error
    measurement_receipt_sha256 = canonical_sha256(
        ["synthetic_calibration_design", "measurement_receipt"]
    )
    encoder_manifest_sha256 = canonical_sha256(["synthetic_calibration_design", "encoder_manifest"])
    crop_manifest_sha256 = canonical_sha256(["synthetic_calibration_design", "crop_manifest"])
    manifest["partitions"]["development_donors"] = list(development)
    manifest["partitions"]["locked_test_donors"] = list(locked)
    manifest["target_gene_panel_sha256"] = canonical_sha256(list(genes))
    manifest["observations"]["supported_fine_type_ids"] = list(fine_types)
    manifest["observations"]["supported_fine_type_ids_sha256"] = canonical_sha256(list(fine_types))
    manifest["prerequisites"]["measurement_report_sha256"] = measurement_receipt_sha256
    manifest["encoder"]["manifest_sha256"] = encoder_manifest_sha256
    manifest["crop_protocols"] = [crop_manifest_sha256]
    minimum_support = int(
        manifest["coverage_requirements"]["minimum_evaluation_cells_per_donor_section_type"]
    )
    return build_confirmatory_design_binding(
        manifest,
        measurement_receipt_sha256=measurement_receipt_sha256,
        ordered_target_gene_ids=genes,
        supported_fine_type_ids=fine_types,
        ordered_planned_stratum_ids=planned_strata,
        planned_stratum_minimum_evaluation_cells=(minimum_support,) * len(planned_strata),
    )


@dataclass(frozen=True)
class CalibrationRunPlan:
    """Frozen inputs for a resumable calibration execution."""

    exact_gate_settings: Mapping[str, object]
    trials_per_condition: int = PRODUCTION_TRIALS_PER_CONDITION
    base_seed: int = 1729
    device: str = "auto"
    smoke_test: bool = False
    checkpoint_every: int = 1
    max_cpu_threads: int = 1
    maximum_process_rss_gib: float = 16.0
    maximum_address_space_gib: float = 64.0

    def validate(self) -> Mapping[str, object]:
        settings = validate_exact_gate_settings(self.exact_gate_settings)
        if isinstance(self.trials_per_condition, bool) or self.trials_per_condition < 1:
            raise ValueError("calibration trials_per_condition must be a positive integer")
        if int(self.trials_per_condition) != self.trials_per_condition:
            raise ValueError("calibration trials_per_condition must be a positive integer")
        if self.trials_per_condition < PRODUCTION_TRIALS_PER_CONDITION and not self.smoke_test:
            raise ValueError(
                "reduced calibration trials require smoke_test=True and are non-authorizing"
            )
        if self.smoke_test and self.trials_per_condition >= PRODUCTION_TRIALS_PER_CONDITION:
            raise ValueError("smoke_test=True is restricted to non-authorizing reduced trials")
        if isinstance(self.base_seed, bool) or int(self.base_seed) != self.base_seed:
            raise ValueError("calibration base_seed must be an integer")
        if self.checkpoint_every != 1:
            raise ValueError("resource-safe calibration requires checkpoint_every=1")
        if (
            isinstance(self.max_cpu_threads, bool)
            or int(self.max_cpu_threads) != self.max_cpu_threads
            or not 1 <= int(self.max_cpu_threads) <= max(int(os.cpu_count() or 1), 1)
        ):
            raise ValueError("calibration max_cpu_threads is outside the available CPU range")
        if (
            isinstance(self.maximum_process_rss_gib, bool)
            or not np.isfinite(float(self.maximum_process_rss_gib))
            or float(self.maximum_process_rss_gib) <= 0.0
        ):
            raise ValueError("calibration maximum_process_rss_gib must be positive")
        if (
            isinstance(self.maximum_address_space_gib, bool)
            or not np.isfinite(float(self.maximum_address_space_gib))
            or float(self.maximum_address_space_gib) <= 0.0
        ):
            raise ValueError("calibration maximum_address_space_gib must be positive")
        if not str(self.device).strip():
            raise ValueError("calibration device must be explicit")
        return settings

    @property
    def production_contract_satisfied(self) -> bool:
        return bool(
            not self.smoke_test and self.trials_per_condition >= PRODUCTION_TRIALS_PER_CONDITION
        )


def load_calibration_run_config(path: Path) -> Mapping[str, object]:
    """Load and validate the v2 calibration configuration's gate settings."""

    try:
        content = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("calibration configuration is not valid JSON") from error
    if not isinstance(content, Mapping) or set(content) != {
        "schema",
        "confidence_level",
        "exact_gate_settings",
        "thresholds",
    }:
        raise ValueError("calibration configuration is incomplete or contains extras")
    if content["schema"] != "heir.morphology_gate_calibration_config.v2":
        raise ValueError("calibration configuration schema is unsupported")
    settings = content["exact_gate_settings"]
    validate_exact_gate_settings(settings)
    return settings


def _trial_seed(
    base_seed: int,
    scenario: str,
    condition: str,
    trial_index: int,
) -> int:
    conditions = (
        PRELIMINARY_CALIBRATION_CONDITIONS
        if condition == PRELIMINARY_ALTERNATIVE_CONDITION
        else AUTHORIZING_CALIBRATION_CONDITIONS
    )
    return calibration_trial_seed(
        base_seed,
        scenario,
        condition,
        trial_index,
        ordered_conditions=conditions,
    )


def _source_digest(*parts: object) -> str:
    return canonical_sha256([CALIBRATION_GENERATOR_VERSION, *parts])


def _disease_by_donor(scenario: str, role: str, donor_count: int) -> tuple[str, ...]:
    if scenario == "disease_imbalance":
        controls = donor_count - 2 if role == "development" else 1
        return ("Control",) * controls + ("Disease",) * (donor_count - controls)
    return tuple("Control" if index % 2 == 0 else "Disease" for index in range(donor_count))


def _crop_signal_scale(crop_id: str, scenario: str) -> float:
    value = PRELIMINARY_CROP_SIGNAL_SCALES[crop_id]
    if scenario == "crop_family_multiplicity" and crop_id not in {
        "crop_112um",
        "blank_patch",
    }:
        value = min(value + 0.12, 0.90)
    return value


def _synthetic_measurement_rows(
    *,
    contract: Mapping[str, object],
    donor_ids: np.ndarray,
    section_ids: np.ndarray,
    type_labels: np.ndarray,
    coordinate_features: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Simulate pre-QC geometry and recompute the production QC row mask."""

    rows = len(donor_ids)
    best_cutoff = float(contract["best_registration_quality_max_fraction_of_limit"])
    intermediate_cutoff = float(contract["intermediate_registration_quality_max_fraction_of_limit"])
    score_levels = np.asarray(
        [
            0.8 * best_cutoff,
            0.5 * (best_cutoff + intermediate_cutoff),
            0.5 * (intermediate_cutoff + 1.0),
        ],
        dtype=np.float64,
    )
    intended_scores = np.empty(rows, dtype=np.float64)
    pre_qc_outlier = np.zeros(rows, dtype=bool)
    strata = sorted(
        set(
            zip(
                donor_ids.astype(str).tolist(),
                section_ids.astype(str).tolist(),
                type_labels.astype(int).tolist(),
            )
        )
    )
    for donor, section, type_index in strata:
        selected = np.flatnonzero(
            (donor_ids.astype(str) == donor)
            & (section_ids.astype(str) == section)
            & (type_labels == type_index)
        )
        intended_scores[selected] = np.resize(score_levels, len(selected))
        # One prespecified registration failure per 63+ pre-QC rows stays
        # below the frozen five-percent section outlier limit, while ensuring
        # the calibration really exercises source-QC filtering.
        pre_qc_outlier[selected[-1]] = True
        intended_scores[selected[-1]] = 1.10

    nucleus_diameter_um = np.full(rows, 16.0, dtype=np.float64)
    nearest_neighbor_um = np.full(rows, 18.0, dtype=np.float64)
    annotation_nucleus_um = intended_scores * (
        float(contract["maximum_registration_nucleus_diameter_ratio_p95"]) * nucleus_diameter_um
    )
    annotation_cell_um = intended_scores * 10.0
    cell_nucleus_um = intended_scores * 6.0
    nucleus_area_um2 = np.pi * np.square(nucleus_diameter_um / 2.0)
    cell_area_um2 = nucleus_area_um2 / 0.30
    nucleus_inside_cell = np.ones(rows, dtype=bool)

    diameter_denominator = np.median(nucleus_diameter_um) * float(
        contract["maximum_registration_nucleus_diameter_ratio_p95"]
    )
    neighbor_denominator = np.median(nearest_neighbor_um) * float(
        contract["maximum_registration_nearest_neighbor_ratio_p95"]
    )
    diameter_scores = (
        annotation_nucleus_um / diameter_denominator
        if diameter_denominator > 0.0
        else np.full(rows, np.inf, dtype=np.float64)
    )
    neighbor_scores = (
        annotation_nucleus_um / neighbor_denominator
        if neighbor_denominator > 0.0
        else np.full(rows, np.inf, dtype=np.float64)
    )
    quality_scores = np.maximum(diameter_scores, neighbor_scores)
    quality_strata = np.full(rows, "near_threshold", dtype="<U16")
    quality_strata[quality_scores <= intermediate_cutoff] = "intermediate"
    quality_strata[quality_scores <= best_cutoff] = "best"

    registration_qc_pass = (
        (annotation_nucleus_um <= float(contract["maximum_annotation_nucleus_p95_um"]))
        & (annotation_cell_um <= float(contract["maximum_annotation_cell_p95_um"]))
        & (cell_nucleus_um <= float(contract["maximum_cell_nucleus_p95_um"]))
        & (diameter_scores <= 1.0)
        & (neighbor_scores <= 1.0)
    )
    area_ratio = nucleus_area_um2 / cell_area_um2
    segmentation_qc_pass = (
        nucleus_inside_cell
        & (area_ratio >= float(contract["minimum_nucleus_cell_area_ratio"]))
        & (area_ratio <= float(contract["maximum_nucleus_cell_area_ratio"]))
    )
    local_coordinates = np.mod(coordinate_features, 1.0)
    edge_distance = np.min(
        np.column_stack((local_coordinates, 1.0 - local_coordinates)),
        axis=1,
    )
    base_padding = np.clip(0.08 - edge_distance, 0.0, 1.0)
    crop_padding = np.column_stack(
        [
            np.clip(base_padding + 0.002 * crop_index, 0.0, 1.0)
            for crop_index in range(len(HEST_CROP_CONTRACT))
        ]
    )
    crop_qc_pass = np.all(
        crop_padding <= float(contract["maximum_crop_padding_p95"]),
        axis=1,
    )
    return {
        "annotation_nucleus_um": annotation_nucleus_um,
        "annotation_cell_um": annotation_cell_um,
        "cell_nucleus_um": cell_nucleus_um,
        "nucleus_diameter_um": nucleus_diameter_um,
        "nearest_neighbor_um": nearest_neighbor_um,
        "nucleus_area_um2": nucleus_area_um2,
        "cell_area_um2": cell_area_um2,
        "nucleus_inside_cell": nucleus_inside_cell,
        "crop_padding": crop_padding,
        "registration_quality_scores": quality_scores,
        "registration_quality_strata": quality_strata,
        "registration_qc_pass": registration_qc_pass,
        "segmentation_qc_pass": segmentation_qc_pass,
        "crop_qc_pass": crop_qc_pass,
        "qualified_qc_pass": registration_qc_pass & segmentation_qc_pass & crop_qc_pass,
        "prespecified_pre_qc_outlier": pre_qc_outlier,
    }


def _synthetic_locked_measurement_audit(
    *,
    contract: Mapping[str, object],
    scenario: str,
    trial_index: int,
    seed: int,
    donor_ids: np.ndarray,
    section_ids: np.ndarray,
    type_labels: np.ndarray,
    type_names: Sequence[str],
    gene_ids: Sequence[str],
    planned_stratum_ids: Sequence[str],
    molecular_residual: np.ndarray,
    registration_quality_scores: np.ndarray,
    technical_covariates: np.ndarray,
    coordinate_features: np.ndarray,
    measurement_rows: Optional[Mapping[str, np.ndarray]] = None,
) -> Mapping[str, object]:
    """Construct row-level inputs and run the production locked-measurement audit."""

    audit_rng = np.random.default_rng(seed + 90_001)
    if scenario == "variable_transcript_reliability":
        # Ten percent of trials deliberately cross the frozen measurement
        # boundary. This is condition-independent and measures fail-closed
        # behavior when the production H-MEAS audit rejects split halves.
        half_noise = 4.0 if trial_index % 10 == 0 else 0.55
    else:
        half_noise = 0.20

    # Simulated transcript halves are non-negative count matrices. The exact
    # production audit performs its own library-size normalization and
    # split-half reliability calculation on these row-level inputs.
    half_a_log_rates = np.clip(
        molecular_residual + audit_rng.normal(scale=half_noise, size=molecular_residual.shape),
        -6.0,
        6.0,
    )
    half_b_log_rates = np.clip(
        molecular_residual + audit_rng.normal(scale=half_noise, size=molecular_residual.shape),
        -6.0,
        6.0,
    )
    half_a_counts = np.rint(100.0 * np.exp(half_a_log_rates)).astype(np.uint32)
    half_b_counts = np.rint(100.0 * np.exp(half_b_log_rates)).astype(np.uint32)
    target_total_a = half_a_counts.sum(axis=1, dtype=np.uint64)
    target_total_b = half_b_counts.sum(axis=1, dtype=np.uint64)
    technical = np.asarray(technical_covariates, dtype=np.float64)
    if technical.ndim != 2 or len(technical) != len(donor_ids) or not np.isfinite(technical).all():
        raise ValueError("synthetic split-half library covariates are malformed")
    centered_library_proxy = technical[:, 0] - float(np.mean(technical[:, 0]))
    background_mean = 20_000.0 * np.exp(np.clip(0.15 * centered_library_proxy, -1.0, 1.0))
    background_a = audit_rng.poisson(background_mean).astype(np.uint64) + 1
    background_b = audit_rng.poisson(background_mean).astype(np.uint64) + 1
    half_a_library_sizes = target_total_a + background_a
    half_b_library_sizes = target_total_b + background_b
    if not np.all(half_a_library_sizes > target_total_a) or not np.all(
        half_b_library_sizes > target_total_b
    ):
        raise AssertionError("synthetic full-transcript libraries must exceed panel counts")

    measurement = (
        measurement_rows
        if measurement_rows is not None
        else _synthetic_measurement_rows(
            contract=contract,
            donor_ids=donor_ids,
            section_ids=section_ids,
            type_labels=type_labels,
            coordinate_features=coordinate_features,
        )
    )
    if measurement_rows is not None and not np.allclose(
        registration_quality_scores,
        np.asarray(measurement["registration_quality_scores"]),
    ):
        raise AssertionError("synthetic registration quality differs from simulated geometry")

    source_qc_pass = np.asarray(measurement["qualified_qc_pass"], dtype=bool)
    fine_type_ids = np.asarray(
        [tuple(type_names)[int(index)] for index in type_labels],
        dtype=str,
    )
    report = locked_measurement_audit_report(
        contract=contract,
        donor_ids=np.asarray(donor_ids).astype(str),
        section_ids=np.asarray(section_ids).astype(str),
        fine_type_ids=fine_type_ids,
        locked_donors=tuple(sorted(set(np.asarray(donor_ids).astype(str).tolist()))),
        supported_types=tuple(str(value) for value in type_names),
        planned_stratum_ids=tuple(str(value) for value in planned_stratum_ids),
        gene_ids=tuple(str(value) for value in gene_ids),
        half_a_counts=half_a_counts,
        half_b_counts=half_b_counts,
        half_a_library_sizes=half_a_library_sizes,
        half_b_library_sizes=half_b_library_sizes,
        source_locked_measurement_qc_pass=source_qc_pass,
        target_qc_pass=np.ones(len(donor_ids), dtype=bool),
        registration_qc_pass=np.asarray(measurement["registration_qc_pass"], dtype=bool),
        segmentation_qc_pass=np.asarray(measurement["segmentation_qc_pass"], dtype=bool),
        crop_qc_pass=np.asarray(measurement["crop_qc_pass"], dtype=bool),
        annotation_nucleus_um=np.asarray(measurement["annotation_nucleus_um"], dtype=np.float64),
        annotation_cell_um=np.asarray(measurement["annotation_cell_um"], dtype=np.float64),
        cell_nucleus_um=np.asarray(measurement["cell_nucleus_um"], dtype=np.float64),
        nucleus_area_um2=np.asarray(measurement["nucleus_area_um2"], dtype=np.float64),
        nearest_neighbor_um=np.asarray(measurement["nearest_neighbor_um"], dtype=np.float64),
        nucleus_inside_cell=np.asarray(measurement["nucleus_inside_cell"], dtype=bool),
        cell_area_um2=np.asarray(measurement["cell_area_um2"], dtype=np.float64),
        crop_ids=tuple(HEST_CROP_CONTRACT),
        crop_padding_fractions=np.asarray(measurement["crop_padding"], dtype=np.float64),
    )
    return report


def _balanced_reference_mask(
    donor_ids: np.ndarray,
    section_ids: np.ndarray,
    type_labels: np.ndarray,
    balance_groups: np.ndarray,
    continuous_features: np.ndarray,
    *,
    split_index: int,
) -> np.ndarray:
    """Assign paired, outcome-free pseudo-reference rows within every stratum."""

    reference = np.zeros(len(donor_ids), dtype=bool)
    strata = sorted(
        set(
            zip(
                donor_ids.astype(str).tolist(),
                section_ids.astype(str).tolist(),
                type_labels.astype(int).tolist(),
            )
        )
    )
    for donor, section, type_index in strata:
        indices = np.flatnonzero(
            (donor_ids.astype(str) == donor)
            & (section_ids.astype(str) == section)
            & (type_labels == type_index)
        )
        local_features = continuous_features[indices]
        local_scale = np.maximum(local_features.std(axis=0), 1.0e-8)
        standardized = (local_features - local_features.mean(axis=0)) / local_scale
        local_groups = balance_groups[indices].astype(str)
        pairs: list[tuple[int, int]] = []
        leftovers: list[int] = []
        for group in sorted(set(local_groups.tolist())):
            remaining = list(np.flatnonzero(local_groups == group))
            while len(remaining) >= 2:
                left = remaining.pop(0)
                right_position = min(
                    range(len(remaining)),
                    key=lambda position: (
                        float(
                            np.square(standardized[left] - standardized[remaining[position]]).sum()
                        ),
                        remaining[position],
                    ),
                )
                pairs.append((left, remaining.pop(right_position)))
            leftovers.extend(remaining)
        while len(leftovers) >= 2:
            left = leftovers.pop(0)
            right_position = min(
                range(len(leftovers)),
                key=lambda position: (
                    float(np.square(standardized[left] - standardized[leftovers[position]]).sum()),
                    leftovers[position],
                ),
            )
            pairs.append((left, leftovers.pop(right_position)))
        if leftovers:
            raise AssertionError("synthetic balance stratum has an odd row count")

        differences = np.asarray(
            [standardized[left] - standardized[right] for left, right in pairs],
            dtype=np.float64,
        )
        pair_order = sorted(
            range(len(pairs)),
            key=lambda index: (
                -float(np.square(differences[index]).sum()),
                pairs[index],
            ),
        )
        if pair_order:
            shift = split_index % len(pair_order)
            pair_order = pair_order[shift:] + pair_order[:shift]
            if split_index % 2:
                pair_order.reverse()

        signs = np.zeros(len(pairs), dtype=np.int8)
        running_difference = np.zeros(continuous_features.shape[1], dtype=np.float64)
        for order_index, pair_index in enumerate(pair_order):
            difference = differences[pair_index]
            positive_score = float(np.max(np.abs(running_difference + difference)))
            negative_score = float(np.max(np.abs(running_difference - difference)))
            sign = (
                1
                if (
                    positive_score < negative_score
                    or (positive_score == negative_score and (order_index + split_index) % 2 == 0)
                )
                else -1
            )
            signs[pair_index] = sign
            running_difference += sign * difference

        # Deterministic single-pair coordinate descent tightens the minimax
        # covariate imbalance without inspecting any molecular outcome.
        for _ in range(max(1, 2 * len(pairs))):
            current_score = float(np.max(np.abs(running_difference)))
            best_index = None
            best_score = current_score
            for pair_index, difference in enumerate(differences):
                candidate = running_difference - 2.0 * signs[pair_index] * difference
                candidate_score = float(np.max(np.abs(candidate)))
                if candidate_score < best_score - 1.0e-12:
                    best_index = pair_index
                    best_score = candidate_score
            if best_index is None:
                break
            running_difference -= 2.0 * signs[best_index] * differences[best_index]
            signs[best_index] *= -1

        local_reference = np.zeros(len(indices), dtype=bool)
        for pair_index, (left, right) in enumerate(pairs):
            local_reference[left if signs[pair_index] > 0 else right] = True

        # Pair orientations alone can reach a local minimax optimum. Outcome-
        # free reference/evaluation swaps within each registration-quality band
        # preserve the frozen band counts while balancing the complete feature
        # matrix used by the production report.
        for _ in range(max(1, 4 * len(indices))):
            current_score = float(np.max(np.abs(running_difference)))
            best_swap = None
            best_score = current_score
            for group in sorted(set(local_groups.tolist())):
                group_rows = local_groups == group
                reference_rows = np.flatnonzero(group_rows & local_reference)
                evaluation_rows = np.flatnonzero(group_rows & ~local_reference)
                for reference_row in reference_rows:
                    candidates = (
                        running_difference[None, :]
                        - 2.0 * standardized[reference_row][None, :]
                        + 2.0 * standardized[evaluation_rows]
                    )
                    scores = np.max(np.abs(candidates), axis=1)
                    candidate_index = int(np.argmin(scores))
                    candidate_score = float(scores[candidate_index])
                    if candidate_score < best_score - 1.0e-12:
                        best_swap = (reference_row, int(evaluation_rows[candidate_index]))
                        best_score = candidate_score
            if best_swap is None:
                break
            reference_row, evaluation_row = best_swap
            running_difference += 2.0 * (standardized[evaluation_row] - standardized[reference_row])
            local_reference[reference_row] = False
            local_reference[evaluation_row] = True

        reference[indices[local_reference]] = True
    return reference


def _synthetic_reference_evaluation_balance(
    *,
    contract: Mapping[str, object],
    split_ids: Sequence[str],
    observation_ids: Sequence[str],
    donor_ids: np.ndarray,
    section_ids: np.ndarray,
    type_labels: np.ndarray,
    type_names: Sequence[str],
    disease_states: Sequence[str],
    site_ids: Sequence[str],
    batch_ids: Sequence[str],
    balance_groups: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[Mapping[str, object], Mapping[str, str]]:
    """Construct pool memberships and run the production balance report."""

    values = {
        "section_ids": np.asarray(section_ids).astype(str),
        "disease_states": np.asarray(disease_states).astype(str),
        "site_ids": np.asarray(site_ids).astype(str),
        "batch_ids": np.asarray(batch_ids).astype(str),
    }
    reports = {}
    memberships = {}
    for split_index, split_id in enumerate(split_ids):
        reference = _balanced_reference_mask(
            donor_ids,
            section_ids,
            type_labels,
            np.asarray(balance_groups).astype(str),
            feature_matrix,
            split_index=split_index,
        )
        evaluation = ~reference
        reports[str(split_id)] = reference_evaluation_balance_report(
            values,
            reference,
            evaluation,
            np.asarray(donor_ids).astype(str),
            np.asarray(type_labels, dtype=np.int64),
            tuple(str(value) for value in type_names),
            np.asarray(feature_matrix, dtype=np.float64),
            tuple(str(value) for value in feature_names),
            float(contract["maximum_reference_evaluation_absolute_smd"]),
            float(contract["maximum_reference_evaluation_categorical_total_variation"]),
        )
        memberships[str(split_id)] = canonical_sha256(
            [
                str(observation_id)
                for observation_id, selected in zip(observation_ids, reference)
                if selected
            ]
        )
    return reports, memberships


def _synthetic_artifact(
    *,
    scenario: str,
    condition: str,
    trial_index: int,
    base_seed: int,
    role: str,
    design_binding: Mapping[str, object],
    shared_target_weights: np.ndarray,
    component_feature_weights: Mapping[str, np.ndarray],
) -> MorphologyRidgeDatasetArtifact:
    """Construct one valid, explicitly synthetic side of a calibration trial."""

    if role not in {"development", "locked_test"}:
        raise ValueError("synthetic calibration role is unsupported")
    seed = _trial_seed(base_seed, scenario, condition, trial_index)
    role_offset = 101 if role == "development" else 211
    rng = np.random.default_rng(seed + role_offset)
    binding = validate_confirmatory_design_binding(design_binding)
    donors = tuple(
        str(value)
        for value in binding[
            "development_donor_ids" if role == "development" else "locked_test_donor_ids"
        ]
    )
    type_names = tuple(str(value) for value in binding["ordered_supported_fine_type_ids"])
    gene_ids = tuple(str(value) for value in binding["ordered_target_gene_ids"])
    donor_count = len(donors)
    gene_count = len(gene_ids)
    disease_lookup = dict(zip(donors, _disease_by_donor(scenario, role, donor_count)))
    donor_index_by_id = {donor: index for index, donor in enumerate(donors)}
    type_index_by_id = {fine_type: index for index, fine_type in enumerate(type_names)}
    bound_strata = tuple(str(value) for value in binding["ordered_planned_stratum_ids"])
    bound_support = tuple(
        int(value) for value in binding["planned_stratum_minimum_evaluation_cells"]
    )
    role_strata: list[tuple[str, str, str, int, int]] = []
    section_index_by_key: dict[tuple[str, str], int] = {}
    for stratum_id, minimum_count in zip(bound_strata, bound_support):
        donor, section, fine_type = stratum_id.split("|")
        if donor not in donor_index_by_id:
            continue
        section_key = (donor, section)
        if section_key not in section_index_by_key:
            section_index_by_key[section_key] = sum(
                existing_donor == donor
                for existing_donor, _existing_section in section_index_by_key
            )
        role_strata.append(
            (
                stratum_id,
                donor,
                section,
                type_index_by_id[fine_type],
                minimum_count,
            )
        )
    planned_strata = tuple(value[0] for value in role_strata)

    observation_ids: list[str] = []
    donor_ids: list[str] = []
    block_ids: list[str] = []
    roi_ids: list[str] = []
    section_ids: list[str] = []
    disease_states: list[str] = []
    site_ids: list[str] = []
    batch_ids: list[str] = []
    section_ordinals: list[int] = []
    labels: list[int] = []
    component_latent_rows: dict[str, list[np.ndarray]] = {
        "shared_local_context": [],
        "nucleus_intrinsic": [],
        "cell_intrinsic": [],
        "extrinsic_context": [],
    }
    independent_target_latent_rows: list[np.ndarray] = []
    coordinate_rows: list[tuple[float, float]] = []
    technical_rows: list[tuple[float, float]] = []

    for _stratum_id, donor, section, type_index, minimum_count in role_strata:
        donor_index = donor_index_by_id[donor]
        section_index = section_index_by_key[(donor, section)]
        if (
            scenario == "missing_fine_types"
            and role == "locked_test"
            and donor_index == 0
            and type_index == len(type_names) - 1
        ):
            continue
        required_pre_qc_support = max(
            minimum_count,
            int(binding["locked_measurement_audit_contract"]["minimum_reliability_rows"]),
            3 * minimum_count,
        )
        calibration_band_minimum = required_pre_qc_support + (3 - required_pre_qc_support % 3)
        if scenario == "section_effects":
            row_count = calibration_band_minimum
        elif scenario == "unbalanced_donor_cell_counts":
            imbalance_index = (
                donor_count - donor_index - 1 if role == "locked_test" else donor_index
            )
            row_count = calibration_band_minimum + 2 * imbalance_index
        else:
            row_count = calibration_band_minimum
        for row_index in range(row_count):
            block_index = row_index // 4
            # Freeze exactly 5% of the default synthetic development rows into
            # singleton ROI strata.  This exercises the live 95% activity rule.
            if scenario == "inactive_permutation_strata" and donor_index == 0 and type_index == 0:
                roi_index = row_index
            else:
                roi_index = row_index // 4
            observation_ids.append(
                "%s|%s|%s|cell_%03d" % (donor, section, type_names[type_index], row_index)
            )
            donor_ids.append(donor)
            block_ids.append("%s/%s/block_%d" % (donor, section, block_index))
            roi_ids.append("%s/%s/type_%d/roi_%d" % (donor, section, type_index, roi_index))
            section_ids.append(section)
            disease_states.append(disease_lookup[donor])
            site_ids.append("site_%d" % (donor_index % 2))
            batch_ids.append("batch_%d" % ((donor_index + section_index) % 2))
            section_ordinals.append(section_index)
            labels.append(type_index)

            for component_rows in component_latent_rows.values():
                component_rows.append(rng.normal(size=6))
            independent_target_latent_rows.append(rng.normal(size=6))
            x = (row_index % 4) / 3.0 + donor_index * 0.03
            y = (row_index // 4) / max(row_count // 4, 1) + type_index * 0.05
            coordinate_rows.append((x, y))
            technical_rows.append(
                (
                    np.log1p(90.0 + 3.0 * row_index + 5.0 * type_index),
                    1.0 + 0.1 * donor_index + rng.normal(scale=0.03),
                )
            )

    component_latents = {
        name: np.asarray(values, dtype=np.float64) for name, values in component_latent_rows.items()
    }
    latent = sum(component_latents.values()) / math.sqrt(float(len(component_latents)))
    target_latent = np.asarray(independent_target_latent_rows, dtype=np.float64)
    coordinates = np.asarray(coordinate_rows, dtype=np.float64)
    technical = np.asarray(technical_rows, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    rows = len(label_array)

    independent_molecular = target_latent @ shared_target_weights
    if condition == PRELIMINARY_ALTERNATIVE_CONDITION:
        residual = component_latents["shared_local_context"] @ shared_target_weights
    elif condition == GLOBAL_NULL_CONDITION:
        residual = independent_molecular
    else:
        condition_definition = AUTHORIZING_DGP_EFFECT_SPEC["effect_definition"][
            "condition_definitions"
        ][condition]
        coefficients = condition_definition["target_component_coefficients"]
        residual = independent_molecular.copy()
        for component_name, coefficient in coefficients.items():
            residual = residual + float(coefficient) * (
                component_latents[component_name] @ shared_target_weights
            )

    if scenario == "spatial_autocorrelation":
        spatial_weights = np.linspace(0.10, 0.35, gene_count)
        residual = residual + coordinates[:, :1] * spatial_weights[None, :]
    if scenario == "section_effects":
        section_indicator = np.asarray(section_ordinals, dtype=np.float64) % 2.0
        residual = residual + section_indicator[:, None] * np.linspace(0.10, 0.25, gene_count)
    if scenario == "disease_imbalance":
        disease_indicator = np.asarray(
            [float(value == "Disease") for value in disease_states], dtype=np.float64
        )
        residual = residual + disease_indicator[:, None] * np.linspace(0.08, 0.20, gene_count)
    if scenario == "target_panel_selection":
        residual = residual * np.linspace(1.0, 0.25, gene_count)
    noise_scale = (
        np.linspace(0.0, 1.0, gene_count)
        if scenario == "variable_transcript_reliability"
        else np.zeros(gene_count)
    )
    if np.any(noise_scale):
        residual = residual + rng.normal(scale=noise_scale, size=(rows, gene_count))

    base_axis = np.linspace(-0.4, 0.3, gene_count)
    type_baseline = np.vstack(
        tuple(
            np.roll(base_axis, type_index % gene_count) + 0.08 * type_index
            for type_index in range(len(type_names))
        )
    )
    donor_positions = {donor: index for index, donor in enumerate(donors)}
    donor_baseline = np.asarray(
        [(donor_positions[value] - (donor_count - 1) / 2.0) * 0.04 for value in donor_ids]
    )
    reference_means = type_baseline[label_array] + donor_baseline[:, None]
    molecular_targets = reference_means + residual

    feature_widths = {value.shape[1] for value in component_feature_weights.values()}
    if len(feature_widths) != 1:
        raise ValueError("synthetic calibration component feature widths differ")
    feature_width = next(iter(feature_widths))
    crop_features = []
    for crop_index, crop_id in enumerate(HEST_CROP_CONTRACT):
        crop_noise_sd = (
            0.12 + 0.01 * crop_index if condition == PRELIMINARY_ALTERNATIVE_CONDITION else 0.08
        )
        crop_noise = rng.normal(
            scale=crop_noise_sd,
            size=(rows, feature_width),
        )
        if condition == PRELIMINARY_ALTERNATIVE_CONDITION:
            scale = _crop_signal_scale(crop_id, scenario)
            morphology = (
                component_latents["shared_local_context"]
                @ component_feature_weights["shared_local_context"]
            )
            if crop_id == "blank_patch":
                crop_features.append(crop_noise * 0.01)
            elif "random_location" in crop_id or "blurred" in crop_id:
                crop_features.append(scale * rng.normal(size=(rows, feature_width)) + crop_noise)
            else:
                crop_features.append(scale * morphology + crop_noise)
        else:
            component_names = []
            if crop_id in _NONBLANK_CROPS:
                component_names.append("shared_local_context")
            if crop_id in _NUCLEUS_SIGNAL_CROPS:
                component_names.append("nucleus_intrinsic")
            if crop_id in _CELL_SIGNAL_CROPS:
                component_names.append("cell_intrinsic")
            if crop_id in _CONTEXT_SIGNAL_CROPS:
                component_names.append("extrinsic_context")
            crop_signal = sum(
                (
                    component_latents[name] @ component_feature_weights[name]
                    for name in component_names
                ),
                np.zeros((rows, feature_width), dtype=np.float64),
            )
            if scenario == "crop_family_multiplicity" and crop_id != "crop_112um":
                crop_signal = crop_signal * (0.94 + 0.02 * float(crop_index % 7))
            crop_features.append(crop_signal + crop_noise)
    image_feature_tensor = np.stack(crop_features, axis=1)
    primary_index = tuple(HEST_CROP_CONTRACT).index("crop_112um")
    frozen_features = image_feature_tensor[:, primary_index, :]

    nuisance_rng = np.random.default_rng(seed + role_offset + 10_000)
    nuisance_noise = nuisance_rng.normal(size=(rows, 2))
    if scenario == "nuisance_selection" and role == "development":
        nuisance_noise[:, 0] = residual[:, 0] + nuisance_rng.normal(scale=0.25, size=rows)
    stain_features = np.column_stack(
        (nuisance_noise[:, 0], 0.2 * coordinates[:, 0] + nuisance_noise[:, 1])
    )
    nuclear_morphometrics = np.column_stack(
        (0.12 * latent[:, 0] + nuisance_rng.normal(size=rows), nuisance_rng.normal(size=rows))
    )
    cell_morphometrics = np.column_stack(
        (0.15 * latent[:, 1] + nuisance_rng.normal(size=rows), nuisance_rng.normal(size=rows))
    )
    cellvit_context = np.column_stack(
        (0.10 * latent[:, 2] + nuisance_rng.normal(size=rows), nuisance_rng.normal(size=rows))
    )
    local_density = np.column_stack(
        (
            coordinates[:, 0] + nuisance_rng.normal(scale=0.5, size=rows),
            nuisance_rng.normal(size=rows),
        )
    )
    boundary = np.column_stack((np.abs(coordinates[:, 0] - 0.5), np.abs(coordinates[:, 1] - 0.5)))
    spatial = np.column_stack(
        (coordinates, np.square(coordinates[:, 0]), np.square(coordinates[:, 1]))
    )

    split_ids = ("primary", "reference_hash_fold_0", "reference_hash_fold_1")
    reference_means_by_split = np.stack(
        (
            reference_means,
            reference_means + rng.normal(scale=0.01, size=reference_means.shape),
            reference_means + rng.normal(scale=0.01, size=reference_means.shape),
        ),
        axis=1,
    )
    shared_identity = (scenario, condition, trial_index)
    # The row count is derived from both bound contracts: every section meets
    # locked split-half reliability support and every registration-quality
    # band has at least the frozen H-CELL minimum support.
    audit_contract = binding["locked_measurement_audit_contract"]
    best_quality_cutoff = float(audit_contract["best_registration_quality_max_fraction_of_limit"])
    intermediate_quality_cutoff = float(
        audit_contract["intermediate_registration_quality_max_fraction_of_limit"]
    )
    donor_array = np.asarray(donor_ids, dtype=str)
    section_array = np.asarray(section_ids, dtype=str)
    measurement_rows = _synthetic_measurement_rows(
        contract=audit_contract,
        donor_ids=donor_array,
        section_ids=section_array,
        type_labels=label_array,
        coordinate_features=coordinates,
    )
    registration_quality_scores = np.asarray(
        measurement_rows["registration_quality_scores"],
        dtype=np.float64,
    )
    registration_quality_strata = np.asarray(
        measurement_rows["registration_quality_strata"]
    ).astype(str)
    locked_measurement_audit = (
        _synthetic_locked_measurement_audit(
            contract=binding["locked_measurement_audit_contract"],
            scenario=scenario,
            trial_index=trial_index,
            seed=seed,
            donor_ids=donor_array,
            section_ids=section_array,
            type_labels=label_array,
            type_names=type_names,
            gene_ids=gene_ids,
            planned_stratum_ids=planned_strata,
            molecular_residual=molecular_targets - reference_means,
            registration_quality_scores=registration_quality_scores,
            technical_covariates=technical,
            coordinate_features=coordinates,
            measurement_rows=measurement_rows,
        )
        if role == "locked_test"
        else None
    )
    qualified = np.asarray(measurement_rows["qualified_qc_pass"], dtype=bool)
    pre_qc_rows = len(qualified)
    if not np.any(~qualified):
        raise AssertionError("synthetic calibration failed to exercise source-QC filtering")
    observation_array = np.asarray(observation_ids, dtype=str)[qualified]
    donor_array = donor_array[qualified]
    block_array = np.asarray(block_ids, dtype=str)[qualified]
    roi_array = np.asarray(roi_ids, dtype=str)[qualified]
    section_array = section_array[qualified]
    disease_array = np.asarray(disease_states, dtype=str)[qualified]
    site_array = np.asarray(site_ids, dtype=str)[qualified]
    batch_array = np.asarray(batch_ids, dtype=str)[qualified]
    label_array = label_array[qualified]
    coordinates = coordinates[qualified]
    technical = technical[qualified]
    frozen_features = frozen_features[qualified]
    molecular_targets = molecular_targets[qualified]
    reference_means = reference_means[qualified]
    stain_features = stain_features[qualified]
    image_feature_tensor = image_feature_tensor[qualified]
    nuclear_morphometrics = nuclear_morphometrics[qualified]
    cell_morphometrics = cell_morphometrics[qualified]
    cellvit_context = cellvit_context[qualified]
    local_density = local_density[qualified]
    boundary = boundary[qualified]
    spatial = spatial[qualified]
    reference_means_by_split = reference_means_by_split[qualified]
    registration_quality_scores = registration_quality_scores[qualified]
    registration_quality_strata = registration_quality_strata[qualified]
    rows = int(np.count_nonzero(qualified))
    observed_strata = {
        "%s|%s|%s" % (donor, section, type_names[label])
        for donor, section, label in zip(donor_array, section_array, label_array)
    }
    retained_fraction = len(observed_strata) / len(planned_strata)
    balance_features = np.column_stack(
        (
            coordinates,
            technical,
            stain_features,
            nuclear_morphometrics,
            cell_morphometrics,
            cellvit_context,
            local_density,
            boundary,
            spatial,
        )
    )
    balance_feature_names = (
        "coordinate::0",
        "coordinate::1",
        "technical::log1p_library_size",
        "technical::segmentation_area_proxy",
        "stain::stain_intensity",
        "stain::stain_gradient",
        "nuclear::synthetic_nuclear_area",
        "nuclear::synthetic_nuclear_texture",
        "cell::synthetic_cell_area",
        "cell::synthetic_cell_texture",
        "cellvit::synthetic_neighbor_1",
        "cellvit::synthetic_neighbor_2",
        "density::synthetic_density_1",
        "density::synthetic_density_2",
        "boundary::synthetic_boundary_x",
        "boundary::synthetic_boundary_y",
        "spatial::x",
        "spatial::y",
        "spatial::x_squared",
        "spatial::y_squared",
    )
    reference_evaluation_balance, memberships = _synthetic_reference_evaluation_balance(
        contract=binding["reference_evaluation_balance_contract"],
        split_ids=split_ids,
        observation_ids=observation_array,
        donor_ids=donor_array,
        section_ids=section_array,
        type_labels=label_array,
        type_names=type_names,
        disease_states=disease_array,
        site_ids=site_array,
        batch_ids=batch_array,
        balance_groups=registration_quality_strata,
        feature_matrix=balance_features,
        feature_names=balance_feature_names,
    )
    artifact = MorphologyRidgeDatasetArtifact(
        observation_ids=observation_array,
        donor_ids=donor_array,
        block_ids=block_array,
        roi_ids=roi_array,
        type_labels=label_array,
        type_names=type_names,
        frozen_features=frozen_features,
        molecular_targets=molecular_targets,
        reference_means=reference_means,
        coordinate_features=coordinates,
        stain_features=stain_features,
        stain_feature_names=("stain_intensity", "stain_gradient"),
        composition_features=np.empty((rows, 0), dtype=np.float64),
        composition_feature_names=(),
        technical_covariates=technical,
        technical_covariate_names=("log1p_library_size", "segmentation_area_proxy"),
        gene_ids=gene_ids,
        type_marker_gene_ids=("SYNTHETIC_TYPE_MARKER",),
        feature_space_id="synthetic-calibration-image-features-v1",
        feature_checkpoint_sha256=_source_digest("feature_checkpoint", *shared_identity),
        molecular_space_id="synthetic-calibration-residual-expression-v1",
        reference_source_sha256=_source_digest("reference", *shared_identity, role),
        label_source_sha256=_source_digest("labels", *shared_identity),
        target_source_sha256=_source_digest("targets", *shared_identity, role),
        registration_source_sha256=_source_digest("registration", *shared_identity),
        exclusion_policy_sha256=_source_digest("exclusions", *shared_identity),
        registration_method="synthetic_one_to_one_generator",
        encoder_name="synthetic-calibration-encoder",
        crop_scale="small_cell_centered",
        cohort_id="SYNTHETIC_CALIBRATION",
        cohort_release=CALIBRATION_GENERATOR_VERSION,
        assay="synthetic_registered_cell_expression",
        observation_level="synthetic_cell",
        target_construction="synthetic_registered_cell_expression",
        reference_pool_independent=True,
        labels_independent_of_images=True,
        registration_is_one_to_one=True,
        role=role,
        section_ids=section_array,
        disease_states=disease_array,
        site_ids=site_array,
        batch_ids=batch_array,
        image_feature_tensor=image_feature_tensor,
        crop_ids=tuple(HEST_CROP_CONTRACT),
        crop_roles=tuple(value[0] for value in HEST_CROP_CONTRACT.values()),
        crop_comparison_families=tuple(value[1] for value in HEST_CROP_CONTRACT.values()),
        primary_crop_id="crop_112um",
        nuclear_morphometrics=nuclear_morphometrics,
        nuclear_morphometric_names=("synthetic_nuclear_area", "synthetic_nuclear_texture"),
        cell_morphometrics=cell_morphometrics,
        cell_morphometric_names=("synthetic_cell_area", "synthetic_cell_texture"),
        cellvit_context_features=cellvit_context,
        cellvit_context_feature_names=("synthetic_neighbor_1", "synthetic_neighbor_2"),
        local_density_features=local_density,
        local_density_feature_names=("synthetic_density_1", "synthetic_density_2"),
        boundary_features=boundary,
        boundary_feature_names=("synthetic_boundary_x", "synthetic_boundary_y"),
        spatial_control_features=spatial,
        spatial_control_feature_names=("x", "y", "x_squared", "y_squared"),
        planned_stratum_ids=planned_strata,
        planned_stratum_manifest_sha256=str(binding["planned_stratum_manifest_sha256"]),
        coverage_audit={
            "retained_fraction": retained_fraction,
            "source_rows_before_frozen_qc": pre_qc_rows,
            "evaluation_rows_after_frozen_qc": rows,
            "source_qc_filtered_rows": pre_qc_rows - rows,
            "source_qc_mask_sha256": canonical_sha256(qualified.astype(int).tolist()),
            "reference_membership_sha256_by_split": memberships,
            "locked_measurement_audit": locked_measurement_audit,
        },
        reference_evaluation_balance=reference_evaluation_balance,
        study_manifest_sha256=_source_digest("study_manifest", *shared_identity),
        measurement_receipt_sha256=str(binding["measurement_receipt_sha256"]),
        measurement_source_sha256=_source_digest("measurement_source", *shared_identity),
        hypothesis_ids=("H-CELL", "H-INTRINSIC"),
        scientific_scope="registered_cell_local_context_association",
        evidence_scope="internal_locked_hest",
        authorizes_nucleus_intrinsic_claim=False,
        registration_quality_scores=registration_quality_scores,
        registration_quality_strata=registration_quality_strata,
        registration_quality_cutoffs={
            "best": best_quality_cutoff,
            "intermediate": intermediate_quality_cutoff,
            "near_threshold": 1.0,
        },
        registration_quality_definition=(
            "max(annotation_nucleus_error/section_median_nucleus_diameter/diameter_limit,"
            "annotation_nucleus_error/section_median_nearest_neighbor_distance/neighbor_limit)"
        ),
        registration_quality_applicable=True,
        reference_split_ids=split_ids,
        reference_means_by_split=reference_means_by_split,
    )
    artifact.validate()
    return artifact


def build_synthetic_calibration_pair(
    scenario: str,
    condition: str,
    trial_index: int,
    *,
    base_seed: int = 1729,
    design_binding: Optional[Mapping[str, object]] = None,
) -> tuple[MorphologyRidgeDatasetArtifact, MorphologyRidgeDatasetArtifact]:
    """Build a development/locked pair containing synthetic values only."""

    binding = validate_confirmatory_design_binding(
        design_binding or synthetic_completed_confirmatory_design_binding()
    )
    seed = _trial_seed(base_seed, scenario, condition, trial_index)
    shared_rng = np.random.default_rng(seed)
    target_weights = shared_rng.normal(size=(6, int(binding["target_gene_count"]))) / np.sqrt(6.0)
    component_names = (
        "shared_local_context",
        "nucleus_intrinsic",
        "cell_intrinsic",
        "extrinsic_context",
    )
    feature_weights = {}
    for component_index, name in enumerate(component_names):
        weights = np.zeros((6, 24), dtype=np.float64)
        start = component_index * 6
        weights[:, start : start + 6] = shared_rng.normal(size=(6, 6)) / np.sqrt(6.0)
        feature_weights[name] = weights
    development = _synthetic_artifact(
        scenario=scenario,
        condition=condition,
        trial_index=trial_index,
        base_seed=base_seed,
        role="development",
        design_binding=binding,
        shared_target_weights=target_weights,
        component_feature_weights=feature_weights,
    )
    locked = _synthetic_artifact(
        scenario=scenario,
        condition=condition,
        trial_index=trial_index,
        base_seed=base_seed,
        role="locked_test",
        design_binding=binding,
        shared_target_weights=target_weights,
        component_feature_weights=feature_weights,
    )
    development.validate_compatible(locked)
    if len(set(locked.donor_ids.tolist())) != 5:
        raise AssertionError("synthetic calibration must contain exactly five locked donors")
    if tuple(development.crop_ids) != tuple(HEST_CROP_CONTRACT):
        raise AssertionError("synthetic calibration crop ladder differs from HEST")
    if len(development.reference_split_ids) < 3:
        raise AssertionError("synthetic calibration requires at least three reference splits")
    return development, locked


def _run_contract(plan: CalibrationRunPlan, settings: Mapping[str, object]) -> Mapping[str, object]:
    provenance = current_calibration_executable_provenance()
    dgp_spec = dict(PRELIMINARY_DGP_EFFECT_SPEC if plan.smoke_test else AUTHORIZING_DGP_EFFECT_SPEC)
    conditions = (
        PRELIMINARY_CALIBRATION_CONDITIONS
        if plan.smoke_test
        else AUTHORIZING_CALIBRATION_CONDITIONS
    )
    return {
        "schema": CALIBRATION_RUN_CONTRACT_SCHEMA,
        "generator_version": CALIBRATION_GENERATOR_VERSION,
        **provenance,
        "dgp_effect_spec": dgp_spec,
        "dgp_effect_spec_sha256": canonical_sha256(dgp_spec),
        "actual_gate_entrypoint": ACTUAL_GATE_ENTRYPOINT,
        "exact_gate_settings": dict(settings),
        "exact_gate_settings_sha256": canonical_sha256(settings),
        "permutations_per_null": int(settings["permutations_per_null"]),
        "permutation_seeds": list(settings["permutation_seeds"]),
        "permutations_per_seed": int(settings["permutations_per_seed"]),
        "scenario_families": list(REQUIRED_CALIBRATION_SCENARIOS),
        "conditions": list(conditions),
        "trials_per_condition": int(plan.trials_per_condition),
        "base_seed": int(plan.base_seed),
        "device": str(plan.device),
        "smoke_test": bool(plan.smoke_test),
        "process_isolation": (
            "dedicated_cli_process"
            if os.environ.get(DEDICATED_PROCESS_ENV) == "1"
            else "in_process_smoke"
        ),
        "max_cpu_threads": int(plan.max_cpu_threads),
        "maximum_process_rss_gib": float(plan.maximum_process_rss_gib),
        "maximum_address_space_gib": float(plan.maximum_address_space_gib),
        "trial_report_manifest_schema": CALIBRATION_TRIAL_REPORT_MANIFEST_SCHEMA,
        "trial_report_storage_layout": CALIBRATION_TRIAL_REPORT_STORAGE_LAYOUT,
    }


@contextmanager
def _cpu_thread_limit(max_cpu_threads: int):
    """Temporarily limit numerical-library CPU pools for this process."""

    threads = str(int(max_cpu_threads))
    variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    original_environment = {name: os.environ.get(name) for name in variables}
    for name in variables:
        os.environ[name] = threads
    torch = None
    original_torch_threads = None
    observed: dict[str, int | bool] = {}
    try:
        import torch

        original_torch_threads = int(torch.get_num_threads())
        torch.set_num_threads(int(max_cpu_threads))
        observed = {
            "torch_intraop_threads": int(torch.get_num_threads()),
            "torch_interop_threads": int(torch.get_num_interop_threads()),
            "torch_interop_limited_by_cpu_affinity": True,
        }
    except ImportError:
        pass
    try:
        yield observed
    finally:
        if torch is not None and original_torch_threads is not None:
            torch.set_num_threads(original_torch_threads)
        for name, original in original_environment.items():
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original


def _task_ids() -> tuple[int, ...]:
    try:
        return tuple(sorted(int(path.name) for path in Path("/proc/self/task").iterdir()))
    except (OSError, ValueError) as error:
        raise RuntimeError("calibration CPU limits require Linux task affinity") from error


@contextmanager
def _cpu_affinity_limit(max_cpu_threads: int):
    """Hard-cap this process and all existing worker tasks to N logical CPUs."""

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("calibration CPU limits require sched affinity support")
    task_ids = _task_ids()
    original = {task_id: os.sched_getaffinity(task_id) for task_id in task_ids}
    available = sorted(set.intersection(*(set(value) for value in original.values())))
    if len(available) < int(max_cpu_threads):
        raise RuntimeError("requested calibration CPUs exceed this process affinity")
    limited = set(available[: int(max_cpu_threads)])
    try:
        for task_id in task_ids:
            os.sched_setaffinity(task_id, limited)
        yield {
            "logical_cpu_ids": sorted(limited),
            "logical_cpu_count": len(limited),
            "tasks_limited_at_start": len(task_ids),
        }
    finally:
        fallback = original[task_ids[0]]
        for task_id in _task_ids():
            try:
                os.sched_setaffinity(task_id, original.get(task_id, fallback))
            except ProcessLookupError:
                pass


def _process_rss_gib() -> float:
    """Return current resident memory without adding a monitoring dependency."""

    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / (1024.0**2)
    except (OSError, IndexError, ValueError):
        pass
    # Linux reports KiB here; the calibration runtime is Linux-only in the frozen plan.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


def _process_virtual_memory_gib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmSize:"):
                return float(line.split()[1]) / (1024.0**2)
    except (OSError, IndexError, ValueError):
        pass
    raise RuntimeError("calibration hard memory limits require Linux /proc VmSize")


@contextmanager
def _address_space_limit(maximum_gib: float):
    """Apply a restorable OS-enforced soft allocation ceiling to this process."""

    if not hasattr(resource, "RLIMIT_AS"):
        raise RuntimeError("calibration hard memory limits require RLIMIT_AS")
    requested_bytes = int(float(maximum_gib) * (1024**3))
    original_soft, original_hard = resource.getrlimit(resource.RLIMIT_AS)
    effective_bytes = requested_bytes
    if original_soft != resource.RLIM_INFINITY:
        effective_bytes = min(effective_bytes, int(original_soft))
    if original_hard != resource.RLIM_INFINITY:
        effective_bytes = min(effective_bytes, int(original_hard))
    current_vms = _process_virtual_memory_gib()
    effective_gib = effective_bytes / float(1024**3)
    if current_vms > effective_gib:
        raise MemoryError(
            "calibration process virtual memory %.3f GiB already exceeds the %.3f-GiB "
            "address-space ceiling" % (current_vms, effective_gib)
        )
    resource.setrlimit(resource.RLIMIT_AS, (effective_bytes, original_hard))
    try:
        yield {
            "limit_kind": "RLIMIT_AS_soft",
            "maximum_bytes": effective_bytes,
            "maximum_gib": effective_gib,
            "requested_maximum_gib": float(maximum_gib),
            "preexisting_soft_limit_preserved": effective_bytes < requested_bytes,
            "initial_virtual_memory_gib": current_vms,
        }
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (original_soft, original_hard))


def _enforce_process_rss_limit(
    plan: CalibrationRunPlan,
    *,
    phase: str,
    observed: Optional[float] = None,
) -> None:
    observed = _process_rss_gib() if observed is None else float(observed)
    if observed > float(plan.maximum_process_rss_gib):
        raise MemoryError(
            "calibration process RSS %.3f GiB exceeds the frozen %.3f-GiB limit during %s; "
            "stop this process and resume from its last checkpoint when available"
            % (observed, float(plan.maximum_process_rss_gib), phase)
        )


def _empty_checkpoint(contract: Mapping[str, object]) -> dict[str, object]:
    checkpoint = {
        "schema": CALIBRATION_CHECKPOINT_SCHEMA,
        "run_contract": dict(contract),
        "run_contract_sha256": canonical_sha256(contract),
        "authorizes_scientific_claims": False,
        "authorizes_final_inference": False,
        "completed_trials": {
            "%s.%s" % (scenario, condition): []
            for scenario in REQUIRED_CALIBRATION_SCENARIOS
            for condition in contract["conditions"]
        },
    }
    checkpoint["checkpoint_content_sha256"] = _checkpoint_content_sha256(checkpoint)
    return checkpoint


def _checkpoint_content_sha256(checkpoint: Mapping[str, object]) -> str:
    return canonical_sha256(
        {
            str(name): value
            for name, value in checkpoint.items()
            if name != "checkpoint_content_sha256"
        }
    )


def _integer_equal(value: object, expected: int) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and int(value) == value
        and int(value) == expected
    )


def _save_checkpoint(checkpoint: dict[str, object], path: Path) -> None:
    checkpoint["checkpoint_content_sha256"] = _checkpoint_content_sha256(checkpoint)
    atomic_json_dump(checkpoint, path)


def _load_checkpoint(
    path: Path,
    contract: Mapping[str, object],
    *,
    report_store: Path,
) -> dict[str, object]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("calibration checkpoint is not valid JSON") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != (
        CALIBRATION_CHECKPOINT_SCHEMA
    ):
        raise ValueError("calibration checkpoint schema is unsupported")
    if checkpoint.get("checkpoint_content_sha256") != _checkpoint_content_sha256(checkpoint):
        raise ValueError("calibration checkpoint content hash differs")
    expected_hash = canonical_sha256(contract)
    if (
        checkpoint.get("run_contract_sha256") != expected_hash
        or checkpoint.get("run_contract") != contract
    ):
        raise ValueError("calibration checkpoint belongs to a different run contract")
    if (
        checkpoint.get("authorizes_scientific_claims") is not False
        or checkpoint.get("authorizes_final_inference") is not False
    ):
        raise ValueError("calibration checkpoint contains an invalid authorization state")
    expected_keys = {
        "%s.%s" % (scenario, condition)
        for scenario in REQUIRED_CALIBRATION_SCENARIOS
        for condition in contract["conditions"]
    }
    completed = checkpoint.get("completed_trials")
    if not isinstance(completed, dict) or set(completed) != expected_keys:
        raise ValueError("calibration checkpoint trial families are incomplete")
    for key, values in completed.items():
        if not isinstance(values, list):
            raise ValueError("calibration checkpoint trial list is malformed")
        indices = [value.get("trial_index") for value in values if isinstance(value, Mapping)]
        if len(indices) != len(values) or indices != list(range(len(values))):
            raise ValueError("calibration checkpoint trial sequence is not contiguous: %s" % key)
        for record in values:
            report_hash = str(record.get("actual_report_sha256", ""))
            decisions = record.get("hypothesis_decision_passes")
            if (
                set(record)
                != {
                    "trial_index",
                    "calibration_trial_identity",
                    "calibration_run_contract_sha256",
                    "calibration_development_artifact_sha256",
                    "calibration_locked_artifact_sha256",
                    "calibration_trial_realization_sha256",
                    "component_pass",
                    "actual_report_sha256",
                    "actual_report_relative_path",
                    "exact_gate_settings_sha256",
                    "required_checks_present",
                    "hypothesis_decision_passes",
                    "any_false_hypothesis_decision",
                    "morphology_source_conclusion",
                    "scientific_authorization_suppressed",
                    "local_roi_permutations",
                    "spatial_block_permutations",
                    "local_roi_seed_counts",
                    "spatial_block_seed_counts",
                }
                or not isinstance(record.get("component_pass"), bool)
                or len(report_hash) != 64
                or any(character not in "0123456789abcdef" for character in report_hash)
                or record.get("actual_report_relative_path")
                != _report_relative_path(report_hash).as_posix()
                or record.get("calibration_run_contract_sha256") != expected_hash
                or record.get("exact_gate_settings_sha256")
                != contract["exact_gate_settings_sha256"]
                or record.get("required_checks_present") is not True
                or record.get("scientific_authorization_suppressed") is not True
                or not isinstance(decisions, Mapping)
                or set(decisions) != set(REQUIRED_HYPOTHESIS_DECISIONS)
                or any(not isinstance(value, bool) for value in decisions.values())
                or not isinstance(record.get("any_false_hypothesis_decision"), bool)
                or record.get("morphology_source_conclusion")
                not in CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES
                or not _integer_equal(
                    record.get("local_roi_permutations"),
                    int(contract["permutations_per_null"]),
                )
                or not _integer_equal(
                    record.get("spatial_block_permutations"),
                    int(contract["permutations_per_null"]),
                )
                or record.get("local_roi_seed_counts")
                != {
                    str(seed): int(contract["exact_gate_settings"]["permutations_per_seed"])
                    for seed in contract["exact_gate_settings"]["permutation_seeds"]
                }
                or record.get("spatial_block_seed_counts")
                != {
                    str(seed): int(contract["exact_gate_settings"]["permutations_per_seed"])
                    for seed in contract["exact_gate_settings"]["permutation_seeds"]
                }
            ):
                raise ValueError("calibration checkpoint trial record is malformed: %s" % key)
            scenario, condition = key.split(".", 1)
            expected_identity = _trial_identity(
                scenario=scenario,
                condition=condition,
                trial_index=int(record["trial_index"]),
                contract=contract,
            )
            if record.get("calibration_trial_identity") != expected_identity:
                raise ValueError("calibration checkpoint trial identity differs: %s" % key)
            preserved = _read_stored_actual_gate_report(report_store, report_hash)
            outcome = actual_gate_trial_outcome(
                preserved,
                exact_gate_settings=contract["exact_gate_settings"],
                expected_trial_identity=expected_identity,
                expected_run_contract_sha256=expected_hash,
                expected_decision_truth=contract["dgp_effect_spec"]["decision_truth_by_condition"][
                    condition
                ],
            )
            if any(record[field] != outcome[field] for field in outcome):
                raise ValueError(
                    "calibration checkpoint summary differs from preserved report: %s" % key
                )
    return checkpoint


def _report_relative_path(report_sha256: str) -> Path:
    return Path(report_sha256[:2]) / (report_sha256 + ".json")


def _trial_identity(
    *,
    scenario: str,
    condition: str,
    trial_index: int,
    contract: Mapping[str, object],
) -> Mapping[str, object]:
    return {
        "scenario": scenario,
        "condition": condition,
        "trial_index": int(trial_index),
        "trial_seed": calibration_trial_seed(
            int(contract["base_seed"]),
            scenario,
            condition,
            int(trial_index),
            ordered_conditions=tuple(str(value) for value in contract["conditions"]),
        ),
    }


def _attest_actual_gate_report(
    report: Mapping[str, object],
    *,
    identity: Mapping[str, object],
    run_contract_sha256: str,
    development_artifact_sha256: str,
    locked_artifact_sha256: str,
) -> Mapping[str, object]:
    realization = canonical_sha256(
        {
            "calibration_trial_identity": dict(identity),
            "calibration_run_contract_sha256": run_contract_sha256,
            "development_artifact_sha256": development_artifact_sha256,
            "locked_artifact_sha256": locked_artifact_sha256,
        }
    )
    if (
        report.get("calibration_trial_identity") != identity
        or report.get("calibration_run_contract_sha256") != run_contract_sha256
        or report.get("calibration_development_artifact_sha256") != development_artifact_sha256
        or report.get("calibration_locked_artifact_sha256") != locked_artifact_sha256
        or report.get("calibration_trial_realization_sha256") != realization
    ):
        raise ValueError("actual gate report differs from its input realization attestation")
    return dict(report)


def _persist_actual_gate_report(
    report: Mapping[str, object],
    *,
    report_store: Path,
) -> tuple[str, str]:
    """Write one immutable report under its canonical JSON content hash."""

    report_sha256 = canonical_sha256(report)
    relative = _report_relative_path(report_sha256)
    destination = report_store / relative
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("stored actual-gate report is not valid JSON") from error
        if canonical_sha256(existing) != report_sha256 or existing != report:
            raise ValueError("stored actual-gate report differs from its content address")
    else:
        atomic_json_dump(dict(report), destination)
    return report_sha256, relative.as_posix()


def _read_stored_actual_gate_report(report_store: Path, report_sha256: str) -> Mapping[str, object]:
    path = (report_store / _report_relative_path(report_sha256)).resolve()
    root = report_store.resolve()
    if root not in path.parents:
        raise ValueError("actual-gate report content address escapes its report store")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("preserved actual-gate report is missing or invalid") from error
    if not isinstance(report, Mapping) or canonical_sha256(report) != report_sha256:
        raise ValueError("preserved actual-gate report differs from its content address")
    return report


def _trial_record(
    report: Mapping[str, object],
    *,
    trial_index: int,
    settings_sha256: str,
    settings: Mapping[str, object],
    expected_trial_identity: Mapping[str, object],
    run_contract_sha256: str,
    expected_decision_truth: Mapping[str, bool],
) -> Mapping[str, object]:
    outcome = actual_gate_trial_outcome(
        report,
        exact_gate_settings=settings,
        expected_trial_identity=expected_trial_identity,
        expected_run_contract_sha256=run_contract_sha256,
        expected_decision_truth=expected_decision_truth,
    )
    if outcome["exact_gate_settings_sha256"] != settings_sha256:
        raise AssertionError("validated trial outcome lost its exact settings binding")
    return {
        "trial_index": int(trial_index),
        **outcome,
    }


def _condition_evidence(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not records:
        raise ValueError("calibration condition has no completed production-gate trials")
    for field in ("local_roi_seed_counts", "spatial_block_seed_counts"):
        if any(record[field] != records[0][field] for record in records):
            raise ValueError("calibration trials used inconsistent permutation seed counts")
    passes = sum(bool(record["component_pass"]) for record in records)
    report_hashes = [str(record["actual_report_sha256"]) for record in records]
    realization_hashes = [str(record["calibration_trial_realization_sha256"]) for record in records]
    return {
        "trials": len(records),
        "complete_gate_passes": passes,
        "hypothesis_decision_passes": {
            name: sum(bool(record["hypothesis_decision_passes"][name]) for record in records)
            for name in REQUIRED_HYPOTHESIS_DECISIONS
        },
        "any_false_hypothesis_decision_passes": sum(
            bool(record["any_false_hypothesis_decision"]) for record in records
        ),
        "morphology_source_conclusion_counts": {
            name: sum(record["morphology_source_conclusion"] == name for record in records)
            for name in CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES
        },
        "actual_gate_executions": len(records),
        "trial_report_set_sha256": canonical_sha256(
            {"ordered_actual_gate_report_sha256": report_hashes}
        ),
        "trial_realization_set_sha256": canonical_sha256(
            {"ordered_trial_realization_sha256": realization_hashes}
        ),
        "all_trial_reports_use_exact_settings": all(
            record.get("exact_gate_settings_sha256") == records[0]["exact_gate_settings_sha256"]
            for record in records
        ),
        "all_trial_reports_include_required_checks": all(
            record.get("required_checks_present") is True for record in records
        ),
        "permutation_nulls": {
            "local_roi_permutations": min(
                int(record["local_roi_permutations"]) for record in records
            ),
            "spatial_block_permutations": min(
                int(record["spatial_block_permutations"]) for record in records
            ),
            "local_roi_seed_counts": dict(records[0]["local_roi_seed_counts"]),
            "spatial_block_seed_counts": dict(records[0]["spatial_block_seed_counts"]),
        },
    }


def _trial_report_manifest(
    completed: Mapping[str, object],
    *,
    conditions: Sequence[str],
    report_store: Path,
) -> Mapping[str, object]:
    ordered = {}
    all_hashes: list[str] = []
    for scenario in REQUIRED_CALIBRATION_SCENARIOS:
        ordered[scenario] = {}
        for condition in conditions:
            records = completed["%s.%s" % (scenario, condition)]
            if not isinstance(records, list):
                raise AssertionError("validated checkpoint lost a trial-report list")
            hashes = [str(record["actual_report_sha256"]) for record in records]
            ordered[scenario][condition] = hashes
            all_hashes.extend(hashes)
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("calibration trial reports are not uniquely bound to trial identities")
    core = {
        "schema": CALIBRATION_TRIAL_REPORT_MANIFEST_SCHEMA,
        "storage": {
            "kind": "content_addressed_directory",
            "layout": CALIBRATION_TRIAL_REPORT_STORAGE_LAYOUT,
            "root_path": str(report_store.resolve()),
        },
        "ordered_report_sha256s_by_scenario_condition": ordered,
        "report_reference_count": len(all_hashes),
        "unique_report_count": len(set(all_hashes)),
    }
    return {**core, "manifest_content_sha256": canonical_sha256(core)}


def run_actual_gate_calibration(
    plan: CalibrationRunPlan,
    *,
    checkpoint_path: Optional[Path] = None,
    report_store_path: Optional[Path] = None,
    resume: bool = True,
) -> Mapping[str, object]:
    """Execute calibration under active CPU-pool and process-RSS limits."""

    settings = dict(plan.validate())
    dedicated_process = os.environ.get(DEDICATED_PROCESS_ENV) == "1"
    if not plan.smoke_test and not dedicated_process:
        raise RuntimeError(
            "non-smoke calibration must run through the dedicated calibration CLI process"
        )
    if report_store_path is None and checkpoint_path is None:
        raise ValueError(
            "calibration requires a persistent checkpoint or explicit trial-report store"
        )
    checkpoint_resolved = checkpoint_path.expanduser().resolve() if checkpoint_path else None
    report_store = (
        report_store_path.expanduser().resolve()
        if report_store_path is not None
        else checkpoint_resolved.parent / (checkpoint_resolved.name + ".reports")
    )
    report_store.mkdir(parents=True, exist_ok=True)
    with _address_space_limit(float(plan.maximum_address_space_gib)) as address_space:
        with _cpu_thread_limit(int(plan.max_cpu_threads)) as torch_pools:
            with _cpu_affinity_limit(int(plan.max_cpu_threads)) as affinity:
                return _run_actual_gate_calibration_with_limits(
                    plan,
                    settings=settings,
                    checkpoint_path=checkpoint_resolved,
                    report_store=report_store,
                    resume=resume,
                    observed_thread_pools={
                        **torch_pools,
                        "cpu_affinity": affinity,
                        "address_space": address_space,
                    },
                )


def _run_actual_gate_calibration_with_limits(
    plan: CalibrationRunPlan,
    *,
    settings: Mapping[str, object],
    checkpoint_path: Optional[Path],
    report_store: Path,
    resume: bool,
    observed_thread_pools: Mapping[str, object],
) -> Mapping[str, object]:
    """Execute or resume all actual-gate stress trials and return evidence.

    The returned execution wrapper is always non-authorizing.  Its ``evidence``
    member is the exact aggregate object consumed by
    ``compile_actual_gate_calibration_receipt``.  Smoke evidence remains below
    the compiler's frozen 1,000-trial minimum.
    """

    _enforce_process_rss_limit(plan, phase="startup")
    settings = dict(settings)
    settings_sha256 = canonical_sha256(settings)
    contract = _run_contract(plan, settings)
    contract_sha256 = canonical_sha256(contract)
    validate_calibration_run_contract(
        contract,
        expected_settings_sha256=settings_sha256,
        require_authorizing_boundary=False,
    )
    checkpoint_file = checkpoint_path
    if checkpoint_file is not None and checkpoint_file.exists():
        if not resume:
            raise ValueError("calibration checkpoint exists but resume=False")
        checkpoint = _load_checkpoint(checkpoint_file, contract, report_store=report_store)
    else:
        checkpoint = _empty_checkpoint(contract)
    if checkpoint_file is not None:
        # Persist even an empty run contract before the first allocation-heavy trial.
        _save_checkpoint(checkpoint, checkpoint_file)

    completed = checkpoint["completed_trials"]
    if not isinstance(completed, dict):
        raise AssertionError("validated calibration checkpoint lost its trial mapping")
    completed_since_checkpoint = 0
    conditions = tuple(str(value) for value in contract["conditions"])
    for scenario in REQUIRED_CALIBRATION_SCENARIOS:
        for condition in conditions:
            key = "%s.%s" % (scenario, condition)
            records = completed[key]
            if not isinstance(records, list):
                raise AssertionError("validated calibration checkpoint has invalid records")
            for trial_index in range(len(records), int(plan.trials_per_condition)):
                before_rss = _process_rss_gib()
                if before_rss > float(plan.maximum_process_rss_gib):
                    if checkpoint_file is not None:
                        _save_checkpoint(checkpoint, checkpoint_file)
                    _enforce_process_rss_limit(
                        plan,
                        phase="before trial",
                        observed=before_rss,
                    )
                development, locked = build_synthetic_calibration_pair(
                    scenario,
                    condition,
                    trial_index,
                    base_seed=int(plan.base_seed),
                    design_binding=settings["confirmatory_design_binding"],
                )
                identity = _trial_identity(
                    scenario=scenario,
                    condition=condition,
                    trial_index=trial_index,
                    contract=contract,
                )
                development_artifact_sha256 = morphology_artifact_content_sha256(development)
                locked_artifact_sha256 = morphology_artifact_content_sha256(locked)
                report = evaluate_morphology_ridge_gate(
                    development,
                    locked,
                    ranks=tuple(int(value) for value in settings["target_rank_grid"]),
                    alphas=tuple(float(value) for value in settings["ridge_penalty_grid"]),
                    permutation_seeds=tuple(int(value) for value in settings["permutation_seeds"]),
                    permutations_per_seed=int(settings["permutations_per_seed"]),
                    total_permutations=int(settings["permutations_per_null"]),
                    final_inference=True,
                    minimum_final_permutations=int(
                        settings["gate_parameters"]["minimum_final_permutations"]
                    ),
                    minimum_support=int(settings["gate_parameters"]["minimum_support"]),
                    minimum_development_donors=int(
                        settings["gate_parameters"]["minimum_development_donors"]
                    ),
                    minimum_locked_donors=int(settings["gate_parameters"]["minimum_locked_donors"]),
                    minimum_macro_r2=float(settings["gate_parameters"]["minimum_macro_r2"]),
                    minimum_shuffle_delta=float(
                        settings["gate_parameters"]["minimum_shuffle_delta"]
                    ),
                    minimum_coordinate_delta=float(
                        settings["gate_parameters"]["minimum_coordinate_delta"]
                    ),
                    minimum_stain_delta=float(settings["gate_parameters"]["minimum_stain_delta"]),
                    maximum_direct_contrast_p=float(
                        settings["gate_parameters"]["maximum_direct_contrast_p"]
                    ),
                    minimum_mask_implementation_pass_fraction=float(
                        settings["gate_parameters"]["minimum_mask_implementation_pass_fraction"]
                    ),
                    minimum_null_shuffled_fraction=float(
                        settings["gate_parameters"]["minimum_null_shuffled_fraction"]
                    ),
                    minimum_strata_coverage=float(
                        settings["gate_parameters"]["minimum_strata_coverage"]
                    ),
                    maximum_permutation_p=float(
                        settings["gate_parameters"]["maximum_permutation_p"]
                    ),
                    minimum_positive_strata_fraction=float(
                        settings["gate_parameters"]["minimum_positive_strata_fraction"]
                    ),
                    minimum_expression_error_reduction=float(
                        settings["gate_parameters"]["minimum_expression_error_reduction"]
                    ),
                    minimum_basis_ceiling_r2=float(
                        settings["gate_parameters"]["minimum_basis_ceiling_r2"]
                    ),
                    donor_bootstrap_iterations=int(
                        settings["gate_parameters"]["donor_bootstrap_iterations"]
                    ),
                    donor_bootstrap_seed=int(settings["gate_parameters"]["donor_bootstrap_seed"]),
                    prespecified_fixed_hyperparameters=bool(
                        settings["gate_parameters"]["prespecified_fixed_hyperparameters"]
                    ),
                    confirmatory_analysis_plan_sha256=str(
                        settings["confirmatory_analysis_plan_sha256"]
                    ),
                    confirmatory_design_binding=settings["confirmatory_design_binding"],
                    synthetic_calibration_mode=True,
                    calibration_trial_identity=identity,
                    calibration_run_contract_sha256=contract_sha256,
                    device=str(plan.device),
                )
                report = _attest_actual_gate_report(
                    report,
                    identity=identity,
                    run_contract_sha256=contract_sha256,
                    development_artifact_sha256=development_artifact_sha256,
                    locked_artifact_sha256=locked_artifact_sha256,
                )
                report_sha256, report_relative_path = _persist_actual_gate_report(
                    report,
                    report_store=report_store,
                )
                record = dict(
                    _trial_record(
                        report,
                        trial_index=trial_index,
                        settings_sha256=settings_sha256,
                        settings=settings,
                        expected_trial_identity=identity,
                        run_contract_sha256=contract_sha256,
                        expected_decision_truth=contract["dgp_effect_spec"][
                            "decision_truth_by_condition"
                        ][condition],
                    )
                )
                if record["actual_report_sha256"] != report_sha256:
                    raise AssertionError("preserved report hash differs from attested outcome")
                record["actual_report_relative_path"] = report_relative_path
                records.append(record)
                completed_since_checkpoint += 1
                observed_rss = _process_rss_gib()
                rss_limit_exceeded = observed_rss > float(plan.maximum_process_rss_gib)
                if checkpoint_file is not None and (
                    completed_since_checkpoint >= plan.checkpoint_every or rss_limit_exceeded
                ):
                    _save_checkpoint(checkpoint, checkpoint_file)
                    completed_since_checkpoint = 0
                _enforce_process_rss_limit(
                    plan,
                    phase="after trial",
                    observed=observed_rss,
                )

    if checkpoint_file is not None:
        _save_checkpoint(checkpoint, checkpoint_file)
    scenario_results = {
        scenario: {
            condition: _condition_evidence(completed["%s.%s" % (scenario, condition)])
            for condition in conditions
        }
        for scenario in REQUIRED_CALIBRATION_SCENARIOS
    }
    report_manifest = _trial_report_manifest(
        completed,
        conditions=conditions,
        report_store=report_store,
    )
    evidence = {
        "schema": CALIBRATION_EVIDENCE_SCHEMA,
        "engine": CALIBRATION_ENGINE,
        "actual_gate_entrypoint": ACTUAL_GATE_ENTRYPOINT,
        "exact_gate_settings_sha256": settings_sha256,
        "run_contract": contract,
        "run_contract_sha256": canonical_sha256(contract),
        "trial_report_manifest": report_manifest,
        "scenario_results": scenario_results,
    }
    return {
        "schema": CALIBRATION_EXECUTION_SCHEMA,
        "production_contract_satisfied": plan.production_contract_satisfied,
        "smoke_test": bool(plan.smoke_test),
        "authorizes_scientific_claims": False,
        "authorizes_final_inference": False,
        "synthetic_data_only": True,
        "locked_outcomes_used": False,
        "resource_limits": {
            "max_cpu_threads": int(plan.max_cpu_threads),
            "maximum_process_rss_gib": float(plan.maximum_process_rss_gib),
            "maximum_address_space_gib": float(plan.maximum_address_space_gib),
            "process_isolation": contract["process_isolation"],
            "final_process_rss_gib": _process_rss_gib(),
            "observed_thread_pools": dict(observed_thread_pools),
        },
        "actual_gate_entrypoint": ACTUAL_GATE_ENTRYPOINT,
        "exact_gate_settings_sha256": settings_sha256,
        "run_contract_sha256": canonical_sha256(contract),
        "generator_version": contract["generator_version"],
        "dgp_effect_spec_sha256": contract["dgp_effect_spec_sha256"],
        "trial_report_manifest_content_sha256": report_manifest["manifest_content_sha256"],
        "evidence": evidence,
        "evidence_content_sha256": canonical_sha256(evidence),
        "non_authorizing_reason": (
            "smoke execution is preliminary and has fewer than 1000 trials per condition"
            if plan.smoke_test
            else (
                "the execution wrapper cannot authorize; its six-condition preserved-report "
                "evidence requires successful compilation into a separate receipt"
            )
        ),
    }


__all__ = [
    "CALIBRATION_CHECKPOINT_SCHEMA",
    "CALIBRATION_CONDITIONS",
    "AUTHORIZING_CALIBRATION_CONDITIONS",
    "AUTHORIZING_DGP_EFFECT_SPEC",
    "CALIBRATION_EXECUTION_SCHEMA",
    "CALIBRATION_GENERATOR_VERSION",
    "PRELIMINARY_CROP_SIGNAL_SCALES",
    "PRELIMINARY_CALIBRATION_CONDITIONS",
    "PRELIMINARY_DGP_EFFECT_SPEC",
    "PRODUCTION_TRIALS_PER_CONDITION",
    "CalibrationRunPlan",
    "build_synthetic_calibration_pair",
    "load_calibration_run_config",
    "run_actual_gate_calibration",
    "synthetic_completed_confirmatory_design_binding",
]
