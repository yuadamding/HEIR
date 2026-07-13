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
from .morphology_gate import evaluate_morphology_ridge_gate
from .power import (
    ACTUAL_GATE_ENTRYPOINT,
    ACTUAL_GATE_REPORT_SCHEMA,
    CALIBRATION_DGP_SPEC_SCHEMA,
    CALIBRATION_ENGINE,
    CALIBRATION_EVIDENCE_SCHEMA,
    CALIBRATION_GENERATOR_VERSION,
    CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES,
    CALIBRATION_RUN_CONTRACT_SCHEMA,
    GLOBAL_NULL_CONDITION,
    PRELIMINARY_ALTERNATIVE_CONDITION,
    REQUIRED_CALIBRATION_SCENARIOS,
    REQUIRED_COMPLETE_GATE_CHECKS,
    REQUIRED_HYPOTHESIS_DECISIONS,
    canonical_sha256,
    current_calibration_executable_provenance,
    planned_donor_type_support_pattern_sha256,
    validate_calibration_run_contract,
    validate_confirmatory_design_binding,
    validate_exact_gate_settings,
)

PRODUCTION_TRIALS_PER_CONDITION = 1000
CALIBRATION_CONDITIONS = (GLOBAL_NULL_CONDITION, PRELIMINARY_ALTERNATIVE_CONDITION)
CALIBRATION_CHECKPOINT_SCHEMA = "heir.morphology_gate_calibration_checkpoint.v2"
CALIBRATION_EXECUTION_SCHEMA = "heir.morphology_gate_calibration_execution.v2"
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
    planned_support = [
        {"stratum_id": stratum_id, "minimum_evaluation_cells": 20} for stratum_id in planned_strata
    ]
    binding = {
        "status": "complete",
        "scientific_manifest_projection_sha256": canonical_sha256(
            {"synthetic_calibration_design": CALIBRATION_GENERATOR_VERSION}
        ),
        "measurement_receipt_sha256": canonical_sha256(
            ["synthetic_calibration_design", "measurement_receipt"]
        ),
        "ordered_target_gene_ids": list(genes),
        "target_panel_sha256": canonical_sha256(list(genes)),
        "target_gene_count": len(genes),
        "ordered_supported_fine_type_ids": list(fine_types),
        "supported_fine_type_ids_sha256": canonical_sha256(list(fine_types)),
        "development_donor_ids": list(development),
        "locked_test_donor_ids": list(locked),
        "encoder_manifest_sha256": canonical_sha256(
            ["synthetic_calibration_design", "encoder_manifest"]
        ),
        "crop_manifest_sha256s": [
            canonical_sha256(["synthetic_calibration_design", "crop_manifest"])
        ],
        "planned_donor_type_support_pattern_sha256": planned_donor_type_support_pattern_sha256(
            development,
            locked,
            fine_types,
        ),
        "planned_stratum_topology_status": "complete",
        "ordered_planned_stratum_ids": list(planned_strata),
        "planned_stratum_manifest_sha256": canonical_sha256(list(planned_strata)),
        "planned_stratum_minimum_evaluation_cells": [20] * len(planned_strata),
        "planned_stratum_support_pattern_sha256": canonical_sha256(planned_support),
    }
    return validate_confirmatory_design_binding(binding)


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
            PRELIMINARY_DGP_EFFECT_SPEC["authorizing_boundary_calibration"] is True
            and not self.smoke_test
            and self.trials_per_condition >= PRODUCTION_TRIALS_PER_CONDITION
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
    if scenario not in REQUIRED_CALIBRATION_SCENARIOS:
        raise ValueError("unknown morphology calibration scenario: %s" % scenario)
    if condition not in CALIBRATION_CONDITIONS:
        raise ValueError("unknown morphology calibration condition: %s" % condition)
    if trial_index < 0:
        raise ValueError("calibration trial_index must be non-negative")
    scenario_index = REQUIRED_CALIBRATION_SCENARIOS.index(scenario)
    condition_index = CALIBRATION_CONDITIONS.index(condition)
    return int(base_seed + scenario_index * 10_000_000 + condition_index * 1_000_000 + trial_index)


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


def _synthetic_artifact(
    *,
    scenario: str,
    condition: str,
    trial_index: int,
    base_seed: int,
    role: str,
    design_binding: Mapping[str, object],
    shared_target_weights: np.ndarray,
    shared_feature_weights: np.ndarray,
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
    latent_rows: list[np.ndarray] = []
    target_latent_rows: list[np.ndarray] = []
    coordinate_rows: list[tuple[float, float]] = []
    technical_rows: list[tuple[float, float]] = []

    for _stratum_id, donor, section, type_index, minimum_count in role_strata:
        donor_index = donor_index_by_id[donor]
        section_index = section_index_by_key[(donor, section)]
        if (
            scenario == "missing_fine_types"
            and role == "locked_test"
            and type_index == len(type_names) - 1
        ):
            continue
        if scenario == "section_effects":
            row_count = minimum_count
        elif scenario == "unbalanced_donor_cell_counts":
            imbalance_index = (
                donor_count - donor_index - 1 if role == "locked_test" else donor_index
            )
            row_count = minimum_count + 2 * imbalance_index
        else:
            row_count = minimum_count + 4
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

            type_shift = (type_index - (len(type_names) - 1) / 2.0) * 0.25
            donor_shift = (donor_index - 2) * 0.05
            latent = rng.normal(size=6) + type_shift + donor_shift
            latent_rows.append(latent)
            target_latent_rows.append(rng.normal(size=6))
            x = (row_index % 4) / 3.0 + donor_index * 0.03
            y = (row_index // 4) / max(row_count // 4, 1) + type_index * 0.05
            coordinate_rows.append((x, y))
            technical_rows.append(
                (
                    np.log1p(90.0 + 3.0 * row_index + 5.0 * type_index),
                    1.0 + 0.1 * donor_index + rng.normal(scale=0.03),
                )
            )

    latent = np.asarray(latent_rows, dtype=np.float64)
    target_latent = np.asarray(target_latent_rows, dtype=np.float64)
    coordinates = np.asarray(coordinate_rows, dtype=np.float64)
    technical = np.asarray(technical_rows, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    rows = len(label_array)

    morphology = latent @ shared_feature_weights + rng.normal(scale=0.08, size=(rows, 8))
    independent_molecular = target_latent @ shared_target_weights
    associated_molecular = latent @ shared_target_weights
    if condition == PRELIMINARY_ALTERNATIVE_CONDITION:
        residual = associated_molecular
    else:
        residual = independent_molecular

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
        np.linspace(0.12, 0.65, gene_count)
        if scenario == "variable_transcript_reliability"
        else np.full(gene_count, 0.18)
    )
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

    crop_features = []
    for crop_index, crop_id in enumerate(HEST_CROP_CONTRACT):
        scale = _crop_signal_scale(crop_id, scenario)
        crop_noise = rng.normal(scale=0.12 + 0.01 * crop_index, size=(rows, 8))
        if crop_id == "blank_patch":
            crop_features.append(crop_noise * 0.01)
        elif "random_location" in crop_id or "blurred" in crop_id:
            crop_features.append(scale * rng.normal(size=(rows, 8)) + crop_noise)
        else:
            crop_features.append(scale * morphology + crop_noise)
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
    observed_strata = {
        "%s|%s|%s" % (donor, section, type_names[label])
        for donor, section, label in zip(donor_ids, section_ids, labels)
    }
    retained_fraction = len(observed_strata) / len(planned_strata)
    memberships = {
        split_id: _source_digest(
            "reference_membership",
            scenario,
            condition,
            trial_index,
            role,
            split_id,
        )
        for split_id in split_ids
    }
    shared_identity = (scenario, condition, trial_index)
    artifact = MorphologyRidgeDatasetArtifact(
        observation_ids=np.asarray(observation_ids, dtype=str),
        donor_ids=np.asarray(donor_ids, dtype=str),
        block_ids=np.asarray(block_ids, dtype=str),
        roi_ids=np.asarray(roi_ids, dtype=str),
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
        section_ids=np.asarray(section_ids, dtype=str),
        disease_states=np.asarray(disease_states, dtype=str),
        site_ids=np.asarray(site_ids, dtype=str),
        batch_ids=np.asarray(batch_ids, dtype=str),
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
            "reference_membership_sha256_by_split": memberships,
            "locked_measurement_audit": {
                "pass": True,
                "synthetic_calibration_only": True,
            },
        },
        reference_evaluation_balance={
            split_id: {"pass": True, "synthetic_calibration_only": True} for split_id in split_ids
        },
        study_manifest_sha256=_source_digest("study_manifest", *shared_identity),
        measurement_receipt_sha256=str(binding["measurement_receipt_sha256"]),
        measurement_source_sha256=_source_digest("measurement_source", *shared_identity),
        hypothesis_ids=("H-CELL", "H-INTRINSIC"),
        scientific_scope="registered_cell_local_context_association",
        evidence_scope="internal_locked_hest",
        authorizes_nucleus_intrinsic_claim=False,
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
    feature_weights = shared_rng.normal(size=(6, 8)) / np.sqrt(6.0)
    development = _synthetic_artifact(
        scenario=scenario,
        condition=condition,
        trial_index=trial_index,
        base_seed=base_seed,
        role="development",
        design_binding=binding,
        shared_target_weights=target_weights,
        shared_feature_weights=feature_weights,
    )
    locked = _synthetic_artifact(
        scenario=scenario,
        condition=condition,
        trial_index=trial_index,
        base_seed=base_seed,
        role="locked_test",
        design_binding=binding,
        shared_target_weights=target_weights,
        shared_feature_weights=feature_weights,
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
    dgp_spec = dict(PRELIMINARY_DGP_EFFECT_SPEC)
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
        "conditions": list(CALIBRATION_CONDITIONS),
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
            for condition in CALIBRATION_CONDITIONS
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


def _load_checkpoint(path: Path, contract: Mapping[str, object]) -> dict[str, object]:
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
        for condition in CALIBRATION_CONDITIONS
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
                    "component_pass",
                    "actual_report_sha256",
                    "exact_gate_settings_sha256",
                    "required_checks_present",
                    "hypothesis_decision_passes",
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
                or record.get("exact_gate_settings_sha256")
                != contract["exact_gate_settings_sha256"]
                or record.get("required_checks_present") is not True
                or record.get("scientific_authorization_suppressed") is not True
                or not isinstance(decisions, Mapping)
                or set(decisions) != set(REQUIRED_HYPOTHESIS_DECISIONS)
                or any(not isinstance(value, bool) for value in decisions.values())
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
    return checkpoint


def _authorization_is_suppressed(report: Mapping[str, object]) -> bool:
    explicit_flags = (
        "authorizes_full_heir",
        "authorizes_population_inference",
        "authorizes_external_generalization",
        "authorizes_validated_regional_association",
        "authorizes_nucleus_intrinsic_claim",
        "authorizes_cell_intrinsic_claim",
    )
    return bool(
        report.get("scientific_authorization_suppressed") is True
        and all(report.get(name) is False for name in explicit_flags)
    )


def _trial_record(
    report: Mapping[str, object],
    *,
    trial_index: int,
    settings_sha256: str,
    settings: Mapping[str, object],
) -> Mapping[str, object]:
    if report.get("schema_version") != ACTUAL_GATE_REPORT_SCHEMA:
        raise ValueError("actual morphology gate returned an unsupported report schema")
    if (
        report.get("synthetic_calibration_execution") is not True
        or report.get("final_inference") is not True
        or not _authorization_is_suppressed(report)
    ):
        raise ValueError("synthetic calibration report did not suppress scientific authorization")
    if report.get("calibration_exact_gate_settings_sha256") != settings_sha256:
        raise ValueError("actual-gate trial report differs from the frozen settings")
    checks = report.get("checks")
    decisions = report.get("hypothesis_decisions")
    if not isinstance(checks, Mapping) or not set(REQUIRED_COMPLETE_GATE_CHECKS).issubset(checks):
        raise ValueError("actual-gate trial report lacks required complete-gate checks")
    if not isinstance(decisions, Mapping) or not set(REQUIRED_HYPOTHESIS_DECISIONS).issubset(
        decisions
    ):
        raise ValueError("actual-gate trial report lacks required hypothesis decisions")
    decision_passes = {}
    for name in REQUIRED_HYPOTHESIS_DECISIONS:
        decision = decisions[name]
        if not isinstance(decision, Mapping) or not isinstance(decision.get("pass"), bool):
            raise ValueError("actual-gate trial hypothesis decision is malformed: %s" % name)
        decision_passes[name] = bool(decision["pass"])
    source_conclusion = report.get("morphology_source_conclusion")
    if source_conclusion not in CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES:
        raise ValueError("actual-gate trial lacks a frozen morphology-source conclusion")
    local = report.get("permutation_control")
    spatial = report.get("spatial_block_permutation_control")
    if not isinstance(local, Mapping) or not isinstance(spatial, Mapping):
        raise ValueError("actual-gate trial report lacks both permutation nulls")
    local_count = int(local.get("total_permutations", -1))
    spatial_count = int(spatial.get("total_permutations", -1))
    minimum_permutations = int(settings["permutations_per_null"])
    if local_count != minimum_permutations or spatial_count != minimum_permutations:
        raise ValueError("actual-gate trial differs from the exact frozen permutation total")

    expected_seeds = tuple(int(value) for value in settings["permutation_seeds"])
    expected_per_seed = int(settings["permutations_per_seed"])

    def seed_counts(null_report: Mapping[str, object], name: str) -> Mapping[str, int]:
        rows = null_report.get("seeds")
        if not isinstance(rows, list) or len(rows) != len(expected_seeds):
            raise ValueError("actual-gate %s null lacks exact seed-stream evidence" % name)
        counts = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("actual-gate %s seed row is malformed" % name)
            seed = int(row.get("seed", -1))
            required = int(row.get("required_unique_permutations", -1))
            generated = int(row.get("generated_unique_permutations", -1))
            if seed in counts or required != expected_per_seed or generated != expected_per_seed:
                raise ValueError("actual-gate %s seed stream differs from frozen counts" % name)
            counts[str(seed)] = generated
        if set(counts) != {str(seed) for seed in expected_seeds}:
            raise ValueError("actual-gate %s seed identities differ from frozen streams" % name)
        return counts

    local_seed_counts = seed_counts(local, "local ROI")
    spatial_seed_counts = seed_counts(spatial, "spatial block")
    report_hash = canonical_sha256(report)
    return {
        "trial_index": int(trial_index),
        "component_pass": bool(report.get("component_pass") is True),
        "actual_report_sha256": report_hash,
        "exact_gate_settings_sha256": settings_sha256,
        "required_checks_present": True,
        "hypothesis_decision_passes": decision_passes,
        "morphology_source_conclusion": source_conclusion,
        "scientific_authorization_suppressed": True,
        "local_roi_permutations": local_count,
        "spatial_block_permutations": spatial_count,
        "local_roi_seed_counts": local_seed_counts,
        "spatial_block_seed_counts": spatial_seed_counts,
    }


def _condition_evidence(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not records:
        raise ValueError("calibration condition has no completed production-gate trials")
    for field in ("local_roi_seed_counts", "spatial_block_seed_counts"):
        if any(record[field] != records[0][field] for record in records):
            raise ValueError("calibration trials used inconsistent permutation seed counts")
    passes = sum(bool(record["component_pass"]) for record in records)
    report_hashes = [str(record["actual_report_sha256"]) for record in records]
    return {
        "trials": len(records),
        "complete_gate_passes": passes,
        "hypothesis_decision_passes": {
            name: sum(bool(record["hypothesis_decision_passes"][name]) for record in records)
            for name in REQUIRED_HYPOTHESIS_DECISIONS
        },
        "morphology_source_conclusion_counts": {
            name: sum(record["morphology_source_conclusion"] == name for record in records)
            for name in CALIBRATION_MORPHOLOGY_SOURCE_OUTCOMES
        },
        "actual_gate_executions": len(records),
        "trial_report_set_sha256": canonical_sha256(
            {"ordered_actual_gate_report_sha256": report_hashes}
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


def run_actual_gate_calibration(
    plan: CalibrationRunPlan,
    *,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
) -> Mapping[str, object]:
    """Execute calibration under active CPU-pool and process-RSS limits."""

    settings = dict(plan.validate())
    dedicated_process = os.environ.get(DEDICATED_PROCESS_ENV) == "1"
    if not plan.smoke_test and not dedicated_process:
        raise RuntimeError(
            "non-smoke calibration must run through the dedicated calibration CLI process"
        )
    with _address_space_limit(float(plan.maximum_address_space_gib)) as address_space:
        with _cpu_thread_limit(int(plan.max_cpu_threads)) as torch_pools:
            with _cpu_affinity_limit(int(plan.max_cpu_threads)) as affinity:
                return _run_actual_gate_calibration_with_limits(
                    plan,
                    settings=settings,
                    checkpoint_path=checkpoint_path,
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
    validate_calibration_run_contract(
        contract,
        expected_settings_sha256=settings_sha256,
        require_authorizing_boundary=False,
    )
    checkpoint_file = checkpoint_path.expanduser().resolve() if checkpoint_path else None
    if checkpoint_file is not None and checkpoint_file.exists():
        if not resume:
            raise ValueError("calibration checkpoint exists but resume=False")
        checkpoint = _load_checkpoint(checkpoint_file, contract)
    else:
        checkpoint = _empty_checkpoint(contract)
    if checkpoint_file is not None:
        # Persist even an empty run contract before the first allocation-heavy trial.
        _save_checkpoint(checkpoint, checkpoint_file)

    completed = checkpoint["completed_trials"]
    if not isinstance(completed, dict):
        raise AssertionError("validated calibration checkpoint lost its trial mapping")
    completed_since_checkpoint = 0
    for scenario in REQUIRED_CALIBRATION_SCENARIOS:
        for condition in CALIBRATION_CONDITIONS:
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
                    device=str(plan.device),
                )
                records.append(
                    dict(
                        _trial_record(
                            report,
                            trial_index=trial_index,
                            settings_sha256=settings_sha256,
                            settings=settings,
                        )
                    )
                )
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
            for condition in CALIBRATION_CONDITIONS
        }
        for scenario in REQUIRED_CALIBRATION_SCENARIOS
    }
    evidence = {
        "schema": CALIBRATION_EVIDENCE_SCHEMA,
        "engine": CALIBRATION_ENGINE,
        "actual_gate_entrypoint": ACTUAL_GATE_ENTRYPOINT,
        "exact_gate_settings_sha256": settings_sha256,
        "run_contract": contract,
        "run_contract_sha256": canonical_sha256(contract),
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
        "evidence": evidence,
        "evidence_content_sha256": canonical_sha256(evidence),
        "non_authorizing_reason": (
            "preliminary full-shared-latent DGP is diagnostic and cannot authorize inference"
            if not plan.smoke_test
            else "smoke execution is preliminary and has fewer than 1000 trials per condition"
        ),
    }


__all__ = [
    "CALIBRATION_CHECKPOINT_SCHEMA",
    "CALIBRATION_CONDITIONS",
    "CALIBRATION_EXECUTION_SCHEMA",
    "CALIBRATION_GENERATOR_VERSION",
    "PRELIMINARY_CROP_SIGNAL_SCALES",
    "PRELIMINARY_DGP_EFFECT_SPEC",
    "PRODUCTION_TRIALS_PER_CONDITION",
    "CalibrationRunPlan",
    "build_synthetic_calibration_pair",
    "load_calibration_run_config",
    "run_actual_gate_calibration",
    "synthetic_completed_confirmatory_design_binding",
]
