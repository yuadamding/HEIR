"""Audited donor-balanced morphology-to-molecular-state gate."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from heir.data import MorphologyRidgeDatasetArtifact
from heir.utils import resolve_device

from .control_models import paired_feature_families
from .hierarchical_metrics import (
    donor_bootstrap,
    donor_dominance,
    donor_section_type_coverage,
    donor_type_coverage,
    exact_paired_randomization,
    group_stratification,
    leave_one_donor_out,
    macro_error_reduction,
    macro_reconstruction_r2,
    paired_donor_effects,
    within_group_donor_type_r2,
)
from .model_selection import select_hyperparameters
from .permutations import (
    block_null_activity,
    donor_type_block_permutation,
    donor_type_roi_permutation,
    null_stratum_activity,
)
from .power import (
    ACTUAL_GATE_ENTRYPOINT,
    G2_MULTIPLICITY_METHOD,
    G3_MULTIPLICITY_METHOD,
    REQUIRED_COMPLETE_GATE_CHECKS,
    REQUIRED_G3_CONTRAST_PAIRS,
    REQUIRED_HYPOTHESIS_DECISIONS,
    REQUIRED_MORPHOLOGY_SOURCE_CONCLUSIONS,
    REQUIRED_NUISANCE_FAMILIES,
    REQUIRED_PERMUTATION_TRANSFORMS,
    canonical_sha256,
    exact_gate_settings_fingerprint,
    pending_confirmatory_design_binding,
    planned_donor_type_support_pattern_sha256,
    validate_calibration_receipt,
    validate_confirmatory_design_binding,
)
from .residual_targets import correct_residuals, endpoint_covariates
from .ridge_probe import fit_and_score, target_coordinates

MORPHOLOGY_RIDGE_REPORT_SCHEMA = "heir.morphology_ridge_evaluation.v5"
REGIONAL_UNI2H_TECHNICAL_COVARIATES = ("log1p_library_size",)
REGIONAL_UNI2H_COMPOSITION_FEATURES = (
    "composition_epithelial",
    "composition_immune",
    "composition_stromal",
    "composition_endothelial",
)


def _exact_positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("%s must be an exact positive integer" % name)
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise ValueError("%s must be an exact positive integer" % name)
    return int(numeric)


def _familywise_frozen_contrasts(
    control_internal: Mapping[
        str,
        Tuple[
            float,
            Mapping[str, float],
            int,
            float,
            Sequence[Mapping[str, object]],
        ],
    ],
    pairs: Mapping[str, tuple[str, str]],
    *,
    minimum_delta: float,
    maximum_p: float,
) -> Mapping[str, object]:
    """Exact max-statistic inference for a fully prespecified contrast family."""

    available = {
        name: (control_internal.get(focal), control_internal.get(comparator))
        for name, (focal, comparator) in pairs.items()
    }
    missing = sorted(
        name
        for name, (focal, comparator) in available.items()
        if focal is None or comparator is None
    )
    if missing:
        return {
            "tested": False,
            "pass": False,
            "missing_contrasts": missing,
            "multiplicity_method": G3_MULTIPLICITY_METHOD,
        }
    effects = {
        name: paired_donor_effects(focal[1], comparator[1])
        for name, (focal, comparator) in available.items()
        if focal is not None and comparator is not None
    }
    donors = tuple(sorted(next(iter(effects.values()))))
    if any(tuple(sorted(values)) != donors for values in effects.values()):
        raise ValueError("frozen contrast family has inconsistent donor support")
    matrix = np.asarray(
        [[effects[name][donor] for donor in donors] for name in pairs], dtype=np.float64
    )
    observed = matrix.mean(axis=1)
    sign_patterns = np.asarray(
        [
            [1.0 if (pattern >> index) & 1 else -1.0 for index in range(len(donors))]
            for pattern in range(1 << len(donors))
        ],
        dtype=np.float64,
    )
    permuted = matrix @ sign_patterns.T / max(len(donors), 1)
    maximum_null = np.max(permuted, axis=0)
    adjusted_p = np.mean(maximum_null[None, :] >= observed[:, None], axis=1)
    reports = {}
    for index, name in enumerate(pairs):
        focal_name, comparator_name = pairs[name]
        passes = bool(observed[index] >= minimum_delta and adjusted_p[index] <= maximum_p)
        reports[name] = {
            "focal_family": focal_name,
            "comparator_family": comparator_name,
            "donor_effects": effects[name],
            "mean_delta_r2": float(observed[index]),
            "minimum_delta_r2": float(minimum_delta),
            "familywise_adjusted_p": float(adjusted_p[index]),
            "maximum_familywise_p": float(maximum_p),
            "pass": passes,
        }
    return {
        "tested": True,
        "multiplicity_method": G3_MULTIPLICITY_METHOD,
        "donors": list(donors),
        "exact_sign_patterns": int(len(sign_patterns)),
        "contrasts": reports,
        "pass": bool(all(report["pass"] for report in reports.values())),
    }


def _classify_morphology_source(
    *,
    intrinsic_prespecified: bool,
    final_inference: bool,
    familywise_tested: bool,
    measurement_valid: bool,
    component_pass: bool,
    any_source_contrast_positive: bool,
    nucleus_specific: bool,
    cell_specific: bool,
    context_specific: bool,
    full_vs_context: bool,
    full_vs_intrinsic: bool,
) -> str:
    """Return a fail-closed conclusion from the complete frozen G3 family.

    A missing or exploratory family is not a negative biological result.  A
    complete family with no positive source contrast reports the prespecified
    ``no_morphology_specific_information`` outcome; this means no source arm
    was supported, not that a biological null was proven.
    """

    if not intrinsic_prespecified or not final_inference:
        return "not_tested"
    if not familywise_tested:
        return "inconclusive"
    if not measurement_valid:
        return "inconclusive"

    intrinsic_specific = nucleus_specific or cell_specific
    if not intrinsic_specific and not context_specific:
        if any_source_contrast_positive:
            return "inconclusive"
        return "no_morphology_specific_information"
    if not component_pass:
        return "inconclusive"
    mixed_information = bool(
        context_specific and intrinsic_specific and full_vs_context and full_vs_intrinsic
    )
    if mixed_information:
        return "mixed_intrinsic_and_contextual_information"
    if nucleus_specific and not cell_specific and not context_specific:
        return "nucleus_dominant"
    if cell_specific and not nucleus_specific and not context_specific:
        return "cell_dominant"
    if context_specific and not nucleus_specific and not cell_specific:
        return "context_dominant"
    return "multiple_sources_without_incremental_combination"


def _disease_adjusted_pair(
    development: MorphologyRidgeDatasetArtifact,
    locked_test: MorphologyRidgeDatasetArtifact,
) -> tuple[MorphologyRidgeDatasetArtifact, MorphologyRidgeDatasetArtifact]:
    """Append a shared disease design without changing the inclusive endpoint."""

    categories = sorted(set(development.disease_states.astype(str).tolist()))
    if not set(locked_test.disease_states.astype(str).tolist()).issubset(categories):
        raise ValueError("locked test contains a disease state absent from development")
    if len(categories) < 2:
        return development, locked_test
    names = tuple("disease::%s" % value for value in categories[1:])
    development_design = np.column_stack(
        [(development.disease_states.astype(str) == value) for value in categories[1:]]
    ).astype(np.float64)
    locked_design = np.column_stack(
        [(locked_test.disease_states.astype(str) == value) for value in categories[1:]]
    ).astype(np.float64)
    return (
        replace(
            development,
            technical_covariates=np.concatenate(
                (development.technical_covariates, development_design), axis=1
            ),
            technical_covariate_names=development.technical_covariate_names + names,
        ),
        replace(
            locked_test,
            technical_covariates=np.concatenate(
                (locked_test.technical_covariates, locked_design), axis=1
            ),
            technical_covariate_names=locked_test.technical_covariate_names + names,
        ),
    )


def _direct_contrast(
    control_internal: Mapping[
        str,
        Tuple[
            float,
            Mapping[str, float],
            int,
            float,
            Sequence[Mapping[str, object]],
        ],
    ],
    focal_family: str,
    comparator_families: Sequence[str],
    *,
    minimum_delta: float,
    maximum_p: float,
    bootstrap_seed: int,
    bootstrap_iterations: int,
) -> Mapping[str, object]:
    focal = control_internal.get(focal_family)
    available = {
        family: control_internal[family]
        for family in comparator_families
        if family in control_internal
    }
    if focal is None or not available:
        return {
            "tested": False,
            "pass": False,
            "reason": "focal image or a required direct comparator is unavailable",
        }
    comparator = sorted(available, key=lambda family: (-available[family][0], family))[0]
    effects = paired_donor_effects(focal[1], available[comparator][1])
    bootstrap = donor_bootstrap(effects, seed=bootstrap_seed, iterations=bootstrap_iterations)
    randomization = exact_paired_randomization(effects)
    dominance = donor_dominance(effects)
    delta = float(focal[0] - available[comparator][0])
    return {
        "tested": True,
        "focal_family": focal_family,
        "focal_macro_r2": float(focal[0]),
        "strongest_comparator_family": comparator,
        "strongest_comparator_macro_r2": float(available[comparator][0]),
        "focal_minus_comparator_macro_r2": delta,
        "minimum_delta": float(minimum_delta),
        "donor_effects": effects,
        "donor_bootstrap": bootstrap,
        "exact_donor_paired_randomization": randomization,
        "donor_dominance": dominance,
        "maximum_largest_positive_donor_share": 0.5,
        "maximum_exact_one_sided_p": float(maximum_p),
        "pass": bool(
            delta >= minimum_delta
            and bootstrap["ci_95"][0] > 0.0
            and randomization["one_sided_p"] <= maximum_p
            and dominance["largest_positive_share"] <= 0.5
        ),
    }


def _robust_mask_contrast(
    control_internal: Mapping[
        str,
        Tuple[
            float,
            Mapping[str, float],
            int,
            float,
            Sequence[Mapping[str, object]],
        ],
    ],
    implementations: Mapping[str, str],
    comparator_families: Sequence[str],
    artifact_control_families: Sequence[str],
    *,
    minimum_delta: float,
    maximum_p: float,
    minimum_pass_fraction: float,
    bootstrap_seed: int,
    bootstrap_iterations: int,
) -> Mapping[str, object]:
    """Require prespecified fill implementations to survive matched controls."""

    contrasts = {}
    for offset, (implementation, focal_family) in enumerate(implementations.items()):
        contrasts[implementation] = _direct_contrast(
            control_internal,
            focal_family,
            tuple(comparator_families) + tuple(artifact_control_families),
            minimum_delta=minimum_delta,
            maximum_p=maximum_p,
            bootstrap_seed=bootstrap_seed + offset,
            bootstrap_iterations=bootstrap_iterations,
        )
    tested = bool(contrasts) and all(value["tested"] for value in contrasts.values())
    passing = sum(value.get("pass") is True for value in contrasts.values())
    pass_fraction = float(passing / len(contrasts)) if contrasts else 0.0
    primary = contrasts.get("white_fill", next(iter(contrasts.values()), {}))
    sensitivities = {
        family: {
            "available": family in control_internal,
            "macro_r2": (
                float(control_internal[family][0]) if family in control_internal else None
            ),
        }
        for family in artifact_control_families
    }
    return {
        **primary,
        "tested": tested,
        "implementation_contrasts": contrasts,
        "mask_artifact_control_sensitivities": sensitivities,
        "passing_implementations": passing,
        "required_implementations": len(contrasts),
        "implementation_pass_fraction": pass_fraction,
        "minimum_mask_implementation_pass_fraction": float(minimum_pass_fraction),
        "pass": bool(tested and pass_fraction >= minimum_pass_fraction),
    }


def validate_experiment_identity(
    artifact: MorphologyRidgeDatasetArtifact, experiment_role: str
) -> None:
    """Compatibility validator retained until manifest binding supersedes role names."""

    expected_encoder = {
        "primary_hest_uni2h": "MahmoodLab/UNI2-h",
        "primary_hoptimus1": "bioptimus/H-optimus-1",
        "replication_h0mini": "bioptimus/H0-mini",
        "confirmation_xenium": "bioptimus/H-optimus-1",
        "regional_hescape_hoptimus1": "bioptimus/H-optimus-1",
        "regional_hescape_uni2h": "MahmoodLab/UNI2-h",
    }.get(experiment_role)
    if experiment_role not in {
        "primary_hoptimus1",
        "primary_hest_uni2h",
        "replication_h0mini",
        "context_sensitivity",
        "confirmation_xenium",
        "regional_hescape_hoptimus1",
        "regional_hescape_uni2h",
    }:
        raise ValueError("morphology-ridge experiment role is unsupported")
    if expected_encoder is not None and artifact.encoder_name != expected_encoder:
        raise ValueError("%s requires frozen %s features" % (experiment_role, expected_encoder))
    expected_crop = {
        "primary_hest_uni2h": "registered_cell_local_context_112um",
        "context_sensitivity": "full_context",
        "confirmation_xenium": "nucleus_centered",
        "regional_hescape_hoptimus1": "full_context",
        "regional_hescape_uni2h": "full_context",
    }.get(experiment_role, "small_cell_centered")
    regional_crop_ok = experiment_role in {
        "regional_hescape_hoptimus1",
        "regional_hescape_uni2h",
    } and artifact.crop_scale in {"full_context", "target_matched_55um"}
    if artifact.crop_scale != expected_crop and not regional_crop_ok:
        raise ValueError("%s requires the %s crop" % (experiment_role, expected_crop))
    if experiment_role in {"primary_hoptimus1", "replication_h0mini"} and (
        artifact.observation_level not in {"cell", "nucleus"}
        or artifact.target_construction != "registered_cell_expression"
    ):
        raise ValueError("the decisive morphology gate requires registered cell-level targets")
    if experiment_role == "primary_hest_uni2h" and (
        artifact.cohort_id != "HEST"
        or artifact.assay != "Xenium"
        or artifact.observation_level != "cell"
        or artifact.registration_method != "native_xenium_cell_id_join"
        or artifact.target_construction != "nucleus_overlapping_xenium_transcripts"
        or artifact.scientific_scope != "registered_cell_local_context_association"
        or artifact.primary_crop_id != "crop_112um"
        or artifact.crop_comparison_families[artifact.crop_ids.index(artifact.primary_crop_id)]
        != "g2_primary"
    ):
        raise ValueError("primary HEST UNI2-h must be the explicit G2 local-context arm")
    if experiment_role in {"regional_hescape_hoptimus1", "regional_hescape_uni2h"} and (
        artifact.cohort_id != "HESCAPE"
        or artifact.cohort_release != "human-lung-healthy-panel"
        or artifact.observation_level != "pseudo_spot_55um"
        or artifact.target_construction != "sum_pooled_xenium_transcripts"
    ):
        raise ValueError("the HESCAPE role is restricted to the regional pseudo-spot control")
    if experiment_role == "regional_hescape_uni2h":
        if artifact.technical_covariate_names != REGIONAL_UNI2H_TECHNICAL_COVARIATES:
            raise ValueError("the UNI2-h regional role requires the frozen log-library covariate")
        if artifact.composition_feature_names != REGIONAL_UNI2H_COMPOSITION_FEATURES:
            raise ValueError(
                "the UNI2-h regional role requires four frozen RNA-only composition scores"
            )
        if artifact.composition_features.shape != (
            len(artifact.observation_ids),
            len(REGIONAL_UNI2H_COMPOSITION_FEATURES),
        ) or any(
            len(np.unique(artifact.composition_features[:, index])) < 3
            for index in range(len(REGIONAL_UNI2H_COMPOSITION_FEATURES))
        ):
            raise ValueError("the UNI2-h composition controls must be continuous scores")
        if not artifact.stain_feature_names:
            raise ValueError("the UNI2-h regional role requires stain-statistics controls")
    if experiment_role == "confirmation_xenium" and (
        artifact.cohort_id != "HEST" or artifact.assay != "Xenium"
    ):
        raise ValueError("confirmation requires a non-overlapping HEST Xenium artifact")


def _evaluate_permutation_null(
    development: MorphologyRidgeDatasetArtifact,
    locked_test: MorphologyRidgeDatasetArtifact,
    *,
    null_kind: str,
    matched: float,
    ranks: Sequence[int],
    alphas: Sequence[float],
    permutation_seeds: Sequence[int],
    total_permutations: int,
    permutations_per_seed: Optional[int] = None,
    minimum_support: int,
    minimum_shuffle_delta: float,
    maximum_permutation_p: float,
    minimum_shuffled_fraction: float,
    include_composition: bool,
    prespecified_fixed_hyperparameters: bool,
    device: str,
) -> Mapping[str, object]:
    """Evaluate one unique, reproducible permutation pool as one scientific test."""

    if null_kind not in {"local_within_roi", "spatial_block_reassignment"}:
        raise ValueError("permutation null kind is unsupported")
    seeds = tuple(sorted(set(int(value) for value in permutation_seeds)))
    if not seeds or total_permutations <= 0:
        raise ValueError("permutation pool needs seeds and a positive size")
    if permutations_per_seed is not None:
        if permutations_per_seed < 1 or permutations_per_seed * len(seeds) != total_permutations:
            raise ValueError("permutation pool must preserve the frozen count in every seed stream")
        target_seed_counts = {seed: int(permutations_per_seed) for seed in seeds}
    else:
        quotient, remainder = divmod(total_permutations, len(seeds))
        target_seed_counts = {
            seed: quotient + int(index < remainder) for index, seed in enumerate(seeds)
        }
    rank_grid = tuple(sorted(set(int(value) for value in ranks)))
    alpha_grid = tuple(sorted(set(float(value) for value in alphas)))
    fixed_model = (
        prespecified_fixed_hyperparameters and len(rank_grid) == 1 and len(alpha_grid) == 1
    )
    values = []
    changed_fractions = []
    cross_block_fractions = []
    selections: Dict[Tuple[int, float], int] = {}
    donor_values: Dict[str, list[float]] = {}
    seed_counts = {seed: 0 for seed in seeds}
    seed_changed = {seed: [] for seed in seeds}
    seed_cross_block = {seed: [] for seed in seeds}
    seen = set()
    pool_hasher = hashlib.sha256()
    attempts_by_seed = {seed: 0 for seed in seeds}
    attempt = 0
    maximum_attempts = max(total_permutations * 500, 1000)
    while len(values) < total_permutations and attempt < maximum_attempts:
        seed = next(value for value in seeds if seed_counts[value] < target_seed_counts[value])
        stream_index = attempts_by_seed[seed]
        attempts_by_seed[seed] += 1
        local_seed = seed + stream_index * 104729
        attempt += 1
        if null_kind == "local_within_roi":
            permutation = donor_type_roi_permutation(
                development.donor_ids,
                development.type_labels,
                development.roi_ids,
                seed=local_seed,
            )
            cross_block = None
        else:
            permutation = donor_type_block_permutation(
                development.donor_ids,
                development.type_labels,
                development.block_ids,
                seed=local_seed,
            )
            cross_block = float(
                np.mean(development.block_ids != development.block_ids[permutation])
            )
        digest = hashlib.sha256(np.ascontiguousarray(permutation).view(np.uint8)).digest()
        if digest in seen:
            continue
        seen.add(digest)
        pool_hasher.update(digest)
        changed = float(np.mean(permutation != np.arange(len(permutation))))
        changed_fractions.append(changed)
        seed_changed[seed].append(changed)
        if cross_block is not None:
            cross_block_fractions.append(cross_block)
            seed_cross_block[seed].append(cross_block)
        seed_counts[seed] += 1
        permuted_features = development.frozen_features[permutation]
        if fixed_model:
            rank, alpha = rank_grid[0], alpha_grid[0]
        else:
            rank, alpha, _ = select_hyperparameters(
                development,
                permuted_features,
                ranks=rank_grid,
                alphas=alpha_grid,
                minimum_support=minimum_support,
                device=device,
                include_composition=include_composition,
            )
        selections[(rank, alpha)] = selections.get((rank, alpha), 0) + 1
        shuffled, *_, local_donors = fit_and_score(
            development,
            locked_test,
            permuted_features,
            locked_test.frozen_features,
            rank=rank,
            alpha=alpha,
            minimum_support=minimum_support,
            device=device,
            include_composition=include_composition,
        )
        values.append(shuffled)
        for donor, value in local_donors.items():
            donor_values.setdefault(donor, []).append(float(value))
    if len(values) != total_permutations:
        raise ValueError(
            "%s cannot generate the prespecified number of unique active permutations" % null_kind
        )
    if seed_counts != target_seed_counts:
        raise AssertionError("permutation seed streams differ from their frozen counts")
    array = np.asarray(values, dtype=np.float64)
    empirical_p = float((1 + np.sum(array >= matched)) / (len(array) + 1))
    minimum_changed = float(min(changed_fractions))
    minimum_cross_block = float(min(cross_block_fractions)) if cross_block_fractions else None
    activity_pass = minimum_changed >= minimum_shuffled_fraction and (
        minimum_cross_block is None or minimum_cross_block >= minimum_shuffled_fraction
    )
    delta = float(matched - array.mean())
    pool_pass = bool(
        activity_pass and delta >= minimum_shuffle_delta and empirical_p <= maximum_permutation_p
    )
    seed_rows = [
        {
            "seed": seed,
            "required_unique_permutations": target_seed_counts[seed],
            "generated_unique_permutations": seed_counts[seed],
            "minimum_shuffled_fraction": (
                float(min(seed_changed[seed])) if seed_changed[seed] else None
            ),
            "minimum_cross_block_fraction": (
                float(min(seed_cross_block[seed])) if seed_cross_block[seed] else None
            ),
            "empirical_p": empirical_p,
            "pass": pool_pass,
        }
        for seed in seeds
    ]
    activity = (
        null_stratum_activity(development.donor_ids, development.type_labels, development.roi_ids)
        if null_kind == "local_within_roi"
        else block_null_activity(
            development.donor_ids, development.type_labels, development.block_ids
        )
    )
    return {
        "null_kind": null_kind,
        "training_probe_refit_for_each_permutation": True,
        "hyperparameter_selection": (
            "manifest_prespecified_single_candidate"
            if fixed_model
            else "repeated_development_donor_fold_selection"
        ),
        "full_pipeline_hyperparameters_reselected": not fixed_model,
        "one_combined_scientific_permutation_pool": True,
        "seeds_are_generation_streams_not_independent_tests": True,
        "unique_permutations": True,
        "permutation_pool_sha256": pool_hasher.hexdigest(),
        "preserves_donor_and_fine_type": True,
        "preserves_roi": null_kind == "local_within_roi",
        "preserves_donor_type_roi": null_kind == "local_within_roi",
        "reassigns_spatial_block": null_kind == "spatial_block_reassignment",
        "activity": activity,
        "total_permutations": len(values),
        "seeds": seed_rows,
        "mean_macro_r2": float(array.mean()),
        "matched_minus_shuffle_macro_r2": delta,
        "empirical_p": empirical_p,
        "minimum_shuffled_fraction": minimum_changed,
        "minimum_cross_block_fraction": minimum_cross_block,
        "activity_pass": activity_pass,
        "pass": pool_pass,
        "selected_hyperparameter_counts": [
            {"rank": rank, "alpha": alpha, "count": count}
            for (rank, alpha), count in sorted(selections.items())
        ],
        "donor_null_mean_r2": {
            donor: float(np.mean(local_values))
            for donor, local_values in sorted(donor_values.items())
        },
    }


def evaluate_morphology_ridge_gate(
    development: MorphologyRidgeDatasetArtifact,
    locked_test: MorphologyRidgeDatasetArtifact,
    *,
    ranks: Sequence[int] = (2, 4, 6),
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    permutation_seeds: Sequence[int] = (17, 29, 41),
    permutations_per_seed: int = 100,
    total_permutations: Optional[int] = None,
    final_inference: bool = False,
    minimum_final_permutations: int = 999,
    minimum_support: int = 10,
    minimum_development_donors: int = 5,
    minimum_locked_donors: Optional[int] = None,
    minimum_macro_r2: float = 0.05,
    minimum_shuffle_delta: float = 0.03,
    minimum_coordinate_delta: float = 0.01,
    minimum_stain_delta: float = 0.01,
    maximum_direct_contrast_p: float = 0.05,
    minimum_mask_implementation_pass_fraction: float = 1.0,
    minimum_null_shuffled_fraction: float = 0.50,
    minimum_strata_coverage: float = 0.80,
    maximum_permutation_p: float = 0.01,
    minimum_positive_strata_fraction: float = 0.70,
    minimum_expression_error_reduction: float = 0.05,
    minimum_basis_ceiling_r2: float = 0.10,
    donor_bootstrap_iterations: int = 2000,
    donor_bootstrap_seed: int = 1701,
    prespecified_fixed_hyperparameters: bool = False,
    calibration_receipt: Optional[Mapping[str, object]] = None,
    confirmatory_analysis_plan_sha256: Optional[str] = None,
    confirmatory_design_binding: Optional[Mapping[str, object]] = None,
    synthetic_calibration_mode: bool = False,
    device: str = "auto",
) -> Mapping[str, object]:
    """Run a locked-donor gate without treating observations as replicates."""

    development.validate_compatible(locked_test)
    if synthetic_calibration_mode and not final_inference:
        raise ValueError("synthetic calibration must execute the final-inference code path")
    synthetic_cohort = (
        development.cohort_id == "SYNTHETIC_CALIBRATION"
        and locked_test.cohort_id == "SYNTHETIC_CALIBRATION"
    )
    if synthetic_calibration_mode != synthetic_cohort:
        raise ValueError(
            "synthetic calibration mode accepts only explicitly synthetic calibration artifacts"
        )
    if development.cohort_id == "HESCAPE" or locked_test.cohort_id == "HESCAPE":
        raise ValueError(
            "HESCAPE is development-only; reserved HEST donors cannot enter a locked gate"
        )
    primary_crop_index = development.crop_ids.index(development.primary_crop_id)
    primary_crop_family = development.crop_comparison_families[primary_crop_index]
    local_context_hypothesis = bool(
        "H-CELL" in development.hypothesis_ids
        and primary_crop_family == "g2_primary"
        and "local_context" in development.scientific_scope
    )
    intrinsic_prespecified = "H-INTRINSIC" in development.hypothesis_ids
    regional = "H-REGIONAL" in development.hypothesis_ids
    regional_uni2h = regional and development.encoder_name == "MahmoodLab/UNI2-h"
    family_pairs = paired_feature_families(development, locked_test)
    if final_inference and local_context_hypothesis:
        missing_nuisance = [
            family for family in REQUIRED_NUISANCE_FAMILIES if family_pairs.get(family) is None
        ]
        if missing_nuisance:
            raise ValueError(
                "final H-CELL inference omits frozen nuisance families: %s"
                % ", ".join(missing_nuisance)
            )
    required_locked_donors = (
        minimum_locked_donors if minimum_locked_donors is not None else (4 if regional_uni2h else 5)
    )
    development_donors = sorted(set(development.donor_ids.tolist()))
    locked_donors = sorted(set(locked_test.donor_ids.tolist()))
    if len(development_donors) < minimum_development_donors:
        raise ValueError("too few development donors for a morphology decision")
    if len(locked_donors) < required_locked_donors:
        raise ValueError("too few locked donors for the prespecified test")
    development_type_support = {
        development.type_names[type_index]: sum(
            np.count_nonzero(
                (development.donor_ids == donor) & (development.type_labels == type_index)
            )
            >= minimum_support
            for donor in development_donors
        )
        for type_index in range(len(development.type_names))
    }
    locked_type_support = {
        locked_test.type_names[type_index]: sum(
            np.count_nonzero(
                (locked_test.donor_ids == donor) & (locked_test.type_labels == type_index)
            )
            >= minimum_support
            for donor in locked_donors
        )
        for type_index in range(len(locked_test.type_names))
    }
    if any(value < minimum_development_donors for value in development_type_support.values()):
        raise ValueError("a frozen fine type lacks the required development-donor support")
    if any(0 < value < required_locked_donors for value in locked_type_support.values()):
        raise ValueError(
            "an under-supported locked fine type entered the evaluated analysis population"
        )
    design_binding = (
        validate_confirmatory_design_binding(confirmatory_design_binding)
        if confirmatory_design_binding is not None
        else pending_confirmatory_design_binding()
    )
    if design_binding.get("status") == "complete":
        topology_complete = design_binding["planned_stratum_topology_status"] == "complete"
        bound_planned_strata = tuple(design_binding["ordered_planned_stratum_ids"])
        bound_development_donors = set(design_binding["development_donor_ids"])
        bound_locked_donors = set(design_binding["locked_test_donor_ids"])
        bound_development_strata = tuple(
            value
            for value in bound_planned_strata
            if value.split("|", 1)[0] in bound_development_donors
        )
        bound_locked_strata = tuple(
            value for value in bound_planned_strata if value.split("|", 1)[0] in bound_locked_donors
        )

        def observed_strata(artifact: MorphologyRidgeDatasetArtifact) -> set[str]:
            return {
                "%s|%s|%s" % (donor, section, artifact.type_names[int(type_index)])
                for donor, section, type_index in zip(
                    artifact.donor_ids,
                    artifact.section_ids,
                    artifact.type_labels,
                )
            }

        development_observed_strata = observed_strata(development)
        locked_observed_strata = observed_strata(locked_test)
        development_coverage = len(development_observed_strata) / len(
            development.planned_stratum_ids
        )
        locked_coverage = len(locked_observed_strata) / len(locked_test.planned_stratum_ids)
        if (
            set(design_binding["development_donor_ids"]) != set(development_donors)
            or set(design_binding["locked_test_donor_ids"]) != set(locked_donors)
            or list(design_binding["ordered_supported_fine_type_ids"])
            != list(development.type_names)
            or development.type_names != locked_test.type_names
            or design_binding["target_gene_count"] != len(development.gene_ids)
            or list(design_binding["ordered_target_gene_ids"]) != list(development.gene_ids)
            or design_binding["target_panel_sha256"] != canonical_sha256(list(development.gene_ids))
            or development.gene_ids != locked_test.gene_ids
            or design_binding["measurement_receipt_sha256"]
            != development.measurement_receipt_sha256
            or design_binding["measurement_receipt_sha256"]
            != locked_test.measurement_receipt_sha256
            or design_binding["planned_donor_type_support_pattern_sha256"]
            != planned_donor_type_support_pattern_sha256(
                design_binding["development_donor_ids"],
                design_binding["locked_test_donor_ids"],
                design_binding["ordered_supported_fine_type_ids"],
            )
            or (
                topology_complete
                and (
                    tuple(development.planned_stratum_ids) != bound_development_strata
                    or tuple(locked_test.planned_stratum_ids) != bound_locked_strata
                    or development.planned_stratum_manifest_sha256
                    != design_binding["planned_stratum_manifest_sha256"]
                    or locked_test.planned_stratum_manifest_sha256
                    != design_binding["planned_stratum_manifest_sha256"]
                    or not development_observed_strata <= set(development.planned_stratum_ids)
                    or not locked_observed_strata <= set(locked_test.planned_stratum_ids)
                    or not np.isclose(
                        float(development.coverage_audit.get("retained_fraction", -1.0)),
                        development_coverage,
                    )
                    or not np.isclose(
                        float(locked_test.coverage_audit.get("retained_fraction", -1.0)),
                        locked_coverage,
                    )
                    or any(
                        int(value) != minimum_support
                        for value in design_binding["planned_stratum_minimum_evaluation_cells"]
                    )
                )
            )
        ):
            raise ValueError("morphology artifacts differ from the confirmatory design binding")
        if (final_inference or synthetic_calibration_mode) and not topology_complete:
            raise ValueError(
                "final morphology inference requires a complete donor/section/type topology"
            )
    elif final_inference or synthetic_calibration_mode:
        raise ValueError("final morphology inference requires a completed design binding")
    if not permutation_seeds:
        raise ValueError("ridge gate requires a nonempty permutation pool")
    normalized_seed_stream = tuple(
        _exact_positive_integer(value, "permutation seed") for value in permutation_seeds
    )
    if len(set(normalized_seed_stream)) != len(normalized_seed_stream):
        raise ValueError("ridge gate permutation seeds must be unique")
    seeds = tuple(sorted(normalized_seed_stream))
    permutations_per_seed = _exact_positive_integer(permutations_per_seed, "permutations_per_seed")
    minimum_final_permutations = _exact_positive_integer(
        minimum_final_permutations, "minimum_final_permutations"
    )
    requested_permutations = (
        _exact_positive_integer(total_permutations, "total_permutations")
        if total_permutations is not None
        else permutations_per_seed * len(seeds)
    )
    if requested_permutations < 100:
        raise ValueError("ridge gate requires at least 100 total permutations")
    if final_inference and requested_permutations < minimum_final_permutations:
        raise ValueError(
            "final morphology inference requires at least %d unique permutations"
            % minimum_final_permutations
        )
    if final_inference and requested_permutations != permutations_per_seed * len(seeds):
        raise ValueError(
            "final morphology inference requires the frozen count in every seed stream"
        )
    if not ranks or not alphas:
        raise ValueError("ridge rank and alpha grids must be positive")
    ranks = tuple(_exact_positive_integer(value, "ridge rank") for value in ranks)
    if len(set(ranks)) != len(ranks) or any(
        value > development.molecular_targets.shape[1] for value in ranks
    ):
        raise ValueError("ridge rank grid is invalid for the frozen target panel")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(float(value))
        or float(value) <= 0.0
        for value in alphas
    ):
        raise ValueError("ridge rank and alpha grids must be positive")
    alphas = tuple(float(value) for value in alphas)
    if not (0.0 < minimum_null_shuffled_fraction <= 1.0) or not (
        0.0 < minimum_strata_coverage <= 1.0
    ):
        raise ValueError("null activity and stratum coverage thresholds must be in (0, 1]")
    if donor_bootstrap_iterations < 100:
        raise ValueError("donor bootstrap requires at least 100 iterations")
    if not 0.0 < maximum_direct_contrast_p <= 1.0 or not (
        0.0 < minimum_mask_implementation_pass_fraction <= 1.0
    ):
        raise ValueError("G3 direct-contrast thresholds must be in (0, 1]")
    expected_calibration_settings = {
        "actual_gate_entrypoint": ACTUAL_GATE_ENTRYPOINT,
        "actual_gate_report_schema": MORPHOLOGY_RIDGE_REPORT_SCHEMA,
        "confirmatory_analysis_plan_sha256": confirmatory_analysis_plan_sha256,
        "confirmatory_design_binding": dict(design_binding),
        "development_donors": len(development_donors),
        "evaluation_donors": len(locked_donors),
        "crop_family_ids": list(development.crop_ids),
        "nuisance_families": list(REQUIRED_NUISANCE_FAMILIES),
        "g2_multiplicity_method": G2_MULTIPLICITY_METHOD,
        "g2_nuisance_family_ids": list(REQUIRED_NUISANCE_FAMILIES),
        "g3_multiplicity_method": G3_MULTIPLICITY_METHOD,
        "g3_contrast_pairs": dict(REQUIRED_G3_CONTRAST_PAIRS),
        "allowed_morphology_source_conclusions": list(REQUIRED_MORPHOLOGY_SOURCE_CONCLUSIONS),
        "target_rank_grid": [int(value) for value in ranks],
        "ridge_penalty_grid": [float(value) for value in alphas],
        "reference_split_ids": list(development.reference_split_ids),
        "permutation_transforms": list(REQUIRED_PERMUTATION_TRANSFORMS),
        "permutation_seeds": list(seeds),
        "permutations_per_seed": int(permutations_per_seed),
        "permutations_per_null": requested_permutations,
        "maximum_permutation_p": float(maximum_permutation_p),
        "gate_parameters": {
            "minimum_final_permutations": int(minimum_final_permutations),
            "minimum_support": int(minimum_support),
            "minimum_development_donors": int(minimum_development_donors),
            "minimum_locked_donors": int(required_locked_donors),
            "minimum_macro_r2": float(minimum_macro_r2),
            "minimum_shuffle_delta": float(minimum_shuffle_delta),
            "minimum_coordinate_delta": float(minimum_coordinate_delta),
            "minimum_stain_delta": float(minimum_stain_delta),
            "maximum_direct_contrast_p": float(maximum_direct_contrast_p),
            "minimum_mask_implementation_pass_fraction": float(
                minimum_mask_implementation_pass_fraction
            ),
            "minimum_null_shuffled_fraction": float(minimum_null_shuffled_fraction),
            "minimum_strata_coverage": float(minimum_strata_coverage),
            "maximum_permutation_p": float(maximum_permutation_p),
            "minimum_positive_strata_fraction": float(minimum_positive_strata_fraction),
            "minimum_expression_error_reduction": float(minimum_expression_error_reduction),
            "minimum_basis_ceiling_r2": float(minimum_basis_ceiling_r2),
            "donor_bootstrap_iterations": int(donor_bootstrap_iterations),
            "donor_bootstrap_seed": int(donor_bootstrap_seed),
            "prespecified_fixed_hyperparameters": bool(prespecified_fixed_hyperparameters),
        },
        "complete_gate_check_ids": list(REQUIRED_COMPLETE_GATE_CHECKS),
        "hypothesis_decision_ids": list(REQUIRED_HYPOTHESIS_DECISIONS),
        "final_inference": True,
    }
    synthetic_settings_sha256 = (
        exact_gate_settings_fingerprint(expected_calibration_settings)
        if synthetic_calibration_mode
        else None
    )
    calibration = validate_calibration_receipt(
        calibration_receipt,
        required=bool(final_inference and not synthetic_calibration_mode),
        expected_settings=(
            None
            if synthetic_calibration_mode or not final_inference
            else expected_calibration_settings
        ),
    )

    include_composition = regional and bool(development.composition_feature_names)
    rank, alpha, selection = select_hyperparameters(
        development,
        development.frozen_features,
        ranks=ranks,
        alphas=alphas,
        minimum_support=minimum_support,
        device=device,
        include_composition=include_composition,
    )
    if include_composition:
        raw_rank, raw_alpha, raw_selection = select_hyperparameters(
            development,
            development.frozen_features,
            ranks=ranks,
            alphas=alphas,
            minimum_support=minimum_support,
            device=device,
            include_composition=False,
        )
        raw_matched, *_, raw_donor_macro = fit_and_score(
            development,
            locked_test,
            development.frozen_features,
            locked_test.frozen_features,
            rank=raw_rank,
            alpha=raw_alpha,
            minimum_support=minimum_support,
            device=device,
            include_composition=False,
        )
    else:
        raw_rank, raw_alpha, raw_selection = rank, alpha, selection
    (
        matched,
        fit,
        prediction,
        predicted_coordinates,
        truth_coordinates,
        rows,
        donor_macro,
    ) = fit_and_score(
        development,
        locked_test,
        development.frozen_features,
        locked_test.frozen_features,
        rank=rank,
        alpha=alpha,
        minimum_support=minimum_support,
        device=device,
        include_composition=include_composition,
    )
    if not include_composition:
        raw_matched = matched
        raw_donor_macro = donor_macro

    disease_development, disease_locked = _disease_adjusted_pair(development, locked_test)
    disease_adjustment_available = disease_development is not development
    if disease_adjustment_available:
        disease_rank, disease_alpha, disease_selection = select_hyperparameters(
            disease_development,
            disease_development.frozen_features,
            ranks=ranks,
            alphas=alphas,
            minimum_support=minimum_support,
            device=device,
            include_composition=include_composition,
        )
        (
            disease_adjusted_macro,
            _,
            _,
            disease_adjusted_prediction,
            disease_adjusted_truth,
            disease_adjusted_rows,
            disease_adjusted_donors,
        ) = fit_and_score(
            disease_development,
            disease_locked,
            disease_development.frozen_features,
            disease_locked.frozen_features,
            rank=disease_rank,
            alpha=disease_alpha,
            minimum_support=minimum_support,
            device=device,
            include_composition=include_composition,
        )
    else:
        disease_rank, disease_alpha, disease_selection = rank, alpha, selection
        disease_adjusted_macro = matched
        disease_adjusted_prediction = predicted_coordinates
        disease_adjusted_truth = truth_coordinates
        disease_adjusted_rows = rows
        disease_adjusted_donors = donor_macro

    control_internal: Dict[
        str,
        Tuple[
            float,
            Mapping[str, float],
            int,
            float,
            Sequence[Mapping[str, object]],
        ],
    ] = {}
    control_report: Dict[str, Mapping[str, object]] = {}
    for family, pair in family_pairs.items():
        if pair is None:
            control_report[family] = {"available": False}
            continue
        development_features, locked_features = pair
        family_rank, family_alpha, family_selection = select_hyperparameters(
            development,
            development_features,
            ranks=(rank,),
            alphas=alphas,
            minimum_support=minimum_support,
            device=device,
            include_composition=include_composition,
        )
        family_macro, *_, family_donors = fit_and_score(
            development,
            locked_test,
            development_features,
            locked_features,
            rank=family_rank,
            alpha=family_alpha,
            minimum_support=minimum_support,
            device=device,
            include_composition=include_composition,
            target_fit=fit.target,
        )
        control_internal[family] = (
            family_macro,
            family_donors,
            family_rank,
            family_alpha,
            family_selection,
        )
        control_report[family] = {
            "available": True,
            "macro_r2": family_macro,
            "rank": family_rank,
            "alpha": family_alpha,
            "development_donor_folds": family_selection,
            "shared_molecular_target_basis": True,
            "locked_test_used_for_selection": False,
        }
    coordinate_macro, coordinate_donor_macro, coordinate_rank, coordinate_alpha, _ = (
        control_internal["coordinate_only"]
    )
    stain_values = control_internal.get("stain_only")
    stain_macro = stain_values[0] if stain_values is not None else None
    stain_donor_macro = stain_values[1] if stain_values is not None else {}

    _, basis_prediction = target_coordinates(
        fit.target,
        locked_test.molecular_targets,
        locked_test.reference_means,
        endpoint_covariates(locked_test, include_composition),
        locked_test.type_labels,
    )
    corrected_truth = locked_test.reference_means.copy()
    corrected_truth += correct_residuals(
        locked_test.molecular_targets,
        locked_test.reference_means,
        endpoint_covariates(locked_test, include_composition),
        locked_test.type_labels,
        fit.technical_mean,
        fit.technical_coefficients,
    )
    ceiling_r2, ceiling_rows, ceiling_donors = macro_reconstruction_r2(
        corrected_truth,
        basis_prediction,
        locked_test.reference_means,
        locked_test.donor_ids,
        locked_test.type_labels,
        minimum_support,
    )
    expression_reduction, expression_rows, expression_donors = macro_error_reduction(
        corrected_truth,
        prediction,
        locked_test.reference_means,
        locked_test.donor_ids,
        locked_test.type_labels,
        minimum_support,
    )

    local_null = _evaluate_permutation_null(
        development,
        locked_test,
        null_kind="local_within_roi",
        matched=matched,
        ranks=ranks,
        alphas=alphas,
        permutation_seeds=seeds,
        total_permutations=requested_permutations,
        permutations_per_seed=(int(permutations_per_seed) if final_inference else None),
        minimum_support=minimum_support,
        minimum_shuffle_delta=minimum_shuffle_delta,
        maximum_permutation_p=maximum_permutation_p,
        minimum_shuffled_fraction=minimum_null_shuffled_fraction,
        include_composition=include_composition,
        prespecified_fixed_hyperparameters=prespecified_fixed_hyperparameters,
        device=device,
    )
    block_null = _evaluate_permutation_null(
        development,
        locked_test,
        null_kind="spatial_block_reassignment",
        matched=matched,
        ranks=ranks,
        alphas=alphas,
        permutation_seeds=seeds,
        total_permutations=requested_permutations,
        permutations_per_seed=(int(permutations_per_seed) if final_inference else None),
        minimum_support=minimum_support,
        minimum_shuffle_delta=minimum_shuffle_delta,
        maximum_permutation_p=maximum_permutation_p,
        minimum_shuffled_fraction=minimum_null_shuffled_fraction,
        include_composition=include_composition,
        prespecified_fixed_hyperparameters=prespecified_fixed_hyperparameters,
        device=device,
    )

    positive_rows = [row for row in rows if row["residual_coordinate_r2"] > 0]
    positive_fraction = len(positive_rows) / len(rows)
    dominance = donor_dominance(donor_macro)
    largest_donor_share = float(dominance["largest_positive_share"])
    allowed_nonpositive_donors = (
        1 if regional_uni2h and len(donor_macro) >= 4 else (1 if len(donor_macro) >= 10 else 0)
    )
    donor_consistency = (
        sum(value <= 0 for value in donor_macro.values()) <= allowed_nonpositive_donors
    )
    stratum_macro = float(np.mean([float(row["residual_coordinate_r2"]) for row in rows]))
    coverage = donor_type_coverage(
        locked_test.donor_ids,
        locked_test.type_labels,
        minimum_support,
        len(locked_test.type_names),
    )
    matched_bootstrap = donor_bootstrap(
        donor_macro, seed=donor_bootstrap_seed, iterations=donor_bootstrap_iterations
    )
    coordinate_effects = paired_donor_effects(donor_macro, coordinate_donor_macro)
    coordinate_bootstrap = donor_bootstrap(
        coordinate_effects,
        seed=donor_bootstrap_seed + 1,
        iterations=donor_bootstrap_iterations,
    )
    stain_effects: Mapping[str, float] = {}
    stain_bootstrap: Optional[Mapping[str, object]] = None
    if stain_macro is not None:
        stain_effects = paired_donor_effects(donor_macro, stain_donor_macro)
        stain_bootstrap = donor_bootstrap(
            stain_effects,
            seed=donor_bootstrap_seed + 2,
            iterations=donor_bootstrap_iterations,
        )
    nuisance_families = (
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
    available_nuisance = {
        family: control_internal[family]
        for family in nuisance_families
        if family in control_internal
    }
    g2_nuisance_closed_tests = {}
    for family, (family_macro, family_donors, *_) in available_nuisance.items():
        family_effects = paired_donor_effects(donor_macro, family_donors)
        family_randomization = exact_paired_randomization(family_effects)
        g2_nuisance_closed_tests[family] = {
            "matched_minus_nuisance_macro_r2": float(matched - family_macro),
            "donor_effects": family_effects,
            "exact_donor_paired_randomization": family_randomization,
            "pass": bool(
                matched - family_macro >= minimum_coordinate_delta
                and family_randomization["one_sided_p"] <= maximum_direct_contrast_p
            ),
        }
    g2_closed_test_pass = bool(
        g2_nuisance_closed_tests
        and all(report["pass"] for report in g2_nuisance_closed_tests.values())
    )
    best_nuisance_name = sorted(
        available_nuisance, key=lambda family: (-available_nuisance[family][0], family)
    )[0]
    best_nuisance_macro, best_nuisance_donors, *_ = available_nuisance[best_nuisance_name]
    best_nuisance_effects = paired_donor_effects(donor_macro, best_nuisance_donors)
    best_nuisance_bootstrap = donor_bootstrap(
        best_nuisance_effects,
        seed=donor_bootstrap_seed + 3,
        iterations=donor_bootstrap_iterations,
    )
    direct_comparators = tuple(available_nuisance) + (
        "context_only",
        "target_cell_removed_context_image",
        "crop_image::target_cell_removed_mean_fill_112um",
        "crop_image::target_cell_removed_blurred_112um",
        "crop_image::context_ring_32_to_112um",
        "crop_image::context_ring_64_to_112um",
    )
    nucleus_contrast = _robust_mask_contrast(
        control_internal,
        {
            "white_fill": "nucleus_mask_image",
            "mean_fill": "crop_image::nucleus_mask_mean_fill_112um",
        },
        direct_comparators,
        (
            "crop_image::nucleus_shape_random_location_mean_fill_112um",
            "crop_image::nucleus_mask_blurred_112um",
        ),
        minimum_delta=minimum_coordinate_delta,
        maximum_p=maximum_direct_contrast_p,
        minimum_pass_fraction=minimum_mask_implementation_pass_fraction,
        bootstrap_seed=donor_bootstrap_seed + 10,
        bootstrap_iterations=donor_bootstrap_iterations,
    )
    cell_contrast = _robust_mask_contrast(
        control_internal,
        {
            "white_fill": "cell_mask_image",
            "mean_fill": "crop_image::cell_mask_mean_fill_112um",
        },
        direct_comparators,
        (
            "crop_image::cell_shape_random_location_mean_fill_112um",
            "crop_image::cell_mask_blurred_112um",
        ),
        minimum_delta=minimum_coordinate_delta,
        maximum_p=maximum_direct_contrast_p,
        minimum_pass_fraction=minimum_mask_implementation_pass_fraction,
        bootstrap_seed=donor_bootstrap_seed + 11,
        bootstrap_iterations=donor_bootstrap_iterations,
    )
    context_contrast = _robust_mask_contrast(
        control_internal,
        {
            "white_fill": "target_cell_removed_context_image",
            "mean_fill": "crop_image::target_cell_removed_mean_fill_112um",
        },
        tuple(available_nuisance)
        + (
            "nucleus_mask_image",
            "cell_mask_image",
            "crop_image::nucleus_mask_mean_fill_112um",
            "crop_image::cell_mask_mean_fill_112um",
        ),
        (
            "crop_image::random_location_cell_removed_mean_fill_112um",
            "crop_image::target_cell_removed_blurred_112um",
        ),
        minimum_delta=minimum_coordinate_delta,
        maximum_p=maximum_direct_contrast_p,
        minimum_pass_fraction=minimum_mask_implementation_pass_fraction,
        bootstrap_seed=donor_bootstrap_seed + 12,
        bootstrap_iterations=donor_bootstrap_iterations,
    )
    frozen_contrast_pairs = {
        name: (pair[0], pair[1]) for name, pair in REQUIRED_G3_CONTRAST_PAIRS.items()
    }
    frozen_contrast_family = _familywise_frozen_contrasts(
        control_internal,
        frozen_contrast_pairs,
        minimum_delta=minimum_coordinate_delta,
        maximum_p=maximum_direct_contrast_p,
    )
    frozen_reports = frozen_contrast_family.get("contrasts", {})
    any_source_contrast_positive = any(
        isinstance(report, Mapping) and report.get("pass") is True
        for report in frozen_reports.values()
    )

    def family_pass(prefix: str) -> bool:
        selected = [report for name, report in frozen_reports.items() if name.startswith(prefix)]
        return bool(selected and all(report.get("pass") is True for report in selected))

    nucleus_specific = family_pass("nucleus_")
    cell_specific = family_pass("cell_")
    context_specific = family_pass("context_")
    full_vs_context = family_pass("full_context_vs_target_removed_")
    full_vs_nucleus = family_pass("full_context_vs_nucleus_")
    full_vs_cell = family_pass("full_context_vs_cell_")
    intrinsic_specific = nucleus_specific or cell_specific
    full_vs_intrinsic = bool(
        (not nucleus_specific or full_vs_nucleus) and (not cell_specific or full_vs_cell)
    )
    mixed_information = bool(
        context_specific and intrinsic_specific and full_vs_context and full_vs_intrinsic
    )
    local_null_effects = paired_donor_effects(donor_macro, local_null["donor_null_mean_r2"])
    block_null_effects = paired_donor_effects(donor_macro, block_null["donor_null_mean_r2"])
    per_donor_effects = {
        donor: {
            "matched_macro_r2": float(donor_macro[donor]),
            "raw_depth_adjusted_macro_r2": float(raw_donor_macro[donor]),
            "coordinate_only_macro_r2": float(coordinate_donor_macro[donor]),
            "matched_minus_coordinate_macro_r2": float(coordinate_effects[donor]),
            "stain_statistics_only_macro_r2": (
                float(stain_donor_macro[donor]) if stain_macro is not None else None
            ),
            "matched_minus_stain_macro_r2": (
                float(stain_effects[donor]) if stain_macro is not None else None
            ),
            "best_nuisance_family": best_nuisance_name,
            "matched_minus_best_nuisance_macro_r2": float(best_nuisance_effects[donor]),
            "matched_minus_local_null_macro_r2": float(local_null_effects[donor]),
            "matched_minus_block_null_macro_r2": float(block_null_effects[donor]),
        }
        for donor in sorted(donor_macro)
    }
    section_ids = locked_test.section_ids
    section_source = "explicit_artifact_metadata"
    section_coverage = donor_section_type_coverage(
        locked_test.donor_ids,
        section_ids,
        locked_test.type_labels,
        minimum_support,
        len(locked_test.type_names),
    )
    section_stratification = group_stratification(
        truth_coordinates,
        predicted_coordinates,
        section_ids,
        locked_test.type_labels,
        minimum_support,
        group_name="section",
        source=section_source,
    )
    disease_labels = locked_test.disease_states
    disease_stratification = group_stratification(
        truth_coordinates,
        predicted_coordinates,
        disease_labels,
        locked_test.type_labels,
        minimum_support,
        group_name="disease",
        source="explicit_artifact_metadata",
    )
    disease_adjusted_stratification = group_stratification(
        disease_adjusted_truth,
        disease_adjusted_prediction,
        disease_labels,
        locked_test.type_labels,
        minimum_support,
        group_name="disease",
        source="development_fitted_disease_adjustment",
    )
    disease_within_donor_balanced = within_group_donor_type_r2(
        truth_coordinates,
        predicted_coordinates,
        disease_labels,
        locked_test.donor_ids,
        locked_test.type_labels,
        minimum_support,
        group_name="disease",
    )
    disease_adjusted_within_donor_balanced = within_group_donor_type_r2(
        disease_adjusted_truth,
        disease_adjusted_prediction,
        disease_labels,
        locked_test.donor_ids,
        locked_test.type_labels,
        minimum_support,
        group_name="disease",
    )
    rank_sensitivity = []
    for candidate in sorted(set(int(value) for value in ranks)):
        candidate_macro, *_ = fit_and_score(
            development,
            locked_test,
            development.frozen_features,
            locked_test.frozen_features,
            rank=candidate,
            alpha=alpha,
            minimum_support=minimum_support,
            device=device,
            include_composition=include_composition,
        )
        rank_sensitivity.append({"rank": candidate, "macro_r2": candidate_macro})

    reference_split_sensitivity = []
    for split_index, split_id in enumerate(development.reference_split_ids):
        if split_index == 0:
            split_macro = matched
            split_rank = rank
            split_alpha = alpha
            split_selection = selection
        else:
            split_development = replace(
                development,
                reference_means=development.reference_means_by_split[:, split_index, :],
            )
            split_locked = replace(
                locked_test,
                reference_means=locked_test.reference_means_by_split[:, split_index, :],
            )
            split_rank, split_alpha, split_selection = select_hyperparameters(
                split_development,
                split_development.frozen_features,
                ranks=ranks,
                alphas=alphas,
                minimum_support=minimum_support,
                device=device,
                include_composition=include_composition,
            )
            split_macro, *_ = fit_and_score(
                split_development,
                split_locked,
                split_development.frozen_features,
                split_locked.frozen_features,
                rank=split_rank,
                alpha=split_alpha,
                minimum_support=minimum_support,
                device=device,
                include_composition=include_composition,
            )
        development_balance = development.reference_evaluation_balance.get(split_id)
        locked_balance = locked_test.reference_evaluation_balance.get(split_id)
        reference_split_sensitivity.append(
            {
                "split_id": split_id,
                "macro_r2": float(split_macro),
                "rank": int(split_rank),
                "alpha": float(split_alpha),
                "minimum_macro_r2": float(minimum_macro_r2),
                "meets_minimum_effect": bool(split_macro >= minimum_macro_r2),
                "development_selection": split_selection,
                "development_balance": development_balance,
                "locked_balance": locked_balance,
                "balance_pass": bool(
                    isinstance(development_balance, Mapping)
                    and isinstance(locked_balance, Mapping)
                    and development_balance.get("pass") is True
                    and locked_balance.get("pass") is True
                ),
            }
        )

    planned_coverage = {
        "development": development.coverage_audit,
        "locked_test": locked_test.coverage_audit,
    }
    coverage_fraction = min(
        float(development.coverage_audit.get("retained_fraction", 0.0)),
        float(locked_test.coverage_audit.get("retained_fraction", 0.0)),
    )
    locked_measurement_audit = locked_test.coverage_audit.get("locked_measurement_audit")
    locked_support_audit = locked_test.coverage_audit.get("locked_support_audit")
    if isinstance(locked_support_audit, Mapping):
        evaluated_fine_types = [
            fine_type
            for fine_type in locked_test.type_names
            if any(
                isinstance(row, Mapping)
                and row.get("evaluable") is True
                and str(key).endswith("|" + fine_type)
                for key, row in locked_support_audit.items()
            )
        ]
    else:
        evaluated_fine_types = list(locked_test.type_names)
    unevaluable_fine_types = [
        fine_type
        for fine_type in locked_test.type_names
        if fine_type not in set(evaluated_fine_types)
    ]
    best_nuisance_randomization = exact_paired_randomization(best_nuisance_effects)

    checks = {
        "primary_claim_is_explicit_local_context": local_context_hypothesis,
        "matched_macro_r2": matched >= minimum_macro_r2,
        "macro_donor_type_r2": stratum_macro >= minimum_macro_r2,
        "local_roi_null_separates": bool(local_null["pass"]),
        "spatial_block_null_separates": bool(block_null["pass"]),
        "every_required_null_separates": bool(local_null["pass"] and block_null["pass"]),
        "permutations_change_training_rows": bool(
            local_null["activity_pass"] and block_null["activity_pass"]
        ),
        "supported_donor_type_coverage": (
            coverage["supported_fraction"] >= minimum_strata_coverage
            and (
                section_coverage is None
                or section_coverage["retained_fraction"] >= minimum_strata_coverage
            )
        ),
        "positive_supported_strata": positive_fraction >= minimum_positive_strata_fraction,
        "donor_consistency": donor_consistency,
        "not_single_donor_driven": largest_donor_share <= 0.5,
        "beats_coordinate_only": matched - coordinate_macro >= minimum_coordinate_delta,
        "paired_coordinate_effect_ci_positive": coordinate_bootstrap["ci_95"][0] > 0.0,
        "beats_best_independently_tuned_nuisance": (
            matched - best_nuisance_macro >= minimum_coordinate_delta
        ),
        "paired_best_nuisance_effect_ci_positive": (best_nuisance_bootstrap["ci_95"][0] > 0.0),
        "matched_donor_bootstrap_ci_positive": matched_bootstrap["ci_95"][0] > 0.0,
        "expression_relevance": expression_reduction >= minimum_expression_error_reduction,
        "adequate_basis_ceiling": ceiling_r2 >= minimum_basis_ceiling_r2,
        "rank_direction_stable": all(row["macro_r2"] > 0 for row in rank_sensitivity),
        "reference_split_direction_stable": all(
            row["macro_r2"] >= minimum_macro_r2 for row in reference_split_sensitivity
        ),
        "planned_coverage_retained": coverage_fraction >= minimum_strata_coverage,
        "disease_inclusive_endpoint_reported": True,
        "disease_adjusted_or_single_disease_endpoint_reported": bool(
            disease_adjustment_available or len(set(disease_labels.tolist())) == 1
        ),
    }
    if regional_uni2h:
        checks.update(
            {
                "composition_adjusted_positive": bool(include_composition and matched > 0.0),
                "beats_stain_statistics_only": bool(
                    stain_macro is not None and matched - stain_macro >= minimum_stain_delta
                ),
                "paired_stain_effect_ci_positive": bool(
                    stain_bootstrap is not None and stain_bootstrap["ci_95"][0] > 0.0
                ),
            }
        )
    if local_context_hypothesis:
        checks["exact_donor_paired_main_effect"] = g2_closed_test_pass
    if final_inference:
        checks["source_coverage_audit_available"] = bool(
            development.coverage_audit and locked_test.coverage_audit
        )
        checks["reference_evaluation_balance_passes"] = all(
            row["balance_pass"] for row in reference_split_sensitivity
        )
        checks["locked_measurement_audit_passes"] = bool(
            isinstance(locked_measurement_audit, Mapping)
            and locked_measurement_audit.get("pass") is True
        )
    component_pass = all(checks.values())
    familywise_tested = bool(frozen_contrast_family.get("tested") is True)
    measurement_valid = bool(
        not final_inference
        or (
            isinstance(locked_measurement_audit, Mapping)
            and locked_measurement_audit.get("pass") is True
        )
    )
    morphology_conclusion = _classify_morphology_source(
        intrinsic_prespecified=intrinsic_prespecified,
        final_inference=final_inference,
        familywise_tested=familywise_tested,
        measurement_valid=measurement_valid,
        component_pass=component_pass,
        any_source_contrast_positive=any_source_contrast_positive,
        nucleus_specific=nucleus_specific,
        cell_specific=cell_specific,
        context_specific=context_specific,
        full_vs_context=full_vs_context,
        full_vs_intrinsic=full_vs_intrinsic,
    )
    nucleus_level = bool(intrinsic_prespecified and familywise_tested)
    cell_intrinsic_tested = bool(intrinsic_prespecified and familywise_tested)
    regional_endpoints = None
    if regional:
        regional_endpoints = {
            "raw_depth_adjusted": {
                "donor_equal_niche_equal_residual_coordinate_r2": raw_matched,
                "development_fitted_covariates": list(development.technical_covariate_names),
                "selected_hyperparameters": {"rank": raw_rank, "alpha": raw_alpha},
            },
            "composition_adjusted": (
                {
                    "donor_equal_niche_equal_residual_coordinate_r2": matched,
                    "development_fitted_covariates": list(
                        development.technical_covariate_names
                        + development.composition_feature_names
                    ),
                    "coordinate_only_macro_r2": coordinate_macro,
                    "stain_statistics_only_macro_r2": stain_macro,
                    "best_nuisance_family": best_nuisance_name,
                    "best_nuisance_macro_r2": best_nuisance_macro,
                    "uni2h_minus_best_nuisance_macro_r2": (matched - best_nuisance_macro),
                }
                if include_composition
                else None
            ),
            "correction_coefficients_fit_on_development_only": True,
            "correction_coefficients_fit_by_fine_type_on_development_only": True,
            "composition_scores_are_continuous_rna_only_controls": bool(
                development.composition_feature_names
            ),
            "composition_score_genes_excluded_from_scored_targets": bool(
                development.composition_feature_names
                and not (set(development.gene_ids) & set(development.type_marker_gene_ids))
            ),
        }
    checks["every_seed_separates_shuffle"] = checks["every_required_null_separates"]
    return {
        "schema_version": MORPHOLOGY_RIDGE_REPORT_SCHEMA,
        "status": (
            "synthetic_calibration_component_pass"
            if synthetic_calibration_mode and component_pass
            else (
                "synthetic_calibration_component_fail"
                if synthetic_calibration_mode
                else ("component_pass" if component_pass else "stop_or_pivot")
            )
        ),
        "component_pass": component_pass,
        "authorizes_full_heir": False,
        "final_inference": final_inference,
        "synthetic_calibration_execution": synthetic_calibration_mode,
        "scientific_authorization_suppressed": synthetic_calibration_mode,
        "calibration_exact_gate_settings": (
            expected_calibration_settings if synthetic_calibration_mode else None
        ),
        "calibration_exact_gate_settings_sha256": synthetic_settings_sha256,
        "calibration": calibration,
        "nucleus_hypothesis_tested": nucleus_level,
        "cell_intrinsic_hypothesis_tested": cell_intrinsic_tested,
        "local_context_hypothesis_tested": local_context_hypothesis,
        "regional_hypothesis_tested": regional,
        "scientific_scope": development.scientific_scope,
        "evidence_scope": (
            "synthetic_calibration_only"
            if synthetic_calibration_mode
            else (
                "development_only_exploratory" if regional else "within_GSE250346_internal_go_no_go"
            )
        ),
        "authorizes_population_inference": False,
        "authorizes_external_generalization": False,
        "authorizes_validated_regional_association": False,
        "encoder_hierarchy": {
            "primary": "MahmoodLab/UNI2-h",
            "replication_1": "bioptimus/H-optimus-1",
            "replication_2": "bioptimus/H0-mini",
            "hierarchy_frozen_before_locked_outcomes": True,
        },
        "crop_source_not_inferred_from_observation_level": True,
        "hypothesis_decisions": {
            "G2_local_context": {
                "tested": local_context_hypothesis,
                "pass": bool(local_context_hypothesis and component_pass),
                "primary_crop_id": development.primary_crop_id,
                "primary_crop_role": development.crop_roles[primary_crop_index],
                "primary_crop_comparison_family": primary_crop_family,
                "multiplicity_method": G2_MULTIPLICITY_METHOD,
                "nuisance_family_tests": g2_nuisance_closed_tests,
            },
            "G3_nucleus_intrinsic": {
                **nucleus_contrast,
                "legacy_strongest_comparator_diagnostic": nucleus_contrast,
                "familywise_frozen_contrasts": {
                    name: report
                    for name, report in frozen_reports.items()
                    if name.startswith("nucleus_")
                },
                "tested": nucleus_level,
                "pass": bool(
                    nucleus_level and nucleus_specific and final_inference and component_pass
                ),
                "requires_matched_artifact_contrasts": True,
            },
            "G3_cell_intrinsic": {
                **cell_contrast,
                "legacy_strongest_comparator_diagnostic": cell_contrast,
                "familywise_frozen_contrasts": {
                    name: report
                    for name, report in frozen_reports.items()
                    if name.startswith("cell_")
                },
                "tested": cell_intrinsic_tested,
                "pass": bool(
                    cell_intrinsic_tested and cell_specific and final_inference and component_pass
                ),
                "requires_matched_artifact_contrasts": True,
            },
            "G3_context_only": {
                **context_contrast,
                "legacy_strongest_comparator_diagnostic": context_contrast,
                "familywise_frozen_contrasts": {
                    name: report
                    for name, report in frozen_reports.items()
                    if name.startswith("context_")
                },
                "tested": bool(intrinsic_prespecified and familywise_tested),
                "pass": bool(
                    intrinsic_prespecified
                    and context_specific
                    and final_inference
                    and component_pass
                ),
            },
            "G3_mixed_intrinsic_context": {
                "tested": bool(intrinsic_prespecified and familywise_tested),
                "pass": bool(
                    intrinsic_prespecified
                    and mixed_information
                    and final_inference
                    and component_pass
                ),
                "full_context_vs_nucleus_pass": full_vs_nucleus,
                "full_context_vs_cell_pass": full_vs_cell,
                "full_context_vs_intrinsic_pass": full_vs_intrinsic,
                "full_context_vs_context_pass": full_vs_context,
            },
        },
        "frozen_morphology_contrast_family": frozen_contrast_family,
        "morphology_source_conclusion": morphology_conclusion,
        "authorizes_nucleus_intrinsic_claim": bool(
            nucleus_level
            and nucleus_specific
            and final_inference
            and component_pass
            and not synthetic_calibration_mode
        ),
        "authorizes_cell_intrinsic_claim": bool(
            cell_intrinsic_tested
            and cell_specific
            and final_inference
            and component_pass
            and not synthetic_calibration_mode
        ),
        "reason_full_heir_remains_blocked": (
            "HESCAPE pseudo-spots test regional image-to-expression signal, not one nucleus "
            "paired to that nucleus's RNA"
            if regional
            else (
                "requires independent encoder replication, non-overlapping cohort confirmation, "
                "and a separate matched-reference utility gate"
            )
        ),
        "oracle_type_only": True,
        "oracle_label_scope": (
            "rna_only_dominant_regional_niche" if regional else "registered_fine_cell_type"
        ),
        "selected_hyperparameters": {"rank": rank, "alpha": alpha},
        "measurement_qualification": {
            "study_manifest_sha256": development.study_manifest_sha256,
            "measurement_receipt_sha256": development.measurement_receipt_sha256,
            "measurement_source_sha256": development.measurement_source_sha256,
            "ordered_gene_ids": list(development.gene_ids),
            "supported_fine_type_ids": list(development.type_names),
            "evaluated_locked_fine_type_ids": evaluated_fine_types,
            "unevaluable_locked_fine_type_ids": unevaluable_fine_types,
            "claim_applies_to_all_h_meas_selected_fine_types": not unevaluable_fine_types,
        },
        "development_selection": selection,
        "baseline_hyperparameter_selection": {
            "common_molecular_rank_for_baseline_comparability": rank,
            "each_family_tuned_in_its_own_development_donor_folds": True,
            "locked_test_used_for_selection": False,
            "families": control_report,
            "raw_depth_adjusted_image": {
                "rank": raw_rank,
                "alpha": raw_alpha,
                "development_donor_folds": raw_selection,
            },
        },
        "control_models": control_report,
        "best_independently_tuned_nuisance_control": {
            "family": best_nuisance_name,
            "macro_r2": best_nuisance_macro,
            "matched_minus_control_macro_r2": matched - best_nuisance_macro,
            "donor_bootstrap": best_nuisance_bootstrap,
            "exact_donor_paired_randomization": exact_paired_randomization(best_nuisance_effects),
            "donor_dominance": donor_dominance(best_nuisance_effects),
        },
        "direct_crop_contrasts": {
            "nucleus_vs_context_and_nonimage": nucleus_contrast,
            "cell_vs_context_and_nonimage": cell_contrast,
            "target_removed_context_vs_intrinsic_and_nonimage": context_contrast,
            "multiplicity_scope": "G3 decisions are separate from the G2 local-context gate",
            "crop_family_multiplicity_calibrated_by_locked_receipt": bool(
                final_inference and calibration.get("available") is True
            ),
        },
        "primary_metrics": {
            "donor_equal_type_equal_residual_coordinate_r2": matched,
            "macro_donor_type_residual_coordinate_r2": stratum_macro,
            "raw_depth_adjusted_regional_macro_r2": raw_matched if regional else None,
            "composition_adjusted_regional_macro_r2": (matched if include_composition else None),
            "coordinate_only_macro_r2": coordinate_macro,
            "stain_statistics_only_macro_r2": stain_macro,
            "matched_minus_coordinate_macro_r2": matched - coordinate_macro,
            "basis_ceiling_r2": ceiling_r2,
            "expression_error_reduction_vs_reference_mean": expression_reduction,
            "positive_donor_type_fraction": positive_fraction,
            "largest_donor_positive_improvement_share": largest_donor_share,
            "matched_donor_bootstrap_ci_95": matched_bootstrap["ci_95"],
            "matched_minus_coordinate_donor_bootstrap_ci_95": coordinate_bootstrap["ci_95"],
            "matched_minus_stain_donor_bootstrap_ci_95": (
                stain_bootstrap["ci_95"] if stain_bootstrap is not None else None
            ),
            "matched_minus_best_nuisance_donor_bootstrap_ci_95": (best_nuisance_bootstrap["ci_95"]),
        },
        "disease_endpoints": {
            "disease_inclusive": {
                "donor_equal_type_equal_residual_coordinate_r2": matched,
                "development_fitted_covariates": list(development.technical_covariate_names),
                "within_disease": disease_stratification,
                "within_disease_donor_balanced": disease_within_donor_balanced,
            },
            "disease_adjusted": {
                "available": disease_adjustment_available,
                "donor_equal_type_equal_residual_coordinate_r2": (disease_adjusted_macro),
                "selected_hyperparameters": {
                    "rank": disease_rank,
                    "alpha": disease_alpha,
                },
                "development_selection": disease_selection,
                "donor_macro_r2": disease_adjusted_donors,
                "donor_type_rows": disease_adjusted_rows,
                "within_disease": disease_adjusted_stratification,
                "within_disease_donor_balanced": (disease_adjusted_within_donor_balanced),
                "correction_fit_on_development_only": True,
            },
        },
        "regional_endpoints": regional_endpoints,
        "donor_type_rows": rows,
        "donor_macro_r2": donor_macro,
        "per_donor_effects": per_donor_effects,
        "leave_one_locked_donor_out": {
            "method": "recompute locked metric after omission; never refit on locked donors",
            "matched_macro_r2": leave_one_donor_out(donor_macro),
            "matched_minus_coordinate_macro_r2": leave_one_donor_out(coordinate_effects),
            "matched_minus_stain_macro_r2": (
                leave_one_donor_out(stain_effects) if stain_macro is not None else None
            ),
            "matched_minus_best_nuisance_macro_r2": leave_one_donor_out(best_nuisance_effects),
        },
        "donor_bootstrap": {
            "matched_macro_r2": matched_bootstrap,
            "matched_minus_coordinate_macro_r2": coordinate_bootstrap,
            "matched_minus_stain_macro_r2": stain_bootstrap,
            "matched_minus_best_nuisance_macro_r2": best_nuisance_bootstrap,
        },
        "exact_donor_paired_randomization": {
            "matched_minus_coordinate": exact_paired_randomization(coordinate_effects),
            "matched_minus_stain": (
                exact_paired_randomization(stain_effects) if stain_macro is not None else None
            ),
            "matched_minus_best_nuisance": best_nuisance_randomization,
        },
        "donor_dominance": {
            "matched": dominance,
            "matched_minus_coordinate": donor_dominance(coordinate_effects),
            "matched_minus_stain": (
                donor_dominance(stain_effects) if stain_macro is not None else None
            ),
            "matched_minus_best_nuisance": donor_dominance(best_nuisance_effects),
        },
        "coverage": {
            "development_donor_support_by_fine_type": development_type_support,
            "locked_donor_support_by_fine_type": locked_type_support,
            "locked_donor_type": coverage,
            "locked_donor_section_type": section_coverage,
            "locked_observations": len(locked_test.observation_ids),
            "development_observations": len(development.observation_ids),
            "development_rois": len(set(development.roi_ids.tolist())),
            "development_blocks": len(set(development.block_ids.tolist())),
            "locked_rois": len(set(locked_test.roi_ids.tolist())),
            "locked_blocks": len(set(locked_test.block_ids.tolist())),
            "source_exclusion_and_reference_count_audit_available": bool(
                development.coverage_audit and locked_test.coverage_audit
            ),
            "planned_biological_coverage": planned_coverage,
            "minimum_prepared_planned_coverage_fraction": coverage_fraction,
            "reference_evaluation_balance": {
                "development": development.reference_evaluation_balance,
                "locked_test": locked_test.reference_evaluation_balance,
            },
        },
        "stratification": {
            "section": section_stratification,
            "disease": disease_stratification,
            "disease_adjusted": disease_adjusted_stratification,
            "site": group_stratification(
                truth_coordinates,
                predicted_coordinates,
                locked_test.site_ids,
                locked_test.type_labels,
                minimum_support,
                group_name="site",
                source="explicit_artifact_metadata",
            ),
            "batch": group_stratification(
                truth_coordinates,
                predicted_coordinates,
                locked_test.batch_ids,
                locked_test.type_labels,
                minimum_support,
                group_name="batch",
                source="explicit_artifact_metadata",
            ),
        },
        "permutation_control": local_null,
        "spatial_block_permutation_control": block_null,
        "null_controls": {
            "local_within_roi": local_null,
            "spatial_block_reassignment": block_null,
            "label_preserving_synthetic_null": {
                "implemented_as_calibration_hook": True,
                "calibration_receipt": calibration,
            },
        },
        "rank_sensitivity": rank_sensitivity,
        "reference_split_sensitivity": reference_split_sensitivity,
        "macro_gate_relevance": {
            "basis_ceiling_donor_type_rows": ceiling_rows,
            "basis_ceiling_donor_macro": ceiling_donors,
            "expression_reduction_donor_type_rows": expression_rows,
            "expression_reduction_donor_macro": expression_donors,
        },
        "thresholds": {
            "minimum_macro_r2": minimum_macro_r2,
            "minimum_shuffle_delta": minimum_shuffle_delta,
            "minimum_coordinate_delta": minimum_coordinate_delta,
            "minimum_stain_delta": minimum_stain_delta,
            "maximum_direct_contrast_p": maximum_direct_contrast_p,
            "minimum_mask_implementation_pass_fraction": (
                minimum_mask_implementation_pass_fraction
            ),
            "minimum_null_shuffled_fraction": minimum_null_shuffled_fraction,
            "minimum_strata_coverage": minimum_strata_coverage,
            "maximum_permutation_p": maximum_permutation_p,
            "minimum_positive_strata_fraction": minimum_positive_strata_fraction,
            "minimum_expression_error_reduction": minimum_expression_error_reduction,
            "minimum_basis_ceiling_r2": minimum_basis_ceiling_r2,
            "minimum_support": minimum_support,
            "minimum_locked_donors": required_locked_donors,
            "donor_bootstrap_iterations": donor_bootstrap_iterations,
            "minimum_final_permutations": minimum_final_permutations,
        },
        "checks": checks,
        "execution": {
            "device": str(resolve_device(device)),
            "development_donors": development_donors,
            "locked_test_donors": locked_donors,
            "nuisance_fit_weighting": "fine_type_specific_equal_donor_development_only",
            "molecular_basis_weighting": "equal_donor_within_fine_type_development_only",
            "gate_aggregation": "donor_equal_then_type_equal_with_macro_stratum_companion",
            "section_batch_indicator_scope": "development_folds_only",
            "unseen_locked_section_or_batch_fully_adjusted": False,
            "shared_molecular_basis_across_control_models": True,
            "prespecified_fixed_hyperparameters": prespecified_fixed_hyperparameters,
            "requested_unique_permutations_per_null": requested_permutations,
            "excluded_components": [
                "oracle_free_type_classifier",
                "neural_residual_head",
                "uot",
                "graph",
                "unknown_mass",
                "refinement",
            ],
        },
    }


__all__ = [
    "MORPHOLOGY_RIDGE_REPORT_SCHEMA",
    "evaluate_morphology_ridge_gate",
    "validate_experiment_identity",
]
