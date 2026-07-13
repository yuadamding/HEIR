"""Deterministic synthetic calibration for the complete morphology gate.

The simulator never accepts a biological artifact.  It creates independent
synthetic development and evaluation donors, repeats all adaptive operations
on development data, and evaluates the final donor-level decision under both
the null and a frozen minimum image effect.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from typing import Mapping, Optional, Sequence

import numpy as np

from .power import (
    CALIBRATION_ENGINE,
    CALIBRATION_RECEIPT_SCHEMA,
    REQUIRED_CALIBRATION_SCENARIOS,
    REQUIRED_COMPLETE_GATE_CHECKS,
)

REQUIRED_SCENARIO_FAMILIES = REQUIRED_CALIBRATION_SCENARIOS


class CalibrationFailure(ValueError):
    """Raised when the empirical complete gate is not calibrated."""

    def __init__(self, message: str, diagnostic: Mapping[str, object]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


@dataclass(frozen=True)
class _Partition:
    donors: np.ndarray
    fine_types: np.ndarray
    active_strata: np.ndarray
    half_a: np.ndarray
    half_b: np.ndarray
    crops: tuple[np.ndarray, ...]
    nuisances: tuple[np.ndarray, ...]


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_float(mapping: Mapping[str, object], name: str) -> float:
    value = float(mapping[name])
    if not np.isfinite(value):
        raise ValueError("calibration %s must be finite" % name)
    return value


def _positive_integer(mapping: Mapping[str, object], name: str) -> int:
    value = mapping[name]
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError("calibration %s must be a positive integer" % name)
    return int(value)


def _validate_inputs(
    scenario_config: Mapping[str, object], thresholds: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    required_config = {
        "seed",
        "replicates_per_condition",
        "development_donors",
        "evaluation_donors",
        "fine_types",
        "cells_per_donor_type",
        "target_genes",
        "crop_families",
        "permutations",
        "minimum_effect_loading",
        "scenario_families",
    }
    required_thresholds = {
        "maximum_complete_gate_false_pass_probability",
        "minimum_power_at_minimum_meaningful_effect",
        "minimum_macro_r2",
        "minimum_image_minus_nuisance_r2",
        "maximum_exact_signflip_p",
        "minimum_positive_donor_fraction",
        "maximum_largest_positive_donor_share",
        "minimum_supported_strata_fraction",
        "minimum_active_permutation_strata_fraction",
        "minimum_gene_reliability",
        "minimum_reliable_gene_fraction",
        "minimum_reliable_donor_fraction",
        "ridge_alpha",
    }
    if set(str(value) for value in scenario_config) != required_config:
        raise ValueError("calibration scenario configuration is incomplete")
    if set(str(value) for value in thresholds) != required_thresholds:
        raise ValueError("calibration thresholds are incomplete")
    scenarios = tuple(str(value) for value in scenario_config["scenario_families"])
    if len(scenarios) != len(set(scenarios)) or set(scenarios) != set(
        REQUIRED_SCENARIO_FAMILIES
    ):
        raise ValueError("calibration must contain every required scenario family exactly once")
    config = {
        "seed": _positive_integer(scenario_config, "seed"),
        "replicates_per_condition": _positive_integer(
            scenario_config, "replicates_per_condition"
        ),
        "development_donors": _positive_integer(
            scenario_config, "development_donors"
        ),
        "evaluation_donors": _positive_integer(
            scenario_config, "evaluation_donors"
        ),
        "fine_types": _positive_integer(scenario_config, "fine_types"),
        "cells_per_donor_type": _positive_integer(
            scenario_config, "cells_per_donor_type"
        ),
        "target_genes": _positive_integer(scenario_config, "target_genes"),
        "crop_families": _positive_integer(scenario_config, "crop_families"),
        "permutations": _positive_integer(scenario_config, "permutations"),
        "minimum_effect_loading": _finite_float(
            scenario_config, "minimum_effect_loading"
        ),
        "scenario_families": list(scenarios),
    }
    if (
        config["replicates_per_condition"] < 10
        or config["development_donors"] < 4
        or config["evaluation_donors"] < 5
        or config["fine_types"] < 2
        or config["cells_per_donor_type"] < 6
        or config["target_genes"] < 4
        or config["crop_families"] < 2
        or config["permutations"] < 19
        or config["minimum_effect_loading"] <= 0
    ):
        raise ValueError("calibration scenario configuration is too small for the complete gate")
    if (
        config["replicates_per_condition"] > 200
        or config["development_donors"] > 12
        or config["evaluation_donors"] > 12
        or config["fine_types"] > 6
        or config["cells_per_donor_type"] > 64
        or config["target_genes"] > 32
        or config["crop_families"] > 8
        or config["permutations"] > 99
    ):
        raise ValueError("calibration scenario configuration exceeds compact runtime bounds")
    frozen_thresholds = {
        name: _finite_float(thresholds, name) for name in sorted(required_thresholds)
    }
    fractions = (
        "maximum_complete_gate_false_pass_probability",
        "minimum_power_at_minimum_meaningful_effect",
        "maximum_exact_signflip_p",
        "minimum_positive_donor_fraction",
        "maximum_largest_positive_donor_share",
        "minimum_supported_strata_fraction",
        "minimum_active_permutation_strata_fraction",
        "minimum_gene_reliability",
        "minimum_reliable_gene_fraction",
        "minimum_reliable_donor_fraction",
    )
    if any(not 0 <= frozen_thresholds[name] <= 1 for name in fractions):
        raise ValueError("calibration probability thresholds must be in [0, 1]")
    if frozen_thresholds["ridge_alpha"] <= 0:
        raise ValueError("calibration ridge_alpha must be positive")
    if frozen_thresholds["maximum_complete_gate_false_pass_probability"] > 0.05:
        raise ValueError("calibration false-pass requirement cannot exceed 0.05")
    if frozen_thresholds["minimum_power_at_minimum_meaningful_effect"] < 0.80:
        raise ValueError("calibration power requirement cannot be below 0.80")
    return config, frozen_thresholds


def _scenario_settings(name: str) -> Mapping[str, float]:
    settings = {
        "spatial_strength": 0.35,
        "disease_strength": 0.30,
        "section_strength": 0.30,
        "low_reliability_fraction": 0.0,
        "missing_fraction": 0.0,
        "inactive_fraction": 0.0,
        "unbalanced": 0.0,
        "nuisance_candidates": 3.0,
        "target_selection_noise": 0.0,
        "crop_noise_multiplier": 1.0,
    }
    if name == "spatial_autocorrelation":
        settings["spatial_strength"] = 1.25
    elif name == "disease_imbalance":
        settings["disease_strength"] = 1.25
        settings["disease_imbalance"] = 1.0
    elif name == "section_effects":
        settings["section_strength"] = 1.25
    elif name == "missing_fine_types":
        settings["missing_fraction"] = 0.20
    elif name == "variable_transcript_reliability":
        settings["low_reliability_fraction"] = 0.40
    elif name == "unbalanced_donor_cell_counts":
        settings["unbalanced"] = 1.0
    elif name == "inactive_permutation_strata":
        settings["inactive_fraction"] = 0.20
    elif name == "nuisance_selection":
        settings["nuisance_candidates"] = 7.0
    elif name == "target_panel_selection":
        settings["target_selection_noise"] = 1.0
        settings["low_reliability_fraction"] = 0.25
    elif name == "crop_family_multiplicity":
        settings["crop_noise_multiplier"] = 1.20
    else:  # pragma: no cover - protected by configuration validation
        raise ValueError("unknown calibration scenario family")
    return settings


def _partition(
    rng: np.random.Generator,
    *,
    donor_count: int,
    fine_type_count: int,
    cells_per_type: int,
    genes: int,
    crop_families: int,
    effect_loading: float,
    scenario: str,
    evaluation: bool,
) -> _Partition:
    settings = _scenario_settings(scenario)
    donor_values: list[str] = []
    type_values: list[int] = []
    active_values: list[bool] = []
    states: list[np.ndarray] = []
    disease_values: list[np.ndarray] = []
    section_values: list[np.ndarray] = []
    spatial_values: list[np.ndarray] = []
    coordinate_values: list[np.ndarray] = []
    for donor_index in range(donor_count):
        if settings.get("disease_imbalance", 0.0):
            cutoff = max(1, int(round(donor_count * (0.80 if evaluation else 0.50))))
            disease = float(donor_index < cutoff)
        else:
            disease = float(donor_index % 2)
        section = float(rng.normal())
        for fine_type in range(fine_type_count):
            stratum_index = donor_index * fine_type_count + fine_type
            unit = ((stratum_index * 37 + 11) % 100) / 100.0
            if unit < settings["missing_fraction"]:
                continue
            if unit < settings["missing_fraction"] + settings["inactive_fraction"]:
                count = 1
            elif settings["unbalanced"]:
                lower = max(6, cells_per_type // 2)
                count = int(rng.integers(lower, cells_per_type * 2 + 1))
            else:
                count = cells_per_type
            coordinate = np.linspace(-1.0, 1.0, count) + rng.normal(0.0, 0.04, count)
            spatial = np.sin(np.pi * coordinate) + rng.normal(0.0, 0.10, count)
            state = rng.normal(0.0, 1.0, count)
            donor_values.extend(["D%02d" % donor_index] * count)
            type_values.extend([fine_type] * count)
            active_values.extend([count > 1] * count)
            states.append(state)
            disease_values.append(np.full(count, disease))
            section_values.append(np.full(count, section))
            spatial_values.append(spatial)
            coordinate_values.append(coordinate)
    donors = np.asarray(donor_values)
    fine_types = np.asarray(type_values, dtype=np.int64)
    active = np.asarray(active_values, dtype=np.bool_)
    state = np.concatenate(states)
    disease = np.concatenate(disease_values)
    section = np.concatenate(section_values)
    spatial = np.concatenate(spatial_values)
    coordinate = np.concatenate(coordinate_values)
    biological_target = (
        state
        + settings["disease_strength"] * disease
        + settings["section_strength"] * section
        + settings["spatial_strength"] * spatial
    )
    half_a = np.empty((len(state), genes), dtype=np.float64)
    half_b = np.empty_like(half_a)
    low_count = int(round(genes * settings["low_reliability_fraction"]))
    for gene in range(genes):
        measurement_noise = 2.5 if gene >= genes - low_count and low_count else 0.25
        if settings["target_selection_noise"] and gene % 3 == 2:
            measurement_noise = 1.25
        loading = 1.0 - 0.05 * (gene % 4)
        shared = loading * biological_target + rng.normal(0.0, 0.15, len(state))
        half_a[:, gene] = shared + rng.normal(0.0, measurement_noise, len(state))
        half_b[:, gene] = shared + rng.normal(0.0, measurement_noise, len(state))
    nuisance_base = np.column_stack(
        (disease, section, spatial, coordinate, coordinate * coordinate)
    )
    crops = []
    for crop in range(crop_families):
        quality = max(0.35, 1.0 - 0.15 * crop)
        image_state = effect_loading * quality * state
        noise = (0.45 + 0.06 * crop) * settings["crop_noise_multiplier"]
        crops.append(
            np.column_stack(
                (
                    image_state
                    + 0.35 * disease
                    + 0.35 * section
                    + 0.35 * spatial
                    + rng.normal(0.0, noise, len(state)),
                    disease + rng.normal(0.0, 0.25, len(state)),
                    section + rng.normal(0.0, 0.25, len(state)),
                    spatial + rng.normal(0.0, 0.25, len(state)),
                )
            )
        )
    nuisance_count = int(settings["nuisance_candidates"])
    nuisances = [
        nuisance_base[:, :2],
        nuisance_base[:, 2:],
        nuisance_base,
    ]
    while len(nuisances) < nuisance_count:
        nuisances.append(rng.normal(0.0, 1.0, (len(state), 2)))
    return _Partition(
        donors=donors,
        fine_types=fine_types,
        active_strata=active,
        half_a=half_a,
        half_b=half_b,
        crops=tuple(crops),
        nuisances=tuple(nuisances),
    )


def _correlation(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    if len(first) < 4 or np.std(first) <= 1.0e-12 or np.std(second) <= 1.0e-12:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    if not np.isfinite(value):
        return None
    return max(0.0, 2.0 * value / (1.0 + value)) if value > 0 else 0.0


def _select_genes(
    partition: _Partition, thresholds: Mapping[str, float]
) -> tuple[np.ndarray, Mapping[str, object]]:
    donors = sorted(set(partition.donors.tolist()))
    types = sorted(set(partition.fine_types.tolist()))
    selected = []
    records = {}
    for gene in range(partition.half_a.shape[1]):
        donor_values = []
        within_values = []
        evaluable_donors = 0
        for donor in donors:
            donor_rows = partition.donors == donor
            value = _correlation(
                partition.half_a[donor_rows, gene],
                partition.half_b[donor_rows, gene],
            )
            if value is not None:
                donor_values.append(value)
                evaluable_donors += 1
            for fine_type in types:
                rows = donor_rows & (partition.fine_types == fine_type)
                within = _correlation(
                    partition.half_a[rows, gene], partition.half_b[rows, gene]
                )
                if within is not None:
                    within_values.append(within)
        donor_fraction = evaluable_donors / len(donors)
        donor_macro = float(np.median(donor_values)) if donor_values else None
        within_macro = float(np.median(within_values)) if within_values else None
        passed = bool(
            donor_macro is not None
            and within_macro is not None
            and donor_macro >= thresholds["minimum_gene_reliability"]
            and within_macro >= thresholds["minimum_gene_reliability"]
            and donor_fraction >= thresholds["minimum_reliable_donor_fraction"]
        )
        if passed:
            selected.append(gene)
        records["gene_%03d" % gene] = {
            "donor_macro_reliability": donor_macro,
            "within_type_donor_macro_reliability": within_macro,
            "evaluable_donor_fraction": donor_fraction,
            "pass": passed,
        }
    return np.asarray(selected, dtype=np.int64), records


def _target(partition: _Partition, selected: np.ndarray) -> np.ndarray:
    if not len(selected):
        return np.zeros(len(partition.donors), dtype=np.float64)
    return 0.5 * (
        partition.half_a[:, selected].mean(axis=1)
        + partition.half_b[:, selected].mean(axis=1)
    )


def _ridge_predict(
    training_features: np.ndarray,
    training_target: np.ndarray,
    evaluation_features: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean = training_features.mean(axis=0)
    scale = training_features.std(axis=0)
    scale[scale <= 1.0e-12] = 1.0
    training = (training_features - mean) / scale
    evaluation = (evaluation_features - mean) / scale
    design = np.column_stack((np.ones(len(training)), training))
    evaluation_design = np.column_stack((np.ones(len(evaluation)), evaluation))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ training_target)
    return evaluation_design @ coefficients


def _macro_r2(
    target: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    fine_types: np.ndarray,
    *,
    minimum_rows: int = 4,
) -> tuple[float, Mapping[str, float], float]:
    donor_values = {}
    planned = 0
    supported = 0
    for donor in sorted(set(donors.tolist())):
        type_values = []
        for fine_type in sorted(set(fine_types.tolist())):
            planned += 1
            rows = (donors == donor) & (fine_types == fine_type)
            if int(rows.sum()) < minimum_rows:
                continue
            denominator = float(np.sum((target[rows] - target[rows].mean()) ** 2))
            if denominator <= 1.0e-12:
                continue
            supported += 1
            residual = float(np.sum((target[rows] - prediction[rows]) ** 2))
            type_values.append(1.0 - residual / denominator)
        if type_values:
            donor_values[donor] = float(np.mean(type_values))
    macro = float(np.mean(list(donor_values.values()))) if donor_values else -np.inf
    return macro, donor_values, supported / planned if planned else 0.0


def _select_crop(
    development: _Partition,
    target: np.ndarray,
    *,
    alpha: float,
) -> tuple[int, Sequence[float]]:
    donors = sorted(set(development.donors.tolist()))
    scores = []
    for features in development.crops:
        fold_values = []
        for heldout in donors:
            evaluation = development.donors == heldout
            training = ~evaluation
            prediction = _ridge_predict(
                features[training], target[training], features[evaluation], alpha
            )
            macro, _, _ = _macro_r2(
                target[evaluation],
                prediction,
                development.donors[evaluation],
                development.fine_types[evaluation],
            )
            if np.isfinite(macro):
                fold_values.append(macro)
        scores.append(float(np.mean(fold_values)) if fold_values else -np.inf)
    selected = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[0]
    return selected, scores


def _permutation_indices(
    partition: _Partition,
    rng: np.random.Generator,
    *,
    kind: str,
) -> tuple[np.ndarray, float]:
    indices = np.arange(len(partition.donors), dtype=np.int64)
    result = indices.copy()
    active_strata = 0
    planned_strata = 0
    for donor in sorted(set(partition.donors.tolist())):
        for fine_type in sorted(set(partition.fine_types.tolist())):
            rows = np.flatnonzero(
                (partition.donors == donor) & (partition.fine_types == fine_type)
            )
            if not len(rows):
                continue
            planned_strata += 1
            if len(rows) < 2:
                continue
            if kind == "local_within_donor_type":
                permuted = rng.permutation(rows)
            elif kind == "spatial_block_shift":
                block_size = max(2, len(rows) // 3)
                shifts = tuple(
                    value
                    for value in range(block_size, len(rows), block_size)
                    if value < len(rows)
                )
                if not shifts:
                    continue
                permuted = np.roll(rows, int(shifts[int(rng.integers(len(shifts)))]))
            else:  # pragma: no cover - internal fixed contract
                raise ValueError("unknown calibration permutation kind")
            result[rows] = permuted
            active_strata += int(np.any(permuted != rows))
    return result, active_strata / planned_strata if planned_strata else 0.0


def _permutation_null(
    development_features: np.ndarray,
    development_target: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_target: np.ndarray,
    evaluation: _Partition,
    *,
    observed_macro_r2: float,
    alpha: float,
    permutations: int,
    seed: int,
    kind: str,
) -> Mapping[str, object]:
    rng = np.random.default_rng(seed)
    values = []
    activity = []
    for _ in range(permutations):
        indices, active_fraction = _permutation_indices(
            evaluation, rng, kind=kind
        )
        prediction = _ridge_predict(
            development_features,
            development_target,
            evaluation_features[indices],
            alpha,
        )
        macro, _, _ = _macro_r2(
            evaluation_target,
            prediction,
            evaluation.donors,
            evaluation.fine_types,
        )
        values.append(macro)
        activity.append(active_fraction)
    empirical_p = (
        1 + sum(value >= observed_macro_r2 - 1.0e-12 for value in values)
    ) / (permutations + 1)
    return {
        "kind": kind,
        "permutations": permutations,
        "plus_one_empirical_p": float(empirical_p),
        "mean_null_macro_r2": float(np.mean(values)),
        "minimum_active_strata_fraction": float(min(activity)),
        "training_model_refit": True,
        "crop_and_target_selection_partition": "synthetic_development_only",
    }


def _exact_signflip_p(effects: Mapping[str, float]) -> float:
    values = np.asarray([effects[name] for name in sorted(effects)], dtype=np.float64)
    observed = float(values.mean())
    exceedances = 0
    permutations = 0
    for signs in product((-1.0, 1.0), repeat=len(values)):
        permutations += 1
        if float(np.mean(values * np.asarray(signs))) >= observed - 1.0e-12:
            exceedances += 1
    return exceedances / permutations


def _complete_gate(
    development: _Partition,
    evaluation: _Partition,
    thresholds: Mapping[str, float],
    *,
    permutations: int,
    permutation_seed: int,
) -> Mapping[str, object]:
    selected_genes, reliability = _select_genes(development, thresholds)
    reliable_fraction = len(selected_genes) / development.half_a.shape[1]
    development_target = _target(development, selected_genes)
    evaluation_target = _target(evaluation, selected_genes)
    crop, crop_scores = _select_crop(
        development, development_target, alpha=thresholds["ridge_alpha"]
    )
    image_prediction = _ridge_predict(
        development.crops[crop],
        development_target,
        evaluation.crops[crop],
        thresholds["ridge_alpha"],
    )
    image_macro, image_donors, coverage = _macro_r2(
        evaluation_target,
        image_prediction,
        evaluation.donors,
        evaluation.fine_types,
    )
    nuisance_reports = []
    for development_features, evaluation_features in zip(
        development.nuisances, evaluation.nuisances
    ):
        prediction = _ridge_predict(
            development_features,
            development_target,
            evaluation_features,
            thresholds["ridge_alpha"],
        )
        nuisance_reports.append(
            _macro_r2(
                evaluation_target,
                prediction,
                evaluation.donors,
                evaluation.fine_types,
            )
        )
    nuisance_index = sorted(
        range(len(nuisance_reports)),
        key=lambda index: (-nuisance_reports[index][0], index),
    )[0]
    nuisance_macro, nuisance_donors, _ = nuisance_reports[nuisance_index]
    common_donors = sorted(set(image_donors) & set(nuisance_donors))
    effects = {
        donor: image_donors[donor] - nuisance_donors[donor]
        for donor in common_donors
    }
    positive = [value for value in effects.values() if value > 0]
    positive_fraction = len(positive) / len(effects) if effects else 0.0
    largest_share = max(positive) / sum(positive) if positive else 1.0
    signflip_p = _exact_signflip_p(effects) if effects else 1.0
    active_by_stratum = []
    for donor in sorted(set(evaluation.donors.tolist())):
        for fine_type in sorted(set(evaluation.fine_types.tolist())):
            rows = (evaluation.donors == donor) & (
                evaluation.fine_types == fine_type
            )
            if rows.any():
                active_by_stratum.append(bool(evaluation.active_strata[rows].all()))
    active_fraction = (
        float(np.mean(active_by_stratum)) if active_by_stratum else 0.0
    )
    local_null = _permutation_null(
        development.crops[crop],
        development_target,
        evaluation.crops[crop],
        evaluation_target,
        evaluation,
        observed_macro_r2=image_macro,
        alpha=thresholds["ridge_alpha"],
        permutations=permutations,
        seed=permutation_seed,
        kind="local_within_donor_type",
    )
    block_null = _permutation_null(
        development.crops[crop],
        development_target,
        evaluation.crops[crop],
        evaluation_target,
        evaluation,
        observed_macro_r2=image_macro,
        alpha=thresholds["ridge_alpha"],
        permutations=permutations,
        seed=permutation_seed + 1,
        kind="spatial_block_shift",
    )
    checks = {
        "development_target_panel_reliable": bool(
            len(selected_genes)
            and reliable_fraction >= thresholds["minimum_reliable_gene_fraction"]
        ),
        "supported_strata": coverage
        >= thresholds["minimum_supported_strata_fraction"],
        "permutation_strata_active": active_fraction
        >= thresholds["minimum_active_permutation_strata_fraction"],
        "image_macro_r2": image_macro >= thresholds["minimum_macro_r2"],
        "beats_strongest_nuisance": (
            image_macro - nuisance_macro
            >= thresholds["minimum_image_minus_nuisance_r2"]
        ),
        "exact_donor_signflip": signflip_p
        <= thresholds["maximum_exact_signflip_p"],
        "local_roi_permutation_null": local_null["plus_one_empirical_p"]
        <= thresholds["maximum_exact_signflip_p"],
        "spatial_block_permutation_null": block_null["plus_one_empirical_p"]
        <= thresholds["maximum_exact_signflip_p"],
        "both_permutation_nulls_active": min(
            local_null["minimum_active_strata_fraction"],
            block_null["minimum_active_strata_fraction"],
        )
        >= thresholds["minimum_active_permutation_strata_fraction"],
        "positive_donor_fraction": positive_fraction
        >= thresholds["minimum_positive_donor_fraction"],
        "not_single_donor_driven": largest_share
        <= thresholds["maximum_largest_positive_donor_share"],
        "development_only_crop_selection": True,
        "development_only_target_selection": True,
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "selected_gene_count": int(len(selected_genes)),
        "reliable_gene_fraction": float(reliable_fraction),
        "gene_reliability": reliability,
        "selected_crop_index": int(crop),
        "development_crop_scores": list(crop_scores),
        "strongest_nuisance_index": int(nuisance_index),
        "image_macro_r2": float(image_macro),
        "strongest_nuisance_macro_r2": float(nuisance_macro),
        "image_minus_nuisance_r2": float(image_macro - nuisance_macro),
        "positive_donor_fraction": float(positive_fraction),
        "largest_positive_donor_share": float(largest_share),
        "exact_signflip_p": float(signflip_p),
        "local_roi_permutation_null": local_null,
        "spatial_block_permutation_null": block_null,
        "supported_strata_fraction": float(coverage),
        "active_permutation_strata_fraction": active_fraction,
    }


def _condition_summary(reports: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    passes = int(sum(report["pass"] is True for report in reports))
    trials = len(reports)
    failures = {}
    for report in reports:
        for name, passed in report["checks"].items():
            if passed is not True:
                failures[name] = failures.get(name, 0) + 1
    selected_counts = [int(report["selected_gene_count"]) for report in reports]
    deltas = [float(report["image_minus_nuisance_r2"]) for report in reports]
    local_p = [
        float(report["local_roi_permutation_null"]["plus_one_empirical_p"])
        for report in reports
    ]
    block_p = [
        float(report["spatial_block_permutation_null"]["plus_one_empirical_p"])
        for report in reports
    ]
    return {
        "trials": trials,
        "complete_gate_passes": passes,
        "complete_gate_pass_fraction": float(passes / trials),
        "monte_carlo_standard_error": float(
            np.sqrt((passes / trials) * (1.0 - passes / trials) / trials)
        ),
        "failed_check_counts": dict(sorted(failures.items())),
        "selected_gene_count": {
            "minimum": min(selected_counts),
            "median": float(np.median(selected_counts)),
            "maximum": max(selected_counts),
        },
        "image_minus_nuisance_r2": {
            "median": float(np.median(deltas)),
            "p05": float(np.quantile(deltas, 0.05)),
            "p95": float(np.quantile(deltas, 0.95)),
        },
        "permutation_nulls": {
            "local_roi_median_empirical_p": float(np.median(local_p)),
            "spatial_block_median_empirical_p": float(np.median(block_p)),
            "local_roi_permutations": int(
                reports[0]["local_roi_permutation_null"]["permutations"]
            ),
            "spatial_block_permutations": int(
                reports[0]["spatial_block_permutation_null"]["permutations"]
            ),
        },
    }


def calibrate_morphology_gate(
    scenario_config: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> Mapping[str, object]:
    """Run all synthetic stress families and return a receipt only if calibrated."""

    config, frozen_thresholds = _validate_inputs(scenario_config, thresholds)
    scenario_results = {}
    for scenario_index, scenario in enumerate(config["scenario_families"]):
        null_reports = []
        effect_reports = []
        for replicate in range(config["replicates_per_condition"]):
            seed = np.random.SeedSequence(
                [config["seed"], scenario_index, replicate]
            )
            development_seed, evaluation_seed = seed.spawn(2)
            for effect, output in (
                (0.0, null_reports),
                (config["minimum_effect_loading"], effect_reports),
            ):
                development = _partition(
                    np.random.default_rng(development_seed),
                    donor_count=config["development_donors"],
                    fine_type_count=config["fine_types"],
                    cells_per_type=config["cells_per_donor_type"],
                    genes=config["target_genes"],
                    crop_families=config["crop_families"],
                    effect_loading=effect,
                    scenario=scenario,
                    evaluation=False,
                )
                evaluation = _partition(
                    np.random.default_rng(evaluation_seed),
                    donor_count=config["evaluation_donors"],
                    fine_type_count=config["fine_types"],
                    cells_per_type=config["cells_per_donor_type"],
                    genes=config["target_genes"],
                    crop_families=config["crop_families"],
                    effect_loading=effect,
                    scenario=scenario,
                    evaluation=True,
                )
                output.append(
                    _complete_gate(
                        development,
                        evaluation,
                        frozen_thresholds,
                        permutations=config["permutations"],
                        permutation_seed=(
                            config["seed"]
                            + scenario_index * 100_000
                            + replicate * 100
                        ),
                    )
                )
        scenario_results[scenario] = {
            "null": _condition_summary(null_reports),
            "minimum_meaningful_effect": _condition_summary(effect_reports),
        }
    maximum_false_pass = max(
        result["null"]["complete_gate_pass_fraction"]
        for result in scenario_results.values()
    )
    minimum_power = min(
        result["minimum_meaningful_effect"]["complete_gate_pass_fraction"]
        for result in scenario_results.values()
    )
    calibrated = bool(
        maximum_false_pass
        <= frozen_thresholds["maximum_complete_gate_false_pass_probability"]
        and minimum_power
        >= frozen_thresholds["minimum_power_at_minimum_meaningful_effect"]
    )
    scenario_config_sha256 = _sha256(config)
    thresholds_sha256 = _sha256(frozen_thresholds)
    simulation_core = {
        "engine": CALIBRATION_ENGINE,
        "scenario_config_sha256": scenario_config_sha256,
        "thresholds_sha256": thresholds_sha256,
        "scenario_results": scenario_results,
    }
    diagnostic = {
        "schema": "heir.morphology_gate_calibration_diagnostic.v1",
        "engine": CALIBRATION_ENGINE,
        "synthetic_data_only": True,
        "locked_outcomes_used": False,
        "complete_gate_executed": True,
        "scenario_config": config,
        "scenario_config_sha256": scenario_config_sha256,
        "thresholds": frozen_thresholds,
        "thresholds_sha256": thresholds_sha256,
        "scenario_families": list(config["scenario_families"]),
        "complete_gate_check_ids": list(REQUIRED_COMPLETE_GATE_CHECKS),
        "scenario_results": scenario_results,
        "maximum_complete_gate_false_pass_probability": float(maximum_false_pass),
        "power_at_minimum_meaningful_effect": float(minimum_power),
        "simulation_sha256": _sha256(simulation_core),
        "calibrated": calibrated,
        "device": "cpu",
        "cuda_used": False,
        "device_rationale": "small deterministic linear-algebra problems are CPU efficient",
    }
    if not calibrated:
        raise CalibrationFailure(
            "complete morphology gate failed empirical type-I-error or power calibration",
            diagnostic,
        )
    receipt_core = {
        **diagnostic,
        "schema": CALIBRATION_RECEIPT_SCHEMA,
        "pass": True,
    }
    return {
        **receipt_core,
        "receipt_content_sha256": _sha256(receipt_core),
    }


__all__ = [
    "CALIBRATION_ENGINE",
    "REQUIRED_SCENARIO_FAMILIES",
    "CalibrationFailure",
    "calibrate_morphology_gate",
]
