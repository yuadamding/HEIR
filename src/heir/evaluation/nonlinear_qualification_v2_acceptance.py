"""Shared scientific scoring and synthetic decision calibration for HEST v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.evaluation.hierarchical_metrics import (
    donor_dominance,
    paired_donor_effects,
)
from heir.evaluation.nonlinear_qualification_v2_contract import (
    ANALYSIS_STATUS,
    CALIBRATION_CONDITION_PARAMETERS,
    CALIBRATION_COORDINATE_RANK,
    CALIBRATION_EXPECTATIONS,
    CALIBRATION_MOLECULAR_WIDTH,
    CALIBRATION_NULL_P,
    CALIBRATION_RARE_DECILE,
    CALIBRATION_SEED_BASE,
    evaluate_decisions,
    exact_sign_flip_max_test,
)
from heir.evaluation.registration_sensitivity_v2 import (
    COMPARISON_IDS,
    evaluate_registration_noninferiority,
)

SCHEMA = "heir.hest_nonlinear_v2_scientific_acceptance.v1"
CALIBRATION_TRIAL_SCHEMA = "heir.hest_nonlinear_v2_synthetic_trial.v1"
REQUIRED_ARMS = (
    "B2",
    "N0",
    "N1",
    "N3",
    "N4",
    "N5",
    "A_star",
    "C_blank",
    "C_blank_single",
)
NULL_FAMILIES = (
    "within_section_type_refitted_null_empirical_p",
    "different_spatial_block_refitted_null_empirical_p",
)


def _group_indices(*values: np.ndarray) -> tuple[tuple[tuple[object, ...], np.ndarray], ...]:
    arrays = tuple(np.asarray(value) for value in values)
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("qualification grouping vectors are not row aligned")
    groups: dict[tuple[object, ...], list[int]] = {}
    for row, key in enumerate(zip(*(array.tolist() for array in arrays))):
        groups.setdefault(key, []).append(row)
    return tuple((key, np.asarray(groups[key], dtype=np.int64)) for key in sorted(groups))


@dataclass(frozen=True)
class ArmPrediction:
    """One row-aligned outer-crossfit prediction under its selected target basis."""

    coordinate_truth: np.ndarray
    coordinate_prediction: np.ndarray
    coordinate_ranks: np.ndarray
    molecular_prediction: np.ndarray
    rare_lower: np.ndarray
    rare_upper: np.ndarray
    prediction_available: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RegistrationComparison:
    """Focal-basis coordinates for one frozen paired registration contrast."""

    truth: np.ndarray
    focal_prediction: np.ndarray
    comparator_prediction: np.ndarray
    coordinate_ranks: np.ndarray


@dataclass(frozen=True)
class QualificationInputs:
    donor_ids: np.ndarray
    section_ids: np.ndarray
    type_labels: np.ndarray
    molecular_truth: np.ndarray
    reference_prediction: np.ndarray
    arms: Mapping[str, ArmPrediction]
    registration: Mapping[str, RegistrationComparison]
    quality_mask: np.ndarray
    all_rows_mask: np.ndarray
    best_mask: np.ndarray
    included_donor_ids: Sequence[str]
    refitted_null_empirical_p: Mapping[str, Optional[float]]


def _vectors(inputs: QualificationInputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    donors = np.asarray(inputs.donor_ids).astype(str)
    sections = np.asarray(inputs.section_ids).astype(str)
    labels = np.asarray(inputs.type_labels, dtype=np.int64)
    rows = len(donors)
    if rows == 0 or sections.shape != (rows,) or labels.shape != (rows,):
        raise ValueError("qualification identities must be nonempty row-aligned vectors")
    if any(not value for value in donors.tolist()) or any(not value for value in sections.tolist()):
        raise ValueError("qualification donor and section identifiers cannot be empty")
    if np.any(labels < 0):
        raise ValueError("qualification type labels must be nonnegative")
    return donors, sections, labels


def _matrix(value: object, name: str, rows: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or len(result) != rows or result.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty row-aligned matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _rank_vector(value: object, rows: int, width: int) -> np.ndarray:
    ranks = np.asarray(value)
    if ranks.shape != (rows,) or ranks.dtype.kind not in {"i", "u"}:
        raise ValueError("coordinate ranks must be a row-aligned integer vector")
    result = ranks.astype(np.int64, copy=False)
    if np.any(result <= 0) or np.any(result > width):
        raise ValueError("coordinate ranks lie outside the padded coordinate width")
    return result


def _donor_type_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: tuple[tuple[tuple[object, ...], np.ndarray], ...],
    minimum_support: int,
    *,
    baseline: Optional[np.ndarray] = None,
) -> tuple[float, list[Mapping[str, object]], Mapping[str, float]]:
    rows = []
    donor_values: dict[str, list[float]] = {}
    for (donor, type_index), indices in groups:
        support = len(indices)
        if support < minimum_support:
            continue
        local_truth = truth[indices]
        if baseline is None:
            centered = local_truth - local_truth.mean(axis=0, keepdims=True)
            denominator = float(np.square(centered).sum())
            error = float(np.square(prediction[indices] - local_truth).sum())
            value = 1.0 - error / denominator if denominator > 1.0e-12 else float("nan")
            if not np.isfinite(value):
                continue
            metric = {"residual_coordinate_r2": float(value)}
        else:
            baseline_rmse = float(np.sqrt(np.mean(np.square(local_truth - baseline[indices]))))
            model_rmse = float(np.sqrt(np.mean(np.square(local_truth - prediction[indices]))))
            value = (baseline_rmse - model_rmse) / max(baseline_rmse, 1.0e-12)
            metric = {"relative_rmse_reduction": float(value)}
        rows.append(
            {
                "donor_id": donor,
                "type_index": int(type_index),
                "support": support,
                **metric,
            }
        )
        donor_values.setdefault(str(donor), []).append(float(value))
    donor_macro = {
        donor: float(np.mean(values)) for donor, values in donor_values.items() if values
    }
    if not donor_macro:
        raise ValueError(
            "no supported locked donor/type stratum is evaluable"
            if baseline is None
            else "no supported donor/type error-reduction stratum is evaluable"
        )
    return float(np.mean(list(donor_macro.values()))), rows, donor_macro


def _donor_section_type_r2(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: tuple[tuple[tuple[object, ...], np.ndarray], ...],
    minimum_support: int,
) -> tuple[
    float,
    list[Mapping[str, object]],
    Mapping[str, float],
    Mapping[str, Mapping[str, float]],
]:
    rows = []
    grouped: dict[str, dict[str, list[tuple[object, np.ndarray]]]] = {}
    for (donor, section, type_index), indices in groups:
        grouped.setdefault(str(donor), {}).setdefault(str(section), []).append(
            (type_index, indices)
        )
    donor_sections: dict[str, dict[str, float]] = {}
    donor_macro: dict[str, float] = {}
    for donor in sorted(grouped):
        sections = {}
        for section in sorted(grouped[donor]):
            type_values = []
            for type_index, indices in grouped[donor][section]:
                support = len(indices)
                row = {
                    "donor_id": donor,
                    "section_id": section,
                    "type_index": int(type_index),
                    "support": support,
                }
                if support < minimum_support:
                    rows.append(
                        {
                            **row,
                            "evaluable": False,
                            "residual_coordinate_r2": None,
                            "reason": "support_below_minimum",
                        }
                    )
                    continue
                local_truth = truth[indices]
                centered = local_truth - local_truth.mean(axis=0, keepdims=True)
                denominator = float(np.square(centered).sum())
                error = float(np.square(prediction[indices] - local_truth).sum())
                value = 1.0 - error / denominator if denominator > 1.0e-12 else float("nan")
                if not np.isfinite(value):
                    rows.append(
                        {
                            **row,
                            "evaluable": False,
                            "residual_coordinate_r2": None,
                            "reason": "zero_truth_variance",
                        }
                    )
                    continue
                rows.append({**row, "evaluable": True, "residual_coordinate_r2": float(value)})
                type_values.append(float(value))
            if not type_values:
                raise ValueError(
                    f"observed donor/section has no evaluable type support: {donor}/{section}"
                )
            sections[section] = float(np.mean(type_values))
        donor_sections[donor] = sections
        donor_macro[donor] = float(np.mean(list(sections.values())))
    if not donor_macro:
        raise ValueError("no donor/section/type stratum is evaluable")
    return float(np.mean(list(donor_macro.values()))), rows, donor_macro, donor_sections


def _validated_arm(
    arm: ArmPrediction,
    *,
    name: str,
    rows: int,
    molecular_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = _matrix(arm.coordinate_truth, f"{name} coordinate truth", rows)
    prediction = _matrix(arm.coordinate_prediction, f"{name} coordinate prediction", rows)
    if prediction.shape != truth.shape:
        raise ValueError(f"{name} coordinate truth and prediction shapes differ")
    ranks = _rank_vector(arm.coordinate_ranks, rows, truth.shape[1])
    active = np.arange(truth.shape[1])[None, :] < ranks[:, None]
    if np.any(truth[~active] != 0.0) or np.any(prediction[~active] != 0.0):
        raise ValueError(f"{name} inactive padded coordinates must be zero")
    molecular = _matrix(arm.molecular_prediction, f"{name} molecular prediction", rows)
    if molecular.shape[1] != molecular_width:
        raise ValueError(f"{name} molecular prediction width differs from truth")
    lower = _matrix(arm.rare_lower, f"{name} rare lower", rows)
    upper = _matrix(arm.rare_upper, f"{name} rare upper", rows)
    if lower.shape != truth.shape or upper.shape != truth.shape or np.any(lower >= upper):
        raise ValueError(f"{name} rare-state thresholds are malformed")
    available = (
        np.ones(rows, dtype=bool)
        if arm.prediction_available is None
        else np.asarray(arm.prediction_available)
    )
    if available.dtype != np.bool_ or available.shape != (rows,):
        raise ValueError(f"{name} prediction availability must be a row-aligned boolean mask")
    return truth, prediction, ranks, molecular, lower, upper, available


def _molecular_variance_ratio(
    truth: np.ndarray,
    prediction: np.ndarray,
    minimum_support: int,
    groups: tuple[tuple[tuple[object, ...], np.ndarray], ...],
) -> float:
    grouped: dict[str, list[float]] = {}
    for (donor, _), indices in groups:
        if len(indices) < minimum_support:
            continue
        truth_scale = np.std(truth[indices], axis=0)
        prediction_scale = np.std(prediction[indices], axis=0)
        evaluable = truth_scale > 1.0e-12
        if np.any(evaluable):
            grouped.setdefault(str(donor), []).append(
                float(np.median(prediction_scale[evaluable] / truth_scale[evaluable]))
            )
    donor_values = [float(np.mean(grouped[donor])) for donor in sorted(grouped)]
    if not donor_values:
        raise ValueError("molecular variance ratio has no supported donor/type strata")
    return float(np.mean(donor_values))


def _rare_state_recall(
    truth: np.ndarray,
    prediction: np.ndarray,
    ranks: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: tuple[tuple[tuple[object, ...], np.ndarray], ...],
) -> Mapping[str, object]:
    donor_values: dict[str, float] = {}
    evaluable_tails = 0
    grouped: dict[str, list[float]] = {}
    for (donor, _), indices in groups:
        local_truth = truth[indices]
        local_prediction = prediction[indices]
        local_ranks = ranks[indices]
        local_lower = lower[indices]
        local_upper = upper[indices]
        coordinate_values = []
        for coordinate in range(truth.shape[1]):
            active = local_ranks > coordinate
            if not active.any():
                continue
            tail_values = []
            low = active & (local_truth[:, coordinate] <= local_lower[:, coordinate])
            high = active & (local_truth[:, coordinate] >= local_upper[:, coordinate])
            if low.any():
                tail_values.append(
                    float(
                        np.mean(local_prediction[low, coordinate] <= local_lower[low, coordinate])
                    )
                )
                evaluable_tails += 1
            if high.any():
                tail_values.append(
                    float(
                        np.mean(local_prediction[high, coordinate] >= local_upper[high, coordinate])
                    )
                )
                evaluable_tails += 1
            if tail_values:
                coordinate_values.append(float(np.mean(tail_values)))
        if coordinate_values:
            grouped.setdefault(str(donor), []).append(float(np.mean(coordinate_values)))
    donor_values = {donor: float(np.mean(grouped[donor])) for donor in sorted(grouped)}
    return {
        "mean_recall": float(np.mean(list(donor_values.values()))) if donor_values else None,
        "per_donor": donor_values,
        "evaluable_tails": evaluable_tails,
    }


def _score_arm(
    arm: ArmPrediction,
    *,
    name: str,
    donors: np.ndarray,
    sections: np.ndarray,
    labels: np.ndarray,
    molecular_truth: np.ndarray,
    reference: np.ndarray,
    minimum_support: int,
    donor_type_groups: tuple[tuple[tuple[object, ...], np.ndarray], ...],
    donor_section_type_groups: tuple[tuple[tuple[object, ...], np.ndarray], ...],
) -> Mapping[str, object]:
    truth, prediction, ranks, molecular, lower, upper, available = _validated_arm(
        arm,
        name=name,
        rows=len(donors),
        molecular_width=molecular_truth.shape[1],
    )
    donor_type, strata, donor_values = _donor_type_summary(
        truth,
        prediction,
        donor_type_groups,
        minimum_support,
    )
    donor_section_type, section_rows, section_donors, _ = _donor_section_type_r2(
        truth,
        prediction,
        donor_section_type_groups,
        minimum_support,
    )
    reduction, reduction_rows, reduction_donors = _donor_type_summary(
        molecular_truth,
        molecular,
        donor_type_groups,
        minimum_support,
        baseline=reference,
    )
    type_coverage = [
        float(np.mean(available[labels == label]))
        for label in sorted(set(labels.tolist()))
        if np.any(labels == label)
    ]
    rare = _rare_state_recall(
        truth,
        prediction,
        ranks,
        lower,
        upper,
        donor_type_groups,
    )
    return {
        "arm_id": name,
        "donor_type_macro_residual_coordinate_R2": donor_type,
        "donor_section_type_macro_residual_coordinate_R2": donor_section_type,
        "relative_molecular_error_reduction_over_reference_mean": reduction,
        "molecular_variance_ratio": _molecular_variance_ratio(
            molecular_truth,
            molecular,
            minimum_support,
            donor_type_groups,
        ),
        "median_type_coverage": float(np.median(type_coverage)),
        "abstention_rate": float(np.mean(~available)),
        "positive_donor_type_strata_fraction": float(
            np.mean([float(row["residual_coordinate_r2"]) > 0.0 for row in strata])
        ),
        "donor_type_strata": strata,
        "donor_section_type_strata": section_rows,
        "molecular_error_reduction_strata": reduction_rows,
        "donor_type_R2": donor_values,
        "donor_section_type_R2": section_donors,
        "donor_molecular_error_reduction": reduction_donors,
        "rare_state": rare,
    }


def _comparison_effects(
    scores: Mapping[str, Mapping[str, object]], focal: str, comparator: str
) -> Mapping[str, float]:
    focal_values = scores[focal]["donor_type_R2"]
    comparator_values = scores[comparator]["donor_type_R2"]
    if not isinstance(focal_values, Mapping) or not isinstance(comparator_values, Mapping):
        raise AssertionError("arm scores must expose per-donor R2")
    return paired_donor_effects(focal_values, comparator_values)


def _registration(
    inputs: QualificationInputs,
    donors: np.ndarray,
    labels: np.ndarray,
    *,
    bootstrap_resamples: int,
) -> Mapping[str, object]:
    if set(inputs.registration) != set(COMPARISON_IDS):
        raise ValueError("qualification registration comparisons differ from v2")
    result = {}
    for offset, comparison_id in enumerate(COMPARISON_IDS):
        comparison = inputs.registration[comparison_id]
        result[comparison_id] = evaluate_registration_noninferiority(
            comparison.truth,
            comparison.focal_prediction,
            comparison.comparator_prediction,
            donors,
            labels,
            rank=comparison.coordinate_ranks,
            quality_mask=inputs.quality_mask,
            all_rows_mask=inputs.all_rows_mask,
            best_mask=inputs.best_mask,
            included_donor_ids=inputs.included_donor_ids,
            comparison_id=comparison_id,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=20_260_721 + offset * 10,
        )
    return result


def score_qualification(
    inputs: QualificationInputs,
    *,
    minimum_support: int = 20,
    registration_bootstrap_resamples: int = 20_000,
) -> Mapping[str, object]:
    """Score one observed or synthetic held-out prediction set and apply D1/D2."""

    if minimum_support != 20:
        raise ValueError("v2 primary scoring requires frozen support 20")
    donors, sections, labels = _vectors(inputs)
    molecular_truth = _matrix(inputs.molecular_truth, "molecular truth", len(donors))
    reference = _matrix(inputs.reference_prediction, "reference prediction", len(donors))
    if reference.shape != molecular_truth.shape:
        raise ValueError("molecular truth and reference prediction shapes differ")
    if not set(REQUIRED_ARMS) <= set(inputs.arms):
        raise ValueError("qualification predictions omit a required v2 arm")
    donor_type_groups = _group_indices(donors, labels)
    donor_section_type_groups = _group_indices(donors, sections, labels)
    scores = {
        name: _score_arm(
            arm,
            name=name,
            donors=donors,
            sections=sections,
            labels=labels,
            molecular_truth=molecular_truth,
            reference=reference,
            minimum_support=minimum_support,
            donor_type_groups=donor_type_groups,
            donor_section_type_groups=donor_section_type_groups,
        )
        for name, arm in inputs.arms.items()
    }
    direct_names = {
        "A_star_vs_B2": ("A_star", "B2"),
        "A_star_vs_N0": ("A_star", "N0"),
        "A_star_vs_C_blank": ("A_star", "C_blank"),
    }
    localization_names = {
        "N1_vs_N5": ("N1", "N5"),
        "N3_vs_N5": ("N3", "N5"),
        "N4_vs_N5": ("N4", "N5"),
    }
    direct_effects = {
        name: _comparison_effects(scores, *arms) for name, arms in direct_names.items()
    }
    localization_effects = {
        name: _comparison_effects(scores, *arms) for name, arms in localization_names.items()
    }
    direct_test = exact_sign_flip_max_test(direct_effects)
    localization_test = exact_sign_flip_max_test(localization_effects)
    registration = _registration(
        inputs,
        donors,
        labels,
        bootstrap_resamples=registration_bootstrap_resamples,
    )
    focal = scores["A_star"]
    b2_rare = scores["B2"]["rare_state"]["mean_recall"]
    focal_rare = focal["rare_state"]["mean_recall"]
    dominance = donor_dominance(direct_effects["A_star_vs_N0"])
    null_p = dict(inputs.refitted_null_empirical_p)
    if set(null_p) != set(NULL_FAMILIES):
        raise ValueError("qualification requires exactly both refitted-null p-values")
    primary_registration = registration["A_star_vs_N0"]["noninferiority"]
    d1_metrics = {
        "donor_type_macro_residual_coordinate_R2": focal["donor_type_macro_residual_coordinate_R2"],
        "donor_section_type_macro_residual_coordinate_R2": focal[
            "donor_section_type_macro_residual_coordinate_R2"
        ],
        "A_star_minus_B2_R2": focal["donor_type_macro_residual_coordinate_R2"]
        - scores["B2"]["donor_type_macro_residual_coordinate_R2"],
        "A_star_minus_N0_R2": focal["donor_type_macro_residual_coordinate_R2"]
        - scores["N0"]["donor_type_macro_residual_coordinate_R2"],
        "A_star_minus_C_blank_R2": focal["donor_type_macro_residual_coordinate_R2"]
        - scores["C_blank"]["donor_type_macro_residual_coordinate_R2"],
        "relative_molecular_error_reduction_over_reference_mean": focal[
            "relative_molecular_error_reduction_over_reference_mean"
        ],
        "positive_donor_type_strata_fraction": focal["positive_donor_type_strata_fraction"],
        "positive_donors_versus_N0_fraction": float(
            np.mean(np.asarray(list(direct_effects["A_star_vs_N0"].values())) > 0.0)
        ),
        "maximum_single_donor_gain_fraction": dominance["largest_positive_share"],
        "molecular_variance_ratio": focal["molecular_variance_ratio"],
        "median_type_coverage": focal["median_type_coverage"],
        "abstention_rate": focal["abstention_rate"],
        "rare_state_recall_drop_vs_B2": (
            None if b2_rare is None or focal_rare is None else float(b2_rare - focal_rare)
        ),
        "A_star_vs_B2_familywise_adjusted_p": direct_test["familywise_adjusted_p"]["A_star_vs_B2"],
        "A_star_vs_N0_familywise_adjusted_p": direct_test["familywise_adjusted_p"]["A_star_vs_N0"],
        "A_star_vs_C_blank_familywise_adjusted_p": direct_test["familywise_adjusted_p"][
            "A_star_vs_C_blank"
        ],
        **null_p,
        "registration_quality_point_difference": primary_registration["estimate"],
        "registration_quality_one_sided_lower_bound": primary_registration["one_sided_lower_bound"],
    }
    d2_metrics = {}
    for arm in ("N3", "N4"):
        comparison_id = f"{arm}_vs_N5"
        d2_metrics[arm] = {
            "increment_over_N5_R2": scores[arm]["donor_type_macro_residual_coordinate_R2"]
            - scores["N5"]["donor_type_macro_residual_coordinate_R2"],
            "localization_familywise_adjusted_p": localization_test["familywise_adjusted_p"][
                comparison_id
            ],
            "registration_quality_one_sided_lower_bound": registration[comparison_id][
                "noninferiority"
            ]["one_sided_lower_bound"],
            "absolute_donor_type_macro_R2": scores[arm]["donor_type_macro_residual_coordinate_R2"],
            "increment_over_N0_R2": scores[arm]["donor_type_macro_residual_coordinate_R2"]
            - scores["N0"]["donor_type_macro_residual_coordinate_R2"],
            "increment_over_C_blank_R2": scores[arm]["donor_type_macro_residual_coordinate_R2"]
            - scores["C_blank_single"]["donor_type_macro_residual_coordinate_R2"],
        }
    decisions = evaluate_decisions(d1_metrics, d2_metrics)
    return {
        "schema": SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "rows": len(donors),
        "donors": len(set(donors.tolist())),
        "minimum_primary_support": minimum_support,
        "arm_scores": scores,
        "direct_family": direct_test,
        "localization_family": localization_test,
        "registration_sensitivity": registration,
        "decision_metrics": {"D1": d1_metrics, "D2": d2_metrics},
        "decisions": decisions,
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_h_ref": False,
        "authorizes_full_heir": False,
    }


def _noise_prediction(
    truth: np.ndarray,
    rng: np.random.Generator,
    scale: float,
) -> np.ndarray:
    noise = rng.normal(size=truth.shape)
    norm = np.linalg.norm(noise, axis=1, keepdims=True)
    noise = noise / np.maximum(norm, np.finfo(np.float64).tiny)
    return truth + noise * (float(scale) * np.sqrt(truth.shape[1]))


def synthetic_trial_inputs(
    donor_ids: object,
    section_ids: object,
    type_labels: object,
    quality_mask: object,
    best_mask: object,
    included_donor_ids: Sequence[str],
    *,
    condition: str,
    trial_index: int,
) -> QualificationInputs:
    """Generate one deterministic held-out-prediction trial on real HEST topology."""

    if condition not in CALIBRATION_EXPECTATIONS or trial_index < 0:
        raise ValueError("synthetic calibration condition or trial index is invalid")
    donors = np.asarray(donor_ids).astype(str)
    sections = np.asarray(section_ids).astype(str)
    labels = np.asarray(type_labels, dtype=np.int64)
    rows = len(donors)
    if sections.shape != (rows,) or labels.shape != (rows,) or rows == 0:
        raise ValueError("synthetic topology vectors must be nonempty and aligned")
    quality = np.asarray(quality_mask)
    best = np.asarray(best_mask)
    if (
        quality.dtype != np.bool_
        or best.dtype != np.bool_
        or quality.shape != (rows,)
        or best.shape != (rows,)
    ):
        raise ValueError("synthetic registration masks must be aligned booleans")
    if np.any(best & ~quality):
        raise ValueError("synthetic best-registration rows must be nested in Q3")
    condition_index = tuple(CALIBRATION_EXPECTATIONS).index(condition)
    seed = CALIBRATION_SEED_BASE + 10_000 * condition_index + trial_index
    rng = np.random.default_rng(seed)
    rank = CALIBRATION_COORDINATE_RANK
    truth = rng.normal(size=(rows, rank))
    ranks = np.full(rows, rank, dtype=np.int64)
    donor_names = sorted(set(donors.tolist()))
    donor_index = {name: index for index, name in enumerate(donor_names)}
    compact_labels = {value: index for index, value in enumerate(sorted(set(labels.tolist())))}
    donor_codes = np.fromiter(
        (donor_index[value] for value in donors),
        dtype=np.float64,
        count=rows,
    )
    label_codes = np.fromiter(
        (compact_labels[int(value)] for value in labels),
        dtype=np.float64,
        count=rows,
    )
    reference = (0.02 * donor_codes + 0.01 * label_codes)[:, None] * np.arange(
        1,
        CALIBRATION_MOLECULAR_WIDTH + 1,
        dtype=np.float64,
    )
    basis_rng = np.random.default_rng(2_026_072_101)
    basis, _ = np.linalg.qr(basis_rng.normal(size=(CALIBRATION_MOLECULAR_WIDTH, rank)))
    molecular_truth = reference + truth @ basis.T
    lower = np.full_like(truth, -CALIBRATION_RARE_DECILE)
    upper = -lower

    scales = {
        "B2": 1.0,
        "N0": 1.0,
        "N1": 1.0,
        "N3": 1.0,
        "N4": 1.0,
        "N5": 1.0,
        "A_star": 1.0,
        "C_blank": 1.0,
        "C_blank_single": 1.0,
    }
    parameters = CALIBRATION_CONDITION_PARAMETERS[condition]
    scales.update(parameters["arm_error_scales"])
    aliases = dict(parameters["prediction_aliases"])
    special = parameters["special_construction"]

    predictions: dict[str, np.ndarray] = {}
    for arm, scale in scales.items():
        if arm in aliases:
            continue
        if special == "A_star_and_N1_predictions_equal_0.1_times_truth" and arm in {
            "A_star",
            "N1",
        }:
            predictions[arm] = 0.1 * truth
        elif special == ("A_star_and_N1_error_scale_1.15_on_Q3_and_zero_outside_Q3") and arm in {
            "A_star",
            "N1",
        }:
            values = _noise_prediction(truth, rng, 1.15)
            values[~quality] = truth[~quality]
            predictions[arm] = values
        elif special == "donor_constant_predictions_for_A_star_N1_N3_N4" and arm in {
            "A_star",
            "N1",
            "N3",
            "N4",
        }:
            code = rng.normal(size=(len(donor_names), rank))
            predictions[arm] = np.asarray([code[donor_index[value]] for value in donors])
        else:
            predictions[arm] = _noise_prediction(truth, rng, scale)
    for arm, source in aliases.items():
        predictions[arm] = predictions[source].copy()

    arms = {
        arm: ArmPrediction(
            coordinate_truth=truth,
            coordinate_prediction=prediction,
            coordinate_ranks=ranks,
            molecular_prediction=reference + prediction @ basis.T,
            rare_lower=lower,
            rare_upper=upper,
        )
        for arm, prediction in predictions.items()
    }
    registration_pairs = {
        "A_star_vs_N0": ("A_star", "N0"),
        "N1_vs_N5": ("N1", "N5"),
        "N3_vs_N5": ("N3", "N5"),
        "N4_vs_N5": ("N4", "N5"),
    }
    registration = {
        comparison: RegistrationComparison(
            truth=truth,
            focal_prediction=predictions[focal],
            comparator_prediction=predictions[comparator],
            coordinate_ranks=ranks,
        )
        for comparison, (focal, comparator) in registration_pairs.items()
    }
    return QualificationInputs(
        donor_ids=donors,
        section_ids=sections,
        type_labels=labels,
        molecular_truth=molecular_truth,
        reference_prediction=reference,
        arms=arms,
        registration=registration,
        quality_mask=quality,
        all_rows_mask=np.ones(rows, dtype=bool),
        best_mask=best,
        included_donor_ids=tuple(included_donor_ids),
        refitted_null_empirical_p={name: CALIBRATION_NULL_P for name in NULL_FAMILIES},
    )


def run_synthetic_trial(
    inputs: QualificationInputs,
    *,
    condition: str,
    trial_index: int,
    registration_bootstrap_resamples: int = 20_000,
) -> Mapping[str, object]:
    scored = score_qualification(
        inputs,
        registration_bootstrap_resamples=registration_bootstrap_resamples,
    )
    return {
        "schema": CALIBRATION_TRIAL_SCHEMA,
        "condition": condition,
        "trial_index": trial_index,
        "D1": bool(scored["decisions"]["D1"]["passed"]),
        "D2": bool(scored["decisions"]["D2"]["passed"]),
        "expected": dict(CALIBRATION_EXPECTATIONS[condition]),
        "matches_expectation": all(
            bool(scored["decisions"][decision]["passed"]) == expected
            for decision, expected in CALIBRATION_EXPECTATIONS[condition].items()
        ),
        "molecular_outcomes_read": False,
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_h_ref": False,
        "authorizes_full_heir": False,
        "scored": scored,
    }


__all__ = [
    "ArmPrediction",
    "CALIBRATION_TRIAL_SCHEMA",
    "NULL_FAMILIES",
    "QualificationInputs",
    "REQUIRED_ARMS",
    "RegistrationComparison",
    "SCHEMA",
    "run_synthetic_trial",
    "score_qualification",
    "synthetic_trial_inputs",
]
