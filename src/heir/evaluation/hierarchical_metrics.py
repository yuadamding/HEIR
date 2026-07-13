"""Hierarchical metrics that treat donors as biological replicates."""

from __future__ import annotations

import itertools
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def macro_r2(
    truth: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    labels: np.ndarray,
    minimum_support: int,
) -> Tuple[float, Sequence[Mapping[str, object]], Mapping[str, float]]:
    """Average types within donor and then donors with equal biological weight."""

    rows = []
    donor_values: Dict[str, list[float]] = {}
    for donor in sorted(set(donors.tolist())):
        for type_index in sorted(set(labels[donors == donor].tolist())):
            selected = (donors == donor) & (labels == type_index)
            support = int(selected.sum())
            if support < minimum_support:
                continue
            centered = truth[selected] - truth[selected].mean(axis=0, keepdims=True)
            denominator = float(np.square(centered).sum())
            error = float(np.square(prediction[selected] - truth[selected]).sum())
            value = float(1.0 - error / denominator) if denominator > 1.0e-12 else float("nan")
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "donor_id": donor,
                    "type_index": int(type_index),
                    "support": support,
                    "residual_coordinate_r2": value,
                }
            )
            donor_values.setdefault(donor, []).append(value)
    donor_macro = {
        donor: float(np.mean(values)) for donor, values in donor_values.items() if values
    }
    if not donor_macro:
        raise ValueError("no supported locked donor/type stratum is evaluable")
    return float(np.mean(list(donor_macro.values()))), rows, donor_macro


def donor_section_type_macro_r2(
    truth: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    labels: np.ndarray,
    minimum_support: int,
) -> Tuple[
    float,
    Sequence[Mapping[str, object]],
    Mapping[str, float],
    Mapping[str, Mapping[str, float]],
]:
    """Average types within section, sections within donor, then donors.

    Types below ``minimum_support`` remain visible in the stratum rows but do
    not contribute to the section mean.  An observed donor/section with no
    evaluable type fails closed instead of silently disappearing from the
    endpoint.
    """

    truth_values = np.asarray(truth)
    prediction_values = np.asarray(prediction)
    donor_values = np.asarray(donors).astype(str)
    section_values = np.asarray(sections).astype(str)
    label_values = np.asarray(labels)
    row_count = len(truth_values)
    if prediction_values.shape != truth_values.shape:
        raise ValueError("section-balanced R2 truth and prediction are not row aligned")
    if any(values.shape != (row_count,) for values in (donor_values, section_values, label_values)):
        raise ValueError("section-balanced R2 identities are not row aligned")
    if minimum_support <= 0:
        raise ValueError("section-balanced R2 minimum support must be positive")
    if row_count == 0 or any(not value.strip() for value in donor_values.tolist()):
        raise ValueError("section-balanced R2 donor identities are missing")
    if any(not value.strip() for value in section_values.tolist()):
        raise ValueError("section-balanced R2 section identities are missing")

    rows = []
    donor_section_values: Dict[str, Dict[str, float]] = {}
    donor_macro: Dict[str, float] = {}
    for donor in sorted(set(donor_values.tolist())):
        sections_for_donor: Dict[str, float] = {}
        observed_sections = sorted(set(section_values[donor_values == donor].tolist()))
        for section in observed_sections:
            section_selected = (donor_values == donor) & (section_values == section)
            type_r2 = []
            for type_index in sorted(set(label_values[section_selected].tolist())):
                selected = section_selected & (label_values == type_index)
                support = int(selected.sum())
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
                centered = truth_values[selected] - truth_values[selected].mean(
                    axis=0, keepdims=True
                )
                denominator = float(np.square(centered).sum())
                error = float(np.square(prediction_values[selected] - truth_values[selected]).sum())
                value = float(1.0 - error / denominator) if denominator > 1.0e-12 else float("nan")
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
                rows.append(
                    {
                        **row,
                        "evaluable": True,
                        "residual_coordinate_r2": value,
                    }
                )
                type_r2.append(value)
            if not type_r2:
                raise ValueError(
                    "observed donor/section has no evaluable type support: %s/%s" % (donor, section)
                )
            sections_for_donor[section] = float(np.mean(type_r2))
        if not sections_for_donor:
            raise ValueError("observed donor has no evaluable section support: %s" % donor)
        donor_section_values[donor] = sections_for_donor
        donor_macro[donor] = float(np.mean(list(sections_for_donor.values())))
    if not donor_macro:
        raise ValueError("no donor/section/type stratum is evaluable")
    return (
        float(np.mean(list(donor_macro.values()))),
        rows,
        donor_macro,
        donor_section_values,
    )


def macro_error_reduction(
    truth: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    donors: np.ndarray,
    labels: np.ndarray,
    minimum_support: int,
) -> Tuple[float, Sequence[Mapping[str, object]], Mapping[str, float]]:
    """Macro-average relative RMSE reduction over donor/type strata."""

    rows = []
    donor_values: Dict[str, list[float]] = {}
    for donor in sorted(set(donors.tolist())):
        for type_index in sorted(set(labels[donors == donor].tolist())):
            selected = (donors == donor) & (labels == type_index)
            support = int(selected.sum())
            if support < minimum_support:
                continue
            baseline_rmse = float(np.sqrt(np.mean(np.square(truth[selected] - baseline[selected]))))
            model_rmse = float(np.sqrt(np.mean(np.square(truth[selected] - prediction[selected]))))
            value = (baseline_rmse - model_rmse) / max(baseline_rmse, 1.0e-12)
            rows.append(
                {
                    "donor_id": donor,
                    "type_index": int(type_index),
                    "support": support,
                    "relative_rmse_reduction": float(value),
                }
            )
            donor_values.setdefault(donor, []).append(float(value))
    donor_macro = {donor: float(np.mean(values)) for donor, values in donor_values.items()}
    if not donor_macro:
        raise ValueError("no supported donor/type error-reduction stratum is evaluable")
    return float(np.mean(list(donor_macro.values()))), rows, donor_macro


def macro_reconstruction_r2(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    baseline: np.ndarray,
    donors: np.ndarray,
    labels: np.ndarray,
    minimum_support: int,
) -> Tuple[float, Sequence[Mapping[str, object]], Mapping[str, float]]:
    """Macro-average representational ceiling over donor/type strata."""

    rows = []
    donor_values: Dict[str, list[float]] = {}
    for donor in sorted(set(donors.tolist())):
        for type_index in sorted(set(labels[donors == donor].tolist())):
            selected = (donors == donor) & (labels == type_index)
            support = int(selected.sum())
            if support < minimum_support:
                continue
            denominator = float(np.square(truth[selected] - baseline[selected]).sum())
            error = float(np.square(truth[selected] - reconstruction[selected]).sum())
            value = 1.0 - error / max(denominator, 1.0e-12)
            rows.append(
                {
                    "donor_id": donor,
                    "type_index": int(type_index),
                    "support": support,
                    "reconstruction_r2": float(value),
                }
            )
            donor_values.setdefault(donor, []).append(float(value))
    donor_macro = {donor: float(np.mean(values)) for donor, values in donor_values.items()}
    if not donor_macro:
        raise ValueError("no supported donor/type reconstruction stratum is evaluable")
    return float(np.mean(list(donor_macro.values()))), rows, donor_macro


def donor_type_coverage(
    donors: np.ndarray, labels: np.ndarray, minimum_support: int, num_types: int
) -> Mapping[str, object]:
    counts = []
    unsupported = []
    donor_values = np.asarray(donors).astype(str)
    label_values = np.asarray(labels, dtype=np.int64)
    for donor in sorted(set(donor_values.tolist())):
        for type_index in range(num_types):
            support = int(np.count_nonzero((donor_values == donor) & (label_values == type_index)))
            counts.append(support)
            if support < minimum_support:
                unsupported.append(
                    {"donor_id": donor, "type_index": int(type_index), "support": support}
                )
    supported = len(counts) - len(unsupported)
    return {
        "planned_donor_type_strata": len(counts),
        "retained_donor_type_strata": supported,
        "unevaluable_donor_type_strata": len(unsupported),
        "supported_fraction": float(supported / max(len(counts), 1)),
        "minimum_support": minimum_support,
        "minimum_observed_support": int(min(counts)) if counts else 0,
        "unsupported_strata": unsupported,
    }


def donor_section_type_coverage(
    donors: np.ndarray,
    sections: np.ndarray,
    labels: np.ndarray,
    minimum_support: int,
    num_types: int,
) -> Mapping[str, object]:
    """Audit every observed donor/section crossed with the frozen fine-type ontology."""

    donor_values = np.asarray(donors).astype(str)
    section_values = np.asarray(sections).astype(str)
    label_values = np.asarray(labels, dtype=np.int64)
    if section_values.shape != donor_values.shape:
        raise ValueError("section coverage identities are not row aligned")
    rows = []
    for donor in sorted(set(donor_values.tolist())):
        for section in sorted(set(section_values[donor_values == donor].tolist())):
            for type_index in range(num_types):
                support = int(
                    np.count_nonzero(
                        (donor_values == donor)
                        & (section_values == section)
                        & (label_values == type_index)
                    )
                )
                rows.append(
                    {
                        "donor_id": donor,
                        "section_id": section,
                        "type_index": type_index,
                        "evaluation_observations": support,
                        "evaluable": support >= minimum_support,
                    }
                )
    retained = sum(bool(row["evaluable"]) for row in rows)
    return {
        "planned_donor_section_type_strata": len(rows),
        "retained_donor_section_type_strata": retained,
        "unevaluable_donor_section_type_strata": len(rows) - retained,
        "retained_fraction": float(retained / max(len(rows), 1)),
        "minimum_support": minimum_support,
        "strata": rows,
    }


def donor_bootstrap(
    values: Mapping[str, float], *, seed: int, iterations: int
) -> Mapping[str, object]:
    if iterations < 100:
        raise ValueError("donor bootstrap requires at least 100 iterations")
    donors = sorted(values)
    if len(donors) < 2:
        raise ValueError("donor bootstrap requires at least two donors")
    array = np.asarray([values[donor] for donor in donors], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(iterations, len(array)))
    draws = array[indices].mean(axis=1)
    return {
        "donors": donors,
        "iterations": iterations,
        "seed": seed,
        "point_estimate": float(array.mean()),
        "ci_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_probability_positive": float(np.mean(draws > 0.0)),
    }


def paired_donor_effects(
    matched: Mapping[str, float], baseline: Mapping[str, float]
) -> Mapping[str, float]:
    donors = sorted(set(matched) & set(baseline))
    if donors != sorted(matched) or donors != sorted(baseline):
        raise ValueError("paired donor effects require identical supported donors")
    return {donor: float(matched[donor] - baseline[donor]) for donor in donors}


def exact_paired_randomization(effects: Mapping[str, float]) -> Mapping[str, object]:
    """Exact one-sided donor sign randomization when enumeration is feasible."""

    donors = sorted(effects)
    if not donors:
        raise ValueError("paired randomization requires donor effects")
    if len(donors) > 20:
        return {
            "available": False,
            "reason": "more than 20 donors makes exact sign enumeration impractical",
            "donors": donors,
        }
    values = np.asarray([effects[donor] for donor in donors], dtype=np.float64)
    observed = float(values.mean())
    null_values = np.asarray(
        [
            np.mean(values * np.asarray(signs, dtype=np.float64))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ],
        dtype=np.float64,
    )
    tolerance = np.finfo(np.float64).eps * max(abs(observed), 1.0) * 8.0
    return {
        "available": True,
        "donors": donors,
        "enumerations": int(len(null_values)),
        "observed_mean_effect": observed,
        "one_sided_p": float(np.mean(null_values >= observed - tolerance)),
    }


def donor_dominance(effects: Mapping[str, float]) -> Mapping[str, object]:
    donors = sorted(effects)
    values = np.asarray([effects[donor] for donor in donors], dtype=np.float64)
    positive = np.maximum(values, 0.0)
    absolute = np.abs(values)
    positive_index = int(np.argmax(positive)) if len(positive) else 0
    absolute_index = int(np.argmax(absolute)) if len(absolute) else 0
    return {
        "largest_positive_donor": donors[positive_index] if donors else None,
        "largest_positive_share": (
            float(positive[positive_index] / positive.sum()) if positive.sum() > 0 else 1.0
        ),
        "largest_absolute_donor": donors[absolute_index] if donors else None,
        "largest_absolute_share": (
            float(absolute[absolute_index] / absolute.sum()) if absolute.sum() > 0 else 1.0
        ),
    }


def leave_one_donor_out(values: Mapping[str, float]) -> Mapping[str, float]:
    donors = sorted(values)
    if len(donors) < 2:
        return {}
    return {
        donor: float(np.mean([values[other] for other in donors if other != donor]))
        for donor in donors
    }


def section_ids_from_blocks(block_ids: np.ndarray) -> Optional[np.ndarray]:
    sections = []
    for value in np.asarray(block_ids).astype(str):
        parts = value.split("/")
        if len(parts) < 3 or not parts[-1].startswith("block_") or not parts[-2].strip():
            return None
        sections.append(parts[-2])
    return np.asarray(sections)


def group_stratification(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    minimum_support: int,
    *,
    group_name: str,
    source: str,
) -> Mapping[str, object]:
    values = np.asarray(groups).astype(str)
    if values.shape != (len(truth),) or any(not value.strip() for value in values.tolist()):
        return {"available": False, "reason": "%s labels are malformed" % group_name}
    try:
        macro, rows, group_values = macro_r2(truth, prediction, values, labels, minimum_support)
    except ValueError as error:
        return {"available": False, "reason": str(error)}
    return {
        "available": True,
        "source": source,
        "%s_equal_type_equal_macro_r2" % group_name: macro,
        "%s_macro_r2" % group_name: group_values,
        "%s_type_rows" % group_name: [
            {
                **{key: value for key, value in row.items() if key != "donor_id"},
                "%s_id" % group_name: row["donor_id"],
            }
            for row in rows
        ],
    }


def within_group_donor_type_r2(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    donors: np.ndarray,
    labels: np.ndarray,
    minimum_support: int,
    *,
    group_name: str,
) -> Mapping[str, object]:
    """Evaluate donor/type macro R2 separately inside each biological group."""

    group_values = np.asarray(groups).astype(str)
    if group_values.shape != (len(truth),):
        raise ValueError("within-group labels are not row aligned")
    reports = {}
    for group in sorted(set(group_values.tolist())):
        selected = group_values == group
        try:
            macro, rows, donor_values = macro_r2(
                truth[selected],
                prediction[selected],
                np.asarray(donors)[selected],
                np.asarray(labels)[selected],
                minimum_support,
            )
        except ValueError as error:
            reports[group] = {"available": False, "reason": str(error)}
            continue
        reports[group] = {
            "available": True,
            "donor_equal_type_equal_macro_r2": macro,
            "donor_macro_r2": donor_values,
            "donor_type_rows": rows,
        }
    available = [
        float(report["donor_equal_type_equal_macro_r2"])
        for report in reports.values()
        if report["available"]
    ]
    return {
        "group_name": group_name,
        "groups": reports,
        "group_equal_macro_r2": float(np.mean(available)) if available else None,
        "available_groups": len(available),
    }


__all__ = [
    "donor_bootstrap",
    "donor_dominance",
    "donor_section_type_macro_r2",
    "donor_section_type_coverage",
    "donor_type_coverage",
    "exact_paired_randomization",
    "group_stratification",
    "leave_one_donor_out",
    "macro_error_reduction",
    "macro_r2",
    "macro_reconstruction_r2",
    "paired_donor_effects",
    "section_ids_from_blocks",
    "within_group_donor_type_r2",
]
