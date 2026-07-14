"""Compact, donor-aware scoring helpers for the retrospective HEST analyses.

The continuous endpoints give every target, donor, section, and cell type equal
weight at the level stated in the metric name.  They intentionally do not pool
rows across biological donors.
"""

from __future__ import annotations

import itertools
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

_EPSILON = 1.0e-12


def _json_float(value: object) -> Optional[float]:
    """Return a plain finite Python float, or ``None`` for JSON safety."""

    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _finite_mean(values: Sequence[float]) -> Optional[float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else None


def _as_target_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("%s must be a row-by-target matrix" % name)
    return array


def _group_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    identities: Sequence[np.ndarray],
    minimum_support: int,
) -> Tuple[Sequence[Tuple[str, ...]], np.ndarray, np.ndarray, np.ndarray]:
    """Compute target-wise values once for every requested identity stratum."""

    row_keys = list(zip(*(values.tolist() for values in identities)))
    keys = sorted(set(row_keys))
    target_count = truth.shape[1]
    support = np.zeros((len(keys), target_count), dtype=np.int64)
    r2 = np.full((len(keys), target_count), np.nan, dtype=np.float64)
    reduction = np.full_like(r2, np.nan)
    finite = np.isfinite(truth) & np.isfinite(prediction) & np.isfinite(reference)

    for group_index, key in enumerate(keys):
        selected = np.ones(len(truth), dtype=bool)
        for values, identity in zip(identities, key):
            selected &= values == identity
        valid = finite[selected]
        count = valid.sum(axis=0)
        support[group_index] = count
        supported = count >= minimum_support
        if not supported.any():
            continue

        y = truth[selected]
        y_hat = prediction[selected]
        y_reference = reference[selected]
        safe_count = np.maximum(count, 1)
        mean = np.where(valid, y, 0.0).sum(axis=0) / safe_count
        centered = np.where(valid, y - mean, 0.0)
        model_difference = np.where(valid, y - y_hat, 0.0)
        reference_difference = np.where(valid, y - y_reference, 0.0)
        centered_sse = np.square(centered).sum(axis=0)
        model_sse = np.square(model_difference).sum(axis=0)
        reference_sse = np.square(reference_difference).sum(axis=0)

        valid_r2 = supported & (centered_sse > _EPSILON)
        valid_reduction = supported & (reference_sse > _EPSILON)
        r2[group_index, valid_r2] = (
            1.0 - model_sse[valid_r2] / centered_sse[valid_r2]
        )
        reduction[group_index, valid_reduction] = (
            1.0 - model_sse[valid_reduction] / reference_sse[valid_reduction]
        )
    return keys, support, r2, reduction


def _donor_type_values(
    keys: Sequence[Tuple[str, ...]], values: np.ndarray
) -> Dict[str, np.ndarray]:
    donors = sorted({key[0] for key in keys})
    result: Dict[str, np.ndarray] = {}
    for donor in donors:
        selected = np.asarray([key[0] == donor for key in keys], dtype=bool)
        with np.errstate(invalid="ignore"):
            counts = np.isfinite(values[selected]).sum(axis=0)
            sums = np.where(np.isfinite(values[selected]), values[selected], 0.0).sum(axis=0)
        result[donor] = np.divide(
            sums,
            counts,
            out=np.full(values.shape[1], np.nan, dtype=np.float64),
            where=counts > 0,
        )
    return result


def _donor_section_type_values(
    keys: Sequence[Tuple[str, ...]], values: np.ndarray
) -> Dict[str, np.ndarray]:
    """Average types within sections, then sections within each donor."""

    result: Dict[str, np.ndarray] = {}
    donors = sorted({key[0] for key in keys})
    for donor in donors:
        sections = sorted({key[1] for key in keys if key[0] == donor})
        section_means = []
        for section in sections:
            selected = np.asarray(
                [key[0] == donor and key[1] == section for key in keys], dtype=bool
            )
            counts = np.isfinite(values[selected]).sum(axis=0)
            sums = np.where(np.isfinite(values[selected]), values[selected], 0.0).sum(axis=0)
            section_means.append(
                np.divide(
                    sums,
                    counts,
                    out=np.full(values.shape[1], np.nan, dtype=np.float64),
                    where=counts > 0,
                )
            )
        stacked = np.asarray(section_means, dtype=np.float64)
        counts = np.isfinite(stacked).sum(axis=0)
        sums = np.where(np.isfinite(stacked), stacked, 0.0).sum(axis=0)
        result[donor] = np.divide(
            sums,
            counts,
            out=np.full(values.shape[1], np.nan, dtype=np.float64),
            where=counts > 0,
        )
    return result


def _macro_by_target(per_donor: Mapping[str, np.ndarray], target_count: int) -> np.ndarray:
    if not per_donor:
        return np.full(target_count, np.nan, dtype=np.float64)
    stacked = np.asarray([per_donor[donor] for donor in sorted(per_donor)])
    counts = np.isfinite(stacked).sum(axis=0)
    sums = np.where(np.isfinite(stacked), stacked, 0.0).sum(axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(target_count, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def score_continuous_targets(
    truth: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    labels: np.ndarray,
    *,
    target_names: Optional[Sequence[str]] = None,
    minimum_support: int = 5,
) -> Mapping[str, object]:
    """Score continuous targets with equal donor/type and section/type weighting.

    ``reference`` is a row-aligned baseline prediction.  Reference error
    reduction is ``1 - SSE(model) / SSE(reference)`` within each stratum.
    Missing or constant-target strata remain visible through evaluable counts,
    but do not enter macro averages.
    """

    y = _as_target_matrix(truth, "truth")
    y_hat = _as_target_matrix(prediction, "prediction")
    y_reference = _as_target_matrix(reference, "reference")
    if y_hat.shape != y.shape or y_reference.shape != y.shape:
        raise ValueError("truth, prediction, and reference must have identical shapes")
    if minimum_support <= 0:
        raise ValueError("minimum_support must be positive")

    donor_values = np.asarray(donors).astype(str)
    section_values = np.asarray(sections).astype(str)
    label_values = np.asarray(labels).astype(str)
    if any(values.shape != (len(y),) for values in (donor_values, section_values, label_values)):
        raise ValueError("donors, sections, and labels must be row aligned")
    if len(y) == 0:
        raise ValueError("continuous scoring requires at least one row")
    if any(not value.strip() for value in donor_values.tolist()):
        raise ValueError("donor identities must be non-empty")
    if any(not value.strip() for value in section_values.tolist()):
        raise ValueError("section identities must be non-empty")
    if any(not value.strip() for value in label_values.tolist()):
        raise ValueError("type labels must be non-empty")

    if target_names is None:
        names = tuple("target_%d" % index for index in range(y.shape[1]))
    else:
        names = tuple(str(name) for name in target_names)
    if len(names) != y.shape[1] or len(set(names)) != len(names) or any(not n for n in names):
        raise ValueError("target_names must uniquely name every target column")

    dt_keys, dt_support, dt_r2, dt_reduction = _group_metrics(
        y,
        y_hat,
        y_reference,
        (donor_values, label_values),
        minimum_support,
    )
    dst_keys, dst_support, dst_r2, dst_reduction = _group_metrics(
        y,
        y_hat,
        y_reference,
        (donor_values, section_values, label_values),
        minimum_support,
    )
    dt_donor_r2 = _donor_type_values(dt_keys, dt_r2)
    dt_donor_reduction = _donor_type_values(dt_keys, dt_reduction)
    dst_donor_r2 = _donor_section_type_values(dst_keys, dst_r2)
    dst_donor_reduction = _donor_section_type_values(dst_keys, dst_reduction)

    metric_arrays = {
        "donor_type_macro_r2": _macro_by_target(dt_donor_r2, y.shape[1]),
        "donor_type_macro_reference_error_reduction": _macro_by_target(
            dt_donor_reduction, y.shape[1]
        ),
        "donor_section_type_macro_r2": _macro_by_target(dst_donor_r2, y.shape[1]),
        "donor_section_type_macro_reference_error_reduction": _macro_by_target(
            dst_donor_reduction, y.shape[1]
        ),
    }
    all_donors = sorted(
        set(dt_donor_r2)
        | set(dt_donor_reduction)
        | set(dst_donor_r2)
        | set(dst_donor_reduction)
    )
    targets: Dict[str, object] = {}
    for target_index, name in enumerate(names):
        targets[name] = {
            **{
                metric: _json_float(values[target_index])
                for metric, values in metric_arrays.items()
            },
            "per_donor": {
                donor: {
                    "donor_type_r2": _json_float(dt_donor_r2[donor][target_index]),
                    "donor_type_reference_error_reduction": _json_float(
                        dt_donor_reduction[donor][target_index]
                    ),
                    "donor_section_type_r2": _json_float(
                        dst_donor_r2[donor][target_index]
                    ),
                    "donor_section_type_reference_error_reduction": _json_float(
                        dst_donor_reduction[donor][target_index]
                    ),
                }
                for donor in all_donors
            },
            "support": {
                "evaluable_donor_type_r2_strata": int(
                    np.isfinite(dt_r2[:, target_index]).sum()
                ),
                "evaluable_donor_type_reference_strata": int(
                    np.isfinite(dt_reduction[:, target_index]).sum()
                ),
                "evaluable_donor_section_type_r2_strata": int(
                    np.isfinite(dst_r2[:, target_index]).sum()
                ),
                "evaluable_donor_section_type_reference_strata": int(
                    np.isfinite(dst_reduction[:, target_index]).sum()
                ),
                "maximum_donor_type_rows": int(dt_support[:, target_index].max(initial=0)),
                "maximum_donor_section_type_rows": int(
                    dst_support[:, target_index].max(initial=0)
                ),
            },
        }

    return {
        "minimum_support": int(minimum_support),
        "rows": int(len(y)),
        "target_count": int(len(names)),
        "targets": targets,
        "target_macro": {
            metric: _finite_mean(values.tolist()) for metric, values in metric_arrays.items()
        },
    }


def multiclass_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    class_labels: Optional[Sequence[str]] = None,
) -> Mapping[str, object]:
    """Return balanced accuracy and macro-F1 without a scikit-learn dependency."""

    y = np.asarray(truth).astype(str)
    y_hat = np.asarray(prediction).astype(str)
    if y.ndim != 1 or y_hat.shape != y.shape or len(y) == 0:
        raise ValueError("classification truth and prediction must be aligned non-empty vectors")
    if class_labels is None:
        classes = tuple(sorted(set(y.tolist()) | set(y_hat.tolist())))
    else:
        classes = tuple(str(label) for label in class_labels)
        if len(classes) != len(set(classes)) or any(not label for label in classes):
            raise ValueError("class_labels must be unique and non-empty")
        unknown = (set(y.tolist()) | set(y_hat.tolist())) - set(classes)
        if unknown:
            raise ValueError(
                "observed labels are absent from class_labels: "
                + ", ".join(sorted(unknown))
            )

    per_class: Dict[str, object] = {}
    recalls = []
    f1_values = []
    for label in classes:
        true_positive = int(np.count_nonzero((y == label) & (y_hat == label)))
        truth_support = int(np.count_nonzero(y == label))
        predicted_support = int(np.count_nonzero(y_hat == label))
        recall = true_positive / truth_support if truth_support else None
        precision = true_positive / predicted_support if predicted_support else None
        if truth_support or predicted_support:
            precision_for_f1 = 0.0 if precision is None else precision
            recall_for_f1 = 0.0 if recall is None else recall
            denominator = precision_for_f1 + recall_for_f1
            f1 = 2.0 * precision_for_f1 * recall_for_f1 / denominator if denominator else 0.0
            f1_values.append(f1)
        else:
            f1 = None
        if recall is not None:
            recalls.append(recall)
        per_class[label] = {
            "truth_support": truth_support,
            "predicted_support": predicted_support,
            "precision": _json_float(precision),
            "recall": _json_float(recall),
            "f1": _json_float(f1),
        }

    return {
        "rows": int(len(y)),
        "class_count": int(len(classes)),
        "accuracy": float(np.mean(y == y_hat)),
        "balanced_accuracy": _finite_mean(recalls),
        "macro_f1": _finite_mean(f1_values),
        "per_class": per_class,
    }


def holm_adjust(p_values: Mapping[str, object]) -> Mapping[str, Optional[float]]:
    """Apply deterministic Holm family-wise adjustment to a named p-value map."""

    parsed: Dict[str, Optional[float]] = {}
    for raw_name, raw_value in p_values.items():
        name = str(raw_name)
        if name in parsed:
            raise ValueError("p-value names must remain unique after string conversion")
        value = _json_float(raw_value)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("p-values must lie in [0, 1]")
        parsed[name] = value

    ordered = sorted(
        ((name, value) for name, value in parsed.items() if value is not None),
        key=lambda item: (item[1], item[0]),
    )
    adjusted: Dict[str, Optional[float]] = {name: None for name in sorted(parsed)}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[name] = float(running)
    return adjusted


def summarize_paired_donor_effects(
    model: Mapping[str, float],
    control: Mapping[str, float],
    *,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 1701,
) -> Mapping[str, object]:
    """Summarize model-minus-control effects over paired biological donors.

    The exact randomization p-value is one-sided for the pre-specified
    alternative that the model improves over its control.
    """

    model_names = {str(name) for name in model}
    control_names = {str(name) for name in control}
    if model_names != control_names:
        raise ValueError("paired donor effects require identical donor identities")
    donors = sorted(model_names)
    if not donors:
        raise ValueError("paired donor effects require at least one donor")
    if len(donors) > 20:
        raise ValueError("exact donor sign-flip inference is limited to 20 donors")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")

    model_by_name = {str(name): float(value) for name, value in model.items()}
    control_by_name = {str(name): float(value) for name, value in control.items()}
    effects = np.asarray(
        [model_by_name[donor] - control_by_name[donor] for donor in donors],
        dtype=np.float64,
    )
    if not np.isfinite(effects).all():
        raise ValueError("paired donor values must be finite")
    observed = float(effects.mean())
    tolerance = np.finfo(np.float64).eps * max(abs(observed), 1.0) * 8.0
    exceedances = 0
    enumerations = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(effects)):
        null_mean = float(np.mean(effects * np.asarray(signs, dtype=np.float64)))
        exceedances += int(null_mean >= observed - tolerance)
        enumerations += 1

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(effects), size=(bootstrap_iterations, len(effects)))
    draws = effects[indices].mean(axis=1)
    return {
        "donor_count": int(len(donors)),
        "donors": donors,
        "per_donor_effect": {
            donor: float(effect) for donor, effect in zip(donors, effects.tolist())
        },
        "mean_effect": observed,
        "positive_fraction": float(np.mean(effects > 0.0)),
        "exact_sign_flip_p": float(exceedances / enumerations),
        "exact_sign_flip_alternative": "greater",
        "exact_sign_flip_enumerations": int(enumerations),
        "bootstrap_ci_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
    }


__all__ = [
    "holm_adjust",
    "multiclass_metrics",
    "score_continuous_targets",
    "summarize_paired_donor_effects",
]
