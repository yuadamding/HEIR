"""Paired fixed-denominator registration sensitivity for HEST qualification v2."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

SCHEMA = "heir.hest_registration_sensitivity.v2"
MINIMUM_DONOR_TYPE_SUPPORT = 5
MINIMUM_FORMAL_DONORS = 12
NONINFERIORITY_MARGIN = -0.01
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_721
ONE_SIDED_ALPHA = 0.05
COMPARISON_IDS = ("A_star_vs_N0", "N1_vs_N5", "N3_vs_N5", "N4_vs_N5")


def _group_indices(*values: np.ndarray) -> tuple[tuple[tuple[object, ...], np.ndarray], ...]:
    arrays = tuple(np.asarray(value) for value in values)
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("registration grouping vectors are not row aligned")
    groups: dict[tuple[object, ...], list[int]] = {}
    for row, key in enumerate(zip(*(array.tolist() for array in arrays))):
        groups.setdefault(key, []).append(row)
    return tuple((key, np.asarray(groups[key], dtype=np.int64)) for key in sorted(groups))


def _matrix(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty row-by-coordinate matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _strings(value: object, name: str, rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != rows or array.dtype.kind not in {"S", "U", "i", "u"}:
        raise ValueError(f"{name} must be a row-aligned identifier vector")
    result = array.astype(str)
    if np.any(np.char.str_len(result) == 0):
        raise ValueError(f"{name} cannot contain empty identifiers")
    return result


def _mask(value: object, name: str, rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != (rows,):
        raise ValueError(f"{name} must be a row-aligned boolean mask")
    return array


def paired_standardized_gain(
    truth: object,
    focal_prediction: object,
    comparator_prediction: object,
    *,
    rank: object,
) -> np.ndarray:
    """Return per-cell comparator error minus focal error, divided by rank."""

    observed = _matrix(truth, "truth")
    focal = _matrix(focal_prediction, "focal_prediction")
    comparator = _matrix(comparator_prediction, "comparator_prediction")
    if focal.shape != observed.shape or comparator.shape != observed.shape:
        raise ValueError("truth and both prediction matrices must have identical shapes")
    raw_rank = np.asarray(rank)
    if raw_rank.ndim == 0:
        if raw_rank.dtype.kind not in {"i", "u"}:
            raise ValueError("coordinate rank must be an integer or row-aligned integer vector")
        ranks = np.full(len(observed), int(raw_rank), dtype=np.int64)
    elif raw_rank.shape == (len(observed),) and raw_rank.dtype.kind in {"i", "u"}:
        ranks = raw_rank.astype(np.int64, copy=False)
    else:
        raise ValueError("coordinate rank must be an integer or row-aligned integer vector")
    if np.any(ranks <= 0) or np.any(ranks > observed.shape[1]):
        raise ValueError("coordinate ranks must lie within the padded matrix width")
    active = np.arange(observed.shape[1])[None, :] < ranks[:, None]
    if (
        np.any(observed[~active] != 0.0)
        or np.any(focal[~active] != 0.0)
        or np.any(comparator[~active] != 0.0)
    ):
        raise ValueError("inactive padded coordinates must be exactly zero")
    comparator_error = np.where(active, np.square(observed - comparator), 0.0).sum(axis=1)
    focal_error = np.where(active, np.square(observed - focal), 0.0).sum(axis=1)
    result = (comparator_error - focal_error) / ranks
    if not np.isfinite(result).all():
        raise ValueError("paired standardized gains are non-finite")
    return result


def _bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
    alpha: float,
) -> Mapping[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("donor bootstrap requires finite donor effects")
    if resamples < 1_000 or not 0.0 < alpha < 0.5:
        raise ValueError("donor bootstrap settings are invalid")
    if len(array) == 1:
        lower = upper = float(array[0])
    else:
        generator = np.random.default_rng(seed)
        means = np.empty(resamples, dtype=np.float64)
        chunk = 2_000
        for start in range(0, resamples, chunk):
            end = min(start + chunk, resamples)
            indices = generator.integers(0, len(array), size=(end - start, len(array)))
            means[start:end] = array[indices].mean(axis=1)
        lower = float(np.quantile(means, alpha, method="linear"))
        upper = float(np.quantile(means, 1.0 - alpha, method="linear"))
    return {
        "method": "equal_donor_nonparametric_bootstrap",
        "resamples": resamples,
        "seed": seed,
        "one_sided_alpha": alpha,
        "one_sided_lower_bound": lower,
        "one_sided_upper_bound": upper,
    }


def hierarchical_gain_summary(
    gains: object,
    donor_ids: object,
    type_labels: object,
    selection_mask: object,
    *,
    minimum_support: int = MINIMUM_DONOR_TYPE_SUPPORT,
    allowed_donors: Optional[Sequence[str]] = None,
    required_strata: Optional[Sequence[tuple[str, str]]] = None,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> Mapping[str, object]:
    """Average cells within type, types within donor, then donors equally."""

    values = np.asarray(gains, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("gains must be a nonempty finite vector")
    donors = _strings(donor_ids, "donor_ids", len(values))
    labels = _strings(type_labels, "type_labels", len(values))
    selected = _mask(selection_mask, "selection_mask", len(values)).copy()
    if isinstance(minimum_support, bool) or minimum_support < 1:
        raise ValueError("minimum donor/type support must be positive")
    if allowed_donors is not None:
        allowed = tuple(str(value) for value in allowed_donors)
        if not allowed or len(set(allowed)) != len(allowed) or any(not value for value in allowed):
            raise ValueError("allowed_donors must be unique nonempty identifiers")
        selected &= np.isin(donors, allowed)

    selected_groups = {
        (str(donor), str(label)): indices[selected[indices]]
        for (donor, label), indices in _group_indices(donors, labels)
        if np.any(selected[indices])
    }
    observed_keys = sorted(selected_groups)
    counts = {key: len(selected_groups[key]) for key in observed_keys}
    if required_strata is None:
        supported = tuple(key for key in observed_keys if counts[key] >= minimum_support)
    else:
        supported = tuple((str(donor), str(label)) for donor, label in required_strata)
        if len(set(supported)) != len(supported):
            raise ValueError("required donor/type strata must be unique")
        missing = [key for key in supported if counts.get(key, 0) < minimum_support]
        if missing:
            raise ValueError("required donor/type strata lack the frozen support")

    type_rows = []
    donor_type_effects: dict[str, list[float]] = {}
    supported_rows = 0
    for donor, label in supported:
        indices = selected_groups[(donor, label)]
        count = len(indices)
        effect = float(values[indices].mean())
        supported_rows += count
        donor_type_effects.setdefault(donor, []).append(effect)
        type_rows.append(
            {
                "donor_id": donor,
                "type_label": label,
                "rows": count,
                "mean_paired_gain": effect,
            }
        )
    donor_effects = {
        donor: float(np.mean(type_effects))
        for donor, type_effects in sorted(donor_type_effects.items())
    }
    estimate = float(np.mean(list(donor_effects.values()))) if donor_effects else None
    uncertainty = (
        _bootstrap_mean(
            list(donor_effects.values()),
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            alpha=ONE_SIDED_ALPHA,
        )
        if donor_effects
        else None
    )
    return {
        "aggregation": "equal_cells_within_type_then_equal_types_within_donor_then_equal_donors",
        "minimum_donor_type_support": minimum_support,
        "rows_in_requested_mask": int(selected.sum()),
        "rows_in_supported_strata": supported_rows,
        "observed_donor_type_strata": len(observed_keys),
        "supported_donor_type_strata": len(supported),
        "supported_donors": len(donor_effects),
        "supported_strata_keys": [list(key) for key in supported],
        "donor_type_effects": type_rows,
        "donor_effects": donor_effects,
        "mean_paired_gain": estimate,
        "uncertainty": uncertainty,
        "evaluable": bool(donor_effects),
    }


def evaluate_registration_noninferiority(
    truth: object,
    focal_prediction: object,
    comparator_prediction: object,
    donor_ids: object,
    type_labels: object,
    *,
    rank: object,
    quality_mask: object,
    all_rows_mask: object,
    best_mask: object,
    included_donor_ids: Sequence[str],
    comparison_id: str,
    selected_quality_band_id: str = "Q3",
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> Mapping[str, object]:
    """Evaluate Q3-versus-all noninferiority while holding donor/type composition fixed."""

    gain = paired_standardized_gain(
        truth,
        focal_prediction,
        comparator_prediction,
        rank=rank,
    )
    rows = len(gain)
    quality = _mask(quality_mask, "quality_mask", rows)
    all_rows = _mask(all_rows_mask, "all_rows_mask", rows)
    best = _mask(best_mask, "best_mask", rows)
    if np.any(quality & ~all_rows) or np.any(best & ~quality):
        raise ValueError("best must be nested in quality, which must be nested in all rows")
    if comparison_id not in COMPARISON_IDS:
        raise ValueError("registration comparison is not frozen in the v2 protocol")
    if selected_quality_band_id != "Q3":
        raise ValueError("v2 formal registration sensitivity requires frozen band Q3")

    selected = hierarchical_gain_summary(
        gain,
        donor_ids,
        type_labels,
        quality,
        allowed_donors=included_donor_ids,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    supported_keys = [tuple(value) for value in selected["supported_strata_keys"]]
    all_matched = hierarchical_gain_summary(
        gain,
        donor_ids,
        type_labels,
        all_rows,
        allowed_donors=included_donor_ids,
        required_strata=supported_keys,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 1,
    )
    selected_donors = selected["donor_effects"]
    all_donors = all_matched["donor_effects"]
    if not isinstance(selected_donors, Mapping) or not isinstance(all_donors, Mapping):
        raise AssertionError("hierarchical summaries must expose donor effects")
    if set(selected_donors) != set(all_donors):
        raise ValueError("quality and all-row summaries have different supported donors")
    donor_differences = {
        donor: float(selected_donors[donor]) - float(all_donors[donor])
        for donor in sorted(selected_donors)
    }
    if donor_differences:
        difference = float(np.mean(list(donor_differences.values())))
        uncertainty = _bootstrap_mean(
            list(donor_differences.values()),
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + 2,
            alpha=ONE_SIDED_ALPHA,
        )
        lower = float(uncertainty["one_sided_lower_bound"])
    else:
        difference = None
        uncertainty = None
        lower = None
    donor_support_pass = len(donor_differences) >= MINIMUM_FORMAL_DONORS
    noninferiority_pass = bool(
        donor_support_pass
        and difference is not None
        and lower is not None
        and difference >= NONINFERIORITY_MARGIN
        and lower >= NONINFERIORITY_MARGIN
    )
    absolute_best = hierarchical_gain_summary(
        gain,
        donor_ids,
        type_labels,
        best,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 3,
    )
    return {
        "schema": SCHEMA,
        "comparison_id": comparison_id,
        "gain_definition": ("(squared_error_comparator_minus_squared_error_focal)/coordinate_rank"),
        "coordinate_rank": (
            int(np.asarray(rank))
            if np.asarray(rank).ndim == 0
            else {
                "minimum": int(np.min(np.asarray(rank))),
                "maximum": int(np.max(np.asarray(rank))),
                "varies_by_outer_donor": bool(len(set(np.asarray(rank).tolist())) > 1),
            }
        ),
        "selected_quality_band_id": selected_quality_band_id,
        "selected_quality": selected,
        "all_rows_same_supported_donor_type_strata": all_matched,
        "composition_rule": "all_rows_reuse_the_selected_quality_supported_donor_type_strata",
        "noninferiority": {
            "contrast": "selected_quality_mean_gain_minus_all_rows_mean_gain",
            "margin": NONINFERIORITY_MARGIN,
            "minimum_formal_donors": MINIMUM_FORMAL_DONORS,
            "supported_donors": len(donor_differences),
            "donor_support_pass": donor_support_pass,
            "donor_differences": donor_differences,
            "estimate": difference,
            "uncertainty": uncertainty,
            "one_sided_lower_bound": lower,
            "passed": noninferiority_pass,
        },
        "absolute_best_descriptive": {
            **absolute_best,
            "formal_primary_band": selected_quality_band_id == "Q1",
            "selection_reason": (
                "selected_by_outcome_blind_topology_rule"
                if selected_quality_band_id == "Q1"
                else "reported_descriptively_not_selected_by_outcome_blind_topology_rule"
            ),
        },
    }


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "COMPARISON_IDS",
    "MINIMUM_DONOR_TYPE_SUPPORT",
    "MINIMUM_FORMAL_DONORS",
    "NONINFERIORITY_MARGIN",
    "ONE_SIDED_ALPHA",
    "SCHEMA",
    "evaluate_registration_noninferiority",
    "hierarchical_gain_summary",
    "paired_standardized_gain",
]
