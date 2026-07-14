"""Leakage-safe split-half measurement utilities for retrospective HEST probes.

The functions in this module are intentionally array based.  They normalize
the two transcript halves independently, form same-donor/section/fine-type
reference residuals without using evaluation cells in a reference mean, and
summarize repeatability at the biological-replicate level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from .reliability import normalize_split_counts, pearson_correlations, spearman_brown

DEFAULT_SUPPORT_THRESHOLDS = (5, 10, 20, 30)


@dataclass(frozen=True)
class HalfReferenceResiduals:
    """Aligned residuals and independently fitted split-half reference means."""

    half_a: np.ndarray
    half_b: np.ndarray
    reference_mean_half_a: np.ndarray
    reference_mean_half_b: np.ndarray
    evaluation_mask: np.ndarray
    strata: tuple[Mapping[str, object], ...]


def _matrix(values: object, name: str, *, finite: bool) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not result.shape[1]:
        raise ValueError(f"{name} must be a non-empty numeric matrix")
    if np.isinf(result).any() or (finite and not np.isfinite(result).all()):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _identifiers(values: object, name: str, rows: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != rows:
        raise ValueError(f"{name} must be a row-aligned identifier vector")
    result = result.astype(str)
    if any(not value.strip() for value in result.tolist()):
        raise ValueError(f"{name} contains an empty identifier")
    return result


def _feature_ids(values: Sequence[str], columns: int) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if (
        len(result) != columns
        or len(set(result)) != len(result)
        or any(not value for value in result)
    ):
        raise ValueError("feature identifiers must be unique, non-empty, and column aligned")
    return result


def _role_masks(pool_roles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lowered = np.char.lower(pool_roles.astype(str))
    reference = np.char.startswith(lowered, "reference")
    evaluation = np.char.startswith(lowered, "evaluation")
    if np.any(reference & evaluation):  # pragma: no cover - impossible for string prefixes
        raise ValueError("a pool role cannot be both reference and evaluation")
    return reference, evaluation


def _stratum_rows(
    donors: np.ndarray,
    sections: np.ndarray,
    fine_types: np.ndarray,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted(set(zip(donors.tolist(), sections.tolist(), fine_types.tolist()))))


def normalize_halves(
    counts_half_a: object,
    counts_half_b: object,
    *,
    library_sizes_half_a: Optional[object] = None,
    library_sizes_half_b: Optional[object] = None,
    scale: float = 10_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return independently library-normalized log1p-CPM transcript halves."""

    first = np.asarray(counts_half_a)
    second = np.asarray(counts_half_b)
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError("split-half count matrices must be aligned")
    return (
        normalize_split_counts(first, library_sizes=library_sizes_half_a, scale=scale),
        normalize_split_counts(second, library_sizes=library_sizes_half_b, scale=scale),
    )


def reference_residualize_halves(
    half_a: object,
    half_b: object,
    donor_ids: object,
    section_ids: object,
    fine_type_ids: object,
    pool_roles: object,
    *,
    minimum_support: int,
) -> HalfReferenceResiduals:
    """Subtract reference-only means from evaluation rows in matched strata.

    A stratum is eligible only when both its reference and evaluation pools
    contain ``minimum_support`` rows.  Half A and half B reference means are
    calculated separately; rows whose role begins with ``evaluation`` are
    never included in either mean.
    """

    first = _matrix(half_a, "half_a", finite=True)
    second = _matrix(half_b, "half_b", finite=True)
    if second.shape != first.shape:
        raise ValueError("split-half features must be aligned")
    if not isinstance(minimum_support, (int, np.integer)) or int(minimum_support) < 1:
        raise ValueError("minimum_support must be a positive integer")
    rows = len(first)
    donors = _identifiers(donor_ids, "donor_ids", rows)
    sections = _identifiers(section_ids, "section_ids", rows)
    fine_types = _identifiers(fine_type_ids, "fine_type_ids", rows)
    roles = _identifiers(pool_roles, "pool_roles", rows)
    reference_role, evaluation_role = _role_masks(roles)

    residual_a = np.full(first.shape, np.nan, dtype=np.float64)
    residual_b = np.full(first.shape, np.nan, dtype=np.float64)
    mean_a = np.full(first.shape, np.nan, dtype=np.float64)
    mean_b = np.full(first.shape, np.nan, dtype=np.float64)
    eligible = np.zeros(rows, dtype=np.bool_)
    reports: list[Mapping[str, object]] = []
    for donor, section, fine_type in _stratum_rows(donors, sections, fine_types):
        stratum = (donors == donor) & (sections == section) & (fine_types == fine_type)
        reference = stratum & reference_role
        evaluation = stratum & evaluation_role
        reference_count = int(reference.sum())
        evaluation_count = int(evaluation.sum())
        supported = bool(
            reference_count >= int(minimum_support)
            and evaluation_count >= int(minimum_support)
        )
        reports.append(
            {
                "donor_id": donor,
                "section_id": section,
                "fine_type_id": fine_type,
                "reference_rows": reference_count,
                "evaluation_rows": evaluation_count,
                "supported": supported,
            }
        )
        if not supported:
            continue
        stratum_mean_a = first[reference].mean(axis=0, dtype=np.float64)
        stratum_mean_b = second[reference].mean(axis=0, dtype=np.float64)
        mean_a[evaluation] = stratum_mean_a
        mean_b[evaluation] = stratum_mean_b
        residual_a[evaluation] = first[evaluation] - stratum_mean_a
        residual_b[evaluation] = second[evaluation] - stratum_mean_b
        eligible[evaluation] = True
    return HalfReferenceResiduals(
        residual_a,
        residual_b,
        mean_a,
        mean_b,
        eligible,
        tuple(reports),
    )


def ordered_program_scores(
    values: object,
    program_names: Sequence[str],
    program_gene_membership: object,
) -> np.ndarray:
    """Average member genes for each boolean program in the supplied order."""

    matrix = _matrix(values, "program input", finite=False)
    names = tuple(str(value) for value in program_names)
    if not names or len(set(names)) != len(names) or any(not value for value in names):
        raise ValueError("program names must be unique and non-empty")
    membership = np.asarray(program_gene_membership)
    if membership.dtype != np.bool_:
        raise ValueError("program_gene_membership must be boolean")
    if membership.shape != (len(names), matrix.shape[1]):
        raise ValueError("program membership must align with programs and genes")
    if np.any(membership.sum(axis=1) == 0):
        raise ValueError("every program must contain at least one gene")
    result = np.empty((len(matrix), len(names)), dtype=np.float64)
    for index in range(len(names)):
        result[:, index] = matrix[:, membership[index]].mean(axis=1, dtype=np.float64)
    return result


def _feature_records(
    first: np.ndarray,
    second: np.ndarray,
    names: tuple[str, ...],
    selected: np.ndarray,
    minimum_rows: int,
) -> dict[str, Mapping[str, object]]:
    records: dict[str, Mapping[str, object]] = {}
    for column, name in enumerate(names):
        finite = selected & np.isfinite(first[:, column]) & np.isfinite(second[:, column])
        rows = int(finite.sum())
        if rows >= minimum_rows:
            raw_value = pearson_correlations(
                first[finite, column],
                second[finite, column],
                minimum_rows=minimum_rows,
            )[0]
            corrected_value = spearman_brown(np.asarray([raw_value]))[0]
        else:
            raw_value = np.nan
            corrected_value = np.nan
        raw = None if not np.isfinite(raw_value) else float(raw_value)
        corrected = None if not np.isfinite(corrected_value) else float(corrected_value)
        records[name] = {
            "finite_rows": rows,
            "evaluable": corrected is not None,
            "raw_split_half_correlation": raw,
            "spearman_brown_reliability": corrected,
        }
    return records


def _aggregate_records(
    records: Sequence[tuple[tuple[str, ...], int, Mapping[str, Mapping[str, object]]]],
    names: tuple[str, ...],
    *,
    identity_name: str,
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    total = len(records)
    for name in names:
        values: list[float] = []
        evaluable_rows = 0
        identities: list[list[str]] = []
        for identity, _rows, report in records:
            value = report[name]["spearman_brown_reliability"]
            if value is None:
                continue
            values.append(float(value))
            evaluable_rows += int(report[name]["finite_rows"])
            identities.append(list(identity))
        result[name] = {
            "median_spearman_brown_reliability": (
                None if not values else float(np.median(values))
            ),
            "mean_spearman_brown_reliability": (
                None if not values else float(np.mean(values))
            ),
            f"evaluable_{identity_name}_ids": identities,
            f"evaluable_{identity_name}_count": int(len(values)),
            f"candidate_{identity_name}_count": int(total),
            f"evaluable_{identity_name}_fraction": (
                float(len(values) / total) if total else 0.0
            ),
            "evaluable_rows": int(evaluable_rows),
        }
    return result


def feature_reliability_report(
    half_a: object,
    half_b: object,
    feature_ids: Sequence[str],
    donor_ids: object,
    fine_type_ids: object,
    *,
    evaluation_mask: Optional[object] = None,
    minimum_rows: int = 3,
) -> Mapping[str, object]:
    """Report pooled, donor/type-macro, and per-type split-half reliability.

    Macro summaries give equal weight to each evaluable biological stratum.
    The median is the primary robust macro statistic; the arithmetic mean is
    also emitted explicitly.  All undefined correlations are represented by
    ``None`` so the returned mapping is valid strict JSON.
    """

    first = _matrix(half_a, "half_a", finite=False)
    second = _matrix(half_b, "half_b", finite=False)
    if second.shape != first.shape:
        raise ValueError("split-half features must be aligned")
    if not isinstance(minimum_rows, (int, np.integer)) or int(minimum_rows) < 2:
        raise ValueError("minimum_rows must be an integer of at least two")
    minimum_rows = int(minimum_rows)
    names = _feature_ids(feature_ids, first.shape[1])
    donors = _identifiers(donor_ids, "donor_ids", len(first))
    fine_types = _identifiers(fine_type_ids, "fine_type_ids", len(first))
    if evaluation_mask is None:
        selected = np.ones(len(first), dtype=np.bool_)
    else:
        selected = np.asarray(evaluation_mask)
        if selected.dtype != np.bool_ or selected.shape != (len(first),):
            raise ValueError("evaluation_mask must be a row-aligned boolean vector")

    overall_features = _feature_records(first, second, names, selected, minimum_rows)
    overall_finite = [
        record["spearman_brown_reliability"]
        for record in overall_features.values()
        if record["spearman_brown_reliability"] is not None
    ]

    donor_type_records = []
    per_type: dict[str, Mapping[str, object]] = {}
    for fine_type in sorted(set(fine_types[selected].tolist())):
        type_selected = selected & (fine_types == fine_type)
        type_records = []
        type_donors = sorted(set(donors[type_selected].tolist()))
        for donor in type_donors:
            stratum_selected = type_selected & (donors == donor)
            records = _feature_records(first, second, names, stratum_selected, minimum_rows)
            entry = ((donor, fine_type), int(stratum_selected.sum()), records)
            donor_type_records.append(entry)
            type_records.append(((donor,), int(stratum_selected.sum()), records))
        pooled = _feature_records(first, second, names, type_selected, minimum_rows)
        per_type[fine_type] = {
            "rows": int(type_selected.sum()),
            "donor_ids": type_donors,
            "pooled_features": pooled,
            "donor_macro_features": _aggregate_records(
                type_records,
                names,
                identity_name="donor",
            ),
        }

    return {
        "rows": int(selected.sum()),
        "minimum_rows": minimum_rows,
        "feature_ids": list(names),
        "overall": {
            "features": overall_features,
            "finite_features": int(len(overall_finite)),
            "median_spearman_brown_reliability": (
                None if not overall_finite else float(np.median(overall_finite))
            ),
        },
        "donor_type_macro": {
            "candidate_strata": int(len(donor_type_records)),
            "features": _aggregate_records(
                donor_type_records,
                names,
                identity_name="donor_type_stratum",
            ),
        },
        "per_type": per_type,
    }


def support_threshold_audit(
    donor_ids: object,
    section_ids: object,
    fine_type_ids: object,
    pool_roles: object,
    *,
    thresholds: Sequence[int] = DEFAULT_SUPPORT_THRESHOLDS,
) -> Mapping[str, object]:
    """Audit matched reference/evaluation support at fixed sensitivity cutoffs."""

    roles_raw = np.asarray(pool_roles)
    if roles_raw.ndim != 1:
        raise ValueError("pool_roles must be a one-dimensional identifier vector")
    rows = len(roles_raw)
    donors = _identifiers(donor_ids, "donor_ids", rows)
    sections = _identifiers(section_ids, "section_ids", rows)
    fine_types = _identifiers(fine_type_ids, "fine_type_ids", rows)
    roles = _identifiers(pool_roles, "pool_roles", rows)
    reference_role, evaluation_role = _role_masks(roles)
    normalized_thresholds = tuple(sorted({int(value) for value in thresholds}))
    if not normalized_thresholds or any(value < 1 for value in normalized_thresholds):
        raise ValueError("support thresholds must contain positive integers")
    if any(int(value) != value for value in thresholds):
        raise ValueError("support thresholds must be integers")

    strata = []
    for donor, section, fine_type in _stratum_rows(donors, sections, fine_types):
        selected = (donors == donor) & (sections == section) & (fine_types == fine_type)
        strata.append(
            (
                donor,
                section,
                fine_type,
                int((selected & reference_role).sum()),
                int((selected & evaluation_role).sum()),
            )
        )
    by_threshold = {}
    for threshold in normalized_thresholds:
        supported = [row for row in strata if row[3] >= threshold and row[4] >= threshold]
        supported_types = sorted({row[2] for row in supported})
        by_threshold[str(threshold)] = {
            "minimum_reference_and_evaluation_rows": int(threshold),
            "supported_strata": int(len(supported)),
            "total_strata": int(len(strata)),
            "supported_strata_fraction": float(len(supported) / len(strata)) if strata else 0.0,
            "supported_reference_rows": int(sum(row[3] for row in supported)),
            "supported_evaluation_rows": int(sum(row[4] for row in supported)),
            "supported_fine_type_ids": supported_types,
            "supported_stratum_ids": [list(row[:3]) for row in supported],
        }
    return {
        "thresholds": list(normalized_thresholds),
        "reference_rows": int(reference_role.sum()),
        "evaluation_rows": int(evaluation_role.sum()),
        "unassigned_rows": int((~reference_role & ~evaluation_role).sum()),
        "total_strata": int(len(strata)),
        "by_threshold": by_threshold,
    }


__all__ = [
    "DEFAULT_SUPPORT_THRESHOLDS",
    "HalfReferenceResiduals",
    "feature_reliability_report",
    "normalize_halves",
    "ordered_program_scores",
    "reference_residualize_halves",
    "support_threshold_audit",
]
