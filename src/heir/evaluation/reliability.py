"""Deterministic transcript split-halves and molecular reliability estimates.

The functions in this module deliberately operate on arrays rather than a
repository-specific artifact class.  A registered-observation loader can bind
its verified fields to these functions without weakening identity checks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

SPLIT_HALF_METHOD = "sha256-final-byte-lsb-v1"


def _identifiers(values: object, name: str, *, unique: bool = False) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError("%s must be a one-dimensional identifier array" % name)
    result = result.astype(str)
    if any(not value.strip() for value in result.tolist()):
        raise ValueError("%s contains an empty identifier" % name)
    if unique and len(set(result.tolist())) != len(result):
        raise ValueError("%s must contain unique identifiers" % name)
    return result


def deterministic_transcript_halves(
    transcript_ids: object,
    *,
    salt: str,
) -> np.ndarray:
    """Assign each unique transcript identity to half 0 or 1.

    SHA-256 makes the assignment stable across processes, input ordering,
    platforms, and Python hash seeds.  Half B is the low bit of the final
    digest byte.  The salt is part of the frozen study contract and prevents
    accidental reuse of a split from another study.
    """

    identifiers = _identifiers(transcript_ids, "transcript_ids", unique=True)
    if not isinstance(salt, str) or not salt:
        raise ValueError("split-half salt must be a non-empty string")
    prefix = (salt + "\x00").encode("utf-8")
    return np.fromiter(
        (
            hashlib.sha256(prefix + value.encode("utf-8")).digest()[-1] & 1
            for value in identifiers.tolist()
        ),
        dtype=np.uint8,
        count=len(identifiers),
    )


@dataclass(frozen=True)
class SplitHalfCounts:
    """Two independent count matrices sharing a frozen row/gene order."""

    half_a: np.ndarray
    half_b: np.ndarray
    observation_ids: np.ndarray
    gene_ids: np.ndarray
    assignments: np.ndarray


def construct_split_half_counts(
    transcript_ids: object,
    transcript_observation_ids: object,
    transcript_gene_ids: object,
    ordered_observation_ids: object,
    ordered_gene_ids: object,
    *,
    salt: str,
) -> SplitHalfCounts:
    """Construct dense split-half counts after exact identity validation."""

    transcripts = _identifiers(transcript_ids, "transcript_ids", unique=True)
    transcript_rows = _identifiers(
        transcript_observation_ids, "transcript_observation_ids"
    )
    transcript_genes = _identifiers(transcript_gene_ids, "transcript_gene_ids")
    observations = _identifiers(
        ordered_observation_ids, "ordered_observation_ids", unique=True
    )
    genes = _identifiers(ordered_gene_ids, "ordered_gene_ids", unique=True)
    if not (len(transcripts) == len(transcript_rows) == len(transcript_genes)):
        raise ValueError("transcript identity arrays must have the same length")

    row_lookup = {value: index for index, value in enumerate(observations.tolist())}
    gene_lookup = {value: index for index, value in enumerate(genes.tolist())}
    unknown_rows = sorted(set(transcript_rows.tolist()) - set(row_lookup))
    unknown_genes = sorted(set(transcript_genes.tolist()) - set(gene_lookup))
    if unknown_rows:
        raise ValueError("transcripts refer to unknown observations")
    if unknown_genes:
        raise ValueError("transcripts contain genes outside the frozen panel")

    row_indices = np.fromiter(
        (row_lookup[value] for value in transcript_rows.tolist()),
        dtype=np.int64,
        count=len(transcripts),
    )
    gene_indices = np.fromiter(
        (gene_lookup[value] for value in transcript_genes.tolist()),
        dtype=np.int64,
        count=len(transcripts),
    )
    assignments = deterministic_transcript_halves(transcripts, salt=salt)
    half_a = np.zeros((len(observations), len(genes)), dtype=np.uint32)
    half_b = np.zeros_like(half_a)
    first = assignments == 0
    np.add.at(half_a, (row_indices[first], gene_indices[first]), 1)
    np.add.at(half_b, (row_indices[~first], gene_indices[~first]), 1)
    return SplitHalfCounts(half_a, half_b, observations, genes, assignments)


def normalize_split_counts(
    counts: object,
    *,
    library_sizes: Optional[object] = None,
    scale: float = 10_000.0,
) -> np.ndarray:
    """Library-size normalize and log-transform one transcript half."""

    values = np.asarray(counts)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("split-half counts must be a numeric matrix")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("split-half counts must be finite and non-negative")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("normalization scale must be positive")
    result = values.astype(np.float32, copy=True)
    target_totals = result.sum(axis=1, dtype=np.float64)
    if library_sizes is None:
        totals = target_totals
    else:
        totals = np.asarray(library_sizes, dtype=np.float64)
        if (
            totals.shape != (len(result),)
            or not np.isfinite(totals).all()
            or np.any(totals < target_totals)
        ):
            raise ValueError("split-half library sizes are malformed or below target counts")
    nonzero = totals > 0
    if np.any(nonzero):
        result[nonzero] *= (scale / totals[nonzero]).astype(np.float32)[:, None]
    np.log1p(result, out=result)
    return result


def pearson_correlations(
    first: object,
    second: object,
    *,
    minimum_rows: int = 3,
) -> np.ndarray:
    """Column-wise Pearson correlations without materializing centered matrices."""

    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("reliability inputs must be aligned finite matrices")
    if minimum_rows < 2:
        raise ValueError("minimum_rows must be at least two")
    result = np.full(x.shape[1], np.nan, dtype=np.float64)
    if len(x) < minimum_rows:
        return result
    count = float(len(x))
    sum_x = x.sum(axis=0)
    sum_y = y.sum(axis=0)
    covariance = np.einsum("ij,ij->j", x, y) - (sum_x * sum_y / count)
    variance_x = np.einsum("ij,ij->j", x, x) - (sum_x * sum_x / count)
    variance_y = np.einsum("ij,ij->j", y, y) - (sum_y * sum_y / count)
    denominator = np.sqrt(np.maximum(variance_x, 0.0) * np.maximum(variance_y, 0.0))
    valid = denominator > np.finfo(np.float64).eps
    result[valid] = np.clip(covariance[valid] / denominator[valid], -1.0, 1.0)
    return result


def spearman_brown(raw_correlations: object) -> np.ndarray:
    """Apply the Spearman--Brown prophecy formula to split-half correlations.

    Negative correlations represent no usable repeatability and are reported as
    zero rather than producing an unbounded negative corrected value.
    Undefined input correlations remain undefined.
    """

    raw = np.asarray(raw_correlations, dtype=np.float64)
    corrected = np.full(raw.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(raw)
    nonnegative = finite & (raw > 0)
    corrected[finite & ~nonnegative] = 0.0
    corrected[nonnegative] = 2.0 * raw[nonnegative] / (1.0 + raw[nonnegative])
    corrected[finite] = np.clip(corrected[finite], 0.0, 1.0)
    return corrected


def feature_reliability(
    half_a: object,
    half_b: object,
    feature_ids: Sequence[str],
    *,
    minimum_rows: int,
) -> Mapping[str, object]:
    """Return raw and corrected split-half reliability for named features."""

    first = np.asarray(half_a)
    second = np.asarray(half_b)
    names = tuple(str(value) for value in feature_ids)
    if first.ndim != 2 or second.shape != first.shape or first.shape[1] != len(names):
        raise ValueError("feature reliability inputs are not aligned")
    if len(set(names)) != len(names) or any(not value for value in names):
        raise ValueError("feature reliability identifiers must be unique and non-empty")
    raw = pearson_correlations(first, second, minimum_rows=minimum_rows)
    corrected = spearman_brown(raw)
    finite = corrected[np.isfinite(corrected)]
    return {
        "rows": int(len(first)),
        "features": {
            name: {
                "raw_split_half_correlation": None if not np.isfinite(value) else float(value),
                "spearman_brown_reliability": (
                    None if not np.isfinite(score) else float(score)
                ),
            }
            for name, value, score in zip(names, raw.tolist(), corrected.tolist())
        },
        "finite_features": int(len(finite)),
        "median_spearman_brown_reliability": (
            None if not len(finite) else float(np.median(finite))
        ),
    }


def program_scores(
    normalized_counts: object,
    gene_ids: Sequence[str],
    programs: Mapping[str, object],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Calculate frozen weighted program scores in deterministic name order.

    A program may be a sequence of gene IDs (equal weights) or a mapping from
    gene ID to a numeric weight.  Missing genes and zero-weight programs are
    rejected so a report cannot silently change its molecular definition.
    """

    values = np.asarray(normalized_counts, dtype=np.float64)
    genes = tuple(str(value) for value in gene_ids)
    if values.ndim != 2 or values.shape[1] != len(genes):
        raise ValueError("program-score counts differ from the ordered gene panel")
    if not programs:
        raise ValueError("at least one frozen molecular program is required")
    lookup = {value: index for index, value in enumerate(genes)}
    names = tuple(sorted(str(value) for value in programs))
    weights = np.zeros((len(genes), len(names)), dtype=np.float64)
    for column, name in enumerate(names):
        definition = programs[name]
        if isinstance(definition, Mapping):
            items = [(str(gene), float(weight)) for gene, weight in definition.items()]
        else:
            if isinstance(definition, (str, bytes)):
                raise ValueError("molecular program definitions must contain gene collections")
            members = tuple(str(gene) for gene in definition)  # type: ignore[arg-type]
            items = [(gene, 1.0 / len(members)) for gene in members] if members else []
        if not items or any(gene not in lookup for gene, _ in items):
            raise ValueError("molecular program %s contains no valid frozen genes" % name)
        for gene, weight in items:
            if not np.isfinite(weight):
                raise ValueError("molecular program weights must be finite")
            weights[lookup[gene], column] += weight
        norm = np.linalg.norm(weights[:, column])
        if norm <= 0:
            raise ValueError("molecular program %s has zero total weight" % name)
        weights[:, column] /= norm
    return values @ weights, names


def program_reliability(
    half_a: object,
    half_b: object,
    gene_ids: Sequence[str],
    programs: Mapping[str, object],
    *,
    minimum_rows: int,
) -> Mapping[str, object]:
    first, names = program_scores(half_a, gene_ids, programs)
    second, second_names = program_scores(half_b, gene_ids, programs)
    if names != second_names:  # pragma: no cover - protected by deterministic sorting
        raise RuntimeError("program identity changed during reliability calculation")
    return feature_reliability(first, second, names, minimum_rows=minimum_rows)


def fit_target_basis(
    normalized_total_counts: object,
    *,
    rank: int,
    sample_weights: Optional[object] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic, optionally donor-balanced target covariance basis."""

    values = np.asarray(normalized_total_counts, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("target-basis fit requires at least two finite rows")
    maximum_rank = min(values.shape[1], len(values) - 1)
    if rank < 1 or rank > maximum_rank:
        raise ValueError("target-basis rank exceeds the available development data")
    if sample_weights is None:
        weights = np.full(len(values), 1.0 / len(values), dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if (
            weights.shape != (len(values),)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError("target-basis sample weights are malformed")
        weights = weights / weights.sum()
    mean = weights @ values
    gram = (values * weights[:, None]).T @ values - np.outer(mean, mean)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues, kind="stable")[::-1][:rank]
    basis = eigenvectors[:, order]
    # Eigenvector signs are arbitrary.  Fix each sign using its largest loading.
    for column in range(basis.shape[1]):
        anchor = int(np.argmax(np.abs(basis[:, column])))
        if basis[anchor, column] < 0:
            basis[:, column] *= -1.0
    return mean, basis


def target_basis_reliability_ceiling(
    half_a: object,
    half_b: object,
    *,
    development_mask: object,
    rank: int,
    minimum_rows: int,
    basis: Optional[object] = None,
    fit_weights: Optional[object] = None,
    group_labels: Optional[object] = None,
    full_targets: Optional[object] = None,
) -> Mapping[str, object]:
    """Estimate the repeatability ceiling for the frozen molecular target basis."""

    first = np.asarray(half_a, dtype=np.float64)
    second = np.asarray(half_b, dtype=np.float64)
    mask = np.asarray(development_mask)
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError("target-basis halves must be aligned matrices")
    if mask.dtype != np.bool_ or mask.shape != (len(first),):
        raise ValueError("development_mask must be a boolean row mask")
    if int(mask.sum()) < max(minimum_rows, 2):
        raise ValueError("too few development rows for target-basis reliability")
    if full_targets is None:
        # This fallback is useful for raw transcript tables that do not carry a
        # separately frozen normalized target.  Confirmatory artifacts should
        # supply the exact full target used downstream.
        total = 0.5 * (first + second)
        fit_source = "mean_of_normalized_halves"
    else:
        total = np.asarray(full_targets, dtype=np.float64)
        if total.shape != first.shape or not np.isfinite(total).all():
            raise ValueError("full molecular targets must align with split halves")
        fit_source = "frozen_full_molecular_target"
    selected_weights = None
    if fit_weights is not None:
        weights = np.asarray(fit_weights, dtype=np.float64)
        if weights.shape != (len(first),):
            raise ValueError("target-basis fit weights must align with all observations")
        selected_weights = weights[mask]
    mean, fitted = fit_target_basis(
        total[mask], rank=rank, sample_weights=selected_weights
    )
    if basis is not None:
        fitted = np.asarray(basis, dtype=np.float64)
        if fitted.ndim != 2 or fitted.shape[0] != first.shape[1] or not fitted.shape[1]:
            raise ValueError("provided target basis is malformed")
        if not np.isfinite(fitted).all():
            raise ValueError("provided target basis is non-finite")
        rank = fitted.shape[1]
    first_scores = (first[mask] - mean) @ fitted
    second_scores = (second[mask] - mean) @ fitted
    result = feature_reliability(
        first_scores,
        second_scores,
        tuple("basis_%03d" % (index + 1) for index in range(rank)),
        minimum_rows=minimum_rows,
    )
    grouped = {}
    macro_components = {}
    if group_labels is not None:
        labels = _identifiers(group_labels, "target_basis_group_labels")
        if labels.shape != (len(first),):
            raise ValueError("target-basis group labels must align with all observations")
        selected_labels = labels[mask]
        for group in sorted(set(selected_labels.tolist())):
            selected = selected_labels == group
            grouped[group] = feature_reliability(
                first_scores[selected],
                second_scores[selected],
                tuple("basis_%03d" % (index + 1) for index in range(rank)),
                minimum_rows=minimum_rows,
            )
        for name in result["features"]:
            values = [
                report["features"][name]["spearman_brown_reliability"]
                for report in grouped.values()
                if report["features"][name]["spearman_brown_reliability"] is not None
            ]
            macro_components[name] = None if not values else float(np.median(values))
    finite_macro = [value for value in macro_components.values() if value is not None]
    return {
        **result,
        "rank": int(rank),
        "fit_partition": "development_only",
        "fit_weighting": "supplied_balanced_weights" if fit_weights is not None else "uniform_rows",
        "fit_source": fit_source,
        "reliability_by_group": grouped,
        "group_macro_component_reliability": macro_components,
        "median_group_macro_reliability": (
            None if not finite_macro else float(np.median(finite_macro))
        ),
    }


__all__ = [
    "SPLIT_HALF_METHOD",
    "SplitHalfCounts",
    "construct_split_half_counts",
    "deterministic_transcript_halves",
    "feature_reliability",
    "fit_target_basis",
    "normalize_split_counts",
    "pearson_correlations",
    "program_reliability",
    "program_scores",
    "spearman_brown",
    "target_basis_reliability_ceiling",
]
