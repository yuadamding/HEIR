"""Reproducible donor-level aggregation for multi-cohort benchmarks.

The benchmark layer intentionally accepts one metric value per biological donor.
It never resamples cells, spots, or image patches as if they were independent
replicates.  Unavailable endpoints remain explicit rows instead of disappearing
from the report.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


class BenchmarkStatus(str, Enum):
    """Availability state for a donor metric or derived benchmark result."""

    OK = "ok"
    MISSING = "missing"
    DATA_LIMITED = "data_limited"


@dataclass(frozen=True)
class DonorMetricRow:
    """A single prespecified endpoint for one donor and one method.

    ``value`` is required only for ``ok`` rows.  Missing and data-limited rows
    deliberately carry no numeric value so they cannot enter an aggregate by
    accident.
    """

    cohort_id: str
    donor_id: str
    method: str
    metric: str
    value: Optional[float]
    status: BenchmarkStatus = BenchmarkStatus.OK
    reason: str = ""
    n_observations: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("cohort_id", "donor_id", "method", "metric"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        try:
            status = BenchmarkStatus(self.status)
        except ValueError as error:
            raise ValueError(f"unsupported benchmark status: {self.status!r}") from error
        object.__setattr__(self, "status", status)
        if status is BenchmarkStatus.OK:
            if self.value is None or not np.isfinite(float(self.value)):
                raise ValueError("ok donor metrics require a finite value")
            object.__setattr__(self, "value", float(self.value))
        elif self.value is not None:
            raise ValueError("missing and data-limited donor metrics cannot carry a value")
        if self.n_observations is not None:
            if (
                isinstance(self.n_observations, bool)
                or int(self.n_observations) != self.n_observations
            ):
                raise ValueError("n_observations must be a positive integer when provided")
            if int(self.n_observations) <= 0:
                raise ValueError("n_observations must be a positive integer when provided")
            object.__setattr__(self, "n_observations", int(self.n_observations))


@dataclass(frozen=True)
class BenchmarkSummaryRow:
    """Donor-bootstrap summary for one cohort, method, and metric."""

    cohort_id: str
    method: str
    metric: str
    estimate: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    n_donors: int
    n_missing: int
    n_data_limited: int
    status: BenchmarkStatus
    reason: str = ""


@dataclass(frozen=True)
class MethodComparisonRow:
    """Paired donor comparison against a prespecified baseline.

    ``mean_difference`` is always ``method - baseline``.  Confidence intervals
    and ``probability_better`` refer to an oriented improvement, which is the
    raw difference for higher-is-better metrics and its negative for
    lower-is-better metrics.
    """

    cohort_id: str
    method: str
    baseline_method: str
    metric: str
    higher_is_better: bool
    mean_difference: Optional[float]
    mean_improvement: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    probability_better: Optional[float]
    n_paired_donors: int
    n_method_donors: int
    n_baseline_donors: int
    status: BenchmarkStatus
    reason: str = ""


BenchmarkRecord = Union[DonorMetricRow, BenchmarkSummaryRow, MethodComparisonRow]


@dataclass(frozen=True)
class BenchmarkReport:
    """Serializable result of a donor-aware multi-cohort benchmark."""

    donor_metrics: Tuple[DonorMetricRow, ...]
    summaries: Tuple[BenchmarkSummaryRow, ...]
    comparisons: Tuple[MethodComparisonRow, ...]
    confidence: float
    iterations: int
    minimum_donors: int
    seed: int
    schema_version: str = "heir-benchmark-v1"

    def to_dict(self) -> Dict[str, object]:
        """Return a stable, JSON-safe representation without NaN sentinels."""

        return {
            "schema_version": self.schema_version,
            "settings": {
                "confidence": self.confidence,
                "iterations": self.iterations,
                "minimum_donors": self.minimum_donors,
                "seed": self.seed,
            },
            "donor_metrics": [_record_dict(row) for row in self.donor_metrics],
            "summaries": [_record_dict(row) for row in self.summaries],
            "comparisons": [_record_dict(row) for row in self.comparisons],
        }


def build_benchmark_report(
    rows: Iterable[DonorMetricRow],
    *,
    baseline_method: Optional[str] = None,
    metric_directions: Optional[Mapping[str, bool]] = None,
    iterations: int = 10000,
    confidence: float = 0.95,
    minimum_donors: int = 2,
    seed: int = 17,
) -> BenchmarkReport:
    """Aggregate donor rows and optionally compare every method with a baseline.

    Parameters
    ----------
    rows:
        One row per ``(cohort, donor, method, metric)``.  Duplicate keys are
        rejected rather than silently averaged.
    baseline_method:
        If provided, create paired comparisons for every other method observed
        in each cohort/metric pair.  Baseline rows may be explicitly missing.
    metric_directions:
        Required for each compared metric.  ``True`` means higher is better;
        ``False`` means lower is better.
    """

    _validate_settings(iterations, confidence, minimum_donors)
    donor_rows = tuple(sorted(tuple(rows), key=_donor_sort_key))
    _validate_unique_rows(donor_rows)
    summaries = _summarize_rows(
        donor_rows,
        iterations=iterations,
        confidence=confidence,
        minimum_donors=minimum_donors,
        seed=seed,
    )
    comparisons: Tuple[MethodComparisonRow, ...] = ()
    if baseline_method is not None:
        if not baseline_method.strip():
            raise ValueError("baseline_method cannot be empty")
        comparisons = _compare_rows(
            donor_rows,
            baseline_method=baseline_method,
            metric_directions={} if metric_directions is None else metric_directions,
            iterations=iterations,
            confidence=confidence,
            minimum_donors=minimum_donors,
            seed=seed,
        )
    return BenchmarkReport(
        donor_metrics=donor_rows,
        summaries=summaries,
        comparisons=comparisons,
        confidence=float(confidence),
        iterations=int(iterations),
        minimum_donors=int(minimum_donors),
        seed=int(seed),
    )


def write_benchmark_json(report: BenchmarkReport, path: Path) -> Path:
    """Write a stable JSON report and reject non-standard NaN output."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return output


def write_benchmark_tsv(report: BenchmarkReport, path: Path) -> Path:
    """Write donor, summary, and comparison records to one normalized TSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    records.extend(_typed_records("donor_metric", report.donor_metrics))
    records.extend(_typed_records("summary", report.summaries))
    records.extend(_typed_records("comparison", report.comparisons))
    fields = [
        "record_type",
        "cohort_id",
        "donor_id",
        "method",
        "baseline_method",
        "metric",
        "value",
        "estimate",
        "mean_difference",
        "mean_improvement",
        "ci_lower",
        "ci_upper",
        "probability_better",
        "higher_is_better",
        "n_observations",
        "n_donors",
        "n_paired_donors",
        "n_method_donors",
        "n_baseline_donors",
        "n_missing",
        "n_data_limited",
        "status",
        "reason",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: _tsv_value(record.get(field)) for field in fields})
    return output


def _validate_settings(iterations: int, confidence: float, minimum_donors: int) -> None:
    if isinstance(iterations, bool) or int(iterations) != iterations or int(iterations) <= 0:
        raise ValueError("iterations must be a positive integer")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if (
        isinstance(minimum_donors, bool)
        or int(minimum_donors) != minimum_donors
        or int(minimum_donors) < 2
    ):
        raise ValueError("minimum_donors must be an integer of at least two")


def _validate_unique_rows(rows: Sequence[DonorMetricRow]) -> None:
    seen = set()
    for row in rows:
        key = (row.cohort_id, row.donor_id, row.method, row.metric)
        if key in seen:
            raise ValueError(
                "duplicate donor metric row for cohort/donor/method/metric: " + "/".join(key)
            )
        seen.add(key)


def _summarize_rows(
    rows: Sequence[DonorMetricRow],
    *,
    iterations: int,
    confidence: float,
    minimum_donors: int,
    seed: int,
) -> Tuple[BenchmarkSummaryRow, ...]:
    grouped: Dict[Tuple[str, str, str], List[DonorMetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.cohort_id, row.method, row.metric), []).append(row)
    output: List[BenchmarkSummaryRow] = []
    for key in sorted(grouped):
        cohort_id, method, metric = key
        group = grouped[key]
        values = np.asarray(
            [row.value for row in group if row.status is BenchmarkStatus.OK], dtype=np.float64
        )
        n_missing = sum(row.status is BenchmarkStatus.MISSING for row in group)
        n_data_limited = sum(row.status is BenchmarkStatus.DATA_LIMITED for row in group)
        estimate = float(values.mean()) if len(values) else None
        ci_lower: Optional[float] = None
        ci_upper: Optional[float] = None
        if len(values) == 0:
            status = BenchmarkStatus.DATA_LIMITED if n_data_limited else BenchmarkStatus.MISSING
            reason = "no finite donor metrics are available"
        elif len(values) < minimum_donors:
            status = BenchmarkStatus.DATA_LIMITED
            reason = f"requires at least {minimum_donors} donor metrics; found {len(values)}"
        else:
            status = BenchmarkStatus.OK
            reason = ""
            ci_lower, ci_upper = _bootstrap_mean_ci(
                values,
                iterations=iterations,
                confidence=confidence,
                seed=_stable_seed(seed, "summary", *key),
            )
        output.append(
            BenchmarkSummaryRow(
                cohort_id=cohort_id,
                method=method,
                metric=metric,
                estimate=estimate,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n_donors=len(values),
                n_missing=n_missing,
                n_data_limited=n_data_limited,
                status=status,
                reason=reason,
            )
        )
    return tuple(output)


def _compare_rows(
    rows: Sequence[DonorMetricRow],
    *,
    baseline_method: str,
    metric_directions: Mapping[str, bool],
    iterations: int,
    confidence: float,
    minimum_donors: int,
    seed: int,
) -> Tuple[MethodComparisonRow, ...]:
    by_endpoint: Dict[Tuple[str, str], List[DonorMetricRow]] = {}
    for row in rows:
        by_endpoint.setdefault((row.cohort_id, row.metric), []).append(row)
    output: List[MethodComparisonRow] = []
    for endpoint in sorted(by_endpoint):
        cohort_id, metric = endpoint
        endpoint_rows = by_endpoint[endpoint]
        methods = sorted({row.method for row in endpoint_rows if row.method != baseline_method})
        if not methods:
            continue
        if metric not in metric_directions:
            raise ValueError(f"metric_directions is missing compared metric {metric!r}")
        higher_is_better = metric_directions[metric]
        if not isinstance(higher_is_better, (bool, np.bool_)):
            raise ValueError(f"metric direction for {metric!r} must be boolean")
        baseline_rows = {
            row.donor_id: row for row in endpoint_rows if row.method == baseline_method
        }
        baseline_ok = {
            donor_id: row.value
            for donor_id, row in baseline_rows.items()
            if row.status is BenchmarkStatus.OK
        }
        for method in methods:
            method_rows = {row.donor_id: row for row in endpoint_rows if row.method == method}
            method_ok = {
                donor_id: row.value
                for donor_id, row in method_rows.items()
                if row.status is BenchmarkStatus.OK
            }
            paired_donors = sorted(set(method_ok).intersection(baseline_ok))
            raw_difference: Optional[float] = None
            improvement: Optional[float] = None
            ci_lower: Optional[float] = None
            ci_upper: Optional[float] = None
            probability_better: Optional[float] = None
            if not paired_donors:
                if _endpoint_is_missing(method_rows) or _endpoint_is_missing(baseline_rows):
                    status = BenchmarkStatus.MISSING
                    reason = "method or baseline has no available donor metrics"
                else:
                    status = BenchmarkStatus.DATA_LIMITED
                    reason = "method and baseline have no overlapping evaluable donors"
            else:
                method_values = np.asarray(
                    [method_ok[donor_id] for donor_id in paired_donors], dtype=np.float64
                )
                baseline_values = np.asarray(
                    [baseline_ok[donor_id] for donor_id in paired_donors], dtype=np.float64
                )
                differences = method_values - baseline_values
                oriented = differences if higher_is_better else -differences
                raw_difference = float(differences.mean())
                improvement = float(oriented.mean())
                if len(paired_donors) < minimum_donors:
                    status = BenchmarkStatus.DATA_LIMITED
                    reason = (
                        f"requires at least {minimum_donors} paired donors; "
                        f"found {len(paired_donors)}"
                    )
                else:
                    status = BenchmarkStatus.OK
                    reason = ""
                    ci_lower, ci_upper, probability_better = _bootstrap_comparison(
                        oriented,
                        iterations=iterations,
                        confidence=confidence,
                        seed=_stable_seed(seed, "comparison", cohort_id, method, metric),
                    )
            output.append(
                MethodComparisonRow(
                    cohort_id=cohort_id,
                    method=method,
                    baseline_method=baseline_method,
                    metric=metric,
                    higher_is_better=bool(higher_is_better),
                    mean_difference=raw_difference,
                    mean_improvement=improvement,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    probability_better=probability_better,
                    n_paired_donors=len(paired_donors),
                    n_method_donors=len(method_ok),
                    n_baseline_donors=len(baseline_ok),
                    status=status,
                    reason=reason,
                )
            )
    return tuple(output)


def _endpoint_is_missing(rows: Mapping[str, DonorMetricRow]) -> bool:
    return not rows or all(row.status is BenchmarkStatus.MISSING for row in rows.values())


def _bootstrap_mean_ci(
    values: np.ndarray, *, iterations: int, confidence: float, seed: int
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(iterations, len(values)))
    sampled = values[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(sampled, alpha)), float(np.quantile(sampled, 1.0 - alpha))


def _bootstrap_comparison(
    improvements: np.ndarray, *, iterations: int, confidence: float, seed: int
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(improvements), size=(iterations, len(improvements)))
    sampled = improvements[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(sampled, alpha)),
        float(np.quantile(sampled, 1.0 - alpha)),
        float(np.mean(sampled > 0.0)),
    )


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)


def _donor_sort_key(row: DonorMetricRow) -> Tuple[str, str, str, str]:
    return row.cohort_id, row.donor_id, row.method, row.metric


def _record_dict(row: BenchmarkRecord) -> Dict[str, object]:
    result = asdict(row)
    status = result.get("status")
    if isinstance(status, BenchmarkStatus):
        result["status"] = status.value
    return result


def _typed_records(record_type: str, rows: Sequence[BenchmarkRecord]) -> List[Dict[str, object]]:
    output = []
    for row in rows:
        record: Dict[str, object] = {"record_type": record_type}
        record.update(_record_dict(row))
        output.append(record)
    return output


def _tsv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value
