import csv
import json

import pytest

from heir.evaluation.benchmark import (
    BenchmarkStatus,
    DonorMetricRow,
    build_benchmark_report,
    write_benchmark_json,
    write_benchmark_tsv,
)


def _complete_rows():
    return [
        DonorMetricRow("cohort1", "d1", "SIGHT", "macro_f1", 0.8),
        DonorMetricRow("cohort1", "d2", "SIGHT", "macro_f1", 0.6),
        DonorMetricRow(
            "cohort1",
            "d3",
            "SIGHT",
            "macro_f1",
            None,
            BenchmarkStatus.MISSING,
            "annotation unavailable",
        ),
        DonorMetricRow("cohort1", "d1", "baseline", "macro_f1", 0.5),
        DonorMetricRow("cohort1", "d2", "baseline", "macro_f1", 0.5),
        DonorMetricRow("cohort1", "d1", "SIGHT", "rmse", 1.0),
        DonorMetricRow("cohort1", "d2", "SIGHT", "rmse", 2.0),
        DonorMetricRow("cohort1", "d1", "baseline", "rmse", 2.0),
        DonorMetricRow("cohort1", "d2", "baseline", "rmse", 4.0),
    ]


def test_report_aggregates_donors_and_orients_method_comparisons():
    report = build_benchmark_report(
        _complete_rows(),
        baseline_method="baseline",
        metric_directions={"macro_f1": True, "rmse": False},
        iterations=500,
        seed=9,
    )
    sight_f1 = next(
        row for row in report.summaries if row.method == "SIGHT" and row.metric == "macro_f1"
    )
    assert sight_f1.status is BenchmarkStatus.OK
    assert sight_f1.estimate == pytest.approx(0.7)
    assert sight_f1.n_donors == 2
    assert sight_f1.n_missing == 1
    assert sight_f1.ci_lower <= sight_f1.estimate <= sight_f1.ci_upper

    f1_comparison = next(row for row in report.comparisons if row.metric == "macro_f1")
    assert f1_comparison.mean_difference == pytest.approx(0.2)
    assert f1_comparison.mean_improvement == pytest.approx(0.2)
    assert f1_comparison.probability_better == 1.0
    assert f1_comparison.n_paired_donors == 2

    rmse_comparison = next(row for row in report.comparisons if row.metric == "rmse")
    assert rmse_comparison.mean_difference == pytest.approx(-1.5)
    assert rmse_comparison.mean_improvement == pytest.approx(1.5)
    assert rmse_comparison.probability_better == 1.0


def test_report_is_order_independent_and_uses_explicit_limited_statuses():
    rows = [
        DonorMetricRow("limited", "d1", "SIGHT", "score", 0.8),
        DonorMetricRow("limited", "d1", "baseline", "score", 0.7),
        DonorMetricRow(
            "missing",
            "d1",
            "SIGHT",
            "score",
            None,
            BenchmarkStatus.MISSING,
            "truth absent",
        ),
        DonorMetricRow(
            "missing",
            "d1",
            "baseline",
            "score",
            None,
            BenchmarkStatus.MISSING,
            "truth absent",
        ),
    ]
    kwargs = {
        "baseline_method": "baseline",
        "metric_directions": {"score": True},
        "iterations": 100,
        "seed": 4,
    }
    forward = build_benchmark_report(rows, **kwargs)
    reverse = build_benchmark_report(reversed(rows), **kwargs)
    assert forward.to_dict() == reverse.to_dict()

    limited_summary = next(
        row for row in forward.summaries if row.cohort_id == "limited" and row.method == "SIGHT"
    )
    assert limited_summary.status is BenchmarkStatus.DATA_LIMITED
    assert limited_summary.estimate == pytest.approx(0.8)
    assert limited_summary.ci_lower is None
    limited_comparison = next(row for row in forward.comparisons if row.cohort_id == "limited")
    assert limited_comparison.status is BenchmarkStatus.DATA_LIMITED
    assert limited_comparison.n_paired_donors == 1

    missing_summary = next(
        row for row in forward.summaries if row.cohort_id == "missing" and row.method == "SIGHT"
    )
    assert missing_summary.status is BenchmarkStatus.MISSING
    assert missing_summary.estimate is None
    missing_comparison = next(row for row in forward.comparisons if row.cohort_id == "missing")
    assert missing_comparison.status is BenchmarkStatus.MISSING


def test_report_writers_emit_json_safe_and_normalized_outputs(tmp_path):
    report = build_benchmark_report(
        _complete_rows(),
        baseline_method="baseline",
        metric_directions={"macro_f1": True, "rmse": False},
        iterations=50,
    )
    json_path = write_benchmark_json(report, tmp_path / "nested" / "benchmark.json")
    tsv_path = write_benchmark_tsv(report, tmp_path / "nested" / "benchmark.tsv")

    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == "heir-benchmark-v1"
    assert parsed["donor_metrics"][0]["status"] == "ok"
    assert "NaN" not in json_path.read_text(encoding="utf-8")
    with tsv_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    assert {record["record_type"] for record in records} == {
        "donor_metric",
        "summary",
        "comparison",
    }
    missing = next(
        record
        for record in records
        if record["record_type"] == "donor_metric" and record["status"] == "missing"
    )
    assert missing["value"] == ""


def test_report_rejects_ambiguous_or_invalid_inputs():
    row = DonorMetricRow("cohort", "donor", "SIGHT", "score", 0.5)
    with pytest.raises(ValueError, match="duplicate donor metric"):
        build_benchmark_report([row, row])
    with pytest.raises(ValueError, match="finite value"):
        DonorMetricRow("cohort", "donor", "SIGHT", "score", float("nan"))
    with pytest.raises(ValueError, match="cannot carry a value"):
        DonorMetricRow(
            "cohort",
            "donor",
            "SIGHT",
            "score",
            0.5,
            BenchmarkStatus.MISSING,
        )
    with pytest.raises(ValueError, match="metric_directions"):
        build_benchmark_report(
            [
                row,
                DonorMetricRow("cohort", "donor", "baseline", "score", 0.4),
            ],
            baseline_method="baseline",
        )
