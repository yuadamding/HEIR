"""Behavioral tests for the retrospective snPATHO-DeepBench evaluator."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import yaml

from heir.data import RNAReference
from heir.evaluation.deepbench import (
    OPTIONAL_ARTIFACTS,
    PRIMARY_METHOD,
    SHUFFLE_METHOD,
    TYPE_MEAN_METHOD,
    DeepBenchPlan,
    _method_macro_summaries,
    _primary_diagnostic,
    _readiness,
    _reference_linear_profiles,
    _type_mean_cells,
    aggregate_cells_to_spots,
    deepbench_expression_metrics,
    validate_deepbench_specification,
    write_deepbench_report,
)
from heir.inference import PredictionBundle


def _specification() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "snpatho_deepbench_v1.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _linear_reference() -> RNAReference:
    return RNAReference(
        sample_id="reference",
        cell_ids=np.asarray(["a1", "a2", "b1"]),
        gene_ids=np.asarray(["g1", "g2"]),
        counts=np.asarray([[10.0, 0.0], [0.0, 20.0], [3.0, 7.0]]),
        library_sizes=np.asarray([100.0, 200.0, 1_000.0]),
        cell_type_labels=np.asarray(["A", "A", "B"]),
    )


def test_attached_deepbench_method_critical_fields_are_frozen() -> None:
    payload = _specification()
    validate_deepbench_specification(payload)

    changed = deepcopy(payload)
    changed["statistics"]["pooled_spot_inference"] = "allowed"
    with pytest.raises(ValueError, match="pooled_spot_inference"):
        validate_deepbench_specification(changed)


def test_rna_mass_spot_aggregation_operates_in_linear_space() -> None:
    expression = np.log1p(np.asarray([[9.0, 1.0], [1.0, 5.0], [7.0, 3.0]]))
    spot_index = np.asarray([0, 0, -1])
    observed, mass = aggregate_cells_to_spots(
        expression,
        spot_index,
        num_spots=2,
        cell_rna_mass=np.asarray([1.0, 3.0, 100.0]),
    )

    np.testing.assert_allclose(observed[0], np.log1p([3.0, 4.0]))
    np.testing.assert_allclose(observed[1], [0.0, 0.0])
    np.testing.assert_allclose(mass, [4.0, 0.0])


def test_reference_profiles_pool_raw_counts_and_full_library_mass() -> None:
    profiles, median_library_sizes = _reference_linear_profiles(
        _linear_reference(),
        ["A", "B"],
    )

    np.testing.assert_allclose(profiles[0], np.asarray([10.0, 20.0]) / 300.0 * 10_000.0)
    np.testing.assert_allclose(profiles[1], np.asarray([3.0, 7.0]) / 1_000.0 * 10_000.0)
    np.testing.assert_allclose(median_library_sizes, [150.0, 1_000.0])


def test_matched_type_mean_uses_hard_assignment_not_soft_averaging() -> None:
    reference = _linear_reference()
    prediction = cast(
        PredictionBundle,
        SimpleNamespace(
            type_names=np.asarray(["A", "B"]),
            type_probabilities=np.asarray([[0.51, 0.49], [0.10, 0.90]]),
        ),
    )

    linear_cells = np.expm1(_type_mean_cells(reference, prediction))
    expected_profiles, _ = _reference_linear_profiles(reference, ["A", "B"])

    np.testing.assert_allclose(linear_cells, expected_profiles, rtol=1.0e-6)
    assert not np.allclose(
        linear_cells[0],
        0.51 * expected_profiles[0] + 0.49 * expected_profiles[1],
    )


def test_constant_prediction_is_zero_not_dropped_when_truth_varies() -> None:
    observed = np.asarray(
        [
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 2.0],
        ],
        dtype=np.float64,
    )
    predicted = np.ones_like(observed)
    coordinates = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
    result = deepbench_expression_metrics(predicted, observed, coordinates)

    assert result["per_gene"]["spearman"] == [0.0, None]
    assert result["summary"]["median_gene_spearman"] == 0.0
    assert result["summary"]["fraction_genes_evaluable"] == 0.5


def test_deepbench_metrics_include_hotspots_locations_and_spatial_agreement() -> None:
    coordinates = np.asarray([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float64)
    observed = np.asarray([[0.0, 1.0], [0.2, 1.2], [0.4, 1.4], [0.6, 1.6], [0.8, 1.8], [1.0, 2.0]])
    result = deepbench_expression_metrics(observed.copy(), observed, coordinates)
    summary = result["summary"]

    assert summary["median_gene_spearman"] == pytest.approx(1.0)
    assert summary["median_gene_mse"] == pytest.approx(0.0)
    assert summary["median_hotspot_dice"] == pytest.approx(1.0)
    assert summary["median_hotspot_jaccard"] == pytest.approx(1.0)
    assert summary["mean_location_cosine"] == pytest.approx(1.0)
    assert summary["morans_i_mae"] == pytest.approx(0.0)


def test_registered_optional_artifacts_do_not_make_requested_plan_ready(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "registered.npz"
    artifact.touch()
    registered = {name: artifact for name in OPTIONAL_ARTIFACTS}
    plan = DeepBenchPlan(
        source_path=tmp_path / "plan.yaml",
        source_sha256="0" * 64,
        name="snpatho_deepbench_v1",
        status="retrospective_diagnostic",
        historical_result_name="snpatho_locked_v0_2",
        frozen_plan=tmp_path / "locked-plan.json",
        frozen_plan_sha256="1" * 64,
        historical_report=tmp_path / "locked-report.json",
        historical_report_sha256="2" * 64,
        sample_ids=("4066",),
        minimum_nuclei=3,
        bootstrap_iterations=4,
        primary_seeds=(17,),
        optional_artifacts=registered,
        optional_artifact_sha256={name: "3" * 64 for name in OPTIONAL_ARTIFACTS},
        specification={},
    )
    case = {
        "section_id": "4066",
        "methods": {
            PRIMARY_METHOD: {
                "per_gene": {"spearman": [0.4, 0.2], "observed_mean": [1.0, 2.0]},
                "summary": {"median_gene_mse": 0.5},
            },
            TYPE_MEAN_METHOD: {
                "per_gene": {"spearman": [0.1, 0.1]},
                "summary": {"median_gene_mse": 1.0},
            },
            SHUFFLE_METHOD: {"per_gene": {"spearman": [0.0, 0.0]}},
        },
    }

    readiness = _readiness(plan)
    diagnostic = _primary_diagnostic([case], plan)
    optional_statuses = {
        item["component"]: item["status"]
        for item in readiness
        if item["component"] in OPTIONAL_ARTIFACTS
    }

    assert set(optional_statuses.values()) == {"registered_not_implemented"}
    assert diagnostic["requested_primary_status"] == (
        "not_testable_registered_refined_schema_not_implemented"
    )
    assert not all(item["status"] == "ready" for item in readiness)


def test_macro_summary_and_per_gene_tsv_preserve_biological_units(tmp_path: Path) -> None:
    cases = []
    for section_id, value in (("4066", 0.1), ("4399", 0.3), ("4411", 0.2)):
        cases.append(
            {
                "section_id": section_id,
                "methods": {
                    "method": {
                        "aggregation": "rna_mass",
                        "spots_evaluated": 10,
                        "spot_coverage": 1.0,
                        "summary": {"median_gene_spearman": value},
                        "per_gene": {
                            "gene_names": ["G1"],
                            "correlation_status": ["prediction_constant_scored_zero"],
                            "correlation_reason": ["prediction is constant"],
                            "spearman": [0.0],
                        },
                    }
                },
            }
        )
    macro = _method_macro_summaries(cases)
    assert macro["method"]["metrics"]["median_gene_spearman"] == {
        "macro_mean": pytest.approx(0.2),
        "minimum": pytest.approx(0.1),
        "maximum": pytest.approx(0.3),
        "specimens_evaluable": 3,
    }

    report = {"method_macro": macro, "cases": cases}
    tsv = tmp_path / "report.tsv"
    write_deepbench_report(report, json_path=tmp_path / "report.json", tsv_path=tsv)
    rows = tsv.read_text(encoding="utf-8").splitlines()

    assert any(row.startswith("macro\tmacro\tmethod") for row in rows)
    assert any(
        "\t4066\tmethod\trna_mass\tG1\tspearman\t0\t10\t"
        "prediction_constant_scored_zero\tprediction is constant" in row
        for row in rows
    )
