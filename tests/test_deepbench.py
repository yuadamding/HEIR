"""Behavioral tests for the retrospective snPATHO-DeepBench evaluator."""

import json
from copy import deepcopy
from dataclasses import replace
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
    SOFT_TYPE_MEAN_METHOD,
    TYPE_MEAN_METHOD,
    DeepBenchPlan,
    _bootstrap_macro_delta,
    _method_macro_summaries,
    _primary_diagnostic,
    _readiness,
    _record_shuffle_seed,
    _reference_linear_profiles,
    _reference_type_support,
    _repeated_final_record_shuffle_null,
    _soft_type_mean_cells,
    _top_indices,
    _type_mean_cells,
    _validate_r1_reference_identity,
    aggregate_cells_to_spots,
    deepbench_expression_metrics,
    validate_deepbench_specification,
    write_deepbench_report,
)
from heir.inference import PredictionBundle
from heir.utils import sha256_file


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


def test_committed_workflow_audit_is_internally_consistent() -> None:
    path = Path(__file__).resolve().parents[1] / "reports" / "snpatho_reference_workflow_audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["processing_method_column"] == "processing_method"
    assert payload["cell_type_column"] == "major_annotation"
    assert payload["filters"]["applied"] is False
    for specimen in payload["specimens"].values():
        assert len(specimen["source_sha256"]) == 64
        assert sum(specimen["counts_by_workflow"].values()) == specimen["total_metadata_rows"]
        for workflow, count in specimen["counts_by_workflow"].items():
            assert sum(specimen["cell_type_counts_by_workflow"][workflow].values()) == count


def test_committed_r1_manifest_freezes_exact_ffpe_filter_and_counts() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "reports" / "snpatho_r1_reference_manifest.json").read_text(encoding="utf-8")
    )

    assert payload["filter"] == {
        "column": "processing_method",
        "accepted_values": ["FFPE_snPATHO"],
        "matching": "exact",
    }
    assert payload["gene_panel"]["sha256"] == (
        "22ddb91188b3b124d5cf3ec0f7ae81017399d141e39647b0dce80675119fe927"
    )
    assert payload["cell_type_annotation"]["primary_clean_reannotation_status"] == ("not_complete")
    for specimen in payload["specimens"].values():
        assert specimen["source_observations"] > specimen["selected_observations"]
        assert sum(specimen["cell_type_counts"].values()) == specimen["selected_observations"]
        for name in (
            "source_rds_sha256",
            "h5ad_sha256",
            "conversion_provenance_sha256",
            "panel_reference_sha256",
            "latent_reference_sha256",
            "prototypes_sha256",
        ):
            digest = specimen[name]
            assert len(digest) == 64
            assert set(digest) <= set("0123456789abcdef")


def test_public_summary_uses_current_plan_schema_and_optional_local_report_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (root / "reports" / "snpatho_deepbench_v1_summary.json").read_text(encoding="utf-8")
    )
    plan = root / "configs" / "experiments" / "snpatho_deepbench_v1.yaml"

    assert summary["schema"] == "heir.snpatho_deepbench_public_summary.v2"
    assert summary["report_schema"] == "heir.snpatho_deepbench.v2"
    assert summary["provenance"]["deepbench_plan_sha256"] == sha256_file(plan)
    assert (
        "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean"
        in summary["macro"]
    )
    assert "bootstrap_fraction_delta_positive" in summary["macro"]
    assert "paired_bootstrap_probability_positive" not in summary["macro"]

    local_outputs = {
        "full_local_json_sha256": root / "artifacts/snpatho/deepbench_v1/report.json",
        "full_local_tsv_sha256": root / "artifacts/snpatho/deepbench_v1/report.tsv",
        "full_local_markdown_sha256": root / "artifacts/snpatho/deepbench_v1/report.md",
    }
    for field, path in local_outputs.items():
        if path.is_file():
            assert summary["provenance"][field] == sha256_file(path)
    local_report = local_outputs["full_local_json_sha256"]
    if local_report.is_file():
        report = json.loads(local_report.read_text(encoding="utf-8"))
        assert report["benchmark"]["plan_sha256"] == summary["provenance"]["deepbench_plan_sha256"]
        assert (
            report["primary"]["macro_delta"]
            == summary["macro"][
                "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean"
            ]
        )
        assert (
            report["primary"]["bootstrap"]["bootstrap_fraction_delta_positive"]
            == summary["macro"]["bootstrap_fraction_delta_positive"]
        )


def test_r1_reference_identity_is_bound_to_specimen_and_h5ad_lineage() -> None:
    source_sha256 = "a" * 64
    reference = RNAReference(
        sample_id="4066",
        cell_ids=np.asarray(["a", "b"]),
        gene_ids=np.asarray(["g"]),
        counts=np.asarray([[1.0], [2.0]]),
        donor_ids=np.asarray(["4066", "4066"]),
        sample_ids=np.asarray(["4066", "4066"]),
        block_id="4066_FFPE",
        source_count_sha256=source_sha256,
    )
    manifest_entry = {"h5ad_sha256": source_sha256}

    _validate_r1_reference_identity(reference, "4066", manifest_entry)

    with pytest.raises(ValueError, match="sample_id differs"):
        _validate_r1_reference_identity(
            replace(reference, sample_id="4399"),
            "4066",
            manifest_entry,
        )
    with pytest.raises(ValueError, match="source-count lineage"):
        _validate_r1_reference_identity(
            replace(reference, source_count_sha256="b" * 64),
            "4066",
            manifest_entry,
        )


def test_attached_deepbench_method_critical_fields_are_frozen() -> None:
    payload = _specification()
    validate_deepbench_specification(payload)

    changed = deepcopy(payload)
    changed["statistics"]["pooled_spot_inference"] = "allowed"
    with pytest.raises(ValueError, match="pooled_spot_inference"):
        validate_deepbench_specification(changed)

    too_few_shuffles = deepcopy(payload)
    too_few_shuffles["statistics"]["final_cell_record_shuffle_permutations"] = 99
    with pytest.raises(ValueError, match="final_cell_record_shuffle_permutations"):
        validate_deepbench_specification(too_few_shuffles)


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


def test_soft_type_mean_probability_weights_linear_profiles() -> None:
    reference = _linear_reference()
    prediction = cast(
        PredictionBundle,
        SimpleNamespace(
            type_names=np.asarray(["A", "B"]),
            type_probabilities=np.asarray([[0.51, 0.49], [0.10, 0.90]]),
        ),
    )

    linear_cells = np.expm1(_soft_type_mean_cells(reference, prediction))
    profiles, _ = _reference_linear_profiles(reference, ["A", "B"])

    np.testing.assert_allclose(
        linear_cells,
        np.asarray([[0.51, 0.49], [0.10, 0.90]]).dot(profiles),
        rtol=1.0e-6,
    )
    assert SOFT_TYPE_MEAN_METHOD == "historical_integrated_soft_type_mean"


def test_missing_reference_type_is_audited_then_fails_closed() -> None:
    reference = _linear_reference()
    prediction = cast(
        PredictionBundle,
        SimpleNamespace(
            type_names=np.asarray(["A", "missing"]),
            type_probabilities=np.asarray([[0.1, 0.9], [0.8, 0.2]]),
        ),
    )

    audit = _reference_type_support(reference, prediction)

    assert audit["reference_supported_prediction_cell_types"] == ["A"]
    assert audit["missing_prediction_cell_types"] == ["missing"]
    assert audit["missing_type_policy"] == "fail_closed_no_global_profile_fallback"
    assert audit["hard_assignment_global_fallback_cells"] == 1
    assert audit["hard_assignment_global_fallback_cell_fraction"] == pytest.approx(0.5)
    assert audit["soft_assignment_global_fallback_probability_mass_mean"] == pytest.approx(0.55)
    with pytest.raises(ValueError, match="global-profile fallback is prohibited"):
        _reference_linear_profiles(reference, ["A", "missing"])


def test_top_decile_ties_use_frozen_lower_index_policy() -> None:
    values = np.asarray([9.0, 9.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    selected = _top_indices(values, fraction=0.10)

    np.testing.assert_array_equal(selected, [0, 1])


def test_repeated_final_record_shuffle_is_deterministic_compact_and_preserves_draw_zero() -> None:
    spot_index = np.repeat(np.arange(4, dtype=np.int64), 3)
    expression = np.log1p(
        np.column_stack(
            (
                np.arange(1, 13, dtype=np.float64),
                np.asarray([1, 4, 2, 8, 3, 9, 5, 7, 6, 12, 10, 11], dtype=np.float64),
            )
        )
    )
    weights = np.linspace(0.5, 2.0, len(expression))
    truth = np.asarray(
        [[0.1, 1.2], [0.8, 0.3], [1.4, 1.8], [2.0, 0.9]],
        dtype=np.float64,
    )
    primary_spots = np.ones(4, dtype=bool)

    first = _repeated_final_record_shuffle_null(
        expression,
        weights,
        spot_index,
        primary_spots,
        truth,
        sample="4066",
        seed=17,
        permutations=100,
    )
    repeated = _repeated_final_record_shuffle_null(
        expression,
        weights,
        spot_index,
        primary_spots,
        truth,
        sample="4066",
        seed=17,
        permutations=100,
    )

    assert first[0] == repeated[0]
    np.testing.assert_array_equal(first[1], repeated[1])
    np.testing.assert_array_equal(first[2], repeated[2])
    np.testing.assert_array_equal(first[3], repeated[3])
    assert first[0]["permutations"] == 100
    assert first[0]["statistic"] == "median_gene_spearman"
    assert "values" not in first[0]
    assert set(first[0]["empirical_percentile_interval_95"]) == {"lower", "upper"}

    assigned = np.flatnonzero(spot_index >= 0)
    draw_zero = np.random.default_rng(_record_shuffle_seed(17, "4066", 0)).permutation(assigned)
    expected_spots, expected_mass = aggregate_cells_to_spots(
        expression[draw_zero],
        spot_index[assigned],
        len(truth),
        weights[draw_zero],
    )
    np.testing.assert_allclose(first[1], expected_spots)
    np.testing.assert_allclose(first[2], expected_mass)
    expected_metric = deepbench_expression_metrics(
        expected_spots,
        truth,
        np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64),
    )["summary"]["median_gene_spearman"]
    assert first[3][0] == pytest.approx(expected_metric)


def test_record_shuffle_seeds_are_distinct_across_draws_and_specimens() -> None:
    seeds = {
        _record_shuffle_seed(17, specimen, draw)
        for specimen in ("4066", "4399", "4411")
        for draw in range(100)
    }

    assert len(seeds) == 300


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
    assert result["summary"]["prediction_constant_scored_zero_count"] == 1
    assert result["summary"]["observed_constant_excluded_count"] == 1


def test_deepbench_metrics_include_hotspots_locations_and_spatial_agreement() -> None:
    coordinates = np.asarray([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float64)
    observed = np.asarray([[0.0, 1.0], [0.2, 1.2], [0.4, 1.4], [0.6, 1.6], [0.8, 1.8], [1.0, 2.0]])
    result = deepbench_expression_metrics(observed.copy(), observed, coordinates)
    summary = result["summary"]

    assert summary["median_gene_spearman"] == pytest.approx(1.0)
    assert summary["median_gene_mse"] == pytest.approx(0.0)
    assert summary["median_hotspot_dice"] == pytest.approx(1.0)
    assert summary["median_hotspot_jaccard"] == pytest.approx(1.0)
    assert summary["median_expression_detection_auroc"] == pytest.approx(1.0)
    assert "median_hotspot_auroc" not in summary
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
        final_cell_record_shuffle_permutations=100,
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
                "summary": {"median_gene_spearman": 0.3, "median_gene_mse": 0.5},
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

    assert optional_statuses["primary_ffpe_snpatho_reference_manifest"] == (
        "partial_consumed_retrospective_sensitivity"
    )
    assert {
        status
        for name, status in optional_statuses.items()
        if name != "primary_ffpe_snpatho_reference_manifest"
    } == {"registered_not_implemented"}
    assert diagnostic["requested_primary_status"] == (
        "not_testable_registered_refined_schema_not_implemented"
    )
    assert diagnostic["diagnostic_statistic"]["specimen_formula"] == (
        "median_g(rho_HEIR,g - rho_historical_integrated_hard_type_mean,g)"
    )
    assert diagnostic["specimens"][0][
        "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean"
    ] == pytest.approx(0.2)
    assert diagnostic["specimens"][0][
        "median_paired_per_gene_spearman_delta_vs_final_record_shuffle_draw_0"
    ] == pytest.approx(0.3)
    assert "median_gene_spearman_delta_vs_spatial_shuffle" not in diagnostic["specimens"][0]
    repeated_diagnostic = _primary_diagnostic(
        [case],
        plan,
        repeated_shuffle_statistics={"4066": np.linspace(-0.1, 0.2, 100)},
    )
    repeated_comparison = repeated_diagnostic["specimens"][0][
        "repeated_final_record_shuffle_null_comparison"
    ]
    assert repeated_comparison["null_permutations"] == 100
    assert repeated_comparison["observed_heir_empirical_percentile_in_null"] == 1.0
    assert repeated_comparison["observed_heir_above_null_95_upper"] is True
    assert (
        repeated_diagnostic["rules"][
            "above_repeated_final_record_shuffle_null_95_upper_in_at_least_two_specimens"
        ]
        is False
    )
    assert not all(item["status"] == "ready" for item in readiness)


def test_bootstrap_positive_field_is_descriptive_not_probabilistic() -> None:
    result = _bootstrap_macro_delta(
        [np.asarray([-0.2, 0.1, 0.3, 0.4])],
        [np.asarray([1.0, 2.0, 3.0, 4.0])],
        iterations=20,
        seed=17,
    )

    assert "bootstrap_fraction_delta_positive" in result
    assert "probability_positive" not in result


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


def test_tsv_surfaces_repeated_shuffle_null_summary(tmp_path: Path) -> None:
    distribution = {
        "statistic": "median_gene_spearman",
        "permutations": 100,
        "mean": 0.01,
        "median": 0.02,
        "sample_standard_deviation": 0.03,
        "minimum": -0.04,
        "maximum": 0.05,
        "empirical_percentile_interval_95": {"lower": -0.03, "upper": 0.04},
    }
    comparison = {
        "observed_heir_median_gene_spearman": 0.06,
        "observed_heir_empirical_percentile_in_null": 0.99,
        "observed_heir_minus_null_median": 0.04,
        "observed_heir_above_null_95_upper": True,
        "null_permutations": 100,
    }
    report = {
        "method_macro": {},
        "cases": [],
        "final_cell_record_shuffle_null": {
            "specimens": {"4066": distribution},
            "equal_weight_specimen_macro": distribution,
        },
        "primary": {
            "specimens": [
                {
                    "section_id": "4066",
                    "repeated_final_record_shuffle_null_comparison": comparison,
                }
            ]
        },
    }
    tsv = tmp_path / "shuffle.tsv"

    write_deepbench_report(report, json_path=tmp_path / "shuffle.json", tsv_path=tsv)
    rows = tsv.read_text(encoding="utf-8").splitlines()

    assert any(
        row.startswith(
            "shuffle_null\t4066\t"
            "heir_final_cell_record_shuffle_historical_integrated_reference_library_size_weighted"
        )
        and "\tnull_empirical_95_upper\t0.04\t" in row
        for row in rows
    )
    assert any(
        row.startswith(
            "shuffle_null_comparison\t4066\t"
            "heir_round0_historical_integrated_reference_library_size_weighted"
        )
        and "\tobserved_heir_above_null_95_upper\t1\t" in row
        for row in rows
    )
