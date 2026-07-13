"""Coverage-aware expression endpoint tests."""

import numpy as np
import pytest

from heir.evaluation.coverage import (
    COVERAGE_EVALUATION_SCHEMA,
    build_truth_gene_mask,
    evaluate_methods_on_truth_gene_mask,
    fixed_coverage_selective_aggregation,
    full_coverage_type_mean_aggregation,
)


def _spot_aggregation(
    values: np.ndarray,
    gene_names: list[str],
    spot_ids: list[str],
):
    matrix = np.asarray(values, dtype=np.float64)
    return fixed_coverage_selective_aggregation(
        cell_log_expression=matrix,
        uncertainty=np.zeros(len(matrix), dtype=np.float64),
        target_coverage=1.0,
        cell_ids=["cell-%d" % index for index in range(len(matrix))],
        spot_ids=spot_ids,
        gene_names=gene_names,
        spot_index=np.arange(len(matrix), dtype=np.int64),
        num_spots=len(spot_ids),
        cell_rna_mass=np.ones(len(matrix), dtype=np.float64),
    )


def test_full_coverage_replaces_abstentions_and_preserves_rna_mass() -> None:
    cell_linear = np.asarray(
        [
            [1.0, 10.0],
            [100.0, 100.0],
            [3.0, 30.0],
            [200.0, 200.0],
        ]
    )
    type_mean_linear = np.asarray([[2.0, 20.0], [4.0, 40.0]])
    result = full_coverage_type_mean_aggregation(
        cell_log_expression=np.log1p(cell_linear),
        abstain=np.asarray([False, True, False, True]),
        frozen_type_index=np.asarray([0, 1, 0, 1]),
        frozen_type_mean_log_expression=np.log1p(type_mean_linear),
        cell_ids=["c1", "c2", "c3", "c4"],
        spot_ids=["s1", "s2"],
        gene_names=["g1", "g2"],
        type_names=["A", "B"],
        spot_index=np.asarray([0, 0, 1, 1]),
        num_spots=2,
        cell_rna_mass=np.asarray([1.0, 3.0, 2.0, 1.0]),
    )

    expected_linear = np.asarray(
        [
            [(1.0 * 1.0 + 4.0 * 3.0) / 4.0, (10.0 * 1.0 + 40.0 * 3.0) / 4.0],
            [(3.0 * 2.0 + 4.0 * 1.0) / 3.0, (30.0 * 2.0 + 40.0 * 1.0) / 3.0],
        ]
    )
    np.testing.assert_allclose(result.spot_expression, np.log1p(expected_linear), rtol=1e-6)
    np.testing.assert_allclose(result.spot_mass, [4.0, 3.0])
    np.testing.assert_array_equal(result.selected_cells, [True, True, True, True])
    assert result.schema == COVERAGE_EVALUATION_SCHEMA
    assert result.requested_coverage == pytest.approx(1.0)
    assert result.realized_coverage == pytest.approx(1.0)
    assert result.metadata["fallback_cells"] == 2
    assert len(str(result.metadata["fallback_matrix_sha256"])) == 64
    assert len(str(result.metadata["rna_mass_vector_sha256"])) == 64
    assert len(str(result.metadata["cell_to_spot_mapping_sha256"])) == 64
    assert len(str(result.metadata["abstain_mask_sha256"])) == 64
    assert len(str(result.metadata["ordered_cell_ids_sha256"])) == 64
    assert len(str(result.metadata["ordered_spot_ids_sha256"])) == 64
    assert len(str(result.metadata["ordered_gene_names_sha256"])) == 64
    assert len(str(result.metadata["ordered_type_names_sha256"])) == 64
    assert len(str(result.metadata["aggregated_spot_expression_sha256"])) == 64
    assert len(str(result.metadata["coverage_aggregation_sha256"])) == 64


def test_full_coverage_rejects_changed_prespecified_fallback() -> None:
    arguments = {
        "cell_log_expression": np.log1p([[1.0], [2.0]]),
        "abstain": np.asarray([False, True]),
        "frozen_type_index": np.asarray([0, 0]),
        "frozen_type_mean_log_expression": np.log1p([[3.0]]),
        "cell_ids": ["c1", "c2"],
        "spot_ids": ["s1"],
        "gene_names": ["g1"],
        "type_names": ["A"],
        "spot_index": np.asarray([0, 0]),
        "num_spots": 1,
        "cell_rna_mass": np.ones(2),
    }
    first = full_coverage_type_mean_aggregation(**arguments)
    digest = str(first.metadata["fallback_matrix_sha256"])
    changed = {**arguments, "frozen_type_mean_log_expression": np.log1p([[4.0]])}
    with pytest.raises(ValueError, match="prespecified hash"):
        full_coverage_type_mean_aggregation(
            **changed,
            expected_fallback_matrix_sha256=digest,
        )


def test_selective_endpoint_uses_exact_coverage_and_stable_tie_break() -> None:
    result = fixed_coverage_selective_aggregation(
        cell_log_expression=np.log1p(np.asarray([[1.0], [2.0], [3.0], [4.0], [999.0]])),
        uncertainty=np.asarray([0.2, 0.1, 0.1, 0.4, 0.0]),
        target_coverage=0.5,
        cell_ids=["z", "b", "a", "d", "unassigned"],
        spot_ids=["s1", "s2"],
        gene_names=["g1"],
        spot_index=np.asarray([0, 0, 1, 1, -1]),
        num_spots=2,
        cell_rna_mass=np.ones(5),
    )

    # The two tied least-uncertain assigned cells are selected in stable ID order.
    np.testing.assert_array_equal(result.selected_cells, [False, True, True, False, False])
    np.testing.assert_array_equal(result.eligible_cells, [True, True, True, True, False])
    np.testing.assert_allclose(result.spot_expression, np.log1p([[2.0], [3.0]]), rtol=1e-6)
    np.testing.assert_allclose(result.spot_mass, [1.0, 1.0])
    assert result.requested_coverage == pytest.approx(0.5)
    assert result.realized_coverage == pytest.approx(0.5)
    assert result.metadata["boundary_uncertainty_tie_count"] == 2
    assert len(str(result.metadata["selection_sha256"])) == 64
    assert len(str(result.metadata["uncertainty_vector_sha256"])) == 64
    assert len(str(result.metadata["rna_mass_vector_sha256"])) == 64
    assert len(str(result.metadata["aggregated_spot_expression_sha256"])) == 64
    assert len(str(result.metadata["coverage_aggregation_sha256"])) == 64


def test_selective_endpoint_never_rounds_unattainable_coverage() -> None:
    arguments = {
        "cell_log_expression": np.log1p(np.arange(1.0, 5.0)[:, None]),
        "uncertainty": np.arange(4.0),
        "cell_ids": ["a", "b", "c", "d"],
        "spot_ids": ["s1", "s2"],
        "gene_names": ["g1"],
        "spot_index": np.asarray([0, 0, 1, 1]),
        "num_spots": 2,
        "cell_rna_mass": np.ones(4),
    }
    for coverage in (0.6, 0.500001):
        with pytest.raises(ValueError, match="not exactly attainable"):
            fixed_coverage_selective_aggregation(
                **arguments,
                target_coverage=coverage,
            )


def test_truth_mask_is_shared_and_constant_predictions_score_zero() -> None:
    truth = np.asarray(
        [
            [0.0, 5.0, 3.0],
            [1.0, 5.0, 1.0],
            [2.0, 5.0, 4.0],
            [3.0, 5.0, 2.0],
        ]
    )
    spot_ids = ["s1", "s2", "s3", "s4"]
    mask = build_truth_gene_mask(
        truth,
        ["g1", "constant", "g2"],
        spot_ids=spot_ids,
    )
    np.testing.assert_array_equal(mask.mask, [True, False, True])
    assert mask.selected_gene_names == ("g1", "g2")
    assert len(mask.sha256) == 64

    method_a = truth.copy()
    method_a[:, 0] = 1.0
    method_b = truth.copy()
    method_b[:, 0] = truth[::-1, 0]
    report = evaluate_methods_on_truth_gene_mask(
        aggregations={
            "method_a": _spot_aggregation(method_a, ["g1", "constant", "g2"], spot_ids),
            "method_b": _spot_aggregation(method_b, ["g1", "constant", "g2"], spot_ids),
        },
        truth_expression=truth,
        gene_mask=mask,
        spot_ids=spot_ids,
        comparison_pairs=[("method_a", "method_b")],
    )

    assert report["schema"] == COVERAGE_EVALUATION_SCHEMA
    assert report["truth_gene_mask"]["sha256"] == mask.sha256
    for method in report["methods"].values():
        assert method["truth_gene_mask_sha256"] == mask.sha256
        assert len(method["coverage_aggregation_sha256"]) == 64
        assert method["coverage"]["endpoint"] == "fixed_coverage_selective"
        assert method["genes_evaluated"] == 2
        assert [row["gene"] for row in method["per_gene"]] == ["g1", "g2"]
    assert report["methods"]["method_a"]["per_gene"][0]["pearson"] == pytest.approx(0.0)
    assert report["methods"]["method_a"]["per_gene"][0]["spearman"] == pytest.approx(0.0)
    comparison = report["paired_comparisons"][0]
    assert comparison["truth_gene_mask_sha256"] == mask.sha256
    assert (
        comparison["left_coverage_aggregation_sha256"]
        == report["methods"]["method_a"]["coverage_aggregation_sha256"]
    )
    assert (
        comparison["right_coverage_aggregation_sha256"]
        == report["methods"]["method_b"]["coverage_aggregation_sha256"]
    )
    assert comparison["genes_evaluated"] == 2
    assert [row["gene"] for row in comparison["per_gene"]] == ["g1", "g2"]


def test_truth_gene_mask_hash_binds_order_and_selected_truth_rows() -> None:
    truth = np.asarray([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    spot_ids = ["s1", "s2", "s3"]
    original = build_truth_gene_mask(truth, ["g1", "g2"], spot_ids=spot_ids)
    repeated = build_truth_gene_mask(truth, ["g1", "g2"], spot_ids=spot_ids)
    reordered = build_truth_gene_mask(
        truth[:, ::-1],
        ["g2", "g1"],
        spot_ids=spot_ids,
    )
    first_two = build_truth_gene_mask(
        truth,
        ["g1", "g2"],
        spot_ids=spot_ids,
        spot_mask=np.asarray([True, True, False]),
    )
    last_two = build_truth_gene_mask(
        truth,
        ["g1", "g2"],
        spot_ids=spot_ids,
        spot_mask=np.asarray([False, True, True]),
    )
    assert original.sha256 == repeated.sha256
    assert original.sha256 != reordered.sha256
    assert first_two.sha256 != last_two.sha256
    np.testing.assert_array_equal(first_two.spot_mask, [True, True, False])

    prediction = truth.copy()
    prediction[2] = [100.0, 100.0]
    selected_report = evaluate_methods_on_truth_gene_mask(
        aggregations={"method": _spot_aggregation(prediction, ["g1", "g2"], spot_ids)},
        truth_expression=truth,
        gene_mask=first_two,
        spot_ids=spot_ids,
    )
    assert selected_report["methods"]["method"]["summary"]["median_gene_spearman"] == pytest.approx(
        1.0
    )

    with pytest.raises(ValueError, match="at least two selected spots"):
        build_truth_gene_mask(
            truth,
            ["g1", "g2"],
            spot_ids=spot_ids,
            spot_mask=np.asarray([True, False, False]),
        )


def test_coverage_hashes_bind_cell_order_mapping_mass_and_abstention() -> None:
    arguments = {
        "cell_log_expression": np.log1p(np.asarray([[1.0], [2.0], [3.0], [4.0]])),
        "abstain": np.asarray([False, True, False, True]),
        "frozen_type_index": np.asarray([0, 0, 1, 1]),
        "frozen_type_mean_log_expression": np.log1p([[2.0], [3.0]]),
        "cell_ids": ["c1", "c2", "c3", "c4"],
        "spot_ids": ["s1", "s2"],
        "gene_names": ["g1"],
        "type_names": ["A", "B"],
        "spot_index": np.asarray([0, 0, 1, 1]),
        "num_spots": 2,
        "cell_rna_mass": np.asarray([1.0, 2.0, 3.0, 4.0]),
    }
    original = full_coverage_type_mean_aggregation(**arguments)
    changed_mapping = full_coverage_type_mean_aggregation(
        **{**arguments, "spot_index": np.asarray([1, 0, 1, 0])}
    )
    changed_abstention = full_coverage_type_mean_aggregation(
        **{**arguments, "abstain": np.asarray([True, False, False, True])}
    )
    changed_cell_order = full_coverage_type_mean_aggregation(
        **{**arguments, "cell_ids": ["c2", "c1", "c3", "c4"]}
    )
    changed_mass = full_coverage_type_mean_aggregation(
        **{**arguments, "cell_rna_mass": np.asarray([2.0, 1.0, 3.0, 4.0])}
    )

    assert (
        original.metadata["cell_to_spot_mapping_sha256"]
        != changed_mapping.metadata["cell_to_spot_mapping_sha256"]
    )
    assert (
        original.metadata["rna_mass_vector_sha256"]
        != changed_mapping.metadata["rna_mass_vector_sha256"]
    )
    assert (
        original.metadata["abstain_mask_sha256"]
        != changed_abstention.metadata["abstain_mask_sha256"]
    )
    assert (
        original.metadata["ordered_cell_ids_sha256"]
        != changed_cell_order.metadata["ordered_cell_ids_sha256"]
    )
    assert (
        original.metadata["rna_mass_vector_sha256"]
        != changed_mass.metadata["rna_mass_vector_sha256"]
    )
    with pytest.raises(ValueError, match="cell-to-spot mapping"):
        full_coverage_type_mean_aggregation(
            **arguments,
            expected_cell_to_spot_mapping_sha256=str(
                changed_mapping.metadata["cell_to_spot_mapping_sha256"]
            ),
        )


def test_selective_hashes_bind_spot_order_and_cell_to_spot_mapping() -> None:
    arguments = {
        "cell_log_expression": np.log1p(np.asarray([[1.0], [2.0], [3.0], [4.0]])),
        "uncertainty": np.asarray([0.1, 0.2, 0.3, 0.4]),
        "target_coverage": 0.5,
        "cell_ids": ["c1", "c2", "c3", "c4"],
        "spot_ids": ["s1", "s2"],
        "gene_names": ["g1"],
        "spot_index": np.asarray([0, 0, 1, 1]),
        "num_spots": 2,
        "cell_rna_mass": np.ones(4),
    }
    original = fixed_coverage_selective_aggregation(**arguments)
    changed_mapping = fixed_coverage_selective_aggregation(
        **{**arguments, "spot_index": np.asarray([1, 0, 1, 0])}
    )
    changed_spot_order = fixed_coverage_selective_aggregation(
        **{**arguments, "spot_ids": ["s2", "s1"]}
    )

    assert (
        original.metadata["cell_to_spot_mapping_sha256"]
        != changed_mapping.metadata["cell_to_spot_mapping_sha256"]
    )
    assert (
        original.metadata["ordered_spot_ids_sha256"]
        != changed_spot_order.metadata["ordered_spot_ids_sha256"]
    )
    assert (
        original.metadata["cell_to_spot_mapping_sha256"]
        != changed_spot_order.metadata["cell_to_spot_mapping_sha256"]
    )


def test_truth_mask_rejects_different_spot_order_or_truth_values() -> None:
    truth = np.asarray([[0.0], [1.0], [2.0]])
    mask = build_truth_gene_mask(truth, ["g1"], spot_ids=["s1", "s2", "s3"])
    with pytest.raises(ValueError, match="ordered spot_ids"):
        evaluate_methods_on_truth_gene_mask(
            aggregations={"method": _spot_aggregation(truth, ["g1"], ["s1", "s2", "s3"])},
            truth_expression=truth,
            gene_mask=mask,
            spot_ids=["s2", "s1", "s3"],
        )
    changed_truth = truth.copy()
    changed_truth[1, 0] = 1.5
    with pytest.raises(ValueError, match="truth_expression differs"):
        evaluate_methods_on_truth_gene_mask(
            aggregations={"method": _spot_aggregation(changed_truth, ["g1"], ["s1", "s2", "s3"])},
            truth_expression=changed_truth,
            gene_mask=mask,
            spot_ids=["s1", "s2", "s3"],
        )


def test_scorer_requires_identity_bound_coverage_aggregations() -> None:
    truth = np.asarray([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    spot_ids = ["s1", "s2", "s3"]
    mask = build_truth_gene_mask(truth, ["g1", "g2"], spot_ids=spot_ids)
    aggregation = _spot_aggregation(truth, ["g1", "g2"], spot_ids)

    with pytest.raises(TypeError, match="provenance-bound CoverageAggregation"):
        evaluate_methods_on_truth_gene_mask(
            aggregations={"raw-array": truth},  # type: ignore[dict-item]
            truth_expression=truth,
            gene_mask=mask,
            spot_ids=spot_ids,
        )
    reordered_genes = _spot_aggregation(truth[:, ::-1], ["g2", "g1"], spot_ids)
    with pytest.raises(ValueError, match="ordered gene identities"):
        evaluate_methods_on_truth_gene_mask(
            aggregations={"reordered": reordered_genes},
            truth_expression=truth,
            gene_mask=mask,
            spot_ids=spot_ids,
        )
    missing_spot = fixed_coverage_selective_aggregation(
        cell_log_expression=truth,
        uncertainty=np.asarray([0.0, 0.1, 1.0]),
        target_coverage=2.0 / 3.0,
        cell_ids=["c1", "c2", "c3"],
        spot_ids=spot_ids,
        gene_names=["g1", "g2"],
        spot_index=np.arange(3, dtype=np.int64),
        num_spots=3,
        cell_rna_mass=np.ones(3),
    )
    with pytest.raises(ValueError, match="zero mass in a truth-scored spot"):
        evaluate_methods_on_truth_gene_mask(
            aggregations={"missing-spot": missing_spot},
            truth_expression=truth,
            gene_mask=mask,
            spot_ids=spot_ids,
        )
    with pytest.raises(TypeError):
        aggregation.metadata["coverage_aggregation_sha256"] = "0" * 64  # type: ignore[index]
