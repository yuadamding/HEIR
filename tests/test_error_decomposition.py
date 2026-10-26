"""Bias/centered-error identities and the evaluator's observation weighting."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from heir.personalized.cohort_cli import replicate_scores
from heir.personalized.error_decomposition import (
    aggregate_error_decompositions,
    error_decomposition,
)
from heir.personalized.observation import ObservationOperator
from heir.personalized.schema import Prediction, SpatialTargets


def _example(predicted, observed, *, mask=None, groups=None, patient="P1", specimen="S1"):
    """Fixed normalizer mass 1e4 makes the supplied log1p outcomes exact."""
    predicted, observed = np.asarray(predicted, dtype=float), np.asarray(observed, dtype=float)
    n, g = predicted.shape
    genes = ("A", "B", *tuple(f"outcome{i}" for i in range(g)))
    cells = tuple(f"cell{i}" for i in range(n))
    ids = tuple(f"observation{i}" for i in range(n))
    expression = torch.as_tensor(np.column_stack((np.full((n, 2), 5000), np.expm1(predicted))))
    values = np.column_stack((np.full((n, 2), 5000), np.expm1(observed)))
    available = np.ones(values.shape, dtype=bool)
    if mask is not None:
        available[:, 2:] = mask
        values[~available] = np.nan
    prediction = Prediction(
        cell_ids=cells,
        gene_ids=genes,
        state_ids=("unsupported",),
        coordinates_um=torch.zeros((n, 2)),
        assignment=torch.ones((n, 1)),
        conditional_assignment=torch.empty((n, 0)),
        transfer=expression,
        log_correction=torch.zeros_like(expression),
        expression=expression,
        assignment_entropy=torch.zeros(n),
        reference_support=torch.zeros(n),
        registration_sensitivity=torch.full((n,), float("nan")),
        supervised_genes=torch.ones(len(genes), dtype=torch.bool),
        independently_evaluated_genes=torch.zeros(len(genes), dtype=torch.bool),
        patient_id=patient,
        specimen_id=specimen,
        unsupported_policy="he_only",
    )
    targets = SpatialTargets(ids, genes, values, available, patient, specimen)
    operator = ObservationOperator(cells, ids, np.arange(n), np.arange(n), np.ones(n))
    return prediction, targets, operator, tuple(groups or ("rep1",) * n)


def _decompose(example):
    prediction, targets, operator, groups = example
    return error_decomposition(
        prediction, targets, operator, groups, prediction.gene_ids[2:], ("A", "B")
    )


def _assert_identity(report):
    assert report["normalized_log_mse"] == pytest.approx(
        report["squared_bias"] + report["centered_mse"], abs=1e-12
    )


def test_analytic_bias_and_centered_error_are_distinct_population_moments():
    # First gene has error 2 everywhere; second has errors -1, 0, +1.
    example = _example([[3, 2], [4, 3], [5, 4]], [[1, 3], [2, 3], [3, 3]])
    result = _decompose(example)
    bias, spatial = result["per_gene"]
    assert bias["mean_bias"] == pytest.approx(2)
    assert bias["squared_bias"] == pytest.approx(4)
    assert bias["centered_mse"] == pytest.approx(0, abs=1e-28)
    assert bias["std_ratio"] == pytest.approx(1)
    assert spatial["mean_bias"] == pytest.approx(0, abs=1e-15)
    assert spatial["centered_mse"] == pytest.approx(2 / 3)
    assert spatial["squared_bias"] == pytest.approx(0, abs=1e-28)
    assert spatial["std_ratio"] is None
    assert spatial["std_ratio_undefined_reason"] == "constant_observed_values"
    assert result["squared_bias"] == pytest.approx(2)
    assert result["centered_mse"] == pytest.approx(1 / 3)
    assert result["normalized_log_mse"] == pytest.approx(7 / 3)
    _assert_identity(result)


def test_missing_outcomes_use_observed_entry_weights_not_equal_gene_weights():
    example = _example(
        [[3, 5], [4, 5], [5, 5]],
        [[1, 1], [2, 2], [3, 3]],
        mask=[[True, True], [True, False], [True, False]],
    )
    result = _decompose(example)
    assert [row["n"] for row in result["per_gene"]] == [3, 1]
    assert [row["weight_in_replicate"] for row in result["per_gene"]] == [0.75, 0.25]
    assert result["normalized_log_mse"] == pytest.approx((3 * 4 + 16) / 4)
    prediction, targets, operator, groups = example
    evaluator = replicate_scores(
        prediction, targets, operator, groups, prediction.gene_ids[2:], ("A", "B")
    )
    assert result["normalized_log_mse"] == pytest.approx(evaluator["normalized_log_mse"])
    _assert_identity(result)


def test_unequal_replicates_specimens_and_patients_have_exact_macro_weights():
    # P1/S1 has 3 replicate-1 observations with error 1, one replicate-2 with error 3.
    first = _decompose(
        _example([[2], [3], [4], [4]], [[1], [2], [3], [1]], groups=("1", "1", "1", "2"))
    )
    assert first["normalized_log_mse"] == pytest.approx((1 + 9) / 2)
    assert [row["weight_in_specimen"] for row in first["replicate_summaries"]] == [0.5, 0.5]
    # P1/S2 error 2, P2/S1 error 4. Patient 1 gets one half, not two thirds.
    second = _decompose(_example([[3]], [[1]], specimen="S2"))
    third = _decompose(_example([[5]], [[1]], patient="P2"))
    result = aggregate_error_decompositions([first, second, third])
    assert result["normalized_log_mse"] == pytest.approx(((5 + 4) / 2 + 16) / 2)
    assert [row["weight_in_cohort"] for row in result["specimen_weights"]] == [0.25, 0.25, 0.5]
    weighted = sum(
        weight["weight_in_cohort"] * row["weight_in_specimen"] * row["mse"]
        for report, weight in zip((first, second, third), result["specimen_weights"])
        for row in report["per_gene"]
    )
    assert weighted == pytest.approx(result["normalized_log_mse"])
    _assert_identity(result)


def test_unobserved_genes_and_replicates_are_explicit_and_match_evaluator():
    example = _example(
        [[2, 4], [3, 4], [4, 4]],
        [[1, 1], [2, 2], [3, 3]],
        mask=[[True, False], [True, False], [False, False]],
        groups=("1", "1", "2"),
    )
    result = _decompose(example)
    assert result["n_replicates"] == 2
    assert result["n_defined_replicates"] == 1
    assert result["replicate_summaries"][1]["normalized_log_mse"] is None
    assert result["replicate_summaries"][1]["weight_in_specimen"] == 0
    missing = [row for row in result["per_gene"] if row["n"] == 0]
    assert all(row["mse"] is None and row["weight_in_specimen"] == 0 for row in missing)
    assert all(row["std_ratio_undefined_reason"] == "no_observed_values" for row in missing)
    prediction, targets, operator, groups = example
    evaluator = replicate_scores(
        prediction, targets, operator, groups, prediction.gene_ids[2:], ("A", "B")
    )
    assert result["normalized_log_mse"] == pytest.approx(evaluator["normalized_log_mse"])
    _assert_identity(result)


def test_undefined_specimen_invalidates_patient_and_cohort_headline():
    complete = _decompose(_example([[2]], [[1]], patient="P1"))
    missing = _decompose(_example([[2]], [[1]], mask=[[False]], patient="P2"))
    summary = aggregate_error_decompositions([complete, missing])
    assert summary["normalized_log_mse"] is None
    assert summary["squared_bias"] is None and summary["centered_mse"] is None
    assert summary["n_patients"] == 2 and summary["n_defined_patients"] == 1
    assert summary["patients"][1]["undefined_specimens"] == ["S1"]


def test_cells_target_rows_and_gene_axes_join_by_identity():
    example = _example(
        [[2, 4], [4, 3], [3, 5], [6, 4]],
        [[1, 2], [2, 4], [3, 3], [4, 2]],
        groups=("b", "a", "b", "a"),
    )
    expected = _decompose(example)
    prediction, targets, operator, groups = example
    cells = [3, 1, 0, 2]
    columns = [3, 0, 2, 1]
    prediction = replace(
        prediction,
        cell_ids=tuple(prediction.cell_ids[i] for i in cells),
        gene_ids=tuple(prediction.gene_ids[i] for i in columns),
        expression=prediction.expression[cells][:, columns],
    )
    rows = [2, 0, 3, 1]
    targets = replace(
        targets,
        observation_ids=tuple(targets.observation_ids[i] for i in rows),
        gene_ids=tuple(targets.gene_ids[i] for i in columns),
        expression=targets.expression[rows][:, columns],
        observed=targets.observed[rows][:, columns],
    )
    actual = error_decomposition(
        prediction, targets, operator, groups, ("outcome0", "outcome1"), ("A", "B")
    )
    assert actual == expected


@pytest.mark.parametrize("groups", [("1",), ("1", ""), "12", ("1", 2)])
def test_group_labels_must_align_with_observation_rows(groups):
    prediction, targets, operator, _ = _example([[2], [3]], [[1], [2]])
    with pytest.raises(ValueError, match="replicate labels"):
        error_decomposition(prediction, targets, operator, groups, ("outcome0",), ("A", "B"))


def test_aggregation_rejects_duplicate_specimens_changed_axes_and_inconsistent_components():
    report = _decompose(_example([[2], [3]], [[1], [2]]))
    with pytest.raises(ValueError, match="unique specimen"):
        aggregate_error_decompositions([report, report])
    for changes in (
        {"normalization_gene_ids": ["B", "A"]},
        {"coordinate_units": "registered_pixel"},
        {"evaluation_gene_ids": ["another_outcome"]},
    ):
        with pytest.raises(ValueError, match="ordered gene panels, and units"):
            aggregate_error_decompositions([report, {**report, "patient_id": "P2", **changes}])
    with pytest.raises(ValueError, match="inconsistent"):
        aggregate_error_decompositions([{**report, "squared_bias": report["squared_bias"] + 1}])
