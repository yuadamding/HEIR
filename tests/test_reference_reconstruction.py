"""Fixed-panel local reference fits and evaluation-target-free replicate reuse."""

import inspect
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from heir.personalized.observation import ObservationOperator, normalized_log_expression
from heir.personalized.reconstruction import (
    fit_reference_mixtures,
    predict_reconstruction,
    save_reconstruction,
)
from heir.personalized.schema import RNAReference, SpatialTargets

GENES = ("A", "B", "C")


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reference(values=None, proportions=None):
    values = np.asarray(values if values is not None else [[12, 2, 1], [1, 9, 3], [2, 1, 11]])
    n = len(values)
    return RNAReference(
        state_ids=tuple(f"state{i}" for i in range(n)),
        gene_ids=GENES,
        expression=values,
        embedding=np.log1p(values),
        proportions=np.asarray(proportions if proportions is not None else [1 / n] * n),
        state_types=("type",) * n,
        variance=np.ones(values.shape),
        cell_counts=np.ones(n),
        embedding_gene_ids=GENES,
    )


def targets(values, *, observed=None):
    values = np.asarray(values, dtype=float)
    return SpatialTargets(
        observation_ids=tuple(f"source:{i}" for i in range(len(values))),
        gene_ids=GENES,
        expression=values,
        observed=np.ones(values.shape, dtype=bool) if observed is None else observed,
        patient_id="P1",
        specimen_id="S1",
    )


def fit(ref, target, **kwargs):
    return fit_reference_mixtures(
        ref, target, gene_ids=GENES, normalization_gene_ids=GENES, **kwargs
    )


def operator(observation_ids, regions, *, cell_ids=None):
    cells = cell_ids or tuple(dict.fromkeys(regions))
    return ObservationOperator(
        cell_ids=cells,
        observation_ids=tuple(observation_ids),
        rows=np.arange(len(observation_ids)),
        columns=np.array([cells.index(region) for region in regions]),
        weights=np.ones(len(observation_ids)),
    )


def test_known_mixtures_simplex_exact_linear_reference_and_normalized_objective():
    ref = reference(proportions=[0.7, 0.2, 0.1])
    expected_q = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]])
    target = targets(expected_q @ ref.expression * np.array([[1], [100], [0.05]]))
    result = fit(ref, target, steps=500)
    assert np.all(result.q >= 0)
    np.testing.assert_allclose(result.q.sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(result.expression, result.q @ ref.expression, atol=1e-14)
    np.testing.assert_allclose(result.q, expected_q, atol=1e-6)
    truth = normalized_log_expression(
        torch.tensor(target.expression),
        torch.tensor(target.observed),
        normalization_indices=(0, 1, 2),
    ).numpy()
    np.testing.assert_allclose(
        result.final_loss, ((result.normalized_expression - truth) ** 2).mean(1)
    )
    assert np.all(result.final_loss <= result.initial_loss)
    assert result.metadata["max_simplex_error"] < 1e-14
    assert result.metadata["target_informed"] is True
    assert result.metadata["certified_bound"] is False
    assert result.metadata["he_mapper"] is result.metadata["expression_correction"] is False


def test_named_gene_axes_and_unrelated_rows_do_not_change_a_regions_fit():
    ref = reference(proportions=[0.7, 0.2, 0.1])
    target = targets([[8, 2, 5], [1, 12, 6]])
    result = fit(ref, target, steps=70)
    order = [2, 0, 1]
    permuted_ref = replace(
        ref,
        gene_ids=tuple(GENES[i] for i in order),
        expression=ref.expression[:, order],
        variance=ref.variance[:, order],
    )
    permuted_target = replace(
        target,
        gene_ids=tuple(GENES[i] for i in order),
        expression=target.expression[:, order],
        observed=target.observed[:, order],
    )
    reordered = fit(permuted_ref, permuted_target, steps=70)
    np.testing.assert_array_equal(result.q, reordered.q)
    np.testing.assert_array_equal(result.expression, reordered.expression)
    row = replace(
        target,
        observation_ids=target.observation_ids[:1],
        expression=target.expression[:1],
        observed=target.observed[:1],
    )
    alone = fit(ref, row, steps=70)
    np.testing.assert_allclose(result.q[:1], alone.q, atol=1e-14)
    np.testing.assert_allclose(result.final_loss[:1], alone.final_loss, atol=1e-14)


def test_fixed_panel_normalization_and_missing_outcomes_do_not_renormalize():
    ref = reference()
    target = targets([[8, 4, np.nan], [2, 8, 0]], observed=np.array([[1, 1, 0], [1, 1, 1]], bool))
    arguments = dict(gene_ids=("C", "A"), normalization_gene_ids=("A", "B"), steps=80)
    result = fit_reference_mixtures(ref, target, **arguments)
    assert result.gene_ids == ("C", "A", "B")
    expression = result.expression
    expected = np.log1p(expression / expression[:, 1:].sum(axis=1, keepdims=True) * 1e4)
    np.testing.assert_allclose(result.normalized_expression, expected)
    changed_missing = replace(target, expression=np.array([[8, 4, 1e50], [2, 8, 0]]))
    repeated = fit_reference_mixtures(ref, changed_missing, **arguments)
    np.testing.assert_array_equal(result.q, repeated.q)
    truth = np.log1p(np.array([[0, 8], [0, 2]]) / np.array([[12], [10]]) * 1e4)
    errors = (expected[:, :2] - truth) ** 2
    np.testing.assert_allclose(result.final_loss, [errors[0, 1], errors[1].mean()])


def test_observed_zeros_preserved_and_invalid_normalization_rejected():
    ref = reference([[8, 2, 0], [1, 9, 0]])
    result = fit(ref, targets([[1, 9, 0], [8, 0, 2]]), steps=40)
    np.testing.assert_array_equal(result.expression[:, 2], 0)
    np.testing.assert_array_equal(result.normalized_expression[:, 2], 0)
    assert result.metadata["zero_reference_gene_ids"] == ["C"]
    assert result.final_loss[1] > 1
    with pytest.raises(ValueError, match="denominator"):
        fit(ref, targets([[0, 0, 0]]), steps=2)
    with pytest.raises(ValueError, match="incomplete normalization"):
        fit(ref, targets([[1, 2, np.nan]], observed=np.array([[1, 1, 0]], bool)), steps=2)
    with pytest.raises(ValueError, match="zero expression across"):
        fit(reference([[0, 0, 0]]), targets([[1, 2, 3]]), steps=2)


def test_best_over_all_iterates_is_retained_and_terminal_loss_reported_honestly():
    result = fit(reference(), targets([[7, 5, 3], [2, 6, 7]]), steps=25, learning_rate=2)
    assert np.any(result.start_best_loss < result.start_final_loss)
    assert np.any(result.selected_step < 25)
    rows = np.arange(len(result.observation_ids))
    np.testing.assert_array_equal(
        result.final_loss, result.start_best_loss[result.selected_start, rows]
    )
    np.testing.assert_array_equal(
        result.selected_step, result.start_best_step[result.selected_start, rows]
    )
    assert np.all(result.final_loss <= result.start_initial_loss.min(axis=0))
    assert result.metadata["optimizer_status"] == "fixed_budget_completed"
    assert result.metadata["steps_completed_per_start"] == 25
    assert result.metadata["convergence_assessed"] is False
    fixed = fit(reference([[2, 3, 4]]), targets([[4, 3, 2]]), steps=5)
    np.testing.assert_array_equal(fixed.selected_step, 0)
    np.testing.assert_array_equal(fixed.final_loss, fixed.initial_loss)


def test_cross_replicate_reuses_source_region_fits_without_evaluation_counts():
    target = targets([[8, 2, 5], [1, 12, 6], [2, 4, 10]])
    result = fit(reference(), target, steps=30)
    source = operator(target.observation_ids, ("r0", "r1", "r2"))
    # Sparse triplets and cell-axis positions differ; named region correspondence remains.
    source = replace(source, rows=source.rows[::-1], columns=source.columns[::-1])
    evaluation = operator(("B:2", "B:0"), ("r2", "r0"), cell_ids=("r1", "r0", "r2"))
    cross = predict_reconstruction(result, source, evaluation, patient_id="P1", specimen_id="S1")
    np.testing.assert_array_equal(cross, result.expression[[2, 0]])
    assert "targets" not in inspect.signature(predict_reconstruction).parameters
    # Evaluation expression is never accepted by this geometry-only boundary.
    for arbitrary_evaluation_counts in ([[1, 1, 1], [1, 10000, 0]], [[99, 2, 3], [2, 1, 90]]):
        targets(arbitrary_evaluation_counts)
        np.testing.assert_array_equal(
            predict_reconstruction(result, source, evaluation, patient_id="P1", specimen_id="S1"),
            cross,
        )
    for patient, specimen in (("P2", "S1"), ("P1", "S2")):
        with pytest.raises(ValueError, match="identity mismatch"):
            predict_reconstruction(
                result, source, evaluation, patient_id=patient, specimen_id=specimen
            )
    duplicate = operator(target.observation_ids, ("r0", "r0", "r2"))
    with pytest.raises(ValueError, match="duplicate regions"):
        predict_reconstruction(result, duplicate, evaluation, patient_id="P1", specimen_id="S1")
    unseen = operator(("B:9",), ("r9",))
    with pytest.raises(ValueError, match="absent from source"):
        predict_reconstruction(result, source, unseen, patient_id="P1", specimen_id="S1")
    with pytest.raises(ValueError, match="unit-weight"):
        predict_reconstruction(
            result,
            replace(source, weights=source.weights * 2),
            evaluation,
            patient_id="P1",
            specimen_id="S1",
        )


def test_saved_reconstruction_is_numeric_bound_diagnostic_and_exclusive(tmp_path):
    ref = reference(proportions=[1, 0, 0])
    target = targets([[8, 2, 5], [1, 12, 6]])
    source_values = target.expression.copy()
    result = fit(ref, target, steps=10)
    path = tmp_path / "reconstruction.npz"
    save_reconstruction(result, path)
    with np.load(path, allow_pickle=False) as saved:
        assert saved["observation_ids"].tolist() == list(target.observation_ids)
        assert saved["gene_ids"].tolist() == list(GENES)
        assert str(saved["patient_id"]) == "P1"
        np.testing.assert_array_equal(saved["q"], result.q)
        np.testing.assert_array_equal(saved["reference_expression"], ref.expression)
        assert all(saved[key].dtype.kind != "O" for key in saved.files)
        metadata = json.loads(str(saved["metadata_json"]))
        assert metadata["schema"] == "heir.reference_reconstruction.v1"
        assert metadata["deployable_prediction"] is False
        assert metadata["random_initialization"] is False
    with pytest.raises(FileExistsError):
        save_reconstruction(result, path)
    np.testing.assert_array_equal(target.expression, source_values)


@pytest.mark.parametrize(
    "parameter,value", [("steps", 0), ("steps", True), ("learning_rate", 0), ("seed", -1)]
)
def test_invalid_optimizer_budget_rejected(parameter, value):
    with pytest.raises(ValueError, match="reconstruction"):
        fit(reference(), targets([[1, 2, 3]]), **{parameter: value})
