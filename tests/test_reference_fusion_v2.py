from __future__ import annotations

import inspect
from itertools import combinations, islice

import numpy as np
import pytest

from heir.evaluation import reference_fusion_v2
from heir.evaluation.reference_fusion_v2 import (
    adaptive_residual_fusion,
    build_reference_prototypes,
    fit_reference_calibrator,
    residual_fusion,
    select_fusion_alpha,
    select_reference_calibration_alpha,
)


def test_molecular_kmeans_uses_the_preregistered_completion_cap():
    assert reference_fusion_v2.MOLECULAR_KMEANS_MAXIMUM_ITERATIONS == 1_000
    parameter = inspect.signature(reference_fusion_v2._deterministic_molecular_kmeans).parameters[
        "maximum_iterations"
    ]
    assert parameter.default == reference_fusion_v2.MOLECULAR_KMEANS_MAXIMUM_ITERATIONS


def test_molecular_kmeans_default_converges_after_more_than_100_assignments(monkeypatch):
    assignments = []
    for right_cluster in islice(combinations(range(14), 7), 101):
        labels = np.zeros(14, dtype=np.int64)
        labels[list(right_cluster)] = 1
        assignments.append(labels)
    assignments.append(assignments[-1].copy())

    def run(maximum_iterations):
        iterator = iter(assignments)

        def scripted_argmin(_values, axis):
            assert axis == 1
            return next(iterator)

        monkeypatch.setattr(reference_fusion_v2.np, "argmin", scripted_argmin)
        return reference_fusion_v2._deterministic_molecular_kmeans(
            np.arange(14, dtype=np.float64)[:, None],
            np.asarray([f"cell-{index}" for index in range(14)]),
            2,
            seed=11,
            maximum_iterations=maximum_iterations,
        )

    with pytest.raises(RuntimeError, match="within 100 iterations"):
        run(100)
    _centers, labels = run(reference_fusion_v2.MOLECULAR_KMEANS_MAXIMUM_ITERATIONS)
    np.testing.assert_array_equal(labels, assignments[-1])


def test_molecular_kmeans_fails_closed_on_an_assignment_cycle(monkeypatch):
    assignments = iter(
        (
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 0, 1]),
            np.array([0, 0, 1, 1]),
        )
    )

    def alternating_argmin(_values, axis):
        assert axis == 1
        return next(assignments)

    monkeypatch.setattr(reference_fusion_v2.np, "argmin", alternating_argmin)
    with pytest.raises(RuntimeError, match="assignment cycle"):
        reference_fusion_v2._deterministic_molecular_kmeans(
            np.arange(4, dtype=np.float64)[:, None],
            np.array(["a", "b", "c", "d"]),
            2,
            seed=7,
            maximum_iterations=10,
        )


def test_reference_builder_passes_the_preregistered_completion_cap(monkeypatch):
    observed = []

    def fake_kmeans(values, _observation_ids, clusters, *, seed, maximum_iterations):
        observed.append((clusters, seed, maximum_iterations))
        labels = np.arange(len(values), dtype=np.int64) % clusters
        centers = np.vstack([values[labels == cluster].mean(axis=0) for cluster in range(clusters)])
        return centers, labels

    monkeypatch.setattr(
        reference_fusion_v2,
        "_deterministic_molecular_kmeans",
        fake_kmeans,
    )
    reference_fusion_v2.build_reference_prototypes(
        [[0.0], [1.0], [2.0], [3.0]],
        ["D"] * 4,
        ["T"] * 4,
        ["a", "b", "c", "d"],
        max_prototypes_per_type=2,
        seed=5,
    )
    assert len(observed) == 1
    assert observed[0][0] == 2
    assert observed[0][2] == reference_fusion_v2.MOLECULAR_KMEANS_MAXIMUM_ITERATIONS


def test_molecular_prototypes_resolve_states_and_ignore_input_order():
    values = np.array([[0.0], [10.0], [0.2], [9.8], [0.1], [10.1]])
    observations = np.array(["a", "b", "c", "d", "e", "f"])
    donors = np.array(["D"] * 6)
    types = np.array(["T"] * 6)
    first = build_reference_prototypes(
        values,
        donors,
        types,
        observations,
        max_prototypes_per_type=2,
        seed=4,
    )
    order = np.array([5, 2, 0, 4, 1, 3])
    second = build_reference_prototypes(
        values[order],
        donors[order],
        types[order],
        observations[order],
        max_prototypes_per_type=2,
        seed=4,
    )
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.prototype_ids, second.prototype_ids)
    np.testing.assert_allclose(np.sort(first.states[:, 0]), [0.1, 9.966666666666667])
    np.testing.assert_array_equal(np.sort(first.weights), [3.0, 3.0])


def test_duplicate_states_do_not_create_redundant_empty_prototypes():
    bank = build_reference_prototypes(
        [[1.0, 2.0]] * 5,
        ["D"] * 5,
        ["T"] * 5,
        [f"c{i}" for i in range(5)],
        max_prototypes_per_type=4,
    )
    assert bank.states.shape == (1, 2)
    assert bank.weights.tolist() == [5.0]


def test_one_prototype_is_exact_donor_type_centroid():
    bank = build_reference_prototypes(
        [[0.0], [2.0], [10.0], [14.0]],
        ["A", "A", "B", "B"],
        ["T", "T", "T", "T"],
        ["a0", "a1", "b0", "b1"],
        max_prototypes_per_type=1,
    )
    np.testing.assert_allclose(bank.states[:, 0], [1.0, 12.0])
    np.testing.assert_array_equal(bank.weights, [2.0, 2.0])


def test_calibration_is_diagonal_and_excludes_nonfit_outcomes():
    reference = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    target = np.array([[1.0, 4.0], [3.0, 7.0], [5.0, 10.0], [7.0, 13.0]])
    donors = np.array(["A", "B", "C", "heldout"])
    first = fit_reference_calibrator(reference, donors, target, donors, ["A", "B", "C"])
    altered = target.copy()
    altered[-1] = [1.0e9, -1.0e9]
    second = fit_reference_calibrator(reference, donors, altered, donors, ["A", "B", "C"])
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.target_mean, second.target_mean)
    np.testing.assert_array_equal(first.coefficients, np.diag(np.diag(first.coefficients)))
    assert first.mode == "global_diagonal"
    assert first.fit_donors == ("A", "B", "C")


def test_indication_aware_calibration_requires_and_uses_indication_identity():
    reference = np.array([[0.0], [1.0], [0.0], [1.0], [0.5]])
    target = np.array([[0.0], [1.0], [10.0], [11.0], [10.5]])
    donors = np.array(["A", "B", "C", "D", "E"])
    indications = {"A": "lung", "B": "lung", "C": "lymph", "D": "lymph", "E": "lymph"}
    fit = fit_reference_calibrator(
        reference,
        donors,
        target,
        donors,
        ["A", "B", "C", "D"],
        donor_indications=indications,
    )
    assert fit.mode == "indication_diagonal"
    with pytest.raises(ValueError, match="requires donor_ids"):
        fit.transform([[0.5]])
    transformed = fit.transform([[0.5], [0.5]], indication_ids=["lung", "lymph"])
    np.testing.assert_allclose(transformed[:, 0], [0.5, 10.5])


def test_sparse_and_absent_registered_indications_use_global_fallback():
    reference = np.array([[0.0, 0.0], [2.0, 1.0], [4.0, 2.0]], dtype=np.float64)
    target = np.array([[1.0, 0.0], [5.0, 3.0], [20.0, 8.0]], dtype=np.float64)
    donors = np.array(["A", "B", "C"])
    registrations = {
        "A": "qualified",
        "B": "qualified",
        "C": "sparse",
        "registered_without_pair": "absent",
    }
    fit = fit_reference_calibrator(
        reference,
        donors,
        target,
        donors,
        ["A", "B", "C"],
        donor_indications=registrations,
    )

    assert fit.qualified_indications == ("qualified",)
    assert fit.indication_labels == fit.qualified_indications
    assert fit.fallback_indications == ("absent", "sparse")
    assert fit.coefficients.shape == (2, 2)
    np.testing.assert_array_equal(fit.coefficients, np.diag(np.diag(fit.coefficients)))

    query = np.array([[3.0, 1.5], [3.0, 1.5]])
    expected_global = (query - fit.source_mean) @ fit.coefficients + fit.target_mean
    by_indication = fit.transform(query, indication_ids=["sparse", "absent"])
    by_donor = fit.transform(
        query,
        donor_ids=["C", "registered_without_pair"],
    )
    np.testing.assert_allclose(by_indication, expected_global)
    np.testing.assert_allclose(by_donor, expected_global)

    with pytest.raises(ValueError, match="lacks donor indications"):
        fit.transform([[3.0, 1.5]], donor_ids=["genuinely_unknown"])
    with pytest.raises(ValueError, match="no registered indications"):
        fit.transform([[3.0, 1.5]], indication_ids=["genuinely_unknown"])


def test_indication_qualification_counts_donors_not_pseudobulk_rows():
    donors = np.repeat(np.array(["A", "B", "C"]), 3)
    types = np.tile(np.array(["T1", "T2", "T3"]), 3)
    reference = np.arange(9, dtype=np.float64)[:, None]
    target = 2.0 * reference + 1.0
    fit = fit_reference_calibrator(
        reference,
        donors,
        target,
        donors,
        ["A", "B", "C"],
        reference_type_labels=types,
        target_type_labels=types,
        donor_indications={"A": "one_donor", "B": "two_donors", "C": "two_donors"},
    )

    assert fit.paired_summary_rows == 9
    assert fit.qualified_indications == ("two_donors",)
    assert fit.fallback_indications == ("one_donor",)


def test_donor_type_pseudobulks_are_paired_and_donor_equal():
    donors = np.array(["A", "A", "A", "B", "B", "B"])
    types = np.array(["T", "T", "M", "T", "M", "M"])
    reference = np.array([[0.0], [2.0], [10.0], [4.0], [12.0], [14.0]])
    target = 2.0 * reference + 1.0
    fit = fit_reference_calibrator(
        reference,
        donors,
        target,
        donors,
        ["A", "B"],
        ridge_alpha=0.01,
        reference_type_labels=types,
        target_type_labels=types,
    )
    assert fit.pairing_unit == "donor_x_type"
    assert fit.paired_summary_rows == 4
    np.testing.assert_allclose(fit.transform([[3.0], [11.0]]), [[7.0], [23.0]], atol=0.01)


def test_calibration_alpha_selection_uses_fit_donors_only():
    donors = np.array(["A", "B", "C", "outside"])
    reference = np.array([[0.0], [1.0], [2.0], [3.0]])
    target = np.array([[0.0], [2.0], [4.0], [6.0]])
    indications = {"A": "lung", "B": "lung", "C": "lymph", "outside": "brain"}
    first_alpha, first = select_reference_calibration_alpha(
        reference,
        donors,
        target,
        donors,
        ["A", "B", "C"],
        candidate_alphas=(0.01, 1.0, 100.0),
        donor_indications=indications,
    )
    altered = target.copy()
    altered[-1, 0] = -1.0e12
    second_alpha, second = select_reference_calibration_alpha(
        reference,
        donors,
        altered,
        donors,
        ["A", "B", "C"],
        candidate_alphas=(0.01, 1.0, 100.0),
        donor_indications=indications,
    )
    assert first_alpha == second_alpha
    assert first["candidate_donor_equal_mse"] == second["candidate_donor_equal_mse"]
    assert first["non_fit_donor_outcomes_used"] is False


def test_calibration_alpha_selection_scores_actual_fold_hierarchy():
    donors = np.array(["A", "B", "C", "D"])
    reference = np.array([[0.0], [1.0], [2.0], [3.0]])
    target = np.array([[0.0], [10.0], [4.0], [6.0]])
    registrations = {"A": "lung", "B": "lung", "C": "lymph", "D": "brain"}

    selected, receipt = select_reference_calibration_alpha(
        reference,
        donors,
        target,
        donors,
        donors,
        candidate_alphas=(1.0,),
        donor_indications=registrations,
    )

    assert selected == 1.0
    assert receipt["selection_weighting"] == ("indication_equal_then_donor_equal_within_indication")
    assert receipt["axis_mapping"] == "diagonal_only"
    assert receipt["final_qualified_indications"] == ["lung"]
    assert receipt["final_global_fallback_indications"] == ["brain", "lymph"]
    assert receipt["fold_hierarchical_mapping"]["A"] == {
        "qualified_indications": [],
        "global_fallback_indications": ["brain", "lung", "lymph"],
        "minimum_paired_donors_per_indication": 2,
    }
    assert receipt["fold_hierarchical_mapping"]["C"] == {
        "qualified_indications": ["lung"],
        "global_fallback_indications": ["brain", "lymph"],
        "minimum_paired_donors_per_indication": 2,
    }

    # In the A-validation fold, lung has only B and therefore must use the
    # global B/C/D map.  Recompute that one-axis ridge map independently.
    inner_source = reference[1:, 0]
    inner_target = target[1:, 0]
    source_mean = float(np.mean(inner_source))
    target_mean = float(np.mean(inner_target))
    slope = (float(np.mean((inner_source - source_mean) * (inner_target - target_mean))) + 1.0) / (
        float(np.mean(np.square(inner_source - source_mean))) + 1.0
    )
    expected_a_loss = ((reference[0, 0] - source_mean) * slope + target_mean) ** 2
    assert receipt["candidate_donor_mse"]["1"]["A"] == pytest.approx(expected_a_loss)
    per_donor = receipt["candidate_donor_mse"]["1"]
    expected_indication_equal = np.mean(
        [
            np.mean([per_donor["A"], per_donor["B"]]),
            per_donor["C"],
            per_donor["D"],
        ]
    )
    assert receipt["candidate_selection_mse"]["1"] == pytest.approx(expected_indication_equal)
    assert receipt["candidate_selection_mse"]["1"] != pytest.approx(
        receipt["candidate_donor_equal_mse"]["1"]
    )


def test_full_alpha_grid_can_select_reference_endpoint():
    image = np.array([[0.0], [0.0], [100.0]])
    reference = np.array([[2.0], [4.0], [-100.0]])
    truth = np.array([[2.0], [4.0], [1.0e9]])
    donors = np.array(["A", "B", "heldout"])
    selected, receipt = select_fusion_alpha(
        image,
        reference,
        truth,
        donor_ids=donors,
        fit_donor_ids=["A", "B"],
    )
    assert selected == 1.0
    assert receipt["candidate_alphas"] == [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    assert receipt["selection_scope"] == "fit_donors_only"
    assert receipt["non_fit_donor_outcomes_used"] is False


def test_alpha_one_and_support_shrinkage_are_valid_but_larger_values_fail():
    image = np.array([[0.0], [4.0]])
    reference = np.array([[2.0], [2.0]])
    np.testing.assert_array_equal(residual_fusion(image, reference, 1.0), reference)
    fused, row_alpha = adaptive_residual_fusion(image, reference, [1.0, 0.25], 1.0)
    np.testing.assert_allclose(row_alpha, [1.0, 0.25])
    np.testing.assert_allclose(fused, [[2.0], [3.5]])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        residual_fusion(image, reference, 1.01)
