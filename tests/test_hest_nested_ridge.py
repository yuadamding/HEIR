from __future__ import annotations

import numpy as np
import pytest

from heir.evaluation.hest_nested_ridge import (
    donor_section_type_row_weights,
    donor_type_row_weights,
    fit_weighted_pca,
    fit_weighted_standardizer,
    grouped_donor_folds,
    weighted_ridge_predict_grid,
)


def _group_totals(weights: np.ndarray, *groups: np.ndarray) -> list[float]:
    keys = sorted(set(zip(*(group.tolist() for group in groups))))
    totals = []
    for key in keys:
        selected = np.asarray(
            [tuple(group[row] for group in groups) == key for row in range(len(weights))]
        )
        totals.append(float(weights[selected].sum()))
    return totals


def test_equal_total_weights_and_ridge_are_invariant_to_exact_row_duplication() -> None:
    rng = np.random.default_rng(91)
    features = rng.normal(size=(18, 4))
    targets = features @ rng.normal(size=(4, 2)) + rng.normal(scale=0.1, size=(18, 2))
    donors = np.repeat(["D1", "D2", "D3"], 6)
    labels = np.tile(np.repeat(["T1", "T2"], 3), 3)
    test = rng.normal(size=(5, 4))
    alphas = [0.01, 0.1, 1.0]

    weights = donor_type_row_weights(donors, labels)
    donor_totals = _group_totals(weights, donors)
    assert donor_totals == pytest.approx(np.repeat(donor_totals[0], len(donor_totals)))
    for donor in sorted(set(donors)):
        selected = donors == donor
        type_totals = _group_totals(weights[selected], labels[selected])
        assert type_totals == pytest.approx(np.repeat(type_totals[0], len(type_totals)))
    prediction = weighted_ridge_predict_grid(features, targets, test, alphas, weights, device="cpu")

    duplicated_features = np.repeat(features, 2, axis=0)
    duplicated_targets = np.repeat(targets, 2, axis=0)
    duplicated_donors = np.repeat(donors, 2)
    duplicated_labels = np.repeat(labels, 2)
    duplicated_weights = donor_type_row_weights(duplicated_donors, duplicated_labels)
    duplicated_prediction = weighted_ridge_predict_grid(
        duplicated_features,
        duplicated_targets,
        test,
        alphas,
        duplicated_weights,
        device="cpu",
    )
    assert prediction.shape == (len(alphas), len(test), targets.shape[1])
    assert duplicated_prediction == pytest.approx(prediction, abs=1.0e-11)


def test_hierarchical_weights_preserve_estimand_with_unequal_support() -> None:
    donors = np.asarray(["D1"] * 7 + ["D2"] * 5)
    sections = np.asarray(["S1"] * 5 + ["S2"] * 2 + ["S3"] * 5)
    labels = np.asarray(["A"] * 3 + ["B"] * 2 + ["A"] * 2 + ["A"] * 5)

    donor_type = donor_type_row_weights(donors, labels)
    donor_totals = _group_totals(donor_type, donors)
    assert donor_totals == pytest.approx(np.repeat(donor_totals[0], len(donor_totals)))
    d1_type_totals = _group_totals(donor_type[donors == "D1"], labels[donors == "D1"])
    assert d1_type_totals[0] == pytest.approx(d1_type_totals[1])

    weights = donor_section_type_row_weights(donors, sections, labels)
    assert weights.mean() == pytest.approx(1.0)
    donor_totals = _group_totals(weights, donors)
    assert donor_totals == pytest.approx(np.repeat(donor_totals[0], len(donor_totals)))
    d1 = donors == "D1"
    section_totals = _group_totals(weights[d1], sections[d1])
    assert section_totals[0] == pytest.approx(section_totals[1])
    d1s1 = d1 & (sections == "S1")
    type_totals = _group_totals(weights[d1s1], labels[d1s1])
    assert type_totals[0] == pytest.approx(type_totals[1])


def test_standardization_and_pca_use_training_rows_only() -> None:
    train = np.asarray(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 2.0],
            [2.0, 2.0, 1.0],
        ]
    )
    weights = np.asarray([1.0, 2.0, 1.0, 2.0])
    held_out = np.asarray([[3.0, 4.0, 5.0]])

    standardizer = fit_weighted_standardizer(train, weights)
    pca = fit_weighted_pca(train, 2, weights, device="cpu")
    first_standardized = standardizer.transform(held_out)
    first_projected = pca.transform(held_out)

    # Changing test rows changes their transformed values but cannot refit any
    # train-derived location, scale, component, or eigenvalue.
    changed_held_out = held_out + 10_000.0
    assert not np.allclose(standardizer.transform(changed_held_out), first_standardized)
    assert not np.allclose(pca.transform(changed_held_out), first_projected)
    assert standardizer.mean == pytest.approx(fit_weighted_standardizer(train, weights).mean)
    repeated_pca = fit_weighted_pca(train, 2, weights, device="cpu")
    assert repeated_pca.mean == pytest.approx(pca.mean)
    assert repeated_pca.components == pytest.approx(pca.components)
    assert repeated_pca.explained_variance == pytest.approx(pca.explained_variance)
    normalized = weights / weights.sum()
    centered = train - np.sum(train * normalized[:, None], axis=0)
    expected_eigenvalues = np.linalg.eigvalsh((centered * normalized[:, None]).T @ centered)
    assert pca.explained_variance == pytest.approx(expected_eigenvalues[-2:][::-1])
    for component in pca.components:
        assert component[int(np.argmax(np.abs(component)))] >= 0.0


def test_grouped_donor_folds_are_deterministic_complete_and_disjoint() -> None:
    donors = np.repeat(["D1", "D2", "D3", "D4", "D5", "D6"], [2, 5, 3, 4, 2, 6])
    first = grouped_donor_folds(donors, n_splits=3, seed=71)
    second = grouped_donor_folds(donors, n_splits=3, seed=71)
    assert len(first) == 3
    validation_rows = []
    for (train, validation), (repeated_train, repeated_validation) in zip(first, second):
        assert np.array_equal(train, repeated_train)
        assert np.array_equal(validation, repeated_validation)
        assert set(donors[train]).isdisjoint(set(donors[validation]))
        validation_rows.extend(validation.tolist())
    assert sorted(validation_rows) == list(range(len(donors)))
