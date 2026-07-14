from __future__ import annotations

import json

import numpy as np
import pytest

from heir.evaluation.hest_scoring import (
    holm_adjust,
    multiclass_metrics,
    score_continuous_targets,
    summarize_paired_donor_effects,
)


def test_continuous_targets_use_equal_donor_and_type_weight() -> None:
    truth = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
        ]
    )
    prediction = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],  # D1/A: R2=1, reduction=1
            [1.0, 0.0],
            [1.0, 0.0],  # D1/B: R2=0, reduction=.5
            [0.0, 0.0],
            [0.0, 0.0],  # D2/A: R2=-1, reduction=0
            [2.0, 0.0],
            [0.0, 0.0],  # D2/B: R2=-3, reduction=-1
        ]
    )
    reference = np.zeros_like(truth)
    report = score_continuous_targets(
        truth,
        prediction,
        reference,
        np.asarray(["D1"] * 4 + ["D2"] * 4),
        np.asarray(["S1"] * 4 + ["S2"] * 4),
        np.asarray(["A", "A", "B", "B"] * 2),
        target_names=("variable", "constant"),
        minimum_support=2,
    )

    variable = report["targets"]["variable"]
    assert variable["donor_type_macro_r2"] == pytest.approx(-0.75)
    assert variable["donor_type_macro_reference_error_reduction"] == pytest.approx(0.125)
    assert variable["donor_section_type_macro_r2"] == pytest.approx(-0.75)
    assert variable["donor_section_type_macro_reference_error_reduction"] == pytest.approx(
        0.125
    )
    assert variable["per_donor"]["D1"]["donor_type_r2"] == pytest.approx(0.5)
    assert variable["per_donor"]["D2"]["donor_type_r2"] == pytest.approx(-2.0)
    assert variable["support"]["evaluable_donor_type_r2_strata"] == 4

    constant = report["targets"]["constant"]
    assert constant["donor_type_macro_r2"] is None
    assert constant["donor_type_macro_reference_error_reduction"] is None
    assert report["target_macro"]["donor_type_macro_r2"] == pytest.approx(-0.75)
    json.dumps(report, allow_nan=False)


def test_continuous_targets_balance_sections_before_donors() -> None:
    # D1/S1 is perfect, D1/S2 has R2=-1, and D2/S3 is perfect.  Equal-section,
    # then equal-donor weighting is mean(mean(1, -1), 1) = 0.5.
    truth = np.asarray([0.0, 2.0, 0.0, 2.0, 0.0, 2.0])
    prediction = np.asarray([0.0, 2.0, 0.0, 0.0, 0.0, 2.0])
    report = score_continuous_targets(
        truth,
        prediction,
        np.zeros_like(truth),
        np.asarray(["D1", "D1", "D1", "D1", "D2", "D2"]),
        np.asarray(["S1", "S1", "S2", "S2", "S3", "S3"]),
        np.asarray(["A"] * 6),
        target_names=("p",),
        minimum_support=2,
    )
    score = report["targets"]["p"]
    assert score["donor_section_type_macro_r2"] == pytest.approx(0.5)
    assert score["per_donor"]["D1"]["donor_section_type_r2"] == pytest.approx(0.0)
    assert score["per_donor"]["D2"]["donor_section_type_r2"] == pytest.approx(1.0)


def test_multiclass_balanced_accuracy_and_macro_f1() -> None:
    report = multiclass_metrics(
        np.asarray(["A", "A", "B", "B"]),
        np.asarray(["A", "B", "B", "B"]),
    )
    assert report["accuracy"] == pytest.approx(0.75)
    assert report["balanced_accuracy"] == pytest.approx(0.75)
    assert report["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.8) / 2.0)
    assert report["per_class"]["A"]["recall"] == pytest.approx(0.5)
    assert report["per_class"]["B"]["precision"] == pytest.approx(2.0 / 3.0)


def test_holm_adjust_is_monotone_and_preserves_missing_values() -> None:
    adjusted = holm_adjust({"b": 0.04, "a": 0.01, "missing": np.nan, "c": 0.03})
    assert list(adjusted) == ["a", "b", "c", "missing"]
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["missing"] is None
    json.dumps(adjusted, allow_nan=False)


def test_paired_summary_has_exact_sign_flip_and_deterministic_bootstrap() -> None:
    first = summarize_paired_donor_effects(
        {"D2": 3.0, "D1": 2.0},
        {"D1": 1.0, "D2": 2.0},
        bootstrap_iterations=100,
        bootstrap_seed=9,
    )
    second = summarize_paired_donor_effects(
        {"D1": 2.0, "D2": 3.0},
        {"D2": 2.0, "D1": 1.0},
        bootstrap_iterations=100,
        bootstrap_seed=9,
    )
    assert first == second
    assert first["mean_effect"] == pytest.approx(1.0)
    assert first["positive_fraction"] == pytest.approx(1.0)
    assert first["exact_sign_flip_p"] == pytest.approx(0.25)
    assert first["bootstrap_ci_95"] == pytest.approx([1.0, 1.0])
    assert first["per_donor_effect"] == {"D1": 1.0, "D2": 1.0}
    json.dumps(first, allow_nan=False)
