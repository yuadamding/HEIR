from __future__ import annotations

import json

import numpy as np
import pytest

from heir.evaluation.hest_measurement import (
    feature_reliability_report,
    normalize_halves,
    ordered_program_scores,
    reference_residualize_halves,
    support_threshold_audit,
)


def test_reference_means_are_fitted_separately_for_each_half() -> None:
    first = np.asarray([[2.0], [4.0], [5.0], [7.0], [9.0], [11.0]])
    second = np.asarray([[10.0], [14.0], [15.0], [17.0], [19.0], [21.0]])
    roles = np.asarray(
        ["reference", "reference", "evaluation", "evaluation", "evaluation", "evaluation"]
    )
    result = reference_residualize_halves(
        first,
        second,
        ["d1"] * 6,
        ["s1"] * 6,
        ["epithelial"] * 6,
        roles,
        minimum_support=2,
    )

    np.testing.assert_allclose(result.reference_mean_half_a[result.evaluation_mask], 3.0)
    np.testing.assert_allclose(result.reference_mean_half_b[result.evaluation_mask], 12.0)
    np.testing.assert_allclose(result.half_a[result.evaluation_mask, 0], [2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(result.half_b[result.evaluation_mask, 0], [3.0, 5.0, 7.0, 9.0])


def test_evaluation_rows_never_contribute_to_reference_means() -> None:
    first = np.asarray([[1.0], [3.0], [10.0], [20.0]])
    second = np.asarray([[2.0], [6.0], [11.0], [21.0]])
    roles = np.asarray(["reference", "reference", "evaluation", "evaluation"])
    original = reference_residualize_halves(
        first, second, ["d"] * 4, ["s"] * 4, ["t"] * 4, roles, minimum_support=2
    )
    changed_first = first.copy()
    changed_second = second.copy()
    changed_first[-1] = 1_000_000.0
    changed_second[-1] = -1_000_000.0
    changed = reference_residualize_halves(
        changed_first,
        changed_second,
        ["d"] * 4,
        ["s"] * 4,
        ["t"] * 4,
        roles,
        minimum_support=2,
    )

    np.testing.assert_array_equal(original.evaluation_mask, changed.evaluation_mask)
    np.testing.assert_allclose(
        original.reference_mean_half_a[original.evaluation_mask],
        changed.reference_mean_half_a[changed.evaluation_mask],
    )
    np.testing.assert_allclose(
        original.reference_mean_half_b[original.evaluation_mask],
        changed.reference_mean_half_b[changed.evaluation_mask],
    )
    assert changed.half_a[2, 0] == pytest.approx(original.half_a[2, 0])
    assert changed.half_b[2, 0] == pytest.approx(original.half_b[2, 0])


def test_program_scores_preserve_supplied_order_and_use_member_means() -> None:
    genes = np.asarray([[1.0, 3.0, 9.0], [2.0, 8.0, 4.0]])
    membership = np.asarray([[False, False, True], [True, True, False]])
    scores = ordered_program_scores(genes, ["second_in_alphabet", "first_in_alphabet"], membership)
    np.testing.assert_allclose(scores, [[9.0, 2.0], [4.0, 5.0]])
    with pytest.raises(ValueError, match="boolean"):
        ordered_program_scores(genes, ["program"], np.asarray([[1, 0, 0]]))


def test_reliability_reports_overall_donor_type_macro_and_per_type() -> None:
    donors = np.repeat(np.asarray(["d1", "d2"]), 12)
    fine_types = np.tile(np.repeat(np.asarray(["type_a", "type_b"]), 6), 2)
    pattern = np.tile(np.arange(6, dtype=np.float64), 4)
    first = np.column_stack((pattern, pattern))
    second = np.column_stack((pattern, -pattern))

    report = feature_reliability_report(
        first,
        second,
        ["repeatable", "anti_correlated"],
        donors,
        fine_types,
        minimum_rows=4,
    )

    assert report["overall"]["features"]["repeatable"][
        "spearman_brown_reliability"
    ] == pytest.approx(1.0)
    assert report["overall"]["features"]["anti_correlated"]["spearman_brown_reliability"] == 0.0
    repeatable = report["donor_type_macro"]["features"]["repeatable"]
    assert repeatable["median_spearman_brown_reliability"] == pytest.approx(1.0)
    assert repeatable["evaluable_donor_type_stratum_count"] == 4
    assert report["per_type"]["type_a"]["donor_macro_features"]["repeatable"][
        "evaluable_donor_count"
    ] == 2
    json.dumps(report, allow_nan=False)


def test_normalization_and_support_threshold_audit_are_strict_json() -> None:
    first, second = normalize_halves(
        np.asarray([[1, 1], [2, 0]], dtype=np.uint32),
        np.asarray([[0, 2], [1, 1]], dtype=np.uint32),
        library_sizes_half_a=[4, 4],
        library_sizes_half_b=[4, 4],
    )
    np.testing.assert_allclose(first[0], np.log1p([2_500.0, 2_500.0]), rtol=1e-6)
    np.testing.assert_allclose(second[0], np.log1p([0.0, 5_000.0]), rtol=1e-6)

    donors = np.asarray(["d1"] * 60 + ["d2"] * 20)
    sections = np.asarray(["s1"] * 60 + ["s2"] * 20)
    fine_types = np.asarray(["type_a"] * 60 + ["type_b"] * 20)
    roles = np.asarray(
        ["reference"] * 30
        + ["evaluation"] * 30
        + ["reference"] * 10
        + ["evaluation"] * 10
    )
    report = support_threshold_audit(donors, sections, fine_types, roles)
    assert report["thresholds"] == [5, 10, 20, 30]
    assert report["by_threshold"]["5"]["supported_strata"] == 2
    assert report["by_threshold"]["10"]["supported_strata"] == 2
    assert report["by_threshold"]["20"]["supported_strata"] == 1
    assert report["by_threshold"]["30"]["supported_strata"] == 1
    json.dumps(report, allow_nan=False)
