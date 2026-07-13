from __future__ import annotations

import numpy as np
import pytest

from heir.evaluation.reliability import (
    construct_split_half_counts,
    deterministic_transcript_halves,
    feature_reliability,
    fit_target_basis,
    normalize_split_counts,
    spearman_brown,
    target_basis_reliability_ceiling,
)


def test_transcript_split_is_identity_bound_and_order_invariant() -> None:
    identifiers = np.asarray(["tx-1", "tx-2", "tx-3", "tx-4"])
    assigned = deterministic_transcript_halves(identifiers, salt="locked-study")
    reversed_assigned = deterministic_transcript_halves(
        identifiers[::-1], salt="locked-study"
    )[::-1]
    np.testing.assert_array_equal(assigned, reversed_assigned)
    with pytest.raises(ValueError, match="unique"):
        deterministic_transcript_halves(["tx-1", "tx-1"], salt="locked-study")


def test_split_count_construction_rejects_unknown_or_duplicate_identities() -> None:
    result = construct_split_half_counts(
        ["t1", "t2", "t3"],
        ["c1", "c1", "c2"],
        ["g1", "g2", "g1"],
        ["c1", "c2"],
        ["g1", "g2"],
        salt="study",
    )
    assert result.half_a.shape == (2, 2)
    assert int(result.half_a.sum() + result.half_b.sum()) == 3
    with pytest.raises(ValueError, match="unknown observations"):
        construct_split_half_counts(
            ["t1"], ["missing"], ["g1"], ["c1"], ["g1"], salt="study"
        )


def test_spearman_brown_and_feature_reliability_are_deterministic() -> None:
    corrected = spearman_brown(np.asarray([0.5, 1.0, 0.0, -0.5, np.nan]))
    np.testing.assert_allclose(corrected[:4], [2.0 / 3.0, 1.0, 0.0, 0.0])
    assert np.isnan(corrected[4])
    first = np.asarray([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    report = feature_reliability(first, first, ["a", "b"], minimum_rows=3)
    assert report["median_spearman_brown_reliability"] == pytest.approx(1.0)


def test_split_normalization_uses_the_frozen_full_library_denominator() -> None:
    counts = np.asarray([[1, 1], [2, 0]], dtype=np.uint32)
    normalized = normalize_split_counts(counts, library_sizes=[4, 4])
    np.testing.assert_allclose(normalized[0], np.log1p([2_500.0, 2_500.0]), rtol=1e-6)
    with pytest.raises(ValueError, match="below target counts"):
        normalize_split_counts(counts, library_sizes=[1, 4])


def test_target_basis_ceiling_is_fit_on_development_rows_only() -> None:
    first = np.asarray(
        [[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0], [5.0, 1.0]]
    )
    second = first.copy()
    development = np.asarray([True, True, True, True, False, False])
    report = target_basis_reliability_ceiling(
        first,
        second,
        development_mask=development,
        rank=1,
        minimum_rows=3,
    )
    assert report["fit_partition"] == "development_only"
    assert report["median_spearman_brown_reliability"] == pytest.approx(1.0)
    mean_a, basis_a = fit_target_basis(first[development] + second[development], rank=1)
    altered = first.copy()
    altered[~development] = 1_000.0
    mean_b, basis_b = fit_target_basis(altered[development] + second[development], rank=1)
    np.testing.assert_allclose(mean_a, mean_b)
    np.testing.assert_allclose(basis_a, basis_b)
