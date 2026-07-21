from __future__ import annotations

import numpy as np
import pytest

from heir.evaluation import registration_sensitivity_v2 as sensitivity


def test_paired_gain_uses_fixed_coordinate_rank_denominator() -> None:
    truth = np.asarray([[1.0, 2.0], [0.0, 0.0]])
    focal = np.asarray([[1.0, 1.0], [1.0, 0.0]])
    comparator = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    gain = sensitivity.paired_standardized_gain(truth, focal, comparator, rank=2)
    assert np.allclose(gain, [2.0, 1.5])


def test_hierarchy_weights_types_and_donors_equally() -> None:
    gains = np.asarray([1.0] * 5 + [3.0] * 10 + [9.0] * 5)
    donors = np.asarray(["D1"] * 15 + ["D2"] * 5)
    labels = np.asarray(["A"] * 5 + ["B"] * 10 + ["A"] * 5)
    summary = sensitivity.hierarchical_gain_summary(
        gains,
        donors,
        labels,
        np.ones(len(gains), dtype=bool),
        bootstrap_resamples=1_000,
    )
    assert summary["donor_effects"] == {"D1": 2.0, "D2": 9.0}
    assert summary["mean_paired_gain"] == 5.5
    assert summary["supported_donor_type_strata"] == 3


def test_fewer_than_five_cells_does_not_form_a_sensitivity_stratum() -> None:
    summary = sensitivity.hierarchical_gain_summary(
        np.arange(9, dtype=float),
        np.asarray(["D1"] * 4 + ["D2"] * 5),
        np.asarray(["A"] * 9),
        np.ones(9, dtype=bool),
        bootstrap_resamples=1_000,
    )
    assert summary["supported_donors"] == 1
    assert summary["supported_strata_keys"] == [["D2", "A"]]


def _noninferiority_fixture(quality_shift: float):
    donors = np.repeat([f"D{index:02d}" for index in range(12)], 20)
    labels = np.tile(np.repeat(["A", "B"], 10), 12)
    rows = len(donors)
    truth = np.zeros((rows, 2), dtype=float)
    comparator = np.ones((rows, 2), dtype=float)
    focal = np.ones((rows, 2), dtype=float)
    quality = np.tile(
        np.asarray([True] * 5 + [False] * 5 + [True] * 5 + [False] * 5),
        12,
    )
    # Comparator error is 2. A focal coordinate of sqrt(1 - g) gives gain g
    # after dividing the two-coordinate error difference by rank two.
    focal[quality] = np.sqrt(1.0 - quality_shift)
    all_rows = np.ones(rows, dtype=bool)
    best = np.zeros(rows, dtype=bool)
    best[::4] = True
    best &= quality
    return truth, focal, comparator, donors, labels, quality, all_rows, best


def test_noninferiority_reuses_quality_supported_strata_and_passes() -> None:
    values = _noninferiority_fixture(0.04)
    result = sensitivity.evaluate_registration_noninferiority(
        *values[:5],
        rank=2,
        quality_mask=values[5],
        all_rows_mask=values[6],
        best_mask=values[7],
        included_donor_ids=sorted(set(values[3].tolist())),
        comparison_id="A_star_vs_N0",
        bootstrap_resamples=1_000,
    )
    assert result["selected_quality"]["supported_donors"] == 12
    assert result["selected_quality"]["supported_strata_keys"] == (
        result["all_rows_same_supported_donor_type_strata"]["supported_strata_keys"]
    )
    assert result["noninferiority"]["passed"] is True
    assert result["absolute_best_descriptive"]["formal_primary_band"] is False


def test_noninferiority_fails_when_quality_loss_exceeds_margin() -> None:
    values = _noninferiority_fixture(-0.08)
    result = sensitivity.evaluate_registration_noninferiority(
        *values[:5],
        rank=2,
        quality_mask=values[5],
        all_rows_mask=values[6],
        best_mask=values[7],
        included_donor_ids=sorted(set(values[3].tolist())),
        comparison_id="N3_vs_N5",
        bootstrap_resamples=1_000,
    )
    assert result["noninferiority"]["estimate"] < -0.01
    assert result["noninferiority"]["passed"] is False


def test_masks_must_be_nested_within_all_rows() -> None:
    values = _noninferiority_fixture(0.04)
    all_rows = values[6].copy()
    all_rows[0] = False
    with pytest.raises(ValueError, match="nested"):
        sensitivity.evaluate_registration_noninferiority(
            *values[:5],
            rank=2,
            quality_mask=values[5],
            all_rows_mask=all_rows,
            best_mask=values[7],
            included_donor_ids=sorted(set(values[3].tolist())),
            comparison_id="A_star_vs_N0",
            bootstrap_resamples=1_000,
        )


def test_empty_supported_quality_strata_fail_closed_without_crashing() -> None:
    values = _noninferiority_fixture(0.04)
    quality = np.zeros_like(values[5])
    result = sensitivity.evaluate_registration_noninferiority(
        *values[:5],
        rank=2,
        quality_mask=quality,
        all_rows_mask=values[6],
        best_mask=np.zeros_like(quality),
        included_donor_ids=sorted(set(values[3].tolist())),
        comparison_id="A_star_vs_N0",
        bootstrap_resamples=1_000,
    )
    assert result["selected_quality"]["evaluable"] is False
    assert result["noninferiority"]["passed"] is False


@pytest.mark.parametrize(
    ("comparison_id", "band_id"),
    [("post_hoc_comparison", "Q3"), ("A_star_vs_N0", "Q2")],
)
def test_unregistered_comparison_or_quality_band_is_rejected(
    comparison_id: str, band_id: str
) -> None:
    values = _noninferiority_fixture(0.04)
    with pytest.raises(ValueError, match="frozen"):
        sensitivity.evaluate_registration_noninferiority(
            *values[:5],
            rank=2,
            quality_mask=values[5],
            all_rows_mask=values[6],
            best_mask=values[7],
            included_donor_ids=sorted(set(values[3].tolist())),
            comparison_id=comparison_id,
            selected_quality_band_id=band_id,
            bootstrap_resamples=1_000,
        )
