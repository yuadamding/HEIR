"""Fixed normalization, observed support, ties, and biological-unit weighting."""

import json

import numpy as np
import pytest

from heir.personalized.measurement_scores import (
    compare_measurement_normalizers,
    measurement_pair_scores,
    summarize_measurements,
)


def _arguments(n, *, patient="p1", specimen="s1", genes=("A", "B", "C")):
    return dict(
        region_ids=tuple(f"r{i}" for i in range(n)),
        gene_ids=genes,
        outcome_gene_ids=("A", "B"),
        normalization_gene_ids=("A", "B"),
        left_present=np.ones(n, bool),
        right_present=np.ones(n, bool),
        patient_id=patient,
        specimen_id=specimen,
        replicate_ids=("rep1", "rep2"),
    )


def test_outcome_subset_never_changes_fixed_normalizer_or_gene_scores():
    left = np.asarray([[1, 4, 5], [2, 3, 15], [3, 2, 25], [4, 1, 35]])
    right = np.asarray([[2, 5, 6], [3, 2, 18], [1, 4, 21], [5, 3, 30]])
    arguments = {**_arguments(4), "normalization_gene_ids": ("A", "B", "C")}
    full = measurement_pair_scores(left, right, **arguments)
    subset = measurement_pair_scores(left, right, **{**arguments, "outcome_gene_ids": ("B",)})
    assert subset["scores"]["gene_scores"][0] == full["scores"]["gene_scores"][1]
    assert subset["eligibility"] == full["eligibility"]
    assert subset["exposure_strata"]["cutpoints"] == full["exposure_strata"]["cutpoints"]
    expected = np.log1p(left[:, 1] / left.sum(1) * 1e4) - np.log1p(right[:, 1] / right.sum(1) * 1e4)
    assert subset["scores"]["normalized_log_mse"] == pytest.approx(np.mean(expected**2))
    assert len(full["scores"]["directions"]) == 2
    assert {row["source_replicate"] for row in full["scores"]["directions"]} == {"rep1", "rep2"}
    assert all(
        row["normalized_log_mse"] == full["scores"]["normalized_log_mse"]
        for row in full["scores"]["directions"]
    )


def test_normalizers_report_both_all_eligible_and_common_eligible_support():
    left = np.asarray([[0, 0, 8], [1, 4, 5], [2, 3, 6], [4, 1, 10]])
    right = np.asarray([[0, 0, 9], [2, 5, 3], [3, 2, 8], [5, 3, 12]])
    arguments = _arguments(4)
    arguments.pop("normalization_gene_ids")
    report = compare_measurement_normalizers(
        left,
        right,
        normalization_panels={"nine": ("A", "B"), "broad": ("A", "B", "C")},
        **arguments,
    )
    assert report["common_eligible_region_ids"] == ["r1", "r2", "r3"]
    assert report["normalizers"]["nine"]["all_eligible"]["scores"]["n_regions"] == 3
    broad = report["normalizers"]["broad"]
    assert broad["all_eligible"]["scores"]["n_regions"] == 4
    for view in report["normalizers"].values():
        assert (
            view["common_eligible"]["scores"]["region_ids"] == report["common_eligible_region_ids"]
        )
        assert (
            view["all_eligible"]["exposure_strata"]["cutpoints"]
            == view["common_eligible"]["exposure_strata"]["cutpoints"]
        )
    assert broad["common_eligible"]["eligibility"]["exclusions"]["outside_row_mask"] == ["r0"]


@pytest.mark.parametrize("n", [0, 1, 2])
def test_too_few_regions_keep_explicit_undefined_scores(n):
    counts = np.tile([1, 2, 3], (n, 1))
    report = measurement_pair_scores(counts, counts, **_arguments(n))
    scores = report["scores"]
    assert scores["median_gene_spearman"] is None
    assert scores["n_regions"] == n
    assert all(r["undefined_reason"] == "insufficient_regions" for r in scores["gene_scores"])
    assert scores["normalized_log_mse"] == (0 if n else None)
    json.dumps(report, allow_nan=False)


def test_constant_observed_zero_is_not_absence_or_a_favorable_omission():
    left = np.asarray([[0, 1, 4], [0, 2, 3], [0, 3, 2]])
    right = np.asarray([[0, 2, 4], [0, 3, 3], [0, 4, 2]])
    report = measurement_pair_scores(
        left, right, **{**_arguments(3), "normalization_gene_ids": ("B", "C")}
    )
    scores = report["scores"]
    assert scores["n_regions"] == 3 and scores["n_defined_genes"] == 1
    assert scores["median_gene_spearman"] is None
    assert scores["defined_gene_median_diagnostic"] == pytest.approx(1)
    assert scores["gene_scores"][0]["undefined_reason"] == "constant_both_replicates"
    assert scores["gene_scores"][0]["normalized_log_mse"] == 0


def test_absent_rows_zero_panel_exposure_and_zero_outcomes_remain_distinct():
    left = np.asarray([[np.nan, np.nan, np.nan], [0, 0, 2], [0, 2, 3], [2, 1, 3]])
    right = np.asarray([[1, 2, 3], [0, 0, 4], [0, 2, 3], [2, 1, 3]])
    report = measurement_pair_scores(
        left, right, **{**_arguments(4), "left_present": np.asarray([False, True, True, True])}
    )
    eligibility = report["eligibility"]
    assert eligibility["left_panel_counts"] == [None, 0, 2, 3]
    assert eligibility["selected_region_ids"] == ["r2", "r3"]
    assert eligibility["exclusions"]["absent_left"] == ["r0"]
    assert eligibility["exclusions"]["zero_left_panel_exposure"] == ["r1"]
    assert report["scores"]["gene_scores"][0]["n_regions"] == 2
    json.dumps(report, allow_nan=False)


def test_exposure_ties_stay_together_and_cutpoints_do_not_follow_row_selection():
    left = np.asarray([[1, 2, 0]] * 3 + [[2, 4, 0]] * 3 + [[3, 6, 0]] * 3)
    right = left * 2
    report = measurement_pair_scores(left, right, **_arguments(9))
    strata = report["exposure_strata"]["strata"]
    assert [r["scores"]["n_regions"] for r in strata] == [3, 3, 3]
    selected = measurement_pair_scores(
        left, right, **{**_arguments(9), "row_mask": np.asarray([False] * 6 + [True] * 3)}
    )
    assert selected["exposure_strata"]["cutpoints"] == report["exposure_strata"]["cutpoints"]
    assert [r["scores"]["n_regions"] for r in selected["exposure_strata"]["strata"]] == [0, 0, 3]
    tied = measurement_pair_scores(left[:3], right[:3], **_arguments(3))
    assert tied["exposure_strata"]["cutpoints"] == [3, 3]
    assert [r["scores"]["n_regions"] for r in tied["exposure_strata"]["strata"]] == [0, 0, 3]


def test_sparse_full_gene_axis_is_not_densified(monkeypatch):
    sparse = pytest.importorskip("scipy.sparse")
    counts = sparse.csr_matrix(np.arange(400).reshape(4, 100))
    original = sparse.csr_matrix.toarray

    def guarded(self, *args, **kwargs):
        assert self.shape[1] <= 2, "only requested outcomes may be densified"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", guarded)
    genes = tuple(f"g{i}" for i in range(100))
    kwargs = {
        **_arguments(4, genes=genes),
        "outcome_gene_ids": genes[:2],
        "normalization_gene_ids": genes,
    }
    result = measurement_pair_scores(counts, counts, **kwargs)
    assert result["scores"]["median_gene_spearman"] == pytest.approx(1)
    assert result["scores"]["normalized_log_mse"] == 0


def test_named_gene_axes_and_region_reordering_preserve_statistics():
    left = np.asarray([[1, 4, 5], [2, 3, 15], [3, 2, 25], [4, 1, 35]])
    right = np.asarray([[2, 5, 6], [3, 2, 18], [1, 4, 21], [5, 3, 30]])
    arguments = _arguments(4)
    expected = measurement_pair_scores(left, right, **arguments)
    rows, columns = [3, 1, 0, 2], [2, 0, 1]
    reordered = measurement_pair_scores(
        left[rows][:, columns],
        right[rows][:, columns],
        **{
            **arguments,
            "region_ids": tuple(arguments["region_ids"][i] for i in rows),
            "gene_ids": tuple(arguments["gene_ids"][i] for i in columns),
        },
    )
    for actual, wanted in zip(
        reordered["scores"]["gene_scores"], expected["scores"]["gene_scores"]
    ):
        assert actual["spearman"] == pytest.approx(wanted["spearman"])
        assert actual["normalized_log_mse"] == pytest.approx(wanted["normalized_log_mse"])


def test_patient_macro_summary_retains_singles_and_undefined_pairs():
    counts = np.asarray([[1, 4, 2], [2, 3, 2], [3, 2, 2], [4, 1, 2]])
    first = measurement_pair_scores(counts, counts, **_arguments(4))
    second = measurement_pair_scores(counts, counts, **_arguments(4, specimen="s2"))
    opposite = measurement_pair_scores(counts, counts[::-1], **_arguments(4, patient="p2"))
    single = measurement_pair_scores(
        counts,
        None,
        **{**_arguments(4, patient="p3"), "right_present": None, "replicate_ids": ("rep1",)},
    )
    summary = summarize_measurements(
        [first, second, opposite, single], patient_ids=("p1", "p2", "p3")
    )
    assert summary["n_registry_patients"] == 3
    assert summary["n_structurally_paired_patients"] == 2
    assert summary["n_single_replicate_patients"] == 1
    assert summary["mean_paired_patient_spearman"] == pytest.approx(0)
    assert summary["patients"][2]["status"] == "single_replicate"
    constant = measurement_pair_scores(
        np.tile([1, 1, 2], (4, 1)), counts, **_arguments(4, patient="p2")
    )
    undefined = summarize_measurements([first, constant, single])
    assert undefined["mean_paired_patient_spearman"] is None
    assert undefined["defined_paired_patient_mean_diagnostic"] == pytest.approx(1)
    assert undefined["mean_paired_patient_normalized_log_mse"] is not None
    missing = summarize_measurements([first], patient_ids=("p1", "p2"))
    assert missing["missing_patient_ids"] == ["p2"]
    assert missing["mean_paired_patient_spearman"] is None


def test_reject_invalid_counts_axes_masks_and_duplicate_pairs():
    counts = np.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    for changes in (
        {"region_ids": ("r1", "r1", "r2")},
        {"normalization_gene_ids": ("A",)},
        {"left_present": [1, 1, 1]},
        {"exposure_cutpoints": [10, 1]},
    ):
        with pytest.raises(ValueError):
            measurement_pair_scores(counts, counts, **{**_arguments(3), **changes})
    with pytest.raises(ValueError, match="raw integer"):
        measurement_pair_scores(counts + 0.5, counts, **_arguments(3))
    report = measurement_pair_scores(counts, counts, **_arguments(3))
    with pytest.raises(ValueError, match="duplicate"):
        summarize_measurements([report, report])
