"""Assay agreement from aggregated raw counts with explicit support and panels.

This module scores measured assays only. Broader denominators do not authorize
renormalizing saved predictions that lack the required output genes.
"""

from collections import defaultdict

import numpy as np

from .evaluate import _ranks


def _axis(values, name, *, empty=False):
    if isinstance(values, str):
        raise ValueError(f"{name} requires a sequence of stable IDs")
    values = tuple(values)
    if (
        (not values and not empty)
        or len(set(values)) != len(values)
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise ValueError(f"{name} requires unique nonempty stable IDs")
    return values


def _mask(values, n, name):
    result = np.asarray(values)
    if result.shape != (n,) or (result.dtype != bool and n):
        raise ValueError(f"{name} must be a boolean mask on the region axis")
    return result.astype(bool, copy=False)


def _counts(counts, present, shape):
    counts = counts.tocsr() if hasattr(counts, "tocsr") else np.asarray(counts)
    if counts.shape != shape:
        raise ValueError("raw measurement counts must match the ordered region and gene axes")
    measured = counts[present]
    values = measured.data if hasattr(measured, "tocsr") else measured
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.allclose(values, np.round(values), rtol=0, atol=1e-6)
    ):
        raise ValueError("present measurements require finite nonnegative raw integer counts")
    return counts


def _dense(values):
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def _scores(left, right, ids, genes, replicates, min_regions):
    rows = []
    for column, gene in enumerate(genes):
        x, y = left[:, column], right[:, column]
        reason = None
        if len(ids) < min_regions:
            reason = "insufficient_regions"
        elif np.ptp(x) == 0 and np.ptp(y) == 0:
            reason = "constant_both_replicates"
        elif np.ptp(x) == 0:
            reason = "constant_left_replicate"
        elif np.ptp(y) == 0:
            reason = "constant_right_replicate"
        rows.append(
            {
                "gene_id": gene,
                "n_regions": len(ids),
                "spearman": None if reason else float(np.corrcoef(_ranks(x), _ranks(y))[0, 1]),
                "normalized_log_mse": float(np.mean((x - y) ** 2)) if len(ids) else None,
                "undefined_reason": reason,
            }
        )
    defined = [row["spearman"] for row in rows if row["spearman"] is not None]
    strict = float(np.median(defined)) if len(defined) == len(genes) else None
    mse = float(np.mean((left - right) ** 2)) if len(ids) else None
    result = {
        "n_regions": len(ids),
        "region_ids": list(ids),
        "median_gene_spearman": strict,
        "defined_gene_median_diagnostic": float(np.median(defined)) if defined else None,
        "normalized_log_mse": mse,
        "n_defined_genes": len(defined),
        "n_requested_genes": len(genes),
        "headline_status": "defined" if strict is not None else "undefined_gene_correlations",
        "gene_scores": rows,
    }
    result["directions"] = (
        [
            {
                "source_replicate": source,
                "evaluation_replicate": evaluation,
                "median_gene_spearman": strict,
                "normalized_log_mse": mse,
                "interpretation": "symmetric_assay_agreement_no_fitting",
            }
            for source, evaluation in (replicates, replicates[::-1])
        ]
        if len(replicates) == 2
        else []
    )
    return result


def measurement_pair_scores(
    left_counts,
    right_counts,
    *,
    region_ids,
    gene_ids,
    outcome_gene_ids,
    normalization_gene_ids,
    left_present,
    right_present,
    patient_id,
    specimen_id,
    replicate_ids,
    row_mask=None,
    exposure_cutpoints=None,
    min_regions=3,
):
    """Normalize already aggregated counts, then score one explicitly aligned pair.

    Sparse count matrices stay sparse except for the selected outcome columns.
    Presence is independent of count values: measured zeros are retained when
    the fixed panel has positive exposure; absent rows are never observed zeros.
    For a single replicate pass ``right_counts=None``, ``right_present=None``,
    and one replicate ID. It remains an explicit unavailable pair in summaries.

    Tertile cutpoints use all panel-eligible rows BEFORE ``row_mask``. Values at
    a cutpoint go to the upper stratum, so tied exposures are never split and
    empty strata are valid. A common-support view reuses these same cutpoints.
    """
    ids = _axis(region_ids, "region_ids", empty=True)
    genes = _axis(gene_ids, "gene_ids")
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    outcomes = _axis(outcome_gene_ids, "outcome_gene_ids")
    panel = _axis(normalization_gene_ids, "normalization_gene_ids")
    replicates = _axis(replicate_ids, "replicate_ids")
    if not patient_id or not specimen_id:
        raise ValueError("measurement scores require patient and specimen identity")
    if isinstance(min_regions, bool) or not isinstance(min_regions, int) or min_regions < 2:
        raise ValueError("min_regions must be an integer of at least two")
    if len(panel) < 2 or not set((*outcomes, *panel)) <= set(genes):
        raise ValueError("outcomes and at least two normalization genes must belong to gene_ids")
    paired = right_counts is not None
    if len(replicates) != (2 if paired else 1) or (not paired and right_present is not None):
        raise ValueError("replicate IDs and presence must match the supplied replicate arrays")
    n = len(ids)
    present_left = _mask(left_present, n, "left_present")
    present_right = _mask(right_present, n, "right_present") if paired else np.zeros(n, bool)
    selected = np.ones(n, bool) if row_mask is None else _mask(row_mask, n, "row_mask")
    left = _counts(left_counts, present_left, (n, len(genes)))
    right = _counts(right_counts, present_right, (n, len(genes))) if paired else None
    columns = [gene_lookup[gene] for gene in panel]
    left_exposure = np.asarray(left[:, columns].sum(axis=1)).ravel().astype(float)
    right_exposure = (
        np.asarray(right[:, columns].sum(axis=1)).ravel().astype(float) if paired else np.zeros(n)
    )
    for exposure, present in ((left_exposure, present_left), (right_exposure, present_right)):
        if not np.isfinite(exposure[present]).all():
            raise ValueError("measurement panel exposure must be finite")
    eligible = present_left & present_right & (left_exposure > 0) & (right_exposure > 0)
    keep = eligible & selected
    exposure = np.minimum(left_exposure, right_exposure)
    if exposure_cutpoints is None:
        cutpoints = (
            np.quantile(exposure[eligible], [1 / 3, 2 / 3]).tolist() if eligible.any() else []
        )
    else:
        cutpoints = list(exposure_cutpoints)
        if (
            len(cutpoints) not in (0, 2)
            or (not cutpoints and eligible.any())
            or any(not np.isfinite(value) or value < 0 for value in cutpoints)
            or cutpoints != sorted(cutpoints)
        ):
            raise ValueError("exposure_cutpoints must be two finite ordered nonnegative values")
    indices = [gene_lookup[gene] for gene in outcomes]
    left_values = _dense(left[keep][:, indices]).astype(float)
    right_values = _dense(right[keep][:, indices]).astype(float) if paired else left_values.copy()
    left_values = np.log1p(left_values / left_exposure[keep, None] * 1e4)
    right_values = np.log1p(right_values / right_exposure[keep, None] * 1e4)
    kept_ids = tuple(np.asarray(ids)[keep].tolist())
    bins = np.searchsorted(cutpoints, exposure[keep], side="right")

    def names(mask):
        return list(np.asarray(ids)[mask])

    return {
        "schema": "heir.measurement_pair_scores.v1",
        "patient_id": patient_id,
        "specimen_id": specimen_id,
        "replicate_ids": list(replicates),
        "structurally_paired": paired,
        "outcome_gene_ids": list(outcomes),
        "normalization_gene_ids": list(panel),
        "normalization": "raw_aggregation_then_fixed_panel_cp10000_log1p_float64",
        "min_regions": min_regions,
        "eligibility": {
            "left_present": present_left.tolist(),
            "right_present": present_right.tolist(),
            "region_ids": list(ids),
            "left_panel_counts": [
                float(v) if p else None for v, p in zip(left_exposure, present_left)
            ],
            "right_panel_counts": [
                float(v) if p else None for v, p in zip(right_exposure, present_right)
            ],
            "all_eligible_region_ids": names(eligible),
            "selected_region_ids": names(keep),
            "exclusions": {
                "absent_left": names(~present_left),
                "absent_right": names(~present_right),
                "zero_left_panel_exposure": names(present_left & (left_exposure == 0)),
                "zero_right_panel_exposure": names(present_right & (right_exposure == 0)),
                "outside_row_mask": names(eligible & ~selected),
            },
        },
        "scores": _scores(left_values, right_values, kept_ids, outcomes, replicates, min_regions),
        "exposure_strata": {
            "cutpoints": cutpoints,
            "cutpoint_scope": "all_panel_eligible_regions_before_row_mask",
            "rule": "paired_minimum_panel_count_linear_quantiles_1/3_2/3_ties_to_upper_bin",
            "strata": [
                {
                    "stratum": name,
                    "scores": _scores(
                        left_values[bins == index],
                        right_values[bins == index],
                        tuple(np.asarray(kept_ids)[bins == index].tolist()),
                        outcomes,
                        replicates,
                        min_regions,
                    ),
                }
                for index, name in enumerate(("low", "middle", "high"))
            ],
        },
    }


def compare_measurement_normalizers(left_counts, right_counts, *, normalization_panels, **kwargs):
    """Retain each fixed denominator's eligible rows and their common-row comparison."""
    if not normalization_panels or any(
        not isinstance(k, str) or not k.strip() for k in normalization_panels
    ):
        raise ValueError("normalization panels require nonempty condition names")
    reports = {
        name: measurement_pair_scores(
            left_counts, right_counts, normalization_gene_ids=panel, **kwargs
        )
        for name, panel in normalization_panels.items()
    }
    common = set.intersection(
        *(set(r["eligibility"]["selected_region_ids"]) for r in reports.values())
    )
    row_mask = np.asarray([region in common for region in kwargs["region_ids"]])
    normalizers = {}
    for name, panel in normalization_panels.items():
        common_kwargs = {
            **kwargs,
            "row_mask": row_mask,
            "exposure_cutpoints": reports[name]["exposure_strata"]["cutpoints"],
        }
        normalizers[name] = {
            "all_eligible": reports[name],
            "common_eligible": measurement_pair_scores(
                left_counts, right_counts, normalization_gene_ids=panel, **common_kwargs
            ),
        }
    return {
        "common_eligible_region_ids": [r for r in kwargs["region_ids"] if r in common],
        "normalizers": normalizers,
        "interpretation": "different_denominators_define_different_measurement_endpoints",
        "saved_prediction_compatibility": "unsupported_without_all_required_panel_outputs",
    }


def summarize_measurements(reports, *, patient_ids=None):
    """Equal patients, with single-replicate and undefined paired cases kept explicit.

    Within patients, specimens are equal-weighted and assay pairs are averaged
    within a specimen. The paired headline requires every structurally paired
    patient's score; the defined-patient mean is labeled diagnostic only.
    """
    reports = list(reports)
    expected = (
        _axis(patient_ids, "patient_ids")
        if patient_ids is not None
        else tuple(sorted({r["patient_id"] for r in reports}))
    )
    if not expected or any(r["patient_id"] not in expected for r in reports):
        raise ValueError("measurement summaries need the declared registry patient IDs")
    keys = [(r["patient_id"], r["specimen_id"], tuple(sorted(r["replicate_ids"]))) for r in reports]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate specimen/replicate pair in measurement summary")
    axes = ("outcome_gene_ids", "normalization_gene_ids", "min_regions")
    if reports and any(any(r[k] != reports[0][k] for k in axes) for r in reports):
        raise ValueError(
            "measurement summaries require matching panels and region eligibility rule"
        )

    def complete_mean(values):
        return float(np.mean(values)) if values and all(v is not None for v in values) else None

    patients = []
    for patient in expected:
        selected = [r for r in reports if r["patient_id"] == patient]
        paired = [r for r in selected if r["structurally_paired"]]
        specimens = defaultdict(list)
        for report in paired:
            specimens[report["specimen_id"]].append(report)
        row = {
            "patient_id": patient,
            "structurally_paired": bool(paired) if selected else None,
            "status": "paired" if paired else "single_replicate" if selected else "missing_report",
            "n_specimens": len({r["specimen_id"] for r in selected}),
            "n_pairs": len(paired),
        }
        for metric in ("median_gene_spearman", "normalized_log_mse"):
            row[metric] = complete_mean(
                [
                    complete_mean([r["scores"][metric] for r in values])
                    for values in specimens.values()
                ]
            )
        patients.append(row)
    paired = [r for r in patients if r["structurally_paired"]]
    defined = [r["median_gene_spearman"] for r in paired if r["median_gene_spearman"] is not None]
    missing = [r["patient_id"] for r in patients if r["status"] == "missing_report"]
    return {
        "n_registry_patients": len(expected),
        "n_structurally_paired_patients": len(paired),
        "n_single_replicate_patients": sum(r["status"] == "single_replicate" for r in patients),
        "n_defined_paired_patients": len(defined),
        "missing_patient_ids": missing,
        "mean_paired_patient_spearman": complete_mean([r["median_gene_spearman"] for r in paired])
        if not missing
        else None,
        "defined_paired_patient_mean_diagnostic": float(np.mean(defined)) if defined else None,
        "mean_paired_patient_normalized_log_mse": complete_mean(
            [r["normalized_log_mse"] for r in paired]
        )
        if not missing
        else None,
        "patients": patients,
        "aggregation": "equal_patient_then_equal_paired_specimen_then_equal_assay_pair",
        "headline_policy": (
            "all_structurally_paired_patients_must_be_defined_single_replicates_retained_separately"
        ),
    }
