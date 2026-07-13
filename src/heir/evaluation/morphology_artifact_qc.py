"""Shared scientific QC reports for morphology validation artifacts.

The functions in this module are the single production implementation used to
construct reference/evaluation balance and locked-measurement audit evidence.
They do not select thresholds, genes, fine types, donors, or rows.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from heir.evaluation.reliability import feature_reliability, normalize_split_counts

__all__ = [
    "locked_measurement_audit_report",
    "reference_evaluation_balance_report",
]


def _standardized_mean_difference(reference: np.ndarray, evaluation: np.ndarray) -> np.ndarray:
    if reference.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    pooled = np.sqrt((reference.var(axis=0) + evaluation.var(axis=0)) / 2.0)
    pooled = np.maximum(pooled, 1.0e-8)
    return (evaluation.mean(axis=0) - reference.mean(axis=0)) / pooled


def reference_evaluation_balance_report(
    values: Mapping[str, np.ndarray],
    reference: np.ndarray,
    evaluation: np.ndarray,
    donors: np.ndarray,
    labels: np.ndarray,
    type_names: tuple[str, ...],
    feature_matrix: np.ndarray,
    feature_names: tuple[str, ...],
    continuous_threshold: Optional[float],
    categorical_threshold: Optional[float],
) -> Mapping[str, object]:
    strata = []
    maximum = 0.0
    feature_maxima = np.zeros(feature_matrix.shape[1], dtype=np.float64)
    categorical_maxima = {
        name: 0.0 for name in ("section_ids", "disease_states", "site_ids", "batch_ids")
    }
    for donor in sorted(set(donors[evaluation].tolist())):
        donor_evaluation = evaluation & (donors == donor)
        for section_id in sorted(set(values["section_ids"][donor_evaluation].astype(str).tolist())):
            section = values["section_ids"].astype(str) == section_id
            for type_index, type_name in enumerate(type_names):
                local_reference = reference & (donors == donor) & section & (labels == type_index)
                local_evaluation = evaluation & (donors == donor) & section & (labels == type_index)
                if not local_reference.any() or not local_evaluation.any():
                    continue
                differences = _standardized_mean_difference(
                    feature_matrix[local_reference], feature_matrix[local_evaluation]
                )
                local_maximum = float(np.max(np.abs(differences))) if len(differences) else 0.0
                maximum = max(maximum, local_maximum)
                if len(differences):
                    feature_maxima = np.maximum(feature_maxima, np.abs(differences))
                local_categorical = {}
                for name in categorical_maxima:
                    values_by_name = values[name].astype(str)
                    levels = sorted(
                        set(values_by_name[local_reference].tolist())
                        | set(values_by_name[local_evaluation].tolist())
                    )
                    reference_fraction = np.asarray(
                        [np.mean(values_by_name[local_reference] == level) for level in levels]
                    )
                    evaluation_fraction = np.asarray(
                        [np.mean(values_by_name[local_evaluation] == level) for level in levels]
                    )
                    total_variation = float(
                        0.5 * np.abs(reference_fraction - evaluation_fraction).sum()
                    )
                    local_categorical[name] = total_variation
                    categorical_maxima[name] = max(categorical_maxima[name], total_variation)
                strata.append(
                    {
                        "donor_id": donor,
                        "section_id": section_id,
                        "fine_type_id": type_name,
                        "reference_cells": int(local_reference.sum()),
                        "evaluation_cells": int(local_evaluation.sum()),
                        "maximum_absolute_standardized_mean_difference": local_maximum,
                        "categorical_total_variation": local_categorical,
                    }
                )
    categorical = {}
    for name in ("section_ids", "disease_states", "site_ids", "batch_ids"):
        array = values[name].astype(str)
        categorical[name] = {
            "reference_counts": {
                value: int(np.count_nonzero(reference & (array == value)))
                for value in sorted(set(array[reference].tolist()))
            },
            "evaluation_counts": {
                value: int(np.count_nonzero(evaluation & (array == value)))
                for value in sorted(set(array[evaluation].tolist()))
            },
        }
    return {
        "continuous_feature_names": list(feature_names),
        "maximum_absolute_standardized_mean_difference_by_feature": {
            name: float(value) for name, value in zip(feature_names, feature_maxima)
        },
        "maximum_absolute_standardized_mean_difference": maximum,
        "maximum_allowed_absolute_standardized_mean_difference": continuous_threshold,
        "maximum_categorical_total_variation_by_field": categorical_maxima,
        "maximum_categorical_total_variation": max(categorical_maxima.values()),
        "maximum_allowed_categorical_total_variation": categorical_threshold,
        "strata": strata,
        "categorical_distributions": categorical,
        "pass": (
            None
            if continuous_threshold is None or categorical_threshold is None
            else bool(
                maximum <= continuous_threshold
                and max(categorical_maxima.values()) <= categorical_threshold
            )
        ),
    }


def locked_measurement_audit_report(
    *,
    contract: Mapping[str, object],
    donor_ids: np.ndarray,
    section_ids: np.ndarray,
    fine_type_ids: np.ndarray,
    locked_donors: Sequence[str],
    supported_types: Sequence[str],
    planned_stratum_ids: Sequence[str],
    gene_ids: Sequence[str],
    half_a_counts: np.ndarray,
    half_b_counts: np.ndarray,
    half_a_library_sizes: np.ndarray,
    half_b_library_sizes: np.ndarray,
    source_locked_measurement_qc_pass: np.ndarray,
    target_qc_pass: np.ndarray,
    registration_qc_pass: np.ndarray,
    segmentation_qc_pass: np.ndarray,
    crop_qc_pass: np.ndarray,
    annotation_nucleus_um: np.ndarray,
    annotation_cell_um: np.ndarray,
    cell_nucleus_um: np.ndarray,
    nucleus_area_um2: np.ndarray,
    nearest_neighbor_um: np.ndarray,
    nucleus_inside_cell: np.ndarray,
    cell_area_um2: np.ndarray,
    crop_ids: Sequence[str],
    crop_padding_fractions: np.ndarray,
) -> Mapping[str, object]:
    """Audit locked measurement quality without selecting genes, types, or thresholds."""

    locked = np.isin(donor_ids, np.asarray(tuple(locked_donors)))
    if not locked.any():
        raise ValueError("confirmatory source lacks locked rows for measurement audit")
    nucleus_diameter = 2.0 * np.sqrt(nucleus_area_um2 / np.pi)
    area_ratio = nucleus_area_um2 / np.maximum(cell_area_um2, np.finfo(float).eps)
    maximum_registration_outliers = float(contract["maximum_registration_outlier_fraction"])

    def summarize_threshold(
        values: np.ndarray,
        selected: np.ndarray,
        *,
        maximum: float,
        maximum_outliers: float,
    ) -> tuple[Mapping[str, object], np.ndarray]:
        valid = np.isfinite(values) & (values >= 0.0)
        row_pass = valid & (values <= maximum)
        valid_values = values[selected & valid]
        p95 = float(np.quantile(valid_values, 0.95)) if len(valid_values) else None
        outlier_fraction = float(np.mean(~row_pass[selected])) if selected.any() else 1.0
        report = {
            "rows": int(np.count_nonzero(selected)),
            "p95": p95,
            "maximum_allowed_p95": float(maximum),
            "outlier_fraction": outlier_fraction,
            "maximum_allowed_outlier_fraction": float(maximum_outliers),
            "pass": bool(
                selected.any()
                and np.count_nonzero(selected & valid) == np.count_nonzero(selected)
                and p95 is not None
                and p95 <= maximum
                and outlier_fraction <= maximum_outliers
            ),
        }
        return report, row_pass

    def absolute_metric(
        values: np.ndarray, maximum: float
    ) -> tuple[Mapping[str, object], np.ndarray]:
        overall, row_pass = summarize_threshold(
            values,
            locked,
            maximum=maximum,
            maximum_outliers=maximum_registration_outliers,
        )
        by_section = {}
        for section_id in sorted(set(section_ids[locked].astype(str).tolist())):
            selected = locked & (section_ids.astype(str) == section_id)
            by_section[section_id], _ = summarize_threshold(
                values,
                selected,
                maximum=maximum,
                maximum_outliers=maximum_registration_outliers,
            )
        return {
            **overall,
            "by_section": by_section,
            "pass": bool(overall["pass"] and all(row["pass"] for row in by_section.values())),
        }, row_pass

    def relative_metric(
        errors: np.ndarray, scales: np.ndarray, maximum: float
    ) -> tuple[Mapping[str, object], np.ndarray]:
        valid_scale = np.isfinite(scales) & (scales > 0.0)
        overall_ratios = np.full(len(errors), np.nan, dtype=np.float64)
        if np.any(locked & valid_scale):
            overall_ratios[locked] = errors[locked] / float(np.median(scales[locked & valid_scale]))
        overall, _ = summarize_threshold(
            overall_ratios,
            locked,
            maximum=maximum,
            maximum_outliers=maximum_registration_outliers,
        )
        section_ratios = np.full(len(errors), np.nan, dtype=np.float64)
        by_section = {}
        for section_id in sorted(set(section_ids[locked].astype(str).tolist())):
            selected = locked & (section_ids.astype(str) == section_id)
            selected_scale = selected & valid_scale
            if selected_scale.any():
                section_ratios[selected] = errors[selected] / float(
                    np.median(scales[selected_scale])
                )
            by_section[section_id], _ = summarize_threshold(
                section_ratios,
                selected,
                maximum=maximum,
                maximum_outliers=maximum_registration_outliers,
            )
        row_pass = np.isfinite(section_ratios) & (section_ratios <= maximum)
        return {
            **overall,
            "normalization_denominator": "median_geometry_scale_um",
            "by_section": by_section,
            "pass": bool(overall["pass"] and all(row["pass"] for row in by_section.values())),
        }, row_pass

    annotation_nucleus_report, annotation_nucleus_pass = absolute_metric(
        annotation_nucleus_um, float(contract["maximum_annotation_nucleus_p95_um"])
    )
    annotation_cell_report, annotation_cell_pass = absolute_metric(
        annotation_cell_um, float(contract["maximum_annotation_cell_p95_um"])
    )
    cell_nucleus_report, cell_nucleus_pass = absolute_metric(
        cell_nucleus_um, float(contract["maximum_cell_nucleus_p95_um"])
    )
    diameter_report, diameter_pass = relative_metric(
        annotation_nucleus_um,
        nucleus_diameter,
        float(contract["maximum_registration_nucleus_diameter_ratio_p95"]),
    )
    neighbor_report, neighbor_pass = relative_metric(
        annotation_nucleus_um,
        nearest_neighbor_um,
        float(contract["maximum_registration_nearest_neighbor_ratio_p95"]),
    )
    recomputed_registration_pass = (
        annotation_nucleus_pass
        & annotation_cell_pass
        & cell_nucleus_pass
        & diameter_pass
        & neighbor_pass
    )

    valid_area = (
        np.isfinite(nucleus_area_um2)
        & np.isfinite(cell_area_um2)
        & (nucleus_area_um2 > 0.0)
        & (cell_area_um2 > 0.0)
    )
    area_pass = (
        valid_area
        & (area_ratio >= float(contract["minimum_nucleus_cell_area_ratio"]))
        & (area_ratio <= float(contract["maximum_nucleus_cell_area_ratio"]))
    )
    recomputed_segmentation_pass = nucleus_inside_cell & area_pass

    def segmentation_summary(selected: np.ndarray) -> Mapping[str, object]:
        outside_fraction = float(np.mean(~nucleus_inside_cell[selected]))
        area_outlier_fraction = float(np.mean(~area_pass[selected]))
        return {
            "rows": int(np.count_nonzero(selected)),
            "nucleus_outside_cell_fraction": outside_fraction,
            "maximum_nucleus_outside_cell_fraction": float(
                contract["maximum_nucleus_outside_cell_fraction"]
            ),
            "area_ratio_outlier_fraction": area_outlier_fraction,
            "maximum_area_ratio_outlier_fraction": float(
                contract["maximum_segmentation_outlier_fraction"]
            ),
            "pass": bool(
                outside_fraction <= float(contract["maximum_nucleus_outside_cell_fraction"])
                and area_outlier_fraction
                <= float(contract["maximum_segmentation_outlier_fraction"])
            ),
        }

    segmentation_overall = segmentation_summary(locked)
    segmentation_by_section = {
        section_id: segmentation_summary(locked & (section_ids.astype(str) == section_id))
        for section_id in sorted(set(section_ids[locked].astype(str).tolist()))
    }
    segmentation_report = {
        **segmentation_overall,
        "by_section": segmentation_by_section,
        "pass": bool(
            segmentation_overall["pass"]
            and all(row["pass"] for row in segmentation_by_section.values())
        ),
    }

    padding = np.asarray(crop_padding_fractions, dtype=np.float64)
    if padding.ndim != 2 or padding.shape != (len(donor_ids), len(crop_ids)):
        raise ValueError("locked crop padding audit differs from the frozen crop family")
    valid_padding = np.isfinite(padding) & (padding >= 0.0) & (padding <= 1.0)
    recomputed_crop_pass = np.all(
        valid_padding & (padding <= float(contract["maximum_crop_padding_p95"])), axis=1
    )
    recomputed_nonmolecular_qc_pass = (
        recomputed_registration_pass & recomputed_segmentation_pass & recomputed_crop_pass
    )
    recomputed_qualified_qc_pass = recomputed_nonmolecular_qc_pass & target_qc_pass
    crop_reports = {}
    for column, crop_id in enumerate(crop_ids):
        values = padding[:, column]

        def crop_summary(selected: np.ndarray) -> Mapping[str, object]:
            valid = valid_padding[:, column]
            selected_values = values[selected & valid]
            p95 = float(np.quantile(selected_values, 0.95)) if len(selected_values) else None
            mostly = float(
                np.mean(
                    ~valid[selected] | (values[selected] > float(contract["mostly_padded_cutoff"]))
                )
            )
            return {
                "rows": int(np.count_nonzero(selected)),
                "padding_p95": p95,
                "mostly_padded_fraction": mostly,
                "pass": bool(
                    np.count_nonzero(selected & valid) == np.count_nonzero(selected)
                    and p95 is not None
                    and p95 <= float(contract["maximum_crop_padding_p95"])
                    and mostly <= float(contract["maximum_mostly_padded_fraction"])
                ),
            }

        overall = crop_summary(locked)
        by_section = {
            section_id: crop_summary(locked & (section_ids.astype(str) == section_id))
            for section_id in sorted(set(section_ids[locked].astype(str).tolist()))
        }
        crop_reports[str(crop_id)] = {
            **overall,
            "by_section": by_section,
            "pass": bool(overall["pass"] and all(row["pass"] for row in by_section.values())),
        }

    source_qc_matches = {
        "registration_qc_matches_recomputed": bool(
            np.array_equal(registration_qc_pass[locked], recomputed_registration_pass[locked])
        ),
        "segmentation_qc_matches_recomputed": bool(
            np.array_equal(segmentation_qc_pass[locked], recomputed_segmentation_pass[locked])
        ),
        "crop_qc_matches_recomputed": bool(
            np.array_equal(crop_qc_pass[locked], recomputed_crop_pass[locked])
        ),
        "locked_measurement_qc_matches_recomputed_conjunction": bool(
            np.array_equal(
                source_locked_measurement_qc_pass[locked],
                recomputed_nonmolecular_qc_pass[locked],
            )
        ),
    }
    distribution_checks = {
        "annotation_nucleus": bool(annotation_nucleus_report["pass"]),
        "annotation_cell": bool(annotation_cell_report["pass"]),
        "cell_nucleus": bool(cell_nucleus_report["pass"]),
        "nucleus_diameter_relative": bool(diameter_report["pass"]),
        "nearest_neighbor_relative": bool(neighbor_report["pass"]),
        "segmentation": bool(segmentation_report["pass"]),
        "crop_padding": bool(all(report["pass"] for report in crop_reports.values())),
        **source_qc_matches,
    }
    maximum_crop_padding = np.max(padding, axis=1)
    summaries = {
        "registration": {
            "annotation_to_nucleus_distance_um": annotation_nucleus_report,
            "annotation_to_cell_distance_um": annotation_cell_report,
            "native_cell_to_nucleus_distance_um": cell_nucleus_report,
            "annotation_error_over_median_nucleus_diameter": diameter_report,
            "annotation_error_over_median_nearest_neighbor_distance": neighbor_report,
        },
        "segmentation": segmentation_report,
        "crop_padding": crop_reports,
        "maximum_crop_padding_p95": float(np.quantile(maximum_crop_padding[locked], 0.95)),
        "rows_before_frozen_qc": int(np.count_nonzero(locked)),
        "rows_after_frozen_qc": int(np.count_nonzero(locked & recomputed_qualified_qc_pass)),
        "source_locked_measurement_qc_false_positive_rows": int(
            np.count_nonzero(
                locked & source_locked_measurement_qc_pass & ~recomputed_nonmolecular_qc_pass
            )
        ),
        "reliability_row_policy": (
            "recomputed_registration_and_segmentation_and_crop_and_target_qc"
        ),
    }
    normalized_a = normalize_split_counts(half_a_counts, library_sizes=half_a_library_sizes)
    normalized_b = normalize_split_counts(half_b_counts, library_sizes=half_b_library_sizes)
    minimum_reliability = float(contract["minimum_within_fine_type_reliability"])
    minimum_rows = int(contract["minimum_reliability_rows"])
    minimum_reliable_fraction = float(contract["minimum_locked_donor_type_reliability_fraction"])

    def reliability_report(selected: np.ndarray, *, planned: bool) -> Mapping[str, object]:
        report = feature_reliability(
            normalized_a[selected],
            normalized_b[selected],
            gene_ids,
            minimum_rows=minimum_rows,
        )
        median = report["median_spearman_brown_reliability"]
        evaluable = median is not None
        return {
            **report,
            "planned": planned,
            "evaluable": evaluable,
            "minimum_reliability_rows": minimum_rows,
            "minimum_frozen_reliability": minimum_reliability,
            "passes_frozen_reliability": bool(evaluable and float(median) >= minimum_reliability),
        }

    donor_type_reports = {}
    reliable_donor_types = 0
    donor_type_denominator = 0
    for donor in locked_donors:
        for fine_type in supported_types:
            donor_type_denominator += 1
            selected = (
                (donor_ids == donor) & (fine_type_ids == fine_type) & recomputed_qualified_qc_pass
            )
            report = reliability_report(selected, planned=True)
            reliable_donor_types += int(report["passes_frozen_reliability"])
            donor_type_reports["%s|%s" % (donor, fine_type)] = report
    reliable_donor_type_fraction = float(reliable_donor_types / max(donor_type_denominator, 1))

    locked_donor_set = set(str(value) for value in locked_donors)
    supported_type_set = set(str(value) for value in supported_types)
    parsed_planned_strata = []
    seen_planned_strata = set()
    for raw_stratum_id in planned_stratum_ids:
        stratum_id = str(raw_stratum_id)
        fields = stratum_id.split("|")
        if len(fields) != 3 or any(not value for value in fields):
            raise ValueError("planned donor/section/type stratum identity is malformed")
        if stratum_id in seen_planned_strata:
            raise ValueError("planned donor/section/type stratum identities are not unique")
        seen_planned_strata.add(stratum_id)
        donor, section, fine_type = fields
        if donor in locked_donor_set and fine_type in supported_type_set:
            parsed_planned_strata.append((stratum_id, donor, section, fine_type))
    if not parsed_planned_strata:
        raise ValueError("frozen planned population lacks locked donor/section/type strata")

    planned_pairs = {(donor, fine_type) for _, donor, _, fine_type in parsed_planned_strata}
    missing_pairs = [
        "%s|%s" % (donor, fine_type)
        for donor in locked_donors
        for fine_type in supported_types
        if (str(donor), str(fine_type)) not in planned_pairs
    ]
    if missing_pairs:
        raise ValueError(
            "frozen planned population lacks locked donor/type section strata: %s"
            % ", ".join(missing_pairs)
        )

    locked_supported = locked & np.isin(fine_type_ids, np.asarray(tuple(supported_types)))
    observed_strata = {
        "%s|%s|%s" % row
        for row in zip(
            donor_ids[locked_supported].astype(str).tolist(),
            section_ids[locked_supported].astype(str).tolist(),
            fine_type_ids[locked_supported].astype(str).tolist(),
        )
    }
    unplanned_observed_strata = sorted(observed_strata - seen_planned_strata)
    if unplanned_observed_strata:
        raise ValueError(
            "locked source contains donor/section/type strata outside the frozen planned "
            "population: %s" % ", ".join(unplanned_observed_strata)
        )

    donor_section_type_reports = {}
    planned_sections_by_donor_type: dict[str, list[str]] = {}
    reliable_planned_strata = 0
    evaluable_planned_strata = 0
    for stratum_id, donor, section, fine_type in parsed_planned_strata:
        selected = (
            (donor_ids == donor)
            & (section_ids == section)
            & (fine_type_ids == fine_type)
            & recomputed_qualified_qc_pass
        )
        report = reliability_report(selected, planned=True)
        donor_section_type_reports[stratum_id] = report
        donor_type_id = "%s|%s" % (donor, fine_type)
        planned_sections_by_donor_type.setdefault(donor_type_id, []).append(stratum_id)
        evaluable_planned_strata += int(report["evaluable"])
        reliable_planned_strata += int(report["passes_frozen_reliability"])

    def worst_report_id(stratum_ids: Sequence[str]) -> str:
        def ordering(stratum_id: str) -> tuple[bool, float, str]:
            median = donor_section_type_reports[stratum_id]["median_spearman_brown_reliability"]
            return median is not None, 0.0 if median is None else float(median), stratum_id

        return min(stratum_ids, key=ordering)

    worst_section_by_donor_type = {}
    for donor_type_id, stratum_ids in sorted(planned_sections_by_donor_type.items()):
        worst_id = worst_report_id(stratum_ids)
        worst_report = donor_section_type_reports[worst_id]
        worst_section_by_donor_type[donor_type_id] = {
            "planned_section_count": len(stratum_ids),
            "evaluable_section_count": sum(
                int(donor_section_type_reports[value]["evaluable"]) for value in stratum_ids
            ),
            "reliable_section_count": sum(
                int(donor_section_type_reports[value]["passes_frozen_reliability"])
                for value in stratum_ids
            ),
            "worst_planned_stratum_id": worst_id,
            "worst_median_spearman_brown_reliability": worst_report[
                "median_spearman_brown_reliability"
            ],
            "worst_section_evaluable": worst_report["evaluable"],
            "all_planned_sections_pass_frozen_reliability": all(
                donor_section_type_reports[value]["passes_frozen_reliability"]
                for value in stratum_ids
            ),
        }

    worst_stratum_id = worst_report_id(tuple(donor_section_type_reports))
    worst_donor, worst_section, worst_type = worst_stratum_id.split("|")
    worst_stratum_report = donor_section_type_reports[worst_stratum_id]
    worst_section_summary = {
        "planned_stratum_id": worst_stratum_id,
        "donor_id": worst_donor,
        "section_id": worst_section,
        "fine_type_id": worst_type,
        "rows": worst_stratum_report["rows"],
        "median_spearman_brown_reliability": worst_stratum_report[
            "median_spearman_brown_reliability"
        ],
        "evaluable": worst_stratum_report["evaluable"],
        "passes_frozen_reliability": worst_stratum_report["passes_frozen_reliability"],
    }
    planned_stratum_denominator = len(parsed_planned_strata)
    reliable_planned_stratum_fraction = float(reliable_planned_strata / planned_stratum_denominator)
    donor_type_reliability_pass = bool(reliable_donor_type_fraction >= minimum_reliable_fraction)
    planned_stratum_reliability_pass = bool(
        reliable_planned_stratum_fraction >= minimum_reliable_fraction
    )
    distribution_checks.update(
        {
            "donor_type_reliability_fraction": donor_type_reliability_pass,
            "planned_donor_section_type_reliability_fraction": (planned_stratum_reliability_pass),
        }
    )
    audit_pass = bool(all(distribution_checks.values()))
    return {
        "schema": "heir.locked_measurement_audit.v1",
        "selection_changes_forbidden": True,
        "coverage_denominator": "all_h_meas_supported_fine_types_and_locked_donors",
        "thresholds": dict(contract),
        "summaries": summaries,
        "distribution_checks": distribution_checks,
        "donor_type_reliability": donor_type_reports,
        "planned_donor_type_count": donor_type_denominator,
        "reliable_donor_type_count": reliable_donor_types,
        "reliable_donor_type_fraction": reliable_donor_type_fraction,
        "donor_section_type_reliability": donor_section_type_reports,
        "worst_section_reliability_by_donor_type": worst_section_by_donor_type,
        "worst_section_reliability_summary": worst_section_summary,
        "planned_stratum_reliability": {
            "coverage_denominator": "all_frozen_locked_donor_section_type_strata",
            "planned_count": planned_stratum_denominator,
            "evaluable_count": evaluable_planned_strata,
            "reliable_count": reliable_planned_strata,
            "reliable_fraction": reliable_planned_stratum_fraction,
            "minimum_required_reliable_fraction": minimum_reliable_fraction,
            "pass": planned_stratum_reliability_pass,
        },
        "pass": audit_pass,
    }
