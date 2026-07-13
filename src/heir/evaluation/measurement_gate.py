"""Fail-closed measurement and registration gate for morphology experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from heir.evaluation.reliability import (
    SPLIT_HALF_METHOD,
    construct_split_half_counts,
    cross_fitted_residualize,
    cross_fitted_target_basis_reliability,
    feature_reliability,
    normalize_split_counts,
    program_reliability,
    program_scores,
)
from heir.utils import sha256_file

MEASUREMENT_GATE_SCHEMA = "heir.measurement_gate.v1"
PRIMARY_TARGET_VARIANT = "nucleus_overlapping_transcripts"
SECONDARY_TARGET_VARIANT = "whole_cell_assigned_transcripts"
PathLike = Union[str, Path]


def _nested_thresholds(content: Mapping[str, object], section: str) -> Mapping[str, object]:
    value = content.get(section)
    if not isinstance(value, Mapping):
        raise ValueError("study manifest %s must be a mapping" % section)
    nested = value.get("measurement")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("study manifest %s.measurement must be a mapping" % section)
        return nested
    return value


@dataclass(frozen=True)
class MeasurementThresholds:
    """Every scientific measurement decision threshold, with no hidden defaults."""

    maximum_annotation_nucleus_p95_um: float
    maximum_annotation_cell_p95_um: float
    maximum_cell_nucleus_p95_um: float
    maximum_registration_nucleus_diameter_ratio_p95: float
    maximum_registration_nearest_neighbor_ratio_p95: float
    best_registration_quality_max_fraction_of_limit: float
    intermediate_registration_quality_max_fraction_of_limit: float
    maximum_registration_outlier_fraction: float
    maximum_nucleus_outside_cell_fraction: float
    minimum_nucleus_cell_area_ratio: float
    maximum_nucleus_cell_area_ratio: float
    maximum_segmentation_outlier_fraction: float
    maximum_crop_padding_p95: float
    mostly_padded_cutoff: float
    maximum_mostly_padded_fraction: float
    minimum_transcript_qv: float
    minimum_median_gene_reliability: float
    minimum_median_program_reliability: float
    minimum_target_basis_ceiling: float
    minimum_reliable_gene_fraction: float
    minimum_reliable_development_donor_fraction: float
    minimum_within_fine_type_reliability: float
    minimum_reliability_rows: int
    target_basis_rank: int
    minimum_reliable_development_donors: int
    minimum_reliable_donors_per_fine_type: int
    minimum_coverage_fraction: float
    minimum_reference_cells_per_stratum: int
    minimum_evaluation_cells_per_stratum: int
    minimum_development_donors_per_fine_type: int
    minimum_locked_donors_per_fine_type: int
    maximum_reference_evaluation_row_overlap: int
    maximum_reference_evaluation_block_overlap: int
    maximum_reference_evaluation_source_file_overlap: int

    @classmethod
    def from_study_manifest(cls, content: Mapping[str, object]) -> "MeasurementThresholds":
        decisions = _nested_thresholds(content, "decision_thresholds")
        coverage = _nested_thresholds(content, "coverage_requirements")
        decision_names = (
            "maximum_annotation_nucleus_p95_um",
            "maximum_annotation_cell_p95_um",
            "maximum_cell_nucleus_p95_um",
            "maximum_registration_nucleus_diameter_ratio_p95",
            "maximum_registration_nearest_neighbor_ratio_p95",
            "best_registration_quality_max_fraction_of_limit",
            "intermediate_registration_quality_max_fraction_of_limit",
            "maximum_registration_outlier_fraction",
            "maximum_nucleus_outside_cell_fraction",
            "minimum_nucleus_cell_area_ratio",
            "maximum_nucleus_cell_area_ratio",
            "maximum_segmentation_outlier_fraction",
            "maximum_crop_padding_p95",
            "mostly_padded_cutoff",
            "maximum_mostly_padded_fraction",
            "minimum_transcript_qv",
            "minimum_median_gene_reliability",
            "minimum_median_program_reliability",
            "minimum_target_basis_ceiling",
            "minimum_reliable_gene_fraction",
            "minimum_reliable_development_donor_fraction",
            "minimum_within_fine_type_reliability",
            "minimum_reliability_rows",
            "target_basis_rank",
            "minimum_reliable_development_donors",
            "minimum_reliable_donors_per_fine_type",
        )
        coverage_names = (
            "minimum_coverage_fraction",
            "minimum_reference_cells_per_stratum",
            "minimum_evaluation_cells_per_stratum",
            "minimum_development_donors_per_fine_type",
            "minimum_locked_donors_per_fine_type",
            "maximum_reference_evaluation_row_overlap",
            "maximum_reference_evaluation_block_overlap",
            "maximum_reference_evaluation_source_file_overlap",
        )
        missing = [name for name in decision_names if name not in decisions]
        missing += [name for name in coverage_names if name not in coverage]
        if missing:
            raise ValueError(
                "locked study manifest lacks explicit measurement thresholds: %s"
                % ", ".join(sorted(missing))
            )
        value = cls(
            **{name: decisions[name] for name in decision_names},
            **{name: coverage[name] for name in coverage_names},
        )
        value.validate()
        return value

    def validate(self) -> None:
        nonnegative = (
            "maximum_annotation_nucleus_p95_um",
            "maximum_annotation_cell_p95_um",
            "maximum_cell_nucleus_p95_um",
            "maximum_registration_nucleus_diameter_ratio_p95",
            "maximum_registration_nearest_neighbor_ratio_p95",
            "minimum_nucleus_cell_area_ratio",
            "maximum_nucleus_cell_area_ratio",
            "minimum_transcript_qv",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError("measurement threshold %s must be finite and non-negative" % name)
        fractions = (
            "maximum_registration_outlier_fraction",
            "maximum_nucleus_outside_cell_fraction",
            "maximum_segmentation_outlier_fraction",
            "maximum_crop_padding_p95",
            "mostly_padded_cutoff",
            "maximum_mostly_padded_fraction",
            "minimum_median_gene_reliability",
            "minimum_median_program_reliability",
            "minimum_target_basis_ceiling",
            "minimum_reliable_gene_fraction",
            "minimum_reliable_development_donor_fraction",
            "minimum_within_fine_type_reliability",
            "minimum_coverage_fraction",
        )
        for name in fractions:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("measurement threshold %s must be in [0, 1]" % name)
        if self.minimum_nucleus_cell_area_ratio > self.maximum_nucleus_cell_area_ratio:
            raise ValueError("nucleus/cell area-ratio thresholds are reversed")
        if (
            self.maximum_registration_nucleus_diameter_ratio_p95 <= 0
            or self.maximum_registration_nearest_neighbor_ratio_p95 <= 0
        ):
            raise ValueError("geometry-relative registration thresholds must be positive")
        if not (
            0
            < self.best_registration_quality_max_fraction_of_limit
            < self.intermediate_registration_quality_max_fraction_of_limit
            < 1
        ):
            raise ValueError("registration quality-stratum cutoffs must increase within (0, 1)")
        if self.minimum_reliable_development_donor_fraction <= 0:
            raise ValueError("reliable molecular targets require a positive donor fraction")
        positive_integers = (
            "minimum_reliability_rows",
            "target_basis_rank",
            "minimum_reliable_development_donors",
            "minimum_reliable_donors_per_fine_type",
            "minimum_reference_cells_per_stratum",
            "minimum_evaluation_cells_per_stratum",
            "minimum_development_donors_per_fine_type",
        )
        if any(
            isinstance(getattr(self, name), (bool, np.bool_))
            or int(getattr(self, name)) != getattr(self, name)
            or int(getattr(self, name)) < 1
            for name in positive_integers
        ):
            raise ValueError("measurement minimum counts and target rank must be positive integers")
        if (
            self.minimum_reliable_development_donors < 2
            or self.minimum_reliable_donors_per_fine_type < 2
        ):
            raise ValueError("donor-cross-fitted reliability requires at least two donors")
        locked_minimum = self.minimum_locked_donors_per_fine_type
        if (
            isinstance(locked_minimum, (bool, np.bool_))
            or int(locked_minimum) != locked_minimum
            or int(locked_minimum) < 0
        ):
            raise ValueError("locked donor coverage minimum must be a non-negative integer")
        overlap_names = (
            "maximum_reference_evaluation_row_overlap",
            "maximum_reference_evaluation_block_overlap",
            "maximum_reference_evaluation_source_file_overlap",
        )
        if any(
            isinstance(getattr(self, name), (bool, np.bool_))
            or int(getattr(self, name)) != getattr(self, name)
            or int(getattr(self, name)) < 0
            for name in overlap_names
        ):
            raise ValueError("measurement overlap thresholds must be non-negative integers")


def _first(source: Mapping[str, object], names: Sequence[str]) -> Optional[np.ndarray]:
    for name in names:
        if name in source:
            return np.asarray(source[name])
    return None


def _scalar_value(source: Mapping[str, object], names: Sequence[str]) -> Optional[object]:
    value = _first(source, names)
    if value is None:
        return None
    if value.ndim != 0:
        raise ValueError("measurement field %s must be scalar" % names[0])
    return value.item()


def _nonnegative_integer_scalar(
    source: Mapping[str, object], names: Sequence[str]
) -> Optional[int]:
    value = _scalar_value(source, names)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 0:
        raise ValueError("measurement field %s must be a non-negative integer" % names[0])
    return int(value)


def _vector(
    source: Mapping[str, object],
    names: Sequence[str],
    rows: Optional[int] = None,
) -> Optional[np.ndarray]:
    result = _first(source, names)
    if result is None:
        return None
    if result.ndim != 1 or (rows is not None and len(result) != rows):
        raise ValueError("measurement field %s must be an aligned vector" % names[0])
    return result


def _string_vector(
    source: Mapping[str, object],
    names: Sequence[str],
    rows: Optional[int] = None,
) -> Optional[np.ndarray]:
    result = _vector(source, names, rows)
    if result is None:
        return None
    values = result.astype(str)
    if any(not value.strip() for value in values.tolist()):
        raise ValueError("measurement field %s contains empty identifiers" % names[0])
    return values


def _required_strings(source: Mapping[str, object], names: Sequence[str]) -> np.ndarray:
    values = _string_vector(source, names)
    if values is None:
        raise ValueError("measurement source lacks %s" % names[0])
    return values


def _booleans(values: object, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype == np.bool_:
        return result
    if result.dtype.kind in "iu" and np.all((result == 0) | (result == 1)):
        return result.astype(np.bool_)
    raise ValueError("measurement field %s must contain only booleans" % name)


def _duplicate_count(values: np.ndarray) -> int:
    return int(len(values) - len(set(values.astype(str).tolist())))


def _quantiles(values: object) -> Mapping[str, object]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "rows": 0,
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    quantiles = np.quantile(array, (0.05, 0.5, 0.95))
    return {
        "rows": int(len(array)),
        "minimum": float(array.min()),
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def _distance(
    source: Mapping[str, object],
    direct_names: Sequence[str],
    first_x: Sequence[str],
    first_y: Sequence[str],
    second_x: Sequence[str],
    second_y: Sequence[str],
    rows: int,
) -> Optional[np.ndarray]:
    direct = _vector(source, direct_names, rows)
    if direct is not None:
        return direct.astype(np.float64)
    coordinates = (
        _vector(source, first_x, rows),
        _vector(source, first_y, rows),
        _vector(source, second_x, rows),
        _vector(source, second_y, rows),
    )
    if any(value is None for value in coordinates):
        return None
    first_x_values, first_y_values, second_x_values, second_y_values = coordinates
    return np.hypot(
        first_x_values.astype(np.float64) - second_x_values.astype(np.float64),
        first_y_values.astype(np.float64) - second_y_values.astype(np.float64),
    )


def _nearest_neighbor_distances(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    section_ids: np.ndarray,
    nucleus_diameters: np.ndarray,
) -> np.ndarray:
    """Exact planar nearest-neighbor distances using an expanding spatial grid."""

    x = np.asarray(x_coordinates, dtype=np.float64)
    y = np.asarray(y_coordinates, dtype=np.float64)
    diameters = np.asarray(nucleus_diameters, dtype=np.float64)
    if (
        x.shape != y.shape
        or x.shape != diameters.shape
        or x.shape != section_ids.shape
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("nucleus coordinates for nearest-neighbor QC are malformed")
    positive_diameters = diameters[np.isfinite(diameters) & (diameters > 0)]
    if not len(positive_diameters):
        raise ValueError("nearest-neighbor QC requires positive nucleus diameters")
    cell_size = float(np.median(positive_diameters))
    result = np.full(len(x), np.nan, dtype=np.float64)
    for section in sorted(set(section_ids.tolist())):
        indices = np.flatnonzero(section_ids == section)
        if len(indices) < 2:
            continue
        cell_x = np.floor(x[indices] / cell_size).astype(np.int64)
        cell_y = np.floor(y[indices] / cell_size).astype(np.int64)
        grid: dict[tuple[int, int], list[int]] = {}
        for local_index, key in enumerate(zip(cell_x.tolist(), cell_y.tolist())):
            grid.setdefault(key, []).append(local_index)
        minimum_x, maximum_x = int(cell_x.min()), int(cell_x.max())
        minimum_y, maximum_y = int(cell_y.min()), int(cell_y.max())
        for local_index, global_index in enumerate(indices.tolist()):
            center_x, center_y = int(cell_x[local_index]), int(cell_y[local_index])
            best_squared = np.inf
            maximum_radius = max(
                center_x - minimum_x,
                maximum_x - center_x,
                center_y - minimum_y,
                maximum_y - center_y,
            )
            for radius in range(maximum_radius + 1):
                for grid_x in range(center_x - radius, center_x + radius + 1):
                    for grid_y in range(center_y - radius, center_y + radius + 1):
                        if radius and max(abs(grid_x - center_x), abs(grid_y - center_y)) != radius:
                            continue
                        for candidate in grid.get((grid_x, grid_y), ()):
                            if candidate == local_index:
                                continue
                            delta_x = x[indices[candidate]] - x[global_index]
                            delta_y = y[indices[candidate]] - y[global_index]
                            best_squared = min(best_squared, delta_x * delta_x + delta_y * delta_y)
                left = (center_x - radius) * cell_size
                right = (center_x + radius + 1) * cell_size
                bottom = (center_y - radius) * cell_size
                top = (center_y + radius + 1) * cell_size
                distance_to_unsearched = min(
                    x[global_index] - left,
                    right - x[global_index],
                    y[global_index] - bottom,
                    top - y[global_index],
                )
                if np.isfinite(best_squared) and best_squared <= distance_to_unsearched**2:
                    break
            if np.isfinite(best_squared):
                result[global_index] = np.sqrt(best_squared)
    return result


def _relative_distance_qc(
    errors: np.ndarray,
    scales: np.ndarray,
    section_ids: np.ndarray,
    *,
    maximum_p95: float,
    maximum_outlier_fraction: float,
) -> tuple[Mapping[str, object], np.ndarray, np.ndarray]:
    valid = np.isfinite(errors) & (errors >= 0) & np.isfinite(scales) & (scales > 0)
    row_ratios = np.full(len(errors), np.nan, dtype=np.float64)

    def summarize(selected: np.ndarray) -> Mapping[str, object]:
        selected_valid = selected & valid
        median_scale = float(np.median(scales[selected_valid])) if selected_valid.any() else None
        ratios = np.full(len(errors), np.nan, dtype=np.float64)
        if median_scale is not None:
            ratios[selected_valid] = errors[selected_valid] / median_scale
        selected_pass = selected_valid & (ratios <= maximum_p95)
        summary = dict(_quantiles(ratios[selected_valid]))
        outlier_fraction = float(np.mean(~selected_pass[selected])) if selected.any() else 1.0
        return {
            **summary,
            "normalization_denominator": "median_geometry_scale_um",
            "median_geometry_scale_um": median_scale,
            "maximum_allowed_p95_ratio": float(maximum_p95),
            "outlier_fraction": outlier_fraction,
            "maximum_allowed_outlier_fraction": float(maximum_outlier_fraction),
            "pass": bool(
                selected.any()
                and selected_valid.sum() == selected.sum()
                and summary["p95"] is not None
                and float(summary["p95"]) <= maximum_p95
                and outlier_fraction <= maximum_outlier_fraction
            ),
        }

    overall = np.ones(len(errors), dtype=np.bool_)
    by_section = {}
    for section in sorted(set(section_ids.tolist())):
        selected = section_ids == section
        section_valid = selected & valid
        if section_valid.any():
            median_scale = float(np.median(scales[section_valid]))
            row_ratios[section_valid] = errors[section_valid] / median_scale
        by_section[section] = summarize(selected)
    row_pass = valid & (row_ratios <= maximum_p95)
    report = {
        **summarize(overall),
        "by_section": by_section,
    }
    report["pass"] = bool(
        report["pass"] and all(value["pass"] for value in report["by_section"].values())
    )
    return report, row_pass, row_ratios


def _distance_qc(
    values: Optional[np.ndarray],
    *,
    maximum_p95: float,
    maximum_outlier_fraction: float,
) -> tuple[Mapping[str, object], np.ndarray]:
    if values is None:
        return {"available": False, "pass": False}, np.zeros(0, dtype=np.bool_)
    valid = np.isfinite(values) & (values >= 0)
    row_pass = valid & (values <= maximum_p95)
    summary = dict(_quantiles(values[valid]))
    p95 = summary["p95"]
    outlier_fraction = float(np.mean(~row_pass)) if len(values) else 1.0
    passed = bool(
        len(values)
        and valid.all()
        and p95 is not None
        and float(p95) <= maximum_p95
        and outlier_fraction <= maximum_outlier_fraction
    )
    return {
        "available": True,
        **summary,
        "maximum_allowed_p95_um": float(maximum_p95),
        "outlier_fraction": outlier_fraction,
        "maximum_allowed_outlier_fraction": float(maximum_outlier_fraction),
        "pass": passed,
    }, row_pass


def _distance_qc_by_section(
    values: Optional[np.ndarray],
    section_ids: np.ndarray,
    *,
    maximum_p95: float,
    maximum_outlier_fraction: float,
) -> Mapping[str, object]:
    if values is None:
        return {}
    result = {}
    for section in sorted(set(section_ids.tolist())):
        selected = section_ids == section
        report, _ = _distance_qc(
            values[selected],
            maximum_p95=maximum_p95,
            maximum_outlier_fraction=maximum_outlier_fraction,
        )
        result[section] = report
    return result


def _programs_from_source(
    source: Mapping[str, object],
    gene_ids: np.ndarray,
) -> Optional[Mapping[str, object]]:
    names = _string_vector(source, ("program_names", "molecular_program_names"))
    membership = _first(source, ("program_gene_membership", "molecular_program_gene_weights"))
    if names is None or membership is None:
        return None
    weights = np.asarray(membership, dtype=np.float64)
    if weights.shape != (len(names), len(gene_ids)) or not np.isfinite(weights).all():
        raise ValueError("molecular program membership is malformed")
    return {
        name: {
            gene: float(weight)
            for gene, weight in zip(gene_ids.tolist(), weights[index].tolist())
            if weight != 0
        }
        for index, name in enumerate(names.tolist())
    }


def _variant_membership(
    source: Mapping[str, object],
    names: Sequence[str],
    transcript_rows: int,
) -> Optional[np.ndarray]:
    explicit = _first(source, ("target_variant_membership", "transcript_variant_membership"))
    explicit_names = _string_vector(source, ("target_variant_names", "target_variants"))
    if explicit is not None:
        membership = np.asarray(explicit)
        if explicit_names is None or membership.shape != (transcript_rows, len(explicit_names)):
            raise ValueError("target-variant transcript membership is malformed")
        lookup = {value: index for index, value in enumerate(explicit_names.tolist())}
        if any(name not in lookup for name in names):
            raise ValueError("source lacks a target variant required by the locked manifest")
        selected = membership[:, [lookup[name] for name in names]]
        return _booleans(selected, "target_variant_membership")

    if len(names) == 1:
        return np.ones((transcript_rows, 1), dtype=np.bool_)
    overlaps = _vector(
        source,
        ("transcript_overlaps_nucleus", "overlaps_nucleus"),
        transcript_rows,
    )
    if overlaps is None:
        return None
    result = np.zeros((transcript_rows, len(names)), dtype=np.bool_)
    for index, name in enumerate(names):
        if name == "whole_cell_assigned_transcripts":
            result[:, index] = True
        elif name == "nucleus_overlapping_transcripts":
            result[:, index] = _booleans(overlaps, "transcript_overlaps_nucleus")
        else:
            return None
    return result


def _precomputed_split_counts(
    source: Mapping[str, object],
    variant: str,
    *,
    rows: int,
    genes: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    prefixes = {
        "nucleus_overlapping_transcripts": "nucleus_target_counts",
        "whole_cell_assigned_transcripts": "whole_cell_target_counts",
    }
    prefix = prefixes.get(variant)
    if prefix is None:
        raise ValueError("precomputed split counts do not define target variant %s" % variant)
    first = _first(source, (prefix + "_half_a",))
    second = _first(source, (prefix + "_half_b",))
    total = _first(source, (prefix,))
    if first is None or second is None or total is None:
        raise ValueError("precomputed transcript split is incomplete for %s" % variant)
    matrices = tuple(np.asarray(value) for value in (first, second, total))
    for values in matrices:
        if values.shape != (rows, genes) or not np.issubdtype(values.dtype, np.number):
            raise ValueError("precomputed transcript split count matrix is malformed")
        for start in range(0, rows, 65_536):
            chunk = values[start : start + 65_536]
            valid = np.isfinite(chunk) & (chunk >= 0)
            if values.dtype.kind not in "iu":
                valid &= chunk == np.floor(chunk)
            if not valid.all():
                raise ValueError("precomputed transcript split count matrix is malformed")
    first_counts, second_counts, total_counts = matrices
    for start in range(0, rows, 65_536):
        first_chunk = first_counts[start : start + 65_536].astype(np.uint64)
        second_chunk = second_counts[start : start + 65_536].astype(np.uint64)
        total_chunk = total_counts[start : start + 65_536].astype(np.uint64)
        if not np.array_equal(first_chunk + second_chunk, total_chunk):
            raise ValueError(
                "precomputed transcript halves do not reconstruct frozen target counts"
            )
    return first_counts, second_counts, int(total_counts.sum(dtype=np.uint64))


def _precomputed_half_libraries(
    source: Mapping[str, object],
    variant: str,
    *,
    half_a: np.ndarray,
    half_b: np.ndarray,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    prefixes = {
        "nucleus_overlapping_transcripts": "nucleus_library_size",
        "whole_cell_assigned_transcripts": "whole_cell_library_size",
    }
    prefix = prefixes.get(variant)
    if prefix is None:
        raise ValueError("precomputed library sizes do not define target variant %s" % variant)
    first = _vector(source, (prefix + "_half_a",), rows)
    second = _vector(source, (prefix + "_half_b",), rows)
    total = _vector(source, (prefix + "s", prefix), rows)
    if first is None or second is None or total is None:
        raise ValueError("precomputed split lacks frozen-target library-size halves")
    libraries = tuple(np.asarray(value, dtype=np.float64) for value in (first, second, total))
    for values in libraries:
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError("precomputed split library sizes are malformed")
    first_library, second_library, total_library = libraries
    if not np.array_equal(first_library + second_library, total_library):
        raise ValueError("precomputed library-size halves do not reconstruct full libraries")
    if np.any(first_library < half_a.sum(axis=1)) or np.any(second_library < half_b.sum(axis=1)):
        raise ValueError("precomputed library-size halves are below target-gene counts")
    return first_library, second_library


def _frozen_full_target(
    source: Mapping[str, object],
    variant: str,
    *,
    rows: int,
    genes: int,
    required: bool,
) -> Optional[np.ndarray]:
    names = {
        "nucleus_overlapping_transcripts": (
            "normalized_nucleus_targets",
            "nucleus_molecular_targets",
        ),
        "whole_cell_assigned_transcripts": (
            "normalized_whole_cell_targets",
            "whole_cell_molecular_targets",
        ),
    }.get(variant)
    if names is None:
        raise ValueError("frozen full target does not define target variant %s" % variant)
    values = _first(source, names)
    if values is None:
        if required:
            raise ValueError("precomputed split lacks its frozen normalized molecular target")
        return None
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (rows, genes) or not np.isfinite(result).all():
        raise ValueError("frozen normalized molecular target is malformed")
    return result


def _group_reliability(
    first: np.ndarray,
    second: np.ndarray,
    gene_ids: np.ndarray,
    programs: Mapping[str, object],
    groups: np.ndarray,
    *,
    minimum_rows: int,
) -> Mapping[str, object]:
    result = {}
    for group in sorted(set(groups.astype(str).tolist())):
        selected = groups.astype(str) == group
        gene_report = feature_reliability(
            first[selected], second[selected], gene_ids.tolist(), minimum_rows=minimum_rows
        )
        program_report = program_reliability(
            first[selected],
            second[selected],
            gene_ids.tolist(),
            programs,
            minimum_rows=minimum_rows,
        )
        result[group] = {
            "rows": int(selected.sum()),
            "genes": gene_report["features"],
            "programs": program_report["features"],
            "median_gene_reliability": gene_report["median_spearman_brown_reliability"],
            "median_program_reliability": program_report["median_spearman_brown_reliability"],
        }
    return result


def _macro_reliability(
    grouped: Mapping[str, object],
    selected_groups: Sequence[str],
    feature_family: str,
) -> Mapping[str, object]:
    group_names = tuple(str(value) for value in selected_groups)
    feature_names: set[str] = set()
    for group in group_names:
        report = grouped.get(group)
        if isinstance(report, Mapping) and isinstance(report.get(feature_family), Mapping):
            feature_names.update(str(value) for value in report[feature_family])
    features = {}
    for feature in sorted(feature_names):
        values = []
        evaluable_groups = []
        for group in group_names:
            report = grouped.get(group)
            family = report.get(feature_family) if isinstance(report, Mapping) else None
            record = family.get(feature) if isinstance(family, Mapping) else None
            value = (
                record.get("spearman_brown_reliability") if isinstance(record, Mapping) else None
            )
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
                evaluable_groups.append(group)
        features[feature] = {
            "donor_macro_spearman_brown_reliability": (
                None if not values else float(np.median(values))
            ),
            "evaluable_development_donors": evaluable_groups,
            "evaluable_development_donor_count": int(len(evaluable_groups)),
            "evaluable_development_donor_fraction": float(len(evaluable_groups) / len(group_names))
            if group_names
            else 0.0,
            "development_donors": int(len(group_names)),
        }
    finite = [
        record["donor_macro_spearman_brown_reliability"]
        for record in features.values()
        if record["donor_macro_spearman_brown_reliability"] is not None
    ]
    return {
        "features": features,
        "finite_features": int(len(finite)),
        "median_donor_macro_spearman_brown_reliability": (
            None if not finite else float(np.median(finite))
        ),
    }


def _donor_macro_feature_reliability(
    first: np.ndarray,
    second: np.ndarray,
    feature_ids: Sequence[str],
    donor_ids: np.ndarray,
    development_mask: np.ndarray,
    *,
    minimum_rows: int,
) -> Mapping[str, object]:
    names = tuple(str(value) for value in feature_ids)
    development_donors = sorted(set(donor_ids[development_mask].tolist()))
    donor_reports = {}
    for donor in development_donors:
        selected = development_mask & (donor_ids == donor)
        finite_rows = selected & np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
        donor_reports[donor] = feature_reliability(
            first[finite_rows],
            second[finite_rows],
            names,
            minimum_rows=minimum_rows,
        )
    features = {}
    for feature in names:
        values = {
            donor: report["features"][feature]["spearman_brown_reliability"]
            for donor, report in donor_reports.items()
            if report["features"][feature]["spearman_brown_reliability"] is not None
        }
        features[feature] = {
            "donor_macro_spearman_brown_reliability": (
                None if not values else float(np.median(list(values.values())))
            ),
            "evaluable_development_donor_ids": sorted(values),
            "evaluable_development_donor_count": int(len(values)),
            "evaluable_development_donor_fraction": (
                float(len(values) / len(development_donors)) if development_donors else 0.0
            ),
            "development_donor_ids": development_donors,
        }
    finite = [
        value["donor_macro_spearman_brown_reliability"]
        for value in features.values()
        if value["donor_macro_spearman_brown_reliability"] is not None
    ]
    return {
        "development_donor_ids": development_donors,
        "donor_reports": donor_reports,
        "features": features,
        "finite_features": int(len(finite)),
        "median_donor_macro_spearman_brown_reliability": (
            None if not finite else float(np.median(finite))
        ),
    }


def _within_type_donor_macro_reliability(
    first: np.ndarray,
    second: np.ndarray,
    feature_ids: Sequence[str],
    donor_ids: np.ndarray,
    fine_types: np.ndarray,
    development_mask: np.ndarray,
    *,
    minimum_rows: int,
) -> Mapping[str, object]:
    names = tuple(str(value) for value in feature_ids)
    result = {}
    for fine_type in sorted(set(fine_types.tolist())):
        type_development = development_mask & (fine_types == fine_type)
        type_donors = sorted(set(donor_ids[type_development].tolist()))
        donor_reports = {}
        for donor in type_donors:
            selected = type_development & (donor_ids == donor)
            finite_rows = (
                selected & np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
            )
            donor_reports[donor] = feature_reliability(
                first[finite_rows],
                second[finite_rows],
                names,
                minimum_rows=minimum_rows,
            )
        features = {}
        for feature in names:
            values = {
                donor: report["features"][feature]["spearman_brown_reliability"]
                for donor, report in donor_reports.items()
                if report["features"][feature]["spearman_brown_reliability"] is not None
            }
            features[feature] = {
                "donor_macro_spearman_brown_reliability": (
                    None if not values else float(np.median(list(values.values())))
                ),
                "evaluable_development_donor_ids": sorted(values),
                "evaluable_development_donor_count": int(len(values)),
                "evaluable_development_donor_fraction": (
                    float(len(values) / len(type_donors)) if type_donors else 0.0
                ),
                "development_donor_ids": type_donors,
            }
        result[fine_type] = {
            "development_donor_ids": type_donors,
            "donor_reports": donor_reports,
            "features": features,
        }
    return result


def _feature_meets_reliability_contract(
    feature_id: str,
    overall: Mapping[str, object],
    within_type: Mapping[str, object],
    supported_fine_types: Sequence[str],
    *,
    minimum_reliability: float,
    minimum_development_donors: int,
    minimum_development_donor_fraction: float,
    minimum_donors_per_fine_type: int,
    minimum_within_type_reliability: float,
) -> bool:
    value = overall.get("donor_macro_spearman_brown_reliability")
    if (
        value is None
        or float(value) < minimum_reliability
        or int(overall.get("evaluable_development_donor_count", 0)) < minimum_development_donors
        or float(overall.get("evaluable_development_donor_fraction", 0.0))
        < minimum_development_donor_fraction
    ):
        return False
    for fine_type in supported_fine_types:
        type_report = within_type.get(fine_type)
        features = type_report.get("features") if isinstance(type_report, Mapping) else None
        feature = features.get(feature_id) if isinstance(features, Mapping) else None
        if (
            not isinstance(feature, Mapping)
            or feature.get("donor_macro_spearman_brown_reliability") is None
            or float(feature["donor_macro_spearman_brown_reliability"])
            < minimum_within_type_reliability
            or int(feature.get("evaluable_development_donor_count", 0))
            < minimum_donors_per_fine_type
            or float(feature.get("evaluable_development_donor_fraction", 0.0))
            < minimum_development_donor_fraction
        ):
            return False
    return True


def _target_basis_ceiling_by_type(
    first: np.ndarray,
    second: np.ndarray,
    full_target: Optional[np.ndarray],
    development_mask: np.ndarray,
    donor_ids: np.ndarray,
    fine_types: np.ndarray,
    supported_fine_types: Sequence[str],
    *,
    rank: int,
    minimum_rows: int,
    minimum_ceiling: float,
    minimum_donors: int,
    minimum_donor_fraction: float,
) -> Mapping[str, object]:
    reports = {}
    values = []
    all_evaluable = True
    for fine_type in supported_fine_types:
        type_rows = fine_types == fine_type
        type_development = development_mask & type_rows
        try:
            report = dict(
                cross_fitted_target_basis_reliability(
                    first,
                    second,
                    donor_ids,
                    development_mask=type_development,
                    rank=rank,
                    minimum_rows=minimum_rows,
                    minimum_training_donors=max(1, minimum_donors - 1),
                    full_targets=full_target,
                )
            )
            value = report["minimum_component_reliability"]
            component_coverage = all(
                int(component["evaluable_donor_count"]) >= minimum_donors
                and float(component["evaluable_donor_fraction"]) >= minimum_donor_fraction
                for component in report["components"].values()
            )
            report["pass"] = bool(
                value is not None and value >= minimum_ceiling and component_coverage
            )
            if value is not None:
                values.append(float(value))
            all_evaluable &= value is not None
        except ValueError as error:
            report = {
                "rows": int(type_development.sum()),
                "error": str(error),
                "pass": False,
            }
            all_evaluable = False
        reports[fine_type] = report
    return {
        "fine_types": reports,
        "evaluable_fine_types": int(len(values)),
        "planned_fine_types": int(len(reports)),
        "minimum_fine_type_ceiling": None if not values else float(min(values)),
        "median_fine_type_ceiling": None if not values else float(np.median(values)),
        "pass": bool(
            reports and all_evaluable and all(report["pass"] for report in reports.values())
        ),
    }


def _panel_sha256(gene_ids: Sequence[str]) -> str:
    payload = json.dumps(list(gene_ids), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _multiassigned_transcripts(transcript_ids: np.ndarray, cell_ids: np.ndarray) -> int:
    if _duplicate_count(transcript_ids) == 0:
        return 0
    assignments: dict[str, set[str]] = {}
    for transcript, cell in zip(transcript_ids.tolist(), cell_ids.tolist()):
        assignments.setdefault(str(transcript), set()).add(str(cell))
    return sum(len(values) > 1 for values in assignments.values())


def _distribution_by_section(
    counts: np.ndarray,
    section_ids: np.ndarray,
    *,
    library_sizes: Optional[np.ndarray] = None,
) -> Mapping[str, object]:
    result = {}
    library = (
        counts.sum(axis=1, dtype=np.uint64)
        if library_sizes is None
        else np.asarray(library_sizes, dtype=np.float64)
    )
    if library.shape != (len(counts),) or np.any(library < counts.sum(axis=1)):
        raise ValueError("molecular distribution libraries are malformed")
    detected = np.count_nonzero(counts, axis=1)
    zero_fraction = np.mean(counts == 0, axis=1)
    for section in sorted(set(section_ids.tolist())):
        selected = section_ids == section
        result[section] = {
            "rows": int(selected.sum()),
            "library_size": _quantiles(library[selected]),
            "detected_target_genes": _quantiles(detected[selected]),
            "zero_fraction": _quantiles(zero_fraction[selected]),
        }
    return result


def _intersection_count(first: np.ndarray, second: np.ndarray) -> int:
    return len(set(first.astype(str).tolist()) & set(second.astype(str).tolist()))


def evaluate_measurement_gate(
    source: Mapping[str, object],
    thresholds: MeasurementThresholds,
    *,
    development_donors: Sequence[str],
    locked_test_donors: Sequence[str],
    target_variants: Sequence[str],
    split_salt: str,
    programs: Optional[Mapping[str, object]] = None,
    study_manifest_sha256: Optional[str] = None,
    source_sha256: Optional[str] = None,
) -> Mapping[str, object]:
    """Evaluate measurement validity without using any image-model outcome."""

    thresholds.validate()
    if thresholds.minimum_locked_donors_per_fine_type != 0:
        raise ValueError("measurement-development locked donor coverage minimum must be zero")
    observations = _required_strings(source, ("observation_id", "observation_ids"))
    rows = len(observations)
    if not rows:
        raise ValueError("measurement source contains no observations")
    cell_ids = _required_strings(source, ("cell_id", "cell_ids"))
    if len(cell_ids) != rows:
        raise ValueError("cell IDs do not align with registered observations")
    donor_ids = _string_vector(source, ("donor_id", "donor_ids"), rows)
    split_ids = _string_vector(source, ("split_id", "split_ids"), rows)
    section_ids = _string_vector(source, ("section_id", "section_ids"), rows)
    fine_types = _string_vector(
        source,
        ("fine_type", "fine_type_ids", "fine_type_label", "fine_type_labels"),
        rows,
    )
    block_ids = _string_vector(source, ("block_id", "block_ids"), rows)
    pool_roles = _string_vector(source, ("pool_role", "pool_roles"), rows)
    source_files = _string_vector(
        source,
        ("source_file_id", "source_file_ids", "source_sample_id", "source_sample_ids"),
        rows,
    )
    separation_identities = (
        donor_ids,
        section_ids,
        fine_types,
        block_ids,
        pool_roles,
        source_files,
    )
    if any(value is None for value in separation_identities):
        raise ValueError("measurement source lacks donor/section/type/pool separation identities")
    study_stage = _scalar_value(source, ("study_stage",))
    source_scope = _scalar_value(source, ("source_scope",))
    source_study_manifest_sha256 = _scalar_value(source, ("study_manifest_sha256",))
    locked_outcomes_materialized = _scalar_value(source, ("locked_donor_outcomes_materialized",))
    if study_stage != "measurement_development":
        raise ValueError("H-MEAS source is not marked measurement_development")
    if source_scope != "development_donors_only":
        raise ValueError("H-MEAS source is not development-donor-only")
    if source_study_manifest_sha256 != study_manifest_sha256:
        raise ValueError("H-MEAS source is not bound to the measurement study manifest")
    if locked_outcomes_materialized is not False:
        raise ValueError("H-MEAS source does not prove locked donor outcomes stayed unopened")
    if split_ids is None or set(split_ids.tolist()) != {"development"}:
        raise ValueError("H-MEAS source split_ids must contain development only")
    observed_donors = set(donor_ids.tolist())
    locked_rows = sorted(observed_donors & set(locked_test_donors))
    unexpected_donors = sorted(observed_donors - set(development_donors))
    if locked_rows:
        raise ValueError("H-MEAS source materialized reserved locked donor rows")
    if unexpected_donors:
        raise ValueError("H-MEAS source contains donors outside the development partition")

    annotation_nucleus = _distance(
        source,
        ("registration_distance_um", "annotation_nucleus_distance_um"),
        ("annotation_centroid_x_um",),
        ("annotation_centroid_y_um",),
        ("nucleus_centroid_x_um",),
        ("nucleus_centroid_y_um",),
        rows,
    )
    annotation_cell = _distance(
        source,
        ("annotation_cell_distance_um",),
        ("annotation_centroid_x_um",),
        ("annotation_centroid_y_um",),
        ("cell_centroid_x_um",),
        ("cell_centroid_y_um",),
        rows,
    )
    cell_nucleus = _distance(
        source,
        ("cell_nucleus_centroid_distance_um", "nucleus_cell_centroid_distance_um"),
        ("cell_centroid_x_um",),
        ("cell_centroid_y_um",),
        ("nucleus_centroid_x_um",),
        ("nucleus_centroid_y_um",),
        rows,
    )
    nucleus_area_geometry = _vector(source, ("nucleus_area_um2",), rows)
    nucleus_x = _vector(source, ("nucleus_centroid_x_um",), rows)
    nucleus_y = _vector(source, ("nucleus_centroid_y_um",), rows)
    nearest_neighbor = _vector(source, ("nearest_neighbor_nucleus_distance_um",), rows)
    geometry_available = bool(
        annotation_nucleus is not None
        and nucleus_area_geometry is not None
        and nucleus_x is not None
        and nucleus_y is not None
    )
    if geometry_available:
        nucleus_diameter = 2.0 * np.sqrt(nucleus_area_geometry.astype(np.float64) / np.pi)
        if nearest_neighbor is None:
            nearest_neighbor = _nearest_neighbor_distances(
                nucleus_x,
                nucleus_y,
                section_ids,
                nucleus_diameter,
            )
            nearest_neighbor_method = "exact_expanding_spatial_grid_v1"
        else:
            nearest_neighbor = nearest_neighbor.astype(np.float64)
            nearest_neighbor_method = "registered_source"
        diameter_relative_report, diameter_relative_rows, diameter_ratios = _relative_distance_qc(
            annotation_nucleus,
            nucleus_diameter,
            section_ids,
            maximum_p95=(thresholds.maximum_registration_nucleus_diameter_ratio_p95),
            maximum_outlier_fraction=(thresholds.maximum_registration_outlier_fraction),
        )
        neighbor_relative_report, neighbor_relative_rows, neighbor_ratios = _relative_distance_qc(
            annotation_nucleus,
            nearest_neighbor,
            section_ids,
            maximum_p95=(thresholds.maximum_registration_nearest_neighbor_ratio_p95),
            maximum_outlier_fraction=(thresholds.maximum_registration_outlier_fraction),
        )
        geometry_quality_score = np.maximum(
            diameter_ratios / thresholds.maximum_registration_nucleus_diameter_ratio_p95,
            neighbor_ratios / thresholds.maximum_registration_nearest_neighbor_ratio_p95,
        )
        quality_strata = np.full(rows, "failed", dtype="<U16")
        quality_strata[np.isfinite(geometry_quality_score) & (geometry_quality_score <= 1.0)] = (
            "near_threshold"
        )
        quality_strata[
            np.isfinite(geometry_quality_score)
            & (
                geometry_quality_score
                <= thresholds.intermediate_registration_quality_max_fraction_of_limit
            )
        ] = "intermediate"
        quality_strata[
            np.isfinite(geometry_quality_score)
            & (geometry_quality_score <= thresholds.best_registration_quality_max_fraction_of_limit)
        ] = "best"
        quality_stratum_manifest_sha256 = _panel_sha256(
            [
                observation + "|" + quality
                for observation, quality in zip(observations.tolist(), quality_strata.tolist())
            ]
        )
        quality_strata_report = {
            "definition": (
                "max(error/section_median_nucleus_diameter/diameter_limit,"
                "error/section_median_nearest_neighbor_distance/neighbor_limit)"
            ),
            "cutoffs_fraction_of_limit": {
                "best": thresholds.best_registration_quality_max_fraction_of_limit,
                "intermediate": (
                    thresholds.intermediate_registration_quality_max_fraction_of_limit
                ),
                "near_threshold": 1.0,
            },
            "counts": {
                name: int(np.count_nonzero(quality_strata == name))
                for name in ("best", "intermediate", "near_threshold", "failed")
            },
            "by_section": {
                section: {
                    name: int(np.count_nonzero((section_ids == section) & (quality_strata == name)))
                    for name in ("best", "intermediate", "near_threshold", "failed")
                }
                for section in sorted(set(section_ids.tolist()))
            },
            "observation_stratum_manifest_sha256": quality_stratum_manifest_sha256,
        }
    else:
        nucleus_diameter = np.full(rows, np.nan)
        nearest_neighbor = np.full(rows, np.nan)
        nearest_neighbor_method = None
        diameter_relative_report = {"available": False, "pass": False}
        neighbor_relative_report = {"available": False, "pass": False}
        diameter_relative_rows = np.zeros(rows, dtype=np.bool_)
        neighbor_relative_rows = np.zeros(rows, dtype=np.bool_)
        quality_strata = np.full(rows, "failed", dtype="<U16")
        quality_strata_report = {"available": False}
    annotation_report, annotation_rows = _distance_qc(
        annotation_nucleus,
        maximum_p95=thresholds.maximum_annotation_nucleus_p95_um,
        maximum_outlier_fraction=thresholds.maximum_registration_outlier_fraction,
    )
    annotation_cell_report, annotation_cell_rows = _distance_qc(
        annotation_cell,
        maximum_p95=thresholds.maximum_annotation_cell_p95_um,
        maximum_outlier_fraction=thresholds.maximum_registration_outlier_fraction,
    )
    cell_nucleus_report, cell_nucleus_rows = _distance_qc(
        cell_nucleus,
        maximum_p95=thresholds.maximum_cell_nucleus_p95_um,
        maximum_outlier_fraction=thresholds.maximum_registration_outlier_fraction,
    )
    annotation_by_section = _distance_qc_by_section(
        annotation_nucleus,
        section_ids,
        maximum_p95=thresholds.maximum_annotation_nucleus_p95_um,
        maximum_outlier_fraction=thresholds.maximum_registration_outlier_fraction,
    )
    annotation_cell_by_section = _distance_qc_by_section(
        annotation_cell,
        section_ids,
        maximum_p95=thresholds.maximum_annotation_cell_p95_um,
        maximum_outlier_fraction=thresholds.maximum_registration_outlier_fraction,
    )
    cell_nucleus_by_section = _distance_qc_by_section(
        cell_nucleus,
        section_ids,
        maximum_p95=thresholds.maximum_cell_nucleus_p95_um,
        maximum_outlier_fraction=thresholds.maximum_registration_outlier_fraction,
    )
    cardinality = _vector(source, ("registration_cardinality", "registration_match_count"), rows)
    if cardinality is None:
        unique, inverse, counts = np.unique(cell_ids, return_inverse=True, return_counts=True)
        del unique
        cardinality = counts[inverse]
        cardinality_method = "inferred_from_cell_id_occurrence"
    else:
        cardinality_method = "source_registration_match_count"
    cardinality = np.asarray(cardinality)
    invalid_cardinality = int(np.count_nonzero(cardinality != 1))
    duplicate_observations = _duplicate_count(observations)
    duplicate_cells = _duplicate_count(cell_ids)
    distance_row_masks = (annotation_rows, annotation_cell_rows, cell_nucleus_rows)
    if all(len(value) == rows for value in distance_row_masks):
        registration_row_pass = (
            annotation_rows
            & annotation_cell_rows
            & cell_nucleus_rows
            & diameter_relative_rows
            & neighbor_relative_rows
            & (cardinality == 1)
        )
    else:
        registration_row_pass = np.zeros(rows, dtype=np.bool_)
    supplied_registration_qc = _vector(source, ("registration_qc_pass",), rows)
    registration_disagreements = None
    if supplied_registration_qc is not None:
        supplied_registration_qc = _booleans(supplied_registration_qc, "registration_qc_pass")
        registration_disagreements = int(
            np.count_nonzero(supplied_registration_qc & ~registration_row_pass)
        )
        registration_row_pass &= supplied_registration_qc
    section_registration_pass = bool(
        annotation_by_section
        and annotation_cell_by_section
        and cell_nucleus_by_section
        and all(report["pass"] for report in annotation_by_section.values())
        and all(report["pass"] for report in annotation_cell_by_section.values())
        and all(report["pass"] for report in cell_nucleus_by_section.values())
    )
    registration_pass = bool(
        duplicate_observations == 0
        and duplicate_cells == 0
        and invalid_cardinality == 0
        and annotation_report["pass"]
        and annotation_cell_report["pass"]
        and cell_nucleus_report["pass"]
        and diameter_relative_report["pass"]
        and neighbor_relative_report["pass"]
        and section_registration_pass
        and (registration_disagreements in (None, 0))
    )
    registration = {
        "rows": rows,
        "duplicate_observation_ids": duplicate_observations,
        "duplicate_cell_ids": duplicate_cells,
        "registration_cardinality_method": cardinality_method,
        "invalid_registration_cardinality_rows": invalid_cardinality,
        "annotation_to_nucleus_distance_um": annotation_report,
        "annotation_to_cell_distance_um": annotation_cell_report,
        "native_cell_to_nucleus_distance_um": cell_nucleus_report,
        "geometry_relative_registration": {
            "equivalent_nucleus_diameter_um": _quantiles(nucleus_diameter),
            "nearest_neighbor_nucleus_distance_um": _quantiles(nearest_neighbor),
            "nearest_neighbor_method": nearest_neighbor_method,
            "annotation_error_over_nucleus_diameter": diameter_relative_report,
            "annotation_error_over_nearest_neighbor_distance": (neighbor_relative_report),
            "quality_strata": quality_strata_report,
        },
        "by_section": {
            section: {
                "annotation_to_nucleus_distance_um": annotation_by_section[section],
                "annotation_to_cell_distance_um": annotation_cell_by_section[section],
                "native_cell_to_nucleus_distance_um": cell_nucleus_by_section[section],
                "invalid_registration_cardinality_rows": int(
                    np.count_nonzero((section_ids == section) & (cardinality != 1))
                ),
            }
            for section in sorted(annotation_by_section)
        },
        "median_annotation_nucleus_distance_um": annotation_report.get("median"),
        "p95_annotation_nucleus_distance_um": annotation_report.get("p95"),
        "maximum_allowed_p95_um": thresholds.maximum_annotation_nucleus_p95_um,
        "source_qc_disagreement_rows": registration_disagreements,
        "pass": registration_pass,
    }

    cell_area = _vector(source, ("cell_area_um2",), rows)
    nucleus_area = nucleus_area_geometry
    nucleus_inside = _vector(source, ("nucleus_centroid_inside_cell",), rows)
    if cell_area is None or nucleus_area is None or nucleus_inside is None:
        segmentation_row_pass = np.zeros(rows, dtype=np.bool_)
        segmentation = {"available": False, "pass": False}
    else:
        cell_area = cell_area.astype(np.float64)
        nucleus_area = nucleus_area.astype(np.float64)
        inside = _booleans(nucleus_inside, "nucleus_centroid_inside_cell")
        valid_area = (
            np.isfinite(cell_area)
            & np.isfinite(nucleus_area)
            & (cell_area > 0)
            & (nucleus_area > 0)
        )
        ratio = np.full(rows, np.nan, dtype=np.float64)
        ratio[valid_area] = nucleus_area[valid_area] / cell_area[valid_area]
        ratio_valid = (
            valid_area
            & (ratio >= thresholds.minimum_nucleus_cell_area_ratio)
            & (ratio <= thresholds.maximum_nucleus_cell_area_ratio)
        )
        segmentation_row_pass = inside & ratio_valid
        outside_fraction = float(np.mean(~inside))
        area_outlier_fraction = float(np.mean(~ratio_valid))
        segmentation_by_section = {}
        for section in sorted(set(section_ids.tolist())):
            selected = section_ids == section
            section_outside = float(np.mean(~inside[selected]))
            section_area_outliers = float(np.mean(~ratio_valid[selected]))
            segmentation_by_section[section] = {
                "rows": int(selected.sum()),
                "nucleus_centroid_outside_cell_fraction": section_outside,
                "nucleus_cell_area_ratio": _quantiles(ratio[selected]),
                "area_ratio_outlier_fraction": section_area_outliers,
                "pass": bool(
                    section_outside <= thresholds.maximum_nucleus_outside_cell_fraction
                    and section_area_outliers <= thresholds.maximum_segmentation_outlier_fraction
                ),
            }
        segmentation_pass = bool(
            outside_fraction <= thresholds.maximum_nucleus_outside_cell_fraction
            and area_outlier_fraction <= thresholds.maximum_segmentation_outlier_fraction
            and all(report["pass"] for report in segmentation_by_section.values())
        )
        segmentation = {
            "available": True,
            "nucleus_centroid_outside_cell_rows": int(np.count_nonzero(~inside)),
            "nucleus_centroid_outside_cell_fraction": outside_fraction,
            "maximum_allowed_outside_fraction": thresholds.maximum_nucleus_outside_cell_fraction,
            "nucleus_cell_area_ratio": _quantiles(ratio),
            "area_ratio_outlier_rows": int(np.count_nonzero(~ratio_valid)),
            "area_ratio_outlier_fraction": area_outlier_fraction,
            "minimum_allowed_area_ratio": thresholds.minimum_nucleus_cell_area_ratio,
            "maximum_allowed_area_ratio": thresholds.maximum_nucleus_cell_area_ratio,
            "maximum_allowed_area_outlier_fraction": (
                thresholds.maximum_segmentation_outlier_fraction
            ),
            "by_section": segmentation_by_section,
            "pass": segmentation_pass,
        }

    padding = _first(source, ("crop_padding_fraction", "crop_padding_fractions"))
    if padding is None:
        crop_row_pass = np.zeros(rows, dtype=np.bool_)
        crop_qc = {"available": False, "pass": False}
    else:
        padding = np.asarray(padding, dtype=np.float64)
        if padding.ndim == 1:
            padding = padding[:, None]
        if padding.ndim != 2 or len(padding) != rows:
            raise ValueError("crop padding fractions must be an observation-aligned matrix")
        crop_ids = _string_vector(source, ("crop_ids", "crop_names"))
        if crop_ids is None:
            crop_ids = np.asarray(["primary_crop"] if padding.shape[1] == 1 else [], dtype=str)
        if len(crop_ids) != padding.shape[1]:
            raise ValueError("crop padding columns differ from crop identities")
        valid_padding = np.isfinite(padding) & (padding >= 0) & (padding <= 1)
        crop_row_pass = np.all(
            valid_padding & (padding <= thresholds.maximum_crop_padding_p95), axis=1
        )
        per_crop = {}
        crop_components_pass = []
        for column, crop_id in enumerate(crop_ids.tolist()):
            values = padding[:, column]
            valid = valid_padding[:, column]
            summary = _quantiles(values[valid])
            mostly = float(np.mean(~valid | (values > thresholds.mostly_padded_cutoff)))
            crop_by_section = {}
            for section in sorted(set(section_ids.tolist())):
                selected = section_ids == section
                section_values = values[selected]
                section_valid = valid[selected]
                section_summary = _quantiles(section_values[section_valid])
                section_mostly = float(
                    np.mean(~section_valid | (section_values > thresholds.mostly_padded_cutoff))
                )
                crop_by_section[section] = {
                    "padding_fraction": section_summary,
                    "mostly_padded_fraction": section_mostly,
                    "pass": bool(
                        section_valid.all()
                        and section_summary["p95"] is not None
                        and float(section_summary["p95"]) <= thresholds.maximum_crop_padding_p95
                        and section_mostly <= thresholds.maximum_mostly_padded_fraction
                    ),
                }
            component_pass = bool(
                valid.all()
                and summary["p95"] is not None
                and float(summary["p95"]) <= thresholds.maximum_crop_padding_p95
                and mostly <= thresholds.maximum_mostly_padded_fraction
                and all(report["pass"] for report in crop_by_section.values())
            )
            crop_components_pass.append(component_pass)
            per_crop[crop_id] = {
                "padding_fraction": summary,
                "out_of_bounds_fraction": float(np.mean(valid & (values > 0))),
                "mostly_padded_fraction": mostly,
                "by_section": crop_by_section,
                "pass": component_pass,
            }
        mostly_rows = np.any(~valid_padding | (padding > thresholds.mostly_padded_cutoff), axis=1)
        crop_pass = bool(all(crop_components_pass))
        crop_qc = {
            "available": True,
            "crops": per_crop,
            "maximum_allowed_padding_p95": thresholds.maximum_crop_padding_p95,
            "mostly_padded_cutoff": thresholds.mostly_padded_cutoff,
            "maximum_allowed_mostly_padded_fraction": thresholds.maximum_mostly_padded_fraction,
            "mostly_padded_rows": int(np.count_nonzero(mostly_rows)),
            "pass": crop_pass,
        }

    gene_ids = _required_strings(source, ("ordered_gene_ids", "gene_ids"))
    if _duplicate_count(gene_ids):
        raise ValueError("ordered target genes must be unique")
    variants = tuple(str(value) for value in target_variants)
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("target variants must be unique and non-empty")
    transcript_ids = _string_vector(source, ("transcript_id", "transcript_ids"))
    raw_transcript_mode = transcript_ids is not None
    transcript_identity_manifest_sha256 = None
    receipt_minimum_qv = None
    if raw_transcript_mode:
        transcript_assignments = _required_strings(
            source,
            (
                "transcript_observation_id",
                "transcript_observation_ids",
                "transcript_cell_id",
                "transcript_cell_ids",
            ),
        )
        transcript_genes = _required_strings(
            source,
            ("transcript_gene_id", "transcript_gene_ids", "transcript_feature_name"),
        )
        if not (len(transcript_ids) == len(transcript_assignments) == len(transcript_genes)):
            raise ValueError("transcript identity arrays are not aligned")
        transcript_qv = _vector(
            source, ("transcript_qv", "transcript_qv_values"), len(transcript_ids)
        )
        if transcript_qv is None:
            invalid_qv = len(transcript_ids)
        else:
            qv_values = transcript_qv.astype(np.float64)
            invalid_qv = int(
                np.count_nonzero(
                    ~np.isfinite(qv_values) | (qv_values < thresholds.minimum_transcript_qv)
                )
            )
        duplicate_transcripts = _duplicate_count(transcript_ids)
        multiassigned = _multiassigned_transcripts(transcript_ids, transcript_assignments)
        observation_set = set(observations.tolist())
        cell_to_observation = dict(zip(cell_ids.tolist(), observations.tolist()))
        assignment_set = set(transcript_assignments.tolist())
        if assignment_set.issubset(observation_set):
            normalized_assignments = transcript_assignments
        elif assignment_set.issubset(set(cell_to_observation)):
            normalized_assignments = np.asarray(
                [cell_to_observation[value] for value in transcript_assignments.tolist()],
                dtype=str,
            )
        else:
            normalized_assignments = transcript_assignments
        unknown_assignment_rows = int(
            np.count_nonzero(
                [value not in observation_set for value in normalized_assignments.tolist()]
            )
        )
        gene_set = set(gene_ids.tolist())
        unknown_gene_rows = int(
            np.count_nonzero([value not in gene_set for value in transcript_genes.tolist()])
        )
        variant_membership = _variant_membership(source, variants, len(transcript_ids))
        transcript_count = len(transcript_ids)
        split_receipt_pass = variant_membership is not None
    else:
        transcript_assignments = None
        transcript_genes = None
        normalized_assignments = None
        variant_membership = None
        receipt_fields = {
            "transcript_count": _nonnegative_integer_scalar(
                source, ("eligible_target_transcripts",)
            ),
            "duplicate": _nonnegative_integer_scalar(source, ("duplicate_transcript_ids",)),
            "multiassigned": _nonnegative_integer_scalar(
                source, ("transcripts_assigned_to_multiple_cells",)
            ),
            "invalid_qv": _nonnegative_integer_scalar(source, ("invalid_qv_transcripts",)),
            "unknown_gene": _nonnegative_integer_scalar(source, ("unknown_gene_transcripts",)),
            "unknown_cell": _nonnegative_integer_scalar(source, ("unknown_cell_transcripts",)),
        }
        if any(value is None for value in receipt_fields.values()):
            raise ValueError("precomputed transcript split lacks identity-QC receipts")
        transcript_count = int(receipt_fields["transcript_count"])
        duplicate_transcripts = int(receipt_fields["duplicate"])
        multiassigned = int(receipt_fields["multiassigned"])
        invalid_qv = int(receipt_fields["invalid_qv"])
        unknown_gene_rows = int(receipt_fields["unknown_gene"])
        unknown_assignment_rows = int(receipt_fields["unknown_cell"])
        split_method = str(_scalar_value(source, ("transcript_split_method",)))
        minimum_qv_value = _scalar_value(source, ("transcript_minimum_qv",))
        receipt_minimum_qv = None if minimum_qv_value is None else float(minimum_qv_value)
        split_salt_sha256 = str(_scalar_value(source, ("transcript_split_salt_sha256",)))
        transcript_identity_manifest_sha256 = str(
            _scalar_value(source, ("transcript_identity_manifest_sha256",))
        )
        valid_identity_sha = len(transcript_identity_manifest_sha256) == 64 and all(
            value in "0123456789abcdef" for value in transcript_identity_manifest_sha256
        )
        split_receipt_pass = bool(
            split_method == SPLIT_HALF_METHOD
            and split_salt_sha256 == hashlib.sha256(split_salt.encode("utf-8")).hexdigest()
            and valid_identity_sha
            and receipt_minimum_qv is not None
            and np.isfinite(receipt_minimum_qv)
            and receipt_minimum_qv == thresholds.minimum_transcript_qv
        )
    frozen_programs = programs or _programs_from_source(source, gene_ids)
    technical_covariates_value = _first(source, ("technical_covariates",))
    if technical_covariates_value is None:
        technical_covariates = None
    else:
        technical_covariates = np.asarray(technical_covariates_value, dtype=np.float64)
        if (
            technical_covariates.ndim != 2
            or len(technical_covariates) != rows
            or not np.isfinite(technical_covariates).all()
        ):
            raise ValueError("technical covariates must be a finite row-aligned matrix")
    development = np.isin(donor_ids, np.asarray(tuple(development_donors), dtype=str))
    missing_development_donors = sorted(set(development_donors) - set(donor_ids.tolist()))
    type_selection_support = {}
    minimum_type_development_donors = max(
        thresholds.minimum_development_donors_per_fine_type,
        thresholds.minimum_reliable_donors_per_fine_type,
    )
    for fine_type in sorted(set(fine_types[development].tolist())):
        type_rows = fine_types == fine_type
        development_ids = sorted(set(donor_ids[type_rows].tolist()) & set(development_donors))
        supported = len(development_ids) >= minimum_type_development_donors
        type_selection_support[fine_type] = {
            "development_donor_ids": development_ids,
            "supported": supported,
            "selection_partition": "development_only",
        }
    supported_fine_types = [
        fine_type for fine_type, support in type_selection_support.items() if support["supported"]
    ]
    primary_variant_present = PRIMARY_TARGET_VARIANT in variants
    molecular_prerequisites_pass = bool(
        duplicate_transcripts == 0
        and multiassigned == 0
        and invalid_qv == 0
        and unknown_assignment_rows == 0
        and unknown_gene_rows == 0
        and split_receipt_pass
        and frozen_programs is not None
        and technical_covariates is not None
        and development.any()
        and not missing_development_donors
        and primary_variant_present
        and supported_fine_types
    )
    variant_reports = {}
    molecular_row_has_target = np.zeros(rows, dtype=np.bool_)
    processing_variants = (
        (PRIMARY_TARGET_VARIANT,)
        + tuple(value for value in variants if value != PRIMARY_TARGET_VARIANT)
        if primary_variant_present
        else variants
    )
    if molecular_prerequisites_pass:
        for variant in processing_variants:
            role = "primary" if variant == PRIMARY_TARGET_VARIANT else "secondary_sensitivity"
            affects_primary = variant == PRIMARY_TARGET_VARIANT
            try:
                variant_index = variants.index(variant)
                frozen_full_target = _frozen_full_target(
                    source,
                    variant,
                    rows=rows,
                    genes=len(gene_ids),
                    required=not raw_transcript_mode,
                )
                if raw_transcript_mode:
                    selected_transcripts = variant_membership[:, variant_index]
                    split = construct_split_half_counts(
                        transcript_ids[selected_transcripts],
                        normalized_assignments[selected_transcripts],
                        transcript_genes[selected_transcripts],
                        observations,
                        gene_ids,
                        salt=split_salt,
                    )
                    half_a = split.half_a
                    half_b = split.half_b
                    variant_transcripts = int(selected_transcripts.sum())
                    half_a_library = None
                    half_b_library = None
                    full_library = None
                    normalization_denominator = "target_panel_transcripts"
                else:
                    half_a, half_b, variant_transcripts = _precomputed_split_counts(
                        source, variant, rows=rows, genes=len(gene_ids)
                    )
                    half_a_library, half_b_library = _precomputed_half_libraries(
                        source,
                        variant,
                        half_a=half_a,
                        half_b=half_b,
                        rows=rows,
                    )
                    normalization_denominator = "frozen_target_library_transcripts"
                    full_library = half_a_library + half_b_library
                    if (
                        variant == SECONDARY_TARGET_VARIANT
                        and variant_transcripts != transcript_count
                    ):
                        raise ValueError(
                            "whole-cell split count differs from eligible transcript receipt"
                        )
                total = half_a + half_b
                if affects_primary:
                    molecular_row_has_target = total.sum(axis=1) > 0
                normalized_a = normalize_split_counts(half_a, library_sizes=half_a_library)
                normalized_b = normalize_split_counts(half_b, library_sizes=half_b_library)
                gene_report = feature_reliability(
                    normalized_a[development],
                    normalized_b[development],
                    gene_ids.tolist(),
                    minimum_rows=thresholds.minimum_reliability_rows,
                )
                program_a, program_names = program_scores(
                    normalized_a, gene_ids.tolist(), frozen_programs
                )
                program_b, second_program_names = program_scores(
                    normalized_b, gene_ids.tolist(), frozen_programs
                )
                if program_names != second_program_names:
                    raise RuntimeError("frozen program identities changed")
                residual_a = cross_fitted_residualize(
                    program_a,
                    technical_covariates,
                    donor_ids,
                    fine_types,
                    development_mask=development,
                    minimum_training_donors=max(
                        1, thresholds.minimum_reliable_donors_per_fine_type - 1
                    ),
                )
                residual_b = cross_fitted_residualize(
                    program_b,
                    technical_covariates,
                    donor_ids,
                    fine_types,
                    development_mask=development,
                    minimum_training_donors=max(
                        1, thresholds.minimum_reliable_donors_per_fine_type - 1
                    ),
                )
                if residual_a.fold_training_donors != residual_b.fold_training_donors:
                    raise RuntimeError("split-half residualization folds differ")
                finite_program_rows = (
                    development
                    & np.isfinite(residual_a.values).all(axis=1)
                    & np.isfinite(residual_b.values).all(axis=1)
                )
                program_report = feature_reliability(
                    residual_a.values[finite_program_rows],
                    residual_b.values[finite_program_rows],
                    program_names,
                    minimum_rows=thresholds.minimum_reliability_rows,
                )
                donor_macro_genes = _donor_macro_feature_reliability(
                    normalized_a,
                    normalized_b,
                    gene_ids.tolist(),
                    donor_ids,
                    development,
                    minimum_rows=thresholds.minimum_reliability_rows,
                )
                within_type_genes = _within_type_donor_macro_reliability(
                    normalized_a,
                    normalized_b,
                    gene_ids.tolist(),
                    donor_ids,
                    fine_types,
                    development,
                    minimum_rows=thresholds.minimum_reliability_rows,
                )
                donor_macro_programs = _donor_macro_feature_reliability(
                    residual_a.values,
                    residual_b.values,
                    program_names,
                    donor_ids,
                    development,
                    minimum_rows=thresholds.minimum_reliability_rows,
                )
                within_type_programs = _within_type_donor_macro_reliability(
                    residual_a.values,
                    residual_b.values,
                    program_names,
                    donor_ids,
                    fine_types,
                    development,
                    minimum_rows=thresholds.minimum_reliability_rows,
                )
                ceiling = dict(
                    cross_fitted_target_basis_reliability(
                        normalized_a,
                        normalized_b,
                        donor_ids,
                        development_mask=development,
                        rank=thresholds.target_basis_rank,
                        minimum_rows=thresholds.minimum_reliability_rows,
                        minimum_training_donors=max(
                            1, thresholds.minimum_reliable_development_donors - 1
                        ),
                        full_targets=frozen_full_target,
                    )
                )
                ceiling_component_coverage = all(
                    int(component["evaluable_donor_count"])
                    >= thresholds.minimum_reliable_development_donors
                    and float(component["evaluable_donor_fraction"])
                    >= thresholds.minimum_reliable_development_donor_fraction
                    for component in ceiling["components"].values()
                )
                ceiling["pass"] = bool(
                    ceiling["minimum_component_reliability"] is not None
                    and float(ceiling["minimum_component_reliability"])
                    >= thresholds.minimum_target_basis_ceiling
                    and ceiling_component_coverage
                )
                ceiling_by_type = _target_basis_ceiling_by_type(
                    normalized_a,
                    normalized_b,
                    frozen_full_target,
                    development,
                    donor_ids,
                    fine_types,
                    supported_fine_types,
                    rank=thresholds.target_basis_rank,
                    minimum_rows=thresholds.minimum_reliability_rows,
                    minimum_ceiling=thresholds.minimum_target_basis_ceiling,
                    minimum_donors=(thresholds.minimum_reliable_donors_per_fine_type),
                    minimum_donor_fraction=(thresholds.minimum_reliable_development_donor_fraction),
                )
                selected_genes = [
                    gene
                    for gene in gene_ids.tolist()
                    if _feature_meets_reliability_contract(
                        gene,
                        donor_macro_genes["features"][gene],
                        within_type_genes,
                        supported_fine_types,
                        minimum_reliability=(thresholds.minimum_median_gene_reliability),
                        minimum_development_donors=(thresholds.minimum_reliable_development_donors),
                        minimum_development_donor_fraction=(
                            thresholds.minimum_reliable_development_donor_fraction
                        ),
                        minimum_donors_per_fine_type=(
                            thresholds.minimum_reliable_donors_per_fine_type
                        ),
                        minimum_within_type_reliability=(
                            thresholds.minimum_within_fine_type_reliability
                        ),
                    )
                ]
                selected_programs = [
                    name
                    for name in program_names
                    if _feature_meets_reliability_contract(
                        name,
                        donor_macro_programs["features"][name],
                        within_type_programs,
                        supported_fine_types,
                        minimum_reliability=(thresholds.minimum_median_program_reliability),
                        minimum_development_donors=(thresholds.minimum_reliable_development_donors),
                        minimum_development_donor_fraction=(
                            thresholds.minimum_reliable_development_donor_fraction
                        ),
                        minimum_donors_per_fine_type=(
                            thresholds.minimum_reliable_donors_per_fine_type
                        ),
                        minimum_within_type_reliability=(
                            thresholds.minimum_within_fine_type_reliability
                        ),
                    )
                ]
                gene_fraction = len(selected_genes) / len(gene_ids)
                variant_pass = bool(
                    selected_genes
                    and gene_fraction >= thresholds.minimum_reliable_gene_fraction
                    and selected_programs
                    and ceiling["pass"]
                    and ceiling_by_type["pass"]
                )
                reliability_by_donor = _group_reliability(
                    normalized_a,
                    normalized_b,
                    gene_ids,
                    frozen_programs,
                    donor_ids,
                    minimum_rows=thresholds.minimum_reliability_rows,
                )
                variant_reports[variant] = {
                    "role": role,
                    "affects_primary_gate": affects_primary,
                    "transcripts": variant_transcripts,
                    "split_half_normalization_denominator": (normalization_denominator),
                    "target_genes_before_qc": int(len(gene_ids)),
                    "target_genes_after_qc": int(len(selected_genes)),
                    "ordered_reliable_gene_ids": selected_genes,
                    "ordered_reliable_gene_panel_sha256": _panel_sha256(selected_genes),
                    "ordered_reliable_program_ids": selected_programs,
                    "ordered_reliable_program_panel_sha256": _panel_sha256(selected_programs),
                    "reliable_gene_fraction": float(gene_fraction),
                    "development_gene_reliability": gene_report,
                    "development_residualized_program_reliability": program_report,
                    "development_donor_macro_gene_reliability": donor_macro_genes,
                    "development_donor_macro_residualized_program_reliability": (
                        donor_macro_programs
                    ),
                    "within_fine_type_gene_reliability": within_type_genes,
                    "within_fine_type_residualized_program_reliability": (within_type_programs),
                    "program_technical_residualization": {
                        "fit_partition": "leave_one_development_donor_out_within_fine_type",
                        "heldout_target_used_in_regression_fit": False,
                        "fold_training_donors": {
                            name: list(values)
                            for name, values in sorted(residual_a.fold_training_donors.items())
                        },
                    },
                    "target_basis_measurement_ceiling": ceiling,
                    "target_basis_measurement_ceiling_by_fine_type": ceiling_by_type,
                    "per_section_distributions": _distribution_by_section(
                        total, section_ids, library_sizes=full_library
                    ),
                    "reliability_by_section": _group_reliability(
                        normalized_a,
                        normalized_b,
                        gene_ids,
                        frozen_programs,
                        section_ids,
                        minimum_rows=thresholds.minimum_reliability_rows,
                    ),
                    "reliability_by_donor": reliability_by_donor,
                    "reliability_by_fine_type": _group_reliability(
                        normalized_a,
                        normalized_b,
                        gene_ids,
                        frozen_programs,
                        fine_types,
                        minimum_rows=thresholds.minimum_reliability_rows,
                    ),
                    "pass": variant_pass,
                }
            except ValueError as error:
                if affects_primary:
                    raise
                variant_reports[variant] = {
                    "role": role,
                    "affects_primary_gate": False,
                    "available": False,
                    "error": str(error),
                    "pass": False,
                }
    primary_report = variant_reports.get(PRIMARY_TARGET_VARIANT)
    primary_pass = bool(isinstance(primary_report, Mapping) and primary_report.get("pass") is True)
    reliable_genes = (
        list(primary_report["ordered_reliable_gene_ids"])
        if primary_pass and isinstance(primary_report, Mapping)
        else []
    )
    reliable_programs = (
        list(primary_report["ordered_reliable_program_ids"])
        if primary_pass and isinstance(primary_report, Mapping)
        else []
    )
    molecular_pass = bool(
        molecular_prerequisites_pass and primary_pass and reliable_genes and reliable_programs
    )
    target_selection_definition = {
        "schema": "heir.measurement_target_selection.v1",
        "pass": molecular_pass,
        "selection_partition": "development_only",
        "locked_test_molecular_outcomes_used": False,
        "primary_target_variant": PRIMARY_TARGET_VARIANT,
        "primary_gate_pass": primary_pass,
        "development_donor_ids": [str(value) for value in development_donors],
        "ordered_reliable_gene_ids": reliable_genes,
        "ordered_reliable_gene_panel_sha256": _panel_sha256(reliable_genes),
        "ordered_reliable_program_ids": reliable_programs,
        "ordered_reliable_program_panel_sha256": _panel_sha256(reliable_programs),
        "supported_fine_type_ids": supported_fine_types,
        "supported_fine_type_panel_sha256": _panel_sha256(supported_fine_types),
        "fine_type_partition_support": type_selection_support,
        "reliability_contract": {
            "minimum_development_donors": (thresholds.minimum_reliable_development_donors),
            "minimum_development_donor_fraction": (
                thresholds.minimum_reliable_development_donor_fraction
            ),
            "minimum_donors_per_fine_type": (thresholds.minimum_reliable_donors_per_fine_type),
            "minimum_gene_reliability": (thresholds.minimum_median_gene_reliability),
            "minimum_program_reliability": (thresholds.minimum_median_program_reliability),
            "minimum_within_fine_type_reliability": (
                thresholds.minimum_within_fine_type_reliability
            ),
        },
    }
    target_selection_core = {
        **target_selection_definition,
        "selection_core_sha256": _mapping_sha256(target_selection_definition),
        "study_manifest_sha256": study_manifest_sha256,
        "source_sha256": source_sha256,
        "transcript_identity_manifest_sha256": (transcript_identity_manifest_sha256),
        "transcript_split_salt_sha256": hashlib.sha256(split_salt.encode("utf-8")).hexdigest(),
    }
    target_selection_receipt = {
        **target_selection_core,
        "receipt_content_sha256": _mapping_sha256(target_selection_core),
    }
    program_summary = {
        variant: {
            "ordered_reliable_program_ids": report.get("ordered_reliable_program_ids", []),
            "pass": report.get("pass") is True,
        }
        for variant, report in variant_reports.items()
    }
    molecular = {
        "transcripts": int(transcript_count),
        "transcript_evidence_mode": (
            "raw_identity_rows"
            if raw_transcript_mode
            else "verified_split_count_sufficient_statistics"
        ),
        "transcript_identity_manifest_sha256": transcript_identity_manifest_sha256,
        "duplicate_transcript_ids": duplicate_transcripts,
        "transcripts_assigned_to_multiple_cells": int(multiassigned),
        "invalid_or_below_qv_transcripts": invalid_qv,
        "minimum_transcript_qv": thresholds.minimum_transcript_qv,
        "precomputed_receipt_minimum_qv": receipt_minimum_qv,
        "unknown_assignment_rows": unknown_assignment_rows,
        "unknown_gene_rows": unknown_gene_rows,
        "missing_development_donors": missing_development_donors,
        "split_half_method": SPLIT_HALF_METHOD,
        "split_salt_sha256": hashlib.sha256(split_salt.encode("utf-8")).hexdigest(),
        "target_genes_before_qc": int(len(gene_ids)),
        "target_genes_after_qc": int(len(reliable_genes)),
        "ordered_reliable_gene_ids": reliable_genes,
        "ordered_reliable_gene_panel_sha256": _panel_sha256(reliable_genes),
        "ordered_reliable_program_ids": reliable_programs,
        "ordered_reliable_program_panel_sha256": _panel_sha256(reliable_programs),
        "supported_fine_type_ids": supported_fine_types,
        "primary_target_gate": {
            "variant": PRIMARY_TARGET_VARIANT,
            "pass": primary_pass,
        },
        "secondary_target_gate": variant_reports.get(SECONDARY_TARGET_VARIANT),
        "secondary_target_gates": {
            variant: report
            for variant, report in variant_reports.items()
            if variant != PRIMARY_TARGET_VARIANT
        },
        "secondary_targets_affect_primary_gate": False,
        "target_selection_receipt": target_selection_receipt,
        "program_reliability": program_summary,
        "target_variants": variant_reports,
        "pass": molecular_pass,
    }

    normalized_roles = np.asarray(
        [
            "reference"
            if value in {"reference", "reference_pool", "ref"}
            else "evaluation"
            if value in {"evaluation", "evaluation_pool", "eval"}
            else "unknown"
            for value in pool_roles.tolist()
        ],
        dtype=str,
    )
    reference = normalized_roles == "reference"
    evaluation = normalized_roles == "evaluation"
    row_overlap = _intersection_count(observations[reference], observations[evaluation])
    cell_overlap = _intersection_count(cell_ids[reference], cell_ids[evaluation])
    block_overlap = _intersection_count(block_ids[reference], block_ids[evaluation])
    source_overlap = _intersection_count(source_files[reference], source_files[evaluation])
    separation_pass = bool(
        reference.any()
        and evaluation.any()
        and not np.any(normalized_roles == "unknown")
        and row_overlap <= thresholds.maximum_reference_evaluation_row_overlap
        and cell_overlap <= thresholds.maximum_reference_evaluation_row_overlap
        and block_overlap <= thresholds.maximum_reference_evaluation_block_overlap
        and source_overlap <= thresholds.maximum_reference_evaluation_source_file_overlap
    )
    separation = {
        "reference_rows": int(reference.sum()),
        "evaluation_rows": int(evaluation.sum()),
        "unknown_pool_role_rows": int(np.count_nonzero(normalized_roles == "unknown")),
        "reference_evaluation_observation_id_intersection": row_overlap,
        "reference_evaluation_cell_id_intersection": cell_overlap,
        "reference_evaluation_block_id_intersection": block_overlap,
        "reference_evaluation_source_file_intersection": source_overlap,
        "spatial_separation_verified_by_disjoint_blocks": bool(
            block_overlap <= thresholds.maximum_reference_evaluation_block_overlap
        ),
        "pass": separation_pass,
    }

    supplied_target_qc = _vector(source, ("target_qc_pass",), rows)
    target_row_pass = molecular_row_has_target
    target_qc_disagreements = None
    if supplied_target_qc is not None:
        supplied = _booleans(supplied_target_qc, "target_qc_pass")
        target_qc_disagreements = int(np.count_nonzero(supplied & ~target_row_pass))
        target_row_pass &= supplied
    provenance_value = _scalar_value(source, ("provenance_json",))
    source_exclusion_counts = None
    if provenance_value is not None:
        try:
            provenance = json.loads(str(provenance_value))
        except json.JSONDecodeError as error:
            raise ValueError("registered source provenance_json is invalid") from error
        exclusions = provenance.get("exclusion_counts") if isinstance(provenance, Mapping) else None
        if isinstance(exclusions, Mapping) and all(
            not isinstance(value, bool) and int(value) == value and int(value) >= 0
            for value in exclusions.values()
        ):
            source_exclusion_counts = {
                str(name): int(value) for name, value in sorted(exclusions.items())
            }
    retained_rows = registration_row_pass & segmentation_row_pass & crop_row_pass & target_row_pass
    stratum_ids = np.asarray(
        ["%s|%s|%s" % values for values in zip(donor_ids, section_ids, fine_types)], dtype=str
    )
    declared_planned = _string_vector(source, ("planned_stratum_ids",))
    planned_manifest_value = _scalar_value(source, ("planned_stratum_manifest_sha256",))
    if declared_planned is None:
        planned = sorted(set(stratum_ids.tolist()))
        coverage_plan_bound = raw_transcript_mode
        planned_manifest_sha256 = None
    else:
        if _duplicate_count(declared_planned):
            raise ValueError("planned biological coverage strata must be unique")
        planned = sorted(declared_planned.tolist())
        if any(len(value.split("|")) != 3 for value in planned):
            raise ValueError("planned coverage strata must be donor|section|fine_type")
        planned_donor_ids = {value.split("|", 1)[0] for value in planned}
        if planned_donor_ids & set(locked_test_donors):
            raise ValueError("H-MEAS coverage plan includes reserved locked donors")
        if not planned_donor_ids <= set(development_donors):
            raise ValueError("H-MEAS coverage plan includes non-development donors")
        planned_manifest_sha256 = str(planned_manifest_value)
        coverage_plan_bound = bool(
            len(planned_manifest_sha256) == 64 and planned_manifest_sha256 == _panel_sha256(planned)
        )
    unplanned_observed_strata = sorted(set(stratum_ids.tolist()) - set(planned))
    coverage_plan_bound &= not unplanned_observed_strata
    retained = []
    unevaluable = []
    support = {}
    for stratum in planned:
        selected = stratum_ids == stratum
        reference_cells = int(np.count_nonzero(selected & retained_rows & reference))
        evaluation_cells = int(np.count_nonzero(selected & retained_rows & evaluation))
        supported = bool(
            reference_cells >= thresholds.minimum_reference_cells_per_stratum
            and evaluation_cells >= thresholds.minimum_evaluation_cells_per_stratum
        )
        support[stratum] = {
            "planned_rows": int(selected.sum()),
            "reference_cells": reference_cells,
            "evaluation_cells": evaluation_cells,
            "supported": supported,
        }
        (retained if supported else unevaluable).append(stratum)
    planned_donors = sorted({value.split("|", 2)[0] for value in planned})
    planned_types = sorted({value.split("|", 2)[2] for value in planned})
    candidate_stratum_mask = np.isin(stratum_ids, np.asarray(retained, dtype=str))
    type_donor_coverage = {}
    unsupported_types = set()
    for fine_type in planned_types:
        selected = candidate_stratum_mask & (fine_types == fine_type)
        donors_for_type = set(donor_ids[selected].tolist())
        development_for_type = sorted(donors_for_type & set(development_donors))
        type_pass = bool(
            len(development_for_type) >= thresholds.minimum_development_donors_per_fine_type
        )
        if not type_pass:
            unsupported_types.add(fine_type)
        type_donor_coverage[fine_type] = {
            "development_donor_ids": development_for_type,
            "selection_partition": "development_only",
            "pass": type_pass,
        }
    if unsupported_types:
        donor_unsupported_strata = [
            value for value in retained if value.split("|", 2)[2] in unsupported_types
        ]
        for stratum in donor_unsupported_strata:
            support[stratum]["supported"] = False
            support[stratum]["donor_coverage_supported"] = False
        retained = [value for value in retained if value not in set(donor_unsupported_strata)]
        unevaluable = sorted(set(unevaluable) | set(donor_unsupported_strata))
    coverage_fraction = len(retained) / len(planned) if planned else 0.0
    retained_stratum_mask = np.isin(stratum_ids, np.asarray(retained, dtype=str))
    retained_donors = sorted(set(donor_ids[retained_stratum_mask].tolist()))
    retained_types = sorted(set(fine_types[retained_stratum_mask].tolist()))
    coverage_pass = bool(
        coverage_fraction >= thresholds.minimum_coverage_fraction
        and coverage_plan_bound
        and (raw_transcript_mode or source_exclusion_counts is not None)
        and (target_qc_disagreements in (None, 0))
    )
    retained_row_count = int(np.count_nonzero(retained_rows & retained_stratum_mask))
    coverage = {
        "planned_strata": int(len(planned)),
        "planned_stratum_manifest_sha256": planned_manifest_sha256,
        "coverage_plan_bound_before_exclusions": coverage_plan_bound,
        "unplanned_observed_stratum_ids": unplanned_observed_strata,
        "retained_strata": int(len(retained)),
        "unevaluable_strata": int(len(unevaluable)),
        "unevaluable_stratum_ids": unevaluable,
        "fraction_planned_biological_coverage_retained": float(coverage_fraction),
        "planned_rows": rows,
        "retained_rows": retained_row_count,
        "retained_row_fraction": float(retained_row_count / rows),
        "planned_donors": int(len(planned_donors)),
        "retained_donors": int(len(retained_donors)),
        "removed_donor_ids": sorted(set(planned_donors) - set(retained_donors)),
        "planned_fine_types": int(len(planned_types)),
        "retained_fine_types": int(len(retained_types)),
        "removed_fine_type_ids": sorted(set(planned_types) - set(retained_types)),
        "minimum_required_fraction": thresholds.minimum_coverage_fraction,
        "minimum_reference_cells_per_stratum": thresholds.minimum_reference_cells_per_stratum,
        "minimum_evaluation_cells_per_stratum": thresholds.minimum_evaluation_cells_per_stratum,
        "minimum_development_donors_per_fine_type": (
            thresholds.minimum_development_donors_per_fine_type
        ),
        "minimum_locked_donors_per_fine_type": (thresholds.minimum_locked_donors_per_fine_type),
        "locked_test_support_status": "not_opened",
        "fine_type_donor_coverage": type_donor_coverage,
        "support": support,
        "source_pre_artifact_exclusion_counts": source_exclusion_counts,
        "rows_removed": {
            "registration": int(np.count_nonzero(~registration_row_pass)),
            "segmentation": int(np.count_nonzero(~segmentation_row_pass)),
            "crop": int(np.count_nonzero(~crop_row_pass)),
            "target": int(np.count_nonzero(~target_row_pass)),
            "any_exclusion": int(np.count_nonzero(~retained_rows)),
        },
        "source_target_qc_disagreement_rows": target_qc_disagreements,
        "pass": coverage_pass,
    }

    passed = bool(
        registration["pass"]
        and molecular["pass"]
        and segmentation["pass"]
        and crop_qc["pass"]
        and separation["pass"]
        and coverage["pass"]
    )
    return {
        "schema": MEASUREMENT_GATE_SCHEMA,
        "hypothesis_ids": ["H-MEAS"],
        "study_manifest_sha256": study_manifest_sha256,
        "source_sha256": source_sha256,
        "thresholds": asdict(thresholds),
        "registration": registration,
        "molecular": molecular,
        "target_selection_receipt": target_selection_receipt,
        "segmentation": segmentation,
        "crop_qc": crop_qc,
        "reference_evaluation_separation": separation,
        "coverage": coverage,
        "locked_test_audit": {
            "status": "not_opened",
            "reserved_donor_ids": [str(value) for value in locked_test_donors],
            "source_locked_rows": 0,
            "source_declares_outcomes_materialized": False,
            "used_for_authorization": False,
        },
        "pass": passed,
        "authorizes_morphology_benchmark": passed,
    }


def _require_target_selection_receipt(report: Mapping[str, object]) -> None:
    receipt = report.get("target_selection_receipt")
    molecular = report.get("molecular")
    if not isinstance(receipt, Mapping) or not isinstance(molecular, Mapping):
        raise ValueError("measurement receipt lacks a target-selection receipt")
    if (
        receipt.get("schema") != "heir.measurement_target_selection.v1"
        or receipt.get("pass") is not True
        or receipt.get("selection_partition") != "development_only"
        or receipt.get("locked_test_molecular_outcomes_used") is not False
        or receipt.get("primary_target_variant") != PRIMARY_TARGET_VARIANT
        or receipt.get("primary_gate_pass") is not True
    ):
        raise ValueError("measurement target-selection receipt is not confirmatory")
    if receipt.get("study_manifest_sha256") != report.get("study_manifest_sha256") or receipt.get(
        "source_sha256"
    ) != report.get("source_sha256"):
        raise ValueError("measurement target selection belongs to different inputs")
    panel_fields = (
        ("ordered_reliable_gene_ids", "ordered_reliable_gene_panel_sha256"),
        ("ordered_reliable_program_ids", "ordered_reliable_program_panel_sha256"),
        ("supported_fine_type_ids", "supported_fine_type_panel_sha256"),
    )
    for panel_name, digest_name in panel_fields:
        values = receipt.get(panel_name)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
            or receipt.get(digest_name) != _panel_sha256(values)
        ):
            raise ValueError("measurement target-selection panel is malformed")
    claimed_digest = receipt.get("receipt_content_sha256")
    core = {str(name): value for name, value in receipt.items() if name != "receipt_content_sha256"}
    if claimed_digest != _mapping_sha256(core):
        raise ValueError("measurement target-selection receipt content hash differs")
    binding_fields = {
        "selection_core_sha256",
        "study_manifest_sha256",
        "source_sha256",
        "transcript_identity_manifest_sha256",
        "transcript_split_salt_sha256",
        "receipt_content_sha256",
    }
    selection_definition = {
        str(name): value for name, value in receipt.items() if name not in binding_fields
    }
    if receipt.get("selection_core_sha256") != _mapping_sha256(selection_definition):
        raise ValueError("measurement development-only selection core hash differs")
    if (
        molecular.get("target_selection_receipt") != receipt
        or molecular.get("ordered_reliable_gene_ids") != receipt.get("ordered_reliable_gene_ids")
        or molecular.get("ordered_reliable_program_ids")
        != receipt.get("ordered_reliable_program_ids")
        or molecular.get("supported_fine_type_ids") != receipt.get("supported_fine_type_ids")
    ):
        raise ValueError("measurement molecular report differs from target selection")
    variants = molecular.get("target_variants")
    primary = variants.get(PRIMARY_TARGET_VARIANT) if isinstance(variants, Mapping) else None
    if (
        not isinstance(primary, Mapping)
        or primary.get("role") != "primary"
        or primary.get("affects_primary_gate") is not True
        or primary.get("pass") is not True
        or primary.get("ordered_reliable_gene_ids") != receipt.get("ordered_reliable_gene_ids")
        or primary.get("ordered_reliable_program_ids")
        != receipt.get("ordered_reliable_program_ids")
    ):
        raise ValueError("measurement receipt lacks its passing primary nucleus gate")
    if isinstance(variants, Mapping):
        for name, variant in variants.items():
            if name == PRIMARY_TARGET_VARIANT:
                continue
            if not isinstance(variant, Mapping) or variant.get("affects_primary_gate") is not False:
                raise ValueError("a secondary target is allowed to affect the primary gate")


def require_passing_measurement_receipt(
    report: Mapping[str, object],
    *,
    expected_study_manifest_sha256: str,
    expected_source_sha256: Optional[str] = None,
) -> None:
    """Fail closed when a downstream benchmark lacks its exact G0 receipt."""

    if report.get("schema") != MEASUREMENT_GATE_SCHEMA or report.get("pass") is not True:
        raise ValueError("a passing heir.measurement_gate.v1 receipt is required")
    if report.get("study_manifest_sha256") != expected_study_manifest_sha256:
        raise ValueError("measurement receipt belongs to a different locked study manifest")
    if expected_source_sha256 is not None and report.get("source_sha256") != expected_source_sha256:
        raise ValueError("measurement receipt belongs to a different registered source")
    _require_target_selection_receipt(report)


def load_passing_measurement_receipt(
    path: PathLike,
    *,
    expected_receipt_sha256: str,
    expected_study_manifest_sha256: str,
    expected_source_sha256: Optional[str] = None,
) -> Mapping[str, object]:
    """Load a receipt only when its file hash and bound inputs match exactly."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError("measurement receipt is not an existing file")
    if sha256_file(resolved) != expected_receipt_sha256:
        raise ValueError("measurement receipt SHA-256 differs from the locked receipt")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("measurement receipt is not valid JSON") from error
    if not isinstance(report, Mapping):
        raise ValueError("measurement receipt must be a JSON object")
    require_passing_measurement_receipt(
        report,
        expected_study_manifest_sha256=expected_study_manifest_sha256,
        expected_source_sha256=expected_source_sha256,
    )
    return report


__all__ = [
    "MEASUREMENT_GATE_SCHEMA",
    "MeasurementThresholds",
    "evaluate_measurement_gate",
    "load_passing_measurement_receipt",
    "require_passing_measurement_receipt",
]
