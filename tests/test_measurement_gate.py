from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from heir.evaluation.measurement_gate import (
    MEASUREMENT_GATE_SCHEMA,
    MeasurementThresholds,
    evaluate_measurement_gate,
    load_passing_measurement_receipt,
    require_passing_measurement_receipt,
)
from heir.evaluation.reliability import (
    SPLIT_HALF_METHOD,
    construct_split_half_counts,
    deterministic_transcript_halves,
)
from heir.utils import sha256_file

PRIMARY_VARIANT = "nucleus_overlapping_transcripts"
SECONDARY_VARIANT = "whole_cell_assigned_transcripts"
VARIANTS = (PRIMARY_VARIANT, SECONDARY_VARIANT)
SPLIT_SALT = "locked-measurement-split"


def _thresholds() -> MeasurementThresholds:
    return MeasurementThresholds(
        maximum_annotation_nucleus_p95_um=1.0,
        maximum_annotation_cell_p95_um=1.0,
        maximum_cell_nucleus_p95_um=1.0,
        maximum_registration_nucleus_diameter_ratio_p95=0.25,
        maximum_registration_nearest_neighbor_ratio_p95=0.25,
        best_registration_quality_max_fraction_of_limit=1.0 / 3.0,
        intermediate_registration_quality_max_fraction_of_limit=2.0 / 3.0,
        maximum_registration_outlier_fraction=0.0,
        maximum_nucleus_outside_cell_fraction=0.0,
        minimum_nucleus_cell_area_ratio=0.1,
        maximum_nucleus_cell_area_ratio=0.9,
        maximum_segmentation_outlier_fraction=0.0,
        maximum_crop_padding_p95=0.1,
        mostly_padded_cutoff=0.5,
        maximum_mostly_padded_fraction=0.0,
        minimum_transcript_qv=20.0,
        required_opposite_pool_guard_um=20.0,
        minimum_median_gene_reliability=0.8,
        minimum_median_program_reliability=0.8,
        minimum_target_basis_ceiling=0.8,
        minimum_reliable_gene_fraction=1.0,
        minimum_reliable_development_donor_fraction=1.0,
        minimum_within_fine_type_reliability=0.8,
        minimum_reliability_rows=4,
        target_basis_rank=1,
        minimum_reliable_development_donors=3,
        minimum_reliable_donors_per_fine_type=3,
        minimum_coverage_fraction=1.0,
        minimum_reference_cells_per_stratum=4,
        minimum_evaluation_cells_per_stratum=4,
        minimum_development_donors_per_fine_type=3,
        minimum_locked_donors_per_fine_type=0,
        maximum_reference_evaluation_row_overlap=0,
        maximum_reference_evaluation_block_overlap=0,
        same_section_source_overlap_allowed=True,
    )


def _transcript_id(prefix: str, desired_half: int, used: set[str]) -> str:
    index = 0
    while True:
        candidate = "%s-%d" % (prefix, index)
        index += 1
        if candidate in used:
            continue
        if int(deterministic_transcript_halves([candidate], salt=SPLIT_SALT)[0]) == desired_half:
            used.add(candidate)
            return candidate


def _source() -> dict[str, np.ndarray]:
    rows = 24
    observations = np.asarray(["o%02d" % index for index in range(rows)])
    donors = np.asarray(["d1"] * 8 + ["d2"] * 8 + ["d3"] * 8)
    sections = np.asarray(["s1"] * 8 + ["s2"] * 8 + ["s3"] * 8)
    roles = np.asarray(["reference"] * 4 + ["evaluation"] * 4)  # repeated per donor
    roles = np.tile(roles, 3)
    blocks = np.asarray(
        ["s1-reference"] * 4
        + ["s1-evaluation"] * 4
        + ["s2-reference"] * 4
        + ["s2-evaluation"] * 4
        + ["s3-reference"] * 4
        + ["s3-evaluation"] * 4
    )
    transcript_ids: list[str] = []
    transcript_observations: list[str] = []
    transcript_genes: list[str] = []
    used: set[str] = set()
    for row, observation in enumerate(observations.tolist()):
        counts = {"g1": row % 4 + 1, "g2": 4 - row % 4}
        for gene, count in counts.items():
            for half in (0, 1):
                for copy_index in range(count):
                    transcript_ids.append(
                        _transcript_id(
                            "%s-%s-h%d-c%d" % (observation, gene, half, copy_index),
                            half,
                            used,
                        )
                    )
                    transcript_observations.append(observation)
                    transcript_genes.append(gene)
    transcript_count = len(transcript_ids)
    return {
        "study_stage": np.asarray("measurement_development"),
        "source_scope": np.asarray("development_donors_only"),
        "study_manifest_sha256": np.asarray("a" * 64),
        "locked_donor_outcomes_materialized": np.asarray(False),
        "opposite_pool_guard_um": np.asarray(20.0),
        "observation_id": observations,
        "cell_id": np.asarray(["cell-" + value for value in observations]),
        "donor_id": donors,
        "split_ids": np.asarray(["development"] * rows),
        "section_id": sections,
        "fine_type_label": np.asarray(["epithelial"] * rows),
        "block_id": blocks,
        "pool_role": roles,
        "source_file_id": sections.copy(),
        "registration_distance_um": np.full(rows, 0.1),
        "annotation_cell_distance_um": np.full(rows, 0.1),
        "cell_nucleus_centroid_distance_um": np.full(rows, 0.1),
        "registration_cardinality": np.ones(rows, dtype=np.int64),
        "registration_qc_pass": np.ones(rows, dtype=np.bool_),
        "cell_area_um2": np.full(rows, 100.0),
        "nucleus_area_um2": np.full(rows, 50.0),
        "nucleus_centroid_x_um": np.tile(np.arange(8, dtype=np.float64) * 10.0, 3),
        "nucleus_centroid_y_um": np.concatenate(
            (
                np.zeros(8, dtype=np.float64),
                np.full(8, 100.0),
                np.full(8, 200.0),
            )
        ),
        "nucleus_centroid_inside_cell": np.ones(rows, dtype=np.bool_),
        "crop_padding_fraction": np.zeros((rows, 1)),
        "crop_ids": np.asarray(["cell_context_112um"]),
        "target_qc_pass": np.ones(rows, dtype=np.bool_),
        "transcript_id": np.asarray(transcript_ids),
        "transcript_observation_id": np.asarray(transcript_observations),
        "transcript_gene_id": np.asarray(transcript_genes),
        "transcript_qv": np.full(transcript_count, 30.0),
        "ordered_gene_ids": np.asarray(["g1", "g2"]),
        "target_variant_names": np.asarray(VARIANTS),
        "target_variant_membership": np.ones((transcript_count, 2), dtype=np.bool_),
        "program_names": np.asarray(["g1_program"]),
        "program_gene_membership": np.asarray([[1.0, 0.0]]),
        "technical_covariates": np.zeros((rows, 1), dtype=np.float64),
    }


def _evaluate(
    source: dict[str, np.ndarray],
    thresholds: Optional[MeasurementThresholds] = None,
) -> dict[str, object]:
    return dict(
        evaluate_measurement_gate(
            source,
            _thresholds() if thresholds is None else thresholds,
            development_donors=("d1", "d2", "d3"),
            locked_test_donors=("d4",),
            target_variants=VARIANTS,
            split_salt=SPLIT_SALT,
            study_manifest_sha256="a" * 64,
            source_sha256="b" * 64,
        )
    )


def _compact_source(
    source: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, np.ndarray]:
    source = _source() if source is None else source
    split = construct_split_half_counts(
        source["transcript_id"],
        source["transcript_observation_id"],
        source["transcript_gene_id"],
        source["observation_id"],
        source["ordered_gene_ids"],
        salt=SPLIT_SALT,
    )
    for prefix, normalized_name in (
        ("nucleus", "normalized_nucleus_targets"),
        ("whole_cell", "normalized_whole_cell_targets"),
    ):
        source[prefix + "_target_counts_half_a"] = split.half_a.copy()
        source[prefix + "_target_counts_half_b"] = split.half_b.copy()
        source[prefix + "_target_counts"] = split.half_a + split.half_b
        source[prefix + "_library_size_half_a"] = split.half_a.sum(axis=1)
        source[prefix + "_library_size_half_b"] = split.half_b.sum(axis=1)
        source[prefix + "_library_sizes"] = source[prefix + "_target_counts"].sum(axis=1)
        source[normalized_name] = np.log1p(
            source[prefix + "_target_counts"].astype(np.float64)
            * (10_000.0 / source[prefix + "_library_sizes"][:, None])
        )
    source["eligible_target_transcripts"] = np.asarray(len(source["transcript_id"]))
    source["duplicate_transcript_ids"] = np.asarray(0)
    source["transcripts_assigned_to_multiple_cells"] = np.asarray(0)
    source["invalid_qv_transcripts"] = np.asarray(0)
    source["unknown_gene_transcripts"] = np.asarray(0)
    source["unknown_cell_transcripts"] = np.asarray(0)
    source["transcript_split_method"] = np.asarray(SPLIT_HALF_METHOD)
    source["transcript_minimum_qv"] = np.asarray(20.0)
    source["transcript_split_salt_sha256"] = np.asarray(
        hashlib.sha256(SPLIT_SALT.encode("utf-8")).hexdigest()
    )
    source["transcript_identity_manifest_sha256"] = np.asarray("c" * 64)
    planned = sorted(
        {
            "%s|%s|%s" % values
            for values in zip(
                source["donor_id"].tolist(),
                source["section_id"].tolist(),
                source["fine_type_label"].tolist(),
            )
        }
    )
    source["planned_stratum_ids"] = np.asarray(planned)
    source["planned_stratum_manifest_sha256"] = np.asarray(
        hashlib.sha256(
            json.dumps(planned, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    source["provenance_json"] = np.asarray(
        json.dumps(
            {
                "exclusion_counts": {
                    "low_transcripts": 0,
                    "spatial_guard": 0,
                    "unsupported_donor_fine_type": 0,
                }
            }
        )
    )
    for key in (
        "transcript_id",
        "transcript_observation_id",
        "transcript_gene_id",
        "transcript_qv",
        "target_variant_names",
        "target_variant_membership",
    ):
        del source[key]
    return source


def _refresh_compact_variant(source: dict[str, np.ndarray], prefix: str) -> None:
    half_a = source[prefix + "_target_counts_half_a"]
    half_b = source[prefix + "_target_counts_half_b"]
    total = half_a + half_b
    source[prefix + "_target_counts"] = total
    source[prefix + "_library_size_half_a"] = half_a.sum(axis=1)
    source[prefix + "_library_size_half_b"] = half_b.sum(axis=1)
    source[prefix + "_library_sizes"] = total.sum(axis=1)
    normalized_name = (
        "normalized_nucleus_targets" if prefix == "nucleus" else "normalized_whole_cell_targets"
    )
    source[normalized_name] = np.log1p(
        total.astype(np.float64) * (10_000.0 / source[prefix + "_library_sizes"][:, None])
    )


def test_measurement_gate_reports_all_required_domains_and_passes() -> None:
    report = _evaluate(_source())
    assert report["schema"] == MEASUREMENT_GATE_SCHEMA
    assert report["pass"] is True
    assert report["registration"]["duplicate_cell_ids"] == 0
    assert set(report["registration"]["by_section"]) == {"s1", "s2", "s3"}
    assert report["molecular"]["duplicate_transcript_ids"] == 0
    assert report["molecular"]["target_genes_after_qc"] == 2
    variant = report["molecular"]["target_variants"][PRIMARY_VARIANT]
    assert set(variant["per_section_distributions"]) == {"s1", "s2", "s3"}
    assert variant["target_basis_measurement_ceiling"][
        "median_component_reliability"
    ] == pytest.approx(1.0)
    assert report["target_selection_receipt"]["selection_partition"] == "development_only"
    assert report["molecular"]["secondary_targets_affect_primary_gate"] is False
    assert (
        report["reference_evaluation_separation"]["reference_evaluation_block_id_intersection"] == 0
    )
    assert report["coverage"]["fraction_planned_biological_coverage_retained"] == 1.0


def test_target_selection_uses_only_rows_passing_frozen_qc() -> None:
    source = _source()
    failed_rows = np.asarray([0, 8, 16])
    source["registration_qc_pass"][failed_rows] = False
    thresholds = replace(
        _thresholds(),
        maximum_registration_outlier_fraction=0.13,
        minimum_reference_cells_per_stratum=3,
    )

    report = _evaluate(source, thresholds)
    selection = report["target_selection_receipt"]
    primary = report["molecular"]["target_variants"][PRIMARY_VARIANT]

    assert report["registration"]["pass"] is True
    assert selection["development_rows_before_frozen_qc"] == 24
    assert selection["development_rows_after_frozen_nonmolecular_qc"] == 21
    assert primary["development_rows_after_frozen_qc_and_target_presence"] == 21
    assert report["coverage"]["rows_removed"]["registration"] == 3


def test_same_section_source_overlap_is_allowed_only_with_disjoint_cells_and_blocks() -> None:
    report = _evaluate(_source())
    separation = report["reference_evaluation_separation"]

    assert separation["reference_evaluation_source_file_intersection"] == 3
    assert separation["same_source_file_overlap_allowed_by_design"] is True
    assert separation["source_file_overlap_affects_pass"] is False
    assert separation["reference_evaluation_observation_id_intersection"] == 0
    assert separation["reference_evaluation_cell_id_intersection"] == 0
    assert separation["reference_evaluation_block_id_intersection"] == 0
    assert separation["spatial_separation_verified_by_disjoint_blocks"] is True
    assert separation["pass"] is True
    assert report["pass"] is True


def test_same_section_separation_requires_the_exact_frozen_physical_guard() -> None:
    source = _source()
    source["opposite_pool_guard_um"] = np.asarray(
        _thresholds().required_opposite_pool_guard_um + np.finfo(np.float64).eps * 128
    )

    report = _evaluate(source)

    separation = report["reference_evaluation_separation"]
    assert separation["opposite_pool_guard_matches_frozen_design"] is False
    assert separation["pass"] is False
    assert report["pass"] is False


def test_measurement_gate_fails_duplicate_transcript_identity() -> None:
    source = _source()
    for key in (
        "transcript_id",
        "transcript_observation_id",
        "transcript_gene_id",
        "transcript_qv",
    ):
        source[key] = np.concatenate((source[key], source[key][:1]))
    source["target_variant_membership"] = np.concatenate(
        (source["target_variant_membership"], source["target_variant_membership"][:1]), axis=0
    )
    report = _evaluate(source)
    assert report["molecular"]["duplicate_transcript_ids"] == 1
    assert report["molecular"]["pass"] is False
    assert report["pass"] is False


def test_compact_precomputed_split_counts_match_raw_identity_gate() -> None:
    source = _source()
    raw_report = _evaluate(source)
    source = _compact_source(source)
    compact_report = _evaluate(source)
    assert compact_report["pass"] is True
    assert compact_report["molecular"]["transcript_evidence_mode"].startswith("verified")
    assert (
        compact_report["molecular"]["ordered_reliable_gene_ids"]
        == raw_report["molecular"]["ordered_reliable_gene_ids"]
    )


def test_reserved_locked_rows_are_a_protocol_violation() -> None:
    source = _source()
    source["donor_id"] = source["donor_id"].astype("<U8")
    source["donor_id"][source["donor_id"] == "d3"] = "d4"
    source["split_ids"] = source["split_ids"].astype("<U16")
    source["split_ids"][source["donor_id"] == "d4"] = "locked_test"
    with pytest.raises(ValueError, match="split_ids|reserved locked"):
        _evaluate(source)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("study_stage", "confirmatory_morphology", "measurement_development"),
        ("source_scope", "all_donors", "development-donor-only"),
        (
            "locked_donor_outcomes_materialized",
            True,
            "locked donor outcomes stayed unopened",
        ),
        ("study_manifest_sha256", "c" * 64, "measurement study manifest"),
    ),
)
def test_measurement_source_requires_development_only_protocol_markers(
    field: str, value: object, message: str
) -> None:
    source = _source()
    source[field] = np.asarray(value)
    with pytest.raises(ValueError, match=message):
        _evaluate(source)


def test_gene_must_be_reliable_in_minimum_development_donors() -> None:
    source = _source()
    observations = source["transcript_observation_id"]
    genes = source["transcript_gene_id"]
    keep = ~((np.asarray([8 <= int(value[1:]) < 16 for value in observations])) & (genes == "g1"))
    for key in (
        "transcript_id",
        "transcript_observation_id",
        "transcript_gene_id",
        "transcript_qv",
    ):
        source[key] = source[key][keep]
    source["target_variant_membership"] = source["target_variant_membership"][keep]
    report = _evaluate(source)
    primary = report["molecular"]["target_variants"][PRIMARY_VARIANT]
    g1 = primary["development_donor_macro_gene_reliability"]["features"]["g1"]
    assert g1["evaluable_development_donor_count"] == 2
    assert "g1" not in primary["ordered_reliable_gene_ids"]


def test_program_reliability_is_residualized_and_required_within_fine_type() -> None:
    source = _compact_source()
    source["fine_type_label"] = np.tile(np.asarray(["type_a"] * 4 + ["type_b"] * 4), 3)
    source["pool_role"] = np.tile(
        np.asarray(
            [
                "reference",
                "reference",
                "evaluation",
                "evaluation",
                "reference",
                "reference",
                "evaluation",
                "evaluation",
            ]
        ),
        3,
    )
    source["block_id"] = np.asarray(
        [
            "%s-%s" % (section, role)
            for section, role in zip(source["section_id"], source["pool_role"])
        ]
    )
    for prefix in ("nucleus", "whole_cell"):
        first = np.zeros((24, 2), dtype=np.uint32)
        second = np.zeros_like(first)
        for donor_start in (0, 8, 16):
            first_g1 = np.asarray([10, 20, 30, 40, 160, 170, 180, 190])
            second_g1 = np.asarray([40, 30, 20, 10, 190, 180, 170, 160])
            first[donor_start : donor_start + 8, 0] = first_g1
            first[donor_start : donor_start + 8, 1] = 200 - first_g1
            second[donor_start : donor_start + 8, 0] = second_g1
            second[donor_start : donor_start + 8, 1] = 200 - second_g1
        source[prefix + "_target_counts_half_a"] = first
        source[prefix + "_target_counts_half_b"] = second
        _refresh_compact_variant(source, prefix)
    source["eligible_target_transcripts"] = np.asarray(
        int(source["whole_cell_target_counts"].sum())
    )
    planned = sorted(
        {
            "%s|%s|%s" % values
            for values in zip(
                source["donor_id"].tolist(),
                source["section_id"].tolist(),
                source["fine_type_label"].tolist(),
            )
        }
    )
    source["planned_stratum_ids"] = np.asarray(planned)
    source["planned_stratum_manifest_sha256"] = np.asarray(
        hashlib.sha256(json.dumps(planned, separators=(",", ":")).encode("utf-8")).hexdigest()
    )
    report = _evaluate(
        source,
        replace(
            _thresholds(),
            minimum_reference_cells_per_stratum=2,
            minimum_evaluation_cells_per_stratum=2,
        ),
    )
    primary = report["molecular"]["target_variants"][PRIMARY_VARIANT]
    pooled = primary["reliability_by_donor"]["d1"]["programs"]["g1_program"]
    assert pooled["spearman_brown_reliability"] > 0.8
    within = primary["within_fine_type_residualized_program_reliability"]
    assert (
        within["type_a"]["features"]["g1_program"]["donor_macro_spearman_brown_reliability"] == 0.0
    )
    assert primary["ordered_reliable_program_ids"] == []


def test_supported_fine_types_require_primary_target_qualified_pool_support() -> None:
    source = _source()
    source["fine_type_label"] = np.tile(np.asarray(["type_a", "type_a", "type_b", "type_b"] * 2), 3)
    type_b_observations = set(
        source["observation_id"][source["fine_type_label"] == "type_b"].tolist()
    )
    keep = np.asarray(
        [value not in type_b_observations for value in source["transcript_observation_id"]]
    )
    for key in (
        "transcript_id",
        "transcript_observation_id",
        "transcript_gene_id",
        "transcript_qv",
    ):
        source[key] = source[key][keep]
    source["target_variant_membership"] = source["target_variant_membership"][keep]
    source["target_qc_pass"][source["fine_type_label"] == "type_b"] = False

    report = _evaluate(
        source,
        replace(
            _thresholds(),
            minimum_reference_cells_per_stratum=2,
            minimum_evaluation_cells_per_stratum=2,
            minimum_coverage_fraction=0.5,
        ),
    )
    selection = report["target_selection_receipt"]

    assert report["pass"] is True
    assert selection["supported_fine_type_ids"] == ["type_a"]
    assert selection["fine_type_partition_support"]["type_b"]["supported"] is False
    assert (
        selection["nonmolecular_precheck_fine_type_partition_support"]["type_b"]["supported"]
        is True
    )


def test_secondary_whole_cell_failure_cannot_redefine_primary_gate() -> None:
    source = _compact_source()
    whole_b = source["whole_cell_target_counts_half_b"].copy()
    for start in (0, 8, 16):
        whole_b[start : start + 8] = whole_b[start : start + 8][::-1]
    source["whole_cell_target_counts_half_b"] = whole_b
    _refresh_compact_variant(source, "whole_cell")
    report = _evaluate(source)
    primary = report["molecular"]["target_variants"][PRIMARY_VARIANT]
    secondary = report["molecular"]["target_variants"][SECONDARY_VARIANT]
    assert primary["pass"] is True
    assert secondary["pass"] is False
    assert secondary["affects_primary_gate"] is False
    assert report["molecular"]["ordered_reliable_gene_ids"] == primary["ordered_reliable_gene_ids"]
    assert report["pass"] is True


def test_measurement_gate_fails_registration_and_crop_thresholds() -> None:
    source = _source()
    source["registration_distance_um"][:] = 2.0
    source["registration_qc_pass"][:] = False
    source["crop_padding_fraction"][:] = 0.75
    report = _evaluate(source)
    assert report["registration"]["pass"] is False
    assert report["crop_qc"]["pass"] is False
    assert report["coverage"]["retained_strata"] == 0
    assert report["pass"] is False


def test_registration_gate_uses_geometry_relative_error_and_quality_strata() -> None:
    baseline = _evaluate(_source())
    assert (
        baseline["registration"]["geometry_relative_registration"]["quality_strata"]["counts"][
            "best"
        ]
        == 24
    )
    source = _source()
    source["registration_distance_um"][:] = 0.95
    source["nucleus_area_um2"][:] = 10.0
    report = _evaluate(source)
    assert report["registration"]["annotation_to_nucleus_distance_um"]["pass"] is True
    relative = report["registration"]["geometry_relative_registration"]
    assert relative["annotation_error_over_nucleus_diameter"]["pass"] is False
    assert relative["quality_strata"]["counts"]["failed"] == 24
    assert report["registration"]["pass"] is False


def test_measurement_receipt_is_bound_to_manifest_source_and_file_sha(
    tmp_path: Path,
) -> None:
    report = _evaluate(_source())
    require_passing_measurement_receipt(
        report,
        expected_study_manifest_sha256="a" * 64,
        expected_source_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="different locked study"):
        require_passing_measurement_receipt(
            report,
            expected_study_manifest_sha256="c" * 64,
        )
    failed = copy.deepcopy(report)
    failed["pass"] = False
    with pytest.raises(ValueError, match="passing"):
        require_passing_measurement_receipt(
            failed,
            expected_study_manifest_sha256="a" * 64,
        )
    tampered = copy.deepcopy(report)
    tampered["target_selection_receipt"]["reliability_contract"]["minimum_gene_reliability"] = 0.0
    with pytest.raises(ValueError, match="content hash"):
        require_passing_measurement_receipt(
            tampered,
            expected_study_manifest_sha256="a" * 64,
        )
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    loaded = load_passing_measurement_receipt(
        path,
        expected_receipt_sha256=sha256_file(path),
        expected_study_manifest_sha256="a" * 64,
        expected_source_sha256="b" * 64,
    )
    assert loaded["pass"] is True
    with pytest.raises(ValueError, match="receipt SHA-256"):
        load_passing_measurement_receipt(
            path,
            expected_receipt_sha256="d" * 64,
            expected_study_manifest_sha256="a" * 64,
        )


def test_manifest_threshold_parser_requires_every_explicit_decision() -> None:
    decisions = {
        key: value
        for key, value in _thresholds().__dict__.items()
        if key
        not in {
            "minimum_coverage_fraction",
            "minimum_reference_cells_per_stratum",
            "minimum_evaluation_cells_per_stratum",
            "minimum_development_donors_per_fine_type",
            "minimum_locked_donors_per_fine_type",
            "maximum_reference_evaluation_row_overlap",
            "maximum_reference_evaluation_block_overlap",
            "same_section_source_overlap_allowed",
        }
    }
    coverage = {key: value for key, value in _thresholds().__dict__.items() if key not in decisions}
    parsed = MeasurementThresholds.from_study_manifest(
        {"decision_thresholds": {"measurement": decisions}, "coverage_requirements": coverage}
    )
    assert parsed == _thresholds()
    del decisions["minimum_target_basis_ceiling"]
    with pytest.raises(ValueError, match="explicit measurement thresholds"):
        MeasurementThresholds.from_study_manifest(
            {"decision_thresholds": decisions, "coverage_requirements": coverage}
        )
