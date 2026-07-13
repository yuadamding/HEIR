from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

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

VARIANT = "whole_cell_assigned_transcripts"
SPLIT_SALT = "locked-measurement-split"


def _thresholds() -> MeasurementThresholds:
    return MeasurementThresholds(
        maximum_annotation_nucleus_p95_um=1.0,
        maximum_annotation_cell_p95_um=1.0,
        maximum_cell_nucleus_p95_um=1.0,
        maximum_registration_outlier_fraction=0.0,
        maximum_nucleus_outside_cell_fraction=0.0,
        minimum_nucleus_cell_area_ratio=0.1,
        maximum_nucleus_cell_area_ratio=0.9,
        maximum_segmentation_outlier_fraction=0.0,
        maximum_crop_padding_p95=0.1,
        mostly_padded_cutoff=0.5,
        maximum_mostly_padded_fraction=0.0,
        minimum_transcript_qv=20.0,
        minimum_median_gene_reliability=0.8,
        minimum_median_program_reliability=0.8,
        minimum_target_basis_ceiling=0.8,
        minimum_reliable_gene_fraction=1.0,
        minimum_reliability_rows=4,
        target_basis_rank=1,
        minimum_coverage_fraction=1.0,
        minimum_reference_cells_per_stratum=4,
        minimum_evaluation_cells_per_stratum=4,
        minimum_development_donors_per_fine_type=1,
        minimum_locked_donors_per_fine_type=1,
        maximum_reference_evaluation_row_overlap=0,
        maximum_reference_evaluation_block_overlap=0,
        maximum_reference_evaluation_source_file_overlap=2,
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
    rows = 16
    observations = np.asarray(["o%02d" % index for index in range(rows)])
    donors = np.asarray(["d1"] * 8 + ["d2"] * 8)
    sections = np.asarray(["s1"] * 8 + ["s2"] * 8)
    roles = np.asarray(["reference"] * 4 + ["evaluation"] * 4)  # repeated per donor
    roles = np.concatenate((roles, roles))
    blocks = np.asarray(
        ["s1-reference"] * 4
        + ["s1-evaluation"] * 4
        + ["s2-reference"] * 4
        + ["s2-evaluation"] * 4
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
        "observation_id": observations,
        "cell_id": np.asarray(["cell-" + value for value in observations]),
        "donor_id": donors,
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
        "nucleus_centroid_inside_cell": np.ones(rows, dtype=np.bool_),
        "crop_padding_fraction": np.zeros((rows, 1)),
        "crop_ids": np.asarray(["cell_context_112um"]),
        "target_qc_pass": np.ones(rows, dtype=np.bool_),
        "transcript_id": np.asarray(transcript_ids),
        "transcript_observation_id": np.asarray(transcript_observations),
        "transcript_gene_id": np.asarray(transcript_genes),
        "transcript_qv": np.full(transcript_count, 30.0),
        "ordered_gene_ids": np.asarray(["g1", "g2"]),
        "target_variant_names": np.asarray([VARIANT]),
        "target_variant_membership": np.ones((transcript_count, 1), dtype=np.bool_),
        "program_names": np.asarray(["g1_program"]),
        "program_gene_membership": np.asarray([[1.0, 0.0]]),
    }


def _evaluate(source: dict[str, np.ndarray]) -> dict[str, object]:
    return dict(
        evaluate_measurement_gate(
            source,
            _thresholds(),
            development_donors=("d1",),
            locked_test_donors=("d2",),
            target_variants=(VARIANT,),
            split_salt=SPLIT_SALT,
            study_manifest_sha256="a" * 64,
            source_sha256="b" * 64,
        )
    )


def test_measurement_gate_reports_all_required_domains_and_passes() -> None:
    report = _evaluate(_source())
    assert report["schema"] == MEASUREMENT_GATE_SCHEMA
    assert report["pass"] is True
    assert report["registration"]["duplicate_cell_ids"] == 0
    assert set(report["registration"]["by_section"]) == {"s1", "s2"}
    assert report["molecular"]["duplicate_transcript_ids"] == 0
    assert report["molecular"]["target_genes_after_qc"] == 2
    variant = report["molecular"]["target_variants"][VARIANT]
    assert set(variant["per_section_distributions"]) == {"s1", "s2"}
    assert variant["target_basis_measurement_ceiling"][
        "median_spearman_brown_reliability"
    ] == pytest.approx(1.0)
    assert report["reference_evaluation_separation"][
        "reference_evaluation_block_id_intersection"
    ] == 0
    assert report["coverage"]["fraction_planned_biological_coverage_retained"] == 1.0


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
    split = construct_split_half_counts(
        source["transcript_id"],
        source["transcript_observation_id"],
        source["transcript_gene_id"],
        source["observation_id"],
        source["ordered_gene_ids"],
        salt=SPLIT_SALT,
    )
    source["whole_cell_target_counts_half_a"] = split.half_a
    source["whole_cell_target_counts_half_b"] = split.half_b
    source["whole_cell_target_counts"] = split.half_a + split.half_b
    source["whole_cell_library_size_half_a"] = split.half_a.sum(axis=1)
    source["whole_cell_library_size_half_b"] = split.half_b.sum(axis=1)
    source["whole_cell_library_sizes"] = source["whole_cell_target_counts"].sum(axis=1)
    source["normalized_whole_cell_targets"] = np.log1p(
        source["whole_cell_target_counts"].astype(np.float64)
        * (10_000.0 / source["whole_cell_library_sizes"][:, None])
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
    planned = ["d1|s1|epithelial", "d2|s2|epithelial"]
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
    compact_report = _evaluate(source)
    assert compact_report["pass"] is True
    assert compact_report["molecular"]["transcript_evidence_mode"].startswith("verified")
    assert compact_report["molecular"]["ordered_reliable_gene_ids"] == raw_report[
        "molecular"
    ]["ordered_reliable_gene_ids"]


def test_reliable_gene_panel_is_selected_from_development_donors_only() -> None:
    source = _source()
    baseline = _evaluate(source)
    keep = np.asarray(
        [
            not value.startswith("o08")
            and not value.startswith("o09")
            and not value.startswith("o1")
            for value in source["transcript_observation_id"]
        ],
        dtype=np.bool_,
    )
    # Remove all locked-donor transcripts.  This destroys locked coverage but
    # must not alter the development-selected molecular panel.
    for key in (
        "transcript_id",
        "transcript_observation_id",
        "transcript_gene_id",
        "transcript_qv",
    ):
        source[key] = source[key][keep]
    source["target_variant_membership"] = source["target_variant_membership"][keep]
    altered = _evaluate(source)
    assert altered["molecular"]["ordered_reliable_gene_ids"] == baseline["molecular"][
        "ordered_reliable_gene_ids"
    ]
    assert altered["coverage"]["pass"] is False


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
            "maximum_reference_evaluation_source_file_overlap",
        }
    }
    coverage = {
        key: value
        for key, value in _thresholds().__dict__.items()
        if key not in decisions
    }
    parsed = MeasurementThresholds.from_study_manifest(
        {"decision_thresholds": {"measurement": decisions}, "coverage_requirements": coverage}
    )
    assert parsed == _thresholds()
    del decisions["minimum_target_basis_ceiling"]
    with pytest.raises(ValueError, match="explicit measurement thresholds"):
        MeasurementThresholds.from_study_manifest(
            {"decision_thresholds": decisions, "coverage_requirements": coverage}
        )
