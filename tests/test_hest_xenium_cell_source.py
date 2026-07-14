from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import struct
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from heir.data.study_manifest import current_git_commit, freeze_manifest_content


def _load_builder():
    path = Path(__file__).parents[1] / "scripts" / "build_hest_xenium_cell_source.py"
    spec = importlib.util.spec_from_file_location("build_hest_xenium_cell_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()
REPOSITORY_ROOT = Path(__file__).parents[1]
ENCODER_MANIFEST_PATH = REPOSITORY_ROOT / "manifests" / "encoders" / "uni2h.json"
CROP_MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "crops" / "hest_crop_ladder.json"


def test_xenium_cell_identity_is_section_scoped() -> None:
    first = builder._section_scoped_cell_id("NCBI845", "cell-7")
    second = builder._section_scoped_cell_id("NCBI846", "cell-7")
    assert first == "NCBI845:cell-7"
    assert second == "NCBI846:cell-7"
    assert first != second


def test_registration_quality_strata_match_frozen_h_meas_definition() -> None:
    contract = {
        "maximum_registration_nucleus_diameter_ratio_p95": 0.5,
        "maximum_registration_nearest_neighbor_ratio_p95": 0.4,
        "best_registration_quality_max_fraction_of_limit": 0.25,
        "intermediate_registration_quality_max_fraction_of_limit": 0.6,
    }
    scores, strata, cutoffs = builder._registration_quality_strata(
        np.asarray([0.05, 0.20, 0.45, 0.55]),
        np.asarray([0.04, 0.16, 0.32, 0.20]),
        contract,
    )

    np.testing.assert_allclose(scores, [0.1, 0.4, 0.9, 1.1])
    assert strata.tolist() == ["best", "intermediate", "near_threshold", "failed"]
    assert cutoffs == {"best": 0.25, "intermediate": 0.6, "near_threshold": 1.0}

    malformed = dict(contract)
    malformed["intermediate_registration_quality_max_fraction_of_limit"] = 0.2
    with pytest.raises(ValueError, match="contract is malformed"):
        builder._registration_quality_strata(np.asarray([0.1]), np.asarray([0.1]), malformed)


def _wkb_square(x: float, y: float, half: float = 2.0) -> bytes:
    points = (
        (x - half, y - half),
        (x + half, y - half),
        (x + half, y + half),
        (x - half, y + half),
        (x - half, y - half),
    )
    return b"".join(
        (b"\x01", struct.pack("<I", 3), struct.pack("<I", 1), struct.pack("<I", len(points)))
        + tuple(struct.pack("<dd", *point) for point in points)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _role_blocks(sample_id: str, salt: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for block_x in range(10, 18):
        role = builder._block_role(sample_id, block_x, 9, salt)
        result.setdefault(role, block_x)
    assert set(result) == {"reference", "evaluation"}
    return result


class _FixtureEncoder:
    feature_width = 1536

    def __init__(self, manifest_sha256: str):
        self.manifest_sha256 = manifest_sha256

    def encode(self, patches: np.ndarray) -> np.ndarray:
        mean = patches.mean(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
        return np.repeat(mean[:, None], self.feature_width, axis=1)


def _write_sample(root: Path, sample_id: str, salt: str, image_value: int) -> dict[str, object]:
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    tifffile = pytest.importorskip("tifffile")
    blocks = _role_blocks(sample_id, salt)
    cell_ids = []
    cell_geometries = []
    nucleus_geometries = []
    transcript_cell_ids = []
    transcript_genes = []
    transcript_ids = []
    qvs = []
    overlaps_nucleus = []
    x_locations = []
    y_locations = []
    he_x_locations = []
    he_y_locations = []
    cellvit_geometries = []
    cellvit_classes = []
    next_transcript = 0
    for role in ("reference", "evaluation"):
        x = blocks[role] * 32 + 16
        for type_index, y in enumerate((300, 308)):
            cell_id = "%s_%d" % (role, type_index)
            cell_ids.append(cell_id)
            cell_geometries.append(_wkb_square(x, y, half=3.0))
            nucleus_geometries.append(_wkb_square(x, y, half=2.0))
            cellvit_geometries.append(_wkb_square(x, y, half=1.0))
            cellvit_classes.append("Epithelial" if type_index == 0 else "Inflammatory")
            genes = (["TYPE_A"] if type_index == 0 else ["TYPE_B"]) * 8 + ["G1"] * 4 + ["G2"] * 4
            for gene in genes:
                transcript_cell_ids.append(cell_id)
                transcript_genes.append(gene)
                transcript_ids.append(next_transcript)
                qvs.append(40.0)
                overlaps_nucleus.append(1)
                x_locations.append(float(x) * builder.SOURCE_MPP)
                y_locations.append(float(y) * builder.SOURCE_MPP)
                he_x_locations.append(float(x))
                he_y_locations.append(float(y))
                next_transcript += 1
            for gene, qv, overlaps in (
                ("G1", 40.0, 0),
                ("G2", 10.0, 1),
                ("NegControlProbe_1", 40.0, 1),
            ):
                transcript_cell_ids.append(cell_id)
                transcript_genes.append(gene)
                transcript_ids.append(next_transcript)
                qvs.append(qv)
                overlaps_nucleus.append(overlaps)
                x_locations.append(float(x) * builder.SOURCE_MPP)
                y_locations.append(float(y) * builder.SOURCE_MPP)
                he_x_locations.append(float(x))
                he_y_locations.append(float(y))
                next_transcript += 1
            for control_index in range(10):
                transcript_cell_ids.append(cell_id)
                transcript_genes.append("NegControlCodeword_%04d" % control_index)
                transcript_ids.append(next_transcript)
                qvs.append(40.0)
                overlaps_nucleus.append(1)
                x_locations.append((float(x) + control_index * 0.001) * builder.SOURCE_MPP)
                y_locations.append((float(y) + control_index * 0.001) * builder.SOURCE_MPP)
                he_x_locations.append(float(x) + control_index * 0.001)
                he_y_locations.append(float(y) + control_index * 0.001)
                next_transcript += 1

    wsi = root / "wsis" / (sample_id + ".tif")
    transcripts = root / "transcripts" / (sample_id + "_transcripts.parquet")
    cell_seg = root / "xenium_seg" / (sample_id + "_xenium_cell_seg.parquet")
    nucleus_seg = root / "xenium_seg" / (sample_id + "_xenium_nucleus_seg.parquet")
    cellvit_seg = root / "cellvit_seg" / (sample_id + "_cellvit_seg.parquet")
    for path in (wsi, transcripts, cell_seg, nucleus_seg, cellvit_seg):
        path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((640, 832, 3), image_value, dtype=np.uint8)
    tifffile.imwrite(wsi, image, photometric="rgb")
    parquet.write_table(
        pa.table({"geometry": cell_geometries, "__index_level_0__": cell_ids}), cell_seg
    )
    parquet.write_table(
        pa.table({"geometry": nucleus_geometries, "__index_level_0__": cell_ids}), nucleus_seg
    )
    parquet.write_table(
        pa.table(
            {
                "transcript_id": transcript_ids,
                "cell_id": transcript_cell_ids,
                "feature_name": transcript_genes,
                "qv": qvs,
                "overlaps_nucleus": overlaps_nucleus,
                "x_location": x_locations,
                "y_location": y_locations,
                "he_x": he_x_locations,
                "he_y": he_y_locations,
            }
        ),
        transcripts,
    )
    parquet.write_table(
        pa.table(
            {
                "geometry": cellvit_geometries,
                "class": cellvit_classes,
                "cell_id": list(range(len(cell_ids))),
            }
        ),
        cellvit_seg,
    )

    def declaration(path: Path) -> dict[str, str]:
        return {"path": str(path.relative_to(root)), "sha256": _sha256(path)}

    return {
        "sample_id": sample_id,
        "pixel_size_um": builder.SOURCE_MPP,
        "wsi": declaration(wsi),
        "transcripts": declaration(transcripts),
        "cell_seg": declaration(cell_seg),
        "nucleus_seg": declaration(nucleus_seg),
        "cellvit_seg": declaration(cellvit_seg),
    }


def _protocol(samples: list[dict[str, object]], development: list[str], locked: list[str]):
    encoder_manifest_sha256 = _sha256(ENCODER_MANIFEST_PATH)
    crop_manifest_sha256 = _sha256(CROP_MANIFEST_PATH)
    return {
        "schema": builder.PROTOCOL_SCHEMA,
        "scientific_scope": "registered_cell_local_context_112um_association",
        "g2_claim_scope": "registered_cell_local_context_112um",
        "authorizes_nucleus_intrinsic_claim": False,
        "dataset_repo": builder.DATASET_REPO,
        "dataset_revision": builder.DATASET_REVISION,
        "encoder_manifest_sha256": encoder_manifest_sha256,
        "crop_manifest_sha256": crop_manifest_sha256,
        "normalization": "log1p_cpm_10000",
        "assay": "Xenium",
        "observation_level": "cell",
        "target_construction": "nucleus_overlapping_xenium_transcripts",
        "registration_method": "native_xenium_cell_id_join",
        "disease_estimands": ["disease_inclusive", "disease_adjusted"],
        "nuisance_fields": [
            "log1p_library_size",
            "section_id",
            "disease_status",
            "site_id",
            "batch_id",
            "stain_quality",
            "nuclear_morphology",
            "cell_morphology",
            "local_density",
            "boundary_position",
            "smooth_spatial_basis",
        ],
        "development_donors": development,
        "locked_test_donors": locked,
        "broad_type_names": list(builder.TYPE_NAMES),
        "type_markers": {
            "Endothelial": ["TYPE_C"],
            "Epithelial": ["TYPE_A"],
            "Immune": ["TYPE_B"],
            "Mesenchymal": ["TYPE_D"],
        },
        "fine_type_marker_gene_ids": ["TYPE_FINE"],
        "label_target_independence": {
            "strategy": "conservative_proxy_marker_exclusion_pending_exact_annotation_provenance",
            "evidence_kind": "pending",
            "annotation_receipt_sha256": None,
            "ordered_annotation_feature_ids": [],
            "ordered_annotation_feature_ids_sha256": None,
            "annotation_training_scope": "unknown_pending_provenance",
            "annotation_training_donor_ids": [],
            "annotation_training_donor_ids_sha256": None,
            "training_label_ontology_source": "pending",
            "training_label_provenance_receipt_sha256": None,
            "training_label_target_gene_overlap_count": None,
            "training_labels_establish_target_independence": False,
            "annotation_quality_contract": builder.DEFAULT_ANNOTATION_QUALITY_CONTRACT,
            "locked_donors_used_for_training": None,
            "same_cohort_annotation": True,
            "cross_fitting_method": "pending",
            "cross_fitting_receipt_sha256": None,
            "establishes_full_target_independence": False,
            "limitation": "synthetic pending fixture",
        },
        "gene_ids": ["G1", "G2"],
        "minimum_transcripts_per_cell": 10,
        "minimum_transcript_qv": 20.0,
        "minimum_reference_cells_per_donor_section_type": 1,
        "minimum_evaluation_cells_per_donor_section_type": 1,
        "minimum_development_donors_per_fine_type": 1,
        "minimum_locked_donors_per_fine_type": 1,
        "excluded_feature_prefixes": list(builder.CONTROL_PREFIXES),
        "spatial_block_um": 32.0 * builder.SOURCE_MPP,
        "spatial_roi_um": 8.0 * builder.SOURCE_MPP,
        "opposite_pool_guard_um": 4.0 * builder.SOURCE_MPP,
        "cellvit_sensitivity_radius_um": 1.0,
        "cellvit_class_names": ["Epithelial", "Inflammatory"],
        "maximum_affine_registration_residual_p95_um": 0.000001,
        "maximum_annotation_nucleus_distance_p95_um": 0.000001,
        "maximum_registration_outlier_fraction": 0.01,
        "maximum_crop_padding_fraction": 0.99,
        "pool_assignment_salt": "synthetic-frozen-v1",
        "reference_splits": {
            "primary_split_id": "primary",
            "selection_unit": "spatial_block",
            "primary_evaluation_rows_fixed": True,
            "alternate_splits": [
                {
                    "split_id": "reference_hash_fold_0",
                    "salt": "synthetic-reference-fold-0",
                    "initial_reference_retention_fraction": 0.8,
                },
                {
                    "split_id": "reference_hash_fold_1",
                    "salt": "synthetic-reference-fold-1",
                    "initial_reference_retention_fraction": 0.8,
                },
            ],
        },
        "transcript_split_salt": "synthetic-split-v1",
        "frozen_h_meas_shared_measurement_thresholds": dict(
            builder.FROZEN_H_MEAS_SHARED_MEASUREMENT_THRESHOLDS
        ),
        "confirmatory_primary_endpoint": {
            "name": builder.CONFIRMATORY_PRIMARY_ENDPOINT_NAME,
            "condition_on": "fine_type",
            "decision_rule": builder.CONFIRMATORY_PRIMARY_DECISION_RULE,
            "donor_type_macro": {
                "metric": builder.CONFIRMATORY_DONOR_TYPE_METRIC,
                "minimum_effect": builder.CONFIRMATORY_PRIMARY_MINIMUM_EFFECT,
            },
            "donor_section_type_macro": {
                "metric": builder.CONFIRMATORY_DONOR_SECTION_TYPE_METRIC,
                "minimum_effect": builder.CONFIRMATORY_PRIMARY_MINIMUM_EFFECT,
            },
        },
        "target_programs": {"synthetic_program": ["G1", "G2"]},
        "samples": samples,
    }


def _write_annotations(
    path: Path,
    sample_ids: set[str] | None = None,
    *,
    include_raw_labels: bool = True,
) -> None:
    columns = [
        "hest_id",
        "sample",
        "patient",
        "cell_id",
        "sample_type",
        "sample_affect",
        "disease_status",
        "tma",
        "run",
        "x_centroid",
        "y_centroid",
        "nCount_RNA",
        "nFeature_RNA",
        "perc_negcontrolorunassigned",
    ]
    if include_raw_labels:
        columns[9:9] = ["final_CT", "final_lineage"]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(columns) + "\n")
        for sample_id, (donor_id, source_sample) in sorted(builder.SECTION_IDENTITIES.items()):
            if sample_ids is not None and sample_id not in sample_ids:
                continue
            blocks = _role_blocks(sample_id, "synthetic-frozen-v1")
            for role in ("reference", "evaluation"):
                x = blocks[role] * 32 + 16
                for type_index, y in enumerate((300, 308)):
                    lineage = "Epithelial" if type_index == 0 else "Immune"
                    values = [
                        sample_id,
                        source_sample,
                        donor_id,
                        "%s_%d" % (role, type_index),
                        "synthetic_site",
                        "Less Affected",
                        "Disease",
                        "TMA1",
                        "Run1",
                        str(x * builder.SOURCE_MPP),
                        str(y * builder.SOURCE_MPP),
                        "16",
                        "4",
                        "0.5",
                    ]
                    if include_raw_labels:
                        values[9:9] = [
                            "Synthetic epithelial" if type_index == 0 else "Synthetic immune",
                            lineage,
                        ]
                    handle.write("\t".join(values) + "\n")


def _write_independent_predictions(path: Path, sample_ids: set[str]) -> int:
    lines = ["\t".join(builder.ANNOTATION_PREDICTION_COLUMNS)]
    rows = 0
    for sample_id in sorted(sample_ids):
        for role in ("reference", "evaluation"):
            for type_index in range(2):
                lineage = "Epithelial" if type_index == 0 else "Immune"
                fine_type = "Independent epithelial" if type_index == 0 else "Independent immune"
                lines.append(
                    "\t".join(
                        (
                            sample_id,
                            "%s_%d" % (role, type_index),
                            lineage,
                            fine_type,
                            "0.95",
                            "false",
                            "[0.95,0.05]" if type_index == 0 else "[0.05,0.95]",
                        )
                    )
                )
                rows += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _external_independence_contract(
    annotation_receipt_sha256: str,
    training_label_receipt_sha256: str = "b" * 64,
) -> dict[str, object]:
    annotation_ids = ["ANN1", "ANN2"]
    training_donors = ["external-donor-1"]
    return {
        "strategy": "external gene-disjoint annotation",
        "evidence_kind": "external_gene_disjoint_annotation",
        "annotation_receipt_sha256": annotation_receipt_sha256,
        "ordered_annotation_feature_ids": annotation_ids,
        "ordered_annotation_feature_ids_sha256": builder._canonical_sha256(annotation_ids),
        "annotation_training_scope": "external_donors_only",
        "annotation_training_donor_ids": training_donors,
        "annotation_training_donor_ids_sha256": builder._canonical_sha256(training_donors),
        "training_label_ontology_source": "external_target_disjoint_ontology",
        "training_label_provenance_receipt_sha256": training_label_receipt_sha256,
        "training_label_target_gene_overlap_count": 0,
        "training_labels_establish_target_independence": True,
        "annotation_quality_contract": builder.DEFAULT_ANNOTATION_QUALITY_CONTRACT,
        "locked_donors_used_for_training": False,
        "same_cohort_annotation": False,
        "cross_fitting_method": "not_applicable",
        "cross_fitting_receipt_sha256": None,
        "establishes_full_target_independence": True,
        "limitation": "synthetic external fixture",
    }


def _write_training_label_provenance_receipt(
    path: Path,
    ontology_source_path: Path,
    contract: dict[str, object],
    *,
    target_gene_ids: tuple[str, ...] = ("G1", "G2"),
) -> None:
    fine_types = ("Independent epithelial", "Independent immune")
    ontology_features = ("ONTOLOGY_MARKER_A", "ONTOLOGY_MARKER_B")
    normalization = "log1p_library_size_10000_then_reference_zscore_v1"
    ontology_source = {
        "schema": builder.TRAINING_LABEL_ONTOLOGY_SOURCE_SCHEMA,
        "evidence_kind": contract["evidence_kind"],
        "training_label_ontology_source": contract["training_label_ontology_source"],
        "ontology_id": "synthetic-target-disjoint-lung-ontology-v1",
        "ontology_construction_method": "external expert labels using frozen disjoint markers",
        "training_label_feature_normalization": normalization,
        "ordered_fine_type_ids": list(fine_types),
        "ordered_fine_type_ids_sha256": builder._canonical_sha256(list(fine_types)),
        "broad_lineage_by_fine_type": {
            "Independent epithelial": "Epithelial",
            "Independent immune": "Immune",
        },
        "ordered_training_label_feature_ids": list(ontology_features),
        "ordered_training_label_feature_ids_sha256": builder._canonical_sha256(
            list(ontology_features)
        ),
        "source_donor_ids": contract["annotation_training_donor_ids"],
        "source_donor_ids_sha256": contract["annotation_training_donor_ids_sha256"],
        "created_without_locked_outcomes": True,
    }
    ontology_source_path.write_text(json.dumps(ontology_source, sort_keys=True), encoding="utf-8")
    receipt = {
        "schema": builder.TRAINING_LABEL_PROVENANCE_RECEIPT_SCHEMA,
        "evidence_kind": contract["evidence_kind"],
        "ontology_id": "synthetic-target-disjoint-lung-ontology-v1",
        "training_label_ontology_source": contract["training_label_ontology_source"],
        "ontology_source_artifact_sha256": _sha256(ontology_source_path),
        "ontology_construction_method": "external expert labels using frozen disjoint markers",
        "training_label_feature_normalization": normalization,
        "ordered_fine_type_ids": list(fine_types),
        "ordered_fine_type_ids_sha256": builder._canonical_sha256(list(fine_types)),
        "broad_lineage_by_fine_type": {
            "Independent epithelial": "Epithelial",
            "Independent immune": "Immune",
        },
        "ordered_training_label_feature_ids": list(ontology_features),
        "ordered_training_label_feature_ids_sha256": builder._canonical_sha256(
            list(ontology_features)
        ),
        "ordered_target_gene_ids": list(target_gene_ids),
        "ordered_target_gene_ids_sha256": builder._canonical_sha256(list(target_gene_ids)),
        "training_label_target_gene_overlap_count": 0,
        "training_labels_establish_target_independence": True,
        "annotation_training_scope": contract["annotation_training_scope"],
        "annotation_training_donor_ids": contract["annotation_training_donor_ids"],
        "annotation_training_donor_ids_sha256": contract["annotation_training_donor_ids_sha256"],
        "locked_donors_used_for_training": False,
        "same_cohort_annotation": contract["same_cohort_annotation"],
        "created_without_locked_outcomes": True,
    }
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")


def _write_annotation_validation_export(path: Path) -> int:
    lines = ["\t".join(builder.ANNOTATION_VALIDATION_COLUMNS)]
    donor = "external-validation-donor-1"
    lines.extend(
        (
            f"{donor}\tcell-0\tIndependent epithelial\tIndependent epithelial"
            "\t0.95\tfalse\t[0.95,0.05]",
            f"{donor}\tcell-1\tIndependent immune\tIndependent immune\t0.95\tfalse\t[0.05,0.95]",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 2


def _write_external_annotation_receipt(
    path: Path,
    predictions_path: Path,
    validation_path: Path,
    contract: dict[str, object],
    *,
    prediction_rows: int,
    prediction_donor_ids: tuple[str, ...],
    validation_rows: int,
) -> None:
    fine_types = ("Independent epithelial", "Independent immune")
    quality = contract["annotation_quality_contract"]
    receipt = {
        "schema": builder.ANNOTATION_RECEIPT_SCHEMA,
        "evidence_kind": contract["evidence_kind"],
        "prediction_export_sha256": _sha256(predictions_path),
        "prediction_row_count": prediction_rows,
        "prediction_nonabstained_row_count": prediction_rows,
        "prediction_columns": list(builder.ANNOTATION_PREDICTION_COLUMNS),
        "row_order": "filtered_annotation_export_order",
        "validation_export_sha256": _sha256(validation_path),
        "validation_row_count": validation_rows,
        "validation_columns": list(builder.ANNOTATION_VALIDATION_COLUMNS),
        "validation_row_order": "validation_donor_then_cell_id",
        "prediction_donor_ids": list(prediction_donor_ids),
        "prediction_donor_ids_sha256": builder._canonical_sha256(list(prediction_donor_ids)),
        "ordered_fine_type_ids": list(fine_types),
        "ordered_fine_type_ids_sha256": builder._canonical_sha256(list(fine_types)),
        "broad_lineage_by_fine_type": {
            "Independent epithelial": "Epithelial",
            "Independent immune": "Immune",
        },
        "probability_vector_order": list(fine_types),
        "probability_sum_tolerance": quality["probability_sum_tolerance"],
        "abstention_policy": {
            "method": quality["abstention_method"],
            "confidence_threshold": quality["confidence_threshold"],
            "abstain_label": quality["abstain_label"],
            "tie_breaking": quality["tie_breaking"],
        },
        "performance_criteria": {
            "minimum_fine_type_macro_f1": quality["minimum_fine_type_macro_f1"],
            "calibration_metric": quality["calibration_metric"],
            "calibration_binning": quality["calibration_binning"],
            "calibration_bin_count": quality["calibration_bin_count"],
            "maximum_calibration_error": quality["maximum_calibration_error"],
            "minimum_major_class_sensitivity": quality["minimum_major_class_sensitivity"],
            "minimum_prediction_coverage_fraction": quality["minimum_prediction_coverage_fraction"],
        },
        "performance_evaluation": {
            "validation_design": "external_donor_held_out",
            "validation_donor_ids": ["external-validation-donor-1"],
            "validation_donor_ids_sha256": builder._canonical_sha256(
                ["external-validation-donor-1"]
            ),
            "fine_type_macro_f1": 1.0,
            "calibration_metric": "multiclass_expected_calibration_error",
            "calibration_error": 0.05,
            "major_class_sensitivity_by_fine_type": {
                "Independent epithelial": 1.0,
                "Independent immune": 1.0,
            },
            "prediction_coverage_fraction": 1.0,
            "per_validation_donor_metrics": {
                "external-validation-donor-1": {
                    "fine_type_macro_f1": 1.0,
                    "calibration_error": 0.05,
                    "minimum_major_class_sensitivity": 1.0,
                    "prediction_coverage_fraction": 1.0,
                }
            },
        },
        "ordered_annotation_feature_ids": contract["ordered_annotation_feature_ids"],
        "ordered_annotation_feature_ids_sha256": contract["ordered_annotation_feature_ids_sha256"],
        "annotation_feature_normalization": "log1p_library_size_10000_then_reference_zscore_v1",
        "annotation_training_scope": contract["annotation_training_scope"],
        "annotation_training_donor_ids": contract["annotation_training_donor_ids"],
        "annotation_training_donor_ids_sha256": contract["annotation_training_donor_ids_sha256"],
        "training_label_ontology_source": contract["training_label_ontology_source"],
        "training_label_provenance_receipt_sha256": contract[
            "training_label_provenance_receipt_sha256"
        ],
        "training_label_target_gene_overlap_count": 0,
        "training_labels_establish_target_independence": True,
        "annotation_quality_contract": builder.DEFAULT_ANNOTATION_QUALITY_CONTRACT,
        "locked_donors_used_for_training": False,
        "same_cohort_annotation": False,
        "cross_fitting_method": "not_applicable",
        "cross_fitting_receipt": None,
    }
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")


def test_independent_annotation_artifacts_bind_actual_prediction_bytes(tmp_path: Path) -> None:
    predictions = tmp_path / "independent-labels.tsv"
    predictions.write_text(
        "\t".join(builder.ANNOTATION_PREDICTION_COLUMNS)
        + "\n"
        + "NCBI858\tcell-1\tEpithelial\tIndependent epithelial\t0.95\tfalse\t"
        "[0.95,0.05]\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "annotation-receipt.json"
    training_receipt = tmp_path / "training-label-provenance.json"
    ontology_source = tmp_path / "training-label-ontology-source.json"
    ontology_source.write_text('{"ontology":"synthetic"}\n', encoding="utf-8")
    validation_export = tmp_path / "annotation-validation.tsv"
    validation_rows = _write_annotation_validation_export(validation_export)
    contract = _external_independence_contract("0" * 64)
    _write_training_label_provenance_receipt(training_receipt, ontology_source, contract)
    contract["training_label_provenance_receipt_sha256"] = _sha256(training_receipt)
    _write_external_annotation_receipt(
        receipt,
        predictions,
        validation_export,
        contract,
        prediction_rows=1,
        prediction_donor_ids=("VUILD91",),
        validation_rows=validation_rows,
    )
    contract["annotation_receipt_sha256"] = _sha256(receipt)

    verified = builder._verify_independent_annotation_artifacts(
        receipt,
        predictions,
        training_receipt,
        ontology_source,
        validation_export,
        contract,
        prediction_donor_ids=("VUILD91",),
        target_gene_ids=("G1", "G2"),
    )

    assert verified.receipt_sha256 == _sha256(receipt)
    assert verified.predictions_sha256 == _sha256(predictions)
    query_validated = json.loads(receipt.read_text(encoding="utf-8"))
    query_validated["performance_evaluation"]["validation_donor_ids"] = ["VUILD91"]
    query_validated["performance_evaluation"]["validation_donor_ids_sha256"] = (
        builder._canonical_sha256(["VUILD91"])
    )
    query_validated["performance_evaluation"]["per_validation_donor_metrics"] = {
        "VUILD91": query_validated["performance_evaluation"]["per_validation_donor_metrics"][
            "external-validation-donor-1"
        ]
    }
    with pytest.raises(ValueError, match="invalid donor scope"):
        builder._verify_annotation_performance_receipt(
            query_validated,
            contract,
            prediction_donor_ids=("VUILD91",),
            fine_type_ids=("Independent epithelial", "Independent immune"),
        )
    original_ontology = ontology_source.read_bytes()
    ontology_source.write_bytes(original_ontology + b"tamper")
    with pytest.raises(ValueError, match="ontology source artifact differs"):
        builder._verify_independent_annotation_artifacts(
            receipt,
            predictions,
            training_receipt,
            ontology_source,
            validation_export,
            contract,
            prediction_donor_ids=("VUILD91",),
            target_gene_ids=("G1", "G2"),
        )
    ontology_source.write_bytes(original_ontology)
    original_training_receipt = training_receipt.read_bytes()
    training_receipt.write_bytes(original_training_receipt + b" ")
    with pytest.raises(ValueError, match="provenance receipt differs from the frozen"):
        builder._verify_independent_annotation_artifacts(
            receipt,
            predictions,
            training_receipt,
            ontology_source,
            validation_export,
            contract,
            prediction_donor_ids=("VUILD91",),
            target_gene_ids=("G1", "G2"),
        )
    training_receipt.write_bytes(original_training_receipt)
    predictions.write_text(predictions.read_text(encoding="utf-8") + "tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="predictions differ"):
        builder._verify_independent_annotation_artifacts(
            receipt,
            predictions,
            training_receipt,
            ontology_source,
            validation_export,
            contract,
            prediction_donor_ids=("VUILD91",),
            target_gene_ids=("G1", "G2"),
        )


def test_annotation_quality_is_recomputed_from_held_out_rows(tmp_path: Path) -> None:
    predictions = tmp_path / "independent-labels.tsv"
    predictions.write_text(
        "\t".join(builder.ANNOTATION_PREDICTION_COLUMNS)
        + "\nNCBI858\tcell-1\tEpithelial\tIndependent epithelial"
        "\t0.95\tfalse\t[0.95,0.05]\n",
        encoding="utf-8",
    )
    validation = tmp_path / "annotation-validation.tsv"
    validation_rows = _write_annotation_validation_export(validation)
    ontology_source = tmp_path / "ontology.json"
    ontology_source.write_text('{"ontology":"synthetic"}\n', encoding="utf-8")
    training_receipt = tmp_path / "training-label-receipt.json"
    annotation_receipt = tmp_path / "annotation-receipt.json"
    contract = _external_independence_contract("0" * 64)
    _write_training_label_provenance_receipt(training_receipt, ontology_source, contract)
    contract["training_label_provenance_receipt_sha256"] = _sha256(training_receipt)
    _write_external_annotation_receipt(
        annotation_receipt,
        predictions,
        validation,
        contract,
        prediction_rows=1,
        prediction_donor_ids=("VUILD91",),
        validation_rows=validation_rows,
    )
    overstated = json.loads(annotation_receipt.read_text(encoding="utf-8"))
    overstated["performance_evaluation"]["fine_type_macro_f1"] = 0.99
    annotation_receipt.write_text(json.dumps(overstated, sort_keys=True), encoding="utf-8")
    contract["annotation_receipt_sha256"] = _sha256(annotation_receipt)

    with pytest.raises(ValueError, match="metrics differ from row-level validation"):
        builder._verify_independent_annotation_artifacts(
            annotation_receipt,
            predictions,
            training_receipt,
            ontology_source,
            validation,
            contract,
            prediction_donor_ids=("VUILD91",),
            target_gene_ids=("G1", "G2"),
        )


def test_same_cohort_receipt_binds_lodo_validation_and_prediction_model_scope() -> None:
    development = ("development-1", "development-2", "development-3")
    validation_sha256 = "a" * 64
    prediction_sha256 = "b" * 64

    def receipt_for(
        prediction_donors: tuple[str, ...],
        method: str,
        final_training_donors: tuple[str, ...],
    ) -> dict[str, object]:
        return {
            "schema": builder.ANNOTATION_CROSS_FIT_SCHEMA,
            "validation_export_sha256": validation_sha256,
            "prediction_export_sha256": prediction_sha256,
            "validation_folds": [
                {
                    "validation_donor_id": donor,
                    "training_donor_ids": [
                        training_donor for training_donor in development if training_donor != donor
                    ],
                }
                for donor in development
            ],
            "prediction_assignment": {
                "method": method,
                "prediction_donor_ids": list(prediction_donors),
                "final_model_training_donor_ids": list(final_training_donors),
                "locked_donors_used_for_training": False,
                "created_without_locked_outcomes": True,
            },
        }

    def verify(receipt: dict[str, object], prediction_donors: tuple[str, ...]) -> None:
        contract = {
            "same_cohort_annotation": True,
            "annotation_training_donor_ids": list(development),
            "cross_fitting_receipt_sha256": builder._canonical_sha256(receipt),
        }
        builder._verify_cross_fitting_receipt(
            receipt,
            contract,
            prediction_donors,
            validation_export_sha256=validation_sha256,
            prediction_export_sha256=prediction_sha256,
        )

    development_receipt = receipt_for(
        development,
        "leave_one_development_donor_out_validation_models",
        (),
    )
    verify(development_receipt, development)

    locked = ("locked-1", "locked-2")
    confirmatory_receipt = receipt_for(
        locked,
        "final_all_development_donors_model",
        development,
    )
    verify(confirmatory_receipt, locked)

    mixed_active_donors = development + locked
    mixed_receipt = receipt_for(
        mixed_active_donors,
        "development_lodo_plus_final_all_development_donors_model",
        development,
    )
    verify(mixed_receipt, mixed_active_donors)

    incomplete = receipt_for(
        locked,
        "final_all_development_donors_model",
        development,
    )
    incomplete["validation_folds"] = incomplete["validation_folds"][:-1]
    with pytest.raises(ValueError, match="every development donor"):
        verify(incomplete, locked)


def test_annotation_reader_consumes_independent_labels_instead_of_final_ct(tmp_path: Path) -> None:
    sample_id = "NCBI858"
    metadata = tmp_path / "development-only.tsv.gz"
    _write_annotations(metadata, {sample_id})
    predictions = tmp_path / "independent-labels.tsv"
    lines = ["\t".join(builder.ANNOTATION_PREDICTION_COLUMNS)]
    for role in ("reference", "evaluation"):
        lines.extend(
            (
                f"{sample_id}\t{role}_0\tEpithelial\tIndependent epithelial"
                "\t0.95\tfalse\t[0.95,0.05]",
                f"{sample_id}\t{role}_1\tImmune\tIndependent immune\t0.95\tfalse\t[0.05,0.95]",
            )
        )
    predictions.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = builder.IndependentAnnotationArtifacts(
        receipt_path=tmp_path / "unused-receipt.json",
        receipt_sha256="1" * 64,
        training_label_receipt_path=tmp_path / "unused-training-receipt.json",
        training_label_receipt_sha256="2" * 64,
        training_label_ontology_source_path=tmp_path / "unused-ontology-source.json",
        training_label_ontology_source_sha256="3" * 64,
        predictions_path=predictions,
        predictions_sha256=_sha256(predictions),
        validation_export_path=tmp_path / "unused-validation.tsv",
        validation_export_sha256="4" * 64,
        prediction_row_count=4,
        prediction_nonabstained_row_count=4,
        prediction_donor_ids=("VUILD91",),
        fine_type_ids=("Independent epithelial", "Independent immune"),
        broad_lineage_by_fine_type={
            "Independent epithelial": "Epithelial",
            "Independent immune": "Immune",
        },
        confidence_threshold=0.8,
        probability_sum_tolerance=1.0e-8,
        prediction_coverage_fraction=1.0,
    )

    annotations = builder._read_annotations(
        metadata,
        allowed_sample_ids=(sample_id,),
        independent_artifacts=artifacts,
        expected_row_count=4,
        strict_sample_scope=True,
    )

    assert {cell.fine_type for cell in annotations[sample_id].values()} == {
        "Independent epithelial",
        "Independent immune",
    }
    predictions.write_text(
        "\n".join(lines).replace("[0.95,0.05]", "[0.60,0.50]") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid or unnormalized"):
        builder._read_annotations(
            metadata,
            allowed_sample_ids=(sample_id,),
            independent_artifacts=artifacts,
            expected_row_count=4,
            strict_sample_scope=True,
        )


def test_development_annotation_export_contract_is_required_and_scope_bound() -> None:
    with pytest.raises(ValueError, match="development-only annotation export"):
        builder._parse_development_annotation_export(None, ("NCBI858",))
    declaration = {
        "path": "GSE250346/development.tsv.gz",
        "sha256": "1" * 64,
        "row_count": 4,
        "sample_ids": ["NCBI858", "NCBI856"],
        "sample_ids_sha256": builder._canonical_sha256(["NCBI858", "NCBI856"]),
        "source_annotation_sha256": builder.ANNOTATION_SHA256,
    }
    with pytest.raises(ValueError, match="receipt is malformed"):
        builder._parse_development_annotation_export(declaration, ("NCBI858",))


def test_h_meas_rejects_pending_or_raw_annotation_labels(tmp_path: Path) -> None:
    pending = _protocol(
        [],
        list(builder.DEVELOPMENT_DONORS),
        list(builder.LOCKED_TEST_DONORS),
    )["label_target_independence"]

    with pytest.raises(ValueError, match="requires non-pending independent label provenance"):
        builder._require_h_meas_independent_annotation_contract(pending, ("G1", "G2"))

    raw_metadata = tmp_path / "development-only.tsv.gz"
    _write_annotations(raw_metadata, {"NCBI858"})
    with pytest.raises(ValueError, match="require an independent prediction export"):
        builder._read_annotations(
            raw_metadata,
            allowed_sample_ids=("NCBI858",),
            expected_row_count=4,
            strict_sample_scope=True,
        )


def test_wkb_centroid_and_spatial_pool_are_deterministic() -> None:
    np.testing.assert_allclose(builder._polygon_centroid(_wkb_square(17.0, 23.0)), (17.0, 23.0))
    first = builder._spatial_identity(
        "NCBI1",
        (48.0, 16.0),
        builder.SOURCE_MPP,
        block_um=32.0 * builder.SOURCE_MPP,
        roi_um=8.0 * builder.SOURCE_MPP,
        guard_um=4.0 * builder.SOURCE_MPP,
        salt="frozen",
    )
    second = builder._spatial_identity(
        "NCBI1",
        (48.0, 16.0),
        builder.SOURCE_MPP,
        block_um=32.0 * builder.SOURCE_MPP,
        roi_um=8.0 * builder.SOURCE_MPP,
        guard_um=4.0 * builder.SOURCE_MPP,
        salt="frozen",
    )
    assert first == second
    assert first.guard_pass is True


def test_frozen_ita_strata_are_cartesian_before_locked_labels_are_observed() -> None:
    samples = (
        SimpleNamespace(donor_id="development_donor", sample_id="development_section"),
        SimpleNamespace(donor_id="locked_donor", sample_id="locked_section"),
    )

    planned = builder._frozen_ita_stratum_ids(
        samples,
        ("common_type", "type_with_zero_locked_rows"),
    )

    assert planned == (
        "development_donor|development_section|common_type",
        "development_donor|development_section|type_with_zero_locked_rows",
        "locked_donor|locked_section|common_type",
        "locked_donor|locked_section|type_with_zero_locked_rows",
    )
    assert "locked_donor|locked_section|type_with_zero_locked_rows" in planned


def test_builder_creates_registered_cell_source_and_isolates_cellvit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("zarr")
    development = list(builder.DEVELOPMENT_DONORS)
    locked = list(builder.LOCKED_TEST_DONORS)
    samples = []
    for index, (sample_id, (donor, _)) in enumerate(sorted(builder.SECTION_IDENTITIES.items())):
        sample = _write_sample(tmp_path, sample_id, "synthetic-frozen-v1", 20 + index)
        sample["donor_id"] = donor
        samples.append(sample)
    annotation_path = tmp_path / "GSE250346" / "annotations.tsv.gz"
    annotation_path.parent.mkdir(parents=True)
    _write_annotations(annotation_path)
    monkeypatch.setattr(builder, "ANNOTATION_SHA256", _sha256(annotation_path))
    monkeypatch.setattr(builder, "ANNOTATION_ROWS", 80)
    protocol = _protocol(samples, development, locked)
    protocol["annotation_export"] = {
        "path": str(annotation_path.relative_to(tmp_path)),
        "sha256": builder.ANNOTATION_SHA256,
    }
    development_sample_ids = [
        str(sample["sample_id"]) for sample in samples if sample["donor_id"] in development
    ]
    development_annotation_path = tmp_path / "GSE250346" / "annotations.development-only.tsv.gz"
    _write_annotations(
        development_annotation_path,
        set(development_sample_ids),
        include_raw_labels=False,
    )
    protocol["measurement_development_annotation_export"] = {
        "path": str(development_annotation_path.relative_to(tmp_path)),
        "sha256": _sha256(development_annotation_path),
        "row_count": 4 * len(development_sample_ids),
        "sample_ids": development_sample_ids,
        "sample_ids_sha256": builder._canonical_sha256(development_sample_ids),
        "source_annotation_sha256": builder.ANNOTATION_SHA256,
    }
    annotation_predictions_path = tmp_path / "development-independent-labels.tsv"
    prediction_rows = _write_independent_predictions(
        annotation_predictions_path,
        set(development_sample_ids),
    )
    annotation_receipt_path = tmp_path / "development-annotation-receipt.json"
    training_label_receipt_path = tmp_path / "training-label-provenance.json"
    ontology_source_path = tmp_path / "training-label-ontology-source.json"
    ontology_source_path.write_text('{"ontology":"synthetic"}\n', encoding="utf-8")
    validation_export_path = tmp_path / "annotation-validation.tsv"
    validation_rows = _write_annotation_validation_export(validation_export_path)
    resolved_independence = _external_independence_contract("0" * 64)
    _write_training_label_provenance_receipt(
        training_label_receipt_path, ontology_source_path, resolved_independence
    )
    resolved_independence["training_label_provenance_receipt_sha256"] = _sha256(
        training_label_receipt_path
    )
    _write_external_annotation_receipt(
        annotation_receipt_path,
        annotation_predictions_path,
        validation_export_path,
        resolved_independence,
        prediction_rows=prediction_rows,
        prediction_donor_ids=tuple(sorted(development)),
        validation_rows=validation_rows,
    )
    resolved_independence["annotation_receipt_sha256"] = _sha256(annotation_receipt_path)
    protocol["label_target_independence"] = resolved_independence
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    draft = json.loads(
        (
            REPOSITORY_ROOT
            / "manifests"
            / "studies"
            / "hest_lung_measurement_development.draft.json"
        ).read_text(encoding="utf-8")
    )
    draft["analysis_plan_sha256"] = _sha256(protocol_path)
    draft["candidate_target_gene_panel_sha256"] = builder._canonical_sha256(["G1", "G2"])
    draft["decision_thresholds"]["required_opposite_pool_guard_um"] = protocol[
        "opposite_pool_guard_um"
    ]
    marker_sha256 = builder._canonical_sha256(["TYPE_FINE"])
    draft["type_marker_panel_sha256"] = marker_sha256
    draft["label_target_independence"] = {
        **protocol["label_target_independence"],
        "ordered_target_gene_ids": ["G1", "G2"],
        "ordered_target_gene_ids_sha256": builder._canonical_sha256(["G1", "G2"]),
        "annotation_target_overlap_count": 0,
    }
    locked_manifest = freeze_manifest_content(
        draft,
        git_commit=current_git_commit(REPOSITORY_ROOT),
        container_digest="sha256:" + "1" * 64,
    )
    study_manifest_path = tmp_path / "measurement-study.locked.json"
    study_manifest_path.write_text(json.dumps(locked_manifest), encoding="utf-8")
    for sample in samples:
        if sample["donor_id"] in locked:
            for field in ("wsi", "transcripts", "cell_seg", "nucleus_seg", "cellvit_seg"):
                (tmp_path / sample[field]["path"]).unlink()
    output = tmp_path / "source.npz"
    plan_output = tmp_path / "preparation-plan.json"
    qc_output = tmp_path / "qc.json"
    encoder_manifest = builder._load_encoder_manifest(ENCODER_MANIFEST_PATH)
    builder.build_source(
        protocol_path,
        study_manifest_path,
        ENCODER_MANIFEST_PATH,
        CROP_MANIFEST_PATH,
        tmp_path,
        tmp_path / "unused-model",
        output,
        plan_output,
        qc_output,
        device="cpu",
        batch_size=3,
        encoder=_FixtureEncoder(encoder_manifest.sha256),
        annotation_receipt_path=annotation_receipt_path,
        annotation_predictions_path=annotation_predictions_path,
        training_label_provenance_receipt_path=training_label_receipt_path,
        training_label_ontology_source_path=ontology_source_path,
        annotation_validation_export_path=validation_export_path,
    )
    with np.load(output, allow_pickle=False) as archive:
        assert str(archive["schema_version"]) == builder.SOURCE_SCHEMA
        assert len(archive["observation_ids"]) == 48
        assert len(set(archive["observation_ids"].astype(str))) == 48
        assert set(archive["split_ids"].astype(str)) == {"development"}
        assert str(archive["study_stage"]) == "measurement_development"
        assert str(archive["source_scope"]) == "development_donors_only"
        assert not bool(archive["locked_donor_outcomes_materialized"])
        assert float(archive["opposite_pool_guard_um"]) > 0.0
        assert str(archive["study_manifest_sha256"]) == _sha256(study_manifest_path)
        assert set(archive["pool_roles"].astype(str)) == {"reference", "evaluation"}
        assert tuple(archive["reference_split_ids"].astype(str)) == (
            "primary",
            "reference_hash_fold_0",
            "reference_hash_fold_1",
        )
        assert archive["pool_roles_by_split"].shape == (48, 3)
        primary_evaluation = archive["pool_roles"].astype(str) == "evaluation"
        assert np.all(
            archive["pool_roles_by_split"].astype(str)[primary_evaluation] == "evaluation"
        )
        assert archive["frozen_features"].shape == (48, encoder_manifest.feature_width)
        assert archive["image_features"].shape == (48, 18, encoder_manifest.feature_width)
        assert tuple(archive["crop_ids"].astype(str)) == (
            "crop_112um",
            "nucleus_mask_only",
            "nucleus_mask_mean_fill_112um",
            "nucleus_mask_blurred_112um",
            "nucleus_shape_random_location_mean_fill_112um",
            "cell_mask_only",
            "cell_mask_mean_fill_112um",
            "cell_mask_blurred_112um",
            "cell_shape_random_location_mean_fill_112um",
            "context_ring_32_to_112um",
            "context_ring_64_to_112um",
            "target_cell_removed_112um",
            "target_cell_removed_mean_fill_112um",
            "target_cell_removed_blurred_112um",
            "random_location_cell_removed_mean_fill_112um",
            "crop_32um",
            "crop_64um",
            "blank_patch",
        )
        assert archive["crop_padding_fractions"].shape == (48, 18)
        assert archive["crop_mask_fractions"].shape == (48, 18)
        assert archive["crop_roles"].shape == (18,)
        assert archive["crop_comparison_families"].shape == (18,)
        common = np.isin(
            archive["crop_comparison_families"].astype(str),
            ["intrinsic_common_canvas", "mask_artifact_control", "context_control"],
        )
        np.testing.assert_array_equal(archive["crop_diameters_um"][common], 112.0)
        np.testing.assert_allclose(archive["crop_effective_mpp"][common], 0.5)
        assert set(
            archive["crop_ids"][
                archive["crop_comparison_families"].astype(str) == "resolution_sensitivity"
            ].astype(str)
        ) == {"crop_32um", "crop_64um"}
        crop_metadata = {
            crop_id: (mask_mode, fill_mode)
            for crop_id, mask_mode, fill_mode in zip(
                archive["crop_ids"].astype(str),
                archive["crop_mask_modes"].astype(str),
                archive["crop_fill_modes"].astype(str),
            )
        }
        assert crop_metadata["nucleus_mask_mean_fill_112um"] == (
            "keep_nucleus",
            "mean_color",
        )
        assert crop_metadata["nucleus_mask_blurred_112um"] == (
            "keep_nucleus",
            "blurred",
        )
        assert crop_metadata["cell_shape_random_location_mean_fill_112um"] == (
            "random_keep_cell",
            "mean_color",
        )
        assert crop_metadata["random_location_cell_removed_mean_fill_112um"] == (
            "random_remove_cell",
            "mean_color",
        )
        assert archive["molecular_targets"].shape == (48, 2)
        np.testing.assert_allclose(archive["molecular_targets"], np.log1p(2500.0))
        np.testing.assert_allclose(
            archive["whole_cell_molecular_targets"][:, 0], np.log1p(10_000.0 * 5.0 / 17.0)
        )
        np.testing.assert_allclose(
            archive["whole_cell_molecular_targets"][:, 1], np.log1p(10_000.0 * 4.0 / 17.0)
        )
        np.testing.assert_array_equal(archive["nucleus_library_sizes"], 16)
        np.testing.assert_array_equal(archive["whole_cell_library_sizes"], 17)
        np.testing.assert_array_equal(
            archive["nucleus_library_size_half_a"] + archive["nucleus_library_size_half_b"],
            archive["nucleus_library_sizes"],
        )
        np.testing.assert_array_equal(
            archive["whole_cell_library_size_half_a"] + archive["whole_cell_library_size_half_b"],
            archive["whole_cell_library_sizes"],
        )
        np.testing.assert_array_equal(
            archive["nucleus_target_counts_half_a"] + archive["nucleus_target_counts_half_b"],
            archive["nucleus_target_counts"],
        )
        np.testing.assert_array_equal(
            archive["whole_cell_target_counts_half_a"] + archive["whole_cell_target_counts_half_b"],
            archive["whole_cell_target_counts"],
        )
        assert str(archive["transcript_split_method"]) == "sha256-final-byte-lsb-v1"
        assert (
            str(archive["transcript_split_salt_sha256"])
            == hashlib.sha256(b"synthetic-split-v1").hexdigest()
        )
        assert int(archive["eligible_target_transcripts"]) == 48 * 9
        assert tuple(archive["program_names"].astype(str)) == ("synthetic_program",)
        np.testing.assert_array_equal(archive["program_gene_membership"], [[True, True]])
        assert len(archive["planned_stratum_ids"]) == 24
        assert str(archive["planned_stratum_manifest_sha256"]) == builder._canonical_sha256(
            list(archive["planned_stratum_ids"].astype(str))
        )
        assert archive["coordinate_features"].shape == (48, 24)
        assert archive["frozen_feature_names"].shape == (encoder_manifest.feature_width,)
        assert archive["coordinate_feature_names"].shape == (24,)
        assert archive["stain_features"].shape == (48, 70)
        assert archive["stain_feature_names"].shape == (70,)
        assert archive["composition_features"].shape == (48, 10)
        assert archive["composition_feature_names"].shape == (10,)
        assert archive["nuclear_morphometric_features"].shape == (48, 34)
        assert archive["cell_morphometric_features"].shape == (48, 33)
        assert archive["classical_morphology_features"].shape == (48, 67)
        assert np.isfinite(archive["stain_features"]).all()
        assert np.isfinite(archive["classical_morphology_features"]).all()
        assert tuple(archive["fine_type_marker_gene_ids"].astype(str)) == ("TYPE_FINE",)
        assert str(archive["fine_type_marker_panel_sha256"]) == builder._canonical_sha256(
            ["TYPE_FINE"]
        )
        source_independence = json.loads(str(archive["label_target_independence_json"]))
        assert source_independence == draft["label_target_independence"]
        assert str(archive["label_target_independence_sha256"]) == builder._canonical_sha256(
            source_independence
        )
        assert tuple(archive["annotation_feature_ids"].astype(str)) == ("ANN1", "ANN2")
        assert str(archive["label_source_kind"]) == "independent_annotation_prediction_export"
        assert str(archive["annotation_receipt_sha256"]) == _sha256(annotation_receipt_path)
        assert str(archive["annotation_receipt_path"]) == str(annotation_receipt_path.resolve())
        assert str(archive["annotation_prediction_export_sha256"]) == _sha256(
            annotation_predictions_path
        )
        assert str(archive["annotation_prediction_export_path"]) == str(
            annotation_predictions_path.resolve()
        )
        assert str(archive["training_label_provenance_receipt_sha256"]) == _sha256(
            training_label_receipt_path
        )
        assert str(archive["training_label_ontology_source_sha256"]) == _sha256(
            ontology_source_path
        )
        assert str(archive["annotation_validation_export_sha256"]) == _sha256(
            validation_export_path
        )
        assert set(archive["disease_estimands"].astype(str)) == {
            "disease_inclusive",
            "disease_adjusted",
        }
        assert set(archive["type_labels"].tolist()) == {0, 1}
        assert set(archive["type_names"].astype(str)) == {
            "Independent epithelial",
            "Independent immune",
        }
        assert set(archive["broad_type_labels"].tolist()) == {1, 2}
        assert tuple(archive["broad_type_names"].astype(str)) == builder.TYPE_NAMES
        assert archive["registration_qc_features"].shape == (48, 23)
        assert archive["registration_qc_feature_names"].shape == (23,)
        assert set(archive["section_ids"].astype(str)) == {
            sample_id
            for sample_id, (donor_id, _) in builder.SECTION_IDENTITIES.items()
            if donor_id in development
        }
        scoped_cell_ids = archive["cell_id"].astype(str)
        assert len(np.unique(scoped_cell_ids)) == len(scoped_cell_ids)
        np.testing.assert_array_equal(scoped_cell_ids, archive["observation_id"].astype(str))
        assert all(
            cell_id.startswith(section_id + ":")
            for cell_id, section_id in zip(
                scoped_cell_ids,
                archive["section_ids"].astype(str),
            )
        )
        assert set(archive["disease_statuses"].astype(str)) == {"Disease"}
        assert set(archive["site_ids"].astype(str)) == {"synthetic_site"}
        assert set(archive["batch_ids"].astype(str)) == {"TMA1:Run1"}
        assert str(archive["encoder_name"]) == encoder_manifest.repository
        assert str(archive["target_construction"]) == ("nucleus_overlapping_xenium_transcripts")
        assert str(archive["primary_crop_id"]) == "crop_112um"
        assert str(archive["crop_role"]) == "registered_cell_local_context_112um"
        assert float(archive["crop_diameter_um"]) == 112.0
        assert str(archive["mask_mode"]) == "none"
        assert not bool(archive["authorizes_nucleus_intrinsic_claim"])
        assert not bool(archive["nucleus_hypothesis_tested"])
        assert not bool(archive["cell_intrinsic_hypothesis_tested"])
        assert str(archive["g2_claim_scope"]) == "registered_cell_local_context_112um"
        assert archive["registration_qc_pass"].all()
        np.testing.assert_array_equal(archive["registration_cardinality"], 1)
        assert archive["target_qc_pass"].all()
        assert archive["crop_qc_pass"].all()
        assert archive["cellvit_sensitivity_features"].shape == (48, 2)
        assert archive["cellvit_context_features"].shape == (48, 2)
        assert set(archive["cellvit_sensitivity_feature_names"].astype(str)) == {
            "cellvit_log1p_count_Epithelial",
            "cellvit_log1p_count_Inflammatory",
        }
        provenance = json.loads(str(archive["provenance_json"]))
        assert provenance["native_xenium_registration_only"] is True
        assert provenance["cellvit_target_registration"] is False
        assert provenance["authorizes_nucleus_intrinsic_claim"] is False
        required_row_fields = {
            "observation_id",
            "donor_id",
            "patient_id",
            "section_id",
            "source_sample_id",
            "cell_id",
            "broad_type_label",
            "fine_type_label",
            "disease_state",
            "site_id",
            "batch_id",
            "block_id",
            "roi_id",
            "pool_role",
            "x_coordinate_um",
            "y_coordinate_um",
            "cell_centroid_x_um",
            "cell_centroid_y_um",
            "nucleus_centroid_x_um",
            "nucleus_centroid_y_um",
            "annotation_centroid_x_um",
            "annotation_centroid_y_um",
            "registration_distance_um",
            "cell_area_um2",
            "nucleus_area_um2",
            "library_size",
            "detected_target_genes",
            "transcript_qv_summary",
            "target_qc_pass",
            "registration_qc_pass",
            "registration_quality_scores",
            "registration_quality_strata",
            "registration_quality_cutoffs_json",
            "registration_quality_definition",
        }
        assert required_row_fields <= set(archive.files)
        required_matrices = {
            "nucleus_target_counts",
            "whole_cell_target_counts",
            "normalized_nucleus_targets",
            "normalized_whole_cell_targets",
            "coordinate_features",
            "stain_features",
            "nuclear_morphometric_features",
            "cell_morphometric_features",
            "cellvit_context_features",
            "image_features_by_crop_and_encoder",
            "pool_roles_by_split",
        }
        assert required_matrices <= set(archive.files)
        assert "registration_is_one_to_one" not in archive.files
        assert "study_manifest_sha256" in archive.files
    plan = json.loads(plan_output.read_text(encoding="utf-8"))
    assert plan["source_observations_sha256"] == _sha256(output)
    assert plan["source_schema"] == builder.SOURCE_SCHEMA
    assert plan["study_stage"] == "measurement_development"
    assert plan["source_scope"] == "development_donors_only"
    assert plan["locked_donor_outcomes_materialized"] is False
    assert plan["encoder_name"] == encoder_manifest.repository
    assert plan["target_construction"] == "nucleus_overlapping_xenium_transcripts"
    assert plan["type_names"] == ["Independent epithelial", "Independent immune"]
    assert plan["broad_type_names"] == list(builder.TYPE_NAMES)
    assert len(plan["frozen_feature_names"]) == encoder_manifest.feature_width
    assert plan["coordinate_feature_names"] == list(builder.SPATIAL_FEATURE_NAMES)
    assert plan["crop_metadata"]["primary_crop_id"] == "crop_112um"
    assert plan["crop_metadata"]["crop_role"] == ("registered_cell_local_context_112um")
    assert plan["crop_metadata"]["crop_diameter_um"] == 112.0
    assert plan["crop_metadata"]["source_mpp"] == builder.SOURCE_MPP
    assert plan["crop_metadata"]["mask_mode"] == "none"
    assert plan["crop_metadata"]["fill_mode"] == "none"
    assert len(plan["crop_metadata"]["variants"]) == 18
    assert plan["target"]["primary"] == "nucleus_overlapping_xenium_transcripts"
    assert plan["target"]["secondary"] == "whole_cell_xenium_transcripts"
    assert plan["crop_ids"] == list(np.load(output, allow_pickle=False)["crop_ids"].astype(str))
    assert plan["gene_ids"] == list(np.load(output, allow_pickle=False)["gene_ids"].astype(str))
    assert plan["type_names"] == list(np.load(output, allow_pickle=False)["type_names"].astype(str))
    assert plan["authorizes_nucleus_intrinsic_claim"] is False
    assert plan["nucleus_hypothesis_tested"] is False
    assert plan["cell_intrinsic_hypothesis_tested"] is False
    assert plan["fine_type_marker_panel_sha256"] == builder._canonical_sha256(["TYPE_FINE"])
    assert plan["label_target_independence"] == draft["label_target_independence"]
    assert plan["label_source_kind"] == "independent_annotation_prediction_export"
    assert plan["annotation_receipt_sha256"] == _sha256(annotation_receipt_path)
    assert plan["annotation_receipt_path"] == str(annotation_receipt_path.resolve())
    assert plan["annotation_prediction_export_sha256"] == _sha256(annotation_predictions_path)
    assert plan["annotation_prediction_export_path"] == str(annotation_predictions_path.resolve())
    assert plan["training_label_provenance_receipt_sha256"] == _sha256(training_label_receipt_path)
    assert plan["training_label_provenance_receipt_path"] == str(
        training_label_receipt_path.resolve()
    )
    assert plan["training_label_ontology_source_sha256"] == _sha256(ontology_source_path)
    assert plan["training_label_ontology_source_path"] == str(ontology_source_path.resolve())
    assert plan["annotation_validation_export_sha256"] == _sha256(validation_export_path)
    assert plan["annotation_validation_export_path"] == str(validation_export_path.resolve())
    assert plan["registration_quality"]["definition"] == builder.REGISTRATION_QUALITY_DEFINITION
    assert plan["registration_quality"]["cutoffs_fraction_of_limit"] == {
        "best": 0.25,
        "intermediate": 0.6,
        "near_threshold": 1.0,
    }
    assert plan["reference_splits"] == {
        "primary_split_id": "primary",
        "split_ids": ["primary", "reference_hash_fold_0", "reference_hash_fold_1"],
        "primary_evaluation_rows_fixed": True,
        "selection_unit": "spatial_block",
    }
    qc = json.loads(qc_output.read_text(encoding="utf-8"))
    assert qc["pass"] is True
    assert qc["study_stage"] == "measurement_development"
    assert qc["source_scope"] == "development_donors_only"
    assert qc["locked_donor_outcomes_materialized"] is False
    assert qc["source_observations"]["sha256"] == _sha256(output)
    assert qc["preparation_plan"]["sha256"] == _sha256(plan_output)
    assert qc["crops"]["g2_claim_scope"] == "registered_cell_local_context_112um"
    assert qc["crops"]["inpainting_substitute"] == "blurred_replacement"
    assert qc["feature_families"]["stain_quality_columns"] == 70
    assert qc["feature_families"]["nuclear_morphology_columns"] == 34
    assert qc["feature_families"]["cell_morphology_columns"] == 33
    assert set(qc["reference_evaluation_balance"]) == set(builder.DEVELOPMENT_DONORS)
    assert (
        qc["targets"]["whole_cell_eligible_transcripts"]
        > qc["targets"]["nucleus_eligible_transcripts"]
    )
    source_schema = json.loads(
        (REPOSITORY_ROOT / "configs" / "schemas" / "registered_observation.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert source_schema["properties"]["schema"]["const"] == builder.SOURCE_SCHEMA
    assert (
        "label_target_independence_sha256"
        in source_schema["properties"]["identities"]["items"]["enum"]
    )
    row_vocabulary = source_schema["properties"]["row_fields"]["items"]["enum"]
    identity_vocabulary = source_schema["properties"]["identities"]["items"]["enum"]
    assert {"registration_quality_scores", "registration_quality_strata"} <= set(row_vocabulary)
    assert {
        "registration_quality_cutoffs_json",
        "registration_quality_definition",
    } <= set(identity_vocabulary)


def test_protocol_and_input_hashes_fail_closed(tmp_path: Path) -> None:
    development = list(builder.DEVELOPMENT_DONORS)
    locked = list(builder.LOCKED_TEST_DONORS)
    samples = []
    for index, (sample_id, (donor, _)) in enumerate(sorted(builder.SECTION_IDENTITIES.items())):
        sample = {
            "sample_id": sample_id,
            "donor_id": donor,
            "pixel_size_um": 0.2125,
            "wsi": {"path": "w%d.tif" % index, "sha256": "1" * 64},
            "transcripts": {"path": "t%d.parquet" % index, "sha256": "2" * 64},
            "cell_seg": {"path": "c%d.parquet" % index, "sha256": "3" * 64},
            "nucleus_seg": {"path": "n%d.parquet" % index, "sha256": "4" * 64},
        }
        samples.append(sample)
    protocol = _protocol(samples, development, locked)
    protocol["annotation_export"] = {
        "path": "GSE250346/annotations.tsv.gz",
        "sha256": builder.ANNOTATION_SHA256,
    }
    encoder_manifest = builder._load_encoder_manifest(ENCODER_MANIFEST_PATH)
    crop_manifest = builder._load_crop_manifest(CROP_MANIFEST_PATH)
    builder._validate_protocol(protocol, encoder_manifest, crop_manifest)
    measurement_drift = json.loads(json.dumps(protocol))
    measurement_drift["frozen_h_meas_shared_measurement_thresholds"][
        "maximum_annotation_cell_p95_um"
    ] = 12.1
    with pytest.raises(ValueError, match="frozen H-MEAS measurement contract has drifted"):
        builder._validate_protocol(measurement_drift, encoder_manifest, crop_manifest)
    endpoint_drift = json.loads(json.dumps(protocol))
    endpoint_drift["confirmatory_primary_endpoint"]["donor_section_type_macro"][
        "minimum_effect"
    ] = 0.049
    with pytest.raises(ValueError, match="confirmatory primary endpoint has drifted"):
        builder._validate_protocol(endpoint_drift, encoder_manifest, crop_manifest)
    with pytest.raises(ValueError, match="dataset_revision differs"):
        builder._validate_protocol(
            {**protocol, "dataset_revision": "main"}, encoder_manifest, crop_manifest
        )
    wrong_samples = [dict(value) for value in samples]
    wrong_samples[0]["donor_id"] = "Patient 1"
    with pytest.raises(ValueError, match="frozen partitions|corrected true-donor identity"):
        builder._validate_protocol(
            {**protocol, "samples": wrong_samples}, encoder_manifest, crop_manifest
        )
    with pytest.raises(ValueError, match="broad/fine marker and evaluation genes"):
        builder._validate_protocol(
            {**protocol, "fine_type_marker_gene_ids": ["G1"]},
            encoder_manifest,
            crop_manifest,
        )
    boolean_only = json.loads(json.dumps(protocol))
    boolean_only["label_target_independence"]["establishes_full_target_independence"] = True
    with pytest.raises(ValueError, match="pending.*overstates"):
        builder._validate_protocol(boolean_only, encoder_manifest, crop_manifest)
    incomplete_cross_fit = json.loads(json.dumps(protocol))
    incomplete_cross_fit["label_target_independence"] = {
        "strategy": "same-cohort gene-disjoint annotation",
        "evidence_kind": "development_donor_cross_fitted_gene_disjoint_annotation",
        "annotation_receipt_sha256": "a" * 64,
        "ordered_annotation_feature_ids": ["ANN1", "ANN2"],
        "ordered_annotation_feature_ids_sha256": builder._canonical_sha256(["ANN1", "ANN2"]),
        "annotation_training_scope": "development_donors_only",
        "annotation_training_donor_ids": development,
        "annotation_training_donor_ids_sha256": builder._canonical_sha256(development),
        "training_label_ontology_source": "de_novo_gene_disjoint_development_ontology",
        "training_label_provenance_receipt_sha256": "b" * 64,
        "training_label_target_gene_overlap_count": 0,
        "training_labels_establish_target_independence": True,
        "annotation_quality_contract": builder.DEFAULT_ANNOTATION_QUALITY_CONTRACT,
        "locked_donors_used_for_training": False,
        "same_cohort_annotation": True,
        "cross_fitting_method": "pending",
        "cross_fitting_receipt_sha256": None,
        "establishes_full_target_independence": True,
        "limitation": "synthetic fixture",
    }
    with pytest.raises(ValueError, match="not donor-cross-fitted"):
        builder._validate_protocol(incomplete_cross_fit, encoder_manifest, crop_manifest)
    cohort_kind_conflict = json.loads(json.dumps(protocol))
    cohort_kind_conflict["label_target_independence"] = {
        "strategy": "inconsistent synthetic annotation contract",
        "evidence_kind": "development_donor_cross_fitted_gene_disjoint_annotation",
        "annotation_receipt_sha256": "a" * 64,
        "ordered_annotation_feature_ids": ["ANN1", "ANN2"],
        "ordered_annotation_feature_ids_sha256": builder._canonical_sha256(["ANN1", "ANN2"]),
        "annotation_training_scope": "orthogonal_no_rna_training",
        "annotation_training_donor_ids": [],
        "annotation_training_donor_ids_sha256": None,
        "training_label_ontology_source": "de_novo_gene_disjoint_development_ontology",
        "training_label_provenance_receipt_sha256": "b" * 64,
        "training_label_target_gene_overlap_count": 0,
        "training_labels_establish_target_independence": True,
        "annotation_quality_contract": builder.DEFAULT_ANNOTATION_QUALITY_CONTRACT,
        "locked_donors_used_for_training": False,
        "same_cohort_annotation": False,
        "cross_fitting_method": "not_applicable",
        "cross_fitting_receipt_sha256": None,
        "establishes_full_target_independence": True,
        "limitation": "synthetic fixture",
    }
    with pytest.raises(ValueError, match="conflicts with same-cohort annotation scope"):
        builder._validate_protocol(cohort_kind_conflict, encoder_manifest, crop_manifest)
    missing = builder.InputFile("missing.tif", "1" * 64)
    with pytest.raises(ValueError, match="missing or differs"):
        builder._resolve_input(tmp_path.resolve(), missing)


def test_builder_cannot_materialize_confirmatory_rows_from_an_unopened_manifest(
    tmp_path: Path,
) -> None:
    draft = REPOSITORY_ROOT / "manifests" / "studies" / "hest_lung_cell_association.draft.json"
    with pytest.raises(ValueError, match="must have status opened"):
        builder.build_source(
            tmp_path / "unread-protocol.json",
            draft,
            ENCODER_MANIFEST_PATH,
            CROP_MANIFEST_PATH,
            tmp_path / "unread-data",
            tmp_path / "unread-model",
            tmp_path / "source.npz",
            tmp_path / "plan.json",
            tmp_path / "qc.json",
            device="cpu",
        )


def test_retrospective_builder_requires_resource_cap_above_pool_support(tmp_path: Path) -> None:
    draft = REPOSITORY_ROOT / "manifests" / "studies" / "hest_lung_cell_association.draft.json"
    protocol = REPOSITORY_ROOT / "configs" / "hest_lung_cell_protocol.json"
    with pytest.raises(ValueError, match="cell cap is below"):
        builder.build_source(
            protocol,
            draft,
            ENCODER_MANIFEST_PATH,
            CROP_MANIFEST_PATH,
            tmp_path / "unread-data",
            tmp_path / "unread-model",
            tmp_path / "source.npz",
            tmp_path / "plan.json",
            tmp_path / "qc.json",
            device="cpu",
            retrospective=True,
            retrospective_max_cells_per_section_type_pool=1,
        )


def test_retrospective_builder_materializes_all_donors_and_four_crops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("zarr")
    development = list(builder.DEVELOPMENT_DONORS)
    locked = list(builder.LOCKED_TEST_DONORS)
    samples = []
    for index, (sample_id, (donor, _)) in enumerate(sorted(builder.SECTION_IDENTITIES.items())):
        sample = _write_sample(tmp_path, sample_id, "synthetic-frozen-v1", 90 + index)
        sample["donor_id"] = donor
        samples.append(sample)
    annotation_path = tmp_path / "GSE250346" / "annotations.tsv.gz"
    annotation_path.parent.mkdir(parents=True)
    _write_annotations(annotation_path)
    monkeypatch.setattr(builder, "ANNOTATION_SHA256", _sha256(annotation_path))
    monkeypatch.setattr(builder, "ANNOTATION_ROWS", 80)
    protocol = _protocol(samples, development, locked)
    protocol["annotation_export"] = {
        "path": str(annotation_path.relative_to(tmp_path)),
        "sha256": builder.ANNOTATION_SHA256,
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    draft = json.loads(
        (
            REPOSITORY_ROOT / "manifests" / "studies" / "hest_lung_cell_association.draft.json"
        ).read_text(encoding="utf-8")
    )
    draft["analysis_plan_sha256"] = _sha256(protocol_path)
    draft["candidate_target_gene_panel_sha256"] = builder._canonical_sha256(["G1", "G2"])
    draft["type_marker_panel_sha256"] = builder._canonical_sha256(["TYPE_FINE"])
    draft["label_target_independence"] = {
        **draft["label_target_independence"],
        **protocol["label_target_independence"],
    }
    study_manifest = tmp_path / "retrospective-study.draft.json"
    study_manifest.write_text(json.dumps(draft), encoding="utf-8")
    output = tmp_path / "retrospective-source.npz"
    encoder_manifest = builder._load_encoder_manifest(ENCODER_MANIFEST_PATH)
    builder.build_source(
        protocol_path,
        study_manifest,
        ENCODER_MANIFEST_PATH,
        CROP_MANIFEST_PATH,
        tmp_path,
        tmp_path / "unused-model",
        output,
        tmp_path / "plan.json",
        tmp_path / "qc.json",
        device="cpu",
        batch_size=3,
        encoder=_FixtureEncoder(encoder_manifest.sha256),
        retrospective=True,
        retrospective_max_cells_per_section_type_pool=1,
        retrospective_max_observations=100,
    )
    with np.load(output, allow_pickle=False) as archive:
        assert str(archive["schema_version"]) == builder.RETROSPECTIVE_SOURCE_SCHEMA
        assert str(archive["study_stage"]) == builder.RETROSPECTIVE_SOURCE_STAGE
        assert set(archive["donor_ids"].astype(str)) == set(development + locked)
        assert set(archive["section_ids"].astype(str)) == set(builder.SECTION_IDENTITIES)
        assert tuple(archive["crop_ids"].astype(str)) == builder.RETROSPECTIVE_CROP_IDS
        assert archive["image_features"].shape[1:] == (4, encoder_manifest.feature_width)
        assert not bool(archive["authorizes_h_cell"])
        assert not bool(archive["authorizes_h_intrinsic"])


def test_expression_aggregation_rejects_duplicate_transcript_ids(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "duplicate-transcripts.parquet"
    parquet.write_table(
        pa.table(
            {
                "transcript_id": [7, 7],
                "cell_id": ["cell-1", "cell-1"],
                "feature_name": ["G1", "G1"],
                "qv": [40.0, 40.0],
                "overlaps_nucleus": [1, 1],
            }
        ),
        path,
    )
    with pytest.raises(ValueError, match="duplicate transcript_id"):
        builder._aggregate_expression(
            path,
            ["cell-1"],
            ["cell-1"],
            ["G1"],
            ["G1"],
            minimum_qv=20.0,
            excluded_prefixes=builder.CONTROL_PREFIXES,
            split_salt="synthetic-split-v1",
        )


def test_encoder_manifests_record_access_status_and_handcrafted_control() -> None:
    from heir.features import HandcraftedPatchEncoder, load_encoder_manifest

    uni2h = load_encoder_manifest(ENCODER_MANIFEST_PATH)
    assert uni2h.available
    assert uni2h.repository == "MahmoodLab/UNI2-h"
    hoptimus = REPOSITORY_ROOT / "manifests" / "encoders" / "hoptimus1.json"
    status = load_encoder_manifest(hoptimus)
    assert status.available
    assert status.repository == "bioptimus/H-optimus-1"
    assert status.revision == "3592cb220dec7a150c5d7813fb56e68bd57473b9"
    assert status.feature_width == 1536
    assert status.input_pixels == 224
    assert status.fine_tuning == "prohibited"
    patches = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    features = HandcraftedPatchEncoder().encode(patches)
    assert features.shape == (2, 12)
    assert np.isfinite(features).all()


def test_future_hest_plan_role_tracks_the_selected_primary_encoder() -> None:
    assert (
        builder._experiment_role("bioptimus/H-optimus-1", False)
        == "primary_hest_hoptimus1"
    )
    assert builder._experiment_role("MahmoodLab/UNI2-h", False) == "primary_hest_uni2h"
    assert (
        builder._experiment_role("bioptimus/H-optimus-1", True)
        == "hest_hoptimus1_encoder_qualification"
    )


def test_hoptimus_qualification_is_exact_and_retrospective_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from heir.features import load_encoder_manifest

    manifest = load_encoder_manifest(REPOSITORY_ROOT / "manifests" / "encoders" / "hoptimus1.json")
    builder._validate_retrospective_encoder_qualification(manifest, retrospective=True)
    with pytest.raises(ValueError, match="exact frozen H-optimus-1"):
        builder._validate_retrospective_encoder_qualification(manifest, retrospective=False)
    with pytest.raises(ValueError, match="exact frozen H-optimus-1"):
        builder._validate_retrospective_encoder_qualification(
            replace(manifest, revision="floating-main"), retrospective=True
        )
    native_pixels = int(round(112.0 / 0.2125))
    probe_input = (
        np.arange(native_pixels * native_pixels * 3, dtype=np.uint32).reshape(
            native_pixels, native_pixels, 3
        )
        % 251
    ).astype(np.uint8)
    probe_output = builder._resize_hoptimus1_batch(probe_input[None], 224)[0]
    parity = {
        "schema": builder.HOPTIMUS1_PARITY_SCHEMA,
        "status": "passed",
        "passed": True,
        "repository": builder.HOPTIMUS1_REPOSITORY,
        "revision": builder.HOPTIMUS1_REVISION,
        "encoder_manifest_sha256": manifest.sha256,
        "implementation_sha256": builder.HOPTIMUS1_PARITY_IMPLEMENTATION_SHA256,
        "model": {
            "repository": builder.HOPTIMUS1_REPOSITORY,
            "revision": builder.HOPTIMUS1_REVISION,
            "manifest_sha256": builder.HOPTIMUS1_MANIFEST_SHA256,
            "config_sha256": builder.HOPTIMUS1_CONFIG_SHA256,
            "checkpoint_sha256": builder.HOPTIMUS1_CHECKPOINT_SHA256,
            "checkpoint_filename": "model.safetensors",
            "local_fp16_path": "HOptimus1Encoder.encode_exact_biological_path",
        },
        "embedding_contract": {"shape": [5, 1536], "all_finite": True},
        "production_runtime_contract": {
            "code_sha256": {
                relative: _sha256(REPOSITORY_ROOT / relative)
                for relative in builder.HOPTIMUS1_PRODUCTION_RUNTIME_FILES
            },
            "resampling_probe": {
                "input_shape": list(probe_input.shape),
                "input_dtype": str(probe_input.dtype),
                "input_sha256": hashlib.sha256(probe_input.tobytes()).hexdigest(),
                "output_shape": list(probe_output.shape),
                "output_dtype": str(probe_output.dtype),
                "output_sha256": hashlib.sha256(probe_output.tobytes()).hexdigest(),
                "implementation": "Pillow.Image.Resampling.BICUBIC",
                "resampling_count": 1,
            },
        },
        "comparisons": {
            "official_fp32_vs_local_fp32": {
                "passed": True,
                "minimum_required_cosine": 0.999999,
                "minimum_cosine": 1.0,
                "mean_cosine": 1.0,
                "mean_absolute_error": 0.0,
                "maximum_absolute_error": 0.0,
            },
            "local_fp32_vs_local_fp16": {
                "passed": True,
                "minimum_required_cosine": 0.9999,
                "minimum_cosine": 0.99999,
                "mean_cosine": 0.999999,
                "mean_absolute_error": 0.001,
                "maximum_absolute_error": 0.01,
            },
        },
    }
    parity_path = tmp_path / "parity.json"
    parity_path.write_text(json.dumps(parity), encoding="utf-8")
    monkeypatch.setattr(builder, "HOPTIMUS1_PARITY_RECEIPT_SHA256", _sha256(parity_path))
    assert builder._validate_hoptimus1_parity_receipt(parity_path, manifest) == _sha256(parity_path)
    parity["comparisons"]["local_fp32_vs_local_fp16"]["passed"] = False
    parity_path.write_text(json.dumps(parity), encoding="utf-8")
    monkeypatch.setattr(builder, "HOPTIMUS1_PARITY_RECEIPT_SHA256", _sha256(parity_path))
    with pytest.raises(ValueError, match="comparisons"):
        builder._validate_hoptimus1_parity_receipt(parity_path, manifest)


def test_only_encoder_changed_receipt_compares_every_non_encoder_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comparison = {
        "schema_version": np.asarray(builder.RETROSPECTIVE_SOURCE_SCHEMA),
        "study_stage": np.asarray(builder.RETROSPECTIVE_SOURCE_STAGE),
        "analysis_status": np.asarray("retrospective_exposed_non_authorizing"),
        "encoder_name": np.asarray(builder.UNI2H_REPOSITORY),
        "encoder_manifest_sha256": np.asarray("1" * 64),
        "image_features": np.zeros((2, 1, 3), dtype=np.float32),
        "observation_ids": np.asarray(["S:1", "S:2"]),
        "pool_roles": np.asarray(["reference", "evaluation"]),
        "molecular_targets": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "nucleus_target_counts_half_a": np.asarray([[1, 0], [0, 1]], dtype=np.uint32),
        "program_gene_membership": np.asarray([[True, False]], dtype=np.bool_),
        "crop_ids": np.asarray(["crop_112um"]),
        "technical_covariates": np.asarray([[0.1], [0.2]], dtype=np.float32),
    }
    comparison_path = tmp_path / "registered-uni2h.npz"
    np.savez(comparison_path, **comparison)
    monkeypatch.setattr(
        builder,
        "REGISTERED_UNI2H_COMPARISON_SOURCE_SHA256",
        _sha256(comparison_path),
    )
    assert builder._validate_registered_comparison_source(comparison_path) == comparison_path

    candidate = dict(comparison)
    candidate["encoder_name"] = np.asarray(builder.HOPTIMUS1_REPOSITORY)
    candidate["encoder_manifest_sha256"] = np.asarray(builder.HOPTIMUS1_MANIFEST_SHA256)
    candidate["image_features"] = np.ones((2, 1, 3), dtype=np.float32)
    receipt = builder._validate_only_encoder_changed(candidate, comparison_path)
    assert receipt["compared_field_count"] == 10
    assert len(str(receipt["field_hash_manifest_sha256"])) == 64

    candidate["molecular_targets"] = comparison["molecular_targets"].copy()
    candidate["molecular_targets"][0, 0] += 1.0
    with pytest.raises(ValueError, match="molecular_targets"):
        builder._validate_only_encoder_changed(candidate, comparison_path)


def test_hest_section_cache_binds_geometry_and_rejects_malformed_arrays() -> None:
    sample = builder.Sample(
        sample_id="NCBI856",
        donor_id="D1",
        split_id="development",
        pixel_size_um=builder.SOURCE_MPP,
        wsi=builder.InputFile("wsi.tif", "1" * 64),
        transcripts=builder.InputFile("transcripts.parquet", "2" * 64),
        cell_seg=builder.InputFile("cell.parquet", "3" * 64),
        nucleus_seg=builder.InputFile("nucleus.parquet", "4" * 64),
        cellvit_seg=builder.InputFile("cellvit.geojson", "5" * 64),
    )
    rows = SimpleNamespace(
        observation_ids=["NCBI856:1", "NCBI856:2"],
        centres=[(10.0, 20.0), (30.0, 40.0)],
        cell_centres=[(10.5, 20.5), (30.5, 40.5)],
        nucleus_vertices=[
            np.asarray([[9.0, 19.0], [11.0, 19.0], [10.0, 21.0]]),
            np.asarray([[29.0, 39.0], [31.0, 39.0], [30.0, 41.0]]),
        ],
        cell_vertices=[
            np.asarray([[8.0, 18.0], [12.0, 18.0], [10.0, 22.0]]),
            np.asarray([[28.0, 38.0], [32.0, 38.0], [30.0, 42.0]]),
        ],
        cellvit_names=("inflammatory",),
        cellvit_centres=[(11.0, 21.0), (31.0, 41.0)],
    )
    identity = builder._section_cache_input_identity(rows, 0, 2, sample)
    changed_rows = SimpleNamespace(**vars(rows))
    changed_rows.nucleus_vertices = [value.copy() for value in rows.nucleus_vertices]
    changed_rows.nucleus_vertices[0][0, 0] += 0.25
    assert builder._section_cache_input_identity(changed_rows, 0, 2, sample) != identity

    cache_identity = {
        "schema": builder.HEST_ENCODER_CACHE_SCHEMA,
        "encoder_manifest_sha256": "a" * 64,
        "crop_materialization_sha256": "b" * 64,
        "section_inputs": identity,
    }
    identity_sha256 = builder._canonical_sha256(cache_identity)
    cache = {
        "schema": np.asarray(builder.HEST_ENCODER_CACHE_SCHEMA),
        "cache_identity_sha256": np.asarray(identity_sha256),
        "cache_identity_json": np.asarray(
            json.dumps(cache_identity, sort_keys=True, separators=(",", ":"))
        ),
        "observation_ids": np.asarray(rows.observation_ids),
        "image_features": np.zeros((2, 2, 3), dtype=np.float32),
        "crop_padding_fractions": np.zeros((2, 2), dtype=np.float32),
        "crop_mask_fractions": np.zeros((2, 2), dtype=np.float32),
        "stain_quality_local": np.zeros(
            (2, len(builder.STAIN_QUALITY_FEATURE_NAMES)), dtype=np.float32
        ),
        "nucleus_texture_features": np.zeros(
            (2, len(builder.REGION_TEXTURE_FEATURE_NAMES)), dtype=np.float32
        ),
        "cell_texture_features": np.zeros(
            (2, len(builder.REGION_TEXTURE_FEATURE_NAMES)), dtype=np.float32
        ),
        "coordinate_base_features": np.zeros(
            (2, len(builder.COORDINATE_BASE_FEATURE_NAMES)), dtype=np.float32
        ),
        "cellvit_nearest_features": np.zeros((2, 3), dtype=np.float32),
        "cellvit_nearest_padding_fractions": np.zeros(2, dtype=np.float32),
    }
    loaded = builder._load_hest_encoder_section_cache(
        cache,
        expected_identity_sha256=identity_sha256,
        expected_observation_ids=np.asarray(rows.observation_ids),
        rows=2,
        crops=2,
        feature_width=3,
        has_cellvit=True,
    )
    assert loaded["image_features"].shape == (2, 2, 3)
    cache["image_features"] = cache["image_features"].copy()
    cache["image_features"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="image_features"):
        builder._load_hest_encoder_section_cache(
            cache,
            expected_identity_sha256=identity_sha256,
            expected_observation_ids=np.asarray(rows.observation_ids),
            rows=2,
            crops=2,
            feature_width=3,
            has_cellvit=True,
        )


def test_hoptimus_qualification_resamples_native_canvas_exactly_once() -> None:
    from PIL import Image

    patch = np.arange(7 * 7 * 3, dtype=np.uint8).reshape(7, 7, 3)
    observed = builder._resize_hoptimus1_batch(patch[None], 4)
    expected = np.asarray(
        Image.fromarray(patch, mode="RGB").resize(
            (4, 4), resample=Image.Resampling.BICUBIC
        ),
        dtype=np.uint8,
    )
    assert observed.shape == (1, 4, 4, 3)
    assert observed.dtype == np.uint8
    np.testing.assert_array_equal(observed[0], expected)
    with pytest.raises(ValueError, match="requires one native-canvas-to-224 resampling"):
        builder._resize_hoptimus1_batch(observed, 4)
