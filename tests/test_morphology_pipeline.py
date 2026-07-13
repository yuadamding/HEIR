from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from heir.data import MorphologyRidgeDatasetArtifact, ordered_ids_sha256
from heir.data.study_manifest import DEFAULT_ANNOTATION_QUALITY_CONTRACT
from heir.evaluation.morphology_artifact_qc import (
    locked_measurement_audit_report,
    reference_evaluation_balance_report,
)
from heir.utils import sha256_file


def _script(name: str):
    path = Path(__file__).parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = _script("prepare_morphology_ridge_artifacts")
BENCHMARK = _script("benchmark_morphology_state_gate")
OPENING_RECEIPT_SHA256 = "c" * 64


def test_preparation_uses_shared_scientific_qc_implementation() -> None:
    assert PREPARE.locked_measurement_audit_report is locked_measurement_audit_report
    assert PREPARE.reference_evaluation_balance_report is reference_evaluation_balance_report


def _locked_audit() -> dict[str, object]:
    return {
        "audit_timing": "after_confirmatory_lock_before_morphology_inference",
        "selection_changes_forbidden": True,
        "coverage_denominator": "all_h_meas_supported_fine_types_and_locked_donors",
        "maximum_annotation_nucleus_p95_um": 8.0,
        "maximum_annotation_cell_p95_um": 12.0,
        "maximum_cell_nucleus_p95_um": 8.0,
        "maximum_registration_nucleus_diameter_ratio_p95": 0.5,
        "maximum_registration_nearest_neighbor_ratio_p95": 0.5,
        "best_registration_quality_max_fraction_of_limit": 0.25,
        "intermediate_registration_quality_max_fraction_of_limit": 0.6,
        "maximum_registration_outlier_fraction": 0.05,
        "maximum_nucleus_outside_cell_fraction": 0.01,
        "minimum_nucleus_cell_area_ratio": 0.05,
        "maximum_nucleus_cell_area_ratio": 0.95,
        "maximum_segmentation_outlier_fraction": 0.05,
        "maximum_crop_padding_p95": 0.25,
        "mostly_padded_cutoff": 0.5,
        "maximum_mostly_padded_fraction": 0.01,
        "minimum_within_fine_type_reliability": 0.4,
        "minimum_reliability_rows": 4,
        "minimum_locked_donor_type_reliability_fraction": 0.5,
    }


def _independence() -> dict[str, object]:
    annotation_ids = ["ANN_A", "ANN_B"]
    target_ids = ["g2", "g1"]
    return {
        "strategy": "development-donor cross-fitted gene-disjoint annotation",
        "evidence_kind": "development_donor_cross_fitted_gene_disjoint_annotation",
        "annotation_receipt_sha256": "9" * 64,
        "ordered_annotation_feature_ids": annotation_ids,
        "ordered_annotation_feature_ids_sha256": ordered_ids_sha256(annotation_ids),
        "ordered_target_gene_ids": target_ids,
        "ordered_target_gene_ids_sha256": ordered_ids_sha256(target_ids),
        "annotation_target_overlap_count": 0,
        "annotation_training_scope": "development_donors_only",
        "annotation_training_donor_ids": ["D1", "D2", "D3"],
        "annotation_training_donor_ids_sha256": ordered_ids_sha256(["D1", "D2", "D3"]),
        "training_label_ontology_source": "de_novo_gene_disjoint_development_ontology",
        "training_label_provenance_receipt_sha256": "8" * 64,
        "training_label_target_gene_overlap_count": 0,
        "training_labels_establish_target_independence": True,
        "annotation_quality_contract": copy.deepcopy(DEFAULT_ANNOTATION_QUALITY_CONTRACT),
        "locked_donors_used_for_training": False,
        "same_cohort_annotation": True,
        "cross_fitting_method": "leave_one_donor_out",
        "cross_fitting_receipt_sha256": "a" * 64,
        "establishes_full_target_independence": True,
        "limitation": "Synthetic within-source test contract.",
    }


def _source(
    path: Path,
    *,
    omitted_donor_types: frozenset[tuple[str, str]] = frozenset(),
    locked_audit: dict[str, object] | None = None,
    locked_audit_sha256: str | None = None,
    independence_contract: dict[str, object] | None = None,
    alternate_reference_deficit: tuple[str, str] | None = None,
    two_section_donor: str | None = None,
    unreliable_locked_stratum: tuple[str, str, str] | None = None,
    stale_locked_measurement_composite: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    development = ("D1", "D2", "D3")
    locked = ("L1", "L2")
    donors = development + locked
    observation_ids = []
    donor_ids = []
    section_ids = []
    block_ids = []
    roi_ids = []
    pool_roles = []
    fine_types = []
    labels = []
    targets = []
    coordinates = []
    image_features = []
    for donor_index, donor in enumerate(donors):
        base_section = "section_%s" % donor
        for type_index, fine_type in enumerate(("type_a", "type_b")):
            if (donor, fine_type) in omitted_donor_types:
                continue
            for pool_index, pool in enumerate(("reference", "evaluation")):
                for row in range(4):
                    section = (
                        "%s_%s" % (base_section, "a" if row < 2 else "b")
                        if donor == two_section_donor
                        else base_section
                    )
                    observation_ids.append("%s-%s-%s-%d" % (donor, fine_type, pool, row))
                    donor_ids.append(donor)
                    section_ids.append(section)
                    block_ids.append("%s/%s/block_%s" % (donor, section, pool))
                    roi_ids.append("%s/%s/roi_%s_%d" % (donor, section, pool, type_index))
                    pool_roles.append(pool)
                    fine_types.append(fine_type)
                    labels.append(type_index)
                    state = float(row - 1.5 + type_index * 0.2)
                    section_shift = 100.0 if section.endswith("_b") else 0.0
                    targets.append(
                        [
                            donor_index + state + section_shift,
                            2.0 * type_index - state + section_shift,
                            0.5 * state + section_shift,
                        ]
                    )
                    coordinates.append([row / 4.0, float(type_index)])
                    primary = np.asarray([state, -state], dtype=np.float64)
                    image_features.append(
                        np.stack(
                            tuple(
                                primary * (1.0 - 0.02 * crop_index)
                                for crop_index in range(len(PREPARE.HEST_CROP_CONTRACT))
                            )
                        )
                    )
    rows = len(observation_ids)
    planned = sorted(
        "%s|%s|%s" % (donor, section, fine_type)
        for donor in donors
        for section in (
            ("section_%s_a" % donor, "section_%s_b" % donor)
            if donor == two_section_donor
            else ("section_%s" % donor,)
        )
        for fine_type in ("type_a", "type_b")
    )
    crop_variants = [
        {"crop_id": crop_id, "role": role, "comparison_family": family}
        for crop_id, (role, family) in PREPARE.HEST_CROP_CONTRACT.items()
    ]
    primary_roles = np.asarray(pool_roles)
    alternate_zero = primary_roles.copy()
    alternate_one = primary_roles.copy()
    reference = primary_roles == "reference"
    identities = np.asarray(observation_ids)
    alternate_eligible = reference & (np.asarray(donor_ids) != two_section_donor)
    alternate_zero[alternate_eligible & np.char.endswith(identities, "-0")] = "excluded"
    alternate_one[alternate_eligible & np.char.endswith(identities, "-1")] = "excluded"
    if alternate_reference_deficit is not None:
        donor, fine_type = alternate_reference_deficit
        affected = np.flatnonzero(
            reference & (np.asarray(donor_ids) == donor) & (np.asarray(fine_types) == fine_type)
        )
        alternate_zero[affected] = "excluded"
        alternate_zero[affected[-1]] = "reference"
    roles_by_split = np.column_stack((primary_roles, alternate_zero, alternate_one))
    split_counts = np.maximum(
        1, np.rint((np.asarray(targets, dtype=np.float64) - np.min(targets) + 1.0) * 5)
    ).astype(np.int64)
    split_counts_half_b = split_counts.copy()
    if unreliable_locked_stratum is not None:
        donor, section, fine_type = unreliable_locked_stratum
        selected = np.flatnonzero(
            (np.asarray(donor_ids) == donor)
            & (np.asarray(section_ids) == section)
            & (np.asarray(fine_types) == fine_type)
        )
        if not len(selected):
            raise ValueError("test unreliable locked stratum has no rows")
        split_counts_half_b[selected] = split_counts_half_b[selected[::-1]]
    locked_audit = _locked_audit() if locked_audit is None else locked_audit
    locked_audit_json = json.dumps(locked_audit, sort_keys=True)
    locked_audit_sha = hashlib.sha256(
        json.dumps(locked_audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    independence = _independence() if independence_contract is None else independence_contract
    independence_json = json.dumps(independence, sort_keys=True, separators=(",", ":"))
    independence_sha = hashlib.sha256(independence_json.encode("utf-8")).hexdigest()
    registration_qc_pass = np.ones(rows, dtype=np.bool_)
    registration_quality_scores = np.resize(np.asarray([0.1, 0.4, 0.8]), rows)
    registration_quality_strata = np.asarray(
        [
            "best" if value <= 0.25 else "intermediate" if value <= 0.6 else "near_threshold"
            for value in registration_quality_scores
        ]
    )
    segmentation_qc_pass = np.ones(rows, dtype=np.bool_)
    crop_qc_pass = np.ones(rows, dtype=np.bool_)
    locked_measurement_qc_pass = np.ones(rows, dtype=np.bool_)
    crop_padding_fractions = np.zeros((rows, len(crop_variants)))
    if stale_locked_measurement_composite:
        locked_row = int(np.flatnonzero(np.isin(np.asarray(donor_ids), np.asarray(locked)))[0])
        crop_padding_fractions[locked_row, :] = 0.26
        crop_qc_pass[locked_row] = False
    np.savez_compressed(
        path,
        schema_version=np.asarray("synthetic.registered.v1"),
        observation_ids=np.asarray(observation_ids),
        donor_ids=np.asarray(donor_ids),
        block_ids=np.asarray(block_ids),
        roi_ids=np.asarray(roi_ids),
        section_ids=np.asarray(section_ids),
        disease_statuses=np.asarray(
            ["Disease" if donor in {"D2", "L2"} else "Control" for donor in donor_ids]
        ),
        site_ids=np.repeat("lung", rows),
        batch_ids=np.asarray(["batch_%s" % donor for donor in donor_ids]),
        pool_roles=np.asarray(pool_roles),
        reference_split_ids=np.asarray(
            ["primary", "reference_hash_fold_0", "reference_hash_fold_1"]
        ),
        pool_roles_by_split=roles_by_split,
        type_labels=np.asarray(labels, dtype=np.int64),
        fine_type_ids=np.asarray(fine_types),
        type_names=np.asarray(["type_a", "type_b"]),
        nucleus_molecular_targets=np.asarray(targets, dtype=np.float64),
        nucleus_target_counts_half_a=split_counts,
        nucleus_target_counts_half_b=split_counts_half_b,
        nucleus_library_size_half_a=split_counts.sum(axis=1),
        nucleus_library_size_half_b=split_counts_half_b.sum(axis=1),
        gene_ids=np.asarray(["g1", "g2", "g3"]),
        coordinate_features=np.asarray(coordinates, dtype=np.float64),
        coordinate_feature_names=np.asarray(["x", "y"]),
        spatial_features=np.asarray(coordinates, dtype=np.float64),
        spatial_feature_names=np.asarray(["smooth_x", "smooth_y"]),
        image_features=np.asarray(image_features, dtype=np.float64),
        crop_ids=np.asarray([value["crop_id"] for value in crop_variants]),
        crop_roles=np.asarray([value["role"] for value in crop_variants]),
        crop_comparison_families=np.asarray(
            [value["comparison_family"] for value in crop_variants]
        ),
        primary_crop_id=np.asarray("crop_112um"),
        technical_covariates=np.ones((rows, 1), dtype=np.float64),
        technical_covariate_names=np.asarray(["log1p_library_size"]),
        stain_features=np.column_stack((np.arange(rows) % 3, np.arange(rows) % 5)),
        stain_feature_names=np.asarray(["hematoxylin_od", "eosin_od"]),
        composition_features=np.empty((rows, 0)),
        composition_feature_names=np.asarray([], dtype=str),
        nuclear_morphometric_features=np.column_stack((np.arange(rows) % 7, np.arange(rows) % 11)),
        nuclear_morphometric_feature_names=np.asarray(["area", "eccentricity"]),
        cell_morphometric_features=(np.arange(rows) % 13)[:, None],
        cell_morphometric_feature_names=np.asarray(["cell_area"]),
        cellvit_context_features=(np.arange(rows) % 17)[:, None],
        cellvit_context_feature_names=np.asarray(["cellvit_density"]),
        local_density_features=(np.arange(rows) % 19)[:, None],
        local_density_feature_names=np.asarray(["neighbors_50um"]),
        boundary_features=(np.arange(rows) % 23)[:, None],
        boundary_feature_names=np.asarray(["distance_to_boundary"]),
        planned_stratum_ids=np.asarray(planned),
        planned_stratum_manifest_sha256=np.asarray("1" * 64),
        registration_qc_pass=registration_qc_pass,
        registration_quality_scores=registration_quality_scores,
        registration_quality_strata=registration_quality_strata,
        registration_quality_cutoffs_json=np.asarray(
            json.dumps(
                {"best": 0.25, "intermediate": 0.6, "near_threshold": 1.0},
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        registration_quality_definition=np.asarray(PREPARE.REGISTRATION_QUALITY_DEFINITION),
        segmentation_qc_pass=segmentation_qc_pass,
        locked_measurement_qc_pass=locked_measurement_qc_pass,
        locked_measurement_audit_thresholds_json=np.asarray(locked_audit_json),
        locked_measurement_audit_thresholds_sha256=np.asarray(
            locked_audit_sha if locked_audit_sha256 is None else locked_audit_sha256
        ),
        target_qc_pass=np.ones(rows, dtype=np.bool_),
        crop_qc_pass=crop_qc_pass,
        crop_padding_fractions=crop_padding_fractions,
        registration_cardinality=np.ones(rows, dtype=np.int64),
        registration_distance_um=np.zeros(rows),
        annotation_cell_distance_um=np.zeros(rows),
        cell_nucleus_centroid_distance_um=np.zeros(rows),
        nucleus_centroid_inside_cell=np.ones(rows, dtype=np.bool_),
        nucleus_area_um2=np.full(rows, 50.0),
        cell_area_um2=np.full(rows, 100.0),
        fine_type_marker_gene_ids=np.asarray(["marker"]),
        fine_type_marker_panel_sha256=np.asarray(ordered_ids_sha256(["marker"])),
        label_target_independence_json=np.asarray(independence_json),
        label_target_independence_sha256=np.asarray(independence_sha),
        annotation_feature_ids=np.asarray(independence["ordered_annotation_feature_ids"]),
        annotation_receipt_sha256=np.asarray(independence["annotation_receipt_sha256"]),
        annotation_prediction_export_sha256=np.asarray("3" * 64),
        study_stage=np.asarray("confirmatory_morphology"),
        study_manifest_sha256=np.asarray("7" * 64),
        opening_receipt_sha256=np.asarray(OPENING_RECEIPT_SHA256),
        source_scope=np.asarray("development_and_locked_after_confirmatory_opening"),
        locked_donor_outcomes_materialized=np.asarray(True),
        cohort_id=np.asarray("HEST"),
        cohort_release=np.asarray("synthetic-release"),
        feature_space_id=np.asarray("uni2h-synthetic"),
        feature_checkpoint_sha256=np.asarray("2" * 64),
        encoder_manifest_sha256=np.asarray("a" * 64),
        crop_manifest_sha256=np.asarray("b" * 64),
        molecular_space_id=np.asarray("log1p-cpm-qualified"),
        label_source_sha256=np.asarray("3" * 64),
        label_source_kind=np.asarray("independent_annotation_prediction_export"),
        registration_source_sha256=np.asarray("4" * 64),
        exclusion_policy_sha256=np.asarray("5" * 64),
        registration_method=np.asarray("native_xenium_cell_id_join"),
        encoder_name=np.asarray("MahmoodLab/UNI2-h"),
        crop_scale=np.asarray("registered_cell_local_context_112um"),
        assay=np.asarray("Xenium"),
        observation_level=np.asarray("cell"),
        target_construction=np.asarray("nucleus_overlapping_xenium_transcripts"),
        provenance_json=np.asarray(
            json.dumps(
                {
                    "crop_metadata": {"variants": crop_variants},
                    "label_target_independence": independence,
                    "label_receipt_sha256": independence["annotation_receipt_sha256"],
                    "label_source_sha256": "3" * 64,
                    "label_source_kind": "independent_annotation_prediction_export",
                }
            )
        ),
    )
    return development, locked


def _manifest(
    source_sha: str,
    measurement_sha: str,
    development: tuple[str, ...],
    locked: tuple[str, ...],
    *,
    locked_audit: dict[str, object] | None = None,
) -> SimpleNamespace:
    genes = ("g2", "g1")
    types = ("type_b", "type_a")
    content = {
        "prerequisites": {
            "measurement_report_sha256": measurement_sha,
            "measurement_study_manifest_sha256": "6" * 64,
            "measurement_source_sha256": source_sha,
        },
        "lock_protection": {
            "reserved_exclusively_for": "H-CELL",
            "reserved_donor_ids": list(locked),
            "prior_outcome_access_confirmed_false": True,
            "prior_outcome_access_status": "unopened",
            "prior_outcome_exposure_receipt_sha256": None,
            "prospective_lock_eligible": True,
        },
        "observations": {
            "level": "cell",
            "registration_method": "native_xenium_cell_id_join",
            "fine_type_field": "fine_type",
            "supported_fine_type_ids": list(types),
        },
        "encoder": {
            "manifest_sha256": "a" * 64,
            "feature_space_id": "uni2h-synthetic",
            "checkpoint_sha256": "2" * 64,
        },
        "crop_protocols": ["b" * 64],
        "reference_splits": {
            "primary_split_id": "primary",
            "split_ids": ["primary", "reference_hash_fold_0", "reference_hash_fold_1"],
        },
        "candidate_target_gene_panel_sha256": ordered_ids_sha256(["g1", "g2", "g3"]),
        "target_gene_panel_sha256": ordered_ids_sha256(genes),
        "type_marker_panel_sha256": ordered_ids_sha256(["marker"]),
        "label_target_independence": _independence(),
        "technical_covariates": [
            "log1p_library_size",
            "section_id",
            "disease_status",
            "site_id",
            "batch_id",
        ],
        "controls": list(PREPARE.REQUIRED_HEST_CONTROL_DECLARATIONS),
        "coverage_requirements": {
            "maximum_reference_evaluation_absolute_smd": 100.0,
            "maximum_reference_evaluation_categorical_total_variation": 1.0,
            "minimum_development_donors_per_fine_type": 2,
            "minimum_locked_donors_per_fine_type": 2,
            "minimum_reference_cells_per_donor_section_type": 2,
            "minimum_evaluation_cells_per_donor_section_type": 2,
            "minimum_positive_supported_fraction": 0.5,
        },
        "hyperparameter_grid": {"ranks": [1], "ridge_penalties": [0.25]},
        "randomization": {
            "seeds": [17],
            "permutations_per_seed": 100,
            "unit": "donor_x_fine_type_x_spatial_roi",
        },
        "primary_endpoint": {
            "name": "joint_donor_type_and_donor_section_type_macro_residual_coordinate_r2",
            "condition_on": "fine_type",
            "decision_rule": "both_endpoints_must_meet_frozen_minimum",
            "donor_type_macro": {
                "metric": "donor_equal_type_equal_residual_coordinate_r2",
                "minimum_effect": 0.05,
            },
            "donor_section_type_macro": {
                "metric": "donor_equal_section_equal_type_equal_residual_coordinate_r2",
                "minimum_effect": 0.05,
            },
        },
        "decision_thresholds": {
            "minimum_shuffled_delta_r2": 0.01,
            "maximum_empirical_p": 0.05,
        },
        "morphology_gate": {
            "experiment_role": "primary_hest_uni2h",
            "scientific_scope": "registered_cell_local_context_association",
            "final_inference": False,
            "minimum_final_permutations": 999,
            "minimum_coordinate_delta": 0.01,
            "minimum_stain_delta": 0.01,
            "minimum_null_shuffled_fraction": 0.5,
            "minimum_strata_coverage": 0.5,
            "minimum_expression_error_reduction": 0.01,
            "minimum_basis_ceiling_r2": 0.01,
            "maximum_direct_contrast_p": 0.05,
            "minimum_mask_implementation_pass_fraction": 1.0,
            "donor_bootstrap_iterations": 100,
            "donor_bootstrap_seed": 29,
            "prespecified_fixed_hyperparameters": True,
        },
        "locked_measurement_audit": (_locked_audit() if locked_audit is None else locked_audit),
        "analysis_plan_sha256": "f" * 64,
        "scientific_scope": "registered_cell_local_context_association",
        "opening": {
            "opening_receipt_sha256": OPENING_RECEIPT_SHA256,
            "permitted_claims": ["H-CELL", "H-INTRINSIC"],
        },
    }
    return SimpleNamespace(
        content=content,
        sha256="7" * 64,
        study_stage="confirmatory_morphology",
        status="opened",
        development_donors=development,
        locked_test_donors=locked,
        hypothesis_ids=("H-CELL", "H-INTRINSIC"),
    )


def _selection(
    source_sha: str,
    planned: list[str],
    *,
    locked_audit: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    genes = ["g2", "g1"]
    types = ["type_b", "type_a"]
    receipt = {
        "schema": "heir.measurement_target_selection.v1",
        "pass": True,
        "selection_partition": "development_only",
        "primary_target_variant": "nucleus_overlapping_transcripts",
        "ordered_reliable_gene_ids": genes,
        "ordered_reliable_gene_panel_sha256": ordered_ids_sha256(genes),
        "supported_fine_type_ids": types,
        "supported_fine_type_panel_sha256": ordered_ids_sha256(types),
        "locked_test_molecular_outcomes_used": False,
    }
    audit = _locked_audit() if locked_audit is None else locked_audit
    report = {
        "schema": "heir.measurement_gate.v1",
        "pass": True,
        "source_sha256": source_sha,
        "thresholds": {
            name: value
            for name, value in audit.items()
            if name
            not in {
                "audit_timing",
                "selection_changes_forbidden",
                "coverage_denominator",
                "minimum_locked_donor_type_reliability_fraction",
            }
        },
        "target_selection_receipt": receipt,
        "coverage": {
            "pass": True,
            "support": {value: {"supported": True} for value in planned},
        },
    }
    return report, receipt


def _preparation_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    omitted_donor_types: frozenset[tuple[str, str]] = frozenset(),
    locked_audit: dict[str, object] | None = None,
    locked_audit_sha256: str | None = None,
    source_independence: dict[str, object] | None = None,
    alternate_reference_deficit: tuple[str, str] | None = None,
    two_section_donor: str | None = None,
    unreliable_locked_stratum: tuple[str, str, str] | None = None,
    stale_locked_measurement_composite: bool = False,
) -> SimpleNamespace:
    source = tmp_path / "source.npz"
    development, locked = _source(
        source,
        omitted_donor_types=omitted_donor_types,
        locked_audit=locked_audit,
        locked_audit_sha256=locked_audit_sha256,
        independence_contract=source_independence,
        alternate_reference_deficit=alternate_reference_deficit,
        two_section_donor=two_section_donor,
        unreliable_locked_stratum=unreliable_locked_stratum,
        stale_locked_measurement_composite=stale_locked_measurement_composite,
    )
    measurement_path = tmp_path / "measurement.json"
    measurement_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "study.json"
    manifest_path.write_text("{}", encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "heir.morphology_ridge_preparation_plan.v1",
                "source_schema": "synthetic.registered.v1",
                "source_observations_sha256": sha256_file(source),
                "opening_receipt_sha256": OPENING_RECEIPT_SHA256,
                "development_donors": list(development),
                "locked_test_donors": list(locked),
                "label_target_independence": _independence(),
                "annotation_receipt_sha256": _independence()["annotation_receipt_sha256"],
                "annotation_prediction_export_sha256": "3" * 64,
                "label_source_kind": "independent_annotation_prediction_export",
            }
        ),
        encoding="utf-8",
    )
    measurement_source_sha = "8" * 64
    manifest = _manifest(
        measurement_source_sha,
        sha256_file(measurement_path),
        development,
        locked,
        locked_audit=locked_audit,
    )
    with np.load(source, allow_pickle=False) as archive:
        planned = archive["planned_stratum_ids"].astype(str).tolist()
    measurement, selection = _selection(
        measurement_source_sha,
        planned,
        locked_audit=locked_audit,
    )

    def load_opened_for_preparation(*args, **kwargs):
        assert kwargs["require_status"] == "opened"
        assert kwargs["verify_runtime"] is True
        assert kwargs["require_clean_runtime"] is True
        assert kwargs["verify_container_digest"] is True
        return manifest

    monkeypatch.setattr(PREPARE.StudyManifest, "load", load_opened_for_preparation)
    monkeypatch.setattr(
        PREPARE,
        "load_passing_measurement_receipt",
        lambda *args, **kwargs: measurement,
    )
    return SimpleNamespace(
        source=source,
        measurement_path=measurement_path,
        manifest_path=manifest_path,
        plan_path=plan_path,
        development_path=tmp_path / "development.npz",
        locked_path=tmp_path / "locked.npz",
        manifest=manifest,
        measurement=measurement,
        selection=selection,
    )


def _run_preparation(context: SimpleNamespace) -> int:
    return PREPARE.main(
        (
            "--study-manifest",
            str(context.manifest_path),
            "--measurement-report",
            str(context.measurement_path),
            "--plan",
            str(context.plan_path),
            "--source-observations",
            str(context.source),
            "--development-output",
            str(context.development_path),
            "--locked-test-output",
            str(context.locked_path),
        )
    )


def test_target_selection_rejects_locked_threshold_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _preparation_context(tmp_path, monkeypatch)
    context.measurement["thresholds"]["maximum_annotation_nucleus_p95_um"] = 7.5

    with pytest.raises(ValueError, match="differs from frozen H-MEAS thresholds"):
        PREPARE._target_selection(context.manifest, context.measurement_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seeds", [17.9]),
        ("seeds", [True]),
        ("permutations_per_seed", 333.9),
        ("permutations_per_seed", True),
    ),
)
def test_gate_settings_reject_nonintegral_randomization_contract(field: str, value: object) -> None:
    manifest = _manifest(
        "8" * 64,
        "9" * 64,
        ("D1", "D2", "D3"),
        ("L1", "L2"),
    )
    manifest.content["randomization"][field] = value

    with pytest.raises(ValueError, match="exact|non-number"):
        BENCHMARK._gate_settings(manifest)


def test_gate_settings_reject_joint_primary_endpoint_drift() -> None:
    manifest = _manifest(
        "8" * 64,
        "9" * 64,
        ("D1", "D2", "D3"),
        ("L1", "L2"),
    )
    manifest.content["primary_endpoint"]["donor_section_type_macro"]["minimum_effect"] = 0.049

    with pytest.raises(ValueError, match="differs from the frozen joint endpoint"):
        BENCHMARK._gate_settings(manifest)


def test_benchmark_rejects_calibration_manifest_projection_drift(
    tmp_path: Path,
    calibration_receipt,
) -> None:
    receipt_path = tmp_path / "calibration_receipt.json"
    receipt_path.write_text(json.dumps(calibration_receipt), encoding="utf-8")
    binding = calibration_receipt["exact_gate_settings"]["confirmatory_design_binding"]
    content = copy.deepcopy(binding["scientific_manifest_projection"])
    content["morphology_gate"]["calibration_receipt_sha256"] = sha256_file(receipt_path)
    manifest = SimpleNamespace(content=content)

    assert BENCHMARK._calibration_receipt(manifest, receipt_path) == calibration_receipt

    # Dataset revision is part of the scientific projection but is not one of
    # the benchmark's separately reconstructed gate parameters.
    manifest.content["dataset"]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="projection differs from the live H-CELL manifest"):
        BENCHMARK._calibration_receipt(manifest, receipt_path)


def test_preparation_rejects_locked_audit_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _preparation_context(
        tmp_path,
        monkeypatch,
        locked_audit_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="changed the frozen locked measurement audit"):
        _run_preparation(context)


def test_preparation_rejects_registration_quality_cutoff_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _preparation_context(tmp_path, monkeypatch)
    with np.load(context.source, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["registration_quality_cutoffs_json"] = np.asarray(
        json.dumps({"best": 0.20, "intermediate": 0.6, "near_threshold": 1.0})
    )
    np.savez_compressed(context.source, **payload)
    plan = json.loads(context.plan_path.read_text(encoding="utf-8"))
    plan["source_observations_sha256"] = sha256_file(context.source)
    context.plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="changed the frozen registration-quality definition"):
        _run_preparation(context)


def test_locked_audit_recomputes_and_checks_composite_qc_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _preparation_context(
        tmp_path,
        monkeypatch,
        stale_locked_measurement_composite=True,
    )

    assert _run_preparation(context) == 0
    locked = MorphologyRidgeDatasetArtifact.load_npz(context.locked_path, role="locked_test")
    audit = locked.coverage_audit["locked_measurement_audit"]

    assert audit["distribution_checks"]["crop_qc_matches_recomputed"] is True
    assert (
        audit["distribution_checks"]["locked_measurement_qc_matches_recomputed_conjunction"]
        is False
    )
    assert audit["summaries"]["source_locked_measurement_qc_false_positive_rows"] == 1
    assert audit["summaries"]["reliability_row_policy"].startswith("recomputed_")
    assert audit["pass"] is False


def test_preparation_rejects_label_target_source_contract_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_independence = _independence()
    source_independence["annotation_receipt_sha256"] = "0" * 64
    context = _preparation_context(
        tmp_path,
        monkeypatch,
        source_independence=source_independence,
    )

    with pytest.raises(ValueError, match="independence contract differs"):
        _run_preparation(context)


def test_preparation_rejects_underpowered_locked_alternate_reference_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _preparation_context(
        tmp_path,
        monkeypatch,
        alternate_reference_deficit=("L1", "type_a"),
    )

    with pytest.raises(ValueError, match="lacks the frozen independent reference support"):
        _run_preparation(context)


def test_reference_means_are_donor_section_type_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _preparation_context(tmp_path, monkeypatch, two_section_donor="D1")

    assert _run_preparation(context) == 0
    development = MorphologyRidgeDatasetArtifact.load_npz(
        context.development_path, role="development"
    )
    type_a = development.type_labels == development.type_names.index("type_a")
    first = type_a & (development.section_ids == "section_D1_a")
    second = type_a & (development.section_ids == "section_D1_b")

    assert np.count_nonzero(first) == 2
    assert np.count_nonzero(second) == 2
    np.testing.assert_allclose(development.reference_means[first], [[1.0, -1.0]] * 2)
    np.testing.assert_allclose(development.reference_means[second], [[99.0, 101.0]] * 2)


def test_preparation_carries_frozen_registration_quality_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _preparation_context(tmp_path, monkeypatch)

    assert _run_preparation(context) == 0
    locked = MorphologyRidgeDatasetArtifact.load_npz(context.locked_path, role="locked_test")

    assert locked.registration_quality_applicable is True
    assert set(locked.registration_quality_strata.tolist()) == {
        "best",
        "intermediate",
        "near_threshold",
    }
    assert locked.registration_quality_cutoffs == {
        "best": 0.25,
        "intermediate": 0.6,
        "near_threshold": 1.0,
    }
    assert locked.registration_quality_definition == PREPARE.REGISTRATION_QUALITY_DEFINITION

    with pytest.raises(ValueError, match="labels differ from scores"):
        replace(
            locked,
            registration_quality_strata=np.repeat("best", len(locked.observation_ids)),
        ).validate()


def test_missing_locked_donor_type_stays_in_frozen_population_and_fails_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _locked_audit()
    audit["minimum_locked_donor_type_reliability_fraction"] = 1.0
    context = _preparation_context(
        tmp_path,
        monkeypatch,
        omitted_donor_types=frozenset({("L2", "type_b")}),
        locked_audit=audit,
    )

    assert _run_preparation(context) == 0
    locked = MorphologyRidgeDatasetArtifact.load_npz(context.locked_path, role="locked_test")
    locked_audit = locked.coverage_audit["locked_measurement_audit"]

    assert locked.type_names == ("type_b", "type_a")
    assert "L2|section_L2|type_b" in locked.planned_stratum_ids
    assert locked.coverage_audit["planned_strata"] == 4
    assert locked.coverage_audit["retained_evaluation_strata"] == 2
    support_audit = locked.coverage_audit["locked_support_audit"]
    assert support_audit["L1|section_L1|type_b"]["evaluable"] is False
    assert support_audit["L2|section_L2|type_b"]["evaluable"] is False
    assert support_audit["L1|section_L1|type_b"]["locked_donors_with_primary_support"] == 1
    assert support_audit["L1|section_L1|type_b"]["fine_type_support_pass"] is False
    assert sum(row["evaluable"] is True for row in support_audit.values()) == 2
    assert locked_audit["planned_donor_type_count"] == 4
    assert set(locked_audit["donor_type_reliability"]) == {
        "L1|type_a",
        "L1|type_b",
        "L2|type_a",
        "L2|type_b",
    }
    missing = locked_audit["donor_type_reliability"]["L2|type_b"]
    assert missing["rows"] == 0
    assert missing["evaluable"] is False
    assert missing["passes_frozen_reliability"] is False
    assert locked_audit["reliable_donor_type_fraction"] == pytest.approx(0.75)
    missing_stratum = locked_audit["donor_section_type_reliability"]["L2|section_L2|type_b"]
    assert missing_stratum["rows"] == 0
    assert missing_stratum["evaluable"] is False
    assert missing_stratum["passes_frozen_reliability"] is False
    planned_reliability = locked_audit["planned_stratum_reliability"]
    assert planned_reliability == {
        "coverage_denominator": "all_frozen_locked_donor_section_type_strata",
        "planned_count": 4,
        "evaluable_count": 3,
        "reliable_count": 3,
        "reliable_fraction": pytest.approx(0.75),
        "minimum_required_reliable_fraction": 1.0,
        "pass": False,
    }
    assert locked_audit["worst_section_reliability_summary"] == {
        "planned_stratum_id": "L2|section_L2|type_b",
        "donor_id": "L2",
        "section_id": "section_L2",
        "fine_type_id": "type_b",
        "rows": 0,
        "median_spearman_brown_reliability": None,
        "evaluable": False,
        "passes_frozen_reliability": False,
    }
    assert locked_audit["pass"] is False


def test_locked_reliability_gate_resolves_sections_and_catches_pooled_masking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _locked_audit()
    audit["minimum_locked_donor_type_reliability_fraction"] = 1.0
    context = _preparation_context(
        tmp_path,
        monkeypatch,
        locked_audit=audit,
        two_section_donor="L1",
        unreliable_locked_stratum=("L1", "section_L1_b", "type_a"),
    )

    assert _run_preparation(context) == 0
    locked = MorphologyRidgeDatasetArtifact.load_npz(context.locked_path, role="locked_test")
    locked_audit = locked.coverage_audit["locked_measurement_audit"]

    assert locked_audit["donor_type_reliability"]["L1|type_a"]["passes_frozen_reliability"] is True
    section_reports = locked_audit["donor_section_type_reliability"]
    assert section_reports["L1|section_L1_a|type_a"]["passes_frozen_reliability"] is True
    assert section_reports["L1|section_L1_b|type_a"]["passes_frozen_reliability"] is False
    assert section_reports["L1|section_L1_b|type_a"]["evaluable"] is True
    assert locked_audit["reliable_donor_type_fraction"] == pytest.approx(1.0)
    assert locked_audit["planned_stratum_reliability"]["planned_count"] == 6
    assert locked_audit["planned_stratum_reliability"]["reliable_count"] == 5
    assert locked_audit["planned_stratum_reliability"]["reliable_fraction"] == pytest.approx(
        5.0 / 6.0
    )
    assert locked_audit["worst_section_reliability_by_donor_type"]["L1|type_a"] == {
        "planned_section_count": 2,
        "evaluable_section_count": 2,
        "reliable_section_count": 1,
        "worst_planned_stratum_id": "L1|section_L1_b|type_a",
        "worst_median_spearman_brown_reliability": 0.0,
        "worst_section_evaluable": True,
        "all_planned_sections_pass_frozen_reliability": False,
    }
    assert (
        locked_audit["distribution_checks"]["planned_donor_section_type_reliability_fraction"]
        is False
    )
    assert locked_audit["pass"] is False


def test_preparation_and_benchmark_bind_effective_experiment_end_to_end(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.npz"
    development_donors, locked_donors = _source(source)
    measurement_path = tmp_path / "measurement.json"
    measurement_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "study.json"
    manifest_path.write_text("{}", encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "heir.morphology_ridge_preparation_plan.v1",
                "source_schema": "synthetic.registered.v1",
                "source_observations_sha256": sha256_file(source),
                "opening_receipt_sha256": OPENING_RECEIPT_SHA256,
                "development_donors": list(development_donors),
                "locked_test_donors": list(locked_donors),
                "label_target_independence": _independence(),
                "annotation_receipt_sha256": _independence()["annotation_receipt_sha256"],
                "annotation_prediction_export_sha256": "3" * 64,
                "label_source_kind": "independent_annotation_prediction_export",
            }
        ),
        encoding="utf-8",
    )
    measurement_source_sha = "8" * 64
    manifest = _manifest(
        measurement_source_sha,
        sha256_file(measurement_path),
        development_donors,
        locked_donors,
    )
    with np.load(source, allow_pickle=False) as archive:
        planned = archive["planned_stratum_ids"].astype(str).tolist()
    measurement, selection = _selection(measurement_source_sha, planned)

    monkeypatch.setattr(PREPARE.StudyManifest, "load", lambda *args, **kwargs: manifest)

    def fake_measurement_loader(path, **kwargs):
        assert Path(path) == measurement_path
        assert kwargs["expected_receipt_sha256"] == sha256_file(measurement_path)
        assert kwargs["expected_source_sha256"] == measurement_source_sha
        return measurement

    monkeypatch.setattr(PREPARE, "load_passing_measurement_receipt", fake_measurement_loader)
    development_path = tmp_path / "development.npz"
    locked_path = tmp_path / "locked.npz"
    assert (
        PREPARE.main(
            (
                "--study-manifest",
                str(manifest_path),
                "--measurement-report",
                str(measurement_path),
                "--plan",
                str(plan_path),
                "--source-observations",
                str(source),
                "--development-output",
                str(development_path),
                "--locked-test-output",
                str(locked_path),
            )
        )
        == 0
    )
    development = MorphologyRidgeDatasetArtifact.load_npz(development_path, role="development")
    locked = MorphologyRidgeDatasetArtifact.load_npz(locked_path, role="locked_test")
    assert development.gene_ids == ("g2", "g1")
    assert development.type_names == ("type_b", "type_a")
    assert development.image_feature_tensor.shape[1] == 18
    assert development.crop_comparison_families[0] == "g2_primary"
    assert development.section_ids.shape == development.observation_ids.shape
    assert development.nuclear_morphometrics.shape[1] == 2
    assert development.cellvit_context_features.shape[1] == 1
    assert development.local_density_features.shape[1] == 1
    assert development.boundary_features.shape[1] == 1
    assert development.reference_evaluation_balance["primary"]["pass"] is True
    assert locked.evidence_scope == "internal_locked_hest"

    def load_opened_for_benchmark(*args, **kwargs):
        assert kwargs["require_status"] == "opened"
        assert kwargs["verify_runtime"] is True
        assert kwargs["require_clean_runtime"] is True
        assert kwargs["verify_container_digest"] is True
        return manifest

    monkeypatch.setattr(BENCHMARK.StudyManifest, "load", load_opened_for_benchmark)
    monkeypatch.setattr(
        BENCHMARK,
        "load_passing_measurement_receipt",
        lambda *args, **kwargs: measurement,
    )
    captured: dict[str, object] = {}

    def fake_gate(*args, **kwargs):
        captured.update(kwargs)
        return {"component_pass": True, "schema_version": "synthetic"}

    monkeypatch.setattr(BENCHMARK, "evaluate_morphology_ridge_gate", fake_gate)
    report_path = tmp_path / "report.json"
    assert (
        BENCHMARK.main(
            (
                "--study-manifest",
                str(manifest_path),
                "--measurement-report",
                str(measurement_path),
                "--development-data",
                str(development_path),
                "--locked-test-data",
                str(locked_path),
                "--report-output",
                str(report_path),
                "--device",
                "cpu",
            )
        )
        == 0
    )
    assert captured["ranks"] == (1,)
    assert captured["alphas"] == (0.25,)
    assert captured["permutation_seeds"] == (17,)
    assert captured["minimum_support"] == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scientific_settings_source"] == "opened_study_manifest_only"
    assert report["measurement_gate_pass"] is True


def test_hescape_reserved_outcome_declaration_fails_closed(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "hescape.npz"
    np.savez_compressed(
        source,
        schema_version=np.asarray("synthetic.hescape.v1"),
        cohort_id=np.asarray("HESCAPE"),
        donor_ids=np.asarray(["THD0008"]),
        reserved_hest_locked_donors=np.asarray(
            ["THD0008", "THD0011", "TILD117", "VUILD78", "VUILD96"]
        ),
        reserved_donor_outcomes_loaded=np.asarray(False),
        analysis_scope=np.asarray(
            "development_donors_only_reserved_outcomes_previously_materialized"
        ),
    )
    manifest = SimpleNamespace(
        development_donors=("VUILD91",),
        locked_test_donors=("THD0008", "THD0011", "TILD117", "VUILD78", "VUILD96"),
        content={
            "lock_protection": {
                "reserved_donor_ids": [
                    "THD0008",
                    "THD0011",
                    "TILD117",
                    "VUILD78",
                    "VUILD96",
                ],
                "hescape_analysis_scope": (
                    "development_donors_only_reserved_outcomes_previously_materialized"
                ),
                "hescape_allowed_donor_ids": ["VUILD91"],
            }
        },
    )
    with np.load(source, allow_pickle=False) as archive:
        try:
            PREPARE._lock_protection(
                manifest,
                "HESCAPE",
                archive["donor_ids"],
                archive,
            )
        except ValueError as error:
            assert "reserved HEST locked outcomes" in str(error)
        else:
            raise AssertionError("reserved HEST donor was accepted in HESCAPE")
