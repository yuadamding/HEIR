from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from heir.data import MorphologyRidgeDatasetArtifact, ordered_ids_sha256
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


def _source(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
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
        section = "section_%s" % donor
        for type_index, fine_type in enumerate(("type_a", "type_b")):
            for pool_index, pool in enumerate(("reference", "evaluation")):
                for row in range(4):
                    observation_ids.append("%s-%s-%s-%d" % (donor, fine_type, pool, row))
                    donor_ids.append(donor)
                    section_ids.append(section)
                    block_ids.append("%s/%s/block_%s" % (donor, section, pool))
                    roi_ids.append("%s/%s/roi_%s_%d" % (donor, section, pool, type_index))
                    pool_roles.append(pool)
                    fine_types.append(fine_type)
                    labels.append(type_index)
                    state = float(row - 1.5 + type_index * 0.2)
                    targets.append([donor_index + state, 2.0 * type_index - state, 0.5 * state])
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
    planned = sorted({"%s|%s|%s" % value for value in zip(donor_ids, section_ids, fine_types)})
    crop_variants = [
        {"crop_id": crop_id, "role": role, "comparison_family": family}
        for crop_id, (role, family) in PREPARE.HEST_CROP_CONTRACT.items()
    ]
    primary_roles = np.asarray(pool_roles)
    alternate_zero = primary_roles.copy()
    alternate_one = primary_roles.copy()
    reference = primary_roles == "reference"
    identities = np.asarray(observation_ids)
    alternate_zero[reference & np.char.endswith(identities, "-0")] = "excluded"
    alternate_one[reference & np.char.endswith(identities, "-1")] = "excluded"
    roles_by_split = np.column_stack((primary_roles, alternate_zero, alternate_one))
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
        registration_qc_pass=np.ones(rows, dtype=np.bool_),
        target_qc_pass=np.ones(rows, dtype=np.bool_),
        crop_qc_pass=np.ones(rows, dtype=np.bool_),
        registration_cardinality=np.ones(rows, dtype=np.int64),
        fine_type_marker_gene_ids=np.asarray(["marker"]),
        fine_type_marker_panel_sha256=np.asarray(ordered_ids_sha256(["marker"])),
        study_stage=np.asarray("confirmatory_morphology"),
        study_manifest_sha256=np.asarray("7" * 64),
        source_scope=np.asarray("development_and_locked_after_confirmatory_lock"),
        locked_donor_outcomes_materialized=np.asarray(True),
        cohort_id=np.asarray("HEST"),
        cohort_release=np.asarray("synthetic-release"),
        feature_space_id=np.asarray("uni2h-synthetic"),
        feature_checkpoint_sha256=np.asarray("2" * 64),
        encoder_manifest_sha256=np.asarray("a" * 64),
        crop_manifest_sha256=np.asarray("b" * 64),
        molecular_space_id=np.asarray("log1p-cpm-qualified"),
        label_source_sha256=np.asarray("3" * 64),
        registration_source_sha256=np.asarray("4" * 64),
        exclusion_policy_sha256=np.asarray("5" * 64),
        registration_method=np.asarray("native_xenium_cell_id_join"),
        encoder_name=np.asarray("MahmoodLab/UNI2-h"),
        crop_scale=np.asarray("registered_cell_local_context_112um"),
        assay=np.asarray("Xenium"),
        observation_level=np.asarray("cell"),
        target_construction=np.asarray("nucleus_overlapping_xenium_transcripts"),
        provenance_json=np.asarray(json.dumps({"crop_metadata": {"variants": crop_variants}})),
    )
    return development, locked


def _manifest(
    source_sha: str,
    measurement_sha: str,
    development: tuple[str, ...],
    locked: tuple[str, ...],
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
        },
        "observations": {
            "level": "cell",
            "registration_method": "native_xenium_cell_id_join",
            "fine_type_field": "final_CT",
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
        "label_target_independence": {
            "marker_panel_sha256": ordered_ids_sha256(["marker"]),
            "establishes_full_target_independence": True,
        },
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
            "minimum_evaluation_cells_per_donor_type": 2,
            "minimum_positive_supported_fraction": 0.5,
        },
        "hyperparameter_grid": {"ranks": [1], "ridge_penalties": [0.25]},
        "randomization": {
            "seeds": [17],
            "permutations_per_seed": 100,
            "unit": "donor_x_fine_type_x_spatial_roi",
        },
        "primary_endpoint": {"minimum_effect": 0.01},
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
        "scientific_scope": "registered_cell_local_context_association",
    }
    return SimpleNamespace(
        content=content,
        sha256="7" * 64,
        study_stage="confirmatory_morphology",
        development_donors=development,
        locked_test_donors=locked,
        hypothesis_ids=("H-CELL", "H-INTRINSIC"),
    )


def _selection(source_sha: str, planned: list[str]) -> tuple[dict[str, object], dict[str, object]]:
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
    report = {
        "schema": "heir.measurement_gate.v1",
        "pass": True,
        "source_sha256": source_sha,
        "target_selection_receipt": receipt,
        "coverage": {
            "pass": True,
            "support": {value: {"supported": True} for value in planned},
        },
    }
    return report, receipt


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
                "development_donors": list(development_donors),
                "locked_test_donors": list(locked_donors),
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

    monkeypatch.setattr(BENCHMARK.StudyManifest, "load", lambda *args, **kwargs: manifest)
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
    assert report["scientific_settings_source"] == "locked_study_manifest_only"
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
        analysis_scope=np.asarray("development_donors_only_hest_lock_unopened"),
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
                "hescape_analysis_scope": "development_donors_only_hest_lock_unopened",
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
