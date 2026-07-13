import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from heir.data import HistologyBag
from heir.uncertainty import MahalanobisOOD

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_snpatho_refinement_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_snpatho_refinement_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)

CALIBRATION_SCRIPT = ROOT / "scripts" / "calibrate_target_ood.py"
CALIBRATION_SPEC = importlib.util.spec_from_file_location(
    "calibrate_target_ood_for_preparation_test",
    CALIBRATION_SCRIPT,
)
assert CALIBRATION_SPEC is not None and CALIBRATION_SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(CALIBRATION_SPEC)
sys.modules[CALIBRATION_SPEC.name] = CALIBRATION
CALIBRATION_SPEC.loader.exec_module(CALIBRATION)


def _samples(tmp_path):
    return {
        sample: PREPARE.SamplePaths(
            sample=sample,
            source=tmp_path / "source" / sample,
            scanvi=tmp_path / "scanvi" / sample,
            scanvi_input=tmp_path / "ffpe" / sample,
        )
        for sample in PREPARE.SAMPLES
    }


def test_command_plan_freezes_rare_types_geometry_and_batch_identity(tmp_path):
    stages = PREPARE.build_stages(samples=_samples(tmp_path), heir_command="heir")

    assert [(stage.sample, stage.name) for stage in stages] == [
        (sample, name)
        for sample in PREPARE.SAMPLES
        for name in (
            "prototypes",
            "residual_geometry_v2",
            "batch_train",
            "batch_validation",
        )
    ]
    prototype = stages[0].command(stages[0].output)
    geometry = stages[1].command(stages[1].output)
    train = stages[2].command(stages[2].output)
    assert "--include-rare-types" in prototype
    assert prototype[prototype.index("--seed") + 1] == "17"
    assert geometry[geometry.index("--rank") + 1] == "4"
    assert train[train.index("--analysis-role") + 1] == "development_retrospective"
    assert train[train.index("--block-id") + 1] == "4066_FFPE"
    assert train[train.index("--ood-artifact") + 1].endswith("4066/ood_target_calibrated.npz")


def _target_ood_v2_fixture(tmp_path):
    paths = PREPARE.SamplePaths(
        sample="4066",
        source=tmp_path / "source" / "4066",
        scanvi=tmp_path / "scanvi" / "4066",
        scanvi_input=tmp_path / "ffpe" / "4066",
    )
    rng = np.random.default_rng(71)
    development = rng.normal(size=(40, 3)).astype(np.float32)
    base = MahalanobisOOD().fit(
        development,
        analysis_role="development",
        quantile=0.9,
        training_donors=("B1",),
        feature_space_id="omiclip-test-v1",
    )
    base.source_sha256 = ("a" * 64,)
    base_path = tmp_path / "development_ood.npz"
    base.to_npz(base_path)

    target = rng.normal(loc=3.0, size=(23, 3)).astype(np.float32)
    full = HistologyBag(
        slide_id="4066",
        nucleus_ids=np.asarray(["n%d" % index for index in range(len(target))]),
        features=target,
        coordinates_um=np.column_stack((np.arange(len(target)), np.zeros(len(target)))),
        sample_id="4066",
        donor_id="4066",
        block_id="4066_FFPE",
        feature_space_id="omiclip-test-v1",
    )
    paths.source.mkdir(parents=True, exist_ok=True)
    full.save_npz(paths.histology("full"))
    CALIBRATION.calibrate(
        base_ood_path=base_path,
        histology_path=paths.histology("full"),
        sample_id="4066",
        quantile=0.95,
        output_path=paths.ood,
        provenance_path=paths.ood_provenance,
        score_batch_size=7,
    )
    return paths, full, base_path


def test_preparation_accepts_v2_target_ood_with_frozen_development_threshold(tmp_path):
    paths, full, base_path = _target_ood_v2_fixture(tmp_path)

    validated_base = PREPARE._validate_target_ood_calibration(
        paths=paths,
        sample="4066",
        full=full,
    )

    provenance = json.loads(paths.ood_provenance.read_text(encoding="utf-8"))
    assert validated_base == base_path.resolve()
    assert provenance["schema"] == "heir.target_histology_ood_calibration.v2"
    assert provenance["threshold_source"] == "development_detector"
    assert provenance["threshold"] == MahalanobisOOD.from_npz(base_path).threshold
    assert provenance["descriptive_target_quantile"] == 0.95


def test_preparation_rejects_target_quantile_substituted_for_development_threshold(tmp_path):
    paths, full, _ = _target_ood_v2_fixture(tmp_path)
    provenance = json.loads(paths.ood_provenance.read_text(encoding="utf-8"))
    provenance["threshold"] = provenance["descriptive_target_quantile_value"]
    paths.ood_provenance.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its development source"):
        PREPARE._validate_target_ood_calibration(
            paths=paths,
            sample="4066",
            full=full,
        )


def _true_loo_scope_provenance(held_out="4411"):
    training = [sample for sample in PREPARE.SAMPLES if sample != held_out]
    label_ontology = ["B", "T"]
    return {
        "status": "native_scanvi_true_leave_one_donor_out",
        "analysis_role": "leave_one_donor_out_molecular_audit",
        "training_partition": {
            "mode": "leave_one_donor_out",
            "held_out_sample": held_out,
            "backbone_training_donors": training,
            "decoder_training_donors": training,
            "all_donor_behavior_role": "not_applicable",
            "label_ontology": label_ontology,
            "label_ontology_sha256": PREPARE._ordered_string_sha256(label_ontology),
            "held_out_mapping": {
                "method": "SCANVI.load_query_data_without_query_training",
                "labels_available_to_query_model": False,
                "query_train_called": False,
                "query_parameters_frozen_before_inference": True,
                "inference_guard_enabled_without_optimization": True,
                "held_out_expression_used_for_fitting": False,
                "held_out_annotation_used_for_label_mapping": False,
                "label_mapping_method": "frozen_training_donor_SCANVI_classifier",
                "label_training_donors": training,
                "runtime_audit": {
                    "labels_removed_before_registry_transfer": True,
                    "query_train_called": False,
                    "parameters_frozen_before_inference": True,
                    "inference_guard_enabled_without_optimization": True,
                    "label_predictions_generated_without_target_annotation": True,
                    "label_prediction_rule": "SCANVI.predict(soft=False)",
                    "frozen_parameter_count": 100,
                    "cells_mapped": 10,
                    "predicted_label_sha256": "a" * 64,
                    "predicted_label_counts": {"T": 10},
                },
            },
            "fit_cell_counts": {sample: 10 for sample in training},
        },
    }


def test_true_loo_scope_prepares_only_target_and_excludes_it_from_fit():
    scope = PREPARE._r2_training_scope(_true_loo_scope_provenance("4411"))

    assert scope.true_leave_one_donor_out is True
    assert scope.held_out_sample == "4411"
    assert scope.active_samples == ("4411",)
    assert scope.training_donors == ("4066", "4399")
    assert scope.held_out_sample not in scope.training_donors


def test_true_loo_scope_rejects_target_in_backbone_or_decoder_fit():
    provenance = _true_loo_scope_provenance("4411")
    provenance["training_partition"]["decoder_training_donors"] = list(PREPARE.SAMPLES)
    with pytest.raises(ValueError, match="training donors are invalid"):
        PREPARE._r2_training_scope(provenance)


def test_true_loo_runtime_counts_are_cross_checked_against_loaded_reference():
    provenance = _true_loo_scope_provenance("4411")
    scope = PREPARE._r2_training_scope(provenance)
    PREPARE._validate_r2_count_binding(
        provenance=provenance,
        scope=scope,
        sample="4066",
        observed_cells=10,
        declared_latent={"cells": 10},
    )
    PREPARE._validate_r2_count_binding(
        provenance=provenance,
        scope=scope,
        sample="4411",
        observed_cells=10,
        declared_latent={"cells": 10},
    )

    provenance["training_partition"]["fit_cell_counts"]["4066"] = 9
    with pytest.raises(ValueError, match="fit-cell count differs"):
        PREPARE._validate_r2_count_binding(
            provenance=provenance,
            scope=scope,
            sample="4066",
            observed_cells=10,
            declared_latent={"cells": 10},
        )
    with pytest.raises(ValueError, match="latent cell count differs"):
        PREPARE._validate_r2_count_binding(
            provenance=provenance,
            scope=scope,
            sample="4411",
            observed_cells=10,
            declared_latent={"cells": 11},
        )


def test_stage_plan_can_be_target_specific_for_true_loo(tmp_path):
    samples = _samples(tmp_path)
    stages = PREPARE.build_stages(samples={"4411": samples["4411"]}, heir_command="heir")

    assert len(stages) == 4
    assert {stage.sample for stage in stages} == {"4411"}


def test_dry_run_refuses_untracked_output_without_explicit_adoption(tmp_path):
    source = tmp_path / "input.npz"
    output = tmp_path / "output.npz"
    source.write_bytes(b"input")
    output.write_bytes(b"untracked")
    stage = PREPARE.Stage(
        sample="4066",
        name="synthetic",
        inputs=(("source", source),),
        output=output,
        command=lambda destination: ("never-executed", str(destination)),
        validate=lambda _: None,
    )

    with pytest.raises(RuntimeError, match="untracked stage output"):
        PREPARE._run_stage(
            stage,
            repository=tmp_path,
            receipt_root=tmp_path / "receipts",
            execute=False,
            adopt_existing=False,
        )

    assert not (tmp_path / "receipts").exists()


def test_r2_decoder_metadata_is_bound_to_design_and_targets(tmp_path):
    path = tmp_path / "decoder.pt"
    metadata = {
        "schema": "heir.scvi_distilled_decoder.v3",
        "gene_names": ["G1"],
        "latent_space_id": "sha256:" + "a" * 64,
        "batch_correction_mode": "none",
        "transform_batch": [],
        "training_donors": ["4066", "4399", "4411"],
        "decoder_only": True,
        "posterior_samples": 32,
        "distillation_latent_sha256": "b" * 64,
        "distillation_target_sha256": "c" * 64,
        "validation_mask_sha256": "d" * 64,
    }
    torch.save({"metadata": metadata}, path)
    observed = PREPARE._decoder_metadata(path, "test decoder")
    PREPARE._validate_decoder_metadata(
        observed,
        label="test decoder",
        genes=["G1"],
        latent_space_id="sha256:" + "a" * 64,
        correction_mode="none",
        transform_batch=[],
        training_donors=PREPARE.SAMPLES,
    )

    bad = dict(metadata)
    bad["batch_correction_mode"] = "reference_batch_marginalization"
    with pytest.raises(ValueError, match="batch_correction_mode differs"):
        PREPARE._validate_decoder_metadata(
            bad,
            label="test decoder",
            genes=["G1"],
            latent_space_id="sha256:" + "a" * 64,
            correction_mode="none",
            transform_batch=[],
            training_donors=PREPARE.SAMPLES,
        )


def test_native_r2_manifest_binds_scorer_inputs_and_v2_geometry(tmp_path):
    repository = tmp_path / "repository"
    samples = _samples(repository)
    latent_outputs = {}
    for sample, paths in samples.items():
        for path, value in (
            (paths.reference, b"reference"),
            (paths.prototypes, b"prototypes"),
            (paths.geometry, b"geometry-v2"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value + sample.encode())
        latent_outputs[sample] = {"path": str(paths.reference), "cells": 10, "sha256": "a" * 64}
    gene_panel = repository / "manifests" / "genes.tsv"
    gene_panel.parent.mkdir(parents=True)
    gene_panel.write_text("G1\n", encoding="utf-8")
    native_model = tmp_path / "assets" / "model"
    decoder = tmp_path / "assets" / "decoder.pt"
    provenance = {
        "schema": "heir.snpatho_scanvi_r2.v1",
        "status": "native_scanvi_with_specimen_biology_preserved",
        "workflow_filter": "processing_method == FFPE_snPATHO",
        "annotation_provenance": "independent audit pending",
        "latent_space_id": "sha256:" + "b" * 64,
        "native_model": str(native_model),
        "native_model_sha256": "b" * 64,
        "decoder": str(decoder),
        "decoder_sha256": "c" * 64,
        "decoder_contract": {"schema": "heir.scvi_distilled_decoder.v3"},
        "decoder_validation": {"policy": "donor_rotated_audit_plus_stratified_deployment_split"},
        "molecular_design": {"name": "no_specimen_correction"},
        "latent_outputs": latent_outputs,
        "scvi_tools_version": "test",
        "scvi_epochs": 1,
        "scanvi_epochs": 1,
        "latent_dim": 32,
        "cuda": True,
        "decoder_epochs": 1,
        "decoder_posterior_samples": 32,
    }

    manifest = PREPARE._native_r2_manifest(
        repository=repository,
        provenance=provenance,
        samples=samples,
        gene_panel=gene_panel,
    )

    assert manifest["schema"] == "heir.snpatho_scanvi_r2_manifest.v1"
    assert manifest["molecular_generation"] == "r2"
    assert manifest["expression_space_id"] == "log1p-cpm-10000-v1"
    assert manifest["distilled_decoder"]["sha256"] == "c" * 64
    assert set(manifest["specimens"]) == set(PREPARE.SAMPLES)
    assert all(
        row["residual_geometry"].endswith("residual_geometry_rare_complete_v2.npz")
        and len(row["residual_geometry_sha256"]) == 64
        for row in manifest["specimens"].values()
    )


def test_upstream_validation_rejects_cli_generation_mismatch_first(tmp_path):
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps({"schema": "heir.snpatho_scanvi_r1.v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="molecular-generation requested r2"):
        PREPARE._validate_upstream(
            repository=tmp_path,
            source_root=tmp_path,
            scanvi_root=tmp_path,
            scanvi_input_root=tmp_path,
            provenance_path=provenance,
            expected_molecular_generation="r2",
        )
