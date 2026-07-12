import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_snpatho_refinement_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_snpatho_refinement_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


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
