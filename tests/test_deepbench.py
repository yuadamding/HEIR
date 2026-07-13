"""Behavioral tests for the retrospective snPATHO-DeepBench evaluator."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
import yaml

from heir.data import PrototypeSet, RNAReference
from heir.evaluation.deepbench import (
    EQUAL_CELL_HARD_TYPE_MEAN_METHOD,
    EQUAL_CELL_SOFT_TYPE_MEAN_METHOD,
    HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD,
    OPTIONAL_ARTIFACTS,
    PRIMARY_GATE_SUPPORT_CONTRACTS,
    PRIMARY_GATE_SUPPORT_SCHEMA,
    PRIMARY_METHOD,
    R1_HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD,
    R1_SOFT_TYPE_MEAN_METHOD,
    REFINED_R1_METHOD,
    REFINEMENT_MATRIX_CONTROLS,
    REQUESTED_PRIMARY_CONTRAST,
    SHUFFLE_METHOD,
    SOFT_TYPE_MEAN_METHOD,
    TYPE_MEAN_METHOD,
    DeepBenchPlan,
    FiveSeedPredictionManifest,
    NativeResidualGeometry,
    NativeScanviManifest,
    RefinementMatrixSummary,
    _baseline_estimands,
    _bootstrap_macro_delta,
    _cell_rna_mass,
    _directory_sha256,
    _full_primary_evidence_gates,
    _hard_assigned_cell_rna_mass,
    _load_five_seed_prediction_manifest,
    _load_native_scanvi_manifest,
    _load_refined_prediction_manifest,
    _load_refinement_matrix_summary,
    _matched_r1_baseline_cell_values,
    _method_macro_summaries,
    _primary_diagnostic,
    _primary_gate_support_status,
    _prototype_type_support,
    _readiness,
    _record_shuffle_seed,
    _reference_linear_profiles,
    _reference_prototype_type_support,
    _reference_type_support,
    _repeated_final_record_shuffle_null,
    _requested_refined_primary_contrasts,
    _soft_type_mean_cells,
    _top_indices,
    _type_map_diagnostics,
    _type_mean_cells,
    _validate_r1_reference_identity,
    aggregate_cells_to_spots,
    deepbench_expression_metrics,
    validate_deepbench_specification,
    write_deepbench_report,
)
from heir.inference import PredictionBundle
from heir.prior.residual_geometry import RNAResidualGeometry
from heir.utils import sha256_file


def _specification() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "snpatho_deepbench_v1.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _linear_reference() -> RNAReference:
    return RNAReference(
        sample_id="reference",
        cell_ids=np.asarray(["a1", "a2", "b1"]),
        gene_ids=np.asarray(["g1", "g2"]),
        counts=np.asarray([[10.0, 0.0], [0.0, 20.0], [3.0, 7.0]]),
        library_sizes=np.asarray([100.0, 200.0, 1_000.0]),
        cell_type_labels=np.asarray(["A", "A", "B"]),
    )


def test_refined_prediction_manifest_binds_full_provenance_and_rejects_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section_id = "4066"
    latent_space_id = "sha256:" + "a" * 64
    expression_space_id = "log1p-cpm-10000-v1"
    native_manifest_path = tmp_path / "native.json"
    native_manifest_path.write_text("{}\n", encoding="utf-8")
    native_prototype = tmp_path / "native-prototype.npz"
    native_prototype.write_bytes(b"native")
    residual_geometry = tmp_path / "residual-geometry.npz"
    residual_geometry.write_bytes(b"geometry")
    geometry_identity = NativeResidualGeometry(
        path=residual_geometry.resolve(),
        sha256=sha256_file(residual_geometry),
        type_names=("A",),
        rank=1,
        latent_dim=2,
        bounds=(0.5,),
        source_reference_sha256="c" * 64,
        latent_transform_sha256="",
        basis=np.asarray([[[1.0], [0.0]]], dtype=np.float32),
    )
    native = NativeScanviManifest(
        path=native_manifest_path.resolve(),
        sha256=sha256_file(native_manifest_path),
        latent_space_id=latent_space_id,
        expression_space_id=expression_space_id,
        native_model_sha256="a" * 64,
        decoder_sha256="b" * 64,
        annotation_status="published_integrated_annotation_sensitivity_not_clean_reannotation",
        specimen_prototype_sha256={section_id: sha256_file(native_prototype)},
        training_donors=(section_id, "4399"),
        specimen_residual_geometry={section_id: geometry_identity},
    )
    refined_prototype = tmp_path / "refined-prototype.npz"
    refined_prototype.write_bytes(b"refined-prototype")
    checkpoint = tmp_path / "heir.pt"
    batch = {"source_sha256": [sha256_file(native_prototype)]}
    torch.save(
        {
            "schema": "heir.model.v4",
            "config": {
                "num_cell_types": 1,
                "latent_dim": 2,
                "residual_rank": 1,
            },
            "state_dict": {
                "residual_type_basis": torch.as_tensor(geometry_identity.basis),
            },
            "residual_geometry": {
                "type_max_norms": torch.as_tensor(geometry_identity.bounds),
                "basis_trainable": False,
            },
            "metadata": {
                "schema": "heir.refined_model.v1",
                "type_names": ["A"],
                "seed": 17,
                "refinement_round": 4,
                "latent_space_id": latent_space_id,
                "expression_space_id": expression_space_id,
                "rna_vae_sha256": native.decoder_sha256,
                "residual_geometry": str(residual_geometry.resolve()),
                "residual_geometry_sha256": sha256_file(residual_geometry),
                "residual_basis_trainable": False,
                "training_donors": [section_id, "4399"],
                "direct_training_donors": [section_id],
                "validation_donors": [section_id],
                "refinement_training_donors": [section_id],
                "refinement_validation_donors": [section_id],
                "training_batches": [batch],
                "validation_batches": [batch],
                "refinement_training_batches": [batch],
                "refinement_validation_batches": [batch],
            },
        },
        checkpoint,
    )
    prediction_path = tmp_path / "predictions.npz"
    prediction_path.write_bytes(b"prediction")
    prediction = SimpleNamespace(
        sample_id=section_id,
        donor_id=section_id,
        inference_seed=17,
        refinement_round=4,
        latent_space_id=latent_space_id,
        expression_space_id=expression_space_id,
        checkpoint_sha256=sha256_file(checkpoint),
        prototype_sha256=sha256_file(refined_prototype),
    )
    prototypes = SimpleNamespace(
        donor_id=section_id,
        sample_ids=np.asarray([section_id]),
        latent_space_id=latent_space_id,
    )
    monkeypatch.setattr(
        PredictionBundle,
        "from_npz",
        classmethod(lambda cls, path: prediction),
    )
    monkeypatch.setattr(
        PrototypeSet,
        "load_npz",
        classmethod(lambda cls, path: prototypes),
    )
    audit = tmp_path / "refinement.json"
    audit.write_text(
        json.dumps(
            {
                "selected_round": 4,
                "rounds": [{"round_id": 4, "committed": True}],
                "prototype_artifacts": {
                    "%s::%s" % (section_id, section_id): str(refined_prototype.resolve())
                },
            }
        ),
        encoding="utf-8",
    )
    telemetry = tmp_path / "prediction.telemetry.json"
    clean_control = {
        "graph_node_shuffle": False,
        "image_feature_shuffle": False,
        "no_graph": False,
        "prototype_only": False,
        "wrong_donor": False,
        "prototype_donor_id": section_id,
        "seed": 17,
    }
    telemetry.write_text(
        json.dumps(
            {
                "schema": "heir.inference_telemetry.v1",
                "prediction_sha256": sha256_file(prediction_path),
                "negative_control": clean_control,
            }
        ),
        encoding="utf-8",
    )
    row = {
        "section_id": section_id,
        "predictions": prediction_path.name,
        "predictions_sha256": sha256_file(prediction_path),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "refinement_audit": audit.name,
        "refinement_audit_sha256": sha256_file(audit),
        "telemetry": telemetry.name,
        "telemetry_sha256": sha256_file(telemetry),
        "refined_prototype": refined_prototype.name,
        "refined_prototype_sha256": sha256_file(refined_prototype),
        "native_prototype_sha256": sha256_file(native_prototype),
    }
    manifest_payload = {
        "schema": "heir.snpatho_refined_prediction_manifest.v1",
        "analysis_role": "development_native_scanvi_published_annotation_sensitivity",
        "seed": 17,
        "round_selection_mode": "fixed",
        "selected_round": 4,
        "native_scanvi_manifest": native_manifest_path.name,
        "native_scanvi_manifest_sha256": native.sha256,
        "latent_space_id": latent_space_id,
        "expression_space_id": expression_space_id,
        "cases": [row],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    loaded = _load_refined_prediction_manifest(manifest, (section_id,), native)
    assert loaded[section_id].path == prediction_path.resolve()
    assert loaded[section_id].checkpoint_sha256 == sha256_file(checkpoint)
    assert loaded[section_id].telemetry_sha256 == sha256_file(telemetry)

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_payload["metadata"]["training_donors"] = [section_id]
    torch.save(checkpoint_payload, checkpoint)
    prediction.checkpoint_sha256 = sha256_file(checkpoint)
    row["checkpoint_sha256"] = prediction.checkpoint_sha256
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="all-exposure donor lineage"):
        _load_refined_prediction_manifest(manifest, (section_id,), native)
    checkpoint_payload["metadata"]["training_donors"] = [section_id, "4399"]
    checkpoint_payload["metadata"]["residual_basis_trainable"] = True
    torch.save(checkpoint_payload, checkpoint)
    prediction.checkpoint_sha256 = sha256_file(checkpoint)
    row["checkpoint_sha256"] = prediction.checkpoint_sha256
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="residual basis must remain frozen"):
        _load_refined_prediction_manifest(manifest, (section_id,), native)
    checkpoint_payload["metadata"]["residual_basis_trainable"] = False
    torch.save(checkpoint_payload, checkpoint)
    prediction.checkpoint_sha256 = sha256_file(checkpoint)
    row["checkpoint_sha256"] = prediction.checkpoint_sha256

    controlled = deepcopy(clean_control)
    controlled["image_feature_shuffle"] = True
    telemetry.write_text(
        json.dumps(
            {
                "schema": "heir.inference_telemetry.v1",
                "prediction_sha256": sha256_file(prediction_path),
                "negative_control": controlled,
            }
        ),
        encoding="utf-8",
    )
    row["telemetry_sha256"] = sha256_file(telemetry)
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="negative control"):
        _load_refined_prediction_manifest(manifest, (section_id,), native)


def test_native_scanvi_manifest_is_parsed_and_recursively_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    reports = repository / "reports"
    reports.mkdir(parents=True)
    model = tmp_path / "external" / "scanvi"
    model.mkdir(parents=True)
    (model / "model.pt").write_bytes(b"native-model")
    native_sha256 = _directory_sha256(model)
    latent_space_id = "sha256:" + native_sha256
    panel = repository / "manifests" / "gene_panel_snpatho_500.tsv"
    panel.parent.mkdir(parents=True)
    panel.write_text("# gene\tgroup\ng1\tcurated\n", encoding="utf-8")
    decoder = tmp_path / "external" / "decoder.pt"
    decoder.parent.mkdir(parents=True, exist_ok=True)
    normalization = {
        "method": "scvi.get_normalized_expression",
        "library_size": 10_000.0,
        "library_basis": "full-transcriptome",
        "gene_selection": "after-library-normalization",
        "transform": "log1p",
        "version": 2,
    }
    torch.save(
        {
            "config": {"input_dim": 1, "latent_dim": 2, "nonnegative_output": True},
            "state_dict": {},
            "metadata": {
                "schema": "heir.scvi_distilled_decoder.v2",
                "gene_names": ["g1"],
                "training_donors": ["4066"],
                "latent_space_id": latent_space_id,
                "expression_space_id": "log1p-cpm-10000-v1",
                "expression_normalization_contract": ("full_library_10000_then_panel_log1p_v2"),
                "expression_normalization": normalization,
                "decoder_only": True,
            },
        },
        decoder,
    )
    latent = repository / "latent.npz"
    latent.write_bytes(b"latent")
    prototypes = repository / "prototypes.npz"
    prototypes.write_bytes(b"prototypes")
    residual_geometry = repository / "residual-geometry.npz"
    residual_geometry.write_bytes(b"geometry")
    reference = SimpleNamespace(
        sample_id="4066",
        sample_ids=np.asarray(["4066"]),
        donor_ids=np.asarray(["4066"]),
        latent_space_id=latent_space_id,
        latent=np.zeros((1, 2), dtype=np.float32),
        counts=np.zeros((1, 1), dtype=np.float32),
        gene_ids=np.asarray(["g1"]),
        cell_type_labels=np.asarray(["A"]),
    )
    prototype_set = SimpleNamespace(
        donor_id="4066",
        sample_ids=np.asarray(["4066"]),
        latent_space_id=latent_space_id,
        means=np.zeros((1, 2), dtype=np.float32),
        source_reference_sha256=sha256_file(latent),
        cell_type_labels=np.asarray(["A"]),
        latent_transform_sha256="",
    )
    geometry = SimpleNamespace(
        type_names=np.asarray(["A"]),
        latent_space_id=latent_space_id,
        latent_dim=2,
        source_reference_sha256=sha256_file(latent),
        latent_transform_sha256="",
        training_donors=("4066",),
        n_cells=np.asarray([1]),
        n_prototypes=np.asarray([1]),
        rank=1,
        residual_type_max_norm=np.asarray([0.5], dtype=np.float32),
        residual_type_basis=np.asarray([[[1.0], [0.0]]], dtype=np.float32),
    )
    monkeypatch.setattr(
        RNAReference,
        "load_npz",
        classmethod(lambda cls, path: reference),
    )
    monkeypatch.setattr(
        PrototypeSet,
        "load_npz",
        classmethod(lambda cls, path: prototype_set),
    )
    monkeypatch.setattr(
        RNAResidualGeometry,
        "from_npz",
        classmethod(lambda cls, path: geometry),
    )
    payload = {
        "schema": "heir.snpatho_scanvi_r1_manifest.v1",
        "status": "native_scanvi_with_published_integrated_annotation_sensitivity",
        "workflow_filter": "processing_method == FFPE_snPATHO",
        "annotation_provenance": "published integrated labels",
        "gene_panel_sha256": sha256_file(panel),
        "expression_transform": "full_library_10000_then_panel_log1p_v2",
        "native_model": {
            "external_path": str(model),
            "sha256": native_sha256,
            "latent_dim": 2,
        },
        "distilled_decoder": {
            "external_path": str(decoder),
            "sha256": sha256_file(decoder),
        },
        "latent_space_id": latent_space_id,
        "expression_space_id": "log1p-cpm-10000-v1",
        "specimens": {
            "4066": {
                "cells": 1,
                "latent_reference": str(latent),
                "latent_reference_sha256": sha256_file(latent),
                "rare_complete_prototypes": str(prototypes),
                "rare_complete_prototypes_sha256": sha256_file(prototypes),
                "residual_geometry": str(residual_geometry),
                "residual_geometry_sha256": sha256_file(residual_geometry),
            }
        },
    }
    manifest = reports / "native.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    loaded = _load_native_scanvi_manifest(
        manifest,
        ("4066",),
        manifest_sha256=sha256_file(manifest),
    )
    assert loaded is not None
    assert loaded.native_model_sha256 == native_sha256
    assert loaded.expression_space_id == "log1p-cpm-10000-v1"
    assert loaded.decoder_gene_names == ("g1",)
    assert loaded.specimen_residual_geometry["4066"].rank == 1
    assert loaded.clean_annotation_complete is False

    geometry.source_reference_sha256 = "0" * 64
    with pytest.raises(ValueError, match="residual geometry lineage"):
        _load_native_scanvi_manifest(manifest, ("4066",))
    geometry.source_reference_sha256 = sha256_file(latent)

    decoder_payload = torch.load(decoder, map_location="cpu", weights_only=True)
    decoder_payload["metadata"]["expression_normalization"]["version"] = 1
    torch.save(decoder_payload, decoder)
    payload["distilled_decoder"]["sha256"] = sha256_file(decoder)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="normalization contract"):
        _load_native_scanvi_manifest(manifest, ("4066",))
    decoder_payload["metadata"]["expression_normalization"]["version"] = 2
    torch.save(decoder_payload, decoder)
    payload["distilled_decoder"]["sha256"] = sha256_file(decoder)

    payload.pop("expression_space_id")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expression_space_id"):
        _load_native_scanvi_manifest(manifest, ("4066",))
    payload["expression_space_id"] = "log1p-cpm-10000-v1"

    payload["status"] = "native_scanvi_with_independent_clean_reannotation"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash-bound reannotation manifest"):
        _load_native_scanvi_manifest(manifest, ("4066",))

    clean_artifacts = {}
    for role in ("annotation_table", "ontology", "qc_report", "adjudication_record"):
        artifact = reports / (role + ".txt")
        artifact.write_text(role + "\n", encoding="utf-8")
        clean_artifacts[role] = {"path": artifact.name, "sha256": sha256_file(artifact)}
    clean_manifest = reports / "clean-reannotation.json"
    clean_manifest.write_text(
        json.dumps(
            {
                "schema": "heir.snpatho_clean_reannotation_manifest.v1",
                "status": "complete",
                "workflow_filter": "processing_method == FFPE_snPATHO",
                "sample_ids": ["4066"],
                "label_source": "independent_clean_reannotation",
                "qc_complete": True,
                "adjudication_complete": True,
                **clean_artifacts,
            }
        ),
        encoding="utf-8",
    )
    payload["clean_reannotation"] = {
        "manifest": "reports/" + clean_manifest.name,
        "manifest_sha256": sha256_file(clean_manifest),
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    clean = _load_native_scanvi_manifest(manifest, ("4066",))
    assert clean is not None
    assert clean.clean_annotation_complete is True
    assert clean.clean_reannotation_manifest_sha256 == sha256_file(clean_manifest)


def test_five_seed_manifest_requires_full_bound_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_path = tmp_path / "native.json"
    native_path.write_text("{}", encoding="utf-8")
    native = NativeScanviManifest(
        path=native_path,
        sha256=sha256_file(native_path),
        latent_space_id="latent-v1",
        expression_space_id="expression-v1",
        native_model_sha256="a" * 64,
        decoder_sha256="b" * 64,
        annotation_status="published_integrated_annotation_sensitivity_not_clean_reannotation",
        specimen_prototype_sha256={"4066": "c" * 64},
    )
    rows = []
    predictions = {}
    for seed in (17, 41):
        path = tmp_path / ("prediction-%d.npz" % seed)
        path.write_bytes(("prediction-%d" % seed).encode("ascii"))
        predictions[path.resolve()] = SimpleNamespace(
            sample_id="4066",
            donor_id="4066",
            inference_seed=seed,
            refinement_round=4,
            latent_space_id="latent-v1",
            expression_space_id="expression-v1",
        )
        rows.append(
            {
                "section_id": "4066",
                "seed": seed,
                "predictions": path.name,
                "predictions_sha256": sha256_file(path),
            }
        )
    monkeypatch.setattr(
        PredictionBundle,
        "from_npz",
        classmethod(lambda cls, path: predictions[Path(path).resolve()]),
    )
    payload = {
        "schema": "heir.snpatho_five_seed_refinement_manifest.v1",
        "analysis_role": "prespecified_five_seed_native_scanvi_integrated_annotation_sensitivity",
        "seeds": [17, 41],
        "samples": ["4066"],
        "negative_control": False,
        "native_scanvi_manifest_sha256": native.sha256,
        "latent_space_id": "latent-v1",
        "expression_space_id": "expression-v1",
        "controls_available": ["prototype_only"],
        "cases": rows,
    }
    manifest = tmp_path / "five-seed.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    loaded = _load_five_seed_prediction_manifest(
        manifest,
        ("4066",),
        (17, 41),
        native,
    )
    assert loaded is not None
    assert set(loaded.predictions) == {("4066", 17), ("4066", 41)}
    assert loaded.control_names == ("prototype_only",)

    payload["cases"] = rows[:1]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="full matrix"):
        _load_five_seed_prediction_manifest(
            manifest,
            ("4066",),
            (17, 41),
            native,
        )


def _refinement_matrix_summary_payload(
    native_sha256: str,
    *,
    strict_status: str = "pass",
    evidence_count: int = 0,
    execution_provenance_count: int = 0,
) -> dict:
    totals = {
        "refined_gt_round0": 4,
        "refined_gt_hard_baseline": 4,
        "refined_gt_soft_baseline": 4,
        "refined_gt_prototype_only": 2,
        "round0_gt_prototype_only": 2,
        "refined_gt_image_shuffle": 2,
        "refined_gt_graph_shuffle": 2,
        "refined_gt_no_graph": 2,
        "refined_gt_wrong_donor": 2,
    }
    by_check = {
        name: {"total": total, "pass": total, "fail": 0, "blocked": 0}
        for name, total in totals.items()
    }
    if strict_status == "fail":
        by_check["refined_gt_round0"].update({"pass": 3, "fail": 1})
    pass_count = sum(row["pass"] for row in by_check.values())
    fail_count = sum(row["fail"] for row in by_check.values())
    primary_evidence_status = (
        "complete" if evidence_count == 0 and execution_provenance_count == 0 else "blocked"
    )
    total_blockers = evidence_count + execution_provenance_count
    by_code = {}
    by_requirement = {}
    groups = []
    if evidence_count:
        by_code["missing_required_followup_evidence"] = evidence_count
        by_requirement["required_followup_evidence"] = evidence_count
        groups.append(
            {
                "code": "missing_required_followup_evidence",
                "requirement": "required_followup_evidence",
                "message": "required follow-up evidence is incomplete",
                "count": evidence_count,
                "samples": [],
                "seeds": [],
                "variants": [],
            }
        )
    if execution_provenance_count:
        by_code["unverified_execution_provenance"] = execution_provenance_count
        by_requirement["execution_provenance"] = execution_provenance_count
        groups.append(
            {
                "code": "unverified_execution_provenance",
                "requirement": "execution_provenance",
                "message": "execution provenance is incomplete",
                "count": execution_provenance_count,
                "samples": [],
                "seeds": [],
                "variants": [],
            }
        )
    return {
        "schema": "heir.snpatho_refinement_matrix_public_summary.v1",
        "report_schema": "heir.snpatho_refinement_matrix.v1",
        "status": (
            "blocked_evidence"
            if primary_evidence_status == "blocked"
            else ("complete" if strict_status == "pass" else "complete_ordering_failed")
        ),
        "matrix_status": "complete",
        "primary_evidence_status": primary_evidence_status,
        "execution_provenance_verified": execution_provenance_count == 0,
        "strict_ordering_status": strict_status,
        "analysis_role": "native_scanvi_published_integrated_annotation_sensitivity",
        "request": {
            "samples": ["4066", "4411"],
            "seeds": [17, 41],
            "trajectory_seed": 17,
            "controls": [
                "prototype_only",
                "image_shuffle",
                "graph_shuffle",
                "no_graph",
                "wrong_donor",
            ],
            "control_seeds": [17],
            "wrong_donor_pairings": [
                {"target": "4066", "source": "4411"},
                {"target": "4411", "source": "4066"},
            ],
            "minimum_nuclei": 3,
        },
        "artifacts": {"requested": 24, "scored": 24},
        "strict_ordering": {
            "status": strict_status,
            "required_policy": "test policy",
            "check_counts": {
                "total": pass_count + fail_count,
                "pass": pass_count,
                "fail": fail_count,
                "blocked": 0,
            },
            "by_check": by_check,
        },
        "blockers": {
            "total_count": total_blockers,
            "matrix_count": 0,
            "evidence_count": evidence_count,
            "execution_provenance_count": execution_provenance_count,
            "by_code": by_code,
            "by_requirement": by_requirement,
            "groups": groups,
        },
        "provenance": {
            "manifests": {
                "native_r1": {"sha256": native_sha256},
                "frozen_truth": {"sha256": "f" * 64},
            },
            "inputs": {
                "4066": {
                    "truth": {
                        "sha256": "1" * 64,
                        "hash_validation": "matched_frozen_truth_manifest",
                    },
                    "native_r1_reference": {
                        "sha256": "2" * 64,
                        "hash_validation": "matched_native_scanvi_manifest",
                    },
                },
                "4411": {
                    "truth": {
                        "sha256": "3" * 64,
                        "hash_validation": "matched_frozen_truth_manifest",
                    },
                    "native_r1_reference": {
                        "sha256": "4" * 64,
                        "hash_validation": "matched_native_scanvi_manifest",
                    },
                },
            },
        },
    }


def _matrix_test_native(tmp_path: Path, *, clean: bool = False) -> NativeScanviManifest:
    path = tmp_path / "native.json"
    path.write_text("{}", encoding="utf-8")
    return NativeScanviManifest(
        path=path,
        sha256=sha256_file(path),
        latent_space_id="latent-v1",
        expression_space_id="expression-v1",
        native_model_sha256="a" * 64,
        decoder_sha256="b" * 64,
        annotation_status=(
            "independent_clean_reannotation_complete"
            if clean
            else "published_integrated_annotation_sensitivity_not_clean_reannotation"
        ),
        specimen_prototype_sha256={"4066": "c" * 64, "4411": "d" * 64},
        clean_reannotation_manifest_sha256="e" * 64 if clean else None,
        validated_clean_reannotation=clean,
    )


@pytest.mark.parametrize("strict_status", ["pass", "fail"])
def test_refinement_matrix_summary_validates_pass_and_fail(
    tmp_path: Path,
    strict_status: str,
) -> None:
    native = _matrix_test_native(tmp_path)
    path = tmp_path / "matrix.json"
    path.write_text(
        json.dumps(_refinement_matrix_summary_payload(native.sha256, strict_status=strict_status)),
        encoding="utf-8",
    )

    result = _load_refinement_matrix_summary(
        path,
        ("4066", "4411"),
        (17, 41),
        (17,),
        minimum_nuclei=3,
        frozen_plan_sha256="f" * 64,
        native_scanvi=native,
        summary_sha256=sha256_file(path),
    )

    assert isinstance(result, RefinementMatrixSummary)
    assert result.matrix_complete is True
    assert result.strict_ordering_pass is (strict_status == "pass")
    assert result.required_followup_evidence_complete is True
    assert result.wrong_donor_coverage_complete is True


def test_refinement_matrix_legacy_single_wrong_donor_pair_fails_closed(tmp_path: Path) -> None:
    native = _matrix_test_native(tmp_path)
    payload = _refinement_matrix_summary_payload(native.sha256)
    payload["request"]["wrong_donor_pairings"] = [{"target": "4066", "source": "4411"}]
    payload["artifacts"] = {"requested": 23, "scored": 23}
    payload["strict_ordering"]["by_check"]["refined_gt_wrong_donor"] = {
        "total": 1,
        "pass": 1,
        "fail": 0,
        "blocked": 0,
    }
    payload["strict_ordering"]["check_counts"]["total"] -= 1
    payload["strict_ordering"]["check_counts"]["pass"] -= 1
    path = tmp_path / "legacy-single-pair.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _load_refinement_matrix_summary(
        path,
        ("4066", "4411"),
        (17, 41),
        (17,),
        minimum_nuclei=3,
        frozen_plan_sha256="f" * 64,
        native_scanvi=native,
        summary_sha256=sha256_file(path),
    )

    assert result is not None
    assert result.matrix_complete is False
    assert result.matrix_status == "blocked"
    assert result.strict_ordering_status == "blocked"
    assert result.overall_status == "blocked_matrix"
    assert result.wrong_donor_coverage_complete is False
    assert result.missing_wrong_donor_case_count == 1


def test_refinement_matrix_followup_evidence_blocks_full_primary(tmp_path: Path) -> None:
    native = _matrix_test_native(tmp_path, clean=True)
    path = tmp_path / "matrix-evidence-blocked.json"
    payload = _refinement_matrix_summary_payload(native.sha256, evidence_count=6)
    payload["analysis_role"] = "native_scanvi_clean_independent_reannotation_primary"
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    summary = _load_refinement_matrix_summary(
        path,
        ("4066", "4411"),
        (17, 41),
        (17,),
        minimum_nuclei=3,
        frozen_plan_sha256="f" * 64,
        native_scanvi=native,
        summary_sha256=sha256_file(path),
    )
    assert summary is not None
    assert summary.primary_evidence_status == "blocked"
    assert summary.evidence_blocker_count == 6
    assert summary.required_followup_evidence_complete is False

    five_seed = FiveSeedPredictionManifest(
        path=tmp_path / "five-seed.json",
        sha256="a" * 64,
        predictions={},
        control_names=REFINEMENT_MATRIX_CONTROLS,
        execution_provenance_verified=True,
    )
    gates = _full_primary_evidence_gates(
        cast(
            DeepBenchPlan,
            SimpleNamespace(
                optional_artifacts={name: None for name in OPTIONAL_ARTIFACTS},
                optional_artifact_sha256={name: None for name in OPTIONAL_ARTIFACTS},
                sample_ids=("4066", "4411"),
                frozen_plan_sha256="f" * 64,
            ),
        ),
        native,
        five_seed,
        summary,
    )
    assert gates["gates"]["required_followup_evidence_complete"] is False
    assert gates["blockers"] == [
        "required_followup_evidence_complete",
        "composition_adjusted_residuals_hash_validated",
        "required_he_tissue_fraction_qc_hash_validated",
        "calibrated_segmentation_confidence_hash_validated",
    ]
    assert gates["eligible_for_full_primary_claim"] is False


def test_refinement_matrix_summary_rejects_malformed_and_stale_files(tmp_path: Path) -> None:
    native = _matrix_test_native(tmp_path)
    path = tmp_path / "matrix.json"
    payload = _refinement_matrix_summary_payload(native.sha256)
    payload["artifacts"]["requested"] = 22
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = sha256_file(path)
    with pytest.raises(ValueError, match="artifact coverage"):
        _load_refinement_matrix_summary(
            path,
            ("4066", "4411"),
            (17, 41),
            (17,),
            minimum_nuclei=3,
            frozen_plan_sha256="f" * 64,
            native_scanvi=native,
            summary_sha256=digest,
        )

    payload = _refinement_matrix_summary_payload(native.sha256, evidence_count=1)
    payload["primary_evidence_status"] = "complete"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="primary-evidence status contradicts blocker counts"):
        _load_refinement_matrix_summary(
            path,
            ("4066", "4411"),
            (17, 41),
            (17,),
            minimum_nuclei=3,
            frozen_plan_sha256="f" * 64,
            native_scanvi=native,
            summary_sha256=sha256_file(path),
        )

    payload = _refinement_matrix_summary_payload(
        native.sha256,
        execution_provenance_count=1,
    )
    payload["execution_provenance_verified"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="execution-provenance status contradicts blocker counts"):
        _load_refinement_matrix_summary(
            path,
            ("4066", "4411"),
            (17, 41),
            (17,),
            minimum_nuclei=3,
            frozen_plan_sha256="f" * 64,
            native_scanvi=native,
            summary_sha256=sha256_file(path),
        )

    payload = _refinement_matrix_summary_payload(native.sha256)
    payload["provenance"]["manifests"]["native_r1"]["sha256"] = "9" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound to the native scANVI"):
        _load_refinement_matrix_summary(
            path,
            ("4066", "4411"),
            (17, 41),
            (17,),
            minimum_nuclei=3,
            frozen_plan_sha256="f" * 64,
            native_scanvi=native,
            summary_sha256=sha256_file(path),
        )

    path.write_text(
        json.dumps(_refinement_matrix_summary_payload(native.sha256)),
        encoding="utf-8",
    )
    stale_digest = sha256_file(path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash-mismatched"):
        _load_refinement_matrix_summary(
            path,
            ("4066", "4411"),
            (17, 41),
            (17,),
            minimum_nuclei=3,
            frozen_plan_sha256="f" * 64,
            native_scanvi=native,
            summary_sha256=stale_digest,
        )


def test_committed_workflow_audit_is_internally_consistent() -> None:
    path = Path(__file__).resolve().parents[1] / "reports" / "snpatho_reference_workflow_audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["processing_method_column"] == "processing_method"
    assert payload["cell_type_column"] == "major_annotation"
    assert payload["filters"]["applied"] is False
    for specimen in payload["specimens"].values():
        assert len(specimen["source_sha256"]) == 64
        assert sum(specimen["counts_by_workflow"].values()) == specimen["total_metadata_rows"]
        for workflow, count in specimen["counts_by_workflow"].items():
            assert sum(specimen["cell_type_counts_by_workflow"][workflow].values()) == count


def test_committed_r1_manifest_freezes_exact_ffpe_filter_and_counts() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "reports" / "snpatho_r1_reference_manifest.json").read_text(encoding="utf-8")
    )

    assert payload["filter"] == {
        "column": "processing_method",
        "accepted_values": ["FFPE_snPATHO"],
        "matching": "exact",
    }
    assert payload["gene_panel"]["sha256"] == (
        "22ddb91188b3b124d5cf3ec0f7ae81017399d141e39647b0dce80675119fe927"
    )
    assert payload["cell_type_annotation"]["primary_clean_reannotation_status"] == ("not_complete")
    for specimen in payload["specimens"].values():
        assert specimen["source_observations"] > specimen["selected_observations"]
        assert sum(specimen["cell_type_counts"].values()) == specimen["selected_observations"]
        count_types = set(specimen["cell_type_counts"])
        prototype_types = set(specimen["prototype_supported_types"])
        omitted_types = set(specimen["prototype_omitted_rare_types"])
        assert count_types == prototype_types | omitted_types
        assert prototype_types.isdisjoint(omitted_types)
        for name in (
            "source_rds_sha256",
            "h5ad_sha256",
            "conversion_provenance_sha256",
            "panel_reference_sha256",
            "latent_reference_sha256",
            "prototypes_sha256",
        ):
            digest = specimen[name]
            assert len(digest) == 64
            assert set(digest) <= set("0123456789abcdef")


def test_public_summary_uses_current_plan_schema_and_optional_local_report_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (root / "reports" / "snpatho_deepbench_v1_summary.json").read_text(encoding="utf-8")
    )
    plan = root / "configs" / "experiments" / "snpatho_deepbench_v1.yaml"

    assert summary["schema"] == "heir.snpatho_deepbench_public_summary.v2"
    assert summary["report_schema"] == "heir.snpatho_deepbench.v2"
    assert summary["provenance"]["deepbench_plan_sha256"] == sha256_file(plan)
    assert (
        "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean"
        in summary["macro"]
    )
    assert "bootstrap_fraction_delta_positive" in summary["macro"]
    assert "paired_bootstrap_probability_positive" not in summary["macro"]

    local_outputs = {
        "full_local_json_sha256": root / "artifacts/snpatho/deepbench_v1/report.json",
        "full_local_tsv_sha256": root / "artifacts/snpatho/deepbench_v1/report.tsv",
        "full_local_markdown_sha256": root / "artifacts/snpatho/deepbench_v1/report.md",
    }
    for field, path in local_outputs.items():
        if path.is_file():
            assert summary["provenance"][field] == sha256_file(path)
    local_report = local_outputs["full_local_json_sha256"]
    if local_report.is_file():
        report = json.loads(local_report.read_text(encoding="utf-8"))
        assert report["benchmark"]["plan_sha256"] == summary["provenance"]["deepbench_plan_sha256"]
        assert (
            report["primary"]["macro_delta"]
            == summary["macro"][
                "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean"
            ]
        )
        assert (
            report["primary"]["bootstrap"]["bootstrap_fraction_delta_positive"]
            == summary["macro"]["bootstrap_fraction_delta_positive"]
        )


def test_r1_reference_identity_is_bound_to_specimen_and_h5ad_lineage() -> None:
    source_sha256 = "a" * 64
    reference = RNAReference(
        sample_id="4066",
        cell_ids=np.asarray(["a", "b"]),
        gene_ids=np.asarray(["g"]),
        counts=np.asarray([[1.0], [2.0]]),
        donor_ids=np.asarray(["4066", "4066"]),
        sample_ids=np.asarray(["4066", "4066"]),
        block_id="4066_FFPE",
        source_count_sha256=source_sha256,
    )
    manifest_entry = {"h5ad_sha256": source_sha256}

    _validate_r1_reference_identity(reference, "4066", manifest_entry)

    with pytest.raises(ValueError, match="sample_id differs"):
        _validate_r1_reference_identity(
            replace(reference, sample_id="4399"),
            "4066",
            manifest_entry,
        )
    with pytest.raises(ValueError, match="source-count lineage"):
        _validate_r1_reference_identity(
            replace(reference, source_count_sha256="b" * 64),
            "4066",
            manifest_entry,
        )


def test_attached_deepbench_method_critical_fields_are_frozen() -> None:
    payload = _specification()
    validate_deepbench_specification(payload)

    changed = deepcopy(payload)
    changed["statistics"]["pooled_spot_inference"] = "allowed"
    with pytest.raises(ValueError, match="pooled_spot_inference"):
        validate_deepbench_specification(changed)

    too_few_shuffles = deepcopy(payload)
    too_few_shuffles["statistics"]["final_cell_record_shuffle_permutations"] = 99
    with pytest.raises(ValueError, match="final_cell_record_shuffle_permutations"):
        validate_deepbench_specification(too_few_shuffles)


def test_rna_mass_spot_aggregation_operates_in_linear_space() -> None:
    expression = np.log1p(np.asarray([[9.0, 1.0], [1.0, 5.0], [7.0, 3.0]]))
    spot_index = np.asarray([0, 0, -1])
    observed, mass = aggregate_cells_to_spots(
        expression,
        spot_index,
        num_spots=2,
        cell_rna_mass=np.asarray([1.0, 3.0, 100.0]),
    )

    np.testing.assert_allclose(observed[0], np.log1p([3.0, 4.0]))
    np.testing.assert_allclose(observed[1], [0.0, 0.0])
    np.testing.assert_allclose(mass, [4.0, 0.0])


def test_reference_profiles_pool_raw_counts_and_full_library_mass() -> None:
    profiles, median_library_sizes = _reference_linear_profiles(
        _linear_reference(),
        ["A", "B"],
    )

    np.testing.assert_allclose(profiles[0], np.asarray([10.0, 20.0]) / 300.0 * 10_000.0)
    np.testing.assert_allclose(profiles[1], np.asarray([3.0, 7.0]) / 1_000.0 * 10_000.0)
    np.testing.assert_allclose(median_library_sizes, [150.0, 1_000.0])


def test_matched_type_mean_uses_hard_assignment_not_soft_averaging() -> None:
    reference = _linear_reference()
    prediction = cast(
        PredictionBundle,
        SimpleNamespace(
            type_names=np.asarray(["A", "B"]),
            type_probabilities=np.asarray([[0.51, 0.49], [0.10, 0.90]]),
        ),
    )

    linear_cells = np.expm1(_type_mean_cells(reference, prediction))
    expected_profiles, _ = _reference_linear_profiles(reference, ["A", "B"])

    np.testing.assert_allclose(linear_cells, expected_profiles, rtol=1.0e-6)
    assert not np.allclose(
        linear_cells[0],
        0.51 * expected_profiles[0] + 0.49 * expected_profiles[1],
    )


def test_soft_type_mean_probability_weights_linear_profiles() -> None:
    reference = _linear_reference()
    prediction = cast(
        PredictionBundle,
        SimpleNamespace(
            type_names=np.asarray(["A", "B"]),
            type_probabilities=np.asarray([[0.51, 0.49], [0.10, 0.90]]),
        ),
    )

    linear_cells = np.expm1(_soft_type_mean_cells(reference, prediction))
    profiles, _ = _reference_linear_profiles(reference, ["A", "B"])

    np.testing.assert_allclose(
        linear_cells,
        np.asarray([[0.51, 0.49], [0.10, 0.90]]).dot(profiles),
        rtol=1.0e-6,
    )
    assert SOFT_TYPE_MEAN_METHOD == "historical_integrated_soft_type_mean"


def test_matched_r1_baselines_use_refined_probabilities_not_historical_map() -> None:
    reference = _linear_reference()
    historical = cast(
        PredictionBundle,
        SimpleNamespace(
            type_names=np.asarray(["A", "B"]),
            type_probabilities=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        ),
    )
    refined = cast(
        PredictionBundle,
        SimpleNamespace(
            type_names=np.asarray(["A", "B"]),
            type_probabilities=np.asarray([[0.0, 1.0], [0.25, 0.75]]),
        ),
    )

    soft_mass, hard_mass, hard_cells, soft_cells = _matched_r1_baseline_cell_values(
        reference,
        refined,
    )

    np.testing.assert_allclose(soft_mass, _cell_rna_mass(reference, refined))
    np.testing.assert_allclose(hard_mass, _hard_assigned_cell_rna_mass(reference, refined))
    np.testing.assert_allclose(hard_cells, _type_mean_cells(reference, refined))
    np.testing.assert_allclose(soft_cells, _soft_type_mean_cells(reference, refined))
    assert not np.allclose(hard_cells, _type_mean_cells(reference, historical))
    assert not np.allclose(soft_mass, _cell_rna_mass(reference, historical))


def test_type_mean_ladder_separates_profile_and_cell_mass_estimands() -> None:
    reference = _linear_reference()
    prediction = cast(
        PredictionBundle,
        SimpleNamespace(
            type_names=np.asarray(["A", "B"]),
            type_probabilities=np.asarray([[0.51, 0.49], [0.49, 0.51]]),
        ),
    )

    shared_soft_mass = _cell_rna_mass(reference, prediction)
    hard_assigned_mass = _hard_assigned_cell_rna_mass(reference, prediction)
    contracts = _baseline_estimands()

    assert not np.allclose(shared_soft_mass, hard_assigned_mass)
    np.testing.assert_allclose(
        hard_assigned_mass / hard_assigned_mass[0],
        [1.0, 1_000.0 / 150.0],
    )
    assert contracts[HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD]["cell_rna_mass"] == (
        "hard_assigned_type_median_library_size"
    )
    assert contracts[TYPE_MEAN_METHOD]["cell_rna_mass"] == (
        "shared_soft_expected_type_median_library_size"
    )
    assert contracts[SOFT_TYPE_MEAN_METHOD]["cell_rna_mass"] == (
        "expected_soft_type_median_library_size"
    )
    assert contracts[EQUAL_CELL_HARD_TYPE_MEAN_METHOD]["cell_rna_mass"] == "equal_cell"
    assert contracts[EQUAL_CELL_SOFT_TYPE_MEAN_METHOD]["cell_rna_mass"] == "equal_cell"
    assert contracts[R1_HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD]["reference"].startswith(
        "matched_ffpe_snpatho"
    )


def test_count_reference_and_prototype_support_are_reported_separately() -> None:
    reference = _linear_reference()
    prediction = cast(
        PredictionBundle,
        SimpleNamespace(
            type_names=np.asarray(["A", "B"]),
            type_probabilities=np.asarray([[0.6, 0.4], [0.4, 0.6]]),
        ),
    )
    prototypes = PrototypeSet(
        prototype_ids=np.asarray(["reference:A:0"]),
        sample_ids=np.asarray(["reference"]),
        cell_type_labels=np.asarray(["A"]),
        means=np.asarray([[0.0, 1.0]], dtype=np.float32),
        variances=np.asarray([[1.0, 1.0]], dtype=np.float32),
        weights=np.asarray([1.0]),
    )

    support = _reference_prototype_type_support(reference, prediction, prototypes)

    assert support["count_reference_supported_types"] == ["A", "B"]
    assert support["prototype_supported_types"] == ["A"]
    assert support["prototype_omitted_types"] == ["B"]
    assert support["prototype_omitted_prediction_types"] == ["B"]
    assert support["prototype_support_source"] == "legacy_svd_sensitivity_prototype_bank"

    native_support = _prototype_type_support(
        reference,
        prediction,
        ("A", "B"),
        source="native_scanvi_rare_complete_prototype_bank",
        policy="rare complete",
    )
    assert native_support["prototype_supported_types"] == ["A", "B"]
    assert native_support["prototype_omitted_types"] == []
    assert native_support["prototype_support_source"] == (
        "native_scanvi_rare_complete_prototype_bank"
    )

    invalid = replace(prototypes, cell_type_labels=np.asarray(["not_in_counts"]))
    with pytest.raises(ValueError, match="absent from its count reference"):
        _reference_prototype_type_support(reference, prediction, invalid)


def test_missing_reference_type_is_audited_then_fails_closed() -> None:
    reference = _linear_reference()
    prediction = cast(
        PredictionBundle,
        SimpleNamespace(
            type_names=np.asarray(["A", "missing"]),
            type_probabilities=np.asarray([[0.1, 0.9], [0.8, 0.2]]),
        ),
    )

    audit = _reference_type_support(reference, prediction)

    assert audit["reference_supported_prediction_cell_types"] == ["A"]
    assert audit["missing_prediction_cell_types"] == ["missing"]
    assert audit["missing_type_policy"] == "fail_closed_no_global_profile_fallback"
    assert audit["hard_assignment_global_fallback_cells"] == 1
    assert audit["hard_assignment_global_fallback_cell_fraction"] == pytest.approx(0.5)
    assert audit["soft_assignment_global_fallback_probability_mass_mean"] == pytest.approx(0.55)
    with pytest.raises(ValueError, match="global-profile fallback is prohibited"):
        _reference_linear_profiles(reference, ["A", "missing"])


def test_type_map_diagnostics_distinguish_constant_hard_from_variable_soft_mix() -> None:
    prediction = cast(
        PredictionBundle,
        SimpleNamespace(
            type_names=np.asarray(["A", "B"]),
            type_probabilities=np.asarray(
                [
                    [0.90, 0.10],
                    [0.90, 0.10],
                    [0.80, 0.20],
                    [0.80, 0.20],
                    [0.60, 0.40],
                    [0.60, 0.40],
                ]
            ),
        ),
    )

    audit = _type_map_diagnostics(
        prediction,
        np.asarray([0, 0, 1, 1, 2, 2]),
        np.asarray([True, True, True]),
    )

    assert audit["hard_assignment_counts"] == {"A": 6, "B": 0}
    assert audit["occupied_hard_type_count"] == 1
    assert audit["occupied_hard_types"] == ["A"]
    assert audit["per_type_spot_mixture"]["B"]["hard_assignment_fraction"] == {
        "spatial_standard_deviation": 0.0,
        "spatially_constant": True,
        "minimum": 0.0,
        "maximum": 0.0,
    }
    assert (
        audit["per_type_spot_mixture"]["B"]["soft_expected_fraction"]["spatially_constant"] is False
    )
    assert (
        audit["per_type_spot_mixture"]["B"]["soft_expected_fraction"]["spatial_standard_deviation"]
        > 0
    )
    assert 0 < audit["normalized_probability_entropy"]["mean"] < 1


def test_top_decile_ties_use_frozen_lower_index_policy() -> None:
    values = np.asarray([9.0, 9.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    selected = _top_indices(values, fraction=0.10)

    np.testing.assert_array_equal(selected, [0, 1])


def test_repeated_final_record_shuffle_is_deterministic_compact_and_preserves_draw_zero() -> None:
    spot_index = np.repeat(np.arange(4, dtype=np.int64), 3)
    expression = np.log1p(
        np.column_stack(
            (
                np.arange(1, 13, dtype=np.float64),
                np.asarray([1, 4, 2, 8, 3, 9, 5, 7, 6, 12, 10, 11], dtype=np.float64),
            )
        )
    )
    weights = np.linspace(0.5, 2.0, len(expression))
    truth = np.asarray(
        [[0.1, 1.2], [0.8, 0.3], [1.4, 1.8], [2.0, 0.9]],
        dtype=np.float64,
    )
    primary_spots = np.ones(4, dtype=bool)

    first = _repeated_final_record_shuffle_null(
        expression,
        weights,
        spot_index,
        primary_spots,
        truth,
        sample="4066",
        seed=17,
        permutations=100,
    )
    repeated = _repeated_final_record_shuffle_null(
        expression,
        weights,
        spot_index,
        primary_spots,
        truth,
        sample="4066",
        seed=17,
        permutations=100,
    )

    assert first[0] == repeated[0]
    np.testing.assert_array_equal(first[1], repeated[1])
    np.testing.assert_array_equal(first[2], repeated[2])
    np.testing.assert_array_equal(first[3], repeated[3])
    assert first[0]["permutations"] == 100
    assert first[0]["statistic"] == "median_gene_spearman"
    assert "values" not in first[0]
    assert set(first[0]["empirical_percentile_interval_95"]) == {"lower", "upper"}

    assigned = np.flatnonzero(spot_index >= 0)
    draw_zero = np.random.default_rng(_record_shuffle_seed(17, "4066", 0)).permutation(assigned)
    expected_spots, expected_mass = aggregate_cells_to_spots(
        expression[draw_zero],
        spot_index[assigned],
        len(truth),
        weights[draw_zero],
    )
    np.testing.assert_allclose(first[1], expected_spots)
    np.testing.assert_allclose(first[2], expected_mass)
    expected_metric = deepbench_expression_metrics(
        expected_spots,
        truth,
        np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64),
    )["summary"]["median_gene_spearman"]
    assert first[3][0] == pytest.approx(expected_metric)


def test_record_shuffle_seeds_are_distinct_across_draws_and_specimens() -> None:
    seeds = {
        _record_shuffle_seed(17, specimen, draw)
        for specimen in ("4066", "4399", "4411")
        for draw in range(100)
    }

    assert len(seeds) == 300


def test_constant_prediction_is_zero_not_dropped_when_truth_varies() -> None:
    observed = np.asarray(
        [
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 2.0],
        ],
        dtype=np.float64,
    )
    predicted = np.ones_like(observed)
    coordinates = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
    result = deepbench_expression_metrics(predicted, observed, coordinates)

    assert result["per_gene"]["spearman"] == [0.0, None]
    assert result["summary"]["median_gene_spearman"] == 0.0
    assert result["summary"]["fraction_genes_evaluable"] == 0.5
    assert result["summary"]["prediction_constant_scored_zero_count"] == 1
    assert result["summary"]["observed_constant_excluded_count"] == 1


def test_deepbench_metrics_include_hotspots_locations_and_spatial_agreement() -> None:
    coordinates = np.asarray([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float64)
    observed = np.asarray([[0.0, 1.0], [0.2, 1.2], [0.4, 1.4], [0.6, 1.6], [0.8, 1.8], [1.0, 2.0]])
    result = deepbench_expression_metrics(observed.copy(), observed, coordinates)
    summary = result["summary"]

    assert summary["median_gene_spearman"] == pytest.approx(1.0)
    assert summary["median_gene_mse"] == pytest.approx(0.0)
    assert summary["median_hotspot_dice"] == pytest.approx(1.0)
    assert summary["median_hotspot_jaccard"] == pytest.approx(1.0)
    assert summary["median_expression_detection_auroc"] == pytest.approx(1.0)
    assert "median_hotspot_auroc" not in summary
    assert summary["mean_location_cosine"] == pytest.approx(1.0)
    assert summary["morans_i_mae"] == pytest.approx(0.0)


def test_refined_primary_requirement_must_beat_both_matched_r1_baselines() -> None:
    def method(values: list[float]) -> dict:
        return {"per_gene": {"spearman": values}}

    case = {
        "section_id": "4066",
        "methods": {
            REFINED_R1_METHOD: method([0.4, 0.5]),
            R1_HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD: method([0.2, 0.3]),
            R1_SOFT_TYPE_MEAN_METHOD: method([0.3, 0.4]),
        },
    }

    passing = _requested_refined_primary_contrasts([case])

    assert passing["joint_contrast"] == REQUESTED_PRIMARY_CONTRAST
    assert passing["requires_both_contrasts"] is True
    assert passing["status"] == "passes_developmental_joint_contrast"
    assert passing["full_primary_claim"] is False
    assert passing["refined_beats_both_matched_ffpe_r1_baselines"] is True
    assert passing["contrasts"]["matched_ffpe_r1_hard"]["macro_delta"] == pytest.approx(0.2)
    assert passing["contrasts"]["matched_ffpe_r1_soft"]["macro_delta"] == pytest.approx(0.1)

    case["methods"][R1_HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD] = method([0.6, 0.7])
    failing = _requested_refined_primary_contrasts([case])

    assert failing["status"] == "fails_developmental_joint_contrast"
    assert failing["refined_beats_both_matched_ffpe_r1_baselines"] is False
    assert failing["contrasts"]["matched_ffpe_r1_soft"]["macro_delta_positive"] is True
    assert failing["contrasts"]["matched_ffpe_r1_hard"]["macro_delta_positive"] is False


def _with_primary_gate_support(tmp_path: Path, plan: DeepBenchPlan) -> DeepBenchPlan:
    optional = dict(plan.optional_artifacts)
    optional_hashes = dict(plan.optional_artifact_sha256)
    for gate, (
        artifact_name,
        evidence_kind,
        requirements,
    ) in PRIMARY_GATE_SUPPORT_CONTRACTS.items():
        per_sample = {}
        for sample_id in plan.sample_ids:
            result = tmp_path / ("%s-%s.json" % (gate, sample_id))
            result.write_text('{"validated": true}\n', encoding="utf-8")
            per_sample[sample_id] = {
                "path": result.name,
                "sha256": sha256_file(result),
            }
        manifest = tmp_path / (gate + "-manifest.json")
        manifest.write_text(
            json.dumps(
                {
                    "schema": PRIMARY_GATE_SUPPORT_SCHEMA,
                    "evidence_kind": evidence_kind,
                    "frozen_benchmark_plan_sha256": plan.frozen_plan_sha256,
                    "sample_ids": list(plan.sample_ids),
                    "requirements": dict(requirements),
                    "per_sample_artifacts": per_sample,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        optional[artifact_name] = manifest
        optional_hashes[artifact_name] = sha256_file(manifest)
    return replace(
        plan,
        optional_artifacts=optional,
        optional_artifact_sha256=optional_hashes,
    )


def test_one_seed_integrated_label_sensitivity_never_passes_full_primary(
    tmp_path: Path,
) -> None:
    def method(values: list[float], *, observed: bool = False) -> dict:
        per_gene: dict = {"spearman": values}
        if observed:
            per_gene["observed_mean"] = [1.0, 2.0]
        return {
            "per_gene": per_gene,
            "summary": {"median_gene_spearman": 0.4, "median_gene_mse": 0.5},
        }

    case = {
        "section_id": "4066",
        "methods": {
            PRIMARY_METHOD: method([0.4, 0.5], observed=True),
            TYPE_MEAN_METHOD: method([0.1, 0.1]),
            SHUFFLE_METHOD: method([0.0, 0.0]),
            REFINED_R1_METHOD: method([0.5, 0.6]),
            R1_HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD: method([0.2, 0.3]),
            R1_SOFT_TYPE_MEAN_METHOD: method([0.3, 0.4]),
        },
    }
    refined = tmp_path / "refined.json"
    refined.touch()
    optional = {name: None for name in OPTIONAL_ARTIFACTS}
    optional["refined_predictions"] = refined
    plan = DeepBenchPlan(
        source_path=tmp_path / "plan.yaml",
        source_sha256="0" * 64,
        name="snpatho_deepbench_v1",
        status="retrospective_diagnostic",
        historical_result_name="snpatho_locked_v0_2",
        frozen_plan=tmp_path / "locked-plan.json",
        frozen_plan_sha256="1" * 64,
        historical_report=tmp_path / "locked-report.json",
        historical_report_sha256="2" * 64,
        sample_ids=("4066",),
        minimum_nuclei=3,
        bootstrap_iterations=4,
        final_cell_record_shuffle_permutations=100,
        primary_seeds=(17, 41, 89, 131, 197),
        optional_artifacts=optional,
        optional_artifact_sha256={name: None for name in OPTIONAL_ARTIFACTS},
        specification={},
    )
    native = NativeScanviManifest(
        path=tmp_path / "native.json",
        sha256="a" * 64,
        latent_space_id="latent",
        expression_space_id="expression",
        native_model_sha256="b" * 64,
        decoder_sha256="c" * 64,
        annotation_status="published_integrated_annotation_sensitivity_not_clean_reannotation",
        specimen_prototype_sha256={"4066": "d" * 64},
    )

    result = _primary_diagnostic([case], plan, native_scanvi=native)

    assert result["requested_primary_contrast"] == REQUESTED_PRIMARY_CONTRAST
    assert result["requested_primary_contrast_requirement"]["status"] == (
        "passes_developmental_joint_contrast"
    )
    assert result["requested_primary_status"] == ("developmental_joint_contrast_only_not_primary")
    assert result["full_primary_evidence"]["eligible_for_full_primary_claim"] is False
    assert result["full_primary_evidence"]["gates"] == {
        "independent_clean_reannotation": False,
        "prespecified_five_seed_matrix": False,
        "scored_refinement_matrix_complete": False,
        "refinement_matrix_strict_ordering_pass": False,
        "required_negative_controls": False,
        "execution_provenance_verified": False,
        "required_followup_evidence_complete": False,
        "composition_adjusted_residuals_hash_validated": False,
        "required_he_tissue_fraction_qc_hash_validated": False,
        "calibrated_segmentation_confidence_hash_validated": False,
    }


def test_failed_refinement_matrix_cannot_unlock_full_primary(tmp_path: Path) -> None:
    def method(values: list[float], *, observed: bool = False) -> dict:
        per_gene: dict = {"spearman": values}
        if observed:
            per_gene["observed_mean"] = [1.0, 2.0]
        return {
            "per_gene": per_gene,
            "summary": {"median_gene_spearman": 0.4, "median_gene_mse": 0.5},
        }

    case = {
        "section_id": "4066",
        "methods": {
            PRIMARY_METHOD: method([0.4, 0.5], observed=True),
            TYPE_MEAN_METHOD: method([0.1, 0.1]),
            SHUFFLE_METHOD: method([0.0, 0.0]),
            REFINED_R1_METHOD: method([0.5, 0.6]),
            R1_HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD: method([0.2, 0.3]),
            R1_SOFT_TYPE_MEAN_METHOD: method([0.3, 0.4]),
        },
    }
    refined = tmp_path / "refined.json"
    refined.touch()
    optional = {name: None for name in OPTIONAL_ARTIFACTS}
    optional["refined_predictions"] = refined
    plan = DeepBenchPlan(
        source_path=tmp_path / "plan.yaml",
        source_sha256="0" * 64,
        name="snpatho_deepbench_v1",
        status="retrospective_diagnostic",
        historical_result_name="snpatho_locked_v0_2",
        frozen_plan=tmp_path / "locked-plan.json",
        frozen_plan_sha256="1" * 64,
        historical_report=tmp_path / "locked-report.json",
        historical_report_sha256="2" * 64,
        sample_ids=("4066",),
        minimum_nuclei=3,
        bootstrap_iterations=4,
        final_cell_record_shuffle_permutations=100,
        primary_seeds=(17, 41, 89, 131, 197),
        optional_artifacts=optional,
        optional_artifact_sha256={name: None for name in OPTIONAL_ARTIFACTS},
        specification={},
    )
    native = NativeScanviManifest(
        path=tmp_path / "native.json",
        sha256="a" * 64,
        latent_space_id="latent",
        expression_space_id="expression",
        native_model_sha256="b" * 64,
        decoder_sha256="c" * 64,
        annotation_status="independent_clean_reannotation_complete",
        specimen_prototype_sha256={"4066": "d" * 64},
        clean_reannotation_manifest_sha256="e" * 64,
        validated_clean_reannotation=True,
    )
    five_seed = FiveSeedPredictionManifest(
        path=tmp_path / "five-seed.json",
        sha256="e" * 64,
        predictions={},
        control_names=(
            "graph_shuffle",
            "image_shuffle",
            "no_graph",
            "prototype_only",
            "wrong_donor",
        ),
        execution_provenance_verified=True,
    )
    failed_matrix = RefinementMatrixSummary(
        path=tmp_path / "matrix.json",
        sha256="f" * 64,
        matrix_status="complete",
        strict_ordering_status="fail",
        samples=("4066",),
        seeds=plan.primary_seeds,
        control_seeds=(17, 41, 89),
        control_names=(
            "prototype_only",
            "image_shuffle",
            "graph_shuffle",
            "no_graph",
            "wrong_donor",
        ),
        requested_artifact_count=1,
        scored_artifact_count=1,
        overall_status="complete_ordering_failed",
        primary_evidence_status="complete",
        evidence_blocker_count=0,
        execution_provenance_blocker_count=0,
        execution_provenance_verified=True,
        wrong_donor_coverage_complete=True,
    )

    failed = _primary_diagnostic(
        [case],
        plan,
        native_scanvi=native,
        five_seed_predictions=five_seed,
        refinement_matrix_summary=failed_matrix,
    )

    assert failed["requested_primary_status"] == ("developmental_joint_contrast_only_not_primary")
    assert failed["requested_primary_status"] != "passes_full_primary_requirement"
    assert failed["developmental_seed17_joint_contrast"]["status"] == (
        "passes_developmental_joint_contrast"
    )
    assert failed["full_primary_evidence"]["gates"] == {
        "independent_clean_reannotation": True,
        "prespecified_five_seed_matrix": True,
        "scored_refinement_matrix_complete": True,
        "refinement_matrix_strict_ordering_pass": False,
        "required_negative_controls": True,
        "execution_provenance_verified": True,
        "required_followup_evidence_complete": True,
        "composition_adjusted_residuals_hash_validated": False,
        "required_he_tissue_fraction_qc_hash_validated": False,
        "calibrated_segmentation_confidence_hash_validated": False,
    }

    strict_matrix_without_support = _primary_diagnostic(
        [case],
        plan,
        native_scanvi=native,
        five_seed_predictions=five_seed,
        refinement_matrix_summary=replace(
            failed_matrix,
            strict_ordering_status="pass",
            overall_status="complete",
        ),
    )
    assert strict_matrix_without_support["requested_primary_status"] == (
        "developmental_joint_contrast_only_not_primary"
    )

    supported_plan = _with_primary_gate_support(tmp_path, plan)
    passing = _primary_diagnostic(
        [case],
        supported_plan,
        native_scanvi=native,
        five_seed_predictions=five_seed,
        refinement_matrix_summary=replace(
            failed_matrix,
            strict_ordering_status="pass",
            overall_status="complete",
        ),
    )
    assert passing["requested_primary_status"] == "passes_full_primary_requirement"
    assert passing["requested_primary_blockers"] == []
    assert passing["rules"]["composition_adjusted_residual_positive"] is True
    assert all(_primary_gate_support_status(supported_plan).values())

    composition_manifest = supported_plan.optional_artifacts["spot_composition_covariates"]
    assert composition_manifest is not None
    composition_manifest.write_text("{}\n", encoding="utf-8")
    tampered = _primary_diagnostic(
        [case],
        supported_plan,
        native_scanvi=native,
        five_seed_predictions=five_seed,
        refinement_matrix_summary=replace(
            failed_matrix,
            strict_ordering_status="pass",
            overall_status="complete",
        ),
    )
    assert tampered["requested_primary_status"] == ("developmental_joint_contrast_only_not_primary")
    assert (
        tampered["full_primary_evidence"]["gates"]["composition_adjusted_residuals_hash_validated"]
        is False
    )


def test_complete_failed_matrix_readiness_consumes_outputs_without_stale_absence_claims(
    tmp_path: Path,
) -> None:
    optional = {name: None for name in OPTIONAL_ARTIFACTS}
    for name in (
        "primary_ffpe_snpatho_reference_manifest",
        "refined_predictions",
        "five_seed_predictions",
        "refinement_matrix_summary",
        "native_scanvi_checkpoint",
    ):
        optional[name] = tmp_path / (name + ".json")
    plan = DeepBenchPlan(
        source_path=tmp_path / "plan.yaml",
        source_sha256="0" * 64,
        name="snpatho_deepbench_v1",
        status="retrospective_diagnostic",
        historical_result_name="snpatho_locked_v0_2",
        frozen_plan=tmp_path / "locked-plan.json",
        frozen_plan_sha256="1" * 64,
        historical_report=tmp_path / "locked-report.json",
        historical_report_sha256="2" * 64,
        sample_ids=("4066", "4399", "4411"),
        minimum_nuclei=3,
        bootstrap_iterations=4,
        final_cell_record_shuffle_permutations=100,
        primary_seeds=(17, 41, 89, 131, 197),
        optional_artifacts=optional,
        optional_artifact_sha256={name: None for name in OPTIONAL_ARTIFACTS},
        specification={},
    )
    native = NativeScanviManifest(
        path=tmp_path / "native.json",
        sha256="a" * 64,
        latent_space_id="latent",
        expression_space_id="expression",
        native_model_sha256="b" * 64,
        decoder_sha256="c" * 64,
        annotation_status="published_integrated_annotation_sensitivity_not_clean_reannotation",
        specimen_prototype_sha256={sample: "d" * 64 for sample in plan.sample_ids},
    )
    five_seed = FiveSeedPredictionManifest(
        path=tmp_path / "five-seed.json",
        sha256="e" * 64,
        predictions={},
        control_names=REFINEMENT_MATRIX_CONTROLS,
    )
    matrix = RefinementMatrixSummary(
        path=tmp_path / "matrix.json",
        sha256="f" * 64,
        matrix_status="complete",
        strict_ordering_status="fail",
        samples=plan.sample_ids,
        seeds=plan.primary_seeds,
        control_seeds=(17, 41, 89),
        control_names=REFINEMENT_MATRIX_CONTROLS,
        requested_artifact_count=93,
        scored_artifact_count=93,
        overall_status="complete_ordering_failed",
        primary_evidence_status="complete",
        evidence_blocker_count=0,
        execution_provenance_blocker_count=0,
        wrong_donor_coverage_complete=True,
        wrong_donor_pairing_count=6,
        expected_wrong_donor_pairing_count=6,
    )

    readiness = _readiness(
        plan,
        native_scanvi=native,
        refined_predictions_validated=True,
        five_seed_predictions=five_seed,
        refinement_matrix_summary=matrix,
    )
    rows = {row["component"]: row for row in readiness}

    assert rows["refinement_matrix_summary"]["status"] == (
        "consumed_provenance_validated_matrix_strict_ordering_failed"
    )
    assert "strict ordering fail" in rows["refinement_matrix_summary"]["reason"]
    assert rows["refinement_trajectory_and_ablations"]["status"] == (
        "partial_consumed_via_refinement_matrix"
    )
    assert rows["graph_sensitivity_and_rewiring"]["status"] == (
        "partial_consumed_via_refinement_matrix"
    )
    assert rows["seed_ensemble_stability"]["status"] == ("partial_consumed_performance_matrix_only")
    for component in (
        "wrong_donor_predictions",
        "image_shuffle_predictions",
        "graph_shuffle_predictions",
    ):
        assert rows[component]["status"] == ("consumed_via_provenance_validated_refinement_matrix")
    assert rows["no_geometry_predictions"]["status"] == (
        "partial_no_graph_consumed_via_refinement_matrix"
    )

    emitted_reasons = "\n".join(str(row["reason"]) for row in readiness)
    for stale_claim in (
        "Only seed 17 exists",
        "No round 1-4 predictions",
        "Wrong-donor HEIR predictions have not been generated",
        "Required shuffled-image-feature HEIR predictions are absent",
        "Required coordinate-shuffled graph predictions are absent",
        "no primary native-scANVI",
        "prototype-only prediction/scorer",
    ):
        assert stale_claim not in emitted_reasons
    assert "independent clean reannotation is absent" in emitted_reasons
    assert "generic-atlas RNA" in emitted_reasons
    assert "state omission" in emitted_reasons
    assert "reference downsampling" in emitted_reasons
    assert "map, anchor" in rows["seed_ensemble_stability"]["reason"]
    assert "have not been scored" in rows["seed_ensemble_stability"]["reason"]


def test_registered_optional_artifacts_do_not_make_requested_plan_ready(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "registered.npz"
    artifact.touch()
    registered = {name: artifact for name in OPTIONAL_ARTIFACTS}
    plan = DeepBenchPlan(
        source_path=tmp_path / "plan.yaml",
        source_sha256="0" * 64,
        name="snpatho_deepbench_v1",
        status="retrospective_diagnostic",
        historical_result_name="snpatho_locked_v0_2",
        frozen_plan=tmp_path / "locked-plan.json",
        frozen_plan_sha256="1" * 64,
        historical_report=tmp_path / "locked-report.json",
        historical_report_sha256="2" * 64,
        sample_ids=("4066",),
        minimum_nuclei=3,
        bootstrap_iterations=4,
        final_cell_record_shuffle_permutations=100,
        primary_seeds=(17,),
        optional_artifacts=registered,
        optional_artifact_sha256={name: "3" * 64 for name in OPTIONAL_ARTIFACTS},
        specification={},
    )
    case = {
        "section_id": "4066",
        "methods": {
            PRIMARY_METHOD: {
                "per_gene": {"spearman": [0.4, 0.2], "observed_mean": [1.0, 2.0]},
                "summary": {"median_gene_spearman": 0.3, "median_gene_mse": 0.5},
            },
            TYPE_MEAN_METHOD: {
                "per_gene": {"spearman": [0.1, 0.1]},
                "summary": {"median_gene_mse": 1.0},
            },
            SHUFFLE_METHOD: {"per_gene": {"spearman": [0.0, 0.0]}},
        },
    }

    readiness = _readiness(plan)
    diagnostic = _primary_diagnostic([case], plan)
    optional_statuses = {
        item["component"]: item["status"]
        for item in readiness
        if item["component"] in OPTIONAL_ARTIFACTS
    }

    assert optional_statuses["primary_ffpe_snpatho_reference_manifest"] == (
        "partial_consumed_retrospective_sensitivity"
    )
    assert optional_statuses["refined_predictions"] == ("registered_but_not_provenance_validated")
    assert optional_statuses["native_scanvi_checkpoint"] == (
        "registered_but_not_provenance_validated"
    )
    assert optional_statuses["refinement_matrix_summary"] == (
        "registered_but_not_provenance_validated"
    )
    assert {
        status
        for name, status in optional_statuses.items()
        if name
        not in {
            "primary_ffpe_snpatho_reference_manifest",
            "refined_predictions",
            "five_seed_predictions",
            "refinement_matrix_summary",
            "native_scanvi_checkpoint",
        }
    } == {"registered_not_implemented"}
    assert diagnostic["requested_primary_status"] == "not_testable_missing_report_methods"
    assert diagnostic["requested_primary_contrast_requirement"]["requires_both_contrasts"] is True
    assert (
        diagnostic["requested_primary_contrast_requirement"][
            "refined_beats_both_matched_ffpe_r1_baselines"
        ]
        is None
    )
    assert diagnostic["diagnostic_statistic"]["specimen_formula"] == (
        "median_g(rho_HEIR,g - rho_historical_integrated_hard_type_mean,g)"
    )
    assert diagnostic["specimens"][0][
        "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean"
    ] == pytest.approx(0.2)
    assert diagnostic["specimens"][0][
        "median_paired_per_gene_spearman_delta_vs_final_record_shuffle_draw_0"
    ] == pytest.approx(0.3)
    assert "median_gene_spearman_delta_vs_spatial_shuffle" not in diagnostic["specimens"][0]
    repeated_diagnostic = _primary_diagnostic(
        [case],
        plan,
        repeated_shuffle_statistics={"4066": np.linspace(-0.1, 0.2, 100)},
    )
    repeated_comparison = repeated_diagnostic["specimens"][0][
        "repeated_final_record_shuffle_null_comparison"
    ]
    assert repeated_comparison["null_permutations"] == 100
    assert repeated_comparison["observed_heir_empirical_percentile_in_null"] == 1.0
    assert repeated_comparison["observed_heir_above_null_95_upper"] is True
    assert (
        repeated_diagnostic["rules"][
            "above_repeated_final_record_shuffle_null_95_upper_in_at_least_two_specimens"
        ]
        is False
    )
    assert not all(item["status"] == "ready" for item in readiness)


def test_bootstrap_positive_field_is_descriptive_not_probabilistic() -> None:
    result = _bootstrap_macro_delta(
        [np.asarray([-0.2, 0.1, 0.3, 0.4])],
        [np.asarray([1.0, 2.0, 3.0, 4.0])],
        iterations=20,
        seed=17,
    )

    assert "bootstrap_fraction_delta_positive" in result
    assert "probability_positive" not in result


def test_macro_summary_and_per_gene_tsv_preserve_biological_units(tmp_path: Path) -> None:
    cases = []
    for section_id, value in (("4066", 0.1), ("4399", 0.3), ("4411", 0.2)):
        cases.append(
            {
                "section_id": section_id,
                "methods": {
                    "method": {
                        "aggregation": "rna_mass",
                        "spots_evaluated": 10,
                        "spot_coverage": 1.0,
                        "summary": {"median_gene_spearman": value},
                        "per_gene": {
                            "gene_names": ["G1"],
                            "correlation_status": ["prediction_constant_scored_zero"],
                            "correlation_reason": ["prediction is constant"],
                            "spearman": [0.0],
                        },
                    }
                },
            }
        )
    macro = _method_macro_summaries(cases)
    assert macro["method"]["metrics"]["median_gene_spearman"] == {
        "macro_mean": pytest.approx(0.2),
        "minimum": pytest.approx(0.1),
        "maximum": pytest.approx(0.3),
        "specimens_evaluable": 3,
    }

    cases[0]["r1_reference_type_support"] = {
        "count_reference_supported_types": ["A", "B"],
        "prototype_supported_types": ["A"],
        "prototype_omitted_types": ["B"],
    }
    cases[0]["type_map_diagnostics"] = {
        "spots_evaluated": 10,
        "normalized_probability_entropy": {
            "mean": 0.2,
            "median": 0.1,
            "p95": 0.4,
            "minimum": 0.0,
            "maximum": 0.5,
        },
    }
    report = {
        "method_macro": macro,
        "cases": cases,
        "baseline_estimands": {TYPE_MEAN_METHOD: _baseline_estimands()[TYPE_MEAN_METHOD]},
    }
    tsv = tmp_path / "report.tsv"
    write_deepbench_report(report, json_path=tmp_path / "report.json", tsv_path=tsv)
    rows = tsv.read_text(encoding="utf-8").splitlines()

    assert any(row.startswith("macro\tmacro\tmethod") for row in rows)
    assert any(
        "\t4066\tmethod\trna_mass\tG1\tspearman\t0\t10\t"
        "prediction_constant_scored_zero\tprediction is constant" in row
        for row in rows
    )
    assert any(
        row.startswith("baseline_estimand\tall\thistorical_integrated_hard_type_mean")
        and "shared_soft_expected_type_median_library_size" in row
        for row in rows
    )
    assert any(
        row.startswith("type_support\t4066\tmatched_ffpe_r1_support_ladder\t\tB")
        and "\tprototype_omitted_types\t1\t" in row
        for row in rows
    )
    assert any(
        row.startswith("type_map\t4066\thistorical_type_probability_map")
        and "\tnormalized_probability_entropy_mean\t0.2\t" in row
        for row in rows
    )


def test_tsv_surfaces_repeated_shuffle_null_summary(tmp_path: Path) -> None:
    distribution = {
        "statistic": "median_gene_spearman",
        "permutations": 100,
        "mean": 0.01,
        "median": 0.02,
        "sample_standard_deviation": 0.03,
        "minimum": -0.04,
        "maximum": 0.05,
        "empirical_percentile_interval_95": {"lower": -0.03, "upper": 0.04},
    }
    comparison = {
        "observed_heir_median_gene_spearman": 0.06,
        "observed_heir_empirical_percentile_in_null": 0.99,
        "observed_heir_minus_null_median": 0.04,
        "observed_heir_above_null_95_upper": True,
        "null_permutations": 100,
    }
    report = {
        "method_macro": {},
        "cases": [],
        "final_cell_record_shuffle_null": {
            "specimens": {"4066": distribution},
            "equal_weight_specimen_macro": distribution,
        },
        "primary": {
            "specimens": [
                {
                    "section_id": "4066",
                    "repeated_final_record_shuffle_null_comparison": comparison,
                }
            ]
        },
    }
    tsv = tmp_path / "shuffle.tsv"

    write_deepbench_report(report, json_path=tmp_path / "shuffle.json", tsv_path=tsv)
    rows = tsv.read_text(encoding="utf-8").splitlines()

    assert any(
        row.startswith(
            "shuffle_null\t4066\t"
            "heir_final_cell_record_shuffle_historical_integrated_reference_library_size_weighted"
        )
        and "\tnull_empirical_95_upper\t0.04\t" in row
        for row in rows
    )
    assert any(
        row.startswith(
            "shuffle_null_comparison\t4066\t"
            "heir_round0_historical_integrated_reference_library_size_weighted"
        )
        and "\tobserved_heir_above_null_95_upper\t1\t" in row
        for row in rows
    )
