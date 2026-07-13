"""Regression tests for strict initialization and molecular E/M contracts."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from heir.losses import unbalanced_sinkhorn
from heir.models import HEIRConfig, HEIRModel
from heir.training import (
    MolecularEStepArtifact,
    ValidatedInitializationReceipt,
    array_content_sha256,
    frozen_transport_telemetry,
    ordered_identity_sha256,
    recompute_initialization_validation,
)


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_receipt(
    tmp_path,
    checkpoint,
    *,
    type_names=("A", "B"),
    feature_space_id="pathology-v1",
    latent_space_id="latent-v1",
    training_donors=("development-donor",),
    held_out_donors=("target-donor",),
    excluded=False,
    exclusion_metadata=None,
):
    if tuple(type_names) != ("A", "B"):
        raise ValueError("the tiny replayable receipt fixture uses the A/B ontology")
    torch.manual_seed(7)
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=3,
            num_cell_types=2,
            expression_dim=2,
            latent_dim=2,
            graph_hidden_dim=4,
            graph_output_dim=3,
            graph_layers=1,
            graph_mode="off",
            trunk_hidden_dims=(5,),
            decoder_hidden_dims=(4,),
            dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        trunk = model.trunk[0]
        trunk.weight.zero_()
        trunk.weight[:3, :3].copy_(torch.eye(3))
        trunk.bias.zero_()
        model.trunk[1].weight.fill_(1.0)
        model.trunk[1].bias.zero_()
        model.fine_type_head.weight.copy_(
            torch.tensor([[-5.0, 5.0, 0.0, 0.0, 0.0], [5.0, -5.0, 0.0, 0.0, 0.0]])
        )
        model.fine_type_head.bias.zero_()
    plan = tmp_path / "initialization_plan.json"
    evidence_artifact = tmp_path / "initialization_evidence.npz"
    label_source = tmp_path / "independent_labels.npz"
    latent_source = tmp_path / "registered_latent.npz"
    type_count = len(type_names)
    labels = np.tile(np.repeat(np.arange(type_count, dtype=np.int64), 2), len(held_out_donors))
    donor_ids = np.repeat(
        np.asarray(held_out_donors),
        type_count * 2,
    )
    nucleus_ids = np.asarray(["n%d" % index for index in range(len(labels))])
    morphology = np.where(
        labels[:, None] == 0,
        np.asarray([[-2.0, 2.0, 0.0]], dtype=np.float32),
        np.asarray([[2.0, -2.0, 0.0]], dtype=np.float32),
    ).astype(np.float32)
    target_latent = np.where(
        labels[:, None] == 0,
        np.asarray([[-1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    ).astype(np.float32)
    with torch.no_grad():
        embedding, _, _ = model.encode_frozen_morphology(torch.from_numpy(morphology))
        first = embedding[labels == 0][0]
        second = embedding[labels == 1][0]
        direction = second - first
        latent_weight = 2.0 * direction / direction.square().sum()
        model.prototype_query_head.weight.zero_()
        model.prototype_query_head.bias.zero_()
        model.prototype_query_head.weight[0].copy_(latent_weight)
        model.prototype_query_head.bias[0].copy_(-1.0 - torch.dot(latent_weight, first))
    checkpoint_payload = model.checkpoint()
    checkpoint_payload["metadata"] = {
        "type_names": list(type_names),
        "training_donors": list(training_donors),
        "feature_space_id": feature_space_id,
        "latent_space_id": latent_space_id,
        **(
            {
                "excluded_from_primary_claims": bool(excluded),
                "exclusion_reasons": (["test_exclusion"] if excluded else []),
            }
            if exclusion_metadata is None
            else exclusion_metadata
        ),
    }
    torch.save(checkpoint_payload, checkpoint)
    np.savez_compressed(
        label_source,
        schema=np.asarray("heir.independent_initialization_labels.v1"),
        nucleus_ids=nucleus_ids,
        donor_ids=donor_ids,
        type_labels=labels,
        type_names=np.asarray(type_names),
        independent_of_checkpoint=np.asarray(True),
    )
    np.savez_compressed(
        latent_source,
        schema=np.asarray("heir.registered_image_latent_targets.v1"),
        nucleus_ids=nucleus_ids,
        target_latent=target_latent,
        latent_space_id=np.asarray(latent_space_id),
        independent_of_checkpoint=np.asarray(True),
    )
    np.savez_compressed(
        evidence_artifact,
        morphology=morphology,
        edge_index=np.empty((2, 0), dtype=np.int64),
        nucleus_ids=nucleus_ids,
        donor_ids=donor_ids,
        type_labels=labels,
        type_names=np.asarray(type_names),
        target_latent=target_latent,
        feature_space_id=np.asarray(feature_space_id),
        latent_space_id=np.asarray(latent_space_id),
        label_source_sha256=np.asarray(_digest(label_source)),
        latent_target_source_sha256=np.asarray(_digest(latent_source)),
        labels_independent_of_checkpoint=np.asarray(True),
        latent_targets_independent_of_checkpoint=np.asarray(True),
    )
    seeds = [17, 41, 89]
    thresholds = {
        "minimum_macro_f1": 0.65,
        "minimum_image_shuffle_macro_f1_delta": 0.05,
        "minimum_latent_cosine": 0.0,
        "minimum_image_shuffle_latent_cosine_delta": 0.01,
        "maximum_latent_rmse": 1.0,
        "maximum_ece": 0.10,
        "maximum_brier": 0.25,
        "minimum_predicted_class_occupancy_fraction": 0.75,
        "minimum_per_type_support": 2,
    }
    plan.write_text(
        json.dumps(
            {
                "schema": "heir.initialization_validation_plan.v1",
                "status": "ready",
                "checkpoint": {"path": str(checkpoint), "sha256": _digest(checkpoint)},
                "evaluation_artifact": {
                    "path": str(evidence_artifact),
                    "sha256": _digest(evidence_artifact),
                },
                "label_source": {"path": str(label_source), "sha256": _digest(label_source)},
                "latent_target_source": {
                    "path": str(latent_source),
                    "sha256": _digest(latent_source),
                },
                "held_out_donors": list(held_out_donors),
                "seeds": seeds,
                "thresholds": thresholds,
            }
        ),
        encoding="utf-8",
    )
    replay = recompute_initialization_validation(
        checkpoint=checkpoint_payload,
        morphology=morphology,
        edge_index=np.empty((2, 0), dtype=np.int64),
        edge_weight=None,
        labels=labels,
        target_latent=target_latent,
        donor_ids=donor_ids,
        seeds=seeds,
    )
    metrics = replay["metrics"]
    donor_metrics = replay["donor_metrics"]
    shuffle_controls = replay["shuffle_controls"]
    report = tmp_path / "initialization_evidence_report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "heir.initialization_validation_evidence.v1",
                "status": "complete",
                "pass": True,
                "checkpoint": {"path": str(checkpoint), "sha256": _digest(checkpoint)},
                "plan": {"path": str(plan), "sha256": _digest(plan)},
                "evidence_artifact": {
                    "path": str(evidence_artifact),
                    "sha256": _digest(evidence_artifact),
                },
                "label_source": {"path": str(label_source), "sha256": _digest(label_source)},
                "latent_target_source": {
                    "path": str(latent_source),
                    "sha256": _digest(latent_source),
                },
                "feature_space_id": feature_space_id,
                "latent_space_id": latent_space_id,
                "type_ontology_sha256": ordered_identity_sha256(type_names),
                "training_donors": list(training_donors),
                "held_out_donors": list(held_out_donors),
                "capabilities": {"broad_type": True, "image_to_latent": True},
                "thresholds": thresholds,
                "metrics": metrics,
                "donor_metrics": donor_metrics,
                "shuffle_controls": shuffle_controls,
                "checks": {
                    "macro_f1": True,
                    "image_shuffle_macro_f1_delta": True,
                    "latent_cosine": True,
                    "image_shuffle_latent_cosine_delta": True,
                    "latent_rmse": True,
                    "ece": True,
                    "brier": True,
                    "predicted_class_occupancy": True,
                    "per_type_support": True,
                },
                "execution": {"device": "cpu-float32", "seeds": seeds},
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "heir.validated_initialization.v1",
                "status": "complete",
                "pass": True,
                "checkpoint_sha256": _digest(checkpoint),
                "feature_space_id": feature_space_id,
                "latent_space_id": latent_space_id,
                "type_ontology_sha256": ordered_identity_sha256(type_names),
                "training_donors": list(training_donors),
                "held_out_donors": list(held_out_donors),
                "capabilities": {"broad_type": True, "image_to_latent": True},
                "evidence_report": str(report),
                "evidence_report_sha256": _digest(report),
            }
        ),
        encoding="utf-8",
    )
    return receipt


def _artifact(tmp_path) -> MolecularEStepArtifact:
    histology = tmp_path / "histology.npz"
    prototypes = tmp_path / "prototypes.npz"
    reference = tmp_path / "reference.npz"
    histology.write_bytes(b"histology")
    prototypes.write_bytes(b"prototypes")
    reference.write_bytes(b"reference")
    teacher = tmp_path / "teacher.pt"
    receipt = _write_valid_receipt(tmp_path, teacher)
    checkpoint = torch.load(teacher, map_location="cpu", weights_only=True)
    model = HEIRModel.from_checkpoint(checkpoint).to(dtype=torch.float32).eval()
    morphology = np.asarray([[-2.0, 2.0, 0.0], [2.0, -2.0, 0.0]], dtype=np.float32)
    edge_index = np.empty((2, 0), dtype=np.int64)
    means = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    variances = np.ones((2, 2), dtype=np.float32)
    prototype_types = np.asarray([0, 1], dtype=np.int64)
    prototype_weights = np.asarray([0.5, 0.5], dtype=np.float32)
    source_mass = np.ones(2, dtype=np.float32)
    with torch.inference_mode():
        _, type_probabilities, image_latent = model.encode_frozen_morphology(
            torch.from_numpy(morphology), torch.from_numpy(edge_index), None
        )
        variance = torch.from_numpy(variances).clamp_min(model.config.prototype_variance_floor)
        gaussian_cost = 0.5 * (
            (image_latent.unsqueeze(1) - torch.from_numpy(means).unsqueeze(0)).square()
            / variance.unsqueeze(0)
            + variance.unsqueeze(0).log()
        ).mean(dim=2)
        type_cost = (
            -type_probabilities.index_select(1, torch.from_numpy(prototype_types))
            .clamp_min(1.0e-8)
            .log()
        )
        known_cost = gaussian_cost + type_cost
        transport = unbalanced_sinkhorn(
            known_cost,
            source_mass=torch.from_numpy(source_mass),
            target_mass=torch.from_numpy(prototype_weights),
            epsilon=0.5,
            marginal_relaxation=1.0,
            iterations=500,
            convergence_tolerance=1.0e-4,
            unknown_mass=0.2,
            unknown_cost=1.0,
            add_unknown=True,
        )
    assert bool(transport.converged)
    raw_plan = transport.plan.float().numpy()
    row_mass = raw_plan.sum(axis=1, keepdims=True)
    plan = raw_plan / row_mass
    cost = (
        torch.cat((known_cost, known_cost.new_full((len(known_cost), 1), 1.0)), dim=1)
        .float()
        .numpy()
    )
    desired_target = np.asarray([0.4, 0.4, 0.2], dtype=np.float64)
    realized_target = (plan * 0.5).sum(axis=0, dtype=np.float64)
    source_marginal_residual = float(np.max(np.abs(plan.sum(axis=1) - 1.0)))
    target_marginal_residual = float(np.abs(realized_target - desired_target).sum())
    telemetry = frozen_transport_telemetry(
        raw_transport_plan=raw_plan,
        transport_cost=cost,
        source_mass=source_mass,
        target_weights=prototype_weights,
        fixed_unknown_mass=0.2,
        epsilon=0.5,
        marginal_relaxation=1.0,
    )
    return MolecularEStepArtifact(
        transport_plan=plan,
        raw_transport_plan=raw_plan,
        transport_cost=cost,
        source_mass=source_mass,
        nucleus_ids=("n0", "n1"),
        prototype_ids=("p0", "p1"),
        source_artifacts=(str(histology), str(prototypes), str(reference)),
        source_sha256=(_digest(histology), _digest(prototypes), _digest(reference)),
        source_roles=("histology", "prototype_bank", "rna_reference"),
        teacher_checkpoint=str(teacher),
        teacher_checkpoint_sha256=_digest(teacher),
        initialization_receipt=str(receipt),
        initialization_receipt_sha256=_digest(receipt),
        teacher_role="independent_crossmodal_bridge",
        teacher_training_donors=("development-donor",),
        target_donor="target-donor",
        feature_space_id="pathology-v1",
        latent_space_id="latent-v1",
        type_ontology_sha256=ordered_identity_sha256(("A", "B")),
        morphology_sha256=array_content_sha256(morphology),
        prototype_means_sha256=array_content_sha256(means),
        prototype_variances_sha256=array_content_sha256(variances),
        prototype_types_sha256=array_content_sha256(prototype_types),
        prototype_weights_sha256=array_content_sha256(prototype_weights),
        image_latent_sha256=array_content_sha256(image_latent.float().numpy()),
        type_probabilities_sha256=array_content_sha256(type_probabilities.float().numpy()),
        transport_cost_sha256=array_content_sha256(cost),
        source_mass_sha256=array_content_sha256(source_mass),
        artifact_threshold=0.5,
        type_cost_weight=1.0,
        unknown_cost=1.0,
        fixed_unknown_mass=0.2,
        uot_epsilon=0.5,
        uot_marginal_relaxation=1.0,
        uot_iterations=500,
        uot_iterations_run=transport.iterations_run,
        uot_convergence_tolerance=1.0e-4,
        uot_maximum_marginal_residual=2.0,
        converged=True,
        source_marginal_residual=source_marginal_residual,
        target_marginal_residual=target_marginal_residual,
        solver_source_marginal_error=telemetry["solver_source_marginal_error"],
        solver_target_marginal_error=telemetry["solver_target_marginal_error"],
        source_dual_residual=float(transport.source_dual_residual.item()),
        target_dual_residual=float(transport.target_dual_residual.item()),
        transport_objective=telemetry["transport_objective"],
    )


def test_molecular_e_step_roundtrip_is_byte_deterministic(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    first = tmp_path / "e_step_a.npz"
    second = tmp_path / "e_step_b.npz"

    artifact.save_npz(first)
    artifact.save_npz(second)

    assert first.read_bytes() == second.read_bytes()
    loaded = MolecularEStepArtifact.load_npz(first)
    np.testing.assert_array_equal(loaded.transport_plan, artifact.transport_plan)
    np.testing.assert_array_equal(loaded.responsibilities, artifact.responsibilities)


def test_molecular_e_step_rejects_reordered_or_stale_bindings(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    source_hashes = dict(zip(artifact.source_roles, artifact.source_sha256))
    common = {
        "prototype_ids": artifact.prototype_ids,
        "source_sha256_by_role": source_hashes,
        "target_donor": artifact.target_donor,
        "feature_space_id": artifact.feature_space_id,
        "latent_space_id": artifact.latent_space_id,
        "type_names": ("A", "B"),
        "morphology": np.asarray([[-2.0, 2.0, 0.0], [2.0, -2.0, 0.0]], dtype=np.float32),
        "edge_index": np.empty((2, 0), dtype=np.int64),
        "edge_weight": None,
        "prototype_means": np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "prototype_variances": np.ones((2, 2), dtype=np.float32),
        "prototype_types": np.asarray([0, 1], dtype=np.int64),
        "prototype_weights": np.asarray([0.5, 0.5], dtype=np.float32),
        "cell_weights": np.ones(2, dtype=np.float32),
        "artifact_threshold": 0.5,
    }

    artifact.validate_binding(nucleus_ids=artifact.nucleus_ids, **common)
    with pytest.raises(ValueError, match="nucleus order"):
        artifact.validate_binding(nucleus_ids=tuple(reversed(artifact.nucleus_ids)), **common)
    with pytest.raises(ValueError, match="source hashes"):
        artifact.validate_binding(
            nucleus_ids=artifact.nucleus_ids,
            **{**common, "source_sha256_by_role": {**source_hashes, "histology": "0" * 64}},
        )
    with pytest.raises(ValueError, match="raw transport objective"):
        replace(artifact, transport_objective=artifact.transport_objective + 1.0).validate_binding(
            nucleus_ids=artifact.nucleus_ids,
            **common,
        )


def test_molecular_e_step_rejects_teacher_target_donor_overlap(tmp_path) -> None:
    artifact = replace(
        _artifact(tmp_path),
        teacher_training_donors=("target-donor",),
    )
    with pytest.raises(ValueError, match="trained on the target donor"):
        artifact.validate()


def test_molecular_e_step_rejects_mutated_bound_tensor(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    source_hashes = dict(zip(artifact.source_roles, artifact.source_sha256))
    with pytest.raises(ValueError, match="morphology tensor content differs"):
        artifact.validate_binding(
            nucleus_ids=artifact.nucleus_ids,
            prototype_ids=artifact.prototype_ids,
            source_sha256_by_role=source_hashes,
            target_donor=artifact.target_donor,
            feature_space_id=artifact.feature_space_id,
            latent_space_id=artifact.latent_space_id,
            type_names=("A", "B"),
            morphology=np.asarray([[9.0, 2.0, 0.0], [2.0, -2.0, 0.0]], dtype=np.float32),
            edge_index=np.empty((2, 0), dtype=np.int64),
            edge_weight=None,
            prototype_means=np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            prototype_variances=np.ones((2, 2), dtype=np.float32),
            prototype_types=np.asarray([0, 1], dtype=np.int64),
            prototype_weights=np.asarray([0.5, 0.5], dtype=np.float32),
            cell_weights=np.ones(2, dtype=np.float32),
            artifact_threshold=0.5,
        )


def test_molecular_e_step_replay_rejects_forged_cost_and_self_consistent_plan(
    tmp_path,
) -> None:
    artifact = _artifact(tmp_path)
    common = {
        "nucleus_ids": artifact.nucleus_ids,
        "prototype_ids": artifact.prototype_ids,
        "source_sha256_by_role": dict(zip(artifact.source_roles, artifact.source_sha256)),
        "target_donor": artifact.target_donor,
        "feature_space_id": artifact.feature_space_id,
        "latent_space_id": artifact.latent_space_id,
        "type_names": ("A", "B"),
        "morphology": np.asarray([[-2.0, 2.0, 0.0], [2.0, -2.0, 0.0]], dtype=np.float32),
        "edge_index": np.empty((2, 0), dtype=np.int64),
        "edge_weight": None,
        "prototype_means": np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "prototype_variances": np.ones((2, 2), dtype=np.float32),
        "prototype_types": np.asarray([0, 1], dtype=np.int64),
        "prototype_weights": np.asarray([0.5, 0.5], dtype=np.float32),
        "cell_weights": np.ones(2, dtype=np.float32),
        "artifact_threshold": 0.5,
    }

    forged_cost = np.array(artifact.transport_cost, copy=True)
    forged_cost[0, 0] += 0.25
    with pytest.raises(ValueError, match="transport cost differs from teacher replay"):
        replace(
            artifact,
            transport_cost=forged_cost,
            transport_cost_sha256=array_content_sha256(forged_cost),
        ).validate_binding(**common)

    forged_raw = np.array(artifact.raw_transport_plan, copy=True)
    forged_raw[:, [0, 1]] = forged_raw[:, [1, 0]]
    forged_plan = forged_raw / forged_raw.sum(axis=1, keepdims=True)
    forged_telemetry = frozen_transport_telemetry(
        raw_transport_plan=forged_raw,
        transport_cost=artifact.transport_cost,
        source_mass=artifact.source_mass,
        target_weights=common["prototype_weights"],
        fixed_unknown_mass=artifact.fixed_unknown_mass,
        epsilon=artifact.uot_epsilon,
        marginal_relaxation=artifact.uot_marginal_relaxation,
    )
    desired_target = np.asarray([0.4, 0.4, 0.2], dtype=np.float64)
    realized_target = (forged_plan * 0.5).sum(axis=0, dtype=np.float64)
    with pytest.raises(ValueError, match="raw transport plan differs from Sinkhorn replay"):
        replace(
            artifact,
            raw_transport_plan=forged_raw,
            transport_plan=forged_plan,
            target_marginal_residual=float(np.abs(realized_target - desired_target).sum()),
            solver_source_marginal_error=forged_telemetry["solver_source_marginal_error"],
            solver_target_marginal_error=forged_telemetry["solver_target_marginal_error"],
            transport_objective=forged_telemetry["transport_objective"],
        ).validate_binding(**common)


def test_molecular_e_step_rejects_dustbin_or_solver_residual_mismatch(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    bad_plan = np.array(artifact.transport_plan, copy=True)
    bad_plan[:, :-1] *= 0.5 / bad_plan[:, :-1].sum(axis=1, keepdims=True)
    bad_plan[:, -1] = 0.5
    with pytest.raises(ValueError, match="dustbin marginal"):
        replace(
            artifact,
            transport_plan=bad_plan,
            raw_transport_plan=(bad_plan * artifact.raw_transport_plan.sum(axis=1, keepdims=True)),
        ).validate()
    with pytest.raises(ValueError, match="dual_residual exceeds"):
        replace(artifact, source_dual_residual=1.0e-3).validate()


def test_validated_initialization_receipt_binds_evidence_and_holdout(tmp_path) -> None:
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"checkpoint")
    receipt_path = _write_valid_receipt(tmp_path, checkpoint)

    receipt = ValidatedInitializationReceipt.load_json(receipt_path)
    receipt.validate_binding(
        checkpoint_sha256=_digest(checkpoint),
        feature_space_id="pathology-v1",
        latent_space_id="latent-v1",
        type_names=("A", "B"),
        target_donors=("target-donor",),
        receipt_path=receipt_path,
    )
    with pytest.raises(ValueError, match="did not hold out"):
        receipt.validate_binding(
            checkpoint_sha256=_digest(checkpoint),
            feature_space_id="pathology-v1",
            latent_space_id="latent-v1",
            type_names=("A", "B"),
            target_donors=("new-target",),
            receipt_path=receipt_path,
        )


def test_initialization_receipt_rejects_arbitrary_hash_bound_json(tmp_path) -> None:
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"checkpoint")
    receipt_path = _write_valid_receipt(tmp_path, checkpoint)
    receipt = ValidatedInitializationReceipt.load_json(receipt_path)
    arbitrary = tmp_path / "arbitrary.json"
    arbitrary.write_text('{"anything": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="schema is invalid"):
        replace(
            receipt,
            evidence_report=str(arbitrary),
            evidence_report_sha256=_digest(arbitrary),
        ).validate_binding(
            checkpoint_sha256=_digest(checkpoint),
            feature_space_id="pathology-v1",
            latent_space_id="latent-v1",
            type_names=("A", "B"),
            target_donors=("target-donor",),
            receipt_path=receipt_path,
        )


def test_initialization_receipt_recomputes_plan_bound_metric_gates(tmp_path) -> None:
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"checkpoint")
    receipt_path = _write_valid_receipt(tmp_path, checkpoint)
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    report_path = tmp_path / "initialization_evidence_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["donor_metrics"][0]["macro_f1"] = 0.10
    for row in report["shuffle_controls"]:
        if row["donor_id"] == report["donor_metrics"][0]["donor_id"]:
            row["real_minus_image_shuffle_macro_f1"] = -0.30
    report_path.write_text(json.dumps(report), encoding="utf-8")
    receipt_payload["evidence_report_sha256"] = _digest(report_path)
    receipt_path.write_text(json.dumps(receipt_payload), encoding="utf-8")

    receipt = ValidatedInitializationReceipt.load_json(receipt_path)
    with pytest.raises(ValueError, match="donor metrics differ from checkpoint replay"):
        receipt.validate_binding(
            checkpoint_sha256=_digest(checkpoint),
            feature_space_id="pathology-v1",
            latent_space_id="latent-v1",
            type_names=("A", "B"),
            target_donors=("target-donor",),
            receipt_path=receipt_path,
        )


def test_initialization_receipt_rejects_forged_pooled_model_metrics(tmp_path) -> None:
    checkpoint = tmp_path / "initial.pt"
    receipt_path = _write_valid_receipt(tmp_path, checkpoint)
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    report_path = tmp_path / "initialization_evidence_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["metrics"]["macro_f1"] = 0.91
    report_path.write_text(json.dumps(report), encoding="utf-8")
    receipt_payload["evidence_report_sha256"] = _digest(report_path)
    receipt_path.write_text(json.dumps(receipt_payload), encoding="utf-8")

    receipt = ValidatedInitializationReceipt.load_json(receipt_path)
    with pytest.raises(ValueError, match="pooled metrics differ from checkpoint replay"):
        receipt.validate_binding(
            checkpoint_sha256=_digest(checkpoint),
            feature_space_id="pathology-v1",
            latent_space_id="latent-v1",
            type_names=("A", "B"),
            target_donors=("target-donor",),
            receipt_path=receipt_path,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("permutation_sha256", "0" * 64, "shuffle permutation is not reproducible"),
        (
            "real_minus_image_shuffle_macro_f1",
            0.60,
            "shuffle metrics differ from checkpoint replay",
        ),
    ],
)
def test_initialization_receipt_recomputes_shuffle_control_integrity(
    tmp_path, field, value, message
) -> None:
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"checkpoint")
    receipt_path = _write_valid_receipt(tmp_path, checkpoint)
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    report_path = tmp_path / "initialization_evidence_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["shuffle_controls"][0][field] = value
    report_path.write_text(json.dumps(report), encoding="utf-8")
    receipt_payload["evidence_report_sha256"] = _digest(report_path)
    receipt_path.write_text(json.dumps(receipt_payload), encoding="utf-8")

    receipt = ValidatedInitializationReceipt.load_json(receipt_path)
    with pytest.raises(ValueError, match=message):
        receipt.validate_binding(
            checkpoint_sha256=_digest(checkpoint),
            feature_space_id="pathology-v1",
            latent_space_id="latent-v1",
            type_names=("A", "B"),
            target_donors=("target-donor",),
            receipt_path=receipt_path,
        )


def test_initialization_receipt_rejects_excluded_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"checkpoint")
    receipt_path = _write_valid_receipt(tmp_path, checkpoint, excluded=True)
    receipt = ValidatedInitializationReceipt.load_json(receipt_path)

    with pytest.raises(ValueError, match="excluded initialization checkpoint"):
        receipt.validate_binding(
            checkpoint_sha256=_digest(checkpoint),
            feature_space_id="pathology-v1",
            latent_space_id="latent-v1",
            type_names=("A", "B"),
            target_donors=("target-donor",),
            receipt_path=receipt_path,
        )


@pytest.mark.parametrize(
    "exclusion_metadata",
    [
        {"excluded_from_primary_claims": 0, "exclusion_reasons": []},
        {"excluded_from_primary_claims": False, "exclusion_reasons": ""},
        {"excluded_from_primary_claims": False, "exclusion_reasons": ["excluded"]},
    ],
)
def test_initialization_receipt_rejects_malformed_or_nonempty_exclusions(
    tmp_path, exclusion_metadata
) -> None:
    checkpoint = tmp_path / "initial.pt"
    receipt_path = _write_valid_receipt(
        tmp_path,
        checkpoint,
        exclusion_metadata=exclusion_metadata,
    )
    receipt = ValidatedInitializationReceipt.load_json(receipt_path)

    with pytest.raises(ValueError, match="exclusion|excluded"):
        receipt.validate_binding(
            checkpoint_sha256=_digest(checkpoint),
            feature_space_id="pathology-v1",
            latent_space_id="latent-v1",
            type_names=("A", "B"),
            target_donors=("target-donor",),
            receipt_path=receipt_path,
        )
