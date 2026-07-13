"""Training-loop regression tests for donor and sample-level semantics."""

import hashlib
import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from heir.config import LossWeightConfig, OptimizationConfig
from heir.data import HistologyBag
from heir.losses import unbalanced_sinkhorn
from heir.models import HEIRConfig, HEIRModel
from heir.training import (
    HEIRTrainer,
    HEIRTrainingBatch,
    MolecularEStepArtifact,
    TrainingStage,
    aggregate_to_spots,
    array_content_sha256,
    frozen_transport_telemetry,
    ordered_identity_sha256,
    recompute_initialization_validation,
    spatial_block_split_masks,
    subset_histology_bag,
)
from heir.utils import set_seed


def _patch(cells: int, bag_id: str, shift: float) -> HEIRTrainingBatch:
    return HEIRTrainingBatch(
        morphology=torch.randn(cells, 3) + shift,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=None,
        prototype_means=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        prototype_variances=torch.ones(2, 2),
        prototype_types=torch.tensor([0, 1]),
        prototype_weights=torch.tensor([0.5, 0.5]),
        target_composition=torch.tensor([0.4, 0.6]),
        target_pseudobulk=torch.tensor([0.2, 0.3]),
        sample_id="sample-a",
        bag_id=bag_id,
        donor_id="donor-a",
        block_id="block-a",
        analysis_role="development",
        nucleus_ids=tuple("%s-n%d" % (bag_id, index) for index in range(cells)),
    )


def _trainer(
    allow_overlap: bool = True,
    *,
    molecular_e_step_mode: str = "live_student_negative_control",
    hard_anchor_routing: bool = False,
) -> HEIRTrainer:
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=3,
            num_cell_types=2,
            expression_dim=2,
            latent_dim=2,
            graph_hidden_dim=4,
            graph_output_dim=3,
            graph_layers=1,
            trunk_hidden_dims=(4,),
            decoder_hidden_dims=(4,),
            dropout=0.0,
            hard_type_routing=False,
        )
    )
    return HEIRTrainer(
        model,
        TrainingStage.PERSONALIZED,
        OptimizationConfig(
            epochs=1,
            bag_size=8,
            reference_batch_size=8,
            mixed_precision=False,
        ),
        LossWeightConfig(),
        device="cpu",
        allow_split_overlap=allow_overlap,
        molecular_e_step_mode=molecular_e_step_mode,
        hard_anchor_routing=hard_anchor_routing,
    )


def test_bernoulli_uot_gate_adds_each_route_cost_exactly_once_and_backpropagates() -> None:
    base_cost = torch.tensor([[0.25, 1.25], [2.0, 3.0]], requires_grad=True)
    unknown_probability = torch.tensor([0.2, 0.75], requires_grad=True)

    real_cost, dustbin_cost = HEIRTrainer._bernoulli_uot_costs(
        base_cost,
        unknown_probability,
    )

    expected_real = base_cost.detach() - torch.log1p(-unknown_probability.detach()).unsqueeze(1)
    expected_dustbin = -torch.log(unknown_probability.detach())
    torch.testing.assert_close(real_cost, expected_real)
    torch.testing.assert_close(dustbin_cost, expected_dustbin)

    (real_cost.sum() + dustbin_cost.sum()).backward()
    torch.testing.assert_close(base_cost.grad, torch.ones_like(base_cost))
    expected_probability_gradient = 2.0 / (1.0 - unknown_probability.detach()) - (
        1.0 / unknown_probability.detach()
    )
    torch.testing.assert_close(unknown_probability.grad, expected_probability_gradient)


def test_bernoulli_uot_gate_clamps_endpoint_probabilities_in_float32() -> None:
    base_cost = torch.zeros((2, 3), dtype=torch.float16, requires_grad=True)
    unknown_probability = torch.tensor([0.0, 1.0], dtype=torch.float16, requires_grad=True)

    real_cost, dustbin_cost = HEIRTrainer._bernoulli_uot_costs(
        base_cost,
        unknown_probability,
    )

    assert real_cost.dtype == torch.float32
    assert dustbin_cost.dtype == torch.float32
    assert torch.isfinite(real_cost).all()
    assert torch.isfinite(dustbin_cost).all()
    (real_cost.sum() + dustbin_cost.sum()).backward()
    assert base_cost.grad is not None and torch.isfinite(base_cost.grad).all()
    assert unknown_probability.grad is not None
    assert torch.isfinite(unknown_probability.grad).all()


def test_anchor_constraints_mask_local_routing_and_uot_responsibilities() -> None:
    trainer = _trainer(hard_anchor_routing=True)
    with torch.no_grad():
        trainer.model.unknown_head.weight.zero_()
        trainer.model.unknown_head.bias.fill_(-20.0)
    batch = replace(
        _patch(3, "anchors", 0.0),
        anchor_labels=torch.tensor([0, -100, 1]),
        anchor_weights=torch.tensor([1.0, 0.0, 1.0]),
    )
    output = trainer._forward_output(batch)

    assert output.prototype_mask[0].tolist() == [True, False]
    assert output.prototype_mask[2].tolist() == [False, True]
    assert output.prototype_probabilities[0, 1] == 0
    assert output.prototype_probabilities[2, 0] == 0
    responsibilities, result = trainer.transport_responsibilities(batch, output)
    assert not responsibilities.requires_grad
    assert responsibilities[0, 1] == 0
    assert responsibilities[2, 0] == 0
    torch.testing.assert_close(responsibilities.sum(dim=1), torch.ones(3))
    assert result.plan.shape == (3, 3)  # two measured states plus dustbin


def test_prototype_responsibilities_are_aggregated_to_detached_type_targets() -> None:
    responsibilities = torch.tensor(
        [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]],
        requires_grad=True,
    )
    types = torch.tensor([0, 0, 1])
    result = HEIRTrainer._type_responsibilities(responsibilities, types, 2)

    torch.testing.assert_close(result, torch.tensor([[0.5, 0.5], [0.7, 0.3]]))
    assert not result.requires_grad


def test_uot_unknown_mass_is_sample_estimated_with_prior_shrinkage() -> None:
    trainer = _trainer()
    trainer.uot_unknown_mass = 0.1
    trainer.uot_unknown_prior_strength = 2.0
    batch = replace(
        _patch(2, "unknown-mass", 0.0),
        unknown_targets=torch.tensor([0.0, 1.0]),
    )
    output = trainer._forward_output(batch)

    estimate = trainer._estimated_uot_unknown_mass(batch, output)
    # Beta-style prior contributes mass 2 * 0.1, then two observed cells
    # contribute one unknown-equivalent cell.
    torch.testing.assert_close(estimate, torch.tensor(0.3))


def test_uot_unknown_mass_is_fixed_without_independent_targets_by_default() -> None:
    trainer = _trainer()
    trainer.uot_unknown_mass = 0.2
    trainer.uot_unknown_prior_strength = 2.0
    batch = _patch(3, "fixed-unknown-mass", 0.0)
    with torch.no_grad():
        trainer.model.unknown_head.weight.zero_()
        trainer.model.unknown_head.bias.fill_(20.0)
    output = trainer._forward_output(batch)

    torch.testing.assert_close(
        trainer._estimated_uot_unknown_mass(batch, output),
        torch.tensor(0.2),
    )
    trainer.uot_unknown_mass_mode = "model_estimate"
    assert trainer._estimated_uot_unknown_mass(batch, output) > 0.6


def test_biological_weights_use_detached_uot_known_state_mass() -> None:
    base = torch.tensor([2.0, 3.0], requires_grad=True)
    responsibilities = torch.tensor(
        [[0.2, 0.3], [0.0, 0.0]],
        requires_grad=True,
    )

    weights = HEIRTrainer._uot_known_cell_weights(base, responsibilities)

    torch.testing.assert_close(weights, torch.tensor([1.0, 0.0]))
    weights.sum().backward()
    torch.testing.assert_close(base.grad, torch.tensor([0.5, 0.0]))
    assert responsibilities.grad is None


def test_molecular_posterior_receives_base_weights_before_known_mass(
    monkeypatch,
) -> None:
    trainer = _trainer()
    responsibilities = torch.tensor([[0.2, 0.3], [0.05, 0.05]])
    batch = replace(
        _patch(2, "posterior-known-mass", 0.0),
        cell_weights=torch.tensor([2.0, 3.0]),
        molecular_responsibilities=responsibilities,
    )
    output = trainer._forward_output(batch)
    captured = {}

    def capture(output_value, batch_value, responsibility_value, cell_weights):
        del batch_value
        captured["weights"] = cell_weights.detach().clone()
        zero = output_value.latent_mu.sum() * 0.0
        return zero, {}

    monkeypatch.setattr(trainer, "_molecular_posterior_loss", capture)
    trainer._output_loss(output, batch)

    torch.testing.assert_close(captured["weights"], torch.tensor([2.0, 3.0]))


def test_strict_fixed_responsibilities_skip_live_transport_and_direct_uot(
    monkeypatch,
) -> None:
    trainer = _trainer(molecular_e_step_mode="strict_artifact")
    batch = replace(
        _patch(3, "strict-m-step", 0.0),
        molecular_responsibilities=torch.tensor(
            [[0.8, 0.1], [0.2, 0.7], [0.4, 0.4]],
        ),
    )

    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError("strict M-step must not run live transport")

    monkeypatch.setattr(trainer, "transport_responsibilities", fail)
    monkeypatch.setattr("heir.losses.composite.unbalanced_sinkhorn", fail)
    output = trainer._forward_output(batch)
    loss, terms = trainer._output_loss(output, batch)
    loss.backward()

    assert float(terms["uot"].detach()) == 0.0
    assert "molecular_posterior/transport_unassigned" in terms
    assert trainer.model.fine_type_head.weight.grad is not None
    assert trainer.model.unknown_head.weight.grad is not None


def test_frozen_dustbin_mass_supervises_transport_unassignment_without_known_mass() -> None:
    trainer = _trainer(molecular_e_step_mode="strict_artifact")
    batch = replace(
        _patch(3, "strict-all-dustbin", 0.0),
        molecular_responsibilities=torch.zeros((3, 2)),
    )
    output = trainer._forward_output(batch)

    loss, terms = trainer._output_loss(output, batch)
    loss.backward()

    assert float(terms["molecular_posterior/routing"].detach()) == 0.0
    assert float(terms["molecular_posterior/type"].detach()) == 0.0
    assert float(terms["molecular_posterior/latent"].detach()) == 0.0
    assert float(terms["molecular_posterior/transport_unassigned"].detach()) > 0.0
    assert trainer.model.unknown_head.weight.grad is not None


def test_strict_personalized_loss_rejects_missing_frozen_estep() -> None:
    trainer = _trainer(molecular_e_step_mode="strict_artifact")
    batch = _patch(3, "missing-e-step", 0.0)
    with pytest.raises(ValueError, match="requires a frozen E-step artifact"):
        trainer._forward_loss(batch)


def test_strict_fit_rejects_biological_unknown_targets_on_transport_head() -> None:
    trainer = _trainer(molecular_e_step_mode="strict_artifact")
    training = replace(
        _patch(3, "train-biological-unknown", 0.0),
        unknown_targets=torch.zeros(3),
    )
    validation = _patch(3, "validation-biological-unknown", 0.0)
    with pytest.raises(ValueError, match="transport-unassigned head"):
        trainer.fit([training], [validation])


def test_strict_fit_uses_only_hash_bound_frozen_responsibilities(
    tmp_path,
    monkeypatch,
) -> None:
    trainer = _trainer(molecular_e_step_mode="strict_artifact")
    teacher = tmp_path / "teacher.pt"
    receipt = tmp_path / "receipt.json"
    torch.manual_seed(7)
    initializer = HEIRModel(
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
        trunk = initializer.trunk[0]
        trunk.weight.zero_()
        trunk.weight[:3, :3].copy_(torch.eye(3))
        trunk.bias.zero_()
        initializer.trunk[1].weight.fill_(1.0)
        initializer.trunk[1].bias.zero_()
        initializer.fine_type_head.weight.copy_(
            torch.tensor([[-5.0, 5.0, 0.0, 0.0, 0.0], [5.0, -5.0, 0.0, 0.0, 0.0]])
        )
        initializer.fine_type_head.bias.zero_()

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    plan = tmp_path / "plan.json"
    evidence_artifact = tmp_path / "evidence.npz"
    label_source = tmp_path / "labels.npz"
    latent_source = tmp_path / "latent.npz"
    evidence_ids = np.asarray(["e0", "e1", "e2", "e3"])
    evidence_donors = np.asarray(["target-donor"] * 4)
    evidence_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    evidence_morphology = np.asarray(
        [[-2.0, 2.0, 0.0]] * 2 + [[2.0, -2.0, 0.0]] * 2,
        dtype=np.float32,
    )
    evidence_latent = np.asarray(
        [[-1.0, 0.0]] * 2 + [[1.0, 0.0]] * 2,
        dtype=np.float32,
    )
    with torch.no_grad():
        embedding, _, _ = initializer.encode_frozen_morphology(
            torch.from_numpy(evidence_morphology)
        )
        direction = embedding[2] - embedding[0]
        latent_weight = 2.0 * direction / direction.square().sum()
        initializer.prototype_query_head.weight.zero_()
        initializer.prototype_query_head.bias.zero_()
        initializer.prototype_query_head.weight[0].copy_(latent_weight)
        initializer.prototype_query_head.bias[0].copy_(
            -1.0 - torch.dot(latent_weight, embedding[0])
        )
    checkpoint_payload = initializer.checkpoint()
    checkpoint_payload["metadata"] = {
        "type_names": ["A", "B"],
        "training_donors": ["development-donor"],
        "feature_space_id": "feature-test-v1",
        "latent_space_id": "latent-test-v1",
        "excluded_from_primary_claims": False,
        "exclusion_reasons": [],
    }
    torch.save(checkpoint_payload, teacher)
    np.savez_compressed(
        label_source,
        schema=np.asarray("heir.independent_initialization_labels.v1"),
        nucleus_ids=evidence_ids,
        donor_ids=evidence_donors,
        type_labels=evidence_labels,
        type_names=np.asarray(["A", "B"]),
        independent_of_checkpoint=np.asarray(True),
    )
    np.savez_compressed(
        latent_source,
        schema=np.asarray("heir.registered_image_latent_targets.v1"),
        nucleus_ids=evidence_ids,
        target_latent=evidence_latent,
        latent_space_id=np.asarray("latent-test-v1"),
        independent_of_checkpoint=np.asarray(True),
    )
    np.savez_compressed(
        evidence_artifact,
        morphology=evidence_morphology,
        edge_index=np.empty((2, 0), dtype=np.int64),
        nucleus_ids=evidence_ids,
        donor_ids=evidence_donors,
        type_labels=evidence_labels,
        type_names=np.asarray(["A", "B"]),
        target_latent=evidence_latent,
        feature_space_id=np.asarray("feature-test-v1"),
        latent_space_id=np.asarray("latent-test-v1"),
        label_source_sha256=np.asarray(digest(label_source)),
        latent_target_source_sha256=np.asarray(digest(latent_source)),
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
                "checkpoint": {"path": str(teacher), "sha256": digest(teacher)},
                "evaluation_artifact": {
                    "path": str(evidence_artifact),
                    "sha256": digest(evidence_artifact),
                },
                "label_source": {"path": str(label_source), "sha256": digest(label_source)},
                "latent_target_source": {
                    "path": str(latent_source),
                    "sha256": digest(latent_source),
                },
                "held_out_donors": ["target-donor"],
                "seeds": seeds,
                "thresholds": thresholds,
            }
        ),
        encoding="utf-8",
    )
    replay = recompute_initialization_validation(
        checkpoint=checkpoint_payload,
        morphology=evidence_morphology,
        edge_index=np.empty((2, 0), dtype=np.int64),
        edge_weight=None,
        labels=evidence_labels,
        target_latent=evidence_latent,
        donor_ids=evidence_donors,
        seeds=seeds,
    )
    metrics = replay["metrics"]
    evidence_report = tmp_path / "evidence_report.json"
    evidence_report.write_text(
        json.dumps(
            {
                "schema": "heir.initialization_validation_evidence.v1",
                "status": "complete",
                "pass": True,
                "checkpoint": {"path": str(teacher), "sha256": digest(teacher)},
                "plan": {"path": str(plan), "sha256": digest(plan)},
                "evidence_artifact": {
                    "path": str(evidence_artifact),
                    "sha256": digest(evidence_artifact),
                },
                "label_source": {
                    "path": str(label_source),
                    "sha256": digest(label_source),
                },
                "latent_target_source": {
                    "path": str(latent_source),
                    "sha256": digest(latent_source),
                },
                "feature_space_id": "feature-test-v1",
                "latent_space_id": "latent-test-v1",
                "type_ontology_sha256": ordered_identity_sha256(("A", "B")),
                "training_donors": ["development-donor"],
                "held_out_donors": ["target-donor"],
                "capabilities": {"broad_type": True, "image_to_latent": True},
                "thresholds": thresholds,
                "metrics": metrics,
                "donor_metrics": replay["donor_metrics"],
                "shuffle_controls": replay["shuffle_controls"],
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
    receipt.write_text(
        json.dumps(
            {
                "schema": "heir.validated_initialization.v1",
                "status": "complete",
                "pass": True,
                "checkpoint_sha256": digest(teacher),
                "feature_space_id": "feature-test-v1",
                "latent_space_id": "latent-test-v1",
                "type_ontology_sha256": ordered_identity_sha256(("A", "B")),
                "training_donors": ["development-donor"],
                "held_out_donors": ["target-donor"],
                "capabilities": {"broad_type": True, "image_to_latent": True},
                "evidence_report": str(evidence_report),
                "evidence_report_sha256": digest(evidence_report),
            }
        ),
        encoding="utf-8",
    )

    def frozen_batch(name: str, reference_payload: bytes) -> HEIRTrainingBatch:
        base = replace(
            _patch(3, name, 0.0),
            donor_id="target-donor",
            type_names=("A", "B"),
            gene_names=("g0", "g1"),
            prototype_ids=("p0", "p1"),
            latent_space_id="latent-test-v1",
            feature_space_id="feature-test-v1",
        )
        histology = tmp_path / (name + "-histology.npz")
        prototypes = tmp_path / (name + "-prototypes.npz")
        reference = tmp_path / (name + "-reference.npz")
        histology.write_bytes((name + "-histology").encode())
        prototypes.write_bytes((name + "-prototypes").encode())
        reference.write_bytes(reference_payload)
        source_mass = np.ones(3, dtype=np.float32)
        with torch.inference_mode():
            _, type_probabilities, image_latent = initializer.encode_frozen_morphology(
                base.morphology,
                base.edge_index,
                base.edge_weight,
            )
            variance = base.prototype_variances.clamp_min(
                initializer.config.prototype_variance_floor
            )
            gaussian_cost = 0.5 * (
                (image_latent.unsqueeze(1) - base.prototype_means.unsqueeze(0)).square()
                / variance.unsqueeze(0)
                + variance.unsqueeze(0).log()
            ).mean(dim=2)
            type_cost = (
                -type_probabilities.index_select(1, base.prototype_types).clamp_min(1.0e-8).log()
            )
            known_cost = gaussian_cost + type_cost
            transport = unbalanced_sinkhorn(
                known_cost,
                source_mass=torch.from_numpy(source_mass),
                target_mass=base.prototype_weights,
                epsilon=0.5,
                marginal_relaxation=1.0,
                iterations=500,
                convergence_tolerance=1.0e-4,
                unknown_mass=0.05,
                unknown_cost=1.0,
                add_unknown=True,
            )
        assert bool(transport.converged)
        raw_plan = transport.plan.float().numpy()
        plan = raw_plan / raw_plan.sum(axis=1, keepdims=True)
        cost = (
            torch.cat((known_cost, known_cost.new_full((len(known_cost), 1), 1.0)), dim=1)
            .float()
            .numpy()
        )
        desired_target = np.asarray([0.475, 0.475, 0.05], dtype=np.float64)
        realized_target = (plan / 3.0).sum(axis=0, dtype=np.float64)
        target_marginal_residual = float(np.abs(realized_target - desired_target).sum())
        source_marginal_residual = float(np.max(np.abs(plan.sum(axis=1) - 1.0)))
        telemetry = frozen_transport_telemetry(
            raw_transport_plan=raw_plan,
            transport_cost=cost,
            source_mass=source_mass,
            target_weights=base.prototype_weights.numpy(),
            fixed_unknown_mass=0.05,
            epsilon=0.5,
            marginal_relaxation=1.0,
        )
        artifact = MolecularEStepArtifact(
            transport_plan=plan,
            raw_transport_plan=raw_plan,
            transport_cost=cost,
            source_mass=source_mass,
            nucleus_ids=base.nucleus_ids,
            prototype_ids=base.prototype_ids,
            source_artifacts=(str(histology), str(prototypes), str(reference)),
            source_sha256=(digest(histology), digest(prototypes), digest(reference)),
            source_roles=("histology", "prototype_bank", "rna_reference"),
            teacher_checkpoint=str(teacher),
            teacher_checkpoint_sha256=digest(teacher),
            initialization_receipt=str(receipt),
            initialization_receipt_sha256=digest(receipt),
            teacher_role="independent_crossmodal_bridge",
            teacher_training_donors=("development-donor",),
            target_donor="target-donor",
            feature_space_id=base.feature_space_id,
            latent_space_id=base.latent_space_id,
            type_ontology_sha256=ordered_identity_sha256(base.type_names),
            morphology_sha256=array_content_sha256(base.morphology.numpy()),
            prototype_means_sha256=array_content_sha256(base.prototype_means.numpy()),
            prototype_variances_sha256=array_content_sha256(base.prototype_variances.numpy()),
            prototype_types_sha256=array_content_sha256(base.prototype_types.numpy()),
            prototype_weights_sha256=array_content_sha256(base.prototype_weights.numpy()),
            image_latent_sha256=array_content_sha256(image_latent.float().numpy()),
            type_probabilities_sha256=array_content_sha256(type_probabilities.float().numpy()),
            transport_cost_sha256=array_content_sha256(cost),
            source_mass_sha256=array_content_sha256(source_mass),
            artifact_threshold=0.5,
            type_cost_weight=1.0,
            unknown_cost=1.0,
            fixed_unknown_mass=0.05,
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
        artifact_path = tmp_path / (name + "-e-step.npz")
        artifact.save_npz(artifact_path)
        return replace(
            base,
            molecular_responsibilities=torch.from_numpy(artifact.responsibilities),
            weak_target_scope_id="sha256:" + digest(reference),
            weak_target_granularity="complete_rna_specimen",
            source_artifacts=(str(artifact_path),),
            source_sha256=(digest(artifact_path),),
            source_roles=("frozen_e_step",),
            molecular_training_donors=("development-donor",),
        )

    training = frozen_batch("training", b"training specimen RNA")
    validation = frozen_batch("validation", b"validation specimen RNA")

    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError("strict fit must not execute live transport")

    monkeypatch.setattr(trainer, "transport_responsibilities", fail)
    monkeypatch.setattr("heir.losses.composite.unbalanced_sinkhorn", fail)
    result = trainer.fit([training], [validation])

    assert result.best_epoch == 0


def test_strict_split_rejects_reused_specimen_weak_targets() -> None:
    train = replace(
        _patch(3, "train-scope", 0.0),
        weak_target_scope_id="sha256:" + "a" * 64,
        weak_target_granularity="complete_rna_specimen",
    )
    validation = replace(
        _patch(3, "validation-scope", 0.0),
        weak_target_scope_id=train.weak_target_scope_id,
        weak_target_granularity="complete_rna_specimen",
    )

    with pytest.raises(ValueError, match="reuse complete-specimen molecular targets"):
        HEIRTrainer._validate_weak_target_split([train], [validation])


def test_batch_rejects_unverifiable_weak_target_scope_identifier() -> None:
    batch = replace(_patch(3, "bad-scope", 0.0), weak_target_scope_id="same-specimen")
    with pytest.raises(ValueError, match="sha256"):
        batch.validate(TrainingStage.PERSONALIZED)


def test_uot_responsibilities_preserve_dustbin_row_mass() -> None:
    trainer = _trainer()
    trainer.uot_unknown_prior_strength = 0.01
    batch = _patch(3, "unknown-routing", 0.0)
    with torch.no_grad():
        trainer.model.unknown_head.weight.zero_()
        trainer.model.unknown_head.bias.fill_(20.0)
    output = trainer._forward_output(batch)
    responsibilities, result = trainer.transport_responsibilities(batch, output)

    assert result.plan.shape[1] == responsibilities.shape[1] + 1
    assert torch.all(responsibilities.sum(dim=1) < 0.05)
    assert bool(result.converged)
    assert result.iterations_run <= trainer.uot_iterations


def test_spot_aggregation_supports_known_and_rna_mass_weighting() -> None:
    expression = torch.log1p(torch.tensor([[9.0, 1.0], [1.0, 5.0]]))
    assignment = torch.ones((1, 2))
    equal = aggregate_to_spots(expression, assignment)
    weighted = aggregate_to_spots(
        expression,
        assignment,
        cell_rna_mass=torch.tensor([1.0, 3.0]),
    )
    torch.testing.assert_close(equal, torch.log1p(torch.tensor([[5.0, 3.0]])))
    torch.testing.assert_close(weighted, torch.log1p(torch.tensor([[3.0, 4.0]])))


def test_sample_losses_are_computed_once_after_graph_patch_merge() -> None:
    torch.manual_seed(11)
    trainer = _trainer()
    patches = (_patch(3, "left", -2.0), _patch(5, "right", 2.0))
    observed = trainer._epoch(patches, None)
    device_patches = [batch.to(trainer.device) for batch in patches]
    trainer.model.eval()
    with torch.no_grad():
        outputs = [trainer._forward_output(batch) for batch in device_patches]
        merged_batch = trainer._merge_sample_batches(device_patches)
        merged_output = trainer._concatenate_outputs(outputs)
        _, expected = trainer._output_loss(merged_output, merged_batch)
    assert observed["total"] == pytest.approx(float(expected["total"]), rel=1e-6)
    assert merged_batch.morphology.shape[0] == 8
    assert merged_batch.edge_index.shape == (2, 0)


def test_patch_merge_preserves_aligned_source_roles() -> None:
    left = replace(
        _patch(2, "left-provenance", 0.0),
        source_artifacts=("/tmp/shared-reference.npz", "/tmp/left-estep.npz"),
        source_sha256=("1" * 64, "2" * 64),
        source_roles=("sample_assay", "frozen_e_step"),
    )
    right = replace(
        _patch(2, "right-provenance", 0.0),
        source_artifacts=("/tmp/shared-reference.npz", "/tmp/right-estep.npz"),
        source_sha256=("1" * 64, "3" * 64),
        source_roles=("sample_assay", "frozen_e_step"),
    )

    merged = HEIRTrainer._merge_sample_batches((left, right))

    assert len(merged.source_artifacts) == len(merged.source_sha256) == len(merged.source_roles)
    assert merged.source_roles.count("sample_assay") == 1
    assert merged.source_roles.count("frozen_e_step") == 2


def test_non_synthetic_fit_requires_explicit_donor_and_block() -> None:
    trainer = _trainer(allow_overlap=False)
    incomplete = replace(_patch(3, "missing", 0.0), donor_id="", block_id="")
    validation = replace(
        _patch(3, "validation", 0.0),
        sample_id="sample-b",
        donor_id="donor-b",
        block_id="block-b",
    )
    with pytest.raises(ValueError, match="explicit donor_id and block_id"):
        trainer.fit([incomplete], [validation])


def test_fit_tracks_optimizer_step_and_seed_configures_deterministic_cublas(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_seed(17)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    result = _trainer().fit([_patch(3, "train", 0.0)], [_patch(3, "validation", 0.0)])
    assert result.history[0]["optimizer_step_skipped"] == 0.0


def test_spatial_patch_merge_unifies_cross_patch_spots_and_recomputes_pseudobulk() -> None:
    left = replace(
        _patch(3, "left", 0.0),
        analysis_role="pretraining",
        spot_ids=("s1", "s2"),
        spot_assignment=torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        target_spatial_expression=torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    )
    right = replace(
        _patch(5, "right", 0.0),
        analysis_role="pretraining",
        spot_ids=("s2", "s3"),
        spot_assignment=torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]]),
        target_spatial_expression=torch.tensor([[2.0, 2.0], [3.0, 3.0]]),
    )
    merged = HEIRTrainer._merge_sample_batches((left, right))
    assert merged.spot_ids == ("s1", "s2", "s3")
    torch.testing.assert_close(merged.spot_assignment.sum(dim=1), torch.tensor([2.0, 3.0, 3.0]))
    expected = torch.log1p(
        (
            torch.expm1(torch.tensor(1.0)) * 2
            + torch.expm1(torch.tensor(2.0)) * 3
            + torch.expm1(torch.tensor(3.0)) * 3
        )
        / 8
    )
    torch.testing.assert_close(merged.target_pseudobulk, torch.full((2,), expected))

    conflicting = replace(
        right,
        target_spatial_expression=torch.tensor([[9.0, 9.0], [3.0, 3.0]]),
    )
    with pytest.raises(ValueError, match="disagree on spatial target"):
        HEIRTrainer._merge_sample_batches((left, conflicting))


def test_spatial_block_split_is_disjoint_and_reindexes_edges() -> None:
    coordinates = np.asarray(
        [[10.0, 10.0], [20.0, 10.0], [610.0, 10.0], [620.0, 10.0]],
        dtype=np.float64,
    )
    bag = HistologyBag(
        slide_id="slide",
        nucleus_ids=np.asarray(["n0", "n1", "n2", "n3"]),
        features=np.arange(8, dtype=np.float32).reshape(4, 2),
        coordinates_um=coordinates,
        edge_index=np.asarray([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        edge_weight=np.ones(6, dtype=np.float32),
    )
    training, validation = spatial_block_split_masks(
        coordinates,
        validation_fraction=0.5,
        block_size_um=512.0,
        seed=17,
    )
    assert not np.any(training & validation)
    assert np.all(training | validation)
    train_bag = subset_histology_bag(bag, training)
    validation_bag = subset_histology_bag(bag, validation)
    assert set(train_bag.nucleus_ids.tolist()).isdisjoint(validation_bag.nucleus_ids.tolist())
    assert train_bag.edge_index.max(initial=-1) < train_bag.n_nuclei
    assert validation_bag.edge_index.max(initial=-1) < validation_bag.n_nuclei
