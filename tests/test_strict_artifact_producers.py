"""End-to-end smoke test for the strict initializer and frozen E-step producers."""

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import sparse

from heir.cli import main as heir_main
from heir.data import HistologyBag, PrototypeSet, RNAReference
from heir.models import HEIRConfig, HEIRModel
from heir.training import MolecularEStepArtifact, ValidatedInitializationReceipt

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / (name + ".py")
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


VALIDATE = _load_script("validate_initialization_checkpoint")
RECEIPT = _load_script("create_initialization_receipt")
E_STEP = _load_script("create_molecular_e_step")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    ("top_level_extra", "artifact_extra", "boolean_seed", "float_support"),
)
def test_initializer_plan_producer_matches_strict_schema(tmp_path: Path, mutation: str) -> None:
    artifact = {"path": "missing", "sha256": "a" * 64}
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
    plan = {
        "schema": VALIDATE.PLAN_SCHEMA,
        "status": "ready",
        "checkpoint": dict(artifact),
        "evaluation_artifact": dict(artifact),
        "label_source": dict(artifact),
        "latent_target_source": dict(artifact),
        "held_out_donors": ["held-out"],
        "seeds": [17, 41, 89],
        "thresholds": thresholds,
    }
    if mutation == "top_level_extra":
        plan["extra"] = True
    elif mutation == "artifact_extra":
        plan["checkpoint"]["extra"] = True
    elif mutation == "boolean_seed":
        plan["seeds"][0] = True
    else:
        plan["thresholds"]["minimum_per_type_support"] = 2.0
    plan_path = tmp_path / "invalid_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="schema|malformed|integer"):
        VALIDATE.main(["--plan", str(plan_path), "--output", str(tmp_path / "report.json")])


def test_strict_artifact_producers_create_bound_receipt_and_estep(tmp_path: Path) -> None:
    torch.manual_seed(7)
    morphology = np.asarray(
        [[-2.0, 2.0, 0.0]] * 4 + [[2.0, -2.0, 0.0]] * 4,
        dtype=np.float32,
    )
    independent_labels = np.asarray([0] * 4 + [1] * 4, dtype=np.int64)
    independent_latent = np.asarray([[-1.0, 0.0]] * 4 + [[1.0, 0.0]] * 4, dtype=np.float32)
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
        embedding, _, _ = model.encode_frozen_morphology(torch.from_numpy(morphology))
        design = torch.cat((embedding, torch.ones((len(embedding), 1))), dim=1)
        solution = torch.linalg.lstsq(design, torch.from_numpy(independent_latent)).solution
        model.prototype_query_head.weight.copy_(solution[:-1].T)
        model.prototype_query_head.bias.copy_(solution[-1])
    checkpoint = model.checkpoint()
    checkpoint["metadata"] = {
        "type_names": ["A", "B"],
        "gene_names": ["g0", "g1"],
        "feature_space_id": "pathology-test-v1",
        "latent_space_id": "latent-test-v1",
        "expression_space_id": "log1p-cpm-10000-v1",
        "training_donors": ["development-donor"],
    }
    checkpoint_path = tmp_path / "initializer.pt"
    torch.save(checkpoint, checkpoint_path)

    edge_index = np.empty((2, 0), dtype=np.int64)
    label_source = tmp_path / "independent_labels.npz"
    np.savez_compressed(
        label_source,
        schema=np.asarray("heir.independent_initialization_labels.v1"),
        nucleus_ids=np.asarray(["n%d" % index for index in range(8)]),
        donor_ids=np.asarray(["target-donor"] * 8),
        type_labels=independent_labels,
        type_names=np.asarray(["A", "B"]),
        independent_of_checkpoint=np.asarray(True),
    )
    latent_target_source = tmp_path / "registered_latent_targets.npz"
    np.savez_compressed(
        latent_target_source,
        schema=np.asarray("heir.registered_image_latent_targets.v1"),
        nucleus_ids=np.asarray(["n%d" % index for index in range(8)]),
        target_latent=independent_latent,
        latent_space_id=np.asarray("latent-test-v1"),
        independent_of_checkpoint=np.asarray(True),
    )
    evidence_path = tmp_path / "heldout_evidence.npz"
    np.savez_compressed(
        evidence_path,
        morphology=morphology,
        edge_index=edge_index,
        edge_weight=np.empty(0, dtype=np.float32),
        nucleus_ids=np.asarray(["n%d" % index for index in range(8)]),
        donor_ids=np.asarray(["target-donor"] * 8),
        type_labels=independent_labels,
        type_names=np.asarray(["A", "B"]),
        target_latent=independent_latent,
        feature_space_id=np.asarray("pathology-test-v1"),
        latent_space_id=np.asarray("latent-test-v1"),
        label_source_sha256=np.asarray(_digest(label_source)),
        latent_target_source_sha256=np.asarray(_digest(latent_target_source)),
        labels_independent_of_checkpoint=np.asarray(True),
        latent_targets_independent_of_checkpoint=np.asarray(True),
    )
    plan_path = tmp_path / "validation_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": VALIDATE.PLAN_SCHEMA,
                "status": "ready",
                "checkpoint": {"path": checkpoint_path.name, "sha256": _digest(checkpoint_path)},
                "evaluation_artifact": {
                    "path": evidence_path.name,
                    "sha256": _digest(evidence_path),
                },
                "label_source": {
                    "path": label_source.name,
                    "sha256": _digest(label_source),
                },
                "latent_target_source": {
                    "path": latent_target_source.name,
                    "sha256": _digest(latent_target_source),
                },
                "held_out_donors": ["target-donor"],
                "seeds": [17, 41, 89],
                "thresholds": {
                    "minimum_macro_f1": 0.65,
                    "minimum_image_shuffle_macro_f1_delta": 0.05,
                    "minimum_latent_cosine": 0.0,
                    "minimum_image_shuffle_latent_cosine_delta": 0.01,
                    "maximum_latent_rmse": 100.0,
                    "maximum_ece": 0.10,
                    "maximum_brier": 0.25,
                    "minimum_predicted_class_occupancy_fraction": 0.75,
                    "minimum_per_type_support": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    evidence_report = tmp_path / "validation_report.json"
    plan_bytes = plan_path.read_bytes()
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        VALIDATE.main(["--plan", str(plan_path), "--output", str(plan_path), "--device", "cpu"])
    assert plan_path.read_bytes() == plan_bytes
    checkpoint_bytes = checkpoint_path.read_bytes()
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        VALIDATE.main(
            ["--plan", str(plan_path), "--output", str(checkpoint_path), "--device", "cpu"]
        )
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    assert (
        VALIDATE.main(
            ["--plan", str(plan_path), "--output", str(evidence_report), "--device", "cpu"]
        )
        == 0
    )
    receipt_path = tmp_path / "initialization_receipt.json"
    report_bytes = evidence_report.read_bytes()
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        RECEIPT.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--evidence-report",
                str(evidence_report),
                "--output",
                str(evidence_report),
            ]
        )
    assert evidence_report.read_bytes() == report_bytes
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        RECEIPT.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--evidence-report",
                str(evidence_report),
                "--output",
                str(plan_path),
            ]
        )
    assert plan_path.read_bytes() == plan_bytes
    assert (
        RECEIPT.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--evidence-report",
                str(evidence_report),
                "--output",
                str(receipt_path),
            ]
        )
        == 0
    )
    receipt = ValidatedInitializationReceipt.load_json(receipt_path)
    assert receipt.held_out_donors == ("target-donor",)

    histology = HistologyBag(
        slide_id="target-slide",
        nucleus_ids=np.asarray(["n%d" % index for index in range(8)]),
        features=morphology,
        coordinates_um=np.column_stack((np.arange(8), np.zeros(8))),
        segmentation_confidence=np.ones(8, dtype=np.float32),
        artifact_probability=np.zeros(8, dtype=np.float32),
        edge_index=edge_index,
        edge_weight=np.empty(0, dtype=np.float32),
        sample_id="target-sample",
        donor_id="target-donor",
        block_id="target-block",
        feature_space_id="pathology-test-v1",
        histology_source_sha256="a" * 64,
        nuclei_source_sha256="b" * 64,
        feature_source_sha256="c" * 64,
    )
    histology_path = tmp_path / "histology.npz"
    histology.save_npz(histology_path)
    prototypes = PrototypeSet(
        prototype_ids=np.asarray(["pA", "pB"]),
        sample_ids=np.asarray(["target-sample", "target-sample"]),
        cell_type_labels=np.asarray(["A", "B"]),
        means=np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32),
        variances=np.ones((2, 2), dtype=np.float32),
        weights=np.asarray([0.5, 0.5], dtype=np.float32),
        n_cells=np.asarray([4, 4], dtype=np.int64),
        latent_space_id="latent-test-v1",
        donor_id="target-donor",
        block_id="target-block",
        latent_training_donors=("development-donor",),
    )
    prototypes_path = tmp_path / "prototypes.npz"
    prototypes.save_npz(prototypes_path)
    reference = RNAReference(
        sample_id="target-sample",
        cell_ids=np.asarray(["r%d" % index for index in range(4)]),
        gene_ids=np.asarray(["g0", "g1"]),
        counts=sparse.csr_matrix(np.ones((4, 2), dtype=np.float32)),
        latent=np.zeros((4, 2), dtype=np.float32),
        cell_type_labels=np.asarray(["A", "A", "B", "B"]),
        donor_ids=np.asarray(["target-donor"] * 4),
        sample_ids=np.asarray(["target-sample"] * 4),
        latent_space_id="latent-test-v1",
        block_id="target-block",
    )
    reference_path = tmp_path / "reference.npz"
    reference.save_npz(reference_path)
    prototypes = replace(
        prototypes,
        source_reference_sha256=_digest(reference_path),
    )
    prototypes.save_npz(prototypes_path)
    e_step_path = tmp_path / "molecular_e_step.npz"
    assert (
        E_STEP.main(
            [
                "--teacher-checkpoint",
                str(checkpoint_path),
                "--initialization-receipt",
                str(receipt_path),
                "--histology",
                str(histology_path),
                "--prototypes",
                str(prototypes_path),
                "--rna-reference",
                str(reference_path),
                "--output",
                str(e_step_path),
                "--device",
                "cpu",
                "--fixed-unknown-mass",
                "0.2",
                "--uot-epsilon",
                "0.5",
                "--uot-iterations",
                "500",
                "--uot-convergence-tolerance",
                "0.0001",
                "--uot-maximum-marginal-residual",
                "2.0",
            ]
        )
        == 0
    )
    artifact = MolecularEStepArtifact.load_npz(e_step_path)
    assert artifact.target_donor == "target-donor"
    assert artifact.teacher_training_donors == ("development-donor",)
    assert artifact.transport_plan.shape == (8, 3)

    validation_reference = replace(
        reference,
        counts=sparse.csr_matrix(np.full((4, 2), 2.0, dtype=np.float32)),
        library_sizes=np.full(4, 4.0, dtype=np.float64),
    )
    validation_reference_path = tmp_path / "reference_validation.npz"
    validation_reference.save_npz(validation_reference_path)
    validation_prototypes = replace(
        prototypes,
        source_reference_sha256=_digest(validation_reference_path),
    )
    validation_prototypes_path = tmp_path / "prototypes_validation.npz"
    validation_prototypes.save_npz(validation_prototypes_path)
    validation_e_step_path = tmp_path / "molecular_e_step_validation.npz"
    assert (
        E_STEP.main(
            [
                "--teacher-checkpoint",
                str(checkpoint_path),
                "--initialization-receipt",
                str(receipt_path),
                "--histology",
                str(histology_path),
                "--prototypes",
                str(validation_prototypes_path),
                "--rna-reference",
                str(validation_reference_path),
                "--output",
                str(validation_e_step_path),
                "--device",
                "cpu",
                "--fixed-unknown-mass",
                "0.2",
                "--uot-epsilon",
                "0.5",
                "--uot-iterations",
                "500",
                "--uot-convergence-tolerance",
                "0.0001",
                "--uot-maximum-marginal-residual",
                "2.0",
            ]
        )
        == 0
    )

    train_batch = tmp_path / "strict_train_batch.npz"
    validation_batch = tmp_path / "strict_validation_batch.npz"
    for batch_path, reference_input, prototype_input, e_step_input, bag_id in (
        (train_batch, reference_path, prototypes_path, e_step_path, "train"),
        (
            validation_batch,
            validation_reference_path,
            validation_prototypes_path,
            validation_e_step_path,
            "validation",
        ),
    ):
        assert (
            heir_main(
                [
                    "assemble-batch",
                    "--histology",
                    str(histology_path),
                    "--prototypes",
                    str(prototype_input),
                    "--reference",
                    str(reference_input),
                    "--molecular-e-step",
                    str(e_step_input),
                    "--output",
                    str(batch_path),
                    "--donor-id",
                    "target-donor",
                    "--block-id",
                    "target-block",
                    "--bag-id",
                    bag_id,
                ]
            )
            == 0
        )
    trained = tmp_path / "strict_random_initializer_control"
    assert (
        heir_main(
            [
                "train",
                "--train-batch",
                str(train_batch),
                "--validation-batch",
                str(validation_batch),
                "--output",
                str(trained),
                "--epochs",
                "1",
                "--graph-hidden-dim",
                "4",
                "--graph-output-dim",
                "3",
                "--graph-layers",
                "1",
                "--trunk-hidden-dims",
                "5",
                "--decoder-hidden-dims",
                "4",
                "--dropout",
                "0",
                "--uot-unknown-mass",
                "0.2",
                "--uninitialized-morphology-negative-control",
                "--allow-random-decoder",
                "--allow-split-overlap",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    trained_payload = torch.load(trained / "heir.pt", map_location="cpu", weights_only=True)
    assert trained_payload["metadata"]["molecular_e_step_mode"] == "strict_artifact"
    assert (
        "live_student_e_step_negative_control"
        not in trained_payload["metadata"]["exclusion_reasons"]
    )

    refined = tmp_path / "strict_fixed_artifact_refinement"
    assert (
        heir_main(
            [
                "refine",
                "--checkpoint",
                str(trained / "heir.pt"),
                "--train-batch",
                str(train_batch),
                "--validation-batch",
                str(validation_batch),
                "--output",
                str(refined),
                "--maximum-rounds",
                "1",
                "--broad-refinement-rounds",
                "0",
                "--epochs-per-round",
                "1",
                "--uot-unknown-mass",
                "0.2",
                "--allow-no-view-agreement",
                "--allow-split-overlap",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    refined_payload = torch.load(refined / "heir_refined.pt", map_location="cpu", weights_only=True)
    assert refined_payload["metadata"]["molecular_e_step_mode"] == "strict_artifact"
    assert refined_payload["metadata"]["refinement_validation_donors"] == ["target-donor"]
