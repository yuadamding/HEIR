import csv
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import sparse

from heir import cli as cli_module
from heir.cli import (
    _assert_file_records_unchanged,
    _canonical_parent_exclusion_reasons,
    _freeze_file_records,
    _freeze_transitive_batch_source_records,
    _load_refinement_views,
    _log_normalize,
    _reject_output_input_collisions,
    _sha256,
    _structural_model_config,
    _upstream_exclusion_reasons,
    _validate_refinement_parent_validation_scope,
    _validate_residual_geometry_training_provenance,
    _wrong_donor_ontology_intersection,
    build_parser,
    main,
)
from heir.data import HistologyBag, PrototypeSet, RNAReference, SpatialTruthArtifact
from heir.data.manifest import MANIFEST_COLUMNS, ManifestRecord
from heir.inference import PredictionBundle
from heir.losses import unbalanced_sinkhorn
from heir.models import HEIRConfig, HEIRModel
from heir.models.rna import RNAVAE, RNAVAEConfig
from heir.prior import fit_rna_residual_geometry
from heir.training import (
    HEIRTrainingBatch,
    MolecularEStepArtifact,
    TrainingStage,
    array_content_sha256,
    frozen_transport_telemetry,
    ordered_identity_sha256,
    recompute_initialization_validation,
)
from heir.uncertainty import MahalanobisOOD


def test_refine_cli_defaults_use_two_round_trust_and_fixed_prior() -> None:
    args = build_parser().parse_args(
        [
            "refine",
            "--checkpoint",
            "checkpoint.pt",
            "--train-batch",
            "train.npz",
            "--validation-batch",
            "validation.npz",
            "--output",
            "refined",
        ]
    )
    assert args.maximum_rounds == 4
    assert args.broad_refinement_rounds == 2
    assert args.prior_old_weight == 1.0
    assert args.round_selection_mode == "fixed"
    assert args.maximum_validation_loss_degradation == pytest.approx(0.01)
    assert args.objective_relative_stability_tolerance == pytest.approx(0.01)
    assert args.objective_stability_tolerance is None
    assert not args.save_round_checkpoints
    assert args.teacher_ema == 0.0
    assert not args.require_scale_view_agreement
    assert not args.live_student_e_step_negative_control


def test_strict_refinement_accepts_exact_parent_validation_lineage() -> None:
    _validate_refinement_parent_validation_scope(
        validation_donors=("target",),
        checkpoint_donors=("bridge", "target"),
        parent_validation_donors=("target",),
        strict_fixed_artifact=True,
        allow_split_overlap=False,
    )
    with pytest.raises(ValueError, match="exact parent validation set"):
        _validate_refinement_parent_validation_scope(
            validation_donors=("target",),
            checkpoint_donors=("bridge", "target"),
            parent_validation_donors=("different",),
            strict_fixed_artifact=True,
            allow_split_overlap=False,
        )
    with pytest.raises(ValueError, match="trained on validation donors"):
        _validate_refinement_parent_validation_scope(
            validation_donors=("target",),
            checkpoint_donors=("bridge", "target"),
            parent_validation_donors=(),
            strict_fixed_artifact=False,
            allow_split_overlap=False,
        )


def test_train_cli_accepts_frozen_rna_residual_geometry() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--train-batch",
            "train.npz",
            "--validation-batch",
            "validation.npz",
            "--output",
            "trained",
            "--residual-geometry",
            "rna_geometry.npz",
        ]
    )
    assert args.residual_geometry == "rna_geometry.npz"
    assert not args.finetune_residual_basis
    assert not args.unsafe_allow_legacy_residual_geometry_provenance
    assert args.graph_mode == "off"
    assert not args.uninitialized_morphology_negative_control
    assert not args.live_student_e_step_negative_control


def test_upstream_exclusion_metadata_is_canonical_and_propagated() -> None:
    assert _upstream_exclusion_reasons({}, "rna") == []
    assert _upstream_exclusion_reasons(
        {
            "excluded_from_primary_claims": True,
            "exclusion_reasons": ["sensitivity"],
        },
        "rna",
    ) == ["rna:sensitivity"]
    for malformed in (
        {"excluded_from_primary_claims": 1, "exclusion_reasons": []},
        {"excluded_from_primary_claims": True, "exclusion_reasons": []},
        {"excluded_from_primary_claims": False, "exclusion_reasons": ["hidden"]},
        {"excluded_from_primary_claims": True, "exclusion_reasons": "hidden"},
    ):
        with pytest.raises(ValueError, match="exclusion"):
            _upstream_exclusion_reasons(malformed, "rna")


def test_frozen_input_record_rejects_mutation_after_load(tmp_path: Path) -> None:
    source = tmp_path / "batch.npz"
    source.write_bytes(b"loaded bytes")
    records = _freeze_file_records([str(source)], "test batch")
    _assert_file_records_unchanged(records, "test batch")

    source.write_bytes(b"mutated bytes")
    with pytest.raises(ValueError, match="changed after it was loaded"):
        _assert_file_records_unchanged(records, "test batch")


def test_refinement_parent_exclusions_are_canonical_and_preserved() -> None:
    assert _canonical_parent_exclusion_reasons(
        {
            "excluded_from_primary_claims": True,
            "exclusion_reasons": ["negative_control"],
        }
    ) == ("negative_control",)
    assert _canonical_parent_exclusion_reasons({}) == ()
    for metadata in (
        {"excluded_from_primary_claims": 0, "exclusion_reasons": []},
        {"excluded_from_primary_claims": True, "exclusion_reasons": []},
        {"excluded_from_primary_claims": False, "exclusion_reasons": ["hidden"]},
        {"excluded_from_primary_claims": True, "exclusion_reasons": "negative_control"},
    ):
        with pytest.raises(ValueError, match="primary-claim exclusion"):
            _canonical_parent_exclusion_reasons(metadata)


def test_transitive_batch_sources_are_frozen_and_rechecked(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"source bytes")
    batch = HEIRTrainingBatch(
        morphology=torch.zeros((2, 3)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=None,
        prototype_means=torch.zeros((2, 2)),
        prototype_variances=torch.ones((2, 2)),
        prototype_types=torch.tensor([0, 1]),
        prototype_weights=torch.tensor([0.5, 0.5]),
        target_composition=torch.tensor([0.5, 0.5]),
        target_pseudobulk=torch.zeros(2),
        source_artifacts=(str(source),),
        source_sha256=(_sha256(str(source)),),
        source_roles=("sample_assay",),
    )
    records = _freeze_transitive_batch_source_records([batch], "test transitive input")
    _assert_file_records_unchanged(records, "test transitive input")
    source.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed after it was loaded"):
        _assert_file_records_unchanged(records, "test transitive input")


def test_output_paths_cannot_overwrite_direct_or_transitive_inputs(tmp_path: Path) -> None:
    direct = tmp_path / "run" / "heir.pt"
    direct.parent.mkdir()
    direct.write_bytes(b"parent")
    records = _freeze_file_records([str(direct)], "parent checkpoint")
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        _reject_output_input_collisions([direct], records, label="training")
    hardlink = tmp_path / "hardlink-to-parent.pt"
    hardlink.hardlink_to(direct)
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        _reject_output_input_collisions([hardlink], records, label="training")

    transitive = tmp_path / "run" / "prototypes" / "donor__sample.npz"
    transitive.parent.mkdir()
    transitive.write_bytes(b"prototype")
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        _reject_output_input_collisions(
            [transitive],
            (),
            transitive_input_paths=[str(transitive)],
            label="refinement",
        )

    archive = tmp_path / "molecular.tar.gz"
    archive.write_bytes(b"archive")
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        _reject_output_input_collisions(
            [archive],
            (),
            transitive_input_paths=[str(archive) + "::matrix.mtx.gz"],
            label="refinement",
        )

    input_directory = tmp_path / "spaceranger-outs"
    input_directory.mkdir()
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        _reject_output_input_collisions(
            [input_directory / "filtered_feature_bc_matrix.h5"],
            (),
            transitive_input_paths=[str(input_directory)],
            label="refinement",
        )


@pytest.mark.parametrize(
    ("command", "output_name"),
    [("train", "heir.pt"), ("refine", "heir_refined.pt")],
)
def test_training_commands_reject_in_place_checkpoint_overwrite(
    tmp_path: Path, command: str, output_name: str, capsys
) -> None:
    destination = tmp_path / command
    destination.mkdir()
    checkpoint = destination / output_name
    checkpoint.write_bytes(b"bound input checkpoint")
    batch = tmp_path / (command + "_batch.npz")
    batch.write_bytes(b"not parsed because collision fails first")
    arguments = [
        command,
        "--checkpoint" if command == "refine" else "--initial-heir-checkpoint",
        str(checkpoint),
        "--train-batch",
        str(batch),
        "--validation-batch",
        str(batch),
        "--output",
        str(destination),
    ]
    if command == "train":
        receipt = tmp_path / "receipt.json"
        receipt.write_text("{}", encoding="utf-8")
        arguments.extend(["--initialization-receipt", str(receipt)])
    with pytest.raises(SystemExit):
        main(arguments)
    assert "output would overwrite a bound input" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "output_flag"),
    [
        ("evaluate", "--output"),
        ("evaluate-spatial", "--output"),
        ("evaluate-spatial", "--aggregates-output"),
    ],
)
def test_legacy_evaluators_reject_output_input_collisions(
    tmp_path: Path, command: str, output_flag: str, capsys
) -> None:
    predictions = tmp_path / "predictions.npz"
    truth = tmp_path / "truth.npz"
    predictions.write_bytes(b"collision checked before prediction parsing")
    truth.write_bytes(b"collision checked before truth parsing")
    with pytest.raises(SystemExit):
        main(
            [
                command,
                "--predictions",
                str(predictions),
                "--truth",
                str(truth),
                output_flag,
                str(predictions),
            ]
        )
    assert "output would overwrite a bound input" in capsys.readouterr().err


def test_initialization_compatibility_retains_unexposed_checkpoint_gate_semantics() -> None:
    base = HEIRConfig(
        morphology_dim=4,
        num_cell_types=2,
        expression_dim=4,
        latent_dim=2,
        graph_hidden_dim=4,
        graph_output_dim=4,
        graph_layers=1,
        graph_mode="distance_only",
        graph_context_gate_init=0.0,
        trunk_hidden_dims=(4,),
        decoder_hidden_dims=(4,),
    )
    historical = replace(
        base,
        graph_context_gate_init=1.0,
        residual_type_strategy="detached_max_hard",
        residual_type_concentration_threshold=0.7,
        residual_type_concentration_temperature=0.1,
    )
    structurally_different = replace(base, graph_mode="off")
    different_residual_rank = replace(base, residual_rank=1)

    assert _structural_model_config(base) == _structural_model_config(historical)
    assert _structural_model_config(base) != _structural_model_config(structurally_different)
    assert _structural_model_config(base) != _structural_model_config(different_residual_rank)


def test_predict_cli_exposes_prespecified_negative_controls() -> None:
    args = build_parser().parse_args(
        [
            "predict",
            "--checkpoint",
            "model.pt",
            "--histology",
            "histology.npz",
            "--prototypes",
            "prototypes.npz",
            "--genes",
            "genes.tsv",
            "--output",
            "predictions.npz",
            "--donor-id",
            "donor",
            "--prototype-only",
            "--image-feature-shuffle",
            "--graph-node-shuffle",
            "--wrong-donor-control",
        ]
    )
    assert args.prototype_only
    assert args.image_feature_shuffle
    assert args.graph_node_shuffle
    assert args.wrong_donor_control
    assert not args.no_graph


def test_predict_rejects_output_aliases_before_loading_inputs(tmp_path: Path) -> None:
    inputs = {
        "checkpoint": tmp_path / "checkpoint.pt",
        "histology": tmp_path / "histology.npz",
        "prototypes": tmp_path / "prototypes.npz",
        "genes": tmp_path / "genes.tsv",
        "program": tmp_path / "programs.npz",
        "ood": tmp_path / "ood.npz",
    }
    for role, path in inputs.items():
        path.write_bytes(("unparsed-" + role).encode("utf-8"))
    base = [
        "predict",
        "--checkpoint",
        str(inputs["checkpoint"]),
        "--histology",
        str(inputs["histology"]),
        "--prototypes",
        str(inputs["prototypes"]),
        "--genes",
        str(inputs["genes"]),
        "--program-artifact",
        str(inputs["program"]),
        "--ood-artifact",
        str(inputs["ood"]),
        "--donor-id",
        "target",
    ]
    original = {role: path.read_bytes() for role, path in inputs.items()}
    for role, path in inputs.items():
        parsed = build_parser().parse_args(
            [*base, "--output", str(path), "--telemetry-output", str(tmp_path / (role + ".json"))]
        )
        with pytest.raises(ValueError, match="output would overwrite a bound input"):
            parsed.func(parsed)
    parsed = build_parser().parse_args(
        [
            *base,
            "--output",
            str(tmp_path / "prediction.npz"),
            "--telemetry-output",
            str(inputs["genes"]),
        ]
    )
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        parsed.func(parsed)
    shared_output = tmp_path / "shared-output.npz"
    parsed = build_parser().parse_args(
        [
            *base,
            "--output",
            str(shared_output),
            "--telemetry-output",
            str(shared_output),
        ]
    )
    with pytest.raises(ValueError, match="output paths collide with each other"):
        parsed.func(parsed)
    assert {role: path.read_bytes() for role, path in inputs.items()} == original


def test_wrong_donor_ontology_intersection_is_deterministic_and_requires_support() -> None:
    source = PrototypeSet(
        prototype_ids=np.asarray(["p-c", "p-a-1", "p-x", "p-a-2", "p-b"]),
        sample_ids=np.asarray(["source"] * 5),
        cell_type_labels=np.asarray(["C", "A", "unsupported", "A", "B"]),
        means=np.arange(10, dtype=np.float32).reshape(5, 2),
        variances=np.full((5, 2), 0.25, dtype=np.float32),
        weights=np.asarray([0.10, 0.15, 0.40, 0.05, 0.30]),
        n_cells=np.asarray([10, 15, 40, 5, 30]),
        latent_space_id="latent-source",
        donor_id="source-donor",
        block_id="source-block",
        source_reference_sha256="a" * 64,
        latent_training_donors=("atlas-donor",),
        latent_transform_sha256="b" * 64,
    )

    filtered, telemetry = _wrong_donor_ontology_intersection(source, ["B", "A"])

    assert filtered.prototype_ids.tolist() == ["p-a-1", "p-a-2", "p-b"]
    assert filtered.cell_type_labels.tolist() == ["A", "A", "B"]
    np.testing.assert_allclose(filtered.weights, [0.30, 0.10, 0.60])
    assert filtered.donor_id == source.donor_id
    assert filtered.block_id == source.block_id
    assert filtered.source_reference_sha256 == source.source_reference_sha256
    assert filtered.latent_transform_sha256 == source.latent_transform_sha256
    assert telemetry == {
        "policy": "target_checkpoint_ontology_intersection_v1",
        "minimum_retained_type_count": 2,
        "minimum_retained_prototype_count": 2,
        "original_prototype_count": 5,
        "retained_prototype_count": 3,
        "omitted_prototype_count": 2,
        "original_type_count": 4,
        "retained_type_count": 2,
        "omitted_type_count": 2,
        "original_type_names": ["C", "A", "unsupported", "B"],
        "retained_type_names": ["B", "A"],
        "omitted_type_names": ["C", "unsupported"],
        "weights_renormalized": True,
    }
    with pytest.raises(ValueError, match="at least two cell types and two prototypes"):
        _wrong_donor_ontology_intersection(source, ["A"])


def _write_input_artifacts(tmp_path):
    rng = np.random.default_rng(7)
    cells = 12
    histology = HistologyBag(
        slide_id="sample1",
        nucleus_ids=np.asarray(["n%d" % index for index in range(cells)]),
        features=rng.normal(size=(cells, 4)).astype(np.float32),
        coordinates_um=np.column_stack((np.arange(cells), np.arange(cells) % 3)),
        segmentation_confidence=np.linspace(0.7, 1.0, cells, dtype=np.float32),
        artifact_probability=np.asarray([0.8] + [0.0] * (cells - 1), dtype=np.float32),
        edge_index=np.asarray(
            [
                np.concatenate((np.arange(cells - 1), np.arange(1, cells))),
                np.concatenate((np.arange(1, cells), np.arange(cells - 1))),
            ],
            dtype=np.int64,
        ),
        edge_weight=np.linspace(0.2, 1.0, 2 * (cells - 1), dtype=np.float32),
        sample_id="sample1",
        donor_id="donor1",
        block_id="block1",
        feature_space_id="pathology-encoder-v1",
        histology_source_sha256="a" * 64,
        nuclei_source_sha256="b" * 64,
        feature_source_sha256="c" * 64,
    )
    labels = np.asarray(["A"] * 6 + ["B"] * 6)
    counts = np.asarray(
        [
            [10, 8, 1, 0],
            [9, 7, 1, 0],
            [11, 6, 0, 1],
            [8, 9, 1, 0],
            [12, 8, 0, 1],
            [10, 10, 1, 0],
            [0, 1, 8, 11],
            [1, 0, 9, 10],
            [0, 1, 10, 9],
            [1, 0, 7, 12],
            [0, 1, 11, 8],
            [1, 0, 9, 11],
        ],
        dtype=np.float32,
    )
    latent = np.vstack(
        (
            rng.normal((-1.0, 0.0), 0.1, size=(6, 2)),
            rng.normal((1.0, 0.0), 0.1, size=(6, 2)),
        )
    ).astype(np.float32)
    reference = RNAReference(
        sample_id="sample1",
        cell_ids=np.asarray(["r%d" % index for index in range(cells)]),
        gene_ids=np.asarray(["g1", "g2", "g3", "g4"]),
        counts=sparse.csr_matrix(counts),
        latent=latent,
        cell_type_labels=labels,
        donor_ids=np.asarray(["donor1"] * cells),
        sample_ids=np.asarray(["sample1"] * cells),
        latent_space_id="test-latent-v1",
        block_id="block1",
    )
    prototypes = PrototypeSet(
        prototype_ids=np.asarray(["pA", "pB"]),
        sample_ids=np.asarray(["sample1", "sample1"]),
        cell_type_labels=np.asarray(["A", "B"]),
        means=np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        variances=np.full((2, 2), 0.2, dtype=np.float32),
        weights=np.asarray([0.5, 0.5]),
        n_cells=np.asarray([6, 6]),
        latent_space_id="test-latent-v1",
    )
    histology_path = tmp_path / "histology.npz"
    reference_path = tmp_path / "reference.npz"
    prototypes_path = tmp_path / "prototypes.npz"
    genes_path = tmp_path / "genes.tsv"
    histology.save_npz(histology_path)
    reference.save_npz(reference_path)
    prototypes = replace(
        prototypes,
        donor_id="donor1",
        block_id="block1",
        source_reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    )
    prototypes.save_npz(prototypes_path)
    genes_path.write_text("g1\ng2\ng3\ng4\n")
    return histology_path, reference_path, prototypes_path, genes_path


def test_predict_rechecks_frozen_inputs_before_publication(tmp_path: Path, monkeypatch) -> None:
    histology, _, prototypes, genes = _write_input_artifacts(tmp_path)
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=4,
            num_cell_types=2,
            expression_dim=4,
            latent_dim=2,
            graph_hidden_dim=4,
            graph_output_dim=4,
            graph_layers=1,
            graph_mode="off",
            trunk_hidden_dims=(4,),
            decoder_hidden_dims=(4,),
            dropout=0.0,
        )
    )
    checkpoint = model.checkpoint()
    checkpoint["metadata"] = {
        "schema": "heir.test.prediction_input_freeze.v1",
        "type_names": ["A", "B"],
        "gene_names": ["g1", "g2", "g3", "g4"],
        "latent_space_id": "test-latent-v1",
        "feature_space_id": "pathology-encoder-v1",
        "expression_space_id": "log1p-cpm-10000-v1",
    }
    checkpoint_path = tmp_path / "heir.pt"
    torch.save(checkpoint, checkpoint_path)
    original_predict_cells = cli_module.predict_cells

    def mutate_gene_panel_after_inference(*args, **kwargs):
        prediction = original_predict_cells(*args, **kwargs)
        genes.write_text("g1\ng2\ng3\ng4\nchanged-after-load\n", encoding="utf-8")
        return prediction

    monkeypatch.setattr(cli_module, "predict_cells", mutate_gene_panel_after_inference)
    output = tmp_path / "prediction.npz"
    parsed = build_parser().parse_args(
        [
            "predict",
            "--checkpoint",
            str(checkpoint_path),
            "--histology",
            str(histology),
            "--prototypes",
            str(prototypes),
            "--genes",
            str(genes),
            "--output",
            str(output),
            "--donor-id",
            "donor1",
            "--latent-samples",
            "1",
            "--device",
            "cpu",
        ]
    )
    with pytest.raises(ValueError, match="prediction input changed after it was loaded"):
        parsed.func(parsed)
    assert not output.exists()


def _write_test_initialization_receipt(tmp_path, teacher) -> Path:
    torch.manual_seed(7)
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=4,
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
        trunk.weight[:4, :4].copy_(torch.eye(4))
        trunk.bias.zero_()
        model.trunk[1].weight.fill_(1.0)
        model.trunk[1].bias.zero_()
        model.fine_type_head.weight.copy_(
            torch.tensor([[-5.0, 5.0, 0.0, 0.0, 0.0], [5.0, -5.0, 0.0, 0.0, 0.0]])
        )
        model.fine_type_head.bias.zero_()
    nucleus_ids = np.asarray(["e0", "e1", "e2", "e3"])
    donor_ids = np.asarray(["donor1"] * 4)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    morphology = np.asarray(
        [[-2.0, 2.0, 0.0, 0.0]] * 2 + [[2.0, -2.0, 0.0, 0.0]] * 2,
        dtype=np.float32,
    )
    target_latent = np.asarray([[-1.0, 0.0]] * 2 + [[1.0, 0.0]] * 2, dtype=np.float32)
    with torch.no_grad():
        embedding, _, _ = model.encode_frozen_morphology(torch.from_numpy(morphology))
        direction = embedding[2] - embedding[0]
        latent_weight = 2.0 * direction / direction.square().sum()
        model.prototype_query_head.weight.zero_()
        model.prototype_query_head.bias.zero_()
        model.prototype_query_head.weight[0].copy_(latent_weight)
        model.prototype_query_head.bias[0].copy_(-1.0 - torch.dot(latent_weight, embedding[0]))
    checkpoint_payload = model.checkpoint()
    checkpoint_payload["metadata"] = {
        "type_names": ["A", "B"],
        "training_donors": ["development-donor"],
        "feature_space_id": "pathology-encoder-v1",
        "latent_space_id": "test-latent-v1",
        "excluded_from_primary_claims": False,
        "exclusion_reasons": [],
    }
    torch.save(checkpoint_payload, teacher)
    label_source = tmp_path / "independent_labels.npz"
    np.savez_compressed(
        label_source,
        schema=np.asarray("heir.independent_initialization_labels.v1"),
        nucleus_ids=nucleus_ids,
        donor_ids=donor_ids,
        type_labels=labels,
        type_names=np.asarray(["A", "B"]),
        independent_of_checkpoint=np.asarray(True),
    )
    latent_source = tmp_path / "registered_latent.npz"
    np.savez_compressed(
        latent_source,
        schema=np.asarray("heir.registered_image_latent_targets.v1"),
        nucleus_ids=nucleus_ids,
        target_latent=target_latent,
        latent_space_id=np.asarray("test-latent-v1"),
        independent_of_checkpoint=np.asarray(True),
    )
    evidence_artifact = tmp_path / "initialization_evidence.npz"
    np.savez_compressed(
        evidence_artifact,
        morphology=morphology,
        edge_index=np.empty((2, 0), dtype=np.int64),
        nucleus_ids=nucleus_ids,
        donor_ids=donor_ids,
        type_labels=labels,
        type_names=np.asarray(["A", "B"]),
        target_latent=target_latent,
        feature_space_id=np.asarray("pathology-encoder-v1"),
        latent_space_id=np.asarray("test-latent-v1"),
        label_source_sha256=np.asarray(_sha256(str(label_source))),
        latent_target_source_sha256=np.asarray(_sha256(str(latent_source))),
        labels_independent_of_checkpoint=np.asarray(True),
        latent_targets_independent_of_checkpoint=np.asarray(True),
    )
    bindings = {
        "evidence_artifact": {
            "path": str(evidence_artifact),
            "sha256": _sha256(str(evidence_artifact)),
        },
        "label_source": {"path": str(label_source), "sha256": _sha256(str(label_source))},
        "latent_target_source": {
            "path": str(latent_source),
            "sha256": _sha256(str(latent_source)),
        },
    }
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
    plan = tmp_path / "initialization_plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "heir.initialization_validation_plan.v1",
                "status": "ready",
                "checkpoint": {
                    "path": str(teacher),
                    "sha256": hashlib.sha256(teacher.read_bytes()).hexdigest(),
                },
                "evaluation_artifact": bindings["evidence_artifact"],
                "label_source": bindings["label_source"],
                "latent_target_source": bindings["latent_target_source"],
                "held_out_donors": ["donor1"],
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
    report = tmp_path / "initialization_evidence_report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "heir.initialization_validation_evidence.v1",
                "status": "complete",
                "pass": True,
                "checkpoint": {
                    "path": str(teacher),
                    "sha256": hashlib.sha256(teacher.read_bytes()).hexdigest(),
                },
                "plan": {
                    "path": str(plan),
                    "sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                },
                **bindings,
                "feature_space_id": "pathology-encoder-v1",
                "latent_space_id": "test-latent-v1",
                "type_ontology_sha256": ordered_identity_sha256(("A", "B")),
                "training_donors": ["development-donor"],
                "held_out_donors": ["donor1"],
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
    receipt = tmp_path / "bridge_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "heir.validated_initialization.v1",
                "status": "complete",
                "pass": True,
                "checkpoint_sha256": hashlib.sha256(teacher.read_bytes()).hexdigest(),
                "feature_space_id": "pathology-encoder-v1",
                "latent_space_id": "test-latent-v1",
                "type_ontology_sha256": ordered_identity_sha256(("A", "B")),
                "training_donors": ["development-donor"],
                "held_out_donors": ["donor1"],
                "capabilities": {"broad_type": True, "image_to_latent": True},
                "evidence_report": str(report),
                "evidence_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return receipt


def test_predict_wrong_donor_filters_only_in_memory_and_binds_original_source(
    tmp_path,
    capsys,
) -> None:
    histology, _, prototype_path, genes = _write_input_artifacts(tmp_path)
    source = PrototypeSet.load_npz(prototype_path)
    source_with_unsupported_type = PrototypeSet(
        prototype_ids=np.asarray(["pA", "pB", "p-unsupported"]),
        sample_ids=np.asarray(["sample1"] * 3),
        cell_type_labels=np.asarray(["A", "B", "unsupported"]),
        means=np.vstack((source.means, np.asarray([[2.0, 2.0]], dtype=np.float32))),
        variances=np.vstack((source.variances, np.asarray([[0.3, 0.3]], dtype=np.float32))),
        weights=np.asarray([0.3, 0.4, 0.3]),
        n_cells=np.asarray([3, 4, 3]),
        latent_space_id=source.latent_space_id,
        donor_id=source.donor_id,
        block_id=source.block_id,
        source_reference_sha256=source.source_reference_sha256,
    )
    matched_source = tmp_path / "matched_with_unsupported_type.npz"
    source_with_unsupported_type.save_npz(matched_source)
    wrong_donor_source = tmp_path / "wrong_donor_with_unsupported_type.npz"
    replace(
        source_with_unsupported_type,
        sample_ids=np.asarray(["source-sample"] * 3),
        donor_id="source-donor",
        block_id="source-block",
    ).save_npz(wrong_donor_source)

    model = HEIRModel(
        HEIRConfig(
            morphology_dim=4,
            num_cell_types=2,
            expression_dim=4,
            latent_dim=2,
            graph_hidden_dim=4,
            graph_output_dim=4,
            graph_layers=1,
            graph_mode="distance_only",
            graph_context_gate_init=1.0,
            trunk_hidden_dims=(4,),
            decoder_hidden_dims=(4,),
            dropout=0.0,
        )
    )
    checkpoint = model.checkpoint()
    checkpoint["metadata"] = {
        "schema": "heir.test.wrong_donor.v1",
        "type_names": ["A", "B"],
        "gene_names": ["g1", "g2", "g3", "g4"],
        "latent_space_id": "test-latent-v1",
        "feature_space_id": "pathology-encoder-v1",
        "expression_space_id": "log1p-cpm-10000-v1",
    }
    checkpoint_path = tmp_path / "heir.pt"
    torch.save(checkpoint, checkpoint_path)

    shared_arguments = [
        "predict",
        "--checkpoint",
        str(checkpoint_path),
        "--histology",
        str(histology),
        "--genes",
        str(genes),
        "--donor-id",
        "donor1",
        "--latent-samples",
        "1",
        "--device",
        "cpu",
    ]
    with pytest.raises(SystemExit):
        main(
            shared_arguments
            + [
                "--prototypes",
                str(matched_source),
                "--output",
                str(tmp_path / "matched_predictions.npz"),
            ]
        )
    assert (
        "prototype types are absent from the model ontology: unsupported" in capsys.readouterr().err
    )

    with pytest.raises(SystemExit):
        main(
            shared_arguments
            + [
                "--prototypes",
                str(prototype_path),
                "--output",
                str(tmp_path / "matched_wrong_donor_control.npz"),
                "--wrong-donor-control",
            ]
        )
    assert "requires a non-matched PrototypeSet donor" in capsys.readouterr().err

    missing_donor_source = tmp_path / "missing_donor.npz"
    replace(source, donor_id="").save_npz(missing_donor_source)
    with pytest.raises(SystemExit):
        main(
            shared_arguments
            + [
                "--prototypes",
                str(missing_donor_source),
                "--output",
                str(tmp_path / "missing_donor_control.npz"),
                "--wrong-donor-control",
                "--unsafe-allow-missing-prototype-provenance",
            ]
        )
    assert "requires PrototypeSet donor provenance" in capsys.readouterr().err

    expected_permutation = np.asarray(np.random.default_rng(17).permutation(12), dtype="<i8")
    expected_map_sha256 = hashlib.sha256(expected_permutation.tobytes(order="C")).hexdigest()
    for control, flag in (
        ("image_shuffle", "--image-feature-shuffle"),
        ("graph_shuffle", "--graph-node-shuffle"),
    ):
        control_telemetry = tmp_path / (control + ".telemetry.json")
        assert (
            main(
                shared_arguments
                + [
                    "--prototypes",
                    str(prototype_path),
                    "--output",
                    str(tmp_path / (control + ".predictions.npz")),
                    "--telemetry-output",
                    str(control_telemetry),
                    flag,
                ]
            )
            == 0
        )
        transform = json.loads(control_telemetry.read_text())["negative_control"]["transform"]
        assert transform["schema"] == "heir.inference_control_transform.v1"
        assert transform["control"] == control
        assert transform["seed"] == 17
        assert transform["map_sha256"] == expected_map_sha256
        assert transform["expected_transform_map_sha256"] == expected_map_sha256
        assert len(transform["recipe_sha256"]) == 64

    output = tmp_path / "wrong_donor_predictions.npz"
    telemetry_path = tmp_path / "wrong_donor.telemetry.json"
    assert (
        main(
            shared_arguments
            + [
                "--prototypes",
                str(wrong_donor_source),
                "--output",
                str(output),
                "--telemetry-output",
                str(telemetry_path),
                "--wrong-donor-control",
            ]
        )
        == 0
    )

    prediction = PredictionBundle.from_npz(output)
    source_sha256 = hashlib.sha256(wrong_donor_source.read_bytes()).hexdigest()
    assert prediction.prototype_ids.tolist() == ["pA", "pB"]
    assert prediction.prototype_sha256 == source_sha256
    prototype_filter = json.loads(telemetry_path.read_text())["negative_control"][
        "prototype_filter"
    ]
    assert prototype_filter["policy"] == "target_checkpoint_ontology_intersection_v1"
    assert prototype_filter["original_source_prototype_sha256"] == source_sha256
    assert prototype_filter["original_type_names"] == ["A", "B", "unsupported"]
    assert prototype_filter["retained_type_names"] == ["A", "B"]
    assert prototype_filter["omitted_type_names"] == ["unsupported"]
    assert prototype_filter["original_prototype_count"] == 3
    assert prototype_filter["retained_prototype_count"] == 2
    assert prototype_filter["omitted_prototype_count"] == 1


def _residual_geometry_provenance_fixture(tmp_path: Path):
    histology_path, reference_path, prototypes_path, _ = _write_input_artifacts(tmp_path)
    transform_sha256 = "4" * 64
    latent_space_id = "sha256:" + transform_sha256
    reference = replace(
        RNAReference.load_npz(reference_path),
        latent_space_id=latent_space_id,
    )
    reference.save_npz(reference_path)
    reference_sha256 = _sha256(str(reference_path))
    prototypes = replace(
        PrototypeSet.load_npz(prototypes_path),
        latent_space_id=latent_space_id,
        source_reference_sha256=reference_sha256,
        latent_transform_sha256=transform_sha256,
    )
    prototypes.save_npz(prototypes_path)
    batch_path = tmp_path / "geometry_batch.npz"
    assert (
        main(
            [
                "assemble-batch",
                "--histology",
                str(histology_path),
                "--reference",
                str(reference_path),
                "--prototypes",
                str(prototypes_path),
                "--output",
                str(batch_path),
                "--donor-id",
                "donor1",
                "--block-id",
                "block1",
            ]
        )
        == 0
    )
    batch = HEIRTrainingBatch.load_npz(batch_path)
    geometry = fit_rna_residual_geometry(
        reference.latent,
        reference.cell_type_labels,
        1,
        type_names=batch.type_names,
        prototype_means=prototypes.means,
        prototype_labels=prototypes.cell_type_labels,
        prototype_variances=prototypes.variances,
        latent_space_id=latent_space_id,
        source_reference_sha256=reference_sha256,
        training_donors=("donor1",),
        latent_transform_sha256=transform_sha256,
    )
    return batch, geometry


def test_train_residual_geometry_requires_exact_molecular_source_and_type_order(
    tmp_path: Path,
) -> None:
    batch, geometry = _residual_geometry_provenance_fixture(tmp_path)

    assert (
        _validate_residual_geometry_training_provenance(
            geometry,
            [batch],
            batch.type_names,
            unsafe_allow_legacy=False,
        )
        == "strict_hash_bound_prototype_reference_and_latent_source"
    )

    same_latent_wrong_reference = replace(geometry, source_reference_sha256="f" * 64)
    with pytest.raises(ValueError, match="source reference differs from PrototypeSet"):
        _validate_residual_geometry_training_provenance(
            same_latent_wrong_reference,
            [batch],
            batch.type_names,
            unsafe_allow_legacy=False,
        )
    with pytest.raises(ValueError, match="source reference differs from PrototypeSet"):
        _validate_residual_geometry_training_provenance(
            same_latent_wrong_reference,
            [batch],
            batch.type_names,
            unsafe_allow_legacy=True,
        )

    missing_transform_identity = replace(geometry, latent_transform_sha256="")
    with pytest.raises(ValueError, match="latent transform differs from PrototypeSet"):
        _validate_residual_geometry_training_provenance(
            missing_transform_identity,
            [batch],
            batch.type_names,
            unsafe_allow_legacy=False,
        )

    reversed_types = replace(geometry, type_names=geometry.type_names[::-1].copy())
    with pytest.raises(ValueError, match="cell-type order differs"):
        _validate_residual_geometry_training_provenance(
            reversed_types,
            [batch],
            batch.type_names,
            unsafe_allow_legacy=False,
        )


def test_train_residual_geometry_legacy_override_only_waives_unavailable_provenance(
    tmp_path: Path,
) -> None:
    batch, geometry = _residual_geometry_provenance_fixture(tmp_path)
    legacy_batch = replace(
        batch,
        source_artifacts=(),
        source_sha256=(),
        source_roles=(),
    )

    with pytest.raises(ValueError, match="no accessible hash-bound PrototypeSet"):
        _validate_residual_geometry_training_provenance(
            geometry,
            [legacy_batch],
            legacy_batch.type_names,
            unsafe_allow_legacy=False,
        )
    assert (
        _validate_residual_geometry_training_provenance(
            geometry,
            [legacy_batch],
            legacy_batch.type_names,
            unsafe_allow_legacy=True,
        )
        == "unsafe_legacy_missing_provenance"
    )

    args = build_parser().parse_args(
        [
            "train",
            "--train-batch",
            "train.npz",
            "--validation-batch",
            "validation.npz",
            "--output",
            "trained",
            "--unsafe-allow-legacy-residual-geometry-provenance",
        ]
    )
    assert args.unsafe_allow_legacy_residual_geometry_provenance


def test_cli_demo_writes_self_describing_checkpoint(tmp_path):
    output = tmp_path / "demo"
    assert main(["demo", "--output", str(output), "--epochs", "1", "--device", "cpu"]) == 0
    assert (output / "heir_demo.pt").is_file()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["best_epoch"] == 0
    assert metrics["best_validation_loss"] >= 0


def test_log_normalize_uses_full_library_before_panel_selection():
    panel_counts = sparse.csr_matrix([[10.0, 0.0], [5.0, 5.0]])
    normalized = _log_normalize(
        panel_counts,
        library_sizes=np.asarray([100.0, 20.0]),
    ).toarray()
    expected = np.log1p(np.asarray([[1000.0, 0.0], [2500.0, 2500.0]]))
    np.testing.assert_allclose(normalized, expected, rtol=1.0e-6)


def test_prepare_histology_builds_calibrated_weighted_graph(tmp_path):
    nuclei = tmp_path / "nuclei.csv"
    nuclei.write_text("id,x,y,area,confidence\nn0,0,0,10,0.9\nn1,10,0,12,0.8\nn2,20,0,11,0.7\n")
    features = tmp_path / "features.npz"
    histology_source = tmp_path / "slide.tif"
    histology_source.write_bytes(b"fake-slide")
    np.savez_compressed(
        features,
        nucleus_ids=np.asarray(["n0", "n1", "n2"]),
        features=np.eye(3, dtype=np.float32),
        boundary_weight=np.asarray([1.0, 0.5, 0.2], dtype=np.float32),
    )
    output = tmp_path / "histology.npz"
    assert (
        main(
            [
                "prepare-histology",
                "--nuclei",
                str(nuclei),
                "--features",
                str(features),
                "--slide-id",
                "slide",
                "--sample-id",
                "sample",
                "--donor-id",
                "donor",
                "--block-id",
                "block",
                "--histology-source",
                str(histology_source),
                "--feature-space-id",
                "pathology-encoder-v1",
                "--mpp",
                "0.5",
                "--boundary-weight-key",
                "boundary_weight",
                "--graph-k",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    bag = HistologyBag.load_npz(output)
    assert bag.features.shape == (3, 4)
    assert bag.edge_index.shape[1] > 0
    assert np.all(bag.edge_weight <= 1.0)
    np.testing.assert_allclose(bag.coordinates_um[:, 0], [0.0, 5.0, 10.0])
    assert bag.nucleus_ids.tolist() == ["sample::n0", "sample::n1", "sample::n2"]
    assert bag.donor_id == "donor"
    assert bag.feature_space_id == "pathology-encoder-v1"


def test_archive_member_hash_matches_extracted_gzip_content(tmp_path):
    payload = b"synthetic-tiff-payload" * 20
    compressed = gzip.compress(payload)
    archive_path = tmp_path / "raw.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("nested/slide.tif.gz")
        member.size = len(compressed)
        archive.addfile(member, io.BytesIO(compressed))
    extracted = tmp_path / "slide.tif"
    extracted.write_bytes(payload)
    assert _sha256("%s::slide.tif.gz" % archive_path) == _sha256(str(extracted))


def test_fit_ood_uses_embedded_donor_and_feature_space_provenance(tmp_path):
    histology, _, _, _ = _write_input_artifacts(tmp_path)
    output = tmp_path / "ood.npz"
    assert (
        main(
            [
                "fit-ood",
                "--histology",
                str(histology),
                "--analysis-role",
                "development",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    detector = MahalanobisOOD.from_npz(output)
    assert detector.training_donors == ("donor1",)
    assert detector.feature_space_id == "pathology-encoder-v1"
    with pytest.raises(SystemExit):
        main(
            [
                "fit-ood",
                "--histology",
                str(histology),
                "--training-donor",
                "wrong-donor",
                "--output",
                str(output),
            ]
        )


def test_assemble_batch_roundtrip_and_validation(tmp_path):
    histology, reference, prototypes, _ = _write_input_artifacts(tmp_path)
    output = tmp_path / "batch.npz"
    assert (
        main(
            [
                "assemble-batch",
                "--histology",
                str(histology),
                "--reference",
                str(reference),
                "--prototypes",
                str(prototypes),
                "--output",
                str(output),
                "--donor-id",
                "donor1",
                "--block-id",
                "block1",
                "--bag-id",
                "bag1",
                "--latent-space-id",
                "test-latent-v1",
                "--markers-per-type",
                "1",
            ]
        )
        == 0
    )
    batch = HEIRTrainingBatch.load_npz(output)
    batch.validate(TrainingStage.PERSONALIZED)
    assert batch.bag_id == "bag1"
    assert batch.type_names == ("A", "B")
    assert batch.gene_names == ("g1", "g2", "g3", "g4")
    assert batch.source_roles == ("sample_assay",) * 3
    assert batch.marker_mask is not None
    assert batch.edge_weight is not None
    assert batch.cell_weights is not None and batch.cell_weights[0] == 0
    assert not torch.allclose(batch.edge_weight, torch.ones_like(batch.edge_weight))
    assert batch.marker_mask.dtype == torch.bool
    assert torch.all(batch.marker_mask.sum(dim=1) == 1)
    assert not torch.equal(batch.marker_mask[0], batch.marker_mask[1])

    with np.load(output, allow_pickle=False) as archive:
        legacy_payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    legacy_payload["__version__"] = np.asarray(1, dtype=np.int64)
    legacy = tmp_path / "legacy_batch.npz"
    np.savez_compressed(legacy, **legacy_payload)
    with pytest.raises(ValueError, match="regenerate"):
        HEIRTrainingBatch.load_npz(legacy)

    with pytest.raises(TypeError, match="marker_mask"):
        replace(batch, marker_mask=batch.marker_mask.float()).validate(TrainingStage.PERSONALIZED)
    with pytest.raises(ValueError, match="target_spatial_expression"):
        replace(
            batch,
            spot_assignment=torch.ones(2, len(batch.morphology)),
            target_spatial_expression=torch.ones(3, len(batch.gene_names)),
        ).validate(TrainingStage.GENERIC_SPATIAL_PRETRAINING)

    enriched = replace(
        batch,
        scgpt_type_prototypes=torch.randn(2, 3),
        scgpt_type_variances=torch.full((2, 3), 0.2),
        scgpt_space_id="scgpt-test-checkpoint",
        program_matrix=torch.randn(4, 2),
        target_program_scores=torch.randn(2, 2),
    )
    enriched.validate(TrainingStage.PERSONALIZED)
    enriched_path = tmp_path / "enriched_batch.npz"
    enriched.save_npz(enriched_path)
    loaded_enriched = HEIRTrainingBatch.load_npz(enriched_path)
    assert loaded_enriched.scgpt_type_prototypes is not None
    assert loaded_enriched.scgpt_type_prototypes.shape == (2, 3)
    assert loaded_enriched.target_program_scores is not None
    assert loaded_enriched.target_program_scores.shape == (2, 2)
    with pytest.raises(ValueError, match="supplied together"):
        replace(enriched, scgpt_type_variances=None).validate(TrainingStage.PERSONALIZED)


def test_refinement_view_artifact_binds_checkpoint_batch_and_identities(tmp_path):
    histology, reference, prototypes, _ = _write_input_artifacts(tmp_path)
    batch_path = tmp_path / "batch.npz"
    assert (
        main(
            [
                "assemble-batch",
                "--histology",
                str(histology),
                "--reference",
                str(reference),
                "--prototypes",
                str(prototypes),
                "--output",
                str(batch_path),
                "--donor-id",
                "donor1",
                "--block-id",
                "block1",
                "--bag-id",
                "bag1",
                "--latent-space-id",
                "test-latent-v1",
            ]
        )
        == 0
    )
    batch = HEIRTrainingBatch.load_npz(batch_path)
    torch.manual_seed(23)
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=batch.morphology.shape[1],
            num_cell_types=len(batch.type_names),
            expression_dim=len(batch.gene_names),
            latent_dim=batch.prototype_means.shape[1],
            graph_hidden_dim=8,
            graph_output_dim=6,
            graph_layers=1,
            trunk_hidden_dims=(8,),
            decoder_hidden_dims=(6,),
            dropout=0.0,
        )
    )
    checkpoint = model.checkpoint()
    checkpoint["metadata"] = {
        "type_names": list(batch.type_names),
        "gene_names": list(batch.gene_names),
        "feature_space_id": batch.feature_space_id,
        "latent_space_id": batch.latent_space_id,
        "expression_space_id": batch.expression_space_id,
    }
    checkpoint_path = tmp_path / "heir.pt"
    torch.save(checkpoint, checkpoint_path)
    views_path = tmp_path / "views.npz"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "build_refinement_views.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--batch",
            str(batch_path),
            "--output",
            str(views_path),
            "--shared-tail-features",
            "0",
            "--device",
            "cpu",
        ],
        check=True,
    )
    batch_digest_before_collision_check = hashlib.sha256(batch_path.read_bytes()).hexdigest()
    colliding_output = tmp_path / "views-hardlink-to-batch.npz"
    colliding_output.hardlink_to(batch_path)
    collision = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "build_refinement_views.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--batch",
            str(batch_path),
            "--output",
            str(colliding_output),
            "--shared-tail-features",
            "0",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert collision.returncode != 0
    assert "output would overwrite a bound input" in collision.stderr
    assert hashlib.sha256(batch_path.read_bytes()).hexdigest() == (
        batch_digest_before_collision_check
    )
    source_digest_before_collision_check = hashlib.sha256(histology.read_bytes()).hexdigest()
    transitive_collision_output = tmp_path / "views-hardlink-to-histology.npz"
    transitive_collision_output.hardlink_to(histology)
    transitive_collision = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "build_refinement_views.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--batch",
            str(batch_path),
            "--output",
            str(transitive_collision_output),
            "--shared-tail-features",
            "0",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert transitive_collision.returncode != 0
    assert "output would overwrite a bound input" in transitive_collision.stderr
    assert hashlib.sha256(histology.read_bytes()).hexdigest() == (
        source_digest_before_collision_check
    )
    with np.load(views_path, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    metadata = json.loads(str(payload["metadata_json"].item()))
    assert metadata["schema"] == "heir.refinement_views.v2"
    assert metadata["checkpoint_sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert metadata["batch_sha256"] == hashlib.sha256(batch_path.read_bytes()).hexdigest()
    assert metadata["batch_source_sha256"] == list(batch.source_sha256)
    assert metadata["sample_id"] == batch.sample_id
    assert metadata["donor_id"] == batch.donor_id
    assert metadata["bag_id"] == batch.bag_id
    assert metadata["feature_space_id"] == batch.feature_space_id
    assert metadata["latent_space_id"] == batch.latent_space_id
    assert metadata["type_names"] == list(batch.type_names)

    key = "%s::%s::%s" % (batch.donor_id, batch.sample_id, batch.bag_id)

    def load(path):
        return _load_refinement_views(
            ["%s=%s" % (key, path)],
            [batch],
            checkpoint_path=str(checkpoint_path),
            batch_paths=[str(batch_path)],
        )

    loaded = load(views_path)
    assert loaded[key].shape == (2, len(batch.nucleus_ids), len(batch.type_names))

    mismatches = (
        ("checkpoint_sha256", "0" * 64),
        ("batch_sha256", "1" * 64),
        ("batch_source_sha256", ["2" * 64]),
        ("sample_id", "other-sample"),
        ("donor_id", "other-donor"),
        ("bag_id", "other-bag"),
        ("feature_space_id", "other-features"),
        ("latent_space_id", "other-latent"),
        ("type_names", list(reversed(batch.type_names))),
        ("type_ontology_sha256", "3" * 64),
    )
    for index, (field, value) in enumerate(mismatches):
        changed = dict(metadata)
        changed[field] = value
        variant_payload = dict(payload)
        variant_payload["metadata_json"] = np.asarray(json.dumps(changed, sort_keys=True))
        variant = tmp_path / ("views_mismatch_%d.npz" % index)
        np.savez_compressed(variant, **variant_payload)
        with pytest.raises(ValueError, match=field):
            load(variant)

    legacy_fields = {
        "schema",
        "checkpoint",
        "checkpoint_sha256",
        "batch",
        "batch_sha256",
        "view_construction",
        "encoder_blocks",
        "encoder_block_width",
        "shared_tail_features",
        "device",
    }
    legacy_metadata = {name: value for name, value in metadata.items() if name in legacy_fields}
    legacy_metadata["schema"] = "heir.refinement_views.v1"
    legacy_payload = dict(payload)
    legacy_payload["metadata_json"] = np.asarray(json.dumps(legacy_metadata, sort_keys=True))
    legacy = tmp_path / "views_v1.npz"
    np.savez_compressed(legacy, **legacy_payload)
    assert load(legacy)[key].shape == loaded[key].shape

    unversioned_payload = dict(payload)
    unversioned_metadata = dict(metadata)
    del unversioned_metadata["schema"]
    unversioned_payload["metadata_json"] = np.asarray(
        json.dumps(unversioned_metadata, sort_keys=True)
    )
    unversioned = tmp_path / "views_unversioned.npz"
    np.savez_compressed(unversioned, **unversioned_payload)
    with pytest.raises(ValueError, match="unsupported provenance schema"):
        load(unversioned)


def test_assemble_batch_keeps_ood_separate_from_biological_unknown_targets(tmp_path):
    histology, reference, prototypes, _ = _write_input_artifacts(tmp_path)
    bag = HistologyBag.load_npz(histology)
    development_features = np.vstack((bag.features - 0.5, bag.features + 0.5))
    detector = MahalanobisOOD().fit(
        development_features,
        analysis_role="development",
        quantile=0.5,
        training_donors=("B1",),
        feature_space_id=bag.feature_space_id,
    )
    detector.source_sha256 = ("d" * 64,)
    ood = tmp_path / "calibrated_ood.npz"
    detector.to_npz(ood)
    output = tmp_path / "batch_with_ood_mask.npz"
    assert (
        main(
            [
                "assemble-batch",
                "--histology",
                str(histology),
                "--reference",
                str(reference),
                "--prototypes",
                str(prototypes),
                "--ood-artifact",
                str(ood),
                "--output",
                str(output),
                "--donor-id",
                "donor1",
                "--block-id",
                "block1",
                "--analysis-role",
                "locked_validation",
            ]
        )
        == 0
    )
    batch = HEIRTrainingBatch.load_npz(output)
    expected = detector.is_ood(bag.features)
    assert batch.ood_mask is not None
    assert batch.unknown_targets is None
    np.testing.assert_array_equal(batch.ood_mask.numpy(), expected)
    assert str(ood.resolve()) in batch.source_artifacts
    source_index = batch.source_artifacts.index(str(ood.resolve()))
    assert batch.source_sha256[source_index] == hashlib.sha256(ood.read_bytes()).hexdigest()


def test_assemble_batch_uses_only_explicit_biological_unknown_targets(tmp_path):
    histology, reference, prototypes, _ = _write_input_artifacts(tmp_path)
    bag = HistologyBag.load_npz(histology)
    target_values = np.linspace(0.0, 1.0, bag.n_nuclei, dtype=np.float32)
    target_path = tmp_path / "biological_unknown_targets.npz"
    np.savez_compressed(
        target_path,
        nucleus_ids=bag.nucleus_ids[::-1],
        unknown_targets=target_values[::-1],
    )
    output = tmp_path / "batch_with_explicit_unknown_targets.npz"

    assert (
        main(
            [
                "assemble-batch",
                "--histology",
                str(histology),
                "--reference",
                str(reference),
                "--prototypes",
                str(prototypes),
                "--unknown-targets",
                str(target_path),
                "--output",
                str(output),
                "--donor-id",
                "donor1",
                "--block-id",
                "block1",
                "--analysis-role",
                "development",
            ]
        )
        == 0
    )
    batch = HEIRTrainingBatch.load_npz(output)
    assert batch.ood_mask is None
    assert batch.unknown_targets is not None
    np.testing.assert_allclose(batch.unknown_targets.numpy(), target_values)
    assert str(target_path.resolve()) in batch.source_artifacts


def test_assemble_batch_attaches_hash_bound_frozen_molecular_estep(tmp_path):
    histology, reference, prototypes, _ = _write_input_artifacts(tmp_path)
    bag = HistologyBag.load_npz(histology)
    prototype_bank = PrototypeSet.load_npz(prototypes)
    source_paths = (histology, prototypes, reference)
    source_mass = np.asarray(
        bag.segmentation_confidence * (1.0 - bag.artifact_probability), dtype=np.float32
    )
    source_mass[np.asarray(bag.artifact_probability) >= 0.5] = 0.0
    teacher = tmp_path / "bridge_teacher.pt"
    receipt = _write_test_initialization_receipt(tmp_path, teacher)
    teacher_payload = torch.load(teacher, map_location="cpu", weights_only=True)
    teacher_model = HEIRModel.from_checkpoint(teacher_payload).to(dtype=torch.float32).eval()
    prototype_types = np.asarray([0, 1], dtype=np.int64)
    prototype_weights = np.asarray(prototype_bank.weights, dtype=np.float32)
    with torch.inference_mode():
        _, type_probabilities, image_latent = teacher_model.encode_frozen_morphology(
            torch.from_numpy(np.array(bag.features, dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(bag.edge_index, dtype=np.int64, copy=True)),
            torch.from_numpy(np.array(bag.edge_weight, dtype=np.float32, copy=True)),
        )
        means = torch.from_numpy(np.array(prototype_bank.means, dtype=np.float32, copy=True))
        variances = torch.from_numpy(
            np.array(prototype_bank.variances, dtype=np.float32, copy=True)
        ).clamp_min(teacher_model.config.prototype_variance_floor)
        gaussian_cost = 0.5 * (
            (image_latent.unsqueeze(1) - means.unsqueeze(0)).square() / variances.unsqueeze(0)
            + variances.unsqueeze(0).log()
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
            unknown_mass=0.1,
            unknown_cost=1.0,
            add_unknown=True,
        )
    assert bool(transport.converged)
    raw_plan = transport.plan.float().numpy()
    plan = np.zeros_like(raw_plan, dtype=np.float32)
    positive = source_mass > 0
    plan[positive] = raw_plan[positive] / raw_plan[positive].sum(axis=1, keepdims=True)
    plan[~positive, -1] = 1.0
    cost = (
        torch.cat((known_cost, known_cost.new_full((len(known_cost), 1), 1.0)), dim=1)
        .float()
        .numpy()
    )
    normalized_source_mass = source_mass / source_mass.sum(dtype=np.float32)
    desired_target = np.concatenate((prototype_weights * 0.9, np.asarray([0.1])))
    realized_target = (plan * normalized_source_mass[:, None]).sum(axis=0)
    source_marginal_residual = float(np.max(np.abs(plan.sum(axis=1) - 1.0)))
    target_marginal_residual = float(np.abs(realized_target - desired_target).sum())
    telemetry = frozen_transport_telemetry(
        raw_transport_plan=raw_plan,
        transport_cost=cost,
        source_mass=source_mass,
        target_weights=prototype_weights,
        fixed_unknown_mass=0.1,
        epsilon=0.5,
        marginal_relaxation=1.0,
    )
    e_step = MolecularEStepArtifact(
        transport_plan=plan.astype(np.float32),
        raw_transport_plan=raw_plan,
        transport_cost=cost,
        source_mass=source_mass,
        nucleus_ids=tuple(str(value) for value in bag.nucleus_ids.tolist()),
        prototype_ids=tuple(str(value) for value in prototype_bank.prototype_ids.tolist()),
        source_artifacts=tuple(str(path.resolve()) for path in source_paths),
        source_sha256=tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths),
        source_roles=("histology", "prototype_bank", "rna_reference"),
        teacher_checkpoint=str(teacher),
        teacher_checkpoint_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
        initialization_receipt=str(receipt),
        initialization_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        teacher_role="independent_crossmodal_bridge",
        teacher_training_donors=("development-donor",),
        target_donor="donor1",
        feature_space_id=bag.feature_space_id,
        latent_space_id=prototype_bank.latent_space_id,
        type_ontology_sha256=ordered_identity_sha256(("A", "B")),
        morphology_sha256=array_content_sha256(bag.features),
        prototype_means_sha256=array_content_sha256(prototype_bank.means),
        prototype_variances_sha256=array_content_sha256(prototype_bank.variances),
        prototype_types_sha256=array_content_sha256(prototype_types),
        prototype_weights_sha256=array_content_sha256(prototype_weights),
        image_latent_sha256=array_content_sha256(image_latent.float().numpy()),
        type_probabilities_sha256=array_content_sha256(type_probabilities.float().numpy()),
        transport_cost_sha256=array_content_sha256(cost),
        source_mass_sha256=array_content_sha256(source_mass),
        artifact_threshold=0.5,
        type_cost_weight=1.0,
        unknown_cost=1.0,
        fixed_unknown_mass=0.1,
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
    e_step_path = tmp_path / "frozen_e_step.npz"
    e_step.save_npz(e_step_path)
    output = tmp_path / "strict_batch.npz"

    assert (
        main(
            [
                "assemble-batch",
                "--histology",
                str(histology),
                "--reference",
                str(reference),
                "--prototypes",
                str(prototypes),
                "--molecular-e-step",
                str(e_step_path),
                "--output",
                str(output),
                "--donor-id",
                "donor1",
                "--block-id",
                "block1",
            ]
        )
        == 0
    )
    batch = HEIRTrainingBatch.load_npz(output)
    assert batch.molecular_responsibilities is not None
    np.testing.assert_array_equal(
        batch.molecular_responsibilities.numpy(),
        e_step.responsibilities,
    )
    index = batch.source_roles.index("frozen_e_step")
    assert batch.source_sha256[index] == hashlib.sha256(e_step_path.read_bytes()).hexdigest()
    assert "development-donor" in batch.molecular_training_donors


def test_personalized_train_fails_closed_without_validated_initialization(tmp_path):
    histology, reference, prototypes, _ = _write_input_artifacts(tmp_path)
    batch = tmp_path / "personalized_batch.npz"
    assert (
        main(
            [
                "assemble-batch",
                "--histology",
                str(histology),
                "--reference",
                str(reference),
                "--prototypes",
                str(prototypes),
                "--output",
                str(batch),
                "--donor-id",
                "donor1",
                "--block-id",
                "block1",
            ]
        )
        == 0
    )
    with pytest.raises(SystemExit):
        main(
            [
                "train",
                "--train-batch",
                str(batch),
                "--validation-batch",
                str(batch),
                "--output",
                str(tmp_path / "blocked"),
            ]
        )


def test_build_prototypes_binds_emitted_latent_reference_for_safe_assembly(tmp_path):
    histology_path, reference_path, _, _ = _write_input_artifacts(tmp_path)
    reference = RNAReference.load_npz(reference_path)
    latent_free = replace(
        reference,
        latent=np.empty((reference.shape[0], 0), dtype=np.float32),
        latent_space_id="",
    )
    latent_free.save_npz(reference_path)
    manifest_path = tmp_path / "manifest.tsv"
    record = ManifestRecord(
        cohort_id="development-cohort",
        donor_id="donor1",
        specimen_id="sample1",
        block_id="block1",
        section_id="sample1",
        modality="histology+snrna",
        assay_platform="synthetic",
        preservation="FFPE",
        tissue="breast",
        matching_tier="tier_1",
        matching_notes="unit-test matched block",
        analysis_role="development",
        outer_fold="fold_0",
        inner_fold="inner_0",
    )
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerow(record.to_mapping())
    enriched_reference = tmp_path / "reference_latent.npz"
    transform = tmp_path / "shared_svd.npz"
    prototypes = tmp_path / "prototypes_built.npz"
    assert (
        main(
            [
                "build-prototypes",
                "--reference",
                str(reference_path),
                "--reference-with-latent",
                str(enriched_reference),
                "--fit-latent-transform",
                str(transform),
                "--manifest",
                str(manifest_path),
                "--section-id",
                "sample1",
                "--latent-dim",
                "2",
                "--minimum-cells",
                "2",
                "--max-per-type",
                "2",
                "--output",
                str(prototypes),
            ]
        )
        == 0
    )
    prototype_set = PrototypeSet.load_npz(prototypes)
    enriched = RNAReference.load_npz(enriched_reference)
    assert enriched.latent_training_donors == ("donor1",)
    assert enriched.latent_transform_sha256 == hashlib.sha256(transform.read_bytes()).hexdigest()
    assert enriched.latent_space_id == "sha256:%s" % enriched.latent_transform_sha256
    assert (
        prototype_set.source_reference_sha256
        == hashlib.sha256(enriched_reference.read_bytes()).hexdigest()
    )
    assert (
        prototype_set.source_reference_sha256
        != hashlib.sha256(reference_path.read_bytes()).hexdigest()
    )

    batch = tmp_path / "assembled_from_build.npz"
    assert (
        main(
            [
                "assemble-batch",
                "--histology",
                str(histology_path),
                "--reference",
                str(enriched_reference),
                "--prototypes",
                str(prototypes),
                "--output",
                str(batch),
                "--donor-id",
                "donor1",
                "--block-id",
                "block1",
            ]
        )
        == 0
    )
    loaded = HEIRTrainingBatch.load_npz(batch)
    loaded.validate(TrainingStage.PERSONALIZED)
    assert loaded.latent_space_id == "sha256:" + hashlib.sha256(transform.read_bytes()).hexdigest()


def test_spatial_pretraining_uses_spot_mass_weighted_pseudobulk(tmp_path):
    histology, reference, prototypes, _ = _write_input_artifacts(tmp_path)
    assignment = np.zeros((2, 12), dtype=np.float32)
    assignment[0, 0] = 1.0
    assignment[1, 1:] = 1.0
    observed = np.asarray([[1.0] * 4, [9.0] * 4], dtype=np.float32)
    truth = tmp_path / "spatial.npz"
    np.savez_compressed(
        truth,
        nucleus_ids=np.asarray(["n%d" % index for index in range(12)]),
        spot_ids=np.asarray(["spot-small", "spot-large"]),
        spot_assignment=assignment,
        observed_expression=observed,
        gene_names=np.asarray(["g1", "g2", "g3", "g4"]),
        expression_space_id=np.asarray("log1p-cpm-10000-v1"),
    )
    output = tmp_path / "pretraining.npz"
    assert (
        main(
            [
                "assemble-batch",
                "--histology",
                str(histology),
                "--reference",
                str(reference),
                "--prototypes",
                str(prototypes),
                "--spatial-pretraining-truth",
                str(truth),
                "--output",
                str(output),
                "--donor-id",
                "donor1",
                "--block-id",
                "block1",
                "--analysis-role",
                "pretraining",
            ]
        )
        == 0
    )
    batch = HEIRTrainingBatch.load_npz(output)
    batch.validate(TrainingStage.GENERIC_SPATIAL_PRETRAINING)
    assert batch.spot_ids == ("spot-small", "spot-large")
    expected_pseudobulk = np.log1p((np.expm1(1.0) + 11 * np.expm1(9.0)) / 12)
    torch.testing.assert_close(
        batch.target_pseudobulk,
        torch.full((4,), expected_pseudobulk, dtype=torch.float32),
    )
    trained = tmp_path / "generic-trained"
    assert (
        main(
            [
                "train",
                "--train-batch",
                str(output),
                "--validation-batch",
                str(output),
                "--stage",
                "generic_spatial_pretraining",
                "--output",
                str(trained),
                "--epochs",
                "1",
                "--graph-hidden-dim",
                "8",
                "--graph-output-dim",
                "6",
                "--graph-layers",
                "1",
                "--trunk-hidden-dims",
                "8",
                "--decoder-hidden-dims",
                "6",
                "--dropout",
                "0",
                "--allow-random-decoder",
                "--allow-split-overlap",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (trained / "heir.pt").is_file()

    empty_assignment = assignment.copy()
    empty_assignment[0] = 0
    np.savez_compressed(
        truth,
        nucleus_ids=np.asarray(["n%d" % index for index in range(12)]),
        spot_ids=np.asarray(["spot-empty", "spot-large"]),
        spot_assignment=empty_assignment,
        observed_expression=observed,
        gene_names=np.asarray(["g1", "g2", "g3", "g4"]),
        expression_space_id=np.asarray("log1p-cpm-10000-v1"),
    )
    with pytest.raises(SystemExit):
        main(
            [
                "assemble-batch",
                "--histology",
                str(histology),
                "--reference",
                str(reference),
                "--prototypes",
                str(prototypes),
                "--spatial-pretraining-truth",
                str(truth),
                "--output",
                str(output),
                "--donor-id",
                "donor1",
                "--block-id",
                "block1",
                "--analysis-role",
                "pretraining",
            ]
        )


def test_real_artifact_train_and_predict_smoke(tmp_path):
    histology, reference, prototypes, genes = _write_input_artifacts(tmp_path)
    train_batch = tmp_path / "train.npz"
    validation_batch = tmp_path / "validation.npz"
    for output, bag_id in ((train_batch, "train"), (validation_batch, "validation")):
        assert (
            main(
                [
                    "assemble-batch",
                    "--histology",
                    str(histology),
                    "--reference",
                    str(reference),
                    "--prototypes",
                    str(prototypes),
                    "--output",
                    str(output),
                    "--donor-id",
                    "donor1",
                    "--block-id",
                    "block1",
                    "--bag-id",
                    bag_id,
                    "--latent-space-id",
                    "test-latent-v1",
                ]
            )
            == 0
        )
    trained = tmp_path / "trained"
    rna_vae = RNAVAE(
        RNAVAEConfig(
            input_dim=4,
            latent_dim=2,
            hidden_dims=(6,),
            decoder_hidden_dims=(6,),
            dropout=0.0,
            nonnegative_output=True,
        )
    )
    rna_checkpoint = rna_vae.checkpoint()
    rna_checkpoint["metadata"] = {
        "gene_names": ["g1", "g2", "g3", "g4"],
        "training_donors": ["donor0"],
        "latent_space_id": "test-latent-v1",
        "expression_space_id": "log1p-cpm-10000-v1",
        "decoder_only": False,
    }
    rna_checkpoint_path = tmp_path / "rna_vae.pt"
    torch.save(rna_checkpoint, rna_checkpoint_path)
    assert (
        main(
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
                "8",
                "--graph-output-dim",
                "6",
                "--graph-layers",
                "1",
                "--trunk-hidden-dims",
                "8",
                "--decoder-hidden-dims",
                "6",
                "--dropout",
                "0",
                "--abstain-threshold",
                "0.73",
                "--rna-vae-checkpoint",
                str(rna_checkpoint_path),
                "--uninitialized-morphology-negative-control",
                "--live-student-e-step-negative-control",
                "--allow-split-overlap",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (trained / "heir.pt").is_file()
    assert (trained / "history.json").is_file()
    trained_payload = torch.load(trained / "heir.pt", map_location="cpu", weights_only=False)
    assert trained_payload["config"]["abstain_threshold"] == pytest.approx(0.73)
    assert trained_payload["metadata"]["uot_unknown_mass"] == pytest.approx(0.05)
    assert trained_payload["metadata"]["uot_unknown_mass_mode"] == "fixed"
    assert trained_payload["metadata"]["excluded_from_primary_claims"] is True
    assert set(trained_payload["metadata"]["exclusion_reasons"]) >= {
        "uninitialized_morphology_negative_control",
        "live_student_e_step_negative_control",
        "train_validation_source_overlap_allowed",
    }
    assert set(trained_payload["metadata"]["training_donors"]) == {"donor0", "donor1"}
    assert trained_payload["metadata"]["direct_training_donors"] == ["donor1"]
    assert trained_payload["metadata"]["validation_donors"] == ["donor1"]
    assert trained_payload["metadata"]["training_batch_artifacts"] == [
        {"path": str(train_batch.resolve()), "sha256": _sha256(str(train_batch))}
    ]
    assert trained_payload["metadata"]["validation_batch_artifacts"] == [
        {"path": str(validation_batch.resolve()), "sha256": _sha256(str(validation_batch))}
    ]

    predictions = tmp_path / "predictions.npz"
    prediction_telemetry = tmp_path / "prediction_telemetry.json"
    assert (
        main(
            [
                "predict",
                "--checkpoint",
                str(trained / "heir.pt"),
                "--histology",
                str(histology),
                "--prototypes",
                str(prototypes),
                "--genes",
                str(genes),
                "--output",
                str(predictions),
                "--telemetry-output",
                str(prediction_telemetry),
                "--donor-id",
                "donor1",
                "--latent-samples",
                "2",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    bundle = PredictionBundle.from_npz(predictions)
    assert bundle.expression_mean.shape == (12, 4)
    assert bundle.type_names.tolist() == ["A", "B"]
    assert bundle.inference_seed == 17
    telemetry = json.loads(prediction_telemetry.read_text())
    assert telemetry["schema"] == "heir.inference_telemetry.v1"
    assert telemetry["device_type"] == "cpu"
    assert telemetry["peak_cuda_memory_bytes"] == 0
    assert telemetry["nuclei"] == 12
    assert telemetry["prediction_sha256"] == hashlib.sha256(predictions.read_bytes()).hexdigest()
    assert telemetry["residual_diagnostics"]["schema"] == ("heir.residual_gate_diagnostics.v1")
    assert telemetry["residual_diagnostics"]["donor_id"] == "donor1"
    assert set(telemetry["residual_diagnostics"]["by_type"]) == {"A", "B"}
    repeated_predictions = tmp_path / "predictions_repeated.npz"
    assert (
        main(
            [
                "predict",
                "--checkpoint",
                str(trained / "heir.pt"),
                "--histology",
                str(histology),
                "--prototypes",
                str(prototypes),
                "--genes",
                str(genes),
                "--output",
                str(repeated_predictions),
                "--donor-id",
                "donor1",
                "--latent-samples",
                "2",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    repeated_bundle = PredictionBundle.from_npz(repeated_predictions)
    np.testing.assert_array_equal(bundle.expression_mean, repeated_bundle.expression_mean)
    np.testing.assert_array_equal(bundle.expression_lower, repeated_bundle.expression_lower)

    excluded_refinement_rna = tmp_path / "excluded_refinement_rna.pt"
    excluded_rna_checkpoint = dict(rna_checkpoint)
    excluded_rna_checkpoint["metadata"] = {
        **rna_checkpoint["metadata"],
        "excluded_from_primary_claims": True,
        "exclusion_reasons": ["refinement_only_rna_sensitivity"],
    }
    torch.save(excluded_rna_checkpoint, excluded_refinement_rna)
    refined = tmp_path / "refined"
    assert (
        main(
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
                "--rna-vae-checkpoint",
                str(excluded_refinement_rna),
                "--allow-no-view-agreement",
                "--live-student-e-step-negative-control",
                "--allow-split-overlap",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (refined / "heir_refined.pt").is_file()
    assert (refined / "refinement.json").is_file()
    refinement_audit = json.loads((refined / "refinement.json").read_text())
    assert np.isfinite(refinement_audit["round_zero_validation_loss"])
    assert [row["round_id"] for row in refinement_audit["rounds"]] == [1]
    refined_payload = torch.load(
        refined / "heir_refined.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert refined_payload["metadata"]["refinement_round"] == refinement_audit["selected_round"]
    assert refined_payload["metadata"]["refinement_rounds_executed"] == 1
    assert refined_payload["metadata"]["uot_unknown_mass"] == pytest.approx(0.05)
    assert refined_payload["metadata"]["uot_unknown_mass_mode"] == "fixed"
    assert set(refined_payload["metadata"]["training_donors"]) == {"donor0", "donor1"}
    assert refined_payload["metadata"]["refinement_validation_donors"] == ["donor1"]
    assert refined_payload["metadata"]["refinement_training_batch_artifacts"] == [
        {"path": str(train_batch.resolve()), "sha256": _sha256(str(train_batch))}
    ]
    assert (
        "refinement_rna_decoder:refinement_only_rna_sensitivity"
        in refined_payload["metadata"]["exclusion_reasons"]
    )
    assert (refined / "prototypes" / "donor1__sample1.npz").is_file()


def test_evaluate_excludes_unavailable_public_cell_expression(tmp_path):
    cells = 3
    availability = np.asarray([True, False, True])
    expression = np.asarray([[1.0, 2.0], [100.0, 200.0], [3.0, 4.0]], dtype=np.float32)
    lower = expression - 0.1
    upper = expression + 0.1
    lower[~availability] = np.nan
    upper[~availability] = np.nan
    prediction = PredictionBundle(
        nucleus_ids=np.asarray(["n0", "n1", "n2"]),
        coordinates_um=np.zeros((cells, 2), dtype=np.float32),
        type_probabilities=np.asarray([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]], dtype=np.float32),
        type_names=np.asarray(["A", "B"]),
        labels=np.asarray([0, -1, 1], dtype=np.int64),
        prototype_probabilities=np.asarray([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]], dtype=np.float32),
        prototype_ids=np.asarray(["pA", "pB"]),
        latent_mean=np.zeros((cells, 2), dtype=np.float32),
        latent_variance=np.ones((cells, 2), dtype=np.float32),
        expression_mean=expression,
        expression_lower=lower,
        expression_upper=upper,
        gene_names=np.asarray(["g1", "g2"]),
        unknown_probability=np.asarray([0.0, 0.8, 0.0], dtype=np.float32),
        abstain_score=np.asarray([0.0, 0.8, 0.0], dtype=np.float32),
        abstain=~availability,
        ood_score=np.zeros(cells, dtype=np.float32),
        refinement_round=0,
        expression_interval_semantics=PredictionBundle.CONDITIONAL_KNOWN_STATE,
        expression_mean_available=availability,
        expression_interval_available=availability,
        sample_id="sample1",
        donor_id="donor1",
        slide_id="slide1",
        checkpoint_sha256="a" * 64,
        prototype_sha256="b" * 64,
        histology_sha256="c" * 64,
        latent_space_id="latent-test",
        model_version="test",
        inference_seed=17,
        latent_samples=20,
        probability_threshold=0.6,
        artifact_threshold=0.5,
        expression_space_id="log1p-cpm-10000-v1",
    )
    predictions = tmp_path / "predictions.npz"
    prediction.to_npz(predictions)
    assert np.isnan(PredictionBundle.from_npz(predictions).public_cell_expression_mean[1]).all()

    truth = tmp_path / "truth.npz"
    observed = expression.copy()
    observed[1] = -1000.0
    np.savez_compressed(
        truth,
        nucleus_ids=prediction.nucleus_ids,
        observed_expression=observed,
        gene_names=prediction.gene_names,
        expression_space_id=np.asarray(prediction.expression_space_id),
    )
    metrics_path = tmp_path / "metrics.json"
    assert (
        main(
            [
                "evaluate",
                "--predictions",
                str(predictions),
                "--truth",
                str(truth),
                "--output",
                str(metrics_path),
            ]
        )
        == 0
    )
    result = json.loads(metrics_path.read_text())
    assert result["expression"]["median_gene_mse"] == pytest.approx(0.0)
    assert result["expression"]["cells_total"] == 3
    assert result["expression"]["cells_evaluated"] == 2
    assert result["expression"]["cells_unavailable_excluded"] == 1
    assert result["expression"]["availability_policy"] == "prediction.expression_mean_available"

    all_unavailable = replace(
        prediction,
        labels=np.full(cells, -1, dtype=np.int64),
        abstain=np.ones(cells, dtype=bool),
        expression_mean_available=np.zeros(cells, dtype=bool),
        expression_interval_available=np.zeros(cells, dtype=bool),
        expression_lower=np.full_like(expression, np.nan),
        expression_upper=np.full_like(expression, np.nan),
    )
    unavailable_path = tmp_path / "predictions_all_unavailable.npz"
    all_unavailable.to_npz(unavailable_path)
    with pytest.raises(SystemExit):
        main(
            [
                "evaluate",
                "--predictions",
                str(unavailable_path),
                "--truth",
                str(truth),
            ]
        )


def test_evaluate_spatial_aligns_names_and_ignores_empty_spots(tmp_path):
    cells = 6
    expression = np.arange(cells * 3, dtype=np.float32).reshape(cells, 3) + 1
    probabilities = np.asarray(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9], [0.6, 0.4], [0.4, 0.6]],
        dtype=np.float32,
    )
    prediction = PredictionBundle(
        nucleus_ids=np.asarray(["n%d" % index for index in range(cells)]),
        coordinates_um=np.zeros((cells, 2), dtype=np.float32),
        type_probabilities=probabilities,
        type_names=np.asarray(["A", "B"]),
        labels=probabilities.argmax(axis=1),
        prototype_probabilities=probabilities,
        prototype_ids=np.asarray(["pA", "pB"]),
        latent_mean=np.zeros((cells, 2), dtype=np.float32),
        latent_variance=np.ones((cells, 2), dtype=np.float32),
        expression_mean=expression,
        expression_lower=expression - 0.1,
        expression_upper=expression + 0.1,
        gene_names=np.asarray(["g1", "g2", "g3"]),
        unknown_probability=np.zeros(cells, dtype=np.float32),
        abstain_score=np.zeros(cells, dtype=np.float32),
        abstain=np.zeros(cells, dtype=bool),
        ood_score=np.zeros(cells, dtype=np.float32),
        refinement_round=0,
        sample_id="sample1",
        donor_id="donor1",
        slide_id="slide1",
        checkpoint_sha256="a" * 64,
        prototype_sha256="b" * 64,
        histology_sha256="c" * 64,
        latent_space_id="latent-test",
        model_version="test",
        expression_space_id="log1p-cpm-10000-v1",
        inference_seed=17,
        latent_samples=20,
        probability_threshold=0.6,
        artifact_threshold=0.5,
    )
    prediction_path = tmp_path / "prediction.npz"
    prediction.to_npz(prediction_path)
    spot_index = np.asarray([0, 0, 1, 1, 1, -1], dtype=np.int64)
    observed_expression = np.log1p(
        np.vstack(
            (
                np.expm1(expression[:2]).mean(axis=0),
                np.expm1(expression[2:5]).mean(axis=0),
            )
        )
    )
    observed_composition = np.vstack(
        (probabilities[:2].mean(axis=0), probabilities[2:5].mean(axis=0))
    )
    truth_path = tmp_path / "spatial_truth.npz"
    np.savez_compressed(
        truth_path,
        observed_expression=np.vstack((observed_expression[:, ::-1], np.zeros((1, 3)))),
        gene_names=np.asarray(["g3", "g2", "g1"]),
        spot_ids=np.asarray(["s1", "s2", "empty"]),
        nucleus_ids=prediction.nucleus_ids,
        nucleus_spot_index=spot_index,
        observed_composition=np.vstack((observed_composition[:, ::-1], np.zeros((1, 2)))),
        type_names=np.asarray(["B", "A"]),
        expression_space_id=np.asarray("log1p-cpm-10000-v1"),
    )
    metrics_path = tmp_path / "spatial_metrics.json"
    assert (
        main(
            [
                "evaluate-spatial",
                "--predictions",
                str(prediction_path),
                "--truth",
                str(truth_path),
                "--output",
                str(metrics_path),
            ]
        )
        == 0
    )
    metrics = json.loads(metrics_path.read_text())
    assert metrics["spots_evaluated"] == 2
    assert metrics["empty_spots_ignored"] == 1
    assert metrics["expression"]["median_gene_mse"] == pytest.approx(0.0)
    assert metrics["composition"]["rmse"] == pytest.approx(0.0, abs=1e-7)

    with np.load(truth_path, allow_pickle=False) as archive:
        reordered_payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    reordered_payload["nucleus_ids"] = prediction.nucleus_ids[::-1]
    reordered_truth = tmp_path / "reordered_truth.npz"
    np.savez_compressed(reordered_truth, **reordered_payload)
    with pytest.raises(SystemExit):
        main(
            [
                "evaluate-spatial",
                "--predictions",
                str(prediction_path),
                "--truth",
                str(reordered_truth),
            ]
        )


def test_evaluate_spatial_coverage_runs_frozen_full_endpoint(tmp_path):
    cells = 6
    expression = np.log1p(
        np.asarray(
            [[1.0, 2.0], [3.0, 4.0], [4.0, 6.0], [8.0, 10.0], [2.0, 8.0], [6.0, 2.0]],
            dtype=np.float32,
        )
    )
    probabilities = np.tile(np.asarray([[0.75, 0.25]], dtype=np.float32), (cells, 1))
    prediction = PredictionBundle(
        nucleus_ids=np.asarray(["n%d" % index for index in range(cells)]),
        coordinates_um=np.zeros((cells, 2), dtype=np.float32),
        type_probabilities=probabilities,
        type_names=np.asarray(["A", "B"]),
        labels=np.zeros(cells, dtype=np.int64),
        prototype_probabilities=probabilities,
        prototype_ids=np.asarray(["pA", "pB"]),
        latent_mean=np.zeros((cells, 2), dtype=np.float32),
        latent_variance=np.ones((cells, 2), dtype=np.float32),
        expression_mean=expression,
        expression_lower=np.maximum(expression - 0.1, 0.0),
        expression_upper=expression + 0.1,
        gene_names=np.asarray(["g1", "g2"]),
        unknown_probability=np.zeros(cells, dtype=np.float32),
        abstain_score=np.zeros(cells, dtype=np.float32),
        abstain=np.zeros(cells, dtype=bool),
        ood_score=np.zeros(cells, dtype=np.float32),
        refinement_round=0,
        sample_id="sample1",
        donor_id="donor1",
        slide_id="slide1",
        checkpoint_sha256="a" * 64,
        prototype_sha256="b" * 64,
        histology_sha256="c" * 64,
        latent_space_id="latent-test",
        model_version="test",
        expression_space_id="log1p-cpm-10000-v1",
        inference_seed=17,
        latent_samples=20,
        probability_threshold=0.6,
        artifact_threshold=0.5,
    )
    prediction_path = tmp_path / "prediction_coverage.npz"
    prediction.to_npz(prediction_path)
    spot_index = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    rna_mass = np.asarray([1.0, 3.0, 2.0, 1.0, 1.0, 2.0])
    linear = np.expm1(expression[:, ::-1])
    observed = np.vstack(
        [
            np.log1p(
                np.average(linear[spot_index == spot], axis=0, weights=rna_mass[spot_index == spot])
            )
            for spot in range(3)
        ]
    )
    truth_path = tmp_path / "coverage_truth.npz"
    SpatialTruthArtifact(
        observed_expression=observed,
        gene_names=np.asarray(["g2", "g1"]),
        spot_ids=np.asarray(["s1", "s2", "s3"]),
        nucleus_ids=prediction.nucleus_ids,
        nucleus_spot_index=spot_index,
        spot_library_sizes=np.ones(3, dtype=np.float64),
        spot_coordinates_px=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        nucleus_spot_distance_px=np.zeros(cells, dtype=np.float64),
        analysis_role="locked_validation",
        cohort_id="test-cohort",
        donor_id="donor1",
        specimen_id="sample1",
        block_id="block1",
        section_id="section1",
        outer_fold="heldout",
        inner_fold="none",
        barcode_suffix_policy="exact",
        spot_radius_px=1.0,
        source_artifacts=np.asarray(["locked-counts", "frozen-manifest"]),
        source_sha256=np.asarray(["d" * 64, "e" * 64]),
        source_roles=np.asarray(["locked_spatial_counts", "shared_manifest"]),
        expression_space_id=prediction.expression_space_id,
    ).save_npz(truth_path)
    endpoint_path = tmp_path / "coverage_endpoint.npz"
    np.savez_compressed(
        endpoint_path,
        __contract__=np.asarray("heir.coverage_endpoint_input"),
        __version__=np.asarray(1, dtype=np.int64),
        endpoint=np.asarray("full_coverage_type_mean_fallback"),
        nucleus_ids=prediction.nucleus_ids,
        gene_names=np.asarray(["g2", "g1"]),
        spot_ids=np.asarray(["s1", "s2", "s3"]),
        nucleus_spot_index=spot_index,
        evaluation_spot_mask=np.ones(3, dtype=bool),
        cell_rna_mass=rna_mass,
        frozen_type_index=np.zeros(cells, dtype=np.int64),
        type_names=np.asarray(["A", "B"]),
        frozen_type_mean_log_expression=np.log1p(np.asarray([[3.0, 2.0], [5.0, 4.0]])),
    )
    output = tmp_path / "coverage_metrics.json"
    assert (
        main(
            [
                "evaluate-spatial-coverage",
                "--predictions",
                str(prediction_path),
                "--truth",
                str(truth_path),
                "--endpoint-input",
                str(endpoint_path),
                "--endpoint-input-sha256",
                _sha256(str(endpoint_path)),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    method = report["methods"]["HEIR"]
    assert method["coverage"]["endpoint"] == "full_coverage_type_mean_fallback"
    assert method["coverage"]["realized_coverage"] == pytest.approx(1.0)
    assert method["summary"]["median_gene_mse"] == pytest.approx(0.0, abs=1e-12)
    assert report["claim_scope"]["historical_report_rewrite"] is False
    assert report["provenance"]["aggregation_constructed_before_truth_values_loaded"] is True
    assert report["provenance"]["endpoint_input_hash_asserted_before_load"] is True
    assert (
        report["provenance"]["comparison_design_hash_asserted_before_truth_values_loaded"] is False
    )
    assert report["claim_scope"]["eligible_for_paired_method_comparisons"] is False
    assert set(report["provenance"]["source_sha256"]) == {
        "heir.cli",
        "heir.evaluation.coverage",
        "heir.inference",
        "heir.data.spatial_truth",
    }
    assert len(report["provenance"]["endpoint_input_sha256"]) == 64

    prediction_bytes = prediction_path.read_bytes()
    collision_args = build_parser().parse_args(
        [
            "evaluate-spatial-coverage",
            "--predictions",
            str(prediction_path),
            "--truth",
            str(truth_path),
            "--endpoint-input",
            str(endpoint_path),
            "--output",
            str(prediction_path),
        ]
    )
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        collision_args.func(collision_args)
    assert prediction_path.read_bytes() == prediction_bytes

    truth_bytes = truth_path.read_bytes()
    aggregate_collision_args = build_parser().parse_args(
        [
            "evaluate-spatial-coverage",
            "--predictions",
            str(prediction_path),
            "--truth",
            str(truth_path),
            "--endpoint-input",
            str(endpoint_path),
            "--output",
            str(tmp_path / "collision-report.json"),
            "--aggregates-output",
            str(truth_path),
        ]
    )
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        aggregate_collision_args.func(aggregate_collision_args)
    assert truth_path.read_bytes() == truth_bytes

    shared_output = tmp_path / "shared-coverage-output"
    duplicate_output_args = build_parser().parse_args(
        [
            "evaluate-spatial-coverage",
            "--predictions",
            str(prediction_path),
            "--truth",
            str(truth_path),
            "--endpoint-input",
            str(endpoint_path),
            "--output",
            str(shared_output),
            "--aggregates-output",
            str(shared_output),
        ]
    )
    with pytest.raises(ValueError, match="output paths collide with each other"):
        duplicate_output_args.func(duplicate_output_args)

    baseline_prediction_path = tmp_path / "baseline_prediction_coverage.npz"
    baseline_expression = expression + np.asarray(
        [[0.00], [0.05], [0.10], [0.15], [0.20], [0.25]], dtype=np.float32
    )
    replace(
        prediction,
        expression_mean=baseline_expression,
        expression_lower=np.maximum(baseline_expression - 0.1, 0.0),
        expression_upper=baseline_expression + 0.1,
        checkpoint_sha256="f" * 64,
    ).to_npz(baseline_prediction_path)
    plan_path = tmp_path / "coverage_benchmark_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "heir.coverage_benchmark_plan.v1",
                "truth": truth_path.name,
                "truth_sha256": _sha256(str(truth_path)),
                "methods": [
                    {
                        "name": "HEIR",
                        "predictions": prediction_path.name,
                        "predictions_sha256": _sha256(str(prediction_path)),
                        "endpoint_input": endpoint_path.name,
                        "endpoint_input_sha256": _sha256(str(endpoint_path)),
                    },
                    {
                        "name": "frozen-baseline",
                        "predictions": baseline_prediction_path.name,
                        "predictions_sha256": _sha256(str(baseline_prediction_path)),
                        "endpoint_input": endpoint_path.name,
                        "endpoint_input_sha256": _sha256(str(endpoint_path)),
                    },
                ],
                "comparison_pairs": [["HEIR", "frozen-baseline"]],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    planned_output = tmp_path / "planned_coverage_metrics.json"
    assert (
        main(
            [
                "evaluate-spatial-coverage",
                "--plan",
                str(plan_path),
                "--plan-sha256",
                _sha256(str(plan_path)),
                "--output",
                str(planned_output),
            ]
        )
        == 0
    )
    planned_report = json.loads(planned_output.read_text())
    assert set(planned_report["methods"]) == {"HEIR", "frozen-baseline"}
    assert len(planned_report["paired_comparisons"]) == 1
    assert planned_report["paired_comparisons"][0]["left"] == "HEIR"
    assert planned_report["paired_comparisons"][0]["right"] == "frozen-baseline"
    assert planned_report["claim_scope"]["eligible_for_paired_method_comparisons"] is True
    assert planned_report["claim_scope"]["evaluation_design"] == "locked_multi_method_plan"
    assert (
        planned_report["provenance"]["comparison_design_hash_asserted_before_truth_values_loaded"]
        is True
    )
    common_mask = planned_report["truth_gene_mask"]["sha256"]
    assert all(
        value["truth_gene_mask_sha256"] == common_mask
        for value in planned_report["methods"].values()
    )
    plan_bytes = plan_path.read_bytes()
    plan_collision_args = build_parser().parse_args(
        [
            "evaluate-spatial-coverage",
            "--plan",
            str(plan_path),
            "--plan-sha256",
            _sha256(str(plan_path)),
            "--output",
            str(plan_path),
        ]
    )
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        plan_collision_args.func(plan_collision_args)
    assert plan_path.read_bytes() == plan_bytes
    baseline_bytes = baseline_prediction_path.read_bytes()
    transitive_plan_collision_args = build_parser().parse_args(
        [
            "evaluate-spatial-coverage",
            "--plan",
            str(plan_path),
            "--plan-sha256",
            _sha256(str(plan_path)),
            "--output",
            str(baseline_prediction_path),
        ]
    )
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        transitive_plan_collision_args.func(transitive_plan_collision_args)
    assert baseline_prediction_path.read_bytes() == baseline_bytes

    with pytest.raises(SystemExit):
        main(
            [
                "evaluate-spatial-coverage",
                "--plan",
                str(plan_path),
                "--plan-sha256",
                "0" * 64,
                "--output",
                str(planned_output),
            ]
        )

    selective_endpoint_path = tmp_path / "selective_coverage_endpoint.npz"
    np.savez_compressed(
        selective_endpoint_path,
        __contract__=np.asarray("heir.coverage_endpoint_input"),
        __version__=np.asarray(1, dtype=np.int64),
        endpoint=np.asarray("fixed_coverage_selective"),
        nucleus_ids=prediction.nucleus_ids,
        gene_names=np.asarray(["g2", "g1"]),
        spot_ids=np.asarray(["s1", "s2", "s3"]),
        nucleus_spot_index=spot_index,
        evaluation_spot_mask=np.ones(3, dtype=bool),
        cell_rna_mass=rna_mass,
        uncertainty=np.asarray([0.0, 1.0, 0.1, 1.1, 0.2, 1.2]),
        target_coverage=np.asarray(0.5),
    )
    selective_output = tmp_path / "selective_coverage_metrics.json"
    assert (
        main(
            [
                "evaluate-spatial-coverage",
                "--predictions",
                str(prediction_path),
                "--truth",
                str(truth_path),
                "--endpoint-input",
                str(selective_endpoint_path),
                "--output",
                str(selective_output),
            ]
        )
        == 0
    )
    selective_report = json.loads(selective_output.read_text())
    selective_method = selective_report["methods"]["HEIR"]
    assert selective_method["coverage"]["endpoint"] == "fixed_coverage_selective"
    assert selective_method["coverage"]["realized_coverage"] == pytest.approx(0.5)

    with pytest.raises(SystemExit):
        main(
            [
                "evaluate-spatial-coverage",
                "--predictions",
                str(prediction_path),
                "--truth",
                str(truth_path),
                "--endpoint-input",
                str(endpoint_path),
                "--endpoint-input-sha256",
                "0" * 64,
                "--output",
                str(output),
            ]
        )
