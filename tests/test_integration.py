import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from heir.config import load_config
from heir.data import ManifestValidationError, PrototypeSet, load_manifest
from heir.inference import predict_cells
from heir.models import HEIRConfig, HEIRModel
from heir.prior.prototypes import build_sample_prototypes
from heir.training.stages import StageInputs, TrainingStage


def test_checked_experiment_config_loads_relative_paths():
    config = load_config("configs/experiments/natcommun_v0_1.yaml")
    assert config.mode == "personalized"
    assert config.spatial_validation_only
    assert config.manifest.endswith("/manifests/natcommun.tsv")
    assert config.refinement.maximum_rounds == 3


def test_stage_gate_allows_public_pretraining_but_not_locked_spatial():
    image = torch.randn(2, 3)
    rna = torch.randn(2, 3)
    spatial = torch.randn(1, 3)
    StageInputs(
        histology_features=image,
        matched_rna=rna,
        target_spatial_expression=spatial,
        analysis_role="pretraining",
    ).validate(TrainingStage.GENERIC_SPATIAL_PRETRAINING)
    with pytest.raises(ValueError, match="analysis_role=pretraining"):
        StageInputs(
            histology_features=image,
            matched_rna=rna,
            target_spatial_expression=spatial,
            analysis_role="locked_validation",
        ).validate(TrainingStage.GENERIC_SPATIAL_PRETRAINING)


def test_personalized_locked_donor_can_use_rna_but_not_spatial_truth():
    image = torch.randn(2, 3)
    rna = torch.randn(2, 3)
    StageInputs(
        histology_features=image,
        matched_rna=rna,
        analysis_role="locked_validation",
    ).validate(TrainingStage.PERSONALIZED)
    with pytest.raises(ValueError, match="validation-only"):
        StageInputs(
            histology_features=image,
            matched_rna=rna,
            target_spatial_expression=torch.randn(1, 3),
            analysis_role="locked_validation",
        ).validate(TrainingStage.PERSONALIZED)


def test_accession_overlap_between_pretraining_and_locked_is_rejected(tmp_path):
    source = Path("manifests/snpatho.tsv")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
        fieldnames = list(rows[0])
    first = dict(rows[0])
    second = dict(rows[1])
    first["analysis_role"] = "pretraining"
    first["cohort_id"] = "generic_copy"
    first["donor_id"] = "generic_donor"
    first["specimen_id"] = "generic_specimen"
    first["block_id"] = "generic_block"
    first["section_id"] = "generic_section"
    first["outer_fold"] = "pretrain_fold"
    first["inner_fold"] = "pretrain_inner"
    # Retain the same GSE accessions to simulate a duplicate in another portal.
    path = tmp_path / "overlap.tsv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows((first, second))
    with pytest.raises(ManifestValidationError, match="overlaps locked"):
        load_manifest(path, resolve_paths=False)


def test_rare_types_are_not_sample_supported_by_default():
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(105, 4)).astype(np.float32)
    labels = np.asarray(["major"] * 100 + ["rare"] * 5)
    prototypes = build_sample_prototypes(
        latent,
        labels,
        sample_id="s1",
        minimum_cells=20,
        max_prototypes_per_type=3,
    )
    assert set(prototypes.cell_type_labels.tolist()) == {"major"}
    assert np.isclose(prototypes.weights.sum(), 1.0)


def test_inference_respects_model_unknown_abstention():
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
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fine_type_head.bias[0] = 20.0
        model.unknown_head.bias.fill_(20.0)
    prototypes = PrototypeSet(
        prototype_ids=np.asarray(["p0", "p1"]),
        sample_ids=np.asarray(["s", "s"]),
        cell_type_labels=np.asarray(["a", "b"]),
        means=np.zeros((2, 2), dtype=np.float32),
        variances=np.ones((2, 2), dtype=np.float32),
        weights=np.asarray([0.5, 0.5]),
        n_cells=np.asarray([10, 10]),
    )
    result = predict_cells(
        model,
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        ["n0"],
        prototypes,
        ["a", "b"],
        ["g0", "g1"],
        latent_samples=2,
        device="cpu",
    )
    assert result.unknown_probability[0] > 0.9
    assert result.abstain[0]
    assert result.labels[0] == -1
