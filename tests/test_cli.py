import csv
import gzip
import hashlib
import io
import json
import tarfile
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy import sparse

from heir.cli import _log_normalize, _sha256, main
from heir.data import HistologyBag, PrototypeSet, RNAReference
from heir.data.manifest import MANIFEST_COLUMNS, ManifestRecord
from heir.inference import PredictionBundle
from heir.models.rna import RNAVAE, RNAVAEConfig
from heir.training import HEIRTrainingBatch, TrainingStage
from heir.uncertainty import MahalanobisOOD


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


def test_assemble_batch_uses_ood_mask_as_default_unknown_targets(tmp_path):
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
    output = tmp_path / "batch_with_unknown_targets.npz"
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
    assert batch.unknown_targets is not None
    np.testing.assert_array_equal(batch.ood_mask.numpy(), expected)
    np.testing.assert_array_equal(batch.unknown_targets.numpy(), expected.astype(np.float32))
    assert str(ood.resolve()) in batch.source_artifacts
    source_index = batch.source_artifacts.index(str(ood.resolve()))
    assert batch.source_sha256[source_index] == hashlib.sha256(ood.read_bytes()).hexdigest()


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
                "--allow-no-view-agreement",
                "--allow-split-overlap",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (refined / "heir_refined.pt").is_file()
    assert (refined / "refinement.json").is_file()
    assert (refined / "prototypes" / "donor1__sample1.npz").is_file()


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
