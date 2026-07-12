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

from heir.cli import (
    _load_refinement_views,
    _log_normalize,
    _sha256,
    _validate_residual_geometry_training_provenance,
    _wrong_donor_ontology_intersection,
    build_parser,
    main,
)
from heir.data import HistologyBag, PrototypeSet, RNAReference
from heir.data.manifest import MANIFEST_COLUMNS, ManifestRecord
from heir.inference import PredictionBundle
from heir.models import HEIRConfig, HEIRModel
from heir.models.rna import RNAVAE, RNAVAEConfig
from heir.prior import fit_rna_residual_geometry
from heir.training import HEIRTrainingBatch, TrainingStage
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
    assert trained_payload["metadata"]["uot_unknown_mass"] == pytest.approx(0.05)
    assert trained_payload["metadata"]["uot_unknown_mass_mode"] == "fixed"

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
