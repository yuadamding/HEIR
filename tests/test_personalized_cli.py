"""Actual H5AD/Parquet-to-bundle pipeline and held-out target isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("anndata")
pytest.importorskip("pyarrow")

from heir.personalized.cli import main
from heir.personalized.serialization import load_prediction
from heir.personalized.synthetic import create_fixture


def test_file_workflow_never_reads_held_out_targets_for_fit_or_prediction(tmp_path):
    root = tmp_path / "fixture"
    manifests = create_fixture(root)
    heldout = [path.parent / "spatial_targets.h5ad" for path in manifests[1:]]
    contents = [path.read_bytes() for path in heldout]
    for path in heldout:
        path.unlink()
    main(["prepare", "--manifest", str(manifests[2]), "--output", str(root / "prepared")])
    assert (root / "prepared" / "nuclei_regions.svg").is_file()
    main(
        [
            "train",
            "--manifests",
            *map(str, manifests),
            "--folds",
            str(root / "folds.csv"),
            "--outer-fold",
            "0",
            "--config",
            str(root / "model.json"),
            "--epochs",
            "2",
            "--output",
            str(root / "bundle"),
        ]
    )
    command = ["predict", "--manifest", str(manifests[2]), "--bundle", str(root / "bundle")]
    main([*command, "--output", str(root / "without_targets.npz")])
    for path, content in zip(heldout, contents):
        path.write_bytes(content)
    main([*command, "--output", str(root / "with_targets.npz")])
    absent, provenance = load_prediction(root / "without_targets.npz")
    present, _ = load_prediction(root / "with_targets.npz")
    np.testing.assert_array_equal(absent.assignment, present.assignment)
    np.testing.assert_array_equal(absent.expression, present.expression)
    assert provenance["training_patient_ids"] == ["SYNTHETIC_P0"]
    assert not provenance["target_payload_read"]
    bundle = json.loads((root / "bundle" / "bundle.json").read_text())
    sources = bundle["preparation"]["training_manifests"]
    assert len(sources) == 1
    assert sources[0]["manifest_path"] == str(manifests[0].resolve())
    assert sources[0]["patient_id"] == "SYNTHETIC_P0"
    main(
        [
            "evaluate",
            "--prediction",
            str(root / "with_targets.npz"),
            "--manifest",
            str(manifests[2]),
            "--genes",
            str(root / "genes.json"),
            "--output",
            str(root / "scores.json"),
        ]
    )
    scores = json.loads((root / "scores.json").read_text())
    assert scores["patient_id"] == "SYNTHETIC_P2"
    assert scores["requested_genes"] == 3
    assert scores["evaluation_role"] == "test"
    main([*command, "--transfer-only", "--output", str(root / "transfer.npz")])
    transfer, _ = load_prediction(root / "transfer.npz")
    np.testing.assert_array_equal(transfer.expression, transfer.transfer)


def test_cli_prediction_rejects_changed_preparation_meaning(tmp_path):
    main(["demo", "--output", str(tmp_path / "demo")])
    root = tmp_path / "demo"
    manifest_path = root / "SYNTHETIC_P2" / "specimen.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reference"]["library_size"] = 100
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="preparation settings differ"):
        main(
            [
                "predict",
                "--manifest",
                str(manifest_path),
                "--bundle",
                str(root / "bundle"),
                "--output",
                str(root / "changed.npz"),
            ]
        )
    assert not (root / "changed.npz").exists()


def test_cli_training_rejects_patient_omission_before_target_access(tmp_path):
    manifests = create_fixture(tmp_path / "fixture")
    with pytest.raises(ValueError, match="every training patient"):
        main(
            [
                "train",
                "--manifests",
                str(manifests[1]),
                "--folds",
                str(tmp_path / "fixture" / "folds.csv"),
                "--outer-fold",
                "0",
                "--config",
                str(tmp_path / "fixture" / "model.json"),
                "--output",
                str(tmp_path / "bundle"),
            ]
        )


def test_example_configuration_files_are_parseable():
    from heir.personalized.model import ModelConfig

    root = Path(__file__).resolve().parents[1]
    ModelConfig.from_dict(
        json.loads((root / "configs/personalized/model.example.json").read_text())
    )


def test_all_comparison_variants_fit_the_same_file_inputs(tmp_path):
    from dataclasses import replace

    from heir.personalized.data import load_spatial_targets, prepare_sample
    from heir.personalized.graph import LRInteraction
    from heir.personalized.model import VARIANTS, ModelConfig
    from heir.personalized.observation import ObservationOperator
    from heir.personalized.train import TrainingConfig, TrainingSample, fit

    manifest = create_fixture(tmp_path / "variants")[0]
    inputs = prepare_sample(manifest)
    sample = TrainingSample(
        inputs,
        load_spatial_targets(manifest),
        ObservationOperator.load(manifest.parent / "observation.npz"),
    )
    config = ModelConfig.from_dict(json.loads((manifest.parent.parent / "model.json").read_text()))
    settings = TrainingConfig(epochs=1)
    baseline = fit([sample], config, settings)
    frozen_assignment = baseline.predict(inputs).assignment
    for variant in VARIANTS:
        context = variant.startswith(("generic_", "lr_"))
        interaction = (LRInteraction(("GENE_A",), ("GENE_B",), radius_um=120),) if context else ()
        fitted = fit(
            [sample],
            replace(config, variant=variant, lr_interactions=interaction),
            settings,
            baseline=baseline if context else None,
        )
        prediction = fitted.predict(inputs)
        assert np.isfinite(prediction.expression.numpy()).all()
        if variant.endswith("_fixed"):
            np.testing.assert_array_equal(prediction.assignment, frozen_assignment)
        if variant != "he":
            np.testing.assert_array_equal(prediction.expression[:, 2], prediction.transfer[:, 2])
