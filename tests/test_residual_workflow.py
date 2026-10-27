"""Residual orchestration separates scales, modalities, and held-out targets."""

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from heir.personalized import joint_workflow, probe_workflow
from heir.personalized import residual_workflow as workflow
from heir.personalized.joint_model import JointModel, JointModelConfig
from heir.personalized.probe_workflow import FrozenFeatureProbe
from heir.personalized.protein_features import save_joint_inputs
from heir.personalized.schema import JointInputs, RegionalProteinFeatures, RNAReference
from heir.personalized.serialization import save_prediction
from heir.personalized.train import make_patient_folds

GENES = tuple(f"g{i}" for i in range(9))
PIN = "a" * 64


def _input(patient, offset=0):
    n = 6
    features = np.zeros((n, 1536))
    features[:, 0] = np.linspace(-1, 1, n)
    features[:, 1] = np.cos(np.arange(n))
    ids = tuple(f"region:{i}" for i in range(n))
    reference = RNAReference(
        ("a", "b"),
        GENES,
        np.array([np.arange(1, 10) + offset, np.arange(10, 1, -1)], dtype=float),
        np.array([[0.1, 1], [1, 0.1]]),
        np.array([0.4, 0.6]),
        ("typeA", "typeB"),
        np.zeros((2, 9)),
        np.array([4, 6]),
        GENES[:2],
    )
    protein = RegionalProteinFeatures(
        ids,
        ("m",),
        features[:, :1],
        features[:, :1] + 0.2,
        np.ones((n, 1), bool),
        np.ones((n, 1), int),
    )
    return JointInputs(
        patient,
        patient + ":s",
        ids,
        np.column_stack((np.arange(n), np.zeros(n))),
        features,
        reference,
        protein,
        image_encoder_manifest_sha256=PIN,
    )


def _model():
    return JointModelConfig(
        1536,
        2,
        GENES,
        GENES,
        GENES,
        state_gene_ids=GENES[:2],
        marker_ids=("m",),
        image_encoder_manifest_sha256=PIN,
    )


def _direct_prediction(inputs, values=None):
    values = np.arange(54, dtype=float).reshape(6, 9) / 10 - 1 if values is None else values
    return SimpleNamespace(
        cell_ids=inputs.cell_ids,
        gene_ids=GENES,
        coordinates_um=inputs.coordinates,
        patient_id=inputs.patient_id,
        specimen_id=inputs.specimen_id,
        prediction_unit=inputs.prediction_unit,
        coordinate_units=inputs.coordinate_units,
        image_baseline=torch.tensor(values),
        residual=torch.zeros_like(torch.tensor(values)),
        normalized_log_expression=torch.tensor(values),
        linear_reference_transfer=None,
        diagnostics={},
    )


def test_residual_prediction_keeps_direct_scale_and_rejects_false_decomposition(tmp_path):
    inputs = _input("p")
    predicted = _direct_prediction(inputs)
    path = tmp_path / "pred.npz"
    identity = {"encoder_manifest_sha256": PIN, "residual_bound": 1.0}
    workflow.save_residual_prediction(predicted, path, identity)
    arrays, record = workflow.load_residual_prediction(path, inputs, GENES)
    assert arrays["normalized_log_expression"].min() < 0
    np.testing.assert_array_equal(
        arrays["normalized_log_expression"], predicted.normalized_log_expression
    )
    assert not record["normalize_prediction_again"]
    with np.load(path) as archive:
        assert "expression" not in archive and "transfer" not in archive
    predicted.residual[0, 0] = 1
    with pytest.raises(ValueError, match="baseline plus residual"):
        workflow.save_residual_prediction(predicted, tmp_path / "bad.npz", identity)
    with pytest.raises(ValueError, match="named geometry"):
        workflow.load_residual_prediction(
            path, replace(inputs, coordinates_um=inputs.coordinates + 1), GENES
        )


def _fixture(tmp_path, monkeypatch):
    prepared, source, baseline = (tmp_path / name for name in ("prepared", "source", "baseline"))
    for directory in (prepared, source, baseline):
        directory.mkdir()
    model = _model()
    inputs = {f"b{i}": _input(f"p{i}", offset=i) for i in range(5)}
    fold_rows = make_patient_folds([item.patient_id for item in inputs.values()])
    with (source / "folds.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)
    metadata = {
        "biopsies": {key: {"patient_id": value.patient_id} for key, value in inputs.items()},
        "folds_sha256": workflow._sha(source / "folds.csv"),
    }
    for biopsy, value in inputs.items():
        save_joint_inputs(value, prepared / "inputs" / biopsy, {})
    (prepared / "prepared.json").write_text(json.dumps(metadata))
    (source / "config.json").write_text(json.dumps({"model": model.to_dict()}))
    monkeypatch.setattr(workflow, "_prepared", lambda _: (metadata, {}, model))
    exports, baseline_folds = [], {}
    for fold in range(5):
        directory, old_probe = source / f"fold-{fold}", baseline / f"fold-{fold}"
        (directory / "targets").mkdir(parents=True)
        old_probe.mkdir()
        roles = {row["patient_id"]: row["role"] for row in fold_rows if row["outer_fold"] == fold}
        identities = {}
        for biopsy, item in inputs.items():
            path = directory / "targets" / f"{biopsy}.npz"
            x = item.image_features[:, 0]
            y = np.column_stack([2 + np.exp(x * (i + 1) / 9) for i in range(9)])
            np.savez(
                path,
                observation_ids=[f"o{i}" for i in range(6)],
                gene_ids=GENES,
                expression=y,
                observed=np.ones(y.shape, bool),
                cell_ids=item.cell_ids,
                rows=np.arange(6),
                columns=np.arange(6),
                weights=np.ones(6),
                groups=("rep1",) * 3 + ("rep2",) * 3,
            )
            identities[biopsy] = {
                "path": str(path),
                "role": roles[item.patient_id],
                "sha256": workflow._sha(path),
            }
            if roles[item.patient_id] == "test":
                prediction = JointModel(model)(item, correction_enabled=False)
                prediction = replace(prediction, expression=prediction.transfer)
                export = directory / f"{biopsy}.transfer.npz"
                save_prediction(prediction, export, {})
                h = replace(item, reference=None, protein_features=None)
                sample = probe_workflow._sample_from_source(h, path)
                ids = {
                    "arm": "HRP",
                    "component": "transfer",
                    "outer_fold": fold,
                    "seed": 0,
                    "biopsy_id": biopsy,
                }
                report, _, _ = joint_workflow._score_export(
                    export,
                    (sample.targets, sample.observation, sample.observation_groups, {}),
                    model,
                    ids,
                )
                export.with_suffix(".common-coverage.scores.json").write_text(json.dumps(report))
                exports.append({**ids, "path": str(export)})
        coefficients = np.zeros((1536, 9))
        coefficients[0] = np.linspace(-0.3, 0.3, 9)
        fitted = FrozenFeatureProbe(
            np.zeros(1536),
            np.ones(1536),
            coefficients,
            np.full(9, 6.0),
            GENES,
            GENES,
            PIN,
            "tissue_region",
            "registered_pixel",
            10,
            tuple(sorted(p for p, role in roles.items() if role == "train")),
            tuple(sorted(p for p, role in roles.items() if role == "validation")),
            {},
        )
        fitted.save(old_probe / "probe.bundle")
        (old_probe / "target_identity.json").write_text(json.dumps(identities))
        baseline_folds[str(fold)] = {
            "bundle_inventory_sha256": workflow._sha(old_probe / "probe.bundle/sha256.json"),
            "target_identity_sha256": workflow._sha(old_probe / "target_identity.json"),
        }
    (source / "exports.json").write_text(json.dumps(exports))
    identity = {
        "prepared_sha256": workflow._sha(prepared / "prepared.json"),
        "experiment_config_sha256": workflow._sha(source / "config.json"),
        "folds_sha256": metadata["folds_sha256"],
        "input_metadata_sha256": {
            b: workflow._sha(prepared / "inputs" / b / "metadata.json") for b in inputs
        },
    }
    (source / "identity.json").write_text(json.dumps(identity))
    protocol = {
        "schema": "heir.residual_study.v1",
        "prepared_directory": str(prepared),
        "source_study_directory": str(source),
        "baseline_directory": str(baseline),
        "source_identity_sha256": workflow._sha(source / "identity.json"),
        "source_exports_sha256": workflow._sha(source / "exports.json"),
        "prepared_sha256": identity["prepared_sha256"],
        "folds_sha256": metadata["folds_sha256"],
        "baseline_folds": baseline_folds,
        "outer_folds": list(range(5)),
        "seed": 0,
        "alpha": 10,
        "inner_splits": 3,
        "arms": list(workflow.ARMS),
        "expected_neural_fits": 30,
        "expected_inner_readouts": 15,
        "expected_reused_deployment_readouts": 5,
        "model": model.to_dict(),
        "training": {
            "epochs": 100,
            "learning_rate": 0.001,
            "centered_weight": 0.25,
            "residual_weight": 0.001,
            "residual_bound": 1.0,
            "device": "cpu",
        },
        "zero_residual_candidate": True,
        "prediction_scale": probe_workflow.SCALE,
        "transfer_scale": "linear_relative_expression",
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return path, protocol, fold_rows


def test_all_six_arms_share_crossfit_and_export_before_test_targets(tmp_path, monkeypatch):
    from heir.personalized import crossfit as crossfit_module
    from heir.personalized import residual_train

    config, protocol, fold_rows = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    fit_calls, crossfits, saved = [], [], {}
    active_generic = [False]
    original_input = workflow._inputs
    original_load = joint_workflow.load_joint_inputs

    def guarded_load(*args, **kwargs):
        if active_generic[0]:
            assert "R" not in kwargs["modalities"]
        return original_load(*args, **kwargs)

    def arm_inputs(prepared, biopsy, arm, pooled=None):
        active_generic[0] = arm == "HRP_generic"
        try:
            return original_input(prepared, biopsy, arm, pooled)
        finally:
            active_generic[0] = False

    monkeypatch.setattr(joint_workflow, "load_joint_inputs", guarded_load)
    monkeypatch.setattr(workflow, "_inputs", arm_inputs)
    original_target = workflow._sample_from_source

    def guarded_target(inputs, path):
        fold = int(path.parents[1].name.split("-")[-1])
        role = next(
            row["role"]
            for row in fold_rows
            if row["outer_fold"] == fold and row["patient_id"] == inputs.patient_id
        )
        if role == "test":
            receipt = json.loads(
                (output / f"fold-{fold}/test_predictions_complete.json").read_text()
            )
            assert receipt["neural_fits"] == 6 and not receipt["test_targets_loaded"]
            assert len(receipt["prediction_sha256"]) == 11
        return original_target(inputs, path)

    monkeypatch.setattr(workflow, "_sample_from_source", guarded_target)

    def verify(bundle, training, genes, panel, **kwargs):
        baseline = FrozenFeatureProbe.load(bundle)
        assert set(baseline.training_patient_ids) == {s.inputs.patient_id for s in training}
        return baseline, {"reused": True}

    def crossfit(training, genes, panel, **kwargs):
        kwargs["output"].mkdir()
        result = SimpleNamespace(
            models=(None,) * 3,
            predictions={
                (s.inputs.patient_id, s.inputs.specimen_id): np.zeros((len(s.inputs.cell_ids), 9))
                for s in training
            },
        )
        crossfits.append(result)
        return result

    class Fitted:
        def __init__(self, model, baseline, generic):
            self.model, self.baseline, self.generic = JointModel(model), baseline, generic
            self.selection = {
                "selected_epoch": 0,
                "selected_zero_residual": True,
                "mapper_training_status": "untrained_selected_epoch_zero"
                if "R" in model.modalities
                else "not_applicable_no_RNA",
                "mapper_parameters_changed": False,
            }

        def save(self, path):
            path.mkdir()
            saved[str(path)] = self

        @classmethod
        def load(cls, path):
            return saved[str(path)]

        def predict(self, inputs):
            direct = self.baseline.predict(replace(inputs, reference=None, protein_features=None))
            result = _direct_prediction(inputs, direct)
            if "R" in self.model.config.modalities:
                prediction = self.model(inputs, correction_enabled=False)
                result.linear_reference_transfer = replace(
                    prediction, expression=prediction.transfer
                )
            return result

    def fit(training, validation, model, **kwargs):
        assert kwargs["training_baselines"] is crossfits[-1]
        fold, arm = kwargs["preparation"]["outer_fold"], kwargs["preparation"]["arm"]
        for sample in [*training, *validation]:
            assert (sample.inputs.reference is not None) == ("R" in model.modalities)
            assert (sample.inputs.protein_features is not None) == ("P" in model.modalities)
        if arm == "HRP_generic":
            expected = sorted({sample.inputs.patient_id for sample in training})
            assert kwargs["preparation"]["reference_pool"]["training_patient_ids"] == expected
            assert all(
                sample.inputs.reference is kwargs["reference_override"]
                for sample in [*training, *validation]
            )
        fit_calls.append((fold, arm, id(kwargs["training_baselines"]), id(kwargs["baseline"])))
        return Fitted(model, kwargs["baseline"], kwargs["reference_override"])

    monkeypatch.setattr(crossfit_module, "verify_deployment_probe", verify)
    monkeypatch.setattr(crossfit_module, "crossfit_feature_probe", crossfit)
    monkeypatch.setattr(residual_train, "fit_residual", fit)
    monkeypatch.setattr(residual_train, "FittedResidualModel", Fitted)
    report = workflow.run_residual_study(config, output)
    assert report["neural_fits"] == 30 and report["inner_readouts"] == 15
    assert report["export_counts"] == {"baseline": 5, "final": 30, "transfer": 20}
    assert len(fit_calls) == 30 and len(crossfits) == 5
    for fold in range(5):
        calls = [row for row in fit_calls if row[0] == fold]
        assert len({row[2:] for row in calls}) == 1
    assert all(row["selection"]["selected_zero_residual"] for row in report["selections"])
    assert "HRP:final-minus-HRP_generic:final" in report["paired_contrasts"][workflow.SUPPORTS[0]]
    assert (
        "HRP:transfer-minus-historical_HRP:transfer"
        in report["paired_contrasts"][workflow.SUPPORTS[0]]
    )
    with pytest.raises(FileExistsError):
        workflow.run_residual_study(config, output)


def test_changed_historical_source_is_rejected_before_any_fit(tmp_path, monkeypatch):
    config, protocol, _ = _fixture(tmp_path, monkeypatch)
    source = Path(protocol["source_study_directory"])
    (source / "exports.json").write_text("[]")
    with pytest.raises(ValueError, match="source contract checksum"):
        workflow.run_residual_study(config, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_actual_fitted_family_exports_direct_outcome_and_separate_linear_transfer(tmp_path):
    from test_residual_training import training_fixture

    from heir.personalized.residual_train import FittedResidualModel, fit_residual

    training, validation, config, baseline, crossfit = training_fixture()
    fitted = fit_residual(
        training, validation, config, training_baselines=crossfit, baseline=baseline, epochs=1
    )
    fitted.save(tmp_path / "bundle")
    restored = FittedResidualModel.load(tmp_path / "bundle")
    inputs = validation[0].inputs
    prediction = restored.predict(inputs)
    path = tmp_path / "final.npz"
    workflow.save_residual_prediction(
        prediction,
        path,
        {"encoder_manifest_sha256": inputs.image_encoder_manifest_sha256, "residual_bound": 1.0},
    )
    arrays, _ = workflow.load_residual_prediction(path, inputs, config.gene_ids)
    np.testing.assert_array_equal(
        arrays["normalized_log_expression"], prediction.normalized_log_expression.numpy()
    )
    assert prediction.linear_reference_transfer is not None
    save_prediction(prediction.linear_reference_transfer, tmp_path / "linear.npz", {})
    assert not np.array_equal(
        arrays["normalized_log_expression"], prediction.linear_reference_transfer.expression.numpy()
    )
    with pytest.raises(ValueError, match="named geometry"):
        workflow.load_residual_prediction(
            path, replace(inputs, image_encoder_manifest_sha256="f" * 64), config.gene_ids
        )


def test_residual_cli_and_fixed_protocol():
    import argparse

    from heir.personalized.cohort_cli import add_commands

    parser = argparse.ArgumentParser()
    add_commands(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["cohort-residual", "--config", "c.json", "--output", "out"])
    assert args.command == "cohort-residual"
    path = Path(__file__).resolve().parents[1] / "configs/personalized/joint_residual_uni2h.json"
    protocol, model = workflow._protocol(path)
    assert protocol["expected_neural_fits"] == 30 and model.feature_dim == 1536
