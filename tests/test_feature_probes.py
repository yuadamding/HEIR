"""Frozen-feature ridge uses only training transforms and scores direct log outputs."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from heir.personalized import probe_workflow as probe
from heir.personalized.joint_model import JointModel, JointModelConfig
from heir.personalized.observation import ObservationOperator
from heir.personalized.protein_features import save_joint_inputs
from heir.personalized.schema import JointInputs, RegionalProteinFeatures, SpatialTargets
from heir.personalized.serialization import save_prediction
from heir.personalized.train import TrainingSample, make_patient_folds

GENES = ("a", "b", "c")
PIN = "a" * 64


def sample(patient, n=12, feature_dim=3, pin=PIN, shift=0.0):
    x = np.column_stack((np.linspace(-1, 1, n), np.cos(np.arange(n)), np.sin(np.arange(n))))
    features = np.zeros((n, feature_dim))
    features[:, :3] = x + shift
    inputs = JointInputs(
        patient_id=patient,
        specimen_id=patient + "-s",
        cell_ids=tuple(f"r{i}" for i in range(n)),
        coordinates_um=np.column_stack((np.arange(n), np.zeros(n))),
        image_features=features,
        image_encoder_manifest_sha256=pin,
        reference=None,
        protein_features=None,
        prediction_unit="tissue_region",
        coordinate_units="registered_pixel",
    )
    y = np.column_stack((1 + np.exp(x[:, 0]), 2 + np.exp(-x[:, 0]), 1 + np.exp(x[:, 1])))
    target = SpatialTargets(
        tuple(f"o{i}" for i in range(n)),
        GENES,
        y,
        np.ones(y.shape, bool),
        patient,
        inputs.specimen_id,
    )
    observation = ObservationOperator(
        inputs.cell_ids, target.observation_ids, np.arange(n), np.arange(n), np.ones(n)
    )
    groups = ("rep1",) * (n // 2) + ("rep2",) * (n - n // 2) if n >= 6 else ("rep1",) * n
    return TrainingSample(inputs, target, observation, groups)


def test_patient_and_replicate_balancing_handles_unequal_counts_and_missing_outcomes():
    labels = [("p1", "s1", "r1")] * 2 + [("p1", "s1", "r2")] * 8 + [("p2", "s2", "r1")]
    mask = np.ones(len(labels), bool)
    weights = probe.patient_balanced_weights(labels, mask)
    np.testing.assert_allclose(
        [weights[:2].sum(), weights[2:10].sum(), weights[10]], [0.25, 0.25, 0.5]
    )
    mask[:1] = False
    masked = probe.patient_balanced_weights(labels, mask)
    assert masked[0] == 0 and masked[1] == 0.25
    np.testing.assert_allclose(masked.sum(), 1)
    mask[-1] = False
    with pytest.raises(ValueError, match="every training patient"):
        probe.patient_balanced_weights(labels, mask)


def test_ridge_matches_weighted_normal_equations_with_unpenalized_intercept():
    training = [sample("small", 6), sample("large", 18, shift=0.5)]
    validation = [sample("validation", shift=10)]
    fitted = probe.fit_feature_probe(training, validation, GENES, GENES, [0.3])
    expected_mean = (
        training[0].inputs.image_features.mean(0) + training[1].inputs.image_features.mean(0)
    ) / 2
    np.testing.assert_allclose(fitted.feature_mean, expected_mean, atol=1e-15)
    x = np.concatenate(
        [(s.inputs.image_features - fitted.feature_mean) / fitted.feature_scale for s in training]
    )
    y = np.concatenate([probe._truth(s, GENES, GENES)[0] for s in training]).astype(float)
    labels = [
        (s.inputs.patient_id, s.inputs.specimen_id, group)
        for s in training
        for group in s.observation_groups
    ]
    w = probe.patient_balanced_weights(labels, np.ones(len(x), bool))
    xc, yc = x - w @ x, y - w @ y
    expected = np.linalg.solve(xc.T @ (w[:, None] * xc) + 0.3 * np.eye(3), xc.T @ (w[:, None] * yc))
    np.testing.assert_allclose(fitted.coefficients, expected, atol=1e-13)
    np.testing.assert_allclose(fitted.intercept, w @ y - (w @ x) @ expected, atol=1e-13)
    assert fitted.selection["candidate_solutions"] == 1
    assert fitted.selection["additional_refits"] == 0
    assert fitted.selection["weighted_design_decompositions"] == 1
    changed = replace(
        validation[0],
        targets=replace(
            validation[0].targets, expression=validation[0].targets.expression * np.array([8, 2, 1])
        ),
    )
    other = probe.fit_feature_probe(training, [changed], GENES, GENES, [0.3])
    np.testing.assert_array_equal(fitted.coefficients, other.coefficients)
    np.testing.assert_array_equal(fitted.feature_mean, other.feature_mean)
    assert fitted.selection["validation"] != other.selection["validation"]


def test_direct_log_scoring_never_normalizes_or_clips_predictions_again():
    s = sample("patient")
    truth, _ = probe._truth(s, GENES, GENES)
    predicted = truth.astype(float) - 10
    report, genes = probe.score_probe(predicted, s, GENES, GENES)
    assert np.any(predicted < 0)
    assert report["normalized_log_mse"] == pytest.approx(100)
    assert report["squared_bias"] == pytest.approx(100)
    assert report["centered_mse"] == pytest.approx(0, abs=1e-25)
    assert report[probe.SCORE] == pytest.approx(1)
    assert report[probe.VARIABILITY] == pytest.approx(1)
    assert not report["normalize_prediction_again"]
    assert all(g["mean_bias"] == pytest.approx(-10) for g in genes)
    changed = predicted.copy()
    changed[:, 0] = 1
    failed, _ = probe.score_probe(changed, s, GENES, GENES)
    assert failed[probe.SCORE] is None
    assert failed["headline_status"] == "undefined_prediction_on_evaluable_panel"
    mask = np.array([False] * 6 + [True] * 6)
    covered, _ = probe.score_probe(predicted, s, GENES, GENES, row_mask=mask)
    assert covered["missing_coverage_replicates"] == ["rep1"]
    assert covered[probe.SCORE] is None and covered["normalized_log_mse"] is None


def test_nonunit_or_mixed_region_log_observation_bridge_rejected():
    s = sample("patient")
    for operator in (
        replace(s.observation, weights=np.ones(12) * 0.5),
        replace(
            s.observation,
            rows=np.r_[s.observation.rows, 0],
            columns=np.r_[s.observation.columns, 1],
            weights=np.r_[s.observation.weights, 1.0],
        ),
    ):
        with pytest.raises(ValueError, match="one unit-weight region"):
            probe.score_probe(np.ones((12, 3)), replace(s, observation=operator), GENES, GENES)


def test_probe_roundtrip_feature_identity_and_integrity(tmp_path):
    training, validation = sample("training"), sample("validation")
    fitted = probe.fit_feature_probe([training], [validation], GENES, GENES, [0.1, 1, 10])
    fitted.save(tmp_path / "model")
    restored = probe.FrozenFeatureProbe.load(tmp_path / "model")
    np.testing.assert_array_equal(
        restored.predict(validation.inputs), fitted.predict(validation.inputs)
    )
    with pytest.raises(ValueError, match="encoder identity"):
        restored.predict(replace(validation.inputs, image_encoder_manifest_sha256="b" * 64))
    with pytest.raises(ValueError, match="patients must be disjoint"):
        probe.fit_feature_probe([training], [training], GENES, GENES)
    path = tmp_path / "model/probe.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        probe.FrozenFeatureProbe.load(tmp_path / "model")


def test_validation_tie_selects_smallest_declared_alpha(monkeypatch):
    def tied(*args, **kwargs):
        return {
            "patient_id": args[1].inputs.patient_id,
            probe.SCORE: 0.2,
            "normalized_log_mse": 1.0,
            "squared_bias": 0.2,
            "centered_mse": 0.8,
        }, []

    monkeypatch.setattr(probe, "score_probe", tied)
    fitted = probe.fit_feature_probe([sample("train")], [sample("validation")], GENES, GENES)
    assert fitted.alpha == 0.01
    assert fitted.selection["candidate_solutions"] == 5


def _baseline(tmp_path, s):
    from heir.personalized.joint_workflow import _score_export

    config = JointModelConfig(
        s.inputs.image_features.shape[1],
        0,
        GENES,
        GENES,
        GENES,
        modalities=("H",),
        image_encoder_manifest_sha256=s.inputs.image_encoder_manifest_sha256,
    )
    prediction = JointModel(config)(s.inputs)
    path = tmp_path / "baseline.npz"
    save_prediction(prediction, path, {})
    target = (s.targets, s.observation, s.observation_groups, {})
    report, _, _ = _score_export(path, target, config, {})
    return path, report


def test_same_axis_changed_truth_cannot_use_stale_baseline_scores(tmp_path):
    s = sample("patient")
    path, report = _baseline(tmp_path, s)
    probe._verify_baseline_truth(path, report, s, GENES, GENES, None)
    changed = replace(s, targets=replace(s.targets, expression=s.targets.expression * [2, 1, 1]))
    with pytest.raises(ValueError, match="currently loaded target values"):
        probe._verify_baseline_truth(path, report, changed, GENES, GENES, None)


def test_missing_coverage_eligibility_is_joined_by_replicate_name():
    s = sample("patient")
    truth, _ = probe._truth(s, GENES, GENES)
    current, _ = probe.score_probe(
        truth, s, GENES, GENES, row_mask=np.array([False] * 6 + [True] * 6)
    )
    baseline = {**current, "target_gene_eligibility": current["target_gene_eligibility"][::-1]}
    probe._assert_matched_support(current, baseline)
    baseline["evaluation_observation_ids"] = ["wrong"]
    with pytest.raises(ValueError, match="identical target rows"):
        probe._assert_matched_support(current, baseline)


def _fixture_protocol(tmp_path):
    from heir.personalized.joint_workflow import _score_export

    protocol = {
        "schema": "heir.frozen_feature_probe_study.v1",
        "backbones": {},
        "outer_folds": list(range(5)),
        "seed": 0,
        "alpha_grid": list(probe.ALPHAS),
        "selected_readouts": 10,
        "candidate_solutions": 50,
        "feature_dim": 1536,
        "evaluation_gene_ids": list(GENES),
        "normalization_gene_ids": list(GENES),
        "no_train_validation_refit": True,
        "prediction_scale": probe.SCALE,
        "selection": "validation_spatial_correlation_then_centered_error_then_total_mse",
    }
    folds = make_patient_folds([f"p{i}" for i in range(5)])
    for backbone, pin in (("hoptimus1", PIN), ("uni2h", "b" * 64)):
        prepared, source = tmp_path / f"{backbone}-prepared", tmp_path / f"{backbone}-source"
        prepared.mkdir()
        source.mkdir()
        with (source / "folds.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(folds[0]))
            writer.writeheader()
            writer.writerows(folds)
        protocol["folds_sha256"] = probe._sha(source / "folds.csv")
        metadata, samples = {"biopsies": {}}, {}
        for i in range(5):
            biopsy, s = f"b{i}", sample(f"p{i}", n=4, feature_dim=1536, pin=pin)
            samples[biopsy] = s
            protein = RegionalProteinFeatures(
                s.inputs.cell_ids,
                ("m",),
                np.ones((4, 1)),
                np.ones((4, 1)),
                np.ones((4, 1), bool),
                np.ones((4, 1), int),
            )
            save_joint_inputs(
                replace(s.inputs, protein_features=protein), prepared / "inputs" / biopsy, {}
            )
            metadata["biopsies"][biopsy] = {"patient_id": s.inputs.patient_id}
        (prepared / "prepared.json").write_text(json.dumps(metadata))
        config = {
            "model": {
                "gene_ids": list(GENES),
                "normalization_gene_ids": list(GENES),
                "image_encoder_manifest_sha256": pin,
            }
        }
        (source / "config.json").write_text(json.dumps(config))
        exports = []
        for fold in range(5):
            directory = source / f"fold-{fold}"
            (directory / "targets").mkdir(parents=True)
            roles = {r["patient_id"]: r["role"] for r in folds if r["outer_fold"] == fold}
            for biopsy, s in samples.items():
                np.savez(
                    directory / "targets" / f"{biopsy}.npz",
                    observation_ids=s.targets.observation_ids,
                    gene_ids=GENES,
                    expression=s.targets.expression,
                    observed=s.targets.observed,
                    cell_ids=s.observation.cell_ids,
                    rows=s.observation.rows,
                    columns=s.observation.columns,
                    weights=s.observation.weights,
                    groups=s.observation_groups,
                )
                if roles[s.inputs.patient_id] != "test":
                    continue
                model = JointModelConfig(
                    1536,
                    0,
                    GENES,
                    GENES,
                    GENES,
                    modalities=("H",),
                    image_encoder_manifest_sha256=pin,
                )
                predicted = JointModel(model)(s.inputs)
                for component in ("full", "transfer"):
                    path = directory / f"{biopsy}-{component}.npz"
                    save_prediction(predicted, path, {})
                    ids = {
                        "arm": "HRP",
                        "seed": 0,
                        "outer_fold": fold,
                        "biopsy_id": biopsy,
                        "component": component,
                    }
                    report, _, _ = _score_export(
                        path, (s.targets, s.observation, s.observation_groups, {}), model, ids
                    )
                    path.with_suffix(".common-coverage.scores.json").write_text(json.dumps(report))
                    exports.append({"path": str(path), **ids})
        (source / "exports.json").write_text(json.dumps(exports))
        identity = {
            "prepared_sha256": probe._sha(prepared / "prepared.json"),
            "experiment_config_sha256": probe._sha(source / "config.json"),
            "folds_sha256": protocol["folds_sha256"],
            "input_metadata_sha256": {
                b: probe._sha(prepared / "inputs" / b / "metadata.json") for b in samples
            },
        }
        (source / "identity.json").write_text(json.dumps(identity))
        protocol["backbones"][backbone] = {
            "prepared_directory": str(prepared),
            "source_study_directory": str(source),
            "encoder_manifest_sha256": pin,
            "source_identity_sha256": probe._sha(source / "identity.json"),
            "source_exports_sha256": probe._sha(source / "exports.json"),
        }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return path, protocol, folds


def test_full_probe_workflow_counts_candidates_and_exports_before_test_reads(tmp_path, monkeypatch):
    path, protocol, folds = _fixture_protocol(tmp_path)
    output = tmp_path / "output"
    read = probe._sample_from_source
    calls = []

    def guarded(inputs, target_path):
        target_path = Path(target_path)
        backbone = target_path.parents[2].name.removesuffix("-source")
        fold = int(target_path.parents[1].name.split("-")[-1])
        roles = {r["patient_id"]: r["role"] for r in folds if r["outer_fold"] == fold}
        if roles[inputs.patient_id] == "test":
            receipt = output / backbone / f"fold-{fold}/test_predictions_complete.json"
            assert receipt.is_file()
            assert json.loads(receipt.read_text())["test_targets_loaded"] is False
        calls.append((backbone, fold, roles[inputs.patient_id]))
        return read(inputs, target_path)

    monkeypatch.setattr(probe, "_sample_from_source", guarded)
    original_load = probe.load_joint_inputs

    def image_or_coverage_only(*args, **kwargs):
        assert "R" not in kwargs["modalities"]
        return original_load(*args, **kwargs)

    monkeypatch.setattr(probe, "load_joint_inputs", image_or_coverage_only)
    result = probe.run_feature_probes(path, output)
    assert result["selected_readouts"] == 10 and result["candidate_solutions"] == 50
    assert result["additional_refits"] == 0
    assert len(list(output.glob("*/fold-*/probe.bundle"))) == 10
    assert len(json.loads((output / "exports.json").read_text())) == 10
    assert sum(role == "test" for _, _, role in calls) == 10
    for identity_path in output.glob("*/fold-*/target_identity.json"):
        inventory = json.loads(identity_path.read_text())
        assert {r["role"] for r in inventory.values()} == {"train", "validation", "test"}
        assert all(probe._sha(r["path"]) == r["sha256"] for r in inventory.values())
    for values in result["backbones"].values():
        assert values["all_target_eligible"]["n_patients"] == 5


def test_probe_cli_and_committed_protocol():
    import argparse

    from heir.personalized.cohort_cli import add_commands

    parser = argparse.ArgumentParser()
    add_commands(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["cohort-probe", "--config", "p.json", "--output", "out"])
    assert args.command == "cohort-probe"
    path = Path(__file__).resolve().parents[1] / "configs/personalized/frozen_feature_probes.json"
    assert probe._protocol(path)["candidate_solutions"] == 50
