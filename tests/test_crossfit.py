"""Patient-excluding ridge fits, auditable reuse and descriptive output contracts."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_feature_probes import GENES, sample

from heir.personalized import crossfit
from heir.personalized import probe_workflow as probe
from heir.personalized.protein_features import save_joint_inputs
from heir.personalized.schema import RegionalProteinFeatures


def training_samples():
    return [sample(f"p{i}", n=6 + i, shift=i * 0.2) for i in (3, 0, 4, 1, 2)]


def save_target(sample, path):
    np.savez(
        path,
        observation_ids=sample.targets.observation_ids,
        gene_ids=sample.targets.gene_ids,
        expression=sample.targets.expression,
        observed=sample.targets.observed,
        cell_ids=sample.observation.cell_ids,
        rows=sample.observation.rows,
        columns=sample.observation.columns,
        weights=sample.observation.weights,
        groups=sample.observation_groups,
    )


def deployment_fixture(tmp_path):
    training = training_samples()
    prepared, study = tmp_path / "prepared", tmp_path / "probes"
    fold = study / "uni2h/fold-0"
    fold.mkdir(parents=True)
    identities, targets = {}, {}
    for s in training:
        biopsy = s.inputs.patient_id
        folder = prepared / "inputs" / biopsy
        save_joint_inputs(s.inputs, folder, {})
        identities[biopsy] = probe._sha(folder / "metadata.json")
        target = fold / f"{biopsy}-target.npz"
        save_target(s, target)
        targets[biopsy] = {"role": "train", "path": str(target), "sha256": probe._sha(target)}
    probe._write(
        study / "identity.json",
        {
            "sources": {
                "uni2h": {"prepared_directory": str(prepared), "input_metadata_sha256": identities}
            }
        },
    )
    probe._write(fold / "target_identity.json", targets)
    fitted = probe.fit_feature_probe(training, [sample("validation")], GENES, GENES, [10.0])
    fitted.save(fold / "probe.bundle")
    return study, fold / "probe.bundle", training


def test_fixed_solver_is_bit_exact_existing_solver_without_validation_selection():
    training = training_samples()
    old = probe.fit_feature_probe(training, [sample("validation")], GENES, GENES, [10.0])
    fixed = probe.fit_fixed_feature_probe(training, GENES, GENES, alpha=10.0)
    for name in ("coefficients", "intercept", "feature_mean", "feature_scale"):
        np.testing.assert_array_equal(getattr(fixed, name), getattr(old, name))
    np.testing.assert_array_equal(
        fixed.predict(training[0].inputs), old.predict(training[0].inputs)
    )
    assert fixed.validation_patient_ids == ()
    assert fixed.selection["validation"] == []
    assert fixed.selection["candidate_solutions"] == 1
    assert fixed.selection["additional_refits"] == 0


def test_crossfit_sorted_patient_folds_exclude_all_specimens_and_preserve_fit_order():
    training = training_samples()
    original = training[1]
    second = replace(
        original,
        inputs=replace(original.inputs, specimen_id="second"),
        targets=replace(original.targets, specimen_id="second"),
    )
    training.append(second)
    fitted = crossfit.crossfit_feature_probe(training, GENES, GENES)
    assert [r["heldout_patient_ids"] for r in fitted.folds] == [["p0", "p3"], ["p1", "p4"], ["p2"]]
    for fold, model in zip(fitted.folds, fitted.models):
        heldout = set(fold["heldout_patient_ids"])
        expected = probe.fit_fixed_feature_probe(
            [s for s in training if s.inputs.patient_id not in heldout], GENES, GENES
        )
        np.testing.assert_array_equal(model.coefficients, expected.coefficients)
        assert set(model.training_patient_ids) == set(fitted.training_patient_ids) - heldout
    assert fitted.to_metadata()["randomness_used"] is False
    assert fitted.to_metadata()["penalty_searches"] == 0
    assert len(fitted.predictions) == 6


def test_heldout_patient_outcomes_cannot_change_own_baseline_or_scaler():
    training = training_samples()
    fitted = crossfit.crossfit_feature_probe(training, GENES, GENES)
    changed = list(training)
    changed[1] = replace(
        changed[1],
        targets=replace(changed[1].targets, expression=changed[1].targets.expression * [9, 2, 1]),
    )
    other = crossfit.crossfit_feature_probe(changed, GENES, GENES)
    np.testing.assert_array_equal(
        fitted.prediction_for(training[1].inputs), other.prediction_for(training[1].inputs)
    )
    np.testing.assert_array_equal(fitted.models[0].coefficients, other.models[0].coefficients)
    np.testing.assert_array_equal(fitted.models[0].feature_mean, other.models[0].feature_mean)
    changed[1] = replace(
        changed[1],
        inputs=replace(changed[1].inputs, image_features=changed[1].inputs.image_features + 100),
    )
    shifted = crossfit.crossfit_feature_probe(changed, GENES, GENES)
    np.testing.assert_array_equal(fitted.models[0].feature_mean, shifted.models[0].feature_mean)
    np.testing.assert_array_equal(fitted.models[0].coefficients, shifted.models[0].coefficients)


def test_crossfit_roundtrip_bound_axes_and_predictions_are_copies(tmp_path):
    training = training_samples()
    fitted = crossfit.crossfit_feature_probe(training, GENES, GENES, output=tmp_path / "inner")
    restored = crossfit.load_crossfit_baseline(tmp_path / "inner")
    inputs = training[0].inputs
    protein = RegionalProteinFeatures(
        inputs.cell_ids,
        ("m",),
        np.ones((len(inputs.cell_ids), 1)),
        np.ones((len(inputs.cell_ids), 1)),
        np.ones((len(inputs.cell_ids), 1), bool),
        np.ones((len(inputs.cell_ids), 1), int),
    )
    values = restored.prediction_for(replace(inputs, protein_features=protein))
    np.testing.assert_array_equal(values, fitted.prediction_for(inputs))
    values[:] = 0
    assert np.any(restored.prediction_for(inputs))
    assert all(not value.flags.writeable for value in restored.predictions.values())
    assert restored.to_metadata() == fitted.to_metadata()
    with pytest.raises(ValueError, match="exact held-out image"):
        restored.prediction_for(replace(inputs, image_features=inputs.image_features + 1))
    with (tmp_path / "inner/prediction-000.npz").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="inventory changed"):
        crossfit.load_crossfit_baseline(tmp_path / "inner")


@pytest.mark.parametrize("kwargs", [{"alpha": 1}, {"seed": 1}, {"n_splits": 2}])
def test_crossfit_rejects_unfrozen_protocol(kwargs):
    with pytest.raises(ValueError, match="fixed alpha10"):
        crossfit.crossfit_feature_probe(training_samples(), GENES, GENES, **kwargs)


def test_deployment_reuse_checks_sources_and_normal_equations_without_any_solve(
    tmp_path, monkeypatch
):
    study, bundle, training = deployment_fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("reuse must not fit, decompose or solve")

    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    model, receipt = crossfit.verify_deployment_probe(
        bundle, training, GENES, GENES, validation_patient_ids=("validation",), probe_study=study
    )
    assert model.alpha == 10
    assert receipt["feature_scaler_exact"]
    assert receipt["new_readouts_fitted"] == 0
    assert not receipt["decomposition_or_linear_solve_performed"]
    assert len(receipt["source_records"]) == 5
    assert max(r["scaled_slope_residual_max"] for r in receipt["normal_equation_residuals"]) < 1e-12
    model.coefficients[0, 0] += 0.1
    with (bundle / "readout.npz").open("wb") as handle:
        np.savez(
            handle,
            feature_mean=model.feature_mean,
            feature_scale=model.feature_scale,
            coefficients=model.coefficients,
            intercept=model.intercept,
        )
    inventory = json.loads((bundle / "sha256.json").read_text())
    inventory["readout.npz"] = probe._sha(bundle / "readout.npz")
    metadata = json.loads((bundle / "probe.json").read_text())
    metadata["readout_sha256"] = inventory["readout.npz"]
    (bundle / "probe.json").write_text(json.dumps(metadata))
    inventory["probe.json"] = probe._sha(bundle / "probe.json")
    (bundle / "sha256.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="normal equations"):
        crossfit.verify_deployment_probe(
            bundle, training, GENES, GENES, validation_patient_ids=("validation",)
        )


def test_deployment_reuse_rejects_changed_source_values_and_partition(tmp_path):
    _, bundle, training = deployment_fixture(tmp_path)
    with pytest.raises(ValueError, match="partition"):
        crossfit.verify_deployment_probe(bundle, training, GENES, GENES)
    changed = list(training)
    changed[0] = replace(
        changed[0],
        targets=replace(changed[0].targets, expression=changed[0].targets.expression * 2),
    )
    with pytest.raises(ValueError, match="targets or observation"):
        crossfit.verify_deployment_probe(
            bundle, changed, GENES, GENES, validation_patient_ids=("validation",)
        )
    changed[0] = replace(
        training[0],
        inputs=replace(training[0].inputs, image_features=training[0].inputs.image_features + 1),
    )
    with pytest.raises(ValueError, match="image values or axes"):
        crossfit.verify_deployment_probe(
            bundle, changed, GENES, GENES, validation_patient_ids=("validation",)
        )


def test_output_contract_signed_absolute_mass_and_outcome_order_are_descriptive_only():
    values = np.array([[-1, np.log1p(15000), 20], [0, np.log1p(5000), -20.0]])
    original = values.copy()
    report = probe.normalized_log_output_contract(values, GENES, ("a", "b"))
    deviation = np.expm1(values[:, :2]).sum(1) - 10000
    assert report["negative_fraction"] == 0.25
    assert report["above_log10001_fraction"] == 0.25
    assert report["mean_signed_implied_mass_deviation"] == pytest.approx(deviation.mean())
    assert report["mean_absolute_implied_mass_deviation"] == pytest.approx(np.abs(deviation).mean())
    assert report["mean_absolute_implied_mass_deviation"] > abs(
        report["mean_signed_implied_mass_deviation"]
    )
    np.testing.assert_array_equal(values, original)
    reordered = probe.normalized_log_output_contract(
        values[:, [2, 0, 1]], ("c", "a", "b"), ("a", "b")
    )
    assert reordered == report
    overflow = probe.normalized_log_output_contract(
        np.array([[1000.0, 1], [0, 0]]), ("a", "b"), ("a", "b")
    )
    assert overflow["mean_signed_implied_mass_deviation"] is None
    assert overflow["implied_mass_overflow_regions"] == 1
    assert overflow["finite_region_mean_signed_mass_deviation_diagnostic"] == -10000


def test_saved_output_report_never_reads_targets_or_changes_predictions(tmp_path):
    study, directory = tmp_path / "study", tmp_path / "study/uni2h/fold-0"
    directory.mkdir(parents=True)
    training, test = sample("train"), sample("test")
    model = probe.fit_fixed_feature_probe([training], GENES, GENES)
    path = directory / "test.npz"
    ids = {"backbone": "uni2h", "outer_fold": 0, "biopsy_id": "test"}
    probe._export_probe(model, test.inputs, path, ids)
    before = probe._sha(path)
    probe._write(
        directory / "test_predictions_complete.json",
        {"test_targets_loaded": False, "prediction_sha256": {path.name: before}},
    )
    probe._write(study / "exports.json", [{"path": str(path), **ids}])
    probe._write(study / "identity.json", {"fixture": "no target files exist"})
    report = probe.report_saved_probe_outputs(study, tmp_path / "report")
    assert probe._sha(path) == before
    assert report["fitting_performed"] is False and report["targets_loaded"] is False
    assert report["backbones"]["uni2h"]["n_patients"] == 1
    assert report["backbones"]["uni2h"]["n_regions"] == 12
    assert (tmp_path / "report/probe_output_contract.tsv").is_file()
