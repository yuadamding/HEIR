"""Cross-fit-only supervision, epoch-zero selection, and standalone residual bundles."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_joint_model import arm
from test_joint_training import sample_fixture

from heir.personalized.crossfit import crossfit_feature_probe
from heir.personalized.probe_workflow import SCORE, fit_fixed_feature_probe
from heir.personalized.residual_train import FittedResidualModel, fit_residual
from heir.personalized.study import pool_reference


def training_fixture(modalities="HRP"):
    training = []
    for i in range(3):
        sample, config = sample_fixture(f"train{i}")
        inputs, config = arm(sample.inputs, config, modalities)
        inputs = replace(inputs, image_features=inputs.image_features + i * 0.5)
        expression = sample.targets.expression * np.asarray([1 + i, 1, 1 + 0.3 * i])
        training.append(
            replace(sample, inputs=inputs, targets=replace(sample.targets, expression=expression))
        )
    validation, _ = sample_fixture("validation")
    inputs, _ = arm(
        validation.inputs,
        replace(
            config,
            state_dim=2,
            state_gene_ids=("a", "b"),
            marker_ids=("m1", "m2"),
            modalities=("H", "R", "P"),
        ),
        modalities,
    )
    validation = replace(validation, inputs=inputs)
    config = replace(config, supervised_gene_ids=config.gene_ids)
    images = [
        replace(s, inputs=replace(s.inputs, reference=None, protein_features=None))
        for s in training
    ]
    baseline = fit_fixed_feature_probe(
        images, config.gene_ids, config.normalization_gene_ids, alpha=10.0
    )
    crossfit = crossfit_feature_probe(images, config.gene_ids, config.normalization_gene_ids)
    return training, [validation], config, baseline, crossfit


@pytest.mark.parametrize("modalities", ["H", "HR", "HP", "HRP"])
def test_fitting_and_roundtrip_keep_float64_frozen_path_and_training_only_transforms(
    tmp_path, modalities
):
    training, validation, config, baseline, crossfit = training_fixture(modalities)
    before = baseline.coefficients.copy()
    fitted = fit_residual(
        training, validation, config, training_baselines=crossfit, baseline=baseline, epochs=3
    )
    assert len(fitted.history) == 4 and fitted.history[0]["epoch"] == 0
    assert fitted.training_patient_ids == ("train0", "train1", "train2")
    assert fitted.validation_patient_ids == ("validation",)
    np.testing.assert_array_equal(fitted.transformer.feature_mean, baseline.feature_mean)
    np.testing.assert_array_equal(fitted.transformer.feature_scale, baseline.feature_scale)
    np.testing.assert_array_equal(baseline.coefficients, before)
    if "P" in modalities:
        assert fitted.transformer.protein_scaler.training_patient_ids == fitted.training_patient_ids
    if "R" in modalities:
        expected = np.concatenate([s.inputs.reference.embedding for s in training]).mean(0)
        np.testing.assert_array_equal(fitted.transformer.state_mean, expected)
    prediction = fitted.predict(validation[0].inputs)
    assert prediction.normalized_log_expression.dtype == torch.float64
    fitted.save(tmp_path / "model")
    loaded = FittedResidualModel.load(tmp_path / "model")
    actual = loaded.predict(validation[0].inputs)
    for field in ("image_baseline", "residual", "normalized_log_expression"):
        torch.testing.assert_close(
            getattr(prediction, field), getattr(actual, field), rtol=0, atol=0
        )
    if prediction.linear_reference_transfer:
        torch.testing.assert_close(
            prediction.linear_reference_transfer.transfer,
            actual.linear_reference_transfer.transfer,
            rtol=0,
            atol=0,
        )
    assert loaded.selection == fitted.selection
    changed = replace(validation[0].inputs, image_encoder_manifest_sha256="b" * 64)
    with pytest.raises(ValueError, match="encoder identity"):
        loaded.predict(changed)
    with (tmp_path / "model/baseline.bundle/readout.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        FittedResidualModel.load(tmp_path / "model")


def test_epoch_zero_is_validation_candidate_and_persists_untrained_mapper(tmp_path, monkeypatch):
    import heir.personalized.residual_train as module

    training, validation, config, baseline, crossfit = training_fixture()
    enabled = []

    def validate(fitted, samples):
        enabled.append(fitted.selection["residual_enabled"])
        return {
            "eligible": True,
            SCORE: 1.0 if not enabled[-1] else 0.1,
            "centered_mse": 0.0,
            "normalized_log_mse": 0.0,
        }

    monkeypatch.setattr(module, "_validation", validate)
    fitted = fit_residual(
        training, validation, config, training_baselines=crossfit, baseline=baseline, epochs=3
    )
    assert enabled == [False, True, True, True]
    assert fitted.selection["selected_epoch"] == 0
    assert fitted.selection["mapper_training_status"] == "untrained_selected_epoch_zero"
    assert fitted.selection["mapper_parameters_changed"] is False
    image = replace(validation[0].inputs, reference=None, protein_features=None)
    expected = baseline.predict(image)
    np.testing.assert_array_equal(
        fitted.predict(validation[0].inputs).normalized_log_expression.numpy(), expected
    )
    fitted.save(tmp_path / "model")
    loaded = FittedResidualModel.load(tmp_path / "model")
    assert loaded.selection["selected_zero_residual"] is True
    np.testing.assert_array_equal(
        loaded.predict(validation[0].inputs).normalized_log_expression.numpy(), expected
    )


def test_training_rejects_arbitrary_insample_changed_features_targets_or_fold_receipts():
    training, validation, config, baseline, crossfit = training_fixture()
    kwargs = dict(baseline=baseline, epochs=1)
    with pytest.raises(ValueError, match="CrossFittedBaseline"):
        fit_residual(
            training, validation, config, training_baselines=crossfit.predictions, **kwargs
        )
    folds = [dict(f) for f in crossfit.folds]
    folds[0]["training_patient_ids"] = (
        folds[0]["training_patient_ids"] + folds[0]["heldout_patient_ids"]
    )
    with pytest.raises(ValueError, match="exclu"):
        fit_residual(
            training,
            validation,
            config,
            training_baselines=replace(crossfit, folds=tuple(folds)),
            **kwargs,
        )
    changed = list(training)
    changed[0] = replace(
        changed[0],
        inputs=replace(changed[0].inputs, image_features=changed[0].inputs.image_features + 1),
    )
    with pytest.raises(ValueError, match="exact held-out image"):
        fit_residual(changed, validation, config, training_baselines=crossfit, **kwargs)
    changed = list(training)
    changed[0] = replace(
        changed[0],
        targets=replace(changed[0].targets, expression=changed[0].targets.expression * 2),
    )
    with pytest.raises(ValueError, match="target identity"):
        fit_residual(changed, validation, config, training_baselines=crossfit, **kwargs)
    with pytest.raises(ValueError, match="training patients"):
        fit_residual(
            training,
            validation,
            config,
            training_baselines=crossfit,
            baseline=replace(baseline, training_patient_ids=("wrong",)),
            epochs=1,
        )


def test_generic_pool_roundtrip_needs_no_query_rna_and_rejects_wrong_reference(tmp_path):
    training, validation, config, baseline, crossfit = training_fixture()
    pool, metadata = pool_reference(
        [s.inputs for s in training],
        baseline.training_patient_ids,
        gene_ids=config.gene_ids,
        embedding_gene_ids=config.state_gene_ids,
        max_states=3,
    )
    training = [replace(s, inputs=replace(s.inputs, reference=pool)) for s in training]
    validation = [replace(s, inputs=replace(s.inputs, reference=None)) for s in validation]
    fitted = fit_residual(
        training,
        validation,
        config,
        training_baselines=crossfit,
        baseline=baseline,
        epochs=2,
        reference_override=pool,
        preparation={"reference_pool": metadata},
    )
    expected = fitted.predict(validation[0].inputs)
    fitted.save(tmp_path / "model")
    loaded = FittedResidualModel.load(tmp_path / "model")
    actual = loaded.predict(validation[0].inputs)
    torch.testing.assert_close(
        expected.normalized_log_expression, actual.normalized_log_expression, rtol=0, atol=0
    )
    torch.testing.assert_close(
        expected.linear_reference_transfer.transfer,
        actual.linear_reference_transfer.transfer,
        rtol=0,
        atol=0,
    )
    assert loaded.reference_override.state_ids == pool.state_ids
    bad = replace(pool, expression=pool.expression * 2)
    with pytest.raises(ValueError, match="bound training pool"):
        loaded.predict(replace(validation[0].inputs, reference=bad))


def test_nonfinite_gradient_rejected_before_optimizer_update(monkeypatch):
    import heir.personalized.residual_train as module

    training, validation, config, baseline, crossfit = training_fixture()
    original = module.ResidualModel

    def construct(*args, **kwargs):
        model = original(*args, **kwargs)
        model.residual_head[-1].weight.register_hook(
            lambda value: torch.full_like(value, float("nan"))
        )
        return model

    monkeypatch.setattr(module, "ResidualModel", construct)
    with pytest.raises(ValueError, match="nonfinite residual gradient"):
        fit_residual(
            training, validation, config, training_baselines=crossfit, baseline=baseline, epochs=1
        )
