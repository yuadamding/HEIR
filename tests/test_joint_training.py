"""Staged joint optimization, portable transforms, and complementary information."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_joint_model import FIXTURE_BACKBONE, arm, joint_fixture

from heir.personalized.evaluate import normalized_observations
from heir.personalized.joint_model import JointModelConfig
from heir.personalized.observation import ObservationOperator
from heir.personalized.schema import (
    JointInputs,
    RegionalProteinFeatures,
    RNAReference,
    SpatialTargets,
)
from heir.personalized.train import (
    FittedJointModel,
    TrainingSample,
    _centered_expression_loss,
    _joint_validation,
    fit_joint,
    load_fitted_model,
)


def sample_fixture(patient="train"):
    inputs, config = joint_fixture(patient)
    a = np.exp(inputs.image_features[:, 0] + inputs.protein_features.mean[:, 0])
    values = np.column_stack((a + 1, 1 + 3 / (a + 1), np.ones(len(a))))
    targets = SpatialTargets(
        tuple(f"o{i}" for i in range(len(a))),
        config.gene_ids,
        values,
        np.ones(values.shape, bool),
        patient,
        inputs.specimen_id,
    )
    operator = ObservationOperator(
        inputs.cell_ids,
        targets.observation_ids,
        np.arange(len(a)),
        np.arange(len(a)),
        np.ones(len(a)),
    )
    return TrainingSample(inputs, targets, operator, ("r1",) * 6 + ("r2",) * 6), config


@pytest.mark.parametrize("modalities", ["H", "HR", "HP", "RP", "HRP"])
def test_joint_fits_roundtrip_and_only_training_fits_transforms(tmp_path, modalities):
    sample, config = sample_fixture()
    training_inputs, config = arm(sample.inputs, config, modalities)
    sample = replace(sample, inputs=training_inputs)
    validation, _ = sample_fixture("validation")
    validation_inputs, _ = arm(validation.inputs, joint_fixture()[1], modalities)
    validation = replace(validation, inputs=validation_inputs)
    fitted = fit_joint([sample], [validation], config, epochs=4, warmup_epochs=2)
    assert [item["stage"] for item in fitted.history][:2] == (
        ["transfer_warmup"] * 2 if "R" in modalities else ["direct"] * 2
    )
    assert fitted.training_patient_ids == ("train",)
    assert fitted.validation_patient_ids == ("validation",)
    if "H" in modalities:
        np.testing.assert_allclose(fitted.feature_mean, sample.inputs.image_features.mean(0))
    if "P" in modalities:
        assert fitted.protein_scaler.training_patient_ids == ("train",)
    expected = fitted.predict(validation.inputs)
    fitted.save(tmp_path / "bundle")
    loaded = FittedJointModel.load(tmp_path / "bundle")
    actual = loaded.predict(validation.inputs)
    torch.testing.assert_close(actual.expression, expected.expression, rtol=0, atol=0)
    assert isinstance(load_fitted_model(tmp_path / "bundle"), FittedJointModel)
    assert loaded.selection == fitted.selection
    assert loaded.predict_diagnostics(validation.inputs)["fused_embedding"].shape == (12, 32)
    with (tmp_path / "bundle" / "weights.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="integrity mismatch"):
        FittedJointModel.load(tmp_path / "bundle")


def test_selected_warmup_checkpoint_does_not_enable_untrained_correction(monkeypatch, tmp_path):
    sample, config = sample_fixture()
    validation, _ = sample_fixture("validation")
    calls = []

    def fake_validation(model, samples, *, correction_enabled):
        calls.append(correction_enabled)
        return {
            "eligible": True,
            "spatial_spearman": 1.0 if not correction_enabled else 0.0,
            "centered_error": 0.0,
            "total_mse": 0.0,
            "patients": [],
        }

    monkeypatch.setattr("heir.personalized.train._joint_validation", fake_validation)
    fitted = fit_joint([sample], [validation], config, epochs=4, warmup_epochs=2)
    assert calls == [False, False, True, True]
    assert fitted.selection["epoch"] == 1
    assert fitted.selection["stage"] == "transfer_warmup"
    result = fitted.predict(sample.inputs)
    torch.testing.assert_close(result.expression, result.transfer, rtol=0, atol=0)
    fitted.save(tmp_path / "warmup")
    loaded = FittedJointModel.load(tmp_path / "warmup")
    torch.testing.assert_close(
        loaded.predict(sample.inputs).expression, result.expression, rtol=0, atol=0
    )


def test_constant_and_transfer_controls_have_full_budget_without_warmup():
    sample, config = sample_fixture()
    fixed = fit_joint([sample], [], replace(config, assignment_mode="constant_reference"), epochs=3)
    assert [row["stage"] for row in fixed.history] == ["joint"] * 3
    assert fixed.selection["epoch"] == 3
    assert fixed.selection["policy"] == "development_no_validation_final_epoch"
    mapper = fit_joint([sample], [], replace(config, transfer_only=True), epochs=3)
    assert [row["stage"] for row in mapper.history] == ["transfer_only"] * 3
    assert all(row["correction_penalty"] == 0 for row in mapper.history)
    with pytest.raises(ValueError, match="no trainable pathway"):
        fit_joint(
            [sample], [], replace(config, transfer_only=True, assignment_mode="constant_reference")
        )


def test_centered_term_is_within_replicate_and_gene_not_pooled():
    truth = torch.tensor([[1.0, 4], [2, 5], [9, 1], [10, 2]])
    prediction = truth + torch.tensor([[20.0, -3], [20, -3], [-4, 15], [-4, 15]])
    mask = torch.ones_like(truth, dtype=torch.bool)
    assert _centered_expression_loss(prediction, truth, mask, ("a", "a", "b", "b")) == 0
    assert _centered_expression_loss(prediction, truth, mask) > 0
    mask[1, 1] = False
    assert _centered_expression_loss(prediction, truth, mask, ("a", "a", "b", "b")) == 0


def test_training_validation_overlap_and_undefined_selection_rejected(monkeypatch):
    sample, config = sample_fixture()
    with pytest.raises(ValueError, match="patients must be disjoint"):
        fit_joint([sample], [sample], config, epochs=1)
    validation, _ = sample_fixture("validation")
    monkeypatch.setattr(
        "heir.personalized.train._joint_validation",
        lambda *a, **k: {
            "eligible": False,
            "spatial_spearman": None,
            "centered_error": 0.0,
            "total_mse": 1.0,
        },
    )
    with pytest.raises(ValueError, match="every validation patient"):
        fit_joint([sample], [validation], config, epochs=1)


def test_validation_does_not_hide_an_undefined_patient():
    sample, config = sample_fixture()
    fitted = fit_joint([sample], [], config, epochs=1, warmup_epochs=0)
    valid, _ = sample_fixture("good")
    bad, _ = sample_fixture("bad")
    bad = replace(
        bad,
        inputs=replace(
            bad.inputs,
            image_features=np.zeros_like(bad.inputs.image_features),
            protein_features=replace(
                bad.inputs.protein_features,
                mean=np.zeros_like(bad.inputs.protein_features.mean),
                q90=np.zeros_like(bad.inputs.protein_features.q90),
            ),
        ),
    )
    prepared = [replace(s, inputs=fitted.transform(s.inputs)) for s in (valid, bad)]
    report = _joint_validation(fitted.model, prepared, correction_enabled=True)
    assert not report["eligible"]
    assert report["spatial_spearman"] is None
    assert len(report["patients"]) == 2
    assert {row["patient_id"] for row in report["patients"]} == {"good", "bad"}


def complementary_sample(patient, levels):
    a, b = np.meshgrid(levels, levels)
    a, b = a.flatten(), b.flatten()
    ids = tuple(f"r{i}" for i in range(len(a)))
    expression = np.array([[12.0, 1.0], [1.0, 12.0]])
    reference = RNAReference(
        ("s0", "s1"),
        ("g0", "g1"),
        expression,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.ones(2) / 2,
        ("type0", "type1"),
        np.zeros((2, 2)),
        np.ones(2),
        ("g0", "g1"),
    )
    protein = RegionalProteinFeatures(
        ids, ("m",), b[:, None], b[:, None], np.ones((len(a), 1), bool), np.ones((len(a), 1), int)
    )
    inputs = JointInputs(
        patient,
        patient + "-s",
        ids,
        np.column_stack((a, b)),
        a[:, None],
        reference,
        protein,
        image_encoder_manifest_sha256=FIXTURE_BACKBONE,
    )
    q = 1 / (1 + np.exp(-1.5 * (a + b)))
    truth = q[:, None] * expression[0] + (1 - q[:, None]) * expression[1]
    target = SpatialTargets(
        ids, ("g0", "g1"), truth, np.ones_like(truth, bool), patient, patient + "-s"
    )
    op = ObservationOperator(ids, ids, np.arange(len(a)), np.arange(len(a)), np.ones(len(a)))
    return TrainingSample(inputs, target, op)


def test_joint_learns_complementary_image_and_protein_signal_on_unseen_regions():
    training = complementary_sample("train", np.array([-1.0, -0.3, 0.3, 1.0]))
    evaluation = complementary_sample("evaluation", np.array([-0.8, -0.1, 0.1, 0.8]))
    errors = {}
    for modalities in ("HRP", "HR", "RP"):
        config = JointModelConfig(
            1 if "H" in modalities else 0,
            2,
            ("g0", "g1"),
            ("g0", "g1"),
            ("g0", "g1"),
            state_gene_ids=("g0", "g1"),
            marker_ids=("m",) if "P" in modalities else (),
            modalities=tuple(modalities),
            image_encoder_manifest_sha256=FIXTURE_BACKBONE if "H" in modalities else None,
            hidden_dim=8,
            transfer_only=True,
        )

        def restrict(s):
            return replace(
                s,
                inputs=replace(
                    s.inputs,
                    image_features=s.inputs.image_features if "H" in modalities else None,
                    image_encoder_manifest_sha256=FIXTURE_BACKBONE if "H" in modalities else None,
                    protein_features=s.inputs.protein_features if "P" in modalities else None,
                ),
            )

        train, test = restrict(training), restrict(evaluation)
        fitted = fit_joint([train], [], config, epochs=150, learning_rate=0.01, seed=0)
        p, y, _ = normalized_observations(
            fitted.predict(test.inputs),
            test.targets,
            test.observation,
            config.gene_ids,
            normalization_gene_ids=config.normalization_gene_ids,
        )
        errors[modalities] = float((p - y).square().mean())
    assert errors["HRP"] < 0.1 * min(errors["HR"], errors["RP"]), errors


def test_three_modalities_resolve_complementary_ambiguities_on_one_common_fixture():
    """Every two-modality arm lacks information on the same held-out specimens.

    Descriptor genes identify two common states; a specimen's RNA expression
    profile changes one outcome independently of those descriptors. Held-out
    specimens use intermediate RNA profiles never seen during optimization.
    """

    def specimen(patient, factor, levels):
        sample = complementary_sample(patient, levels)
        ref = sample.inputs.reference
        molecular = ref.expression.copy()
        molecular[:, 0] *= factor
        descriptors = np.array([[10.0, 1.0], [1.0, 10.0]])
        reference = replace(
            ref,
            gene_ids=("g0", "g1", "d0", "d1"),
            expression=np.column_stack((molecular, descriptors)),
            variance=np.zeros((2, 4)),
            embedding=np.log1p(descriptors),
            embedding_gene_ids=("d0", "d1"),
        )
        # The known generating weights depend on the independent H and P signals.
        a, b = sample.inputs.image_features[:, 0], sample.inputs.protein_features.mean[:, 0]
        q = 1 / (1 + np.exp(-1.5 * (a + b)))
        truth = q[:, None] * molecular[0] + (1 - q[:, None]) * molecular[1]
        return replace(
            sample,
            inputs=replace(sample.inputs, reference=reference),
            targets=replace(sample.targets, expression=truth),
        )

    training = [
        specimen(f"train{i}", f, np.array([-1.0, -0.3, 0.3, 1.0]))
        for i, f in enumerate((0.35, 2.5))
    ]
    testing = [
        specimen(f"test{i}", f, np.array([-0.8, -0.1, 0.1, 0.8])) for i, f in enumerate((0.5, 2.0))
    ]
    errors = {}
    for modalities in ("HRP", "HR", "HP", "RP"):
        config = JointModelConfig(
            1 if "H" in modalities else 0,
            2 if "R" in modalities else 0,
            ("g0", "g1"),
            ("g0", "g1"),
            ("g0", "g1"),
            state_gene_ids=("d0", "d1") if "R" in modalities else (),
            marker_ids=("m",) if "P" in modalities else (),
            modalities=tuple(modalities),
            image_encoder_manifest_sha256=FIXTURE_BACKBONE if "H" in modalities else None,
            hidden_dim=8,
        )

        def restrict(sample):
            return replace(
                sample,
                inputs=replace(
                    sample.inputs,
                    image_features=sample.inputs.image_features if "H" in modalities else None,
                    image_encoder_manifest_sha256=FIXTURE_BACKBONE if "H" in modalities else None,
                    reference=sample.inputs.reference if "R" in modalities else None,
                    protein_features=sample.inputs.protein_features if "P" in modalities else None,
                ),
            )

        fitted = fit_joint(
            [restrict(s) for s in training],
            [],
            config,
            epochs=150,
            warmup_epochs=30,
            learning_rate=0.01,
            seed=0,
        )
        errors[modalities] = []
        for sample in testing:
            sample = restrict(sample)
            p, y, _ = normalized_observations(
                fitted.predict(sample.inputs),
                sample.targets,
                sample.observation,
                config.gene_ids,
                normalization_gene_ids=config.normalization_gene_ids,
            )
            errors[modalities].append(float((p - y).square().mean()))
    for comparator in ("HR", "HP", "RP"):
        assert np.mean(errors["HRP"]) < 0.1 * np.mean(errors[comparator]), errors


def test_validation_features_do_not_fit_training_transforms_or_updates(monkeypatch):
    sample, config = sample_fixture()
    validation, _ = sample_fixture("validation")
    changed = replace(
        validation,
        inputs=replace(
            validation.inputs,
            image_features=validation.inputs.image_features + 100,
            reference=replace(
                validation.inputs.reference, embedding=validation.inputs.reference.embedding + 100
            ),
            protein_features=replace(
                validation.inputs.protein_features,
                mean=validation.inputs.protein_features.mean + 100,
                q90=validation.inputs.protein_features.q90 + 100,
            ),
        ),
    )
    monkeypatch.setattr(
        "heir.personalized.train._joint_validation",
        lambda *a, **k: {
            "eligible": True,
            "spatial_spearman": 0.0,
            "centered_error": 0.0,
            "total_mse": 0.0,
        },
    )
    original = fit_joint([sample], [validation], config, epochs=2, warmup_epochs=0)
    other = fit_joint([sample], [changed], config, epochs=2, warmup_epochs=0)
    assert original.protein_scaler.to_dict() == other.protein_scaler.to_dict()
    np.testing.assert_array_equal(original.feature_mean, other.feature_mean)
    np.testing.assert_array_equal(original.state_mean, other.state_mean)
    for name, weight in original.model.state_dict().items():
        torch.testing.assert_close(weight, other.model.state_dict()[name], rtol=0, atol=0)
