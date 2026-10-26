"""Opt-in joint controls preserve the completed dot-product family's behavior."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_joint_model import arm, joint_fixture

from heir.personalized.joint_model import JointModel, JointModelConfig
from heir.personalized.protein_features import ProteinFeatureScaler
from heir.personalized.train import FittedJointModel


@pytest.mark.parametrize("network", ["dot_product", "pairwise_mlp"])
def test_zero_output_initialization_keeps_the_correction_trainable(network):
    inputs, config = joint_fixture()
    model = JointModel(
        replace(config, assignment_network=network, correction_initialization="zero")
    )
    initial = model(inputs)
    assert torch.count_nonzero(initial.log_correction) == 0
    # The final factor receives signal first; zero initialization must not be
    # confused with freezing the complete low-rank decoder.
    initial.expression[:, 0].sum().backward()
    assert model.correction_gene.grad.abs().sum() > 0
    assert model.correction.weight.grad.abs().sum() == 0
    with torch.no_grad():
        model.correction_gene -= 0.01 * model.correction_gene.grad
    model.zero_grad()
    after = model(inputs)
    assert after.log_correction[:, 0].abs().sum() > 0
    after.expression[:, 0].sum().backward()
    assert model.correction.weight.grad.abs().sum() > 0
    torch.testing.assert_close(after.expression[:, 2], after.transfer[:, 2], rtol=0, atol=0)


def test_mlp_changes_only_assignment_initialization_and_has_comparable_capacity():
    _, config = joint_fixture()
    config = replace(config, correction_initialization="zero")
    torch.manual_seed(19)
    dot = JointModel(config)
    torch.manual_seed(19)
    mlp = JointModel(replace(config, assignment_network="pairwise_mlp"))
    shared = set(dot.state_dict()) & set(mlp.state_dict())
    assert shared == set(dot.state_dict()) - {"type_query.weight", "state_query.weight"}
    for name in shared:
        torch.testing.assert_close(dot.state_dict()[name], mlp.state_dict()[name], rtol=0, atol=0)
    assert dot.assignment_trainable_parameter_count() == 2048
    assert mlp.assignment_trainable_parameter_count() == 2080
    assert abs(mlp.assignment_trainable_parameter_count() / 2048 - 1) < 0.02
    assert (
        abs(mlp.active_trainable_parameter_count() / dot.active_trainable_parameter_count() - 1)
        < 0.01
    )
    fixed = JointModel(replace(mlp.config, assignment_mode="constant_reference"))
    assert fixed.assignment_trainable_parameter_count() == 0


def _permuted_inputs(inputs):
    order, genes = [3, 1, 0, 2], [2, 0, 1]
    ref = inputs.reference
    ref = replace(
        ref,
        state_ids=tuple(f"renamed-{i}" for i in order),
        state_types=tuple({"tumor": "a", "immune": "z"}[ref.state_types[i]] for i in order),
        gene_ids=tuple(ref.gene_ids[i] for i in genes),
        expression=ref.expression[np.ix_(order, genes)],
        variance=ref.variance[np.ix_(order, genes)],
        embedding=ref.embedding[order][:, ::-1],
        embedding_gene_ids=ref.embedding_gene_ids[::-1],
        proportions=ref.proportions[order],
        cell_counts=ref.cell_counts[order],
    )
    protein = inputs.protein_features
    protein = replace(
        protein,
        marker_ids=protein.marker_ids[::-1],
        mean=protein.mean[:, ::-1],
        q90=protein.q90[:, ::-1],
        observed=protein.observed[:, ::-1],
        valid_counts=protein.valid_counts[:, ::-1],
    )
    return replace(inputs, reference=ref, protein_features=protein), order


@pytest.mark.parametrize("network", ["dot_product", "pairwise_mlp"])
def test_named_axes_and_type_relabeling_do_not_change_predictions(network):
    inputs, config = joint_fixture()
    model = JointModel(replace(config, assignment_network=network))
    expected = model(inputs)
    changed, order = _permuted_inputs(inputs)
    actual = model(changed)
    torch.testing.assert_close(actual.expression, expected.expression)
    torch.testing.assert_close(
        actual.conditional_assignment, expected.conditional_assignment[:, order]
    )
    diag = model.diagnostic_arrays(changed)
    assert tuple(diag["region_ids"]) == inputs.cell_ids
    assert tuple(diag["state_ids"]) == changed.reference.state_ids
    assert tuple(diag["state_types"]) == changed.reference.state_types
    assert tuple(diag["reference_gene_ids"]) == config.gene_ids
    np.testing.assert_array_equal(diag["reference_cell_counts"], changed.reference.cell_counts)
    np.testing.assert_array_equal(diag["reference_variance"], inputs.reference.variance[order])
    for j, label in enumerate(diag["type_ids"]):
        group = diag["state_types"] == label
        np.testing.assert_allclose(diag["within_type_mixture"][:, group].sum(1), 1, atol=1e-6)
        np.testing.assert_allclose(
            diag["state_mixture"][:, group],
            diag["type_mixture"][:, j : j + 1] * diag["within_type_mixture"][:, group],
            atol=1e-7,
        )


@pytest.mark.parametrize("network", ["dot_product", "pairwise_mlp"])
def test_molecular_descriptors_and_raw_transfer_are_separate(network):
    inputs, config = joint_fixture()
    model = JointModel(replace(config, assignment_network=network))
    original = model(inputs)
    changed = replace(
        inputs, reference=replace(inputs.reference, expression=inputs.reference.expression * 3)
    )
    result = model(changed)
    torch.testing.assert_close(
        result.conditional_assignment, original.conditional_assignment, rtol=0, atol=0
    )
    torch.testing.assert_close(result.transfer, original.transfer * 3)
    torch.testing.assert_close(result.log_correction, original.log_correction, rtol=0, atol=0)


@pytest.mark.parametrize("network", ["dot_product", "pairwise_mlp"])
def test_type_mass_has_no_flat_softmax_prototype_count_bias(network):
    inputs, config = joint_fixture()
    model = JointModel(replace(config, assignment_network=network))
    original = model.diagnostic_arrays(inputs)
    ref, order = inputs.reference, [0, 1, 2, 3, 0]
    # Split one state's sampling mass across two identical descriptors. Broad
    # type pooling is unchanged; no claim of within-type duplication invariance.
    duplicated = replace(
        ref,
        state_ids=(*ref.state_ids, "copy-s0"),
        state_types=tuple(ref.state_types[i] for i in order),
        expression=ref.expression[order],
        variance=ref.variance[order],
        embedding=ref.embedding[order],
        proportions=np.array([0.05, 0.2, 0.3, 0.4, 0.05]),
        cell_counts=np.array([0.5, 1, 1, 1, 0.5]),
    )
    result = model.diagnostic_arrays(replace(inputs, reference=duplicated))
    np.testing.assert_allclose(result["type_mixture"], original["type_mixture"], atol=1e-7)


def test_within_type_weights_remain_defined_when_type_mass_underflows(monkeypatch):
    inputs, config = joint_fixture()
    model = JointModel(config)

    def compatibility(query, states, *, broad_type):
        logits = query.new_zeros((len(query), len(states)))
        if broad_type:
            logits[:, 0] = -1000
        return logits

    monkeypatch.setattr(model, "_compatibility", compatibility)
    diag = model.diagnostic_arrays(inputs)
    assert np.all(diag["type_mixture"][:, 0] == 0)
    assert np.all(diag["state_mixture"][:, :2] == 0)
    np.testing.assert_array_equal(diag["within_type_mixture"], np.full((12, 4), 0.5))


def test_constant_zero_prior_type_has_explicit_conditional_weights():
    inputs, config = joint_fixture()
    inputs = replace(
        inputs, reference=replace(inputs.reference, proportions=np.array([0, 0, 0.4, 0.6]))
    )
    model = JointModel(replace(config, assignment_mode="constant_reference"))
    diag = model.diagnostic_arrays(inputs)
    np.testing.assert_array_equal(diag["within_type_mixture"][:, :2], np.full((12, 2), 0.5))
    np.testing.assert_array_equal(diag["state_mixture"][:, :2], np.zeros((12, 2)))


def test_coverage_control_ignores_intensity_but_retains_observation_support():
    torch.manual_seed(5)
    inputs, config = joint_fixture()
    model = JointModel(replace(config, protein_signal="coverage_only"))
    original = model(inputs)
    protein = replace(
        inputs.protein_features,
        mean=inputs.protein_features.mean * 100 + 91,
        q90=inputs.protein_features.q90[::-1].copy() * 7,
    )
    unchanged = model(replace(inputs, protein_features=protein))
    torch.testing.assert_close(original.expression, unchanged.expression, rtol=0, atol=0)
    torch.testing.assert_close(
        original.conditional_assignment, unchanged.conditional_assignment, rtol=0, atol=0
    )
    counts = protein.valid_counts.copy()
    counts[:, 0] *= 20
    altered = model(replace(inputs, protein_features=replace(protein, valid_counts=counts)))
    assert not torch.allclose(original.conditional_assignment, altered.conditional_assignment)
    original.expression[:, 0].sum().backward()
    weight_gradient = model.protein_encoder[0].weight.grad
    assert torch.count_nonzero(weight_gradient[:, :4]) == 0
    assert weight_gradient[:, 4:].abs().sum() > 0
    measured = JointModel(config)
    assert (
        measured.active_trainable_parameter_count() - model.active_trainable_parameter_count()
        == 128
    )


@pytest.mark.parametrize("network", ["dot_product", "pairwise_mlp"])
def test_partial_masks_ignore_hidden_values_through_scaler_and_model(network):
    torch.manual_seed(5)
    inputs, config = joint_fixture()
    observed = np.tile([[False, False], [True, False], [False, True], [True, True]], (3, 1))
    protein = replace(inputs.protein_features, observed=observed)
    inputs = replace(inputs, protein_features=protein)
    fitted = FittedJointModel(
        JointModel(replace(config, assignment_network=network, correction_initialization="zero")),
        np.zeros(3),
        np.ones(3),
        np.zeros(2),
        np.ones(2),
        ProteinFeatureScaler.fit([inputs]),
        (inputs.patient_id,),
        (),
        [],
        {},
        {"prediction_unit": inputs.prediction_unit, "coordinate_units": inputs.coordinate_units},
        {"correction_enabled": True},
    )
    expected = fitted.predict(inputs)
    diagnostics = fitted.predict_diagnostics(inputs)
    for hidden in (np.nan, 1e100, -1e100):
        changed = replace(
            protein,
            mean=np.where(observed, protein.mean, hidden),
            q90=np.where(observed, protein.q90, -hidden),
            valid_counts=np.where(observed, protein.valid_counts, hidden),
        )
        altered_inputs = replace(inputs, protein_features=changed)
        actual = fitted.predict(altered_inputs)
        torch.testing.assert_close(actual.expression, expected.expression, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.conditional_assignment, expected.conditional_assignment, rtol=0, atol=0
        )
        for name, value in fitted.predict_diagnostics(altered_inputs).items():
            np.testing.assert_array_equal(value, diagnostics[name])
    changed = replace(
        protein,
        mean=np.where(observed, protein.mean + 10, protein.mean),
        q90=np.where(observed, protein.q90 - 7, protein.q90),
    )
    measured_inputs = replace(inputs, protein_features=changed)
    measured = fitted.predict(measured_inputs)
    assert not torch.allclose(measured.conditional_assignment, expected.conditional_assignment)
    assert not torch.allclose(measured.expression, expected.expression)
    measured_diagnostics = fitted.predict_diagnostics(measured_inputs)
    for name in ("protein_observed", "protein_valid_counts"):
        np.testing.assert_array_equal(measured_diagnostics[name], diagnostics[name])


@pytest.mark.parametrize("network", ["dot_product", "pairwise_mlp"])
def test_bundle_roundtrip_persists_new_choices_without_new_prediction_semantics(tmp_path, network):
    inputs, config = joint_fixture()
    config = replace(
        config,
        assignment_network=network,
        correction_initialization="zero",
        protein_signal="coverage_only",
    )
    fitted = FittedJointModel(
        JointModel(config),
        np.zeros(3),
        np.ones(3),
        np.zeros(2),
        np.ones(2),
        ProteinFeatureScaler.fit([inputs]),
        (inputs.patient_id,),
        (),
        [],
        {},
        {"prediction_unit": inputs.prediction_unit, "coordinate_units": inputs.coordinate_units},
        {"correction_enabled": True},
    )
    expected = fitted.predict(inputs)
    fitted.save(tmp_path / "bundle")
    restored = FittedJointModel.load(tmp_path / "bundle")
    assert restored.model.config == config
    torch.testing.assert_close(
        restored.predict(inputs).expression, expected.expression, rtol=0, atol=0
    )
    for name, value in fitted.predict_diagnostics(inputs).items():
        np.testing.assert_array_equal(restored.predict_diagnostics(inputs)[name], value)


def test_old_bundle_config_retains_strict_legacy_weights(tmp_path):
    inputs, config = joint_fixture()
    fitted = FittedJointModel(
        JointModel(config),
        np.zeros(3),
        np.ones(3),
        np.zeros(2),
        np.ones(2),
        ProteinFeatureScaler.fit([inputs]),
        (inputs.patient_id,),
        (),
        [],
        {},
        {"prediction_unit": inputs.prediction_unit, "coordinate_units": inputs.coordinate_units},
        {"correction_enabled": True},
    )
    before = fitted.predict(inputs)
    directory = tmp_path / "historical"
    fitted.save(directory)
    metadata = json.loads((directory / "bundle.json").read_text())
    for key in ("assignment_network", "correction_initialization", "protein_signal"):
        metadata["model_config"].pop(key)
    (directory / "bundle.json").write_text(json.dumps(metadata))
    hashes = json.loads((directory / "sha256.json").read_text())
    hashes["bundle.json"] = hashlib.sha256((directory / "bundle.json").read_bytes()).hexdigest()
    (directory / "sha256.json").write_text(json.dumps(hashes))
    loaded = FittedJointModel.load(directory)
    assert loaded.model.config == config
    torch.testing.assert_close(loaded.predict(inputs).expression, before.expression, rtol=0, atol=0)
    assert set(loaded.model.state_dict()) == set(fitted.model.state_dict())
    bad = dict(loaded.model.state_dict(), extra_compatibility_weight=torch.ones(1))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        loaded.model.load_state_dict(bad)


def test_control_configuration_is_explicit_and_modality_specific():
    inputs, config = joint_fixture()
    for field, value in (
        ("assignment_network", "attention"),
        ("correction_initialization", "random"),
        ("protein_signal", "mask"),
    ):
        with pytest.raises(ValueError, match="unknown"):
            JointModelConfig.from_dict({**config.to_dict(), field: value})
    _, hp = arm(inputs, config, "HP")
    with pytest.raises(ValueError, match="requires an RNA"):
        replace(hp, assignment_network="pairwise_mlp")
    _, hr = arm(inputs, config, "HR")
    with pytest.raises(ValueError, match="requires the protein"):
        replace(hr, protein_signal="coverage_only")
