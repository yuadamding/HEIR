"""Controls separating reference assignment from the H&E correction pathway."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_personalized_training import _config, _sample

from heir.personalized.graph import LRInteraction
from heir.personalized.model import ModelConfig, PersonalizedModel
from heir.personalized.train import FittedModel, TrainingConfig, fit


def test_constant_reference_mixture_is_declared_rna_profile_everywhere():
    inputs = _sample().inputs
    reference = replace(inputs.reference, proportions=np.asarray([0.8, 0.2]))
    inputs = replace(inputs, reference=reference)
    model = PersonalizedModel(replace(_config(), assignment_mode="constant_reference"))
    predicted = model(inputs)
    expected = torch.tensor([[0.8, 0.2]]).expand(6, -1)
    torch.testing.assert_close(predicted.conditional_assignment, expected)
    torch.testing.assert_close(
        predicted.transfer, expected @ torch.as_tensor(reference.expression, dtype=torch.float32)
    )
    assert not predicted.reference_support_calibrated
    assert not predicted.conditional_assignment.requires_grad
    assert torch.equal(predicted.reference_support, torch.ones(6))

    # Image features and state embeddings cannot introduce spatial assignment.
    changed = replace(
        inputs,
        image_features=inputs.image_features[::-1].copy() * 3,
        reference=replace(reference, embedding=reference.embedding + 10),
    )
    with torch.no_grad():
        for parameter in model.image_mapper.parameters():
            parameter.fill_(50)
        for parameter in model.state_mapper.parameters():
            parameter.fill_(-50)
    actual = model(changed)
    torch.testing.assert_close(actual.conditional_assignment, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.transfer, predicted.transfer, rtol=0, atol=0)

    # Reference ordering is an identity join, not a new mixture definition.
    order = [1, 0]
    reordered = replace(
        reference,
        state_ids=tuple(reference.state_ids[i] for i in order),
        state_types=tuple(reference.state_types[i] for i in order),
        expression=reference.expression[order],
        embedding=reference.embedding[order],
        proportions=reference.proportions[order],
        variance=reference.variance[order],
        cell_counts=reference.cell_counts[order],
    )
    reordered_prediction = model(replace(inputs, reference=reordered, proteins=None))
    torch.testing.assert_close(reordered_prediction.transfer, predicted.transfer)
    torch.testing.assert_close(reordered_prediction.conditional_assignment, expected[:, order])


def test_constant_control_fits_correction_and_preserves_frozen_mapper(tmp_path):
    sample = _sample()
    config = replace(_config(), assignment_mode="constant_reference")
    settings = TrainingConfig(epochs=12, learning_rate=0.03, abundance_weight=0, seed=7)
    torch.manual_seed(settings.seed)
    initial = PersonalizedModel(config)
    torch.manual_seed(settings.seed)
    learned = PersonalizedModel(replace(config, assignment_mode="learned"))
    # Constructing the control must not change the shared decoder initialization.
    for name in ("decoder_image.0.weight", "residual_cell.weight", "residual_gene"):
        torch.testing.assert_close(initial.state_dict()[name], learned.state_dict()[name])

    fitted = fit([sample], config, settings)
    assert fitted.history[-1]["loss"] < fitted.history[0]["loss"]
    for name, parameter in fitted.model.named_parameters():
        if name.startswith(("image_mapper.", "state_mapper.", "unsupported_head.")):
            assert not parameter.requires_grad
            assert parameter.grad is None
            torch.testing.assert_close(parameter, initial.state_dict()[name], rtol=0, atol=0)
    assert not fitted.model.protein_log_strength.requires_grad
    for name in ("decoder_image.0.weight", "residual_cell.weight", "residual_gene"):
        assert not torch.equal(fitted.model.state_dict()[name], initial.state_dict()[name])
    prediction = fitted.predict(sample.inputs)
    torch.testing.assert_close(
        prediction.conditional_assignment, torch.full((6, 2), 0.5), rtol=0, atol=0
    )
    assert prediction.log_correction[:, :2].std(0).max() > 0

    fitted.save(tmp_path / "control")
    restored = FittedModel.load(tmp_path / "control")
    assert restored.model.config.assignment_mode == "constant_reference"
    assert restored.model.active_trainable_parameter_count() == 26
    assert not any(
        parameter.requires_grad for parameter in restored.model.image_mapper.parameters()
    )
    torch.testing.assert_close(restored.predict(sample.inputs).expression, prediction.expression)


def test_separately_trained_transfer_only_updates_mapper_without_decoder():
    sample = _sample()
    config = replace(_config(), transfer_only=True)
    torch.manual_seed(3)
    initial = PersonalizedModel(config)
    prediction = initial(sample.inputs)
    prediction.expression.square().mean().backward()
    for module in (initial.image_mapper, initial.state_mapper):
        assert sum(parameter.grad.abs().sum() for parameter in module.parameters()) > 0
    for name, parameter in initial.named_parameters():
        if not name.startswith(("image_mapper.", "state_mapper.")):
            assert parameter.grad is None
    assert torch.count_nonzero(prediction.log_correction) == 0
    torch.testing.assert_close(prediction.expression, prediction.transfer, rtol=0, atol=0)

    fitted = fit(
        [sample],
        config,
        TrainingConfig(epochs=12, learning_rate=0.03, abundance_weight=0, seed=3),
    )
    assert fitted.history[-1]["loss"] < fitted.history[0]["loss"]
    assert fitted.model.active_trainable_parameter_count() == 24
    assert not torch.equal(fitted.model.image_mapper[0].weight, initial.image_mapper[0].weight)
    for name in ("decoder_image.0.weight", "residual_cell.weight", "residual_gene"):
        torch.testing.assert_close(fitted.model.state_dict()[name], initial.state_dict()[name])


@pytest.mark.parametrize("variant", ["he", "he_rna_protein", "generic_fixed", "lr_refined"])
def test_constant_reference_rejects_other_arms(variant):
    with pytest.raises(ValueError, match="noncontext he_rna"):
        replace(_config(), variant=variant, assignment_mode="constant_reference")


def test_control_modes_validate_and_legacy_config_defaults_are_preserved():
    with pytest.raises(ValueError, match="assignment_mode"):
        replace(_config(), assignment_mode="constant")
    old_config = _config().to_dict()
    old_config.pop("assignment_mode")
    assert ModelConfig.from_dict(old_config).assignment_mode == "learned"
    control = replace(_config(), assignment_mode="constant_reference", transfer_only=True)
    assert ModelConfig.from_dict(control.to_dict()) == control
    model = PersonalizedModel(control)
    assert model.active_trainable_parameter_count() == 0
    prediction = model(_sample().inputs)
    torch.testing.assert_close(prediction.expression, prediction.transfer, rtol=0, atol=0)
    with pytest.raises(ValueError, match="no trainable prediction pathway"):
        fit([_sample()], control)


@pytest.mark.parametrize("width,expected", [(12, (713, 2112, 2113)), (1536, (49481, 99648, 99649))])
def test_active_parameter_counts_match_comparison_architectures(width, expected):
    genes = tuple(f"g{i}" for i in range(18))
    config = ModelConfig(
        feature_dim=width,
        state_dim=18,
        state_gene_ids=genes,
        gene_ids=genes[:9],
        supervised_gene_ids=genes[:9],
        normalization_gene_ids=genes[:9],
    )
    for variant, count in zip(("he", "he_rna", "he_rna_protein"), expected):
        model = PersonalizedModel(replace(config, variant=variant))
        assert model.active_trainable_parameter_count() == count
        assert count < sum(parameter.numel() for parameter in model.parameters())
    constant = PersonalizedModel(replace(config, assignment_mode="constant_reference"))
    assert constant.active_trainable_parameter_count() == (width + 1) * 32 + 33 * 16 + 9 * 16
    transfer = PersonalizedModel(replace(config, transfer_only=True))
    assert transfer.active_trainable_parameter_count() == (width + 1) * 32 + 19 * 32
    rank2 = PersonalizedModel(replace(config, residual_rank=2))
    assert rank2.active_trainable_parameter_count() == (
        2 * (width + 1) * 32 + 19 * 32 + 33 * 2 + 9 * 2
    )


def test_active_counts_exclude_masked_outputs_unused_messages_and_frozen_context_mapper():
    config = _config()
    assert PersonalizedModel(config).active_trainable_parameter_count() == 50
    # Programs instantiated for a noncontext arm have permanently zero decoder inputs.
    extra = replace(config, lr_interactions=(LRInteraction(("A",), ("B",), radius_um=25),))
    assert PersonalizedModel(extra).active_trainable_parameter_count() == 50
    baseline = PersonalizedModel(replace(config, variant="he_rna_protein"))
    fixed = PersonalizedModel(replace(extra, variant="lr_fixed"))
    refined = PersonalizedModel(replace(extra, variant="lr_refined"))
    fixed.initialize_fixed_mapper(baseline)
    refined.initialize_fixed_mapper(baseline)
    assert fixed.active_trainable_parameter_count() == 30
    assert refined.active_trainable_parameter_count() == 38
    refined.config = replace(refined.config, transfer_only=True)
    assert refined.active_trainable_parameter_count() == 8
    refined.refinement_message.requires_grad_(False)
    assert refined.active_trainable_parameter_count() == 0
