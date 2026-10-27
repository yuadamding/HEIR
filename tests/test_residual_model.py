"""The new family protects float64 baseline predictions and keeps transfer separate."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_joint_architecture import _permuted_inputs
from test_joint_model import arm, joint_fixture

from heir.personalized.residual_model import ResidualModel


def fixture(modalities="HRP"):
    inputs, config = arm(*joint_fixture(), modalities)
    config = replace(config, supervised_gene_ids=config.gene_ids)
    baseline = np.random.default_rng(8).normal(size=(len(inputs.cell_ids), len(config.gene_ids)))
    baseline[0] = [-4.12500000001, 11.00000000001, 0.123456789012345]
    return inputs, config, baseline


@pytest.mark.parametrize("modalities", ["H", "HR", "HP", "HRP"])
def test_zero_head_preserves_float64_baseline_and_separates_expression_scales(modalities):
    inputs, config, baseline = fixture(modalities)
    model = ResidualModel(config)
    result = model(inputs, baseline)
    assert result.normalized_log_expression.dtype == torch.float64
    np.testing.assert_array_equal(result.normalized_log_expression.detach().numpy(), baseline)
    np.testing.assert_array_equal(result.coordinates_um.numpy(), inputs.coordinates_um)
    assert torch.count_nonzero(result.residual) == 0
    assert not hasattr(result, "expression")
    if "R" in modalities:
        linear = result.linear_reference_transfer
        assert linear is not None
        torch.testing.assert_close(linear.expression, linear.transfer, rtol=0, atol=0)
        expected = linear.conditional_assignment @ torch.tensor(
            inputs.reference.expression, dtype=torch.float32
        )
        torch.testing.assert_close(linear.transfer, expected, rtol=0, atol=0)
        assert not torch.equal(result.normalized_log_expression, linear.expression.double())
    else:
        assert result.linear_reference_transfer is None
        assert not any(p.requires_grad for p in model.mapper.direct_head.parameters())
        assert not any(p.requires_grad for p in model.mapper.fusion.parameters())
        assert torch.count_nonzero(result.diagnostics["residual_input"][:, 64:]) == 0
    if "P" not in modalities:
        assert torch.count_nonzero(result.diagnostics["protein_embedding"]) == 0
    with pytest.raises(ValueError, match="float64"):
        model(inputs, baseline.astype(np.float32))


def test_gradients_reach_image_protein_and_rna_mapper_after_zero_head_starts_learning():
    torch.manual_seed(3)
    inputs, config, baseline = fixture()
    model = ResidualModel(config)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    target = torch.tensor(baseline + np.linspace(-0.8, 0.8, baseline.size).reshape(baseline.shape))
    ((model(inputs, baseline).normalized_log_expression - target).square().mean()).backward()
    assert model.residual_head[-1].weight.grad.abs().sum() > 0
    assert model.mapper.type_query.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad()
    output = model(inputs, baseline)
    ((output.normalized_log_expression - target).square().mean()).backward()
    for module in (
        model.mapper.image_encoder[0],
        model.mapper.protein_encoder[0],
        model.mapper.state_encoder[0],
        model.mapper.type_query,
        model.mapper.state_query,
        model.mapper.fusion[0],
        model.residual_head[0],
    ):
        assert module.weight.grad.abs().sum() > 0
    assert model.mapper.correction_gene.grad is None
    assert output.residual.abs().max() <= 1
    changed = replace(
        inputs,
        reference=replace(inputs.reference, expression=inputs.reference.expression * [3, 1, 1]),
    )
    assert not torch.equal(
        output.normalized_log_expression, model(changed, baseline).normalized_log_expression
    )
    with torch.no_grad():
        model.mapper.state_query.weight.mul_(10)
    assert not torch.equal(
        output.normalized_log_expression, model(inputs, baseline).normalized_log_expression
    )


def test_zero_candidate_stays_baseline_even_when_mapper_or_decoder_changes():
    inputs, config, baseline = fixture()
    model = ResidualModel(config)
    original = model(inputs, baseline, residual_enabled=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.2)
    changed = model(inputs, baseline, residual_enabled=False)
    np.testing.assert_array_equal(changed.normalized_log_expression.numpy(), baseline)
    assert not torch.equal(
        original.linear_reference_transfer.transfer, changed.linear_reference_transfer.transfer
    )


def test_fixed_transfer_denominator_is_complete_positive_without_pseudocounts():
    inputs, config, baseline = fixture()
    model = ResidualModel(config)
    reference = replace(inputs.reference, expression=np.zeros_like(inputs.reference.expression))
    with pytest.raises(ValueError, match="denominator"):
        model(replace(inputs, reference=reference), baseline)


def test_coverage_only_and_named_axes_remain_invariant_with_nonzero_residual():
    inputs, config, baseline = fixture()
    torch.manual_seed(2)
    model = ResidualModel(replace(config, protein_signal="coverage_only"))
    torch.nn.init.normal_(model.residual_head[-1].weight, std=0.1)
    original = model(inputs, baseline)
    protein = replace(
        inputs.protein_features,
        mean=inputs.protein_features.mean * 50,
        q90=inputs.protein_features.q90 * -10,
    )
    changed = model(replace(inputs, protein_features=protein), baseline)
    torch.testing.assert_close(
        original.normalized_log_expression, changed.normalized_log_expression, rtol=0, atol=0
    )
    permuted, order = _permuted_inputs(inputs)
    changed = model(permuted, baseline)
    torch.testing.assert_close(
        original.normalized_log_expression, changed.normalized_log_expression, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        original.linear_reference_transfer.conditional_assignment[:, order],
        changed.linear_reference_transfer.conditional_assignment,
        rtol=1e-6,
        atol=1e-7,
    )
    assert model.active_trainable_parameter_count() > 0
