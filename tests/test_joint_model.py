"""Joint family numerical contracts and genuinely separate modality controls."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from heir.personalized.joint_model import JointModel, JointModelConfig
from heir.personalized.model import create_model
from heir.personalized.schema import JointInputs, RegionalProteinFeatures, RNAReference
from heir.personalized.serialization import load_prediction, save_prediction

FIXTURE_BACKBONE = "a" * 64


def joint_fixture(patient="train", n=12):
    rng = np.random.default_rng(42)
    genes = ("a", "b", "untrained")
    reference = RNAReference(
        state_ids=("s0", "s1", "s2", "s3"),
        gene_ids=genes,
        expression=np.array([[7, 2, 4], [2, 8, 3], [4, 5, 2], [8, 4, 6.0]], dtype=float),
        embedding=np.array([[0, 1], [1, 0], [0.2, 0.8], [0.8, 0.2]], dtype=float),
        proportions=np.array([0.1, 0.2, 0.3, 0.4]),
        state_types=("immune", "immune", "tumor", "tumor"),
        variance=np.zeros((4, 3)),
        cell_counts=np.ones(4),
        embedding_gene_ids=("a", "b"),
    )
    ids = tuple(f"r{i}" for i in range(n))
    values = rng.normal(size=(n, 2))
    protein = RegionalProteinFeatures(
        ids, ("m1", "m2"), values, values + 1, np.ones((n, 2), bool), np.ones((n, 2), dtype=int) * 5
    )
    inputs = JointInputs(
        patient,
        f"{patient}-s",
        ids,
        rng.random((n, 2)),
        rng.normal(size=(n, 3)),
        reference,
        protein,
        image_encoder_manifest_sha256=FIXTURE_BACKBONE,
    )
    config = JointModelConfig(
        3,
        2,
        genes,
        ("a", "b"),
        ("a", "b"),
        state_gene_ids=("a", "b"),
        marker_ids=("m1", "m2"),
        image_encoder_manifest_sha256=FIXTURE_BACKBONE,
    )
    return inputs, config


def arm(inputs, config, modalities):
    modalities = tuple(modalities)
    return replace(
        inputs,
        image_features=inputs.image_features if "H" in modalities else None,
        image_encoder_manifest_sha256=FIXTURE_BACKBONE if "H" in modalities else None,
        reference=inputs.reference if "R" in modalities else None,
        protein_features=inputs.protein_features if "P" in modalities else None,
    ), replace(
        config,
        modalities=modalities,
        image_encoder_manifest_sha256=FIXTURE_BACKBONE if "H" in modalities else None,
        feature_dim=3 if "H" in modalities else 0,
        state_dim=2 if "R" in modalities else 0,
        state_gene_ids=config.state_gene_ids if "R" in modalities else (),
        marker_ids=config.marker_ids if "P" in modalities else (),
    )


@pytest.mark.parametrize("modalities", ["H", "HR", "HP", "RP", "HRP"])
def test_separate_modalities_and_export(modalities, tmp_path):
    inputs, config = arm(*joint_fixture(), modalities)
    model = create_model(config)
    prediction = model(inputs)
    assert prediction.expression.shape == (12, 3)
    assert torch.isfinite(prediction.expression).all()
    assert "targets" not in inputs.__dataclass_fields__
    if "R" in modalities:
        torch.testing.assert_close(prediction.conditional_assignment.sum(1), torch.ones(12))
        torch.testing.assert_close(
            prediction.expression[:, 2], prediction.transfer[:, 2], rtol=0, atol=0
        )
        assert prediction.log_correction.abs().max() <= 0.5
    else:
        assert prediction.conditional_assignment.shape == (12, 0)
        assert torch.count_nonzero(prediction.expression[:, 2]) == 0
        assert inputs.reference is None
    save_prediction(prediction, tmp_path / "prediction")
    restored, _ = load_prediction(tmp_path / "prediction")
    torch.testing.assert_close(restored.expression, prediction.expression)


def test_hierarchy_relabeling_and_state_gene_permutations():
    inputs, config = joint_fixture()
    model = JointModel(config)
    original = model(inputs)
    order, gene_order = [3, 1, 0, 2], [2, 0, 1]
    reference = inputs.reference
    changed = replace(
        reference,
        state_ids=tuple(f"new{i}" for i in order),
        state_types=tuple(reference.state_types[i] for i in order),
        gene_ids=tuple(reference.gene_ids[i] for i in gene_order),
        expression=reference.expression[np.ix_(order, gene_order)],
        variance=reference.variance[np.ix_(order, gene_order)],
        embedding=reference.embedding[order][:, ::-1],
        embedding_gene_ids=reference.embedding_gene_ids[::-1],
        proportions=reference.proportions[order],
        cell_counts=reference.cell_counts[order],
    )
    result = model(replace(inputs, reference=changed))
    torch.testing.assert_close(result.expression, original.expression)
    torch.testing.assert_close(
        result.conditional_assignment, original.conditional_assignment[:, order]
    )
    diagnostic = model.diagnostic_arrays(inputs)
    for j, broad_type in enumerate(diagnostic["type_ids"]):
        selected = [t == broad_type for t in reference.state_types]
        np.testing.assert_allclose(
            original.conditional_assignment[:, selected].detach().sum(1),
            diagnostic["type_mixture"][:, j],
            rtol=1e-6,
        )


def test_every_modality_reaches_assignment_and_decoder_with_finite_gradients():
    torch.manual_seed(4)
    inputs, config = joint_fixture()
    model = JointModel(config)
    prediction = model(inputs)
    (prediction.expression[:, 0].sum() + prediction.log_correction[:, 0].sum()).backward()
    for module in (
        model.image_encoder,
        model.protein_encoder,
        model.state_encoder,
        model.type_query,
        model.state_query,
        model.correction,
    ):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0
    changed = replace(
        inputs,
        protein_features=replace(
            inputs.protein_features,
            mean=inputs.protein_features.mean[::-1].copy(),
            q90=inputs.protein_features.q90[::-1].copy(),
        ),
    )
    other = model(changed)
    assert not torch.allclose(other.conditional_assignment, prediction.conditional_assignment)
    assert not torch.allclose(other.log_correction, prediction.log_correction)
    assert not torch.allclose(other.expression, prediction.expression)


def test_missing_protein_is_explicit_and_hidden_values_cannot_leak():
    inputs, config = joint_fixture()
    model = JointModel(config)
    missing = model(replace(inputs, protein_features=None))
    protein = replace(
        inputs.protein_features,
        observed=np.zeros((12, 2), bool),
        mean=np.full((12, 2), np.nan),
        q90=np.full((12, 2), np.inf),
        valid_counts=np.full((12, 2), -99),
    )
    masked = model(replace(inputs, protein_features=protein))
    torch.testing.assert_close(masked.expression, missing.expression, rtol=0, atol=0)
    zero = replace(inputs.protein_features, mean=np.zeros((12, 2)), q90=np.zeros((12, 2)))
    measured_zero = model(replace(inputs, protein_features=zero))
    assert not torch.allclose(measured_zero.expression, missing.expression)


def test_constant_control_fixes_mixture_but_decoder_uses_local_modalities():
    inputs, config = joint_fixture()
    model = JointModel(replace(config, assignment_mode="constant_reference"))
    prediction = model(inputs)
    torch.testing.assert_close(
        prediction.conditional_assignment,
        torch.tensor(inputs.reference.proportions, dtype=torch.float32).expand(12, -1),
    )
    changed = model(replace(inputs, protein_features=None))
    torch.testing.assert_close(prediction.transfer, changed.transfer, rtol=0, atol=0)
    assert not torch.allclose(prediction.log_correction, changed.log_correction)
    assert all(not p.requires_grad for p in model.type_query.parameters())
    transfer = JointModel(replace(config, transfer_only=True))(inputs)
    torch.testing.assert_close(transfer.expression, transfer.transfer, rtol=0, atol=0)
    assert torch.count_nonzero(transfer.log_correction) == 0


def test_disallowed_modalities_and_axes_fail():
    inputs, config = joint_fixture()
    direct_inputs, direct = arm(inputs, config, "HP")
    with pytest.raises(ValueError, match="unused modality R"):
        JointModel(direct)(replace(direct_inputs, reference=inputs.reference))
    rp_inputs, rp = arm(inputs, config, "RP")
    with pytest.raises(ValueError, match="unused modality H"):
        JointModel(rp)(
            replace(
                rp_inputs,
                image_features=inputs.image_features,
                image_encoder_manifest_sha256=FIXTURE_BACKBONE,
            )
        )
    with pytest.raises(ValueError, match="marker panel"):
        JointModel(config)(
            replace(
                inputs, protein_features=replace(inputs.protein_features, marker_ids=("x", "m2"))
            )
        )
    with pytest.raises(ValueError, match="explicit family"):
        create_model(config.to_dict())
    assert isinstance(create_model(config.to_dict(), family="joint"), JointModel)
