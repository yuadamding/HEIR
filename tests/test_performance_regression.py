"""Small deterministic control-ordering fixture run by every CI build."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from heir.models import HEIRConfig, HEIRModel

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "configs" / "performance_regression_fixture_v1.json"


def _fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_performance_fixture_is_source_bound_and_claim_limited() -> None:
    fixture = _fixture()
    assert fixture["schema"] == "heir.synthetic_performance_regression.v1"
    assert fixture["claim_scope"] == (
        "deterministic_forward_control_only_not_biological_performance"
    )
    sources = fixture["sources"]
    assert {row["path"] for row in sources} == {
        "src/heir/models/heir.py",
        "src/heir/models/graph.py",
        "src/heir/models/rna.py",
        "tests/test_performance_regression.py",
    }
    for row in sources:
        source = ROOT / row["path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"]


def _controlled_heir() -> HEIRModel:
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=2,
            num_cell_types=2,
            expression_dim=1,
            latent_dim=1,
            graph_hidden_dim=2,
            graph_output_dim=2,
            graph_layers=1,
            graph_mode="off",
            trunk_hidden_dims=(2,),
            decoder_hidden_dims=(2,),
            dropout=0.0,
            hard_type_routing=False,
            residual_rank=1,
            residual_max_norm=1.0,
            residual_type_concentration_threshold=0.5,
            residual_type_concentration_temperature=0.05,
        )
    ).eval()
    with torch.no_grad():
        linear = model.trunk[0]
        linear.weight.zero_()
        linear.weight[:, :2].copy_(torch.eye(2))
        linear.bias.zero_()
        model.trunk[1].weight.fill_(1.0)
        model.trunk[1].bias.zero_()
        model.fine_type_head.weight.copy_(torch.tensor([[-5.0, 5.0], [5.0, -5.0]]))
        model.fine_type_head.bias.zero_()
        model.prototype_query_head.weight.copy_(torch.tensor([[1.0, -1.0]]))
        model.prototype_query_head.bias.zero_()
        model.unknown_head.weight.zero_()
        model.unknown_head.bias.fill_(-20.0)
        assert model.residual_type_basis is not None
        model.residual_type_basis.fill_(1.0)
        assert model.residual_coefficient_head is not None
        model.residual_coefficient_head.weight.copy_(torch.tensor([[0.577, -0.577]]))
        model.residual_coefficient_head.bias.zero_()
        assert model.residual_gate_head is not None
        model.residual_gate_head.weight.zero_()
        model.residual_gate_head.bias.fill_(20.0)
    return model


def test_controlled_heir_orders_matched_image_bank_and_residual_controls() -> None:
    expected = _fixture()["expected"]
    model = _controlled_heir()
    features = torch.tensor([[-2.0, 2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, -2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    true_latent = torch.tensor([[-1.0], [-1.0], [1.0], [1.0]])
    types = torch.tensor([0, 1])
    matched_means = torch.tensor([[-0.5], [0.5]])
    wrong_means = -matched_means

    matched = model(
        features,
        prototype_means=matched_means,
        prototype_types=types,
        sample_latent=False,
    )
    image_shuffle = model(
        features.flip(0),
        prototype_means=matched_means,
        prototype_types=types,
        sample_latent=False,
    )
    wrong_bank = model(
        features,
        prototype_means=wrong_means,
        prototype_types=types,
        sample_latent=False,
    )
    matched_accuracy = (matched.type_probabilities.argmax(1) == labels).float().mean()
    shuffled_accuracy = (image_shuffle.type_probabilities.argmax(1) == labels).float().mean()
    matched_mse = (matched.latent_mu - true_latent).square().mean()
    wrong_bank_mse = (wrong_bank.latent_mu - true_latent).square().mean()

    assert matched_accuracy > shuffled_accuracy
    assert matched_mse < wrong_bank_mse
    assert float(matched_accuracy) == pytest.approx(expected["matched_accuracy"], abs=1.0e-7)
    assert float(shuffled_accuracy) == pytest.approx(expected["image_shuffle_accuracy"], abs=1.0e-7)
    assert float(matched_mse) == pytest.approx(expected["matched_latent_mse"], abs=1.0e-7)
    assert float(wrong_bank_mse) == pytest.approx(expected["wrong_bank_latent_mse"], abs=1.0e-6)

    assert model.residual_gate_head is not None
    with torch.no_grad():
        saved_bias = model.residual_gate_head.bias.detach().clone()
        model.residual_gate_head.bias.fill_(-100.0)
    residual_off = model(
        features,
        prototype_means=matched_means,
        prototype_types=types,
        sample_latent=False,
    )
    with torch.no_grad():
        model.residual_gate_head.bias.copy_(saved_bias)
    residual_off_mse = (residual_off.latent_mu - true_latent).square().mean()
    assert matched_mse < residual_off_mse
    assert float(residual_off_mse) == pytest.approx(expected["residual_off_latent_mse"], abs=1.0e-6)


def test_enabled_graph_beats_no_graph_and_degree_preserving_cross_type_rewire() -> None:
    expected = _fixture()["expected"]
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=2,
            num_cell_types=2,
            expression_dim=1,
            latent_dim=1,
            graph_hidden_dim=2,
            graph_output_dim=2,
            graph_layers=1,
            graph_mode="distance_only",
            graph_context_gate_init=1.0,
            dropout=0.0,
            graph_residual=False,
            trunk_hidden_dims=(2,),
            decoder_hidden_dims=(2,),
            hard_type_routing=False,
            residual_rank=1,
        )
    ).eval()
    layer = model.graph_encoder.layers[0]
    with torch.no_grad():
        layer.self_projection.weight.zero_()
        layer.self_projection.bias.zero_()
        layer.neighbor_projection.weight.copy_(torch.eye(2))
        layer.normalization.weight.fill_(1.0)
        layer.normalization.bias.zero_()
        trunk = model.trunk[0]
        trunk.weight.zero_()
        trunk.weight[:, 2:].copy_(torch.eye(2))
        trunk.bias.zero_()
        model.trunk[1].weight.fill_(1.0)
        model.trunk[1].bias.zero_()
        model.fine_type_head.weight.copy_(torch.tensor([[-5.0, 5.0], [5.0, -5.0]]))
        model.fine_type_head.bias.zero_()
    features = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    matched_edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    rewired_edges = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]])
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    prototype_means = torch.tensor([[-1.0], [1.0]])
    prototype_types = torch.tensor([0, 1])

    def accuracy(edges: torch.Tensor, *, use_graph=None) -> torch.Tensor:
        output = model(
            features,
            edges,
            prototype_means=prototype_means,
            prototype_types=prototype_types,
            sample_latent=False,
            use_graph=use_graph,
        )
        return (output.type_probabilities.argmax(1) == labels).float().mean()

    assert accuracy(matched_edges) > accuracy(rewired_edges)
    assert accuracy(matched_edges) > accuracy(empty_edges)
    assert float(accuracy(matched_edges)) == pytest.approx(
        expected["matched_graph_accuracy"], abs=1.0e-7
    )
    assert float(accuracy(rewired_edges)) == pytest.approx(
        expected["rewired_graph_accuracy"], abs=1.0e-7
    )
    assert float(accuracy(empty_edges)) == pytest.approx(expected["no_graph_accuracy"], abs=1.0e-7)
    assert float(accuracy(matched_edges, use_graph=False)) == pytest.approx(
        expected["no_graph_accuracy"], abs=1.0e-7
    )
