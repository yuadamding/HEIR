"""Shape, routing, checkpoint, and gradient tests for authoritative HEIR models."""

import copy
import unittest

import torch

from heir.models import RNAVAE, GraphMessageLayer, HEIRConfig, HEIRModel, RNAVAEConfig


def small_config(**overrides: object) -> HEIRConfig:
    values = {
        "morphology_dim": 6,
        "num_cell_types": 3,
        "expression_dim": 9,
        "latent_dim": 4,
        "graph_hidden_dim": 10,
        "graph_output_dim": 8,
        "graph_layers": 2,
        "trunk_hidden_dims": (12, 8),
        "decoder_hidden_dims": (8, 12),
        "dropout": 0.0,
        "fine_to_parent": (0, 0, 1),
    }
    values.update(overrides)
    return HEIRConfig(**values)  # type: ignore[arg-type]


class HEIRModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_shapes_hierarchy_and_gradients(self) -> None:
        model = HEIRModel(small_config(hard_type_routing=False))
        morphology = torch.randn(11, 6, requires_grad=True)
        edges = torch.tensor(
            [[0, 1, 2, 3, 4, 6, 7, 8], [1, 2, 3, 4, 5, 7, 8, 9]],
            dtype=torch.long,
        )
        means = torch.randn(5, 4, requires_grad=True)
        types = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
        output = model(
            morphology,
            edges,
            prototype_means=means,
            prototype_types=types,
            prototype_weights=torch.tensor([2.0, 1.0, 2.0, 1.0, 1.0]),
            sample_latent=False,
        )
        self.assertEqual(output.type_probabilities.shape, (11, 3))
        self.assertEqual(output.parent_type_probabilities.shape, (11, 2))
        self.assertEqual(output.hierarchy_parent_probabilities.shape, (11, 2))
        self.assertEqual(output.prototype_probabilities.shape, (11, 5))
        self.assertEqual(output.residual_logvar.shape, (11, 4))
        self.assertEqual(output.latent.shape, (11, 4))
        self.assertEqual(output.expression.shape, (11, 9))
        self.assertEqual(output.abstain.shape, (11,))
        self.assertTrue(
            torch.allclose(
                output.prototype_probabilities.sum(dim=-1) + output.unknown_probability,
                torch.ones(11),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                output.hierarchy_parent_probabilities.sum(dim=-1),
                torch.ones(11),
                atol=1e-6,
            )
        )
        loss = (
            output.expression.square().mean()
            + output.type_probabilities.square().mean()
            + output.unknown_probability.mean()
        )
        loss.backward()
        self.assertIsNotNone(morphology.grad)
        self.assertIsNotNone(means.grad)
        self.assertTrue(torch.isfinite(morphology.grad).all())
        self.assertGreater(float(morphology.grad.norm().detach()), 0.0)

    def test_graph_projection_before_gather_matches_edge_expanded_reference(self) -> None:
        optimized = GraphMessageLayer(6, 4, dropout=0.0).double()
        reference = copy.deepcopy(optimized)
        optimized_input = torch.randn(7, 6, dtype=torch.float64, requires_grad=True)
        reference_input = optimized_input.detach().clone().requires_grad_(True)
        edge_index = torch.tensor(
            [[0, 0, 1, 2, 2, 2, 4, 5, 6], [1, 2, 2, 0, 3, 4, 5, 6, 0]],
            dtype=torch.long,
        )
        edge_weight = torch.linspace(0.2, 1.0, edge_index.shape[1], dtype=torch.float64)

        optimized_output = optimized(optimized_input, edge_index, edge_weight)
        source, target = edge_index
        messages = reference.neighbor_projection(reference_input.index_select(0, source))
        messages = messages * edge_weight.unsqueeze(-1)
        aggregate = reference_input.new_zeros((len(reference_input), 4))
        aggregate = aggregate.index_add(0, target, messages)
        degree = reference_input.new_zeros(len(reference_input))
        degree = degree.index_add(0, target, edge_weight)
        aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
        reference_output = reference.self_projection(reference_input) + aggregate
        reference_output = reference.dropout(
            reference.activation(reference.normalization(reference_output))
        )
        reference_output = reference_output + reference.residual_projection(reference_input)

        torch.testing.assert_close(optimized_output, reference_output)
        probe = torch.randn_like(optimized_output)
        (optimized_output * probe).sum().backward()
        (reference_output * probe).sum().backward()
        torch.testing.assert_close(optimized_input.grad, reference_input.grad)
        for optimized_parameter, reference_parameter in zip(
            optimized.parameters(), reference.parameters()
        ):
            torch.testing.assert_close(optimized_parameter.grad, reference_parameter.grad)

    def test_optional_scgpt_head_is_checkpointed_and_trainable(self) -> None:
        model = HEIRModel(small_config(hard_type_routing=False, scgpt_embedding_dim=7))
        morphology = torch.randn(6, 6, requires_grad=True)
        output = model(
            morphology,
            prototype_means=torch.randn(3, 4),
            prototype_types=torch.tensor([0, 1, 2]),
            sample_latent=False,
        )
        self.assertIsNotNone(output.scgpt_embedding)
        assert output.scgpt_embedding is not None
        self.assertEqual(output.scgpt_embedding.shape, (6, 7))
        output.scgpt_embedding.square().mean().backward()
        self.assertIsNotNone(model.scgpt_head)
        assert model.scgpt_head is not None
        self.assertIsNotNone(model.scgpt_head.weight.grad)
        restored = HEIRModel.from_checkpoint(model.checkpoint())
        self.assertEqual(restored.config.scgpt_embedding_dim, 7)

    def test_prototype_mask_and_wrong_type_are_exactly_excluded(self) -> None:
        model = HEIRModel(small_config(fine_to_parent=None, hard_type_routing=True))
        with torch.no_grad():
            model.fine_type_head.weight.zero_()
            model.fine_type_head.bias.copy_(torch.tensor([8.0, 0.0, -8.0]))
        output = model(
            torch.randn(7, 6),
            prototype_means=torch.randn(3, 4),
            prototype_types=torch.tensor([0, 1, 0]),
            prototype_mask=torch.tensor([True, True, False]),
            sample_latent=False,
        )
        self.assertTrue(torch.all(output.prototype_probabilities[:, 0] > 0))
        self.assertEqual(
            float(output.prototype_probabilities[:, 1].abs().max().detach()),
            0.0,
        )
        self.assertEqual(
            float(output.prototype_probabilities[:, 2].abs().max().detach()),
            0.0,
        )

        no_compatible = model(
            torch.randn(4, 6),
            prototype_means=torch.randn(2, 4),
            prototype_types=torch.tensor([0, 1]),
            cell_type_constraints=torch.full((4,), 2, dtype=torch.long),
            sample_latent=False,
        )
        self.assertTrue(torch.equal(no_compatible.prototype_probabilities, torch.zeros(4, 2)))
        self.assertTrue(torch.equal(no_compatible.unknown_probability, torch.ones(4)))

    def test_per_sample_prototype_banks(self) -> None:
        model = HEIRModel(small_config(hard_type_routing=False)).eval()
        cells = torch.randn(6, 6)
        means = torch.randn(2, 3, 4)
        types = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
        weights = torch.tensor([[1.0, 2.0, 1.0], [3.0, 1.0, 1.0]])
        samples = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        output = model(
            cells,
            prototype_means=means,
            prototype_types=types,
            prototype_weights=weights,
            sample_index=samples,
            sample_latent=False,
        )
        self.assertEqual(output.prototype_cost.shape, (6, 3))
        self.assertTrue(torch.equal(output.prototype_weights[:3], weights[0].expand(3, -1)))
        self.assertTrue(torch.equal(output.prototype_weights[3:], weights[1].expand(3, -1)))

    def test_exported_prototype_cost_does_not_preapply_unknown_gate(self) -> None:
        model = HEIRModel(small_config(hard_type_routing=False)).eval()
        morphology = torch.randn(4, 6)
        means = torch.randn(3, 4)
        types = torch.tensor([0, 1, 2])
        with torch.no_grad():
            model.unknown_head.weight.zero_()
            model.unknown_head.bias.zero_()
            neutral = model(
                morphology,
                prototype_means=means,
                prototype_types=types,
                sample_latent=False,
            )
            model.unknown_head.bias.fill_(8.0)
            high_unknown = model(
                morphology,
                prototype_means=means,
                prototype_types=types,
                sample_latent=False,
            )

        self.assertTrue(torch.all(high_unknown.unknown_probability > neutral.unknown_probability))
        torch.testing.assert_close(high_unknown.prototype_cost, neutral.prototype_cost)

    def test_deterministic_inference_and_checkpoint_round_trip(self) -> None:
        model = HEIRModel(small_config()).eval()
        morphology = torch.randn(8, 6)
        means = torch.randn(4, 4)
        types = torch.tensor([0, 1, 2, 0])
        first = model(
            morphology,
            prototype_means=means,
            prototype_types=types,
            sample_latent=False,
        )
        second = model(
            morphology,
            prototype_means=means,
            prototype_types=types,
            sample_latent=False,
        )
        self.assertTrue(torch.equal(first.latent, second.latent))
        self.assertTrue(torch.equal(first.expression, second.expression))
        restored = HEIRModel.from_checkpoint(model.checkpoint()).eval()
        third = restored(
            morphology,
            prototype_means=means,
            prototype_types=types,
            sample_latent=False,
        )
        self.assertTrue(torch.equal(first.expression, third.expression))
        self.assertTrue(torch.equal(first.type_probabilities, third.type_probabilities))

    def test_no_prototypes_and_empty_graph_route_to_unknown(self) -> None:
        model = HEIRModel(small_config()).eval()
        output = model(
            torch.randn(5, 6),
            torch.empty((2, 0), dtype=torch.long),
            sample_latent=False,
        )
        self.assertEqual(output.prototype_probabilities.shape, (5, 0))
        self.assertTrue(torch.equal(output.unknown_probability, torch.ones(5)))
        self.assertTrue(torch.isfinite(output.expression).all())

    def test_rna_decoder_transfer_and_freeze(self) -> None:
        vae = RNAVAE(
            RNAVAEConfig(
                input_dim=9,
                latent_dim=4,
                hidden_dims=(12, 8),
                decoder_hidden_dims=(8, 12),
                dropout=0.0,
            )
        )
        expression = torch.randn(10, 9)
        vae_output = vae(expression)
        vae_loss = (vae_output.reconstruction - expression).square().mean()
        vae_loss = vae_loss + 0.01 * vae.kl_divergence(vae_output.mu, vae_output.logvar)
        vae_loss.backward()
        restored = RNAVAE.from_checkpoint(vae.checkpoint()).eval()
        vae.eval()
        self.assertTrue(
            torch.equal(
                vae(expression, sample=False).reconstruction,
                restored(expression, sample=False).reconstruction,
            )
        )
        model = HEIRModel(small_config())
        model.load_rna_decoder(vae, freeze=True)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.expression_decoder.parameters())
        )


if __name__ == "__main__":
    unittest.main()
