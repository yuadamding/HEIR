"""Shape, routing, checkpoint, and gradient tests for authoritative HEIR models."""

import copy
import json
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
        torch.testing.assert_close(
            output.conditional_prototype_probabilities.sum(dim=-1),
            torch.ones(11),
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

    def test_frozen_morphology_bridge_accepts_no_prototypes_and_matches_type_path(self) -> None:
        model = HEIRModel(small_config(graph_mode="off", hard_type_routing=False)).eval()
        morphology = torch.randn(7, 6)
        means = torch.randn(5, 4)
        types = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)

        embedding, probabilities, image_latent = model.encode_frozen_morphology(morphology)
        output = model(
            morphology,
            prototype_means=means,
            prototype_types=types,
            sample_latent=False,
        )

        self.assertEqual(embedding.shape, (7, 8))
        self.assertEqual(image_latent.shape, (7, 4))
        torch.testing.assert_close(probabilities, output.type_probabilities)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for AMP regression")
    def test_cuda_autocast_end_to_end_forward_backward(self) -> None:
        model = HEIRModel(small_config(hard_type_routing=False)).cuda()
        morphology = torch.randn(11, 6, device="cuda", requires_grad=True)
        edges = torch.tensor(
            [[0, 1, 2, 3, 4, 6, 7, 8], [1, 2, 3, 4, 5, 7, 8, 9]],
            dtype=torch.long,
            device="cuda",
        )
        prototype_means = torch.randn(5, 4, device="cuda", requires_grad=True)
        prototype_types = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(
                morphology,
                edges,
                prototype_means=prototype_means,
                prototype_types=prototype_types,
                sample_latent=True,
            )
            loss = (
                output.expression.square().mean()
                + output.type_probabilities.square().mean()
                + output.latent.square().mean()
            )
        loss.backward()

        self.assertTrue(torch.isfinite(loss.detach()))
        self.assertIsNotNone(morphology.grad)
        self.assertIsNotNone(prototype_means.grad)
        assert morphology.grad is not None and prototype_means.grad is not None
        self.assertTrue(torch.isfinite(morphology.grad).all())
        self.assertTrue(torch.isfinite(prototype_means.grad).all())

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

    def test_graph_off_is_exactly_invariant_to_graph_inputs(self) -> None:
        model = HEIRModel(small_config(graph_mode="off")).eval()
        morphology = torch.randn(7, 6)
        empty = torch.empty((2, 0), dtype=torch.long)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        common = {
            "prototype_means": torch.randn(3, 4),
            "prototype_types": torch.tensor([0, 1, 2]),
            "sample_latent": False,
        }

        without_graph = model(morphology, empty, **common)
        with_graph = model(morphology, edges, torch.rand(edges.shape[1]), **common)

        torch.testing.assert_close(with_graph.cell_embedding, without_graph.cell_embedding)
        torch.testing.assert_close(with_graph.expression, without_graph.expression)

    def test_distance_graph_is_explicit_and_starts_with_zero_residual_gate(self) -> None:
        model = HEIRModel(
            small_config(graph_mode="distance_only", graph_context_gate_init=0.0)
        ).eval()
        morphology = torch.randn(7, 6)
        empty = torch.empty((2, 0), dtype=torch.long)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        common = {
            "prototype_means": torch.randn(3, 4),
            "prototype_types": torch.tensor([0, 1, 2]),
            "sample_latent": False,
        }

        zero_empty = model(morphology, empty, **common)
        zero_edges = model(morphology, edges, **common)
        torch.testing.assert_close(zero_edges.cell_embedding, zero_empty.cell_embedding)

        with torch.no_grad():
            model.graph_context_gate.fill_(1.0)
        enabled_empty = model(morphology, empty, **common)
        enabled_edges = model(morphology, edges, **common)
        self.assertFalse(torch.equal(enabled_edges.cell_embedding, enabled_empty.cell_embedding))
        restored = HEIRModel.from_checkpoint(model.checkpoint())
        self.assertEqual(restored.config.graph_mode, "distance_only")
        torch.testing.assert_close(restored.graph_context_gate, torch.tensor(1.0))

    def test_legacy_checkpoint_preserves_full_strength_distance_graph(self) -> None:
        model = HEIRModel(
            small_config(graph_mode="distance_only", graph_context_gate_init=1.0)
        ).eval()
        checkpoint = copy.deepcopy(model.checkpoint())
        checkpoint["config"].pop("graph_mode")
        checkpoint["config"].pop("graph_context_gate_init")
        checkpoint["state_dict"].pop("graph_context_gate")

        restored = HEIRModel.from_checkpoint(checkpoint).eval()

        self.assertEqual(restored.config.graph_mode, "distance_only")
        torch.testing.assert_close(restored.graph_context_gate, torch.tensor(1.0))

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

    def test_local_routing_does_not_double_count_abundance_by_default(self) -> None:
        model = HEIRModel(small_config(fine_to_parent=None, hard_type_routing=False)).eval()
        morphology = torch.randn(3, 6)
        means = torch.zeros(2, 4)
        types = torch.zeros(2, dtype=torch.long)
        first = model(
            morphology,
            prototype_means=means,
            prototype_types=types,
            prototype_weights=torch.tensor([0.99, 0.01]),
            sample_latent=False,
        )
        second = model(
            morphology,
            prototype_means=means,
            prototype_types=types,
            prototype_weights=torch.tensor([0.01, 0.99]),
            sample_latent=False,
        )
        torch.testing.assert_close(
            first.conditional_prototype_probabilities,
            second.conditional_prototype_probabilities,
        )

        prior_weighted = HEIRModel(
            small_config(
                fine_to_parent=None,
                hard_type_routing=False,
                prototype_abundance_logit_weight=1.0,
            )
        ).eval()
        prior_weighted.load_state_dict(model.state_dict())
        weighted = prior_weighted(
            morphology,
            prototype_means=means,
            prototype_types=types,
            prototype_weights=torch.tensor([0.99, 0.01]),
            sample_latent=False,
        )
        assert torch.all(weighted.conditional_prototype_probabilities[:, 0] > 0.98)

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
        torch.testing.assert_close(high_unknown.prototype_latent, neutral.prototype_latent)
        torch.testing.assert_close(high_unknown.latent_mu, neutral.latent_mu)
        torch.testing.assert_close(high_unknown.expression, neutral.expression)

    def test_covariance_aware_uot_uses_final_decoded_latent(self) -> None:
        model = HEIRModel(small_config(hard_type_routing=False)).eval()
        morphology = torch.randn(4, 6)
        means = torch.randn(3, 4)
        variances = torch.rand(3, 4) + 0.2
        types = torch.tensor([0, 1, 2])
        with torch.no_grad():
            output = model(
                morphology,
                prototype_means=means,
                prototype_variances=variances,
                prototype_types=types,
                sample_latent=False,
            )
            model.prototype_query_head.weight.fill_(1000.0)
            model.prototype_query_head.bias.fill_(-1000.0)
            changed_legacy_query = model(
                morphology,
                prototype_means=means,
                prototype_variances=variances,
                prototype_types=types,
                sample_latent=False,
            )

        total_variance = output.residual_logvar.exp().unsqueeze(1) + variances.unsqueeze(0)
        expected = 0.5 * (
            (output.latent_mu.unsqueeze(1) - means.unsqueeze(0)).square() / total_variance
            + total_variance.log()
        ).mean(dim=-1)
        compatibility = output.type_probabilities.gather(
            1, types.unsqueeze(0).expand(len(morphology), -1)
        )
        expected = expected - model.config.prototype_type_cost_weight * compatibility.log()
        torch.testing.assert_close(output.prototype_query, output.latent_mu)
        torch.testing.assert_close(
            output.prototype_latent + output.residual_mu,
            output.latent_mu,
        )
        torch.testing.assert_close(output.prototype_cost, expected)
        self.assertFalse(
            torch.allclose(changed_legacy_query.prototype_latent, output.prototype_latent)
        )

    def test_zero_initialized_restricted_residual_inherits_prototype_baseline(self) -> None:
        model = HEIRModel(
            small_config(
                hard_type_routing=False,
                residual_rank=2,
                residual_max_norm=0.25,
            )
        ).eval()
        output = model(
            torch.randn(7, 6),
            prototype_means=torch.randn(3, 4),
            prototype_types=torch.tensor([0, 1, 2]),
            sample_latent=False,
        )

        torch.testing.assert_close(output.residual_mu, torch.zeros_like(output.residual_mu))
        torch.testing.assert_close(output.latent_mu, output.prototype_latent)
        torch.testing.assert_close(
            output.expression,
            model.expression_decoder(output.prototype_latent),
        )
        assert model.residual_logvar_head is not None
        torch.testing.assert_close(
            model.residual_logvar_head.weight,
            torch.zeros_like(model.residual_logvar_head.weight),
        )
        torch.testing.assert_close(
            model.residual_logvar_head.bias,
            torch.full_like(model.residual_logvar_head.bias, -6.0),
        )
        torch.testing.assert_close(
            output.residual_coefficient_logvar,
            torch.full_like(output.residual_coefficient_logvar, -6.0),
        )

    def test_restricted_residual_has_bounded_norm_and_type_basis_rank(self) -> None:
        rank = 2
        maximum = 0.15
        model = HEIRModel(
            small_config(
                hard_type_routing=False,
                residual_rank=rank,
                residual_max_norm=maximum,
            )
        ).eval()
        assert model.residual_coefficient_head is not None
        assert model.residual_gate_head is not None
        with torch.no_grad():
            model.fine_type_head.weight.zero_()
            model.fine_type_head.bias.copy_(torch.tensor([20.0, -20.0, -20.0]))
            model.residual_coefficient_head.weight.normal_()
            model.residual_coefficient_head.bias.normal_()
            model.residual_gate_head.weight.zero_()
            model.residual_gate_head.bias.fill_(20.0)
        output = model(
            torch.randn(12, 6),
            prototype_means=torch.randn(3, 4),
            prototype_types=torch.tensor([0, 1, 2]),
            sample_latent=False,
        )

        self.assertTrue(torch.all(output.residual_mu.norm(dim=-1) <= maximum + 1.0e-6))
        self.assertLessEqual(int(torch.linalg.matrix_rank(output.residual_mu)), rank)
        sampled = model.sample_residuals(output, draws=64)
        self.assertTrue(torch.all(sampled.norm(dim=-1) <= maximum + 1.0e-6))

    def test_rna_geometry_sets_type_specific_bound_and_survives_checkpoint(self) -> None:
        model = HEIRModel(
            small_config(
                hard_type_routing=False,
                residual_rank=2,
                residual_max_norm=9.0,
            )
        ).eval()
        bases = torch.zeros(3, 4, 2)
        bases[:, 0, 0] = 1.0
        bases[:, 1, 1] = 1.0
        maximums = torch.tensor([0.1, 0.3, 0.7])
        model.configure_residual_geometry(bases, maximums, freeze_basis=True)
        assert model.residual_type_basis is not None
        assert not model.residual_type_basis.requires_grad
        assert model.residual_coefficient_head is not None
        assert model.residual_gate_head is not None
        with torch.no_grad():
            model.fine_type_head.weight.zero_()
            model.fine_type_head.bias.copy_(torch.tensor([30.0, -30.0, -30.0]))
            model.residual_coefficient_head.weight.zero_()
            model.residual_coefficient_head.bias.fill_(100.0)
            model.residual_gate_head.weight.zero_()
            model.residual_gate_head.bias.fill_(30.0)
        output = model(
            torch.randn(5, 6),
            prototype_means=torch.randn(3, 4),
            prototype_types=torch.tensor([0, 1, 2]),
            sample_latent=False,
        )
        self.assertTrue(torch.all(output.residual_mu.norm(dim=-1) <= 0.1 + 1.0e-6))

        restored = HEIRModel.from_checkpoint(model.checkpoint())
        torch.testing.assert_close(restored.residual_type_max_norms, maximums)
        assert restored.residual_type_basis is not None
        assert not restored.residual_type_basis.requires_grad

    def test_uncertain_type_defers_residual_and_concentrated_type_selects_one_basis(
        self,
    ) -> None:
        model = HEIRModel(
            small_config(
                hard_type_routing=False,
                residual_rank=1,
                residual_type_concentration_threshold=0.7,
            )
        ).eval()
        bases = torch.zeros(3, 4, 1)
        bases[0, 0, 0] = 1.0
        bases[1, 1, 0] = 1.0
        bases[2, 2, 0] = 1.0
        maximums = torch.tensor([0.1, 0.2, 0.3])
        model.configure_residual_geometry(bases, maximums, freeze_basis=True)
        assert model.residual_coefficient_head is not None
        assert model.residual_gate_head is not None
        with torch.no_grad():
            model.fine_type_head.weight.zero_()
            model.fine_type_head.bias.zero_()
            assert model.parent_type_head is not None
            model.parent_type_head.weight.zero_()
            model.parent_type_head.bias.zero_()
            model.residual_coefficient_head.weight.zero_()
            model.residual_coefficient_head.bias.fill_(100.0)
            model.residual_gate_head.weight.zero_()
            model.residual_gate_head.bias.fill_(100.0)
        morphology = torch.randn(4, 6)
        prototypes = torch.randn(3, 4)
        types = torch.tensor([0, 1, 2])

        uncertain = model(
            morphology,
            prototype_means=prototypes,
            prototype_types=types,
            sample_latent=False,
        )
        assert uncertain.residual_gate is not None
        expected_uncertain_gate = maximums[0] * torch.sigmoid(
            torch.tensor(
                (1.0 / 3.0 - model.config.residual_type_concentration_threshold)
                / model.config.residual_type_concentration_temperature
            )
        )
        torch.testing.assert_close(
            uncertain.residual_gate,
            expected_uncertain_gate.expand_as(uncertain.residual_gate),
        )
        self.assertTrue(torch.all(uncertain.residual_mu.norm(dim=-1) < 0.01))

        with torch.no_grad():
            model.fine_type_head.bias.copy_(torch.tensor([-30.0, 30.0, -30.0]))
        concentrated = model(
            morphology,
            prototype_means=prototypes,
            prototype_types=types,
            sample_latent=False,
        )
        assert concentrated.residual_basis is not None
        torch.testing.assert_close(
            concentrated.residual_basis,
            bases[1].expand(len(morphology), -1, -1),
        )
        torch.testing.assert_close(
            concentrated.residual_mu[:, [0, 2, 3]],
            torch.zeros_like(concentrated.residual_mu[:, [0, 2, 3]]),
        )
        self.assertTrue(torch.all(concentrated.residual_mu.norm(dim=-1) <= 0.2 + 1.0e-6))

    def test_residual_concentration_gate_is_continuous_around_threshold(self) -> None:
        threshold = 0.6
        temperature = 0.02
        model = HEIRModel(
            small_config(
                fine_to_parent=None,
                hard_type_routing=False,
                residual_type_concentration_threshold=threshold,
                residual_type_concentration_temperature=temperature,
            )
        ).eval()
        assert model.residual_gate_head is not None
        with torch.no_grad():
            model.fine_type_head.weight.zero_()
            model.residual_gate_head.weight.zero_()
            model.residual_gate_head.bias.zero_()

        observed = []
        for concentration in (threshold - 0.001, threshold, threshold + 0.001):
            probabilities = torch.tensor(
                [concentration, (1.0 - concentration) / 2.0, (1.0 - concentration) / 2.0]
            )
            with torch.no_grad():
                model.fine_type_head.bias.copy_(probabilities.log())
            output = model(torch.zeros(1, 6), sample_latent=False)
            assert output.residual_gate is not None
            observed.append(output.residual_gate[0])

        expected = (
            0.5
            * model.config.residual_max_norm
            * torch.sigmoid(
                (torch.tensor([threshold - 0.001, threshold, threshold + 0.001]) - threshold)
                / temperature
            )
        )
        torch.testing.assert_close(torch.stack(observed), expected)
        self.assertTrue(observed[0] < observed[1] < observed[2])
        self.assertLess(float(observed[2] - observed[0]), 0.02)

    def test_residual_gate_cannot_backpropagate_into_type_head(self) -> None:
        model = HEIRModel(
            small_config(
                fine_to_parent=None,
                hard_type_routing=False,
                residual_type_concentration_temperature=0.03,
            )
        )
        assert model.residual_coefficient_head is not None
        assert model.residual_gate_head is not None
        with torch.no_grad():
            model.residual_coefficient_head.bias.fill_(0.5)
        output = model(torch.randn(5, 6), sample_latent=False)
        assert output.residual_gate is not None
        (output.residual_gate.sum() + output.residual_mu.square().sum()).backward()

        self.assertIsNone(model.fine_type_head.weight.grad)
        self.assertIsNone(model.fine_type_head.bias.grad)
        self.assertIsNotNone(model.residual_gate_head.bias.grad)
        assert model.residual_gate_head.bias.grad is not None
        self.assertGreater(float(model.residual_gate_head.bias.grad), 0.0)

    def test_residual_temperature_config_and_checkpoint_contract(self) -> None:
        for invalid in (0.0, -0.01, float("inf"), float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "temperature must be finite and positive"):
                    small_config(residual_type_concentration_temperature=invalid)

        configured = small_config(residual_type_concentration_temperature=0.037)
        restored_config = HEIRConfig.from_dict(configured.to_dict())
        self.assertEqual(restored_config.residual_type_concentration_temperature, 0.037)

        configured_model = HEIRModel(configured).eval()
        checkpoint = configured_model.checkpoint()
        self.assertEqual(checkpoint["schema"], "heir.model.v4")
        self.assertEqual(
            checkpoint["residual_geometry"]["type_concentration_temperature"],
            0.037,
        )
        restored = HEIRModel.from_checkpoint(checkpoint).eval()
        self.assertEqual(restored.config.residual_type_concentration_temperature, 0.037)
        morphology = torch.randn(4, 6)
        expected = configured_model(morphology, sample_latent=False)
        observed = restored(morphology, sample_latent=False)
        torch.testing.assert_close(observed.residual_gate, expected.residual_gate)
        torch.testing.assert_close(observed.expression, expected.expression)

        historical = copy.deepcopy(checkpoint)
        historical["schema"] = "heir.model.v3"
        historical["config"].pop("residual_type_concentration_temperature")
        historical["residual_geometry"].pop("type_concentration_temperature")
        historical_model = HEIRModel.from_checkpoint(historical)
        self.assertEqual(historical_model.config.residual_type_concentration_temperature, 0.05)
        self.assertEqual(historical_model.config.residual_type_strategy, "detached_max_hard")

        missing_v4_temperature = copy.deepcopy(checkpoint)
        missing_v4_temperature["config"].pop("residual_type_concentration_temperature")
        missing_v4_temperature["residual_geometry"].pop("type_concentration_temperature")
        with self.assertRaisesRegex(ValueError, "v4 checkpoint.*temperature is missing"):
            HEIRModel.from_checkpoint(missing_v4_temperature)

        missing_v4_strategy = copy.deepcopy(checkpoint)
        missing_v4_strategy["config"].pop("residual_type_strategy")
        missing_v4_strategy["residual_geometry"].pop("type_strategy")
        with self.assertRaisesRegex(ValueError, "v4 checkpoint.*strategy is missing"):
            HEIRModel.from_checkpoint(
                missing_v4_strategy,
                allow_legacy_mixed_residual_basis=True,
            )

        inconsistent = copy.deepcopy(checkpoint)
        inconsistent["residual_geometry"]["type_concentration_temperature"] = 0.04
        with self.assertRaisesRegex(ValueError, "temperature differs"):
            HEIRModel.from_checkpoint(inconsistent)

    def test_v3_checkpoint_round_trip_preserves_hard_residual_gate_predictions(self) -> None:
        historical_model = HEIRModel(
            small_config(
                fine_to_parent=None,
                hard_type_routing=False,
                graph_mode="distance_only",
                graph_context_gate_init=1.0,
                residual_type_strategy="detached_max_hard",
                residual_type_concentration_threshold=0.6,
            )
        ).eval()
        assert historical_model.residual_coefficient_head is not None
        assert historical_model.residual_gate_head is not None
        with torch.no_grad():
            historical_model.fine_type_head.weight.zero_()
            historical_model.fine_type_head.bias.copy_(torch.tensor([0.55, 0.25, 0.20]).log())
            historical_model.residual_coefficient_head.bias.fill_(0.5)
            historical_model.residual_gate_head.bias.fill_(1.0)

        morphology = torch.randn(4, 6)
        expected = historical_model(morphology, sample_latent=False)
        assert expected.residual_gate is not None
        torch.testing.assert_close(expected.residual_gate, torch.zeros(4))

        checkpoint = historical_model.checkpoint()
        self.assertEqual(checkpoint["schema"], "heir.model.v3")
        self.assertEqual(checkpoint["config"]["residual_type_strategy"], "detached_max")
        self.assertNotIn("residual_type_concentration_temperature", checkpoint["config"])
        checkpoint["config"].pop("graph_mode")
        checkpoint["config"].pop("graph_context_gate_init")
        checkpoint["state_dict"].pop("graph_context_gate")
        restored = HEIRModel.from_checkpoint(checkpoint).eval()
        self.assertEqual(restored.config.residual_type_strategy, "detached_max_hard")
        observed = restored(morphology, sample_latent=False)

        torch.testing.assert_close(observed.residual_gate, expected.residual_gate)
        torch.testing.assert_close(observed.residual_mu, expected.residual_mu)
        torch.testing.assert_close(observed.expression, expected.expression)
        self.assertEqual(restored.checkpoint()["schema"], "heir.model.v3")

    def test_residual_gate_diagnostics_are_json_ready(self) -> None:
        model = HEIRModel(
            small_config(
                fine_to_parent=None,
                hard_type_routing=False,
                residual_type_concentration_temperature=0.025,
            )
        ).eval()
        output = model(torch.randn(5, 6), sample_latent=False)
        diagnostics = model.residual_gate_diagnostics(output)

        json.dumps(diagnostics)
        self.assertEqual(diagnostics["schema"], "heir.residual_gate_diagnostics.v1")
        self.assertEqual(diagnostics["cell_count"], 5)
        self.assertTrue(diagnostics["available"])
        self.assertEqual(diagnostics["concentration_temperature"], 0.025)
        self.assertIn("p95", diagnostics["concentration_gate"])
        self.assertIn("maximum", diagnostics["residual_norm"])

        weighted_model = HEIRModel(
            small_config(
                fine_to_parent=None,
                residual_type_strategy="legacy_weighted_basis",
            )
        ).eval()
        weighted_output = weighted_model(torch.randn(3, 6), sample_latent=False)
        weighted_diagnostics = weighted_model.residual_gate_diagnostics(weighted_output)
        self.assertEqual(weighted_diagnostics["concentration_gate"]["mean"], 1.0)

    def test_residual_sampling_promotes_mixed_precision_geometry(self) -> None:
        basis = torch.zeros(3, 4, 2, dtype=torch.float32)
        basis[:, 0, 0] = 1.0
        basis[:, 1, 1] = 1.0
        coefficients = torch.full((5, 3, 2), 0.25, dtype=torch.float16)
        gate = torch.full((3,), 0.5, dtype=torch.float16)

        residual = HEIRModel._bounded_low_rank_residual(basis, coefficients, gate)

        self.assertEqual(residual.dtype, torch.float32)
        self.assertEqual(residual.shape, (5, 3, 4))
        self.assertTrue(torch.isfinite(residual).all())
        self.assertTrue(torch.all(residual.norm(dim=-1) < 0.5))

    def test_legacy_mixed_basis_checkpoint_requires_explicit_migration(self) -> None:
        model = HEIRModel(small_config(residual_rank=2)).eval()
        checkpoint = model.checkpoint()
        checkpoint["schema"] = "heir.model.v3"
        checkpoint["config"].pop("residual_type_strategy")
        checkpoint["config"].pop("residual_type_concentration_threshold")
        checkpoint["config"].pop("residual_type_concentration_temperature")
        checkpoint["residual_geometry"].pop("type_strategy")
        checkpoint["residual_geometry"].pop("type_concentration_threshold")
        checkpoint["residual_geometry"].pop("type_concentration_temperature")

        with self.assertRaisesRegex(ValueError, "allow_legacy_mixed_residual_basis=True"):
            HEIRModel.from_checkpoint(checkpoint)
        restored = HEIRModel.from_checkpoint(
            checkpoint,
            allow_legacy_mixed_residual_basis=True,
        )
        self.assertEqual(restored.config.residual_type_strategy, "legacy_weighted_basis")
        self.assertEqual(restored.config.residual_type_concentration_threshold, 0.0)

    def test_checkpoint_rejects_residual_strategy_metadata_mismatch(self) -> None:
        checkpoint = HEIRModel(small_config()).checkpoint()
        checkpoint["residual_geometry"]["type_strategy"] = "legacy_weighted_basis"
        with self.assertRaisesRegex(ValueError, "strategy differs"):
            HEIRModel.from_checkpoint(checkpoint)

    def test_restricted_checkpoint_requires_residual_geometry_contract(self) -> None:
        checkpoint = HEIRModel(small_config()).checkpoint()
        checkpoint.pop("residual_geometry")
        with self.assertRaisesRegex(ValueError, "residual geometry is missing"):
            HEIRModel.from_checkpoint(checkpoint)

    def test_rna_geometry_rejects_nonorthonormal_basis(self) -> None:
        model = HEIRModel(small_config(residual_rank=2))
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            model.configure_residual_geometry(
                torch.ones(3, 4, 2),
                torch.ones(3),
            )

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

    def test_unversioned_checkpoint_preserves_legacy_routing_semantics(self) -> None:
        legacy = HEIRModel(
            small_config(
                prototype_abundance_logit_weight=1.0,
                covariance_aware_uot=False,
                legacy_independent_prototype_query=True,
            )
        ).eval()
        checkpoint = legacy.checkpoint()
        checkpoint.pop("schema")
        for key in (
            "prototype_abundance_logit_weight",
            "prototype_variance_floor",
            "covariance_aware_uot",
            "legacy_independent_prototype_query",
            "legacy_unrestricted_residual",
            "residual_rank",
            "residual_max_norm",
        ):
            checkpoint["config"].pop(key)
        restored = HEIRModel.from_checkpoint(checkpoint).eval()

        self.assertTrue(restored.config.legacy_independent_prototype_query)
        self.assertFalse(restored.config.covariance_aware_uot)
        self.assertEqual(restored.config.prototype_abundance_logit_weight, 1.0)
        morphology = torch.randn(4, 6)
        means = torch.randn(3, 4)
        types = torch.tensor([0, 1, 2])
        weights = torch.tensor([0.7, 0.2, 0.1])
        expected = legacy(
            morphology,
            prototype_means=means,
            prototype_types=types,
            prototype_weights=weights,
            sample_latent=False,
        )
        observed = restored(
            morphology,
            prototype_means=means,
            prototype_types=types,
            prototype_weights=weights,
            sample_latent=False,
        )
        torch.testing.assert_close(observed.expression, expected.expression)
        torch.testing.assert_close(observed.prototype_cost, expected.prototype_cost)

    def test_v2_checkpoint_intentionally_preserves_tied_full_rank_residual(self) -> None:
        legacy = HEIRModel(
            small_config(
                hard_type_routing=False,
                legacy_unrestricted_residual=True,
            )
        ).eval()
        morphology = torch.randn(5, 6)
        means = torch.randn(3, 4)
        types = torch.tensor([0, 1, 2])
        expected = legacy(
            morphology,
            prototype_means=means,
            prototype_types=types,
            sample_latent=False,
        )
        checkpoint = legacy.checkpoint()
        checkpoint["schema"] = "heir.model.v2"
        for key in ("legacy_unrestricted_residual", "residual_rank", "residual_max_norm"):
            checkpoint["config"].pop(key)

        restored = HEIRModel.from_checkpoint(checkpoint).eval()
        observed = restored(
            morphology,
            prototype_means=means,
            prototype_types=types,
            sample_latent=False,
        )

        self.assertTrue(restored.config.legacy_unrestricted_residual)
        self.assertIsNotNone(restored.residual_mu_head)
        torch.testing.assert_close(observed.latent_mu, expected.latent_mu)
        torch.testing.assert_close(observed.expression, expected.expression)
        torch.testing.assert_close(observed.prototype_cost, expected.prototype_cost)

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
