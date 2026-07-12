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
