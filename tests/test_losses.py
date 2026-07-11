"""Numerical and gradient tests for HEIR distributional and biological losses."""

import unittest

import torch

from heir.losses import (
    HEIRCompositeLoss,
    HEIRLossConfig,
    anchor_classification_loss,
    boundary_graph_loss,
    cycle_consistency_loss,
    dirichlet_composition_prior_loss,
    hierarchy_consistency_loss,
    jensen_shannon_composition_loss,
    marker_centroid_loss,
    marker_ranking_loss,
    program_score_loss,
    pseudobulk_loss,
    residual_mahalanobis_loss,
    scgpt_representation_loss,
    soft_composition_bounds_loss,
    type_conditioned_program_score_loss,
    unbalanced_sinkhorn,
)
from heir.models import HEIRConfig, HEIRModel


class DistributionLossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)

    def test_unbalanced_sinkhorn_is_finite_and_tracks_marginals(self) -> None:
        cost = torch.tensor([[0.01, 3.0], [3.0, 0.01]], requires_grad=True)
        result = unbalanced_sinkhorn(
            cost,
            target_mass=torch.tensor([0.5, 0.5]),
            epsilon=0.05,
            marginal_relaxation=10.0,
            unknown_mass=0.01,
            iterations=100,
        )
        self.assertEqual(result.plan.shape, (2, 3))
        self.assertTrue(torch.isfinite(result.loss))
        self.assertGreater(float(result.plan[0, 0].detach()), float(result.plan[0, 1].detach()))
        self.assertGreater(float(result.plan[1, 1].detach()), float(result.plan[1, 0].detach()))
        self.assertLess(float(result.source_marginal_error.detach()), 0.02)
        self.assertLess(float(result.target_marginal_error.detach()), 0.02)
        self.assertGreaterEqual(float(result.unassigned_mass.detach()), 0.0)
        result.loss.backward()
        self.assertIsNotNone(cost.grad)
        self.assertTrue(torch.isfinite(cost.grad).all())

    def test_transport_masks_pairs_and_prototypes_exactly(self) -> None:
        cost = torch.tensor([[0.1, 0.1], [0.1, 0.1]], requires_grad=True)
        pair_mask = torch.tensor([[True, False], [False, True]])
        result = unbalanced_sinkhorn(
            cost,
            target_mass=torch.tensor([1.0, 1.0]),
            target_mask=torch.tensor([True, True]),
            pair_mask=pair_mask,
            unknown_mass=0.05,
            iterations=50,
        )
        self.assertEqual(float(result.plan[0, 1].detach()), 0.0)
        self.assertEqual(float(result.plan[1, 0].detach()), 0.0)

        masked = unbalanced_sinkhorn(
            cost.detach(),
            target_mass=torch.tensor([1.0, 1.0]),
            target_mask=torch.tensor([True, False]),
            unknown_mass=0.05,
        )
        self.assertEqual(float(masked.target_marginal[1]), 0.0)

    def test_transport_with_no_real_target_uses_unassigned_mass(self) -> None:
        result = unbalanced_sinkhorn(
            torch.ones(3, 2),
            target_mass=torch.ones(2),
            target_mask=torch.zeros(2, dtype=torch.bool),
            unknown_mass=0.05,
            marginal_relaxation=10.0,
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertGreater(float(result.unassigned_mass), 0.9)

    def test_transport_zero_source_mass_has_finite_gradients(self) -> None:
        cost = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
        result = unbalanced_sinkhorn(
            cost,
            source_mass=torch.tensor([0.0, 1.0]),
            target_mass=torch.tensor([0.5, 0.5]),
        )
        result.loss.backward()
        self.assertTrue(torch.isfinite(cost.grad).all())
        self.assertTrue(torch.equal(cost.grad[0], torch.zeros(2)))

    def test_composition_objectives(self) -> None:
        probabilities = torch.tensor(
            [[0.8, 0.2], [0.2, 0.8]],
            requires_grad=True,
        )
        target = torch.tensor([0.5, 0.5])
        js = jensen_shannon_composition_loss(probabilities, target)
        bounds = soft_composition_bounds_loss(
            probabilities,
            torch.tensor([0.4, 0.4]),
            torch.tensor([0.6, 0.6]),
        )
        prior = dirichlet_composition_prior_loss(
            probabilities,
            torch.tensor([5.0, 5.0]),
        )
        self.assertLess(float(js.detach()), 1e-7)
        self.assertLess(float(bounds.detach()), 1e-7)
        self.assertTrue(torch.isfinite(prior))
        (js + bounds + prior).backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())


class BiologicalLossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)

    def test_pseudobulk_program_and_marker_exact_matches(self) -> None:
        expression = torch.randn(8, 5, requires_grad=True)
        self.assertLess(
            float(pseudobulk_loss(expression, expression.mean(dim=0)).detach()),
            1e-7,
        )
        programs = torch.randn(5, 3)
        target_programs = expression.matmul(programs).mean(dim=0).detach()
        self.assertLess(
            float(program_score_loss(expression, programs, target_programs).detach()),
            1e-7,
        )
        probabilities = torch.zeros(8, 2)
        probabilities[:4, 0] = 1.0
        probabilities[4:, 1] = 1.0
        centroids = torch.stack((expression[:4].mean(dim=0), expression[4:].mean(dim=0))).detach()
        marker = marker_centroid_loss(expression, probabilities, centroids)
        self.assertLess(float(marker.detach()), 1e-7)
        (program_score_loss(expression, programs, target_programs) + marker).backward()
        self.assertTrue(torch.isfinite(expression.grad).all())

    def test_log1p_pseudobulk_aggregates_in_linear_space(self) -> None:
        expression = torch.tensor(
            [
                [torch.log1p(torch.tensor(10_000.0)), 0.0],
                [0.0, torch.log1p(torch.tensor(10_000.0))],
            ],
            requires_grad=True,
        )
        target = torch.full((2,), torch.log1p(torch.tensor(5_000.0)))
        loss = pseudobulk_loss(
            expression,
            target,
            log1p_expression=True,
        )
        self.assertLess(float(loss.detach()), 1.0e-8)
        loss.backward()
        self.assertTrue(torch.isfinite(expression.grad).all())

    def test_type_conditioned_programs_and_marker_ranking(self) -> None:
        expression = torch.tensor(
            [[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]],
            requires_grad=True,
        )
        probabilities = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        programs = torch.eye(2)
        target = torch.tensor([[1.5, 0.0], [0.0, 1.5]])
        conditioned = type_conditioned_program_score_loss(
            expression,
            probabilities,
            programs,
            target,
        )
        ranking = marker_ranking_loss(
            expression,
            probabilities,
            torch.eye(2, dtype=torch.bool),
        )
        self.assertLess(float(conditioned.detach()), 1e-7)
        self.assertLess(float(ranking.detach()), 1e-7)
        (conditioned + ranking).backward()
        self.assertTrue(torch.isfinite(expression.grad).all())

    def test_residual_mahalanobis_and_cycle(self) -> None:
        residual = torch.tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
        assignment = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        precision = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        value = residual_mahalanobis_loss(
            residual,
            precision=precision,
            assignment_probabilities=assignment,
        )
        self.assertTrue(torch.isfinite(value))
        cycle = cycle_consistency_loss(residual, residual)
        self.assertEqual(float(cycle.detach()), 0.0)
        value.backward()
        self.assertTrue(torch.isfinite(residual.grad).all())

    def test_scgpt_alignment_uses_prototypes_and_teacher_variance(self) -> None:
        predicted = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
            requires_grad=True,
        )
        probabilities = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        prototypes = torch.eye(2)
        variances = torch.full((2, 2), 0.01)
        aligned = scgpt_representation_loss(
            predicted,
            probabilities,
            prototypes,
            variances,
        )
        shuffled = scgpt_representation_loss(
            predicted,
            probabilities.flip(1),
            prototypes,
            variances,
        )
        self.assertLess(float(aligned.detach()), float(shuffled.detach()))
        aligned.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_empty_graph_and_boundary_weighting(self) -> None:
        values = torch.randn(4, 3, requires_grad=True)
        empty = boundary_graph_loss(values, torch.empty((2, 0), dtype=torch.long))
        self.assertEqual(float(empty.detach()), 0.0)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        types = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        nonempty = boundary_graph_loss(values, edges, types, boundary_margin=0.1)
        self.assertTrue(torch.isfinite(nonempty))
        nonempty.backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_anchor_and_hierarchy_consistency(self) -> None:
        logits = torch.tensor([[8.0, -8.0], [-8.0, 8.0], [0.0, 0.0]], requires_grad=True)
        unknown = torch.tensor([0.01, 0.01, 0.9], requires_grad=True)
        anchors = anchor_classification_loss(
            logits,
            torch.tensor([0, 1, -1]),
            unknown_probability=unknown,
        )
        fine = torch.tensor([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1]], requires_grad=True)
        parent = torch.stack((fine[:, :2].sum(dim=1), fine[:, 2]), dim=1)
        hierarchy = hierarchy_consistency_loss(fine, parent, (0, 0, 1))
        self.assertLess(float(hierarchy.detach()), 1e-7)
        (anchors + hierarchy).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class CompositeLossTests(unittest.TestCase):
    def test_type_conditioned_biology_uses_detached_molecular_responsibilities(self) -> None:
        live_probabilities = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            requires_grad=True,
        )
        molecular_types = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            requires_grad=True,
        )
        expression = torch.tensor(
            [[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]],
            requires_grad=True,
        )
        output = {
            "type_probabilities": live_probabilities,
            "expression": expression,
            "latent": torch.zeros(4, 2, requires_grad=True),
        }
        criterion = HEIRCompositeLoss(
            HEIRLossConfig(
                cell_type_weight=0.0,
                uot_weight=0.0,
                program_weight=1.0,
                marker_weight=0.0,
                pseudobulk_weight=0.0,
                composition_weight=0.0,
                cycle_weight=0.0,
                residual_weight=0.0,
                latent_kl_weight=0.0,
                graph_weight=0.0,
                calibration_weight=0.0,
                hierarchy_weight=0.0,
                scgpt_weight=0.0,
            )
        )
        total, logs = criterion(
            output,
            molecular_type_responsibilities=molecular_types,
            program_matrix=torch.eye(2),
            target_program_scores=torch.tensor([[1.5, 0.0], [0.0, 1.5]]),
        )

        self.assertLess(float(logs["program"].detach()), 1.0e-7)
        total.backward()
        self.assertIsNotNone(live_probabilities.grad)
        self.assertTrue(torch.equal(live_probabilities.grad, torch.zeros_like(live_probabilities)))
        self.assertIsNone(molecular_types.grad)

    def test_composite_logs_and_backpropagates(self) -> None:
        torch.manual_seed(31)
        config = HEIRConfig(
            morphology_dim=5,
            num_cell_types=3,
            expression_dim=6,
            latent_dim=3,
            graph_hidden_dim=7,
            graph_output_dim=6,
            trunk_hidden_dims=(9, 6),
            decoder_hidden_dims=(6, 9),
            dropout=0.0,
            fine_to_parent=(0, 0, 1),
            hard_type_routing=False,
        )
        model = HEIRModel(config)
        morphology = torch.randn(9, 5, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 4, 5, 6], [1, 2, 3, 5, 6, 7]])
        output = model(
            morphology,
            edge_index,
            prototype_means=torch.randn(4, 3),
            prototype_types=torch.tensor([0, 1, 2, 0]),
            prototype_weights=torch.tensor([2.0, 1.0, 1.0, 1.0]),
            sample_latent=False,
        )
        criterion = HEIRCompositeLoss(
            HEIRLossConfig(
                uot_iterations=30,
                composition_bounds_weight=0.1,
                dirichlet_weight=0.1,
            )
        )
        total, logs = criterion(
            output,
            target_composition=torch.tensor([0.4, 0.3, 0.3]),
            composition_lower=torch.tensor([0.1, 0.1, 0.1]),
            composition_upper=torch.tensor([0.8, 0.8, 0.8]),
            dirichlet_concentration=torch.tensor([4.0, 3.0, 3.0]),
            target_pseudobulk=torch.randn(6),
            program_matrix=torch.randn(6, 2),
            target_program_scores=torch.randn(2),
            marker_centroids=torch.randn(3, 6),
            cycle_latent=output.latent.detach() + 0.1,
            edge_index=edge_index,
            anchor_labels=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]),
            fine_to_parent=(0, 0, 1),
            unknown_targets=torch.zeros(9),
        )
        self.assertTrue(torch.isfinite(total))
        self.assertIn("uot/source_marginal_error", logs)
        self.assertIn("weighted/residual", logs)
        self.assertIn("total", logs)
        total.backward()
        self.assertIsNotNone(morphology.grad)
        self.assertTrue(torch.isfinite(morphology.grad).all())


if __name__ == "__main__":
    unittest.main()
