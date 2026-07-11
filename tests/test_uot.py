"""Focused recovery and convergence tests for unbalanced transport."""

import math
import unittest

import torch

from heir.losses import unbalanced_sinkhorn, unbalanced_sinkhorn_loss


class UnbalancedSinkhornRobustnessTests(unittest.TestCase):
    def test_symmetric_two_by_two_recovers_analytic_uot_solution(self) -> None:
        dtype = torch.float64
        epsilon = 0.1
        relaxation = 1.0
        off_diagonal_cost = 0.2
        result = unbalanced_sinkhorn(
            torch.tensor(
                [[0.0, off_diagonal_cost], [off_diagonal_cost, 0.0]],
                dtype=dtype,
            ),
            source_mass=torch.tensor([0.5, 0.5], dtype=dtype),
            target_mass=torch.tensor([0.5, 0.5], dtype=dtype),
            epsilon=epsilon,
            marginal_relaxation=relaxation,
            iterations=400,
            convergence_tolerance=1.0e-10,
            add_unknown=False,
        )

        # For the symmetric plan [[a, b], [b, a]], stationarity gives
        # a / b = exp(c / epsilon) and an analytic transported row mass s.
        ratio_inverse = math.exp(-off_diagonal_cost / epsilon)
        exponent = 2.0 * relaxation / epsilon
        row_mass = math.exp(
            (math.log1p(ratio_inverse) - exponent * math.log(2.0)) / (1.0 + exponent)
        )
        diagonal = row_mass / (1.0 + ratio_inverse)
        off_diagonal = diagonal * ratio_inverse
        expected = torch.tensor(
            [[diagonal, off_diagonal], [off_diagonal, diagonal]],
            dtype=dtype,
        )

        torch.testing.assert_close(result.plan, expected, rtol=1.0e-8, atol=1.0e-10)
        torch.testing.assert_close(result.plan.sum(dim=-1), result.source_marginal)
        torch.testing.assert_close(result.plan.sum(dim=-2), result.target_marginal)
        self.assertLess(result.iterations_run, 400)
        self.assertTrue(bool(result.converged))

    def test_early_stop_matches_fixed_iterations_and_reports_residuals(self) -> None:
        cost = torch.tensor([[0.0, 0.2], [0.2, 0.0]], dtype=torch.float64)
        kwargs = {
            "source_mass": torch.tensor([0.5, 0.5], dtype=torch.float64),
            "target_mass": torch.tensor([0.5, 0.5], dtype=torch.float64),
            "epsilon": 0.1,
            "marginal_relaxation": 1.0,
            "add_unknown": False,
        }
        fixed = unbalanced_sinkhorn(cost, iterations=200, **kwargs)
        early = unbalanced_sinkhorn(
            cost,
            iterations=200,
            convergence_tolerance=1.0e-6,
            **kwargs,
        )

        self.assertLess(early.iterations_run, fixed.iterations_run)
        self.assertLessEqual(float(early.dual_residual), 1.0e-6)
        torch.testing.assert_close(early.plan, fixed.plan, rtol=2.0e-6, atol=1.0e-7)
        torch.testing.assert_close(early.loss, fixed.loss, rtol=1.0e-10, atol=1.0e-10)
        diagnostics = early.diagnostics()
        self.assertEqual(float(diagnostics["uot/iterations"]), early.iterations_run)
        self.assertEqual(float(diagnostics["uot/converged_fraction"]), 1.0)
        self.assertIn("uot/source_dual_residual", diagnostics)
        self.assertIn("uot/target_dual_residual", diagnostics)
        self.assertIn("uot/dual_residual", diagnostics)

    def test_batched_masks_preserve_plan_invariants_and_gradients(self) -> None:
        cost = torch.tensor(
            [
                [[-0.2, 0.4], [0.3, -0.1]],
                [[-0.1, 0.5], [0.2, -0.3]],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        pair_mask = torch.tensor(
            [
                [[True, False], [True, True]],
                [[True, True], [False, True]],
            ]
        )
        result = unbalanced_sinkhorn(
            cost,
            source_mass=torch.tensor([[0.4, 0.6], [0.7, 0.3]], dtype=torch.float64),
            target_mass=torch.tensor([[0.5, 0.5], [0.6, 0.4]], dtype=torch.float64),
            pair_mask=pair_mask,
            epsilon=0.15,
            marginal_relaxation=2.0,
            iterations=300,
            convergence_tolerance=1.0e-7,
            add_unknown=False,
        )

        self.assertTrue(torch.isfinite(result.plan).all())
        self.assertTrue(bool((result.plan >= 0).all()))
        self.assertTrue(torch.equal(result.plan.masked_select(~pair_mask), torch.zeros(2)))
        torch.testing.assert_close(result.plan.sum(dim=-1), result.source_marginal)
        torch.testing.assert_close(result.plan.sum(dim=-2), result.target_marginal)
        self.assertEqual(result.dual_residual.shape, (2,))
        self.assertEqual(result.converged.shape, (2,))
        result.loss.backward()
        self.assertTrue(torch.isfinite(cost.grad).all())
        self.assertTrue(torch.equal(cost.grad.masked_select(~pair_mask), torch.zeros(2)))

    def test_convergence_tolerance_must_be_finite_and_positive(self) -> None:
        cost = torch.zeros(2, 2)
        for tolerance in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, "convergence_tolerance"):
                    unbalanced_sinkhorn(cost, convergence_tolerance=tolerance)

    def test_scalar_loss_wrapper_retains_its_public_return_modes(self) -> None:
        cost = torch.tensor([[0.0, 0.2], [0.2, 0.0]])
        scalar = unbalanced_sinkhorn_loss(
            cost,
            add_unknown=False,
            convergence_tolerance=1.0e-5,
        )
        with_diagnostics = unbalanced_sinkhorn_loss(
            cost,
            add_unknown=False,
            convergence_tolerance=1.0e-5,
            return_diagnostics=True,
        )

        self.assertIsInstance(scalar, torch.Tensor)
        self.assertEqual(scalar.ndim, 0)
        loss, diagnostics = with_diagnostics
        torch.testing.assert_close(loss, scalar)
        self.assertIn("uot/dual_residual", diagnostics)


if __name__ == "__main__":
    unittest.main()
