"""Behavioral tests for identifiability, anchors, and missing molecular states."""

import numpy as np
import torch

from heir.losses import unbalanced_sinkhorn
from heir.refinement.anchors import AnchorStatus, select_anchors, update_anchor_lifecycle


def _real_responsibilities(plan: torch.Tensor) -> torch.Tensor:
    real = plan[:, :-1]
    return real / real.sum(dim=1, keepdim=True).clamp_min(1.0e-12)


def test_identifiable_morphology_recovers_states_despite_wrong_abundance_prior() -> None:
    cost = torch.tensor([[0.0, 6.0], [6.0, 0.0]])
    result = unbalanced_sinkhorn(
        cost,
        target_mass=torch.tensor([0.99, 0.01]),
        epsilon=0.1,
        marginal_relaxation=1.0,
        iterations=300,
        unknown_mass=0.05,
        unknown_cost=10.0,
    )
    responsibilities = _real_responsibilities(result.plan)

    assert responsibilities.argmax(dim=1).tolist() == [0, 1]
    assert torch.all(responsibilities.max(dim=1).values > 0.999)


def test_non_identifiable_states_remain_uniform_without_spatial_separation() -> None:
    result = unbalanced_sinkhorn(
        torch.zeros((6, 2)),
        target_mass=torch.tensor([0.5, 0.5]),
        epsilon=0.1,
        marginal_relaxation=1.0,
        iterations=300,
        unknown_mass=0.05,
        unknown_cost=10.0,
    )
    responsibilities = _real_responsibilities(result.plan)
    entropy = -(responsibilities * responsibilities.clamp_min(1.0e-12).log()).sum(dim=1)

    torch.testing.assert_close(responsibilities, torch.full_like(responsibilities, 0.5))
    torch.testing.assert_close(entropy, torch.full_like(entropy, np.log(2.0)))


def test_incorrect_anchor_is_challenged_relabelled_and_reconfirmed() -> None:
    wrong_probabilities = np.asarray([[0.96, 0.04]])
    wrong = select_anchors(wrong_probabilities, 0.9, 1.0)
    state = update_anchor_lifecycle(wrong, wrong_probabilities, min_probability=0.9)
    state = update_anchor_lifecycle(wrong, wrong_probabilities, state, min_probability=0.9)
    assert state.status[0] == AnchorStatus.TRUSTED
    assert state.labels[0] == 0

    correct_probabilities = np.asarray([[0.03, 0.97]])
    correct = select_anchors(correct_probabilities, 0.9, 1.0)
    challenged = update_anchor_lifecycle(correct, correct_probabilities, state, min_probability=0.9)
    relabelled = update_anchor_lifecycle(
        correct, correct_probabilities, challenged, min_probability=0.9
    )
    trusted = update_anchor_lifecycle(
        correct, correct_probabilities, relabelled, min_probability=0.9
    )

    assert challenged.status[0] == AnchorStatus.CHALLENGED
    assert relabelled.status[0] == AnchorStatus.PROVISIONAL
    assert relabelled.labels[0] == 1
    assert trusted.status[0] == AnchorStatus.TRUSTED


def test_missing_reference_state_routes_to_dustbin() -> None:
    result = unbalanced_sinkhorn(
        torch.full((4, 2), 8.0),
        target_mass=torch.tensor([0.5, 0.5]),
        epsilon=0.1,
        marginal_relaxation=1.0,
        iterations=300,
        unknown_mass=0.2,
        unknown_cost=0.0,
    )
    dustbin = result.plan[:, -1]
    measured = result.plan[:, :-1].sum(dim=1)

    assert torch.all(dustbin > measured * 100.0)
    assert float(result.unassigned_mass) > 0.45
