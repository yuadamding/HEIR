"""Regression tests for patch-safe constrained HEIR refinement."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from heir.config import RefinementConfig
from heir.models import HEIRConfig, HEIRModel
from heir.refinement import IterativeRefiner
from heir.refinement import iterative as iterative_module
from heir.refinement.anchors import (
    AnchorStatus,
    select_anchors,
    update_anchor_lifecycle,
)
from heir.refinement.ema import EMATeacher
from heir.training import HEIRTrainingBatch, HEIRTrainingResult


class _RecordingTrainer:
    def __init__(self, model: HEIRModel) -> None:
        self.model = model
        self.calls = []

    def fit(self, training_batches, validation_batches):
        self.calls.append(tuple(training_batches))
        return HEIRTrainingResult(0, 1.0, tuple())

    def _anchor_constraints(self, batch):
        del batch
        return None

    def transport_responsibilities(self, batch, output):
        del batch
        responsibilities = output.conditional_prototype_probabilities.detach()
        result = SimpleNamespace(
            source_marginal_error=torch.tensor(0.0),
            target_marginal_error=torch.tensor(0.0),
            target_marginal=responsibilities.mean(dim=0),
        )
        return responsibilities, result


def _batch(cells: int, bag_id: str, offset: float) -> HEIRTrainingBatch:
    morphology = torch.full((cells, 3), offset)
    return HEIRTrainingBatch(
        morphology=morphology,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=None,
        prototype_means=torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
        prototype_variances=torch.ones(2, 2),
        prototype_types=torch.tensor([0, 1]),
        prototype_weights=torch.tensor([0.5, 0.5]),
        target_composition=torch.tensor([0.5, 0.5]),
        target_pseudobulk=torch.zeros(2),
        sample_id="sample-a",
        bag_id=bag_id,
        donor_id="donor-a",
        block_id="block-a",
        analysis_role="development",
    )


def test_refinement_keeps_patch_state_separate_and_updates_one_sample_prior() -> None:
    torch.manual_seed(7)
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=3,
            num_cell_types=2,
            expression_dim=2,
            latent_dim=2,
            graph_hidden_dim=4,
            graph_output_dim=3,
            graph_layers=1,
            trunk_hidden_dims=(4,),
            decoder_hidden_dims=(4,),
            dropout=0.0,
            hard_type_routing=False,
            abstain_threshold=1.0,
        )
    )
    trainer = _RecordingTrainer(model)
    config = RefinementConfig(
        maximum_rounds=2,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
        require_view_agreement=False,
        stable_rounds_required=3,
    )
    batches = (_batch(3, "left", -2.0), _batch(5, "right", 2.0))
    result = IterativeRefiner(lambda: trainer, config).fit(batches, batches)
    assert len(result.rounds) == 2
    assert len(trainer.calls) == 2
    first_round = trainer.calls[0]
    assert first_round[0].anchor_labels.shape == (3,)
    assert first_round[1].anchor_labels.shape == (5,)
    assert torch.all(first_round[0].anchor_labels == -100)
    assert torch.all(first_round[1].anchor_labels == -100)
    assert torch.all(trainer.calls[1][0].anchor_labels >= 0)
    assert torch.all(trainer.calls[1][1].anchor_labels >= 0)
    assert result.rounds[0].provisional == 8
    assert result.rounds[0].trusted == 0
    assert result.rounds[1].trusted == 8
    assert torch.equal(first_round[0].prototype_weights, first_round[1].prototype_weights)
    assert "donor-a::sample-a" in result.sample_prototype_weights


def test_hierarchical_refinement_uses_parent_then_fine_anchors() -> None:
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=3,
            num_cell_types=3,
            expression_dim=2,
            latent_dim=2,
            graph_hidden_dim=4,
            graph_output_dim=3,
            graph_layers=1,
            trunk_hidden_dims=(4,),
            decoder_hidden_dims=(4,),
            fine_to_parent=(0, 0, 1),
            dropout=0.0,
            hard_type_routing=False,
            abstain_threshold=1.0,
        )
    )
    trainer = _RecordingTrainer(model)
    batch = HEIRTrainingBatch(
        morphology=torch.randn(5, 3),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=None,
        prototype_means=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        prototype_variances=torch.ones(3, 2),
        prototype_types=torch.tensor([0, 1, 2]),
        prototype_weights=torch.tensor([0.4, 0.3, 0.3]),
        target_composition=torch.tensor([0.4, 0.3, 0.3]),
        target_pseudobulk=torch.zeros(2),
        sample_id="hierarchical",
        bag_id="bag0",
        donor_id="donor-h",
        block_id="block-h",
        analysis_role="development",
    )
    config = RefinementConfig(
        maximum_rounds=2,
        broad_refinement_rounds=1,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
        require_view_agreement=False,
        stable_rounds_required=3,
    )
    IterativeRefiner(lambda: trainer, config).fit([batch], [batch])
    broad = trainer.calls[0][0]
    fine = trainer.calls[1][0]
    assert broad.parent_anchor_labels is not None
    assert torch.all(broad.anchor_labels == -100)
    assert torch.equal(fine.parent_anchor_labels, broad.parent_anchor_labels)
    assert fine.anchor_labels is not None


def test_view_agreement_fails_closed_without_independent_views() -> None:
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=3,
            num_cell_types=2,
            expression_dim=2,
            latent_dim=2,
            graph_hidden_dim=4,
            graph_output_dim=3,
            graph_layers=1,
            trunk_hidden_dims=(4,),
            decoder_hidden_dims=(4,),
        )
    )
    trainer = _RecordingTrainer(model)
    with pytest.raises(ValueError, match="independent view predictions"):
        IterativeRefiner(lambda: trainer, RefinementConfig()).fit(
            [_batch(3, "bag0", 0.0)],
            [_batch(3, "bag1", 0.0)],
        )


def test_anchor_lifecycle_recomputes_confidence_and_can_relabel_or_revoke() -> None:
    def update(
        probabilities: np.ndarray,
        previous=None,
        *,
        ood: bool = False,
    ):
        selection = select_anchors(
            probabilities,
            min_probability=0.90,
            max_normalized_entropy=1.0,
            ood_mask=np.asarray([ood]),
        )
        return update_anchor_lifecycle(
            selection,
            probabilities,
            previous,
            min_probability=0.90,
        )

    provisional = update(np.asarray([[0.97, 0.03]]))
    assert provisional.status[0] == AnchorStatus.PROVISIONAL
    assert provisional.labels[0] == 0
    assert provisional.confidence[0] == pytest.approx(0.97)

    trusted = update(np.asarray([[0.96, 0.04]]), provisional)
    assert trusted.status[0] == AnchorStatus.TRUSTED
    assert trusted.confidence[0] == pytest.approx(0.96)

    # A soft dip is retained at the lower hysteresis threshold, using the new
    # posterior rather than the permanently frozen 0.96 confidence.
    retained = update(np.asarray([[0.85, 0.15]]), trusted)
    assert retained.status[0] == AnchorStatus.TRUSTED
    assert retained.confidence[0] == pytest.approx(0.85)

    challenged = update(np.asarray([[0.04, 0.96]]), retained)
    assert challenged.status[0] == AnchorStatus.CHALLENGED
    assert not challenged.accepted[0]

    relabelled = update(np.asarray([[0.03, 0.97]]), challenged)
    assert relabelled.status[0] == AnchorStatus.PROVISIONAL
    assert relabelled.labels[0] == 1
    assert relabelled.confidence[0] == pytest.approx(0.97)

    revoked = update(np.asarray([[0.02, 0.98]]), relabelled, ood=True)
    assert revoked.status[0] == AnchorStatus.REVOKED
    assert not revoked.accepted[0]


class _ScriptedRefiner(IterativeRefiner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prediction_round = 0

    def _teacher_probabilities(self, teacher, batch):
        del teacher
        first = self.prediction_round == 0
        self.prediction_round += 1
        probabilities = np.tile([0.95, 0.05] if first else [0.05, 0.95], (3, 1))
        return (
            probabilities,
            probabilities,
            np.zeros(3, dtype=bool),
            np.zeros(3, dtype=np.float32),
            None,
        )

    def _teacher_transport(
        self,
        trainer,
        teacher,
        batch,
        parent_probabilities=None,
        broad_level=False,
    ):
        del trainer, teacher, batch, parent_probabilities, broad_level
        probabilities = np.tile(
            [0.95, 0.05] if self.prediction_round == 1 else [0.05, 0.95],
            (3, 1),
        )
        return probabilities, 0.1, 0.0, probabilities.mean(axis=0)


class _DegradingTrainer(_RecordingTrainer):
    def __init__(self, model: HEIRModel) -> None:
        super().__init__(model)
        self.model_states = []

    def fit(self, training_batches, validation_batches):
        del validation_batches
        self.calls.append(tuple(training_batches))
        call_id = len(self.calls)
        with torch.no_grad():
            next(self.model.parameters()).fill_(float(call_id))
        self.model_states.append(
            {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        )
        loss = 1.0 if call_id == 1 else 2.0
        return HEIRTrainingResult(0, loss, tuple())


def test_degraded_round_rolls_back_model_prior_and_skips_ema(monkeypatch) -> None:
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=3,
            num_cell_types=2,
            expression_dim=2,
            latent_dim=2,
            graph_hidden_dim=4,
            graph_output_dim=3,
            graph_layers=1,
            trunk_hidden_dims=(4,),
            decoder_hidden_dims=(4,),
            dropout=0.0,
            hard_type_routing=False,
            abstain_threshold=1.0,
        )
    )
    trainer = _DegradingTrainer(model)
    ema_updates = []

    class _CountingTeacher(EMATeacher):
        def update(self, student) -> None:
            ema_updates.append(len(trainer.calls))
            super().update(student)

    monkeypatch.setattr(iterative_module, "EMATeacher", _CountingTeacher)
    config = RefinementConfig(
        maximum_rounds=3,
        broad_refinement_rounds=0,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
        require_view_agreement=False,
        stable_rounds_required=3,
        objective_stability_tolerance=0.01,
    )
    result = _ScriptedRefiner(lambda: trainer, config).fit(
        [_batch(3, "bag0", 0.0)],
        [_batch(3, "validation", 0.0)],
    )

    assert result.stopped_reason == "validation_degraded_rollback"
    assert [item.committed for item in result.rounds] == [True, False]
    assert ema_updates == [1]
    assert len(trainer.calls) == 2
    for name, expected in trainer.model_states[0].items():
        assert torch.equal(trainer.model.state_dict()[name], expected)
    np.testing.assert_allclose(
        result.sample_prototype_weights["donor-a::sample-a"],
        trainer.calls[0][0].prototype_weights.numpy(),
    )
    assert not np.allclose(
        result.sample_prototype_weights["donor-a::sample-a"],
        trainer.calls[1][0].prototype_weights.numpy(),
    )
