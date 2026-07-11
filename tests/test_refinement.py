"""Regression tests for patch-safe constrained HEIR refinement."""

import pytest
import torch

from heir.config import RefinementConfig
from heir.models import HEIRConfig, HEIRModel
from heir.refinement import IterativeRefiner
from heir.training import HEIRTrainingBatch, HEIRTrainingResult


class _RecordingTrainer:
    def __init__(self, model: HEIRModel) -> None:
        self.model = model
        self.calls = []

    def fit(self, training_batches, validation_batches):
        self.calls.append(tuple(training_batches))
        return HEIRTrainingResult(0, 1.0, tuple())


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
