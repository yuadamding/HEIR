"""Training-loop regression tests for donor and sample-level semantics."""

import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from heir.config import LossWeightConfig, OptimizationConfig
from heir.data import HistologyBag
from heir.models import HEIRConfig, HEIRModel
from heir.training import (
    HEIRTrainer,
    HEIRTrainingBatch,
    TrainingStage,
    spatial_block_split_masks,
    subset_histology_bag,
)
from heir.utils import set_seed


def _patch(cells: int, bag_id: str, shift: float) -> HEIRTrainingBatch:
    return HEIRTrainingBatch(
        morphology=torch.randn(cells, 3) + shift,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=None,
        prototype_means=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        prototype_variances=torch.ones(2, 2),
        prototype_types=torch.tensor([0, 1]),
        prototype_weights=torch.tensor([0.5, 0.5]),
        target_composition=torch.tensor([0.4, 0.6]),
        target_pseudobulk=torch.tensor([0.2, 0.3]),
        sample_id="sample-a",
        bag_id=bag_id,
        donor_id="donor-a",
        block_id="block-a",
        analysis_role="development",
        nucleus_ids=tuple("%s-n%d" % (bag_id, index) for index in range(cells)),
    )


def _trainer(allow_overlap: bool = True) -> HEIRTrainer:
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
        )
    )
    return HEIRTrainer(
        model,
        TrainingStage.PERSONALIZED,
        OptimizationConfig(
            epochs=1,
            bag_size=8,
            reference_batch_size=8,
            mixed_precision=False,
        ),
        LossWeightConfig(),
        device="cpu",
        allow_split_overlap=allow_overlap,
    )


def test_bernoulli_uot_gate_adds_each_route_cost_exactly_once_and_backpropagates() -> None:
    base_cost = torch.tensor([[0.25, 1.25], [2.0, 3.0]], requires_grad=True)
    unknown_probability = torch.tensor([0.2, 0.75], requires_grad=True)

    real_cost, dustbin_cost = HEIRTrainer._bernoulli_uot_costs(
        base_cost,
        unknown_probability,
    )

    expected_real = base_cost.detach() - torch.log1p(-unknown_probability.detach()).unsqueeze(1)
    expected_dustbin = -torch.log(unknown_probability.detach())
    torch.testing.assert_close(real_cost, expected_real)
    torch.testing.assert_close(dustbin_cost, expected_dustbin)

    (real_cost.sum() + dustbin_cost.sum()).backward()
    torch.testing.assert_close(base_cost.grad, torch.ones_like(base_cost))
    expected_probability_gradient = 2.0 / (1.0 - unknown_probability.detach()) - (
        1.0 / unknown_probability.detach()
    )
    torch.testing.assert_close(unknown_probability.grad, expected_probability_gradient)


def test_bernoulli_uot_gate_clamps_endpoint_probabilities_in_float32() -> None:
    base_cost = torch.zeros((2, 3), dtype=torch.float16, requires_grad=True)
    unknown_probability = torch.tensor([0.0, 1.0], dtype=torch.float16, requires_grad=True)

    real_cost, dustbin_cost = HEIRTrainer._bernoulli_uot_costs(
        base_cost,
        unknown_probability,
    )

    assert real_cost.dtype == torch.float32
    assert dustbin_cost.dtype == torch.float32
    assert torch.isfinite(real_cost).all()
    assert torch.isfinite(dustbin_cost).all()
    (real_cost.sum() + dustbin_cost.sum()).backward()
    assert base_cost.grad is not None and torch.isfinite(base_cost.grad).all()
    assert unknown_probability.grad is not None
    assert torch.isfinite(unknown_probability.grad).all()


def test_sample_losses_are_computed_once_after_graph_patch_merge() -> None:
    torch.manual_seed(11)
    trainer = _trainer()
    patches = (_patch(3, "left", -2.0), _patch(5, "right", 2.0))
    observed = trainer._epoch(patches, None)
    device_patches = [batch.to(trainer.device) for batch in patches]
    trainer.model.eval()
    with torch.no_grad():
        outputs = [trainer._forward_output(batch) for batch in device_patches]
        merged_batch = trainer._merge_sample_batches(device_patches)
        merged_output = trainer._concatenate_outputs(outputs)
        _, expected = trainer._output_loss(merged_output, merged_batch)
    assert observed["total"] == pytest.approx(float(expected["total"]), rel=1e-6)
    assert merged_batch.morphology.shape[0] == 8
    assert merged_batch.edge_index.shape == (2, 0)


def test_non_synthetic_fit_requires_explicit_donor_and_block() -> None:
    trainer = _trainer(allow_overlap=False)
    incomplete = replace(_patch(3, "missing", 0.0), donor_id="", block_id="")
    validation = replace(
        _patch(3, "validation", 0.0),
        sample_id="sample-b",
        donor_id="donor-b",
        block_id="block-b",
    )
    with pytest.raises(ValueError, match="explicit donor_id and block_id"):
        trainer.fit([incomplete], [validation])


def test_fit_tracks_optimizer_step_and_seed_configures_deterministic_cublas(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_seed(17)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    result = _trainer().fit([_patch(3, "train", 0.0)], [_patch(3, "validation", 0.0)])
    assert result.history[0]["optimizer_step_skipped"] == 0.0


def test_spatial_patch_merge_unifies_cross_patch_spots_and_recomputes_pseudobulk() -> None:
    left = replace(
        _patch(3, "left", 0.0),
        analysis_role="pretraining",
        spot_ids=("s1", "s2"),
        spot_assignment=torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        target_spatial_expression=torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    )
    right = replace(
        _patch(5, "right", 0.0),
        analysis_role="pretraining",
        spot_ids=("s2", "s3"),
        spot_assignment=torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]]),
        target_spatial_expression=torch.tensor([[2.0, 2.0], [3.0, 3.0]]),
    )
    merged = HEIRTrainer._merge_sample_batches((left, right))
    assert merged.spot_ids == ("s1", "s2", "s3")
    torch.testing.assert_close(merged.spot_assignment.sum(dim=1), torch.tensor([2.0, 3.0, 3.0]))
    expected = torch.log1p(
        (
            torch.expm1(torch.tensor(1.0)) * 2
            + torch.expm1(torch.tensor(2.0)) * 3
            + torch.expm1(torch.tensor(3.0)) * 3
        )
        / 8
    )
    torch.testing.assert_close(merged.target_pseudobulk, torch.full((2,), expected))

    conflicting = replace(
        right,
        target_spatial_expression=torch.tensor([[9.0, 9.0], [3.0, 3.0]]),
    )
    with pytest.raises(ValueError, match="disagree on spatial target"):
        HEIRTrainer._merge_sample_batches((left, conflicting))


def test_spatial_block_split_is_disjoint_and_reindexes_edges() -> None:
    coordinates = np.asarray(
        [[10.0, 10.0], [20.0, 10.0], [610.0, 10.0], [620.0, 10.0]],
        dtype=np.float64,
    )
    bag = HistologyBag(
        slide_id="slide",
        nucleus_ids=np.asarray(["n0", "n1", "n2", "n3"]),
        features=np.arange(8, dtype=np.float32).reshape(4, 2),
        coordinates_um=coordinates,
        edge_index=np.asarray([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        edge_weight=np.ones(6, dtype=np.float32),
    )
    training, validation = spatial_block_split_masks(
        coordinates,
        validation_fraction=0.5,
        block_size_um=512.0,
        seed=17,
    )
    assert not np.any(training & validation)
    assert np.all(training | validation)
    train_bag = subset_histology_bag(bag, training)
    validation_bag = subset_histology_bag(bag, validation)
    assert set(train_bag.nucleus_ids.tolist()).isdisjoint(validation_bag.nucleus_ids.tolist())
    assert train_bag.edge_index.max(initial=-1) < train_bag.n_nuclei
    assert validation_bag.edge_index.max(initial=-1) < validation_bag.n_nuclei
