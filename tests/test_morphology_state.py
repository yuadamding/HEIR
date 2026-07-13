from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from heir.models import (
    MorphologyStateGate,
    MorphologyStateGateConfig,
    donor_type_preserving_permutation,
    evaluate_morphology_state_checkpoint,
    fit_morphology_state_gate,
)


def _synthetic_donors(donors: tuple[str, ...], cells_per_type: int = 24):
    features = []
    latents = []
    labels = []
    donor_ids = []
    roi_ids = []
    for donor_position, donor in enumerate(donors):
        rng = np.random.default_rng(100 + donor_position)
        for type_index in range(2):
            state = np.linspace(-1.25, 1.25, cells_per_type)
            rng.shuffle(state)
            type_sign = -1.0 if type_index == 0 else 1.0
            features.append(
                np.column_stack(
                    (
                        np.full(cells_per_type, type_sign),
                        state,
                        type_sign * state,
                        rng.normal(0.0, 0.02, cells_per_type),
                    )
                )
            )
            if type_index == 0:
                residual = np.column_stack((np.zeros_like(state), state, 0.5 * state))
                centroid = np.array([2.0, 0.0, 0.0])
            else:
                residual = np.column_stack((np.zeros_like(state), -0.5 * state, state))
                centroid = np.array([-2.0, 0.0, 0.0])
            latents.append(centroid[None, :] + residual)
            labels.extend([type_index] * cells_per_type)
            donor_ids.extend([donor] * cells_per_type)
            roi_ids.extend(["roi_%d" % (index % 2) for index in range(cells_per_type)])
    return (
        np.concatenate(features).astype(np.float32),
        np.concatenate(latents).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(donor_ids),
        np.asarray(roi_ids),
    )


def _fit_gate():
    train = _synthetic_donors(("train_a", "train_b"))
    config = MorphologyStateGateConfig(
        feature_dim=4,
        latent_dim=3,
        num_types=2,
        residual_rank=1,
        residual_hidden_dim=12,
        type_names=("epithelial", "immune"),
    )
    model = MorphologyStateGate.from_training_data(config, *train[:4])
    report = fit_morphology_state_gate(
        model,
        train[0],
        train[1],
        train[2],
        epochs=120,
        batch_size=len(train[0]),
        learning_rate=0.03,
        weight_decay=0.0,
        seed=17,
        device="cpu",
    )
    return model, report, train


def test_gate_trains_on_frozen_features_and_roundtrips_checkpoint(tmp_path: Path) -> None:
    model, training, train = _fit_gate()
    assert training["frozen_input_features"] is True
    assert training["final_epoch"]["residual_loss"] < training["first_epoch"]["residual_loss"]
    assert model.training_donors == ("train_a", "train_b")

    expected_centroids = np.stack(
        [train[1][train[2] == type_index].mean(axis=0) for type_index in range(2)]
    )
    np.testing.assert_allclose(model.type_centroids.numpy(), expected_centroids, atol=1.0e-6)
    heldout = _synthetic_donors(("heldout",), cells_per_type=8)
    assert not np.allclose(
        model.type_centroids.numpy()[0], heldout[1][heldout[2] == 0].mean(axis=0) + 1.0
    )

    feature_tensor = torch.tensor(train[0][:8], requires_grad=True)
    type_tensor = torch.from_numpy(train[2][:8])
    output = model(feature_tensor, type_tensor)
    (output.type_logits.sum() + output.latent.sum()).backward()
    assert feature_tensor.grad is None

    before = model(torch.from_numpy(heldout[0]), torch.from_numpy(heldout[2])).latent
    checkpoint = model.save_checkpoint(tmp_path / "morphology_state.pt")
    restored = MorphologyStateGate.load_checkpoint(checkpoint).eval()
    after = restored(torch.from_numpy(heldout[0]), torch.from_numpy(heldout[2])).latent
    torch.testing.assert_close(after, before)
    assert restored.training_donors == model.training_donors
    with pytest.raises(RuntimeError, match="already initialized"):
        restored.initialize_training_geometry(
            frozen_features=train[0],
            latent_targets=train[1],
            type_labels=train[2],
            donor_ids=train[3],
        )


def test_checkpoint_execution_beats_type_mean_and_preserving_shuffle(tmp_path: Path) -> None:
    model, _, _ = _fit_gate()
    heldout = _synthetic_donors(("heldout",))
    checkpoint = model.save_checkpoint(tmp_path / "morphology_state.pt")
    report = evaluate_morphology_state_checkpoint(
        checkpoint,
        heldout[0],
        heldout[1],
        heldout[2],
        heldout[3],
        decoder=nn.Identity(),
        expression_targets=heldout[1],
        roi_ids=heldout[4],
        seed=41,
        device="cpu",
    )

    assert report["execution"]["checkpoint_executed"] is True
    assert report["heldout"]["donor_disjoint_from_training"] is True
    assert report["type_classification"]["accuracy"] == pytest.approx(1.0)
    endpoints = report["endpoints"]
    oracle = endpoints["oracle_type_image_residual"]
    predicted = endpoints["predicted_type_image_residual"]
    shuffled = endpoints["donor_type_shuffle"]
    assert oracle["latent"]["within_type_r2"] > 0.95
    assert predicted["latent"]["within_type_r2"] > 0.95
    assert oracle["decoded_expression"]["rmse_delta_vs_type_mean"] > 0.25
    assert oracle["state_retrieval"]["mrr"] > shuffled["state_retrieval"]["mrr"]
    assert (
        report["deltas"]["oracle_image_residual_vs_donor_type_shuffle"]["within_type_r2_delta"]
        > 0.5
    )
    roi_control = report["controls"]["donor_type_roi_shuffle"]
    assert roi_control["preserves_donor"] is True
    assert roi_control["preserves_type"] is True
    assert roi_control["preserves_roi"] is True
    assert roi_control["shuffled_fraction"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="overlaps training donors"):
        evaluate_morphology_state_checkpoint(
            checkpoint,
            heldout[0],
            heldout[1],
            heldout[2],
            np.repeat("train_a", len(heldout[0])),
            decoder=nn.Identity(),
            expression_targets=heldout[1],
            device="cpu",
        )


def test_donor_type_roi_permutation_is_a_reproducible_derangement() -> None:
    donors = np.array(["a"] * 8 + ["b"] * 8)
    labels = np.array([0] * 4 + [1] * 4 + [0] * 4 + [1] * 4)
    rois = np.tile(np.repeat(["r1", "r2"], 2), 4)
    first = donor_type_preserving_permutation(donors, labels, roi_ids=rois, seed=7)
    second = donor_type_preserving_permutation(donors, labels, roi_ids=rois, seed=7)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(len(first)))
    assert np.all(first != np.arange(len(first)))
    np.testing.assert_array_equal(donors, donors[first])
    np.testing.assert_array_equal(labels, labels[first])
    np.testing.assert_array_equal(rois, rois[first])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_morphology_state_gate_cuda_training_and_checkpoint_execution(tmp_path: Path) -> None:
    training = _synthetic_donors(("train_a", "train_b"), cells_per_type=8)
    heldout = _synthetic_donors(("heldout",), cells_per_type=8)
    model = MorphologyStateGate.from_training_data(
        MorphologyStateGateConfig(
            feature_dim=4,
            latent_dim=3,
            num_types=2,
            residual_rank=1,
            residual_hidden_dim=8,
        ),
        *training[:4],
    )
    fit_morphology_state_gate(
        model,
        training[0],
        training[1],
        training[2],
        epochs=3,
        batch_size=16,
        device="cuda",
    )
    checkpoint = model.save_checkpoint(tmp_path / "cuda_gate.pt")
    report = evaluate_morphology_state_checkpoint(
        checkpoint,
        heldout[0],
        heldout[1],
        heldout[2],
        heldout[3],
        decoder=nn.Identity(),
        expression_targets=heldout[1],
        device="cuda",
        bootstrap_iterations=20,
        require_wrong_donor_banks=False,
    )
    assert report["execution"]["device"].startswith("cuda")
    assert report["execution"]["checkpoint_executed"] is True
