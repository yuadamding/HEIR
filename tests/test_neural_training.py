from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from heir.evaluation.neural_checkpoint import (
    canonical_model_state_sha256,
    load_neural_checkpoint,
    save_neural_checkpoint,
)
from heir.evaluation.neural_training import (
    NeuralArchitecture,
    predict_neural_model,
    train_neural_model,
)


def _xor_rows(repeats: int = 32) -> tuple[np.ndarray, np.ndarray]:
    patterns = np.asarray(
        [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    features = np.tile(patterns, (repeats, 1))
    targets = (features[:, 0] * features[:, 1])[:, None]
    return features, targets


def test_small_mlp_fits_nonlinear_signal_deterministically_and_row_invariant() -> None:
    torch.set_num_threads(1)
    features, targets = _xor_rows()
    labels = np.zeros(len(features), dtype=np.int64)
    ids = np.asarray([f"row-{index:04d}" for index in range(len(features))])
    weights = np.ones(len(features))
    architecture = NeuralArchitecture("mlp_tiny", 4, 1, 1)
    first = train_neural_model(
        features,
        targets,
        labels,
        ids,
        weights,
        architecture=architecture,
        max_epochs=80,
        batch_size=32,
        patience=10,
        seed=17,
        device="cpu",
    )
    rng = np.random.default_rng(8)
    order = rng.permutation(len(features))
    reordered = train_neural_model(
        features[order],
        targets[order],
        labels[order],
        ids[order],
        weights[order],
        architecture=architecture,
        max_epochs=80,
        batch_size=32,
        patience=10,
        seed=17,
        device="cpu",
    )
    prediction = predict_neural_model(first, features, labels, device="cpu")
    denominator = float(np.square(targets - targets.mean(axis=0)).sum())
    r2 = 1.0 - float(np.square(prediction - targets).sum()) / denominator
    assert r2 > 0.95
    assert first.checkpoint_sha256 == reordered.checkpoint_sha256
    assert prediction == pytest.approx(
        predict_neural_model(reordered, features, labels, device="cpu"), abs=1.0e-7
    )


def test_type_linear_adapter_recovers_opposite_type_effects() -> None:
    torch.set_num_threads(1)
    rng = np.random.default_rng(12)
    features = rng.normal(size=(240, 3)).astype(np.float32)
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), 120)
    targets = np.where(labels[:, None] == 0, features[:, :1], -features[:, :1])
    ids = np.asarray([f"cell-{index:04d}" for index in range(len(features))])
    fit = train_neural_model(
        features,
        targets,
        labels,
        ids,
        np.ones(len(features)),
        architecture=NeuralArchitecture(
            "shared_linear", 3, 1, 2, type_conditioned=True, adapter_rank=2
        ),
        max_epochs=100,
        batch_size=48,
        seed=29,
        device="cpu",
    )
    prediction = predict_neural_model(fit, features, labels, device="cpu")
    assert np.corrcoef(prediction[:, 0], targets[:, 0])[0, 1] > 0.95


def test_checkpoint_hash_is_order_stable_and_metadata_tampering_fails(tmp_path: Path) -> None:
    state_a = {
        "b": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "a": torch.arange(8, dtype=torch.float32).reshape(4, 2).T.contiguous().T,
    }
    state_b = {"a": state_a["a"].clone(), "b": state_a["b"].clone()}
    assert canonical_model_state_sha256(state_a) == canonical_model_state_sha256(state_b)
    changed = {**state_b, "b": state_b["b"].clone()}
    changed["b"][0, 0] += 1.0
    assert canonical_model_state_sha256(changed) != canonical_model_state_sha256(state_a)

    path = tmp_path / "checkpoint.npz"
    save_neural_checkpoint(path, state_a, {"seed": 17, "scope": "test"})
    loaded = load_neural_checkpoint(path)
    assert canonical_model_state_sha256(loaded) == canonical_model_state_sha256(state_a)

    receipt_path = path.with_suffix(".npz.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["metadata"]["seed"] = 29
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata hash"):
        load_neural_checkpoint(path)
