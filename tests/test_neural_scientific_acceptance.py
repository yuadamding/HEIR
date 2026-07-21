from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch

from heir.evaluation.hest_nested_ridge import weighted_ridge_predict_grid
from heir.evaluation.neural_model_selection import nonlinear_complexity_supported
from heir.evaluation.neural_training import (
    NeuralArchitecture,
    predict_neural_model,
    train_neural_model,
)

REGISTERED_SUPPORT_FLOOR = 0.05
REGISTERED_COMPLEXITY_TAX = 0.01


@pytest.fixture(scope="module", autouse=True)
def _limit_torch_to_one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.square(truth_array - truth_array.mean(axis=0)).sum())
    assert denominator > 0.0
    return 1.0 - float(np.square(truth_array - prediction_array).sum()) / denominator


def _fit_mlp_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    identifier: str,
    max_epochs: int = 80,
) -> np.ndarray:
    train_labels = np.zeros(len(train_features), dtype=np.int64)
    fit = train_neural_model(
        train_features,
        train_targets,
        train_labels,
        np.asarray([f"{identifier}-{index:04d}" for index in range(len(train_features))]),
        np.ones(len(train_features)),
        architecture=NeuralArchitecture(
            "mlp_tiny",
            train_features.shape[1],
            train_targets.shape[1],
            1,
        ),
        max_epochs=max_epochs,
        batch_size=32,
        patience=10,
        seed=17,
        device="cpu",
    )
    return predict_neural_model(
        fit,
        test_features,
        np.zeros(len(test_features), dtype=np.int64),
        device="cpu",
    )


def _fit_ridge_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    return weighted_ridge_predict_grid(
        train_features,
        train_targets,
        test_features,
        (1.0e-4,),
        device="cpu",
    )[0]


def _xor_patterns() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_nonlinear_signal_supports_mlp_but_not_ridge() -> None:
    patterns = _xor_patterns()
    train_features = np.tile(patterns, (32, 1))
    test_features = np.tile(patterns, (8, 1))
    train_targets = (train_features[:, 0] * train_features[:, 1])[:, None]
    test_targets = (test_features[:, 0] * test_features[:, 1])[:, None]

    neural_r2 = _r2(
        test_targets,
        _fit_mlp_predict(
            train_features,
            train_targets,
            test_features,
            identifier="nonlinear",
        ),
    )
    ridge_r2 = _r2(
        test_targets,
        _fit_ridge_predict(train_features, train_targets, test_features),
    )

    assert neural_r2 > 0.95
    assert ridge_r2 < REGISTERED_SUPPORT_FLOOR
    assert nonlinear_complexity_supported(
        neural_r2,
        ridge_r2,
        minimum_gain=REGISTERED_COMPLEXITY_TAX,
    )


def test_global_null_supports_neither_mlp_nor_ridge() -> None:
    patterns = _xor_patterns()
    train_features = np.repeat(patterns, 32, axis=0)
    test_features = np.repeat(patterns, 16, axis=0)
    # Each identical feature vector has exactly balanced opposing outcomes, so
    # the conditional signal is globally null for both estimator classes.
    train_targets = np.tile(np.asarray([-1.0, 1.0]), len(train_features) // 2)[:, None]
    test_targets = np.tile(np.asarray([-1.0, 1.0]), len(test_features) // 2)[:, None]

    neural_r2 = _r2(
        test_targets,
        _fit_mlp_predict(
            train_features,
            train_targets,
            test_features,
            identifier="global-null",
            max_epochs=60,
        ),
    )
    ridge_r2 = _r2(
        test_targets,
        _fit_ridge_predict(train_features, train_targets, test_features),
    )

    assert neural_r2 < REGISTERED_SUPPORT_FLOOR
    assert ridge_r2 < REGISTERED_SUPPORT_FLOOR
    assert not nonlinear_complexity_supported(
        neural_r2,
        ridge_r2,
        minimum_gain=REGISTERED_COMPLEXITY_TAX,
    )


def test_random_row_split_exposes_donor_shortcut_that_lodo_rejects() -> None:
    donor_count = 6
    rows_per_donor = 40
    donors = np.repeat(np.arange(donor_count), rows_per_donor)
    features = np.repeat(np.eye(donor_count, dtype=np.float32), rows_per_donor, axis=0)
    donor_effect = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    within_donor = np.tile(np.linspace(-0.05, 0.05, rows_per_donor), donor_count)
    targets = (donor_effect[donors] + within_donor)[:, None]

    rng = np.random.default_rng(913)
    random_test = np.concatenate(
        [
            donor * rows_per_donor + rng.choice(rows_per_donor, 10, replace=False)
            for donor in range(donor_count)
        ]
    )
    random_training_mask = np.ones(len(targets), dtype=bool)
    random_training_mask[random_test] = False
    random_train = np.flatnonzero(random_training_mask)

    donor_train = np.flatnonzero(donors < 4)
    donor_test = np.flatnonzero(donors >= 4)

    random_ridge_r2 = _r2(
        targets[random_test],
        _fit_ridge_predict(features[random_train], targets[random_train], features[random_test]),
    )
    random_neural_r2 = _r2(
        targets[random_test],
        _fit_mlp_predict(
            features[random_train],
            targets[random_train],
            features[random_test],
            identifier="random-row",
            max_epochs=100,
        ),
    )
    donor_ridge_r2 = _r2(
        targets[donor_test],
        _fit_ridge_predict(features[donor_train], targets[donor_train], features[donor_test]),
    )
    donor_neural_r2 = _r2(
        targets[donor_test],
        _fit_mlp_predict(
            features[donor_train],
            targets[donor_train],
            features[donor_test],
            identifier="donor-held-out",
            max_epochs=100,
        ),
    )

    assert min(random_ridge_r2, random_neural_r2) > 0.95
    assert max(donor_ridge_r2, donor_neural_r2) < REGISTERED_SUPPORT_FLOOR
