from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from heir.evaluation.neural_probe import (
    NeuralProbeValidation,
    fit_neural_residual_probe,
    load_neural_residual_fit,
    predict_neural_residual_probe,
    save_neural_residual_fit,
)


def _probe_rows() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(22)
    donors = np.repeat(["D1", "D2", "D3", "D4"], 24)
    labels = np.tile(np.repeat([0, 1], 12), 4)
    sections = np.asarray([f"{donor}-S" for donor in donors])
    ids = np.asarray([f"obs-{index:04d}" for index in range(len(donors))])
    features = rng.normal(size=(len(donors), 5))
    technical = rng.normal(size=(len(donors), 1))
    reference = np.column_stack((labels * 0.2, labels * -0.1, labels * 0.05))
    coordinates = features[:, 0] * features[:, 1]
    targets = reference + np.column_stack((coordinates, -0.5 * coordinates, 0.25 * coordinates))
    targets += technical @ np.asarray([[0.1, -0.05, 0.02]])
    return features, targets, reference, labels, donors, sections, ids, technical


def test_complete_probe_checkpoint_reproduces_predictions(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    arrays = _probe_rows()
    fit = fit_neural_residual_probe(
        *arrays,
        num_types=2,
        rank=1,
        model_id="mlp_tiny",
        type_conditioned=True,
        weight_decay=1.0e-4,
        epochs=30,
        seed=17,
        batch_size=24,
        device="cpu",
    )
    before = predict_neural_residual_probe(fit, arrays[0], arrays[2], arrays[3], device="cpu")
    path = tmp_path / "probe.npz"
    receipt = save_neural_residual_fit(path, fit)
    loaded = load_neural_residual_fit(path)
    after = predict_neural_residual_probe(loaded, arrays[0], arrays[2], arrays[3], device="cpu")
    assert receipt["array_registry_sha256"]
    assert loaded.checkpoint_sha256 == fit.checkpoint_sha256
    assert after[0] == pytest.approx(before[0], abs=1.0e-7)
    assert after[1] == pytest.approx(before[1], abs=1.0e-7)


def test_probe_preprocessing_is_row_order_invariant() -> None:
    torch.set_num_threads(1)
    arrays = _probe_rows()
    first = fit_neural_residual_probe(
        *arrays,
        num_types=2,
        rank=1,
        model_id="mlp_tiny",
        type_conditioned=False,
        weight_decay=1.0e-4,
        epochs=10,
        seed=17,
        batch_size=32,
        device="cpu",
    )
    order = np.random.default_rng(91).permutation(len(arrays[0]))
    second = fit_neural_residual_probe(
        *(np.asarray(value)[order] for value in arrays),
        num_types=2,
        rank=1,
        model_id="mlp_tiny",
        type_conditioned=False,
        weight_decay=1.0e-4,
        epochs=10,
        seed=17,
        batch_size=32,
        device="cpu",
    )
    assert second.feature_mean == pytest.approx(first.feature_mean, abs=1.0e-12)
    assert second.target.bases == pytest.approx(first.target.bases, abs=1.0e-12)
    assert second.checkpoint_sha256 == first.checkpoint_sha256


def test_outer_training_and_validation_donors_must_be_disjoint() -> None:
    arrays = _probe_rows()
    training = np.flatnonzero(arrays[4] != "D4")
    validation = np.flatnonzero(arrays[4] == "D4")
    bad_donors = np.asarray(arrays[4])[validation].copy()
    bad_donors[0] = "D1"
    with pytest.raises(ValueError, match="donors overlap"):
        fit_neural_residual_probe(
            *(np.asarray(value)[training] for value in arrays),
            num_types=2,
            rank=1,
            model_id="mlp_tiny",
            type_conditioned=False,
            weight_decay=1.0e-4,
            epochs=2,
            seed=17,
            device="cpu",
            validation=NeuralProbeValidation(
                features=arrays[0][validation],
                molecular_targets=arrays[1][validation],
                reference_means=arrays[2][validation],
                type_labels=arrays[3][validation],
                donor_ids=bad_donors,
                section_ids=arrays[5][validation],
                observation_ids=arrays[6][validation],
                technical_covariates=arrays[7][validation],
                minimum_support=2,
            ),
        )
