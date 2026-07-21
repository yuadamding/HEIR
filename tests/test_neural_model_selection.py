from __future__ import annotations

import numpy as np

import heir.evaluation.neural_model_selection as selection_module
from heir.evaluation.neural_model_selection import (
    NeuralCandidate,
    _candidate_fold,
    _molecular_variance_ratio,
    nonlinear_complexity_supported,
    select_neural_hyperparameters,
)


def _selection_rows() -> tuple[np.ndarray, ...]:
    donors = np.repeat(["D1", "D2", "D3"], 4)
    rows = len(donors)
    return (
        np.ones((rows, 3)),
        np.ones((rows, 2)),
        np.zeros((rows, 2)),
        np.zeros(rows, dtype=np.int64),
        donors,
        np.asarray([f"{donor}-S" for donor in donors]),
        np.asarray([f"row-{index}" for index in range(rows)]),
        np.ones((rows, 1)),
    )


def test_linear_signal_complexity_tax_retains_ridge() -> None:
    assert nonlinear_complexity_supported(0.509, 0.5) is False
    assert nonlinear_complexity_supported(0.51, 0.5) is True


def test_exact_selection_tie_break_prefers_larger_weight_decay(monkeypatch) -> None:
    def fold(candidate, *args, **kwargs):
        return {
            "heldout_donors": ["D"],
            "donor_type_macro_r2": 0.2,
            "donor_section_type_macro_r2": 0.2,
            "variance_ratio": 0.8,
            "basis_ceiling_r2": 0.8,
            "per_donor_type_macro_r2": {"D": 0.2},
            "per_donor_section_type_macro_r2": {"D": 0.2},
            "seeds": [
                {
                    "seed": 17,
                    "best_epoch": 8,
                    "epochs_run": 8,
                    "checkpoint_sha256": "0" * 64,
                    "parameter_count": 100,
                    "fit_device": "cpu",
                }
            ],
        }

    monkeypatch.setattr(selection_module, "_candidate_fold", fold)
    rows = _selection_rows()
    result = select_neural_hyperparameters(
        *rows,
        num_types=1,
        rank=1,
        candidates=(
            NeuralCandidate("mlp_tiny", False, 1.0e-4),
            NeuralCandidate("mlp_tiny", False, 1.0e-2),
        ),
        seeds=(17,),
        max_epochs=8,
        minimum_support=1,
        device="cpu",
    )
    assert result.selected.weight_decay == 1.0e-2
    assert result.selected_epoch == 8
    assert result.inner_donors == ("D1", "D2", "D3")


def test_basis_ceiling_scores_technically_corrected_gene_truth() -> None:
    rng = np.random.default_rng(4)
    donors = np.repeat(["D1", "D2", "D3"], 24)
    labels = np.zeros(len(donors), dtype=np.int64)
    sections = np.asarray([f"{donor}-S" for donor in donors])
    observation_ids = np.asarray([f"row-{index:03d}" for index in range(len(donors))])
    technical = np.tile(np.linspace(-1.0, 1.0, 24), 3)[:, None]
    features = rng.normal(size=(len(donors), 3))
    coordinate = features[:, 0]
    reference = np.zeros((len(donors), 3))
    targets = np.column_stack(
        (
            30.0 * technical[:, 0] + coordinate,
            -15.0 * technical[:, 0] + 2.0 * coordinate,
            7.0 * technical[:, 0] - coordinate,
        )
    )
    training = np.flatnonzero(donors != "D3")
    validation = np.flatnonzero(donors == "D3")
    fold = _candidate_fold(
        NeuralCandidate("shared_linear", False, 1.0e-4, 1),
        training,
        validation,
        features=features,
        targets=targets,
        reference_means=reference,
        labels=labels,
        donors=donors,
        sections=sections,
        observation_ids=observation_ids,
        technical_covariates=technical,
        num_types=1,
        rank=1,
        seeds=(17,),
        max_epochs=2,
        batch_size=16,
        patience=2,
        learning_rate=1.0e-3,
        gradient_clip=1.0,
        minimum_support=5,
        device="cpu",
        view_dims=None,
    )
    assert fold["basis_ceiling_r2"] == 1.0


def test_molecular_variance_is_gene_wise_within_donor_type() -> None:
    donors = np.repeat(["D1", "D2"], 8)
    labels = np.tile(np.repeat([0, 1], 4), 2)
    local = np.tile(np.asarray([-1.5, -0.5, 0.5, 1.5]), 4)
    truth = np.column_stack((local, 2.0 * local))
    prediction = 0.5 * truth
    assert _molecular_variance_ratio(truth, prediction, donors, labels, 4) == 0.5
