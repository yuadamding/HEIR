from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from heir.evaluation import nonlinear_qualification_v2_models as models
from heir.evaluation.neural_model_selection import NeuralCandidate, NeuralSelectionResult
from heir.evaluation.ridge_probe import MolecularTargetFit


def test_frozen_neural_candidate_registries_match_the_B3_N0_N7_roles() -> None:
    grid = models.FROZEN_TRAINING_GRID
    assert len(models.neural_candidates("B3", grid)) == 6
    assert len(models.neural_candidates("N0", grid)) == 12
    assert len(models.neural_candidates("N1", grid)) == 12
    assert len(models.neural_candidates("N2", grid)) == 12
    assert len(models.neural_candidates("N6", grid)) == 6
    assert len(models.neural_candidates("N7", grid)) == 6
    assert {value.model_id for value in models.neural_candidates("B3", grid)} == {"shared_linear"}
    assert all(value.type_conditioned for value in models.neural_candidates("B3", grid))
    assert all(not value.type_conditioned for value in models.neural_candidates("N1", grid))
    assert all(value.type_conditioned for value in models.neural_candidates("N2", grid))
    assert all(not value.type_conditioned for value in models.neural_candidates("N7", grid))


def _neural_selection(arm_id: str, *, parameters: int) -> models.NeuralArmSelection:
    candidate = NeuralCandidate("mlp_tiny", arm_id == "N2", 1.0e-2, 2)
    receipt = {
        "candidate_id": candidate.candidate_id,
        "model_id": candidate.model_id,
        "weight_decay": candidate.weight_decay,
        "parameter_count": parameters,
        "donor_section_type_macro_r2": 0.2,
        "donor_type_macro_r2": 0.2,
    }
    selection = NeuralSelectionResult(
        selected=candidate,
        selected_epoch=3,
        candidates=(receipt,),
        inner_donors=("D1", "D2", "D3"),
        seeds=(17,),
        selection_rule="test",
    )
    return models.NeuralArmSelection(arm_id, selection, receipt)


def test_A_star_and_A_single_use_the_frozen_cross_arm_tie_breaking() -> None:
    selections = {
        "N1": _neural_selection("N1", parameters=100),
        "N2": _neural_selection("N2", parameters=90),
        "N6": _neural_selection("N6", parameters=120),
        "N7": _neural_selection("N7", parameters=130),
    }
    assert models.choose_neural_algorithm(selections, models.PRIMARY_IMAGE_ARMS) == "N2"
    assert models.choose_neural_algorithm(selections, models.SINGLE_IMAGE_ARMS) == "N2"


def test_comparator_gene_prediction_is_projected_into_the_focal_basis() -> None:
    target = MolecularTargetFit(
        technical_means=np.zeros((2, 1)),
        technical_coefficients=np.zeros((2, 1, 3)),
        residual_means=np.asarray([[1.0, 2.0, 0.0], [-1.0, 0.5, 2.0]]),
        bases=np.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            ]
        ),
        rank=2,
    )
    labels = np.asarray([0, 1])
    reference = np.asarray([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    expected = np.asarray([[0.25, -0.5], [1.5, 2.0]])
    molecular = reference.copy()
    for type_index in (0, 1):
        selected = labels == type_index
        molecular[selected] += target.residual_means[type_index]
        molecular[selected] += expected[selected] @ target.bases[type_index].T

    observed = models.project_molecular_prediction(target, molecular, reference, labels)
    assert np.allclose(observed, expected)


def test_nested_ridge_selection_uses_inner_donors_and_rank_grid() -> None:
    rng = np.random.default_rng(11)
    donors = np.repeat(["D1", "D2", "D3"], 8)
    sections = np.asarray([f"{donor}-S" for donor in donors])
    labels = np.zeros(len(donors), dtype=np.int64)
    ids = np.asarray([f"row-{index:03d}" for index in range(len(donors))])
    features = rng.normal(size=(len(donors), 2))
    signal = features[:, 0] - 0.5 * features[:, 1]
    targets = np.column_stack((signal, -signal, 0.5 * signal))
    reference = np.zeros_like(targets)
    technical = np.zeros((len(donors), 1))
    grid = models.TrainingGrid(
        ranks=(1,),
        ridge_alphas=(0.01,),
        weight_decays=(0.01,),
        seeds=(17,),
        maximum_epochs=1,
        batch_size=8,
        patience=1,
        minimum_support=2,
        minimum_variance_ratio=0.0,
        minimum_basis_ceiling_r2=0.0,
    )

    result = models.select_ridge_hyperparameters(
        {"technical": features},
        targets,
        reference,
        labels,
        donors,
        sections,
        ids,
        technical,
        num_types=1,
        grid=grid,
        device="cpu",
    )

    assert result.selected.rank == 1
    assert result.selected.alpha == 0.01
    assert result.inner_donors == ("D1", "D2", "D3")
    assert len(result.candidates[0]["folds"]) == 3


def test_image_mapping_cannot_cross_a_frozen_null_stratum() -> None:
    data = SimpleNamespace(
        observation_ids=np.asarray(["a", "b", "c", "d"]),
        evaluation_indices=np.arange(4),
        donor_ids=np.asarray(["D1", "D1", "D2", "D2"]),
        section_ids=np.asarray(["S1", "S1", "S2", "S2"]),
        type_labels=np.asarray([0, 0, 0, 0]),
        pool_roles=np.full(4, "evaluation"),
        image_features=np.zeros((4, 4, 3), dtype=np.float32),
        control_families={"combined_nonimage": np.zeros((4, 2), dtype=np.float32)},
    )

    with pytest.raises(ValueError, match="crossed a frozen"):
        models._feature_arms(
            data,
            np.asarray([2, 1, 0, 3]),
            models.FROZEN_TRAINING_GRID,
        )
