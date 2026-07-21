"""Synthetic nested-fit and complete-refit implementation smoke for HEST v2."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from heir.evaluation.hierarchical_metrics import donor_section_type_macro_r2
from heir.evaluation.neural_model_selection import (
    NeuralCandidate,
    refit_selected_neural_probe,
    select_neural_hyperparameters,
)
from heir.evaluation.neural_nulls import NULL_FAMILIES, build_neural_null_design
from heir.evaluation.neural_probe import (
    load_neural_residual_fit,
    predict_neural_residual_probe,
    save_neural_residual_fit,
)
from heir.evaluation.nonlinear_qualification_contract import canonical_sha256
from heir.evaluation.nonlinear_qualification_v2_contract import ANALYSIS_STATUS
from heir.evaluation.ridge_probe import target_coordinates

SCHEMA = "heir.hest_nonlinear_v2_implementation_smoke.v1"
PERMUTATIONS = 20
COMPLETE_REFIT_STEPS = (
    "feature_preprocessing",
    "technical_correction",
    "molecular_target_fitting",
    "rank_selection",
    "neural_architecture_selection",
    "training",
    "checkpoint_selection",
    "prediction",
    "scoring",
)


def run_implementation_smoke(checkpoint_dir: Path, *, device: str) -> Mapping[str, object]:
    """Exercise fitting, replay, selection, and 20 refits per null without HEST RNA."""

    rng = np.random.default_rng(20_260_720)
    donors = np.repeat(["D1", "D2", "D3", "D4"], 24)
    labels = np.tile(np.repeat([0, 1], 12), 4)
    sections = np.asarray([f"{donor}-S" for donor in donors])
    blocks = np.tile(np.repeat(["a", "b"], 6), 8)
    roles = np.full(len(donors), "evaluation", dtype="<U10")
    identities = np.asarray([f"synthetic-{index:04d}" for index in range(len(donors))])
    features = rng.normal(size=(len(donors), 8)).astype(np.float32)
    technical = rng.normal(size=(len(donors), 1))
    reference = labels[:, None] * np.asarray([[0.1, -0.1, 0.05, -0.05]])
    signal = np.column_stack(
        (
            3.0 * features[:, 0] + labels * features[:, 1],
            features[:, 2] - 0.5 * features[:, 3],
            features[:, 4] * features[:, 5],
            features[:, 6] + labels * features[:, 7],
        )
    )
    targets = reference + signal
    targets += technical @ np.asarray([[0.05, -0.02, 0.01, 0.03]])
    candidates = (
        NeuralCandidate("mlp_tiny", True, 1.0e-4, 2),
        NeuralCandidate("mlp_small", True, 1.0e-2, 4),
    )

    def fit_and_score(local_features: np.ndarray) -> tuple[float, tuple[object, ...], object]:
        selection = select_neural_hyperparameters(
            local_features,
            targets,
            reference,
            labels,
            donors,
            sections,
            identities,
            technical,
            num_types=2,
            candidates=candidates,
            seeds=(17,),
            max_epochs=8,
            batch_size=32,
            patience=4,
            minimum_support=5,
            minimum_variance_ratio=0.0,
            minimum_basis_ceiling_r2=0.0,
            device=device,
        )
        fits = refit_selected_neural_probe(
            selection,
            local_features,
            targets,
            reference,
            labels,
            donors,
            sections,
            identities,
            technical,
            num_types=2,
            batch_size=32,
            device=device,
        )
        predictions = [
            predict_neural_residual_probe(
                fit,
                local_features,
                reference,
                labels,
                device=device,
            )[0]
            for fit in fits
        ]
        truth, _ = target_coordinates(fits[0].target, targets, reference, technical, labels)
        score, _, _, _ = donor_section_type_macro_r2(
            truth,
            np.mean(np.stack(predictions), axis=0),
            donors,
            sections,
            labels,
            5,
        )
        return float(score), fits, selection

    observed, fits, selection = fit_and_score(features)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "synthetic_probe.npz"
    checkpoint_receipt = save_neural_residual_fit(checkpoint_path, fits[0])
    replay = load_neural_residual_fit(checkpoint_path)
    first_prediction = predict_neural_residual_probe(
        fits[0], features, reference, labels, device=device
    )[0]
    replay_prediction = predict_neural_residual_probe(
        replay, features, reference, labels, device=device
    )[0]
    if not np.allclose(first_prediction, replay_prediction, rtol=0.0, atol=1.0e-7):
        raise RuntimeError("implementation-smoke checkpoint replay changed predictions")

    nulls = {}
    for family_index, family in enumerate(NULL_FAMILIES):
        design = build_neural_null_design(
            family,
            PERMUTATIONS,
            donors,
            sections,
            labels,
            roles,
            blocks,
            identities,
            seed=20_260_720 + family_index * 1_000_003,
        )
        values = []
        receipts = []
        for mapping, mapping_receipt in zip(design.mappings, design.receipts):
            if float(mapping_receipt["changed_fraction"]) <= 0.0:
                raise RuntimeError("implementation-smoke null mapping changed no image rows")
            score, local_fits, local_selection = fit_and_score(features[mapping])
            values.append(score)
            receipts.append(
                {
                    "mapping_sha256": mapping_receipt["mapping_sha256"],
                    "completed_steps": list(COMPLETE_REFIT_STEPS),
                    "score": score,
                    "selected_candidate_id": local_selection.selected.candidate_id,
                    "selected_epoch": local_selection.selected_epoch,
                    "checkpoint_sha256": [fit.checkpoint_sha256 for fit in local_fits],
                }
            )
        scores = np.asarray(values, dtype=np.float64)
        nulls[family] = {
            "permutations": PERMUTATIONS,
            "observed_statistic": observed,
            "empirical_p": float((1 + np.count_nonzero(scores >= observed)) / (1 + PERMUTATIONS)),
            "mapping_set_sha256": design.mapping_set_sha256,
            "refit_receipts_sha256": canonical_sha256(receipts),
            "completed_steps": list(COMPLETE_REFIT_STEPS),
            "model_refit_for_every_permutation": True,
        }
    return {
        "schema": SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "scope": "synthetic_only_no_registered_HEST_molecular_member_opened",
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "rank_candidates": sorted({int(candidate.rank) for candidate in candidates}),
        "selected_candidate_id": selection.selected.candidate_id,
        "selected_epoch": selection.selected_epoch,
        "checkpoint_receipt": checkpoint_receipt,
        "checkpoint_replay_identical": True,
        "permutations_per_family": PERMUTATIONS,
        "complete_refit_steps": list(COMPLETE_REFIT_STEPS),
        "nulls": nulls,
        "smoke_passed": True,
        "molecular_outcomes_read": False,
        "biological_experiment_run": False,
        "engineering_decision_available": False,
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_h_ref": False,
        "authorizes_full_heir": False,
    }


__all__ = ["COMPLETE_REFIT_STEPS", "PERMUTATIONS", "SCHEMA", "run_implementation_smoke"]
