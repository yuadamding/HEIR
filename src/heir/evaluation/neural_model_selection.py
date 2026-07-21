"""Nested donor-level model selection for bounded neural residual probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from .hierarchical_metrics import (
    donor_section_type_macro_r2,
    macro_r2,
    macro_reconstruction_r2,
)
from .neural_probe import (
    NeuralProbeValidation,
    NeuralResidualFit,
    fit_neural_residual_probe,
    predict_neural_residual_probe,
)
from .residual_targets import correct_residuals
from .ridge_probe import target_coordinates

FIXED_SEEDS = (17, 29, 41)


@dataclass(frozen=True)
class NeuralCandidate:
    model_id: str
    type_conditioned: bool
    weight_decay: float
    rank: Optional[int] = None

    @property
    def candidate_id(self) -> str:
        conditioned = "type_adapter" if self.type_conditioned else "shared"
        rank = "registered" if self.rank is None else str(self.rank)
        return f"{self.model_id}__{conditioned}__rank={rank}__wd={self.weight_decay:.12g}"


@dataclass(frozen=True)
class NeuralSelectionResult:
    selected: NeuralCandidate
    selected_epoch: int
    candidates: tuple[Mapping[str, object], ...]
    inner_donors: tuple[str, ...]
    seeds: tuple[int, ...]
    selection_rule: str


def default_candidates(ranks: Sequence[int] = (2, 4, 6)) -> tuple[NeuralCandidate, ...]:
    rank_values = tuple(int(rank) for rank in ranks)
    if not rank_values or any(rank < 1 for rank in rank_values):
        raise ValueError("neural candidate ranks must be positive")
    return tuple(
        NeuralCandidate(model_id, conditioned, weight_decay, rank)
        for model_id, conditioned in (
            ("mlp_tiny", False),
            ("mlp_small", False),
            ("mlp_tiny", True),
            ("mlp_small", True),
        )
        for weight_decay in (1.0e-4, 1.0e-2)
        for rank in rank_values
    )


def _molecular_variance_ratio(
    truth: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    labels: np.ndarray,
    minimum_support: int,
) -> float:
    """Macro-average gene-wise SD preservation within donor/type strata."""

    truth_values = np.asarray(truth, dtype=np.float64)
    prediction_values = np.asarray(prediction, dtype=np.float64)
    if truth_values.shape != prediction_values.shape or truth_values.ndim != 2:
        raise ValueError("variance-ratio arrays must be aligned matrices")
    donor_values = np.asarray(donors).astype(str)
    label_values = np.asarray(labels, dtype=np.int64)
    if donor_values.shape != (len(truth_values),) or label_values.shape != donor_values.shape:
        raise ValueError("variance-ratio identities must be row aligned")
    donor_scores = []
    for donor in sorted(set(donor_values.tolist())):
        type_scores = []
        for type_index in sorted(set(label_values[donor_values == donor].tolist())):
            selected = (donor_values == donor) & (label_values == type_index)
            if int(selected.sum()) < minimum_support:
                continue
            truth_scale = np.std(truth_values[selected], axis=0)
            prediction_scale = np.std(prediction_values[selected], axis=0)
            valid = truth_scale > 1.0e-12
            if np.any(valid):
                type_scores.append(float(np.median(prediction_scale[valid] / truth_scale[valid])))
        if type_scores:
            donor_scores.append(float(np.mean(type_scores)))
    return float(np.mean(donor_scores)) if donor_scores else float("nan")


def _slice(values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return np.asarray(values)[selected]


def _candidate_rank(candidate: NeuralCandidate, registered_rank: Optional[int]) -> int:
    value = candidate.rank if candidate.rank is not None else registered_rank
    if value is None or int(value) < 1:
        raise ValueError("neural candidate rank is not registered")
    return int(value)


def _candidate_fold(
    candidate: NeuralCandidate,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    features: np.ndarray,
    targets: np.ndarray,
    reference_means: np.ndarray,
    labels: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    observation_ids: np.ndarray,
    technical_covariates: np.ndarray,
    num_types: int,
    rank: int,
    seeds: Sequence[int],
    max_epochs: int,
    batch_size: int,
    patience: int,
    learning_rate: float,
    gradient_clip: float,
    minimum_support: int,
    device: str,
    view_dims: Optional[tuple[int, ...]],
) -> Mapping[str, object]:
    coordinate_predictions = []
    molecular_predictions = []
    seed_receipts = []
    truth = None
    corrected_truth = None
    local_reconstruction = None
    for seed in seeds:
        fit = fit_neural_residual_probe(
            _slice(features, train),
            _slice(targets, train),
            _slice(reference_means, train),
            _slice(labels, train),
            _slice(donors, train),
            _slice(sections, train),
            _slice(observation_ids, train),
            _slice(technical_covariates, train),
            num_types=num_types,
            rank=_candidate_rank(candidate, rank),
            model_id=candidate.model_id,
            type_conditioned=candidate.type_conditioned,
            weight_decay=candidate.weight_decay,
            epochs=max_epochs,
            seed=int(seed),
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience,
            gradient_clip=gradient_clip,
            device=device,
            validation=NeuralProbeValidation(
                features=_slice(features, validation),
                molecular_targets=_slice(targets, validation),
                reference_means=_slice(reference_means, validation),
                type_labels=_slice(labels, validation),
                donor_ids=_slice(donors, validation),
                section_ids=_slice(sections, validation),
                observation_ids=_slice(observation_ids, validation),
                technical_covariates=_slice(technical_covariates, validation),
                minimum_support=minimum_support,
            ),
            view_dims=view_dims,
        )
        predicted, molecular_prediction = predict_neural_residual_probe(
            fit,
            _slice(features, validation),
            _slice(reference_means, validation),
            _slice(labels, validation),
            device=device,
            batch_size=batch_size,
        )
        local_truth, local_reconstruction = target_coordinates(
            fit.target,
            _slice(targets, validation),
            _slice(reference_means, validation),
            _slice(technical_covariates, validation),
            _slice(labels, validation),
        )
        if truth is None:
            truth = local_truth
        elif not np.allclose(truth, local_truth, rtol=1.0e-10, atol=1.0e-10):
            raise RuntimeError("fixed target fitting changed across neural seeds")
        local_corrected_truth = _slice(reference_means, validation).astype(np.float64).copy()
        local_corrected_truth += correct_residuals(
            _slice(targets, validation),
            _slice(reference_means, validation),
            _slice(technical_covariates, validation),
            _slice(labels, validation),
            fit.target.technical_means,
            fit.target.technical_coefficients,
        )
        if corrected_truth is None:
            corrected_truth = local_corrected_truth
        elif not np.allclose(
            corrected_truth, local_corrected_truth, rtol=1.0e-10, atol=1.0e-10
        ):
            raise RuntimeError("technical correction changed across neural seeds")
        coordinate_predictions.append(predicted)
        molecular_predictions.append(molecular_prediction)
        seed_receipts.append(
            {
                "seed": int(seed),
                "best_epoch": fit.training.best_epoch,
                "epochs_run": fit.training.epochs_run,
                "checkpoint_sha256": fit.checkpoint_sha256,
                "parameter_count": fit.training.parameter_count,
                "fit_device": fit.training.fit_device,
            }
        )
    assert truth is not None and corrected_truth is not None and local_reconstruction is not None
    ensemble = np.mean(np.stack(coordinate_predictions, axis=0), axis=0)
    molecular_ensemble = np.mean(np.stack(molecular_predictions, axis=0), axis=0)
    basis_ceiling, _, _ = macro_reconstruction_r2(
        corrected_truth,
        local_reconstruction,
        _slice(reference_means, validation),
        _slice(donors, validation).astype(str),
        _slice(labels, validation).astype(np.int64),
        minimum_support,
    )
    donor_type, _, donor_type_values = macro_r2(
        truth,
        ensemble,
        _slice(donors, validation).astype(str),
        _slice(labels, validation).astype(np.int64),
        minimum_support,
    )
    donor_section_type, _, donor_section_values, _ = donor_section_type_macro_r2(
        truth,
        ensemble,
        _slice(donors, validation).astype(str),
        _slice(sections, validation).astype(str),
        _slice(labels, validation).astype(np.int64),
        minimum_support,
    )
    return {
        "heldout_donors": sorted(set(_slice(donors, validation).astype(str).tolist())),
        "donor_type_macro_r2": float(donor_type),
        "donor_section_type_macro_r2": float(donor_section_type),
        "variance_ratio": _molecular_variance_ratio(
            corrected_truth,
            molecular_ensemble,
            _slice(donors, validation),
            _slice(labels, validation),
            minimum_support,
        ),
        "basis_ceiling_r2": float(basis_ceiling),
        "per_donor_type_macro_r2": dict(donor_type_values),
        "per_donor_section_type_macro_r2": dict(donor_section_values),
        "seeds": seed_receipts,
    }


def select_neural_hyperparameters(
    features: np.ndarray,
    targets: np.ndarray,
    reference_means: np.ndarray,
    labels: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    observation_ids: np.ndarray,
    technical_covariates: np.ndarray,
    *,
    num_types: int,
    rank: Optional[int] = None,
    candidates: Optional[Sequence[NeuralCandidate]] = None,
    seeds: Sequence[int] = FIXED_SEEDS,
    max_epochs: int = 100,
    batch_size: int = 256,
    patience: int = 10,
    learning_rate: float = 1.0e-3,
    gradient_clip: float = 1.0,
    minimum_support: int = 5,
    minimum_variance_ratio: float = 0.5,
    minimum_basis_ceiling_r2: float = 0.3,
    device: str = "auto",
    view_dims: Optional[tuple[int, ...]] = None,
) -> NeuralSelectionResult:
    """Select architecture/decay using inner leave-one-donor-out folds only."""

    donor_values = np.asarray(donors).astype(str)
    row_count = len(donor_values)
    arrays = (
        targets,
        reference_means,
        labels,
        sections,
        observation_ids,
        technical_covariates,
    )
    if len(np.asarray(features)) != row_count or any(
        len(np.asarray(value)) != row_count for value in arrays
    ):
        raise ValueError("nested neural-selection arrays must be row aligned")
    inner_donors = tuple(sorted(set(donor_values.tolist())))
    if len(inner_donors) < 3:
        raise ValueError("nested neural selection requires at least three development donors")
    seed_values = tuple(int(seed) for seed in seeds)
    if len(seed_values) != len(set(seed_values)) or any(seed < 0 for seed in seed_values):
        raise ValueError("neural selection seeds must be unique and non-negative")
    default_ranks = (rank,) if rank is not None else (2, 4, 6)
    candidate_values = tuple(candidates or default_candidates(default_ranks))
    candidate_ids = {candidate.candidate_id for candidate in candidate_values}
    if not candidate_values or len(candidate_ids) != len(candidate_values):
        raise ValueError("neural selection candidates must be non-empty and unique")
    if rank is None and any(candidate.rank is None for candidate in candidate_values):
        raise ValueError("every neural candidate must register a rank")
    if (
        not np.isfinite(minimum_variance_ratio)
        or minimum_variance_ratio < 0.0
        or not np.isfinite(minimum_basis_ceiling_r2)
        or minimum_basis_ceiling_r2 < 0.0
    ):
        raise ValueError("neural selection screens must be finite and non-negative")

    receipts = []
    for candidate in candidate_values:
        folds = []
        failure = None
        for heldout in inner_donors:
            train = np.flatnonzero(donor_values != heldout)
            validation = np.flatnonzero(donor_values == heldout)
            try:
                fold = _candidate_fold(
                    candidate,
                    train,
                    validation,
                    features=np.asarray(features),
                    targets=np.asarray(targets),
                    reference_means=np.asarray(reference_means),
                    labels=np.asarray(labels),
                    donors=donor_values,
                    sections=np.asarray(sections),
                    observation_ids=np.asarray(observation_ids),
                    technical_covariates=np.asarray(technical_covariates),
                    num_types=num_types,
                    rank=_candidate_rank(candidate, rank),
                    seeds=seed_values,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    patience=patience,
                    learning_rate=learning_rate,
                    gradient_clip=gradient_clip,
                    minimum_support=minimum_support,
                    device=device,
                    view_dims=view_dims,
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                failure = f"{type(error).__name__}: {error}"
                break
            folds.append(fold)
        variance_values = [float(fold["variance_ratio"]) for fold in folds]
        basis_values = [float(fold["basis_ceiling_r2"]) for fold in folds]
        valid_variance = bool(
            variance_values
            and np.all(np.isfinite(variance_values))
            and float(np.median(variance_values)) >= minimum_variance_ratio
        )
        valid_basis = bool(
            basis_values
            and np.all(np.isfinite(basis_values))
            and min(basis_values) >= minimum_basis_ceiling_r2
        )
        accepted = (
            failure is None and len(folds) == len(inner_donors) and valid_variance and valid_basis
        )
        epochs = [
            int(seed_receipt["best_epoch"]) for fold in folds for seed_receipt in fold["seeds"]
        ]
        parameter_counts = [
            int(seed_receipt["parameter_count"]) for fold in folds for seed_receipt in fold["seeds"]
        ]
        receipts.append(
            {
                "candidate_id": candidate.candidate_id,
                "model_id": candidate.model_id,
                "type_conditioned": candidate.type_conditioned,
                "weight_decay": candidate.weight_decay,
                "rank": _candidate_rank(candidate, rank),
                "accepted": accepted,
                "rejection_reason": (
                    failure
                    if failure is not None
                    else (
                        "median_variance_ratio_below_threshold"
                        if not valid_variance
                        else "basis_ceiling_below_threshold"
                        if not valid_basis
                        else None
                    )
                ),
                "folds": folds,
                "median_variance_ratio": (
                    float(np.median(variance_values)) if variance_values else None
                ),
                "observed_minimum_basis_ceiling_r2": (
                    min(basis_values) if basis_values else None
                ),
                "required_minimum_variance_ratio": float(minimum_variance_ratio),
                "required_minimum_basis_ceiling_r2": float(minimum_basis_ceiling_r2),
                "donor_section_type_macro_r2": (
                    float(np.mean([fold["donor_section_type_macro_r2"] for fold in folds]))
                    if folds
                    else None
                ),
                "donor_type_macro_r2": (
                    float(np.mean([fold["donor_type_macro_r2"] for fold in folds]))
                    if folds
                    else None
                ),
                "parameter_count": max(parameter_counts) if parameter_counts else None,
                "median_selected_epoch": int(np.median(epochs)) if epochs else None,
            }
        )
    accepted = [receipt for receipt in receipts if bool(receipt["accepted"])]
    if not accepted:
        raise RuntimeError("every registered neural candidate failed nested selection")
    winner = min(
        accepted,
        key=lambda receipt: (
            -float(receipt["donor_section_type_macro_r2"]),
            -float(receipt["donor_type_macro_r2"]),
            int(receipt["parameter_count"]),
            -float(receipt["weight_decay"]),
            str(receipt["model_id"]),
            str(receipt["candidate_id"]),
        ),
    )
    selected = next(
        candidate
        for candidate in candidate_values
        if candidate.candidate_id == winner["candidate_id"]
    )
    return NeuralSelectionResult(
        selected=selected,
        selected_epoch=int(winner["median_selected_epoch"]),
        candidates=tuple(receipts),
        inner_donors=inner_donors,
        seeds=seed_values,
        selection_rule=(
            f"reject_failed_fold; reject_minimum_basis_ceiling_below_"
            f"{minimum_basis_ceiling_r2:.12g}; reject_median_variance_ratio_below_"
            f"{minimum_variance_ratio:.12g}; maximize_"
            "donor_section_type_r2; maximize_donor_type_r2; fewer_parameters; larger_"
            "weight_decay; lexicographically_smaller_model_id"
        ),
    )


def refit_selected_neural_probe(
    selection: NeuralSelectionResult,
    features: np.ndarray,
    targets: np.ndarray,
    reference_means: np.ndarray,
    labels: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    observation_ids: np.ndarray,
    technical_covariates: np.ndarray,
    *,
    num_types: int,
    rank: Optional[int] = None,
    learning_rate: float = 1.0e-3,
    batch_size: int = 256,
    gradient_clip: float = 1.0,
    device: str = "auto",
    view_dims: Optional[tuple[int, ...]] = None,
) -> tuple[NeuralResidualFit, ...]:
    """Refit the selected configuration on all supplied development donors."""

    fits = []
    for seed in selection.seeds:
        fits.append(
            fit_neural_residual_probe(
                features,
                targets,
                reference_means,
                labels,
                donors,
                sections,
                observation_ids,
                technical_covariates,
                num_types=num_types,
                rank=_candidate_rank(selection.selected, rank),
                model_id=selection.selected.model_id,
                type_conditioned=selection.selected.type_conditioned,
                weight_decay=selection.selected.weight_decay,
                epochs=selection.selected_epoch,
                seed=seed,
                learning_rate=learning_rate,
                batch_size=batch_size,
                patience=max(selection.selected_epoch, 1),
                gradient_clip=gradient_clip,
                device=device,
                view_dims=view_dims,
            )
        )
    return tuple(fits)


def nonlinear_complexity_supported(
    neural_r2: float, ridge_r2: float, *, minimum_gain: float = 0.01
) -> bool:
    if minimum_gain < 0.0 or not np.isfinite(neural_r2) or not np.isfinite(ridge_r2):
        raise ValueError("complexity comparison requires finite scores and a non-negative tax")
    return bool(neural_r2 - ridge_r2 >= minimum_gain)


__all__ = [
    "FIXED_SEEDS",
    "NeuralCandidate",
    "NeuralSelectionResult",
    "default_candidates",
    "nonlinear_complexity_supported",
    "refit_selected_neural_probe",
    "select_neural_hyperparameters",
]
