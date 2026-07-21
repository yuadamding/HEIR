"""Fold-local B0-N7 fitting for HEST nonlinear qualification v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from heir.evaluation.hierarchical_metrics import (
    donor_section_type_macro_r2,
    macro_r2,
    macro_reconstruction_r2,
)
from heir.evaluation.neural_model_selection import (
    FIXED_SEEDS,
    NeuralCandidate,
    NeuralSelectionResult,
    _molecular_variance_ratio,
    refit_selected_neural_probe,
    select_neural_hyperparameters,
)
from heir.evaluation.neural_probe import predict_neural_residual_probe
from heir.evaluation.nonlinear_qualification_contract import canonical_sha256
from heir.evaluation.nonlinear_qualification_v2_acceptance import RegistrationComparison
from heir.evaluation.nonlinear_qualification_v2_data import HestBiologicalData
from heir.evaluation.residual_targets import correct_residuals
from heir.evaluation.ridge_probe import (
    MolecularTargetFit,
    OracleRidgeFit,
    fit_molecular_target,
    fit_oracle_ridge_probe,
    predict_oracle_ridge,
    target_coordinates,
)

ARM_IDS = (
    "B0",
    "B1",
    "B2",
    "B3",
    "N0",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "N7",
    "C_blank",
    "C_blank_single",
    "C_target_removed",
    "A_star",
)
PRIMARY_IMAGE_ARMS = ("N1", "N2", "N6", "N7")
SINGLE_IMAGE_ARMS = ("N1", "N2")
RIDGE_FAMILIES = ("technical", "spatial", "stain_qc", "morphometry_density", "combined_nonimage")
REGISTRATION_PAIRS = {
    "A_star_vs_N0": ("A_star", "N0"),
    "N1_vs_N5": ("N1", "N5"),
    "N3_vs_N5": ("N3", "N5"),
    "N4_vs_N5": ("N4", "N5"),
}


@dataclass(frozen=True)
class TrainingGrid:
    ranks: tuple[int, ...] = (2, 4, 6)
    ridge_alphas: tuple[float, ...] = (1.0e-4, 1.0e-2, 1.0, 100.0)
    weight_decays: tuple[float, ...] = (1.0e-4, 1.0e-2)
    seeds: tuple[int, ...] = FIXED_SEEDS
    maximum_epochs: int = 100
    batch_size: int = 256
    patience: int = 10
    learning_rate: float = 1.0e-3
    gradient_clip: float = 1.0
    minimum_support: int = 20
    minimum_variance_ratio: float = 0.5
    minimum_basis_ceiling_r2: float = 0.3


FROZEN_TRAINING_GRID = TrainingGrid()


@dataclass(frozen=True)
class RidgeCandidate:
    family: str
    alpha: float
    rank: int
    feature_width: int

    @property
    def candidate_id(self) -> str:
        return f"{self.family}__rank={self.rank}__alpha={self.alpha:.12g}"

    @property
    def parameter_count(self) -> int:
        return self.feature_width * self.rank + self.rank


@dataclass(frozen=True)
class RidgeSelectionResult:
    selected: RidgeCandidate
    candidates: tuple[Mapping[str, object], ...]
    inner_donors: tuple[str, ...]
    selection_rule: str


@dataclass(frozen=True)
class NeuralArmSelection:
    arm_id: str
    selection: NeuralSelectionResult
    winning_receipt: Mapping[str, object]


@dataclass(frozen=True)
class FeatureArm:
    arm_id: str
    features: np.ndarray
    view_dims: tuple[int, ...]
    candidates: tuple[NeuralCandidate, ...]


@dataclass(frozen=True)
class FittedArm:
    arm_id: str
    target: MolecularTargetFit
    coordinate_truth: np.ndarray
    coordinate_prediction: np.ndarray
    coordinate_ranks: np.ndarray
    molecular_prediction: np.ndarray
    corrected_molecular_truth: np.ndarray
    rare_lower: np.ndarray
    rare_upper: np.ndarray
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class OuterFoldModels:
    heldout_donor: str
    row_indices: np.ndarray
    arms: Mapping[str, FittedArm]
    registration: Mapping[str, RegistrationComparison]
    selected_primary_arm: str
    selected_single_arm: str
    selections: Mapping[str, object]
    corrected_molecular_truth: np.ndarray
    reference_prediction: np.ndarray


def validate_frozen_training_grid(grid: TrainingGrid) -> None:
    if grid != FROZEN_TRAINING_GRID:
        raise ValueError("production HEST v2 execution requires the exact inherited training grid")


def neural_candidates(arm_id: str, grid: TrainingGrid) -> tuple[NeuralCandidate, ...]:
    if arm_id == "B3":
        models = (("shared_linear", True),)
    elif arm_id in {"N0", "N1"}:
        models = (("mlp_tiny", False), ("mlp_small", False))
    elif arm_id == "N2":
        models = (("mlp_tiny", True), ("mlp_small", True))
    elif arm_id in {"N6", "N7"}:
        models = (("late_fusion", False),)
    else:
        raise ValueError(f"no frozen neural candidate registry exists for {arm_id}")
    return tuple(
        NeuralCandidate(model_id, conditioned, decay, rank)
        for model_id, conditioned in models
        for decay in grid.weight_decays
        for rank in grid.ranks
    )


def _corrected_truth(
    target: MolecularTargetFit,
    molecular_targets: np.ndarray,
    reference_means: np.ndarray,
    technical_covariates: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    result = np.asarray(reference_means, dtype=np.float64).copy()
    result += correct_residuals(
        molecular_targets,
        reference_means,
        technical_covariates,
        labels,
        target.technical_means,
        target.technical_coefficients,
    )
    return result


def _same_technical_target(left: MolecularTargetFit, right: MolecularTargetFit) -> bool:
    return bool(
        np.allclose(left.technical_means, right.technical_means, rtol=1.0e-10, atol=1.0e-10)
        and np.allclose(
            left.technical_coefficients,
            right.technical_coefficients,
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    )


def _rare_thresholds(
    target: MolecularTargetFit,
    train_targets: np.ndarray,
    train_reference: np.ndarray,
    train_technical: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates, _ = target_coordinates(
        target,
        train_targets,
        train_reference,
        train_technical,
        train_labels,
    )
    lower = np.empty((len(test_labels), target.rank), dtype=np.float64)
    upper = np.empty_like(lower)
    for type_index in sorted(set(test_labels.tolist())):
        training = train_labels == type_index
        if int(training.sum()) < 3:
            raise ValueError("rare-state thresholds lack outer-training type support")
        local_lower, local_upper = np.quantile(
            coordinates[training],
            (0.1, 0.9),
            axis=0,
            method="linear",
        )
        selected = test_labels == type_index
        lower[selected] = local_lower
        upper[selected] = local_upper
    if np.any(lower >= upper) or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("outer-training rare-state thresholds are degenerate")
    return lower, upper


def _ridge_fold(
    fit: OracleRidgeFit,
    *,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    validation_reference: np.ndarray,
    validation_labels: np.ndarray,
    validation_donors: np.ndarray,
    validation_sections: np.ndarray,
    validation_technical: np.ndarray,
    minimum_support: int,
) -> Mapping[str, object]:
    coordinates, molecular = predict_oracle_ridge(
        fit,
        validation_features,
        validation_reference,
        validation_labels,
    )
    truth, reconstruction = target_coordinates(
        fit.target,
        validation_targets,
        validation_reference,
        validation_technical,
        validation_labels,
    )
    corrected = _corrected_truth(
        fit.target,
        validation_targets,
        validation_reference,
        validation_technical,
        validation_labels,
    )
    donor_type, _, _ = macro_r2(
        truth,
        coordinates,
        validation_donors,
        validation_labels,
        minimum_support,
    )
    donor_section_type, _, _, _ = donor_section_type_macro_r2(
        truth,
        coordinates,
        validation_donors,
        validation_sections,
        validation_labels,
        minimum_support,
    )
    ceiling, _, _ = macro_reconstruction_r2(
        corrected,
        reconstruction,
        validation_reference,
        validation_donors,
        validation_labels,
        minimum_support,
    )
    return {
        "heldout_donors": sorted(set(validation_donors.tolist())),
        "donor_type_macro_r2": float(donor_type),
        "donor_section_type_macro_r2": float(donor_section_type),
        "variance_ratio": _molecular_variance_ratio(
            corrected,
            molecular,
            validation_donors,
            validation_labels,
            minimum_support,
        ),
        "basis_ceiling_r2": float(ceiling),
        "fit_device": "ridge_closed_form",
    }


def select_ridge_hyperparameters(
    feature_families: Mapping[str, np.ndarray],
    targets: np.ndarray,
    reference_means: np.ndarray,
    labels: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    observation_ids: np.ndarray,
    technical_covariates: np.ndarray,
    *,
    num_types: int,
    grid: TrainingGrid,
    device: str,
) -> RidgeSelectionResult:
    families = tuple(feature_families)
    if not families or any(
        name not in RIDGE_FAMILIES and name not in {"B1", "B2"} for name in families
    ):
        raise ValueError("ridge feature-family registry is empty or unknown")
    rows = len(donors)
    if any(len(np.asarray(value)) != rows for value in feature_families.values()):
        raise ValueError("ridge feature families are not row aligned")
    inner_donors = tuple(sorted(set(np.asarray(donors).astype(str).tolist())))
    candidates = tuple(
        RidgeCandidate(name, alpha, rank, np.asarray(feature_families[name]).shape[1])
        for name in families
        for alpha in grid.ridge_alphas
        for rank in grid.ranks
    )
    fold_rows: dict[str, list[Mapping[str, object]]] = {
        candidate.candidate_id: [] for candidate in candidates
    }
    failures: dict[str, str] = {}
    donor_values = np.asarray(donors).astype(str)
    ids = np.asarray(observation_ids).astype(str)
    for heldout in inner_donors:
        train = np.flatnonzero(donor_values != heldout)
        validation = np.flatnonzero(donor_values == heldout)
        train = train[np.argsort(ids[train], kind="stable")]
        target_cache: dict[int, MolecularTargetFit] = {}
        for rank in grid.ranks:
            target_cache[rank] = fit_molecular_target(
                targets[train],
                reference_means[train],
                labels[train],
                donor_values[train],
                technical_covariates[train],
                num_types=num_types,
                rank=rank,
                device=device,
            )
        for candidate in candidates:
            if candidate.candidate_id in failures:
                continue
            try:
                fit = fit_oracle_ridge_probe(
                    np.asarray(feature_families[candidate.family])[train],
                    targets[train],
                    reference_means[train],
                    labels[train],
                    donor_values[train],
                    technical_covariates[train],
                    num_types=num_types,
                    rank=candidate.rank,
                    alpha=candidate.alpha,
                    device=device,
                    target_fit=target_cache[candidate.rank],
                )
                fold_rows[candidate.candidate_id].append(
                    _ridge_fold(
                        fit,
                        validation_features=np.asarray(feature_families[candidate.family])[
                            validation
                        ],
                        validation_targets=targets[validation],
                        validation_reference=reference_means[validation],
                        validation_labels=labels[validation],
                        validation_donors=donor_values[validation],
                        validation_sections=sections[validation],
                        validation_technical=technical_covariates[validation],
                        minimum_support=grid.minimum_support,
                    )
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                failures[candidate.candidate_id] = f"{type(error).__name__}: {error}"
    receipts = []
    for candidate in candidates:
        folds = fold_rows[candidate.candidate_id]
        variance = [float(row["variance_ratio"]) for row in folds]
        ceilings = [float(row["basis_ceiling_r2"]) for row in folds]
        variance_pass = bool(
            variance
            and np.isfinite(variance).all()
            and float(np.median(variance)) >= grid.minimum_variance_ratio
        )
        ceiling_pass = bool(
            ceilings
            and np.isfinite(ceilings).all()
            and min(ceilings) >= grid.minimum_basis_ceiling_r2
        )
        accepted = bool(
            candidate.candidate_id not in failures
            and len(folds) == len(inner_donors)
            and variance_pass
            and ceiling_pass
        )
        receipts.append(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "alpha": candidate.alpha,
                "rank": candidate.rank,
                "feature_width": candidate.feature_width,
                "parameter_count": candidate.parameter_count,
                "accepted": accepted,
                "rejection_reason": (
                    failures.get(candidate.candidate_id)
                    or ("median_variance_ratio_below_threshold" if not variance_pass else None)
                    or ("basis_ceiling_below_threshold" if not ceiling_pass else None)
                ),
                "donor_section_type_macro_r2": (
                    float(np.mean([row["donor_section_type_macro_r2"] for row in folds]))
                    if folds
                    else None
                ),
                "donor_type_macro_r2": (
                    float(np.mean([row["donor_type_macro_r2"] for row in folds])) if folds else None
                ),
                "median_variance_ratio": float(np.median(variance)) if variance else None,
                "minimum_basis_ceiling_r2": min(ceilings) if ceilings else None,
                "folds": folds,
            }
        )
    accepted = [row for row in receipts if row["accepted"]]
    if not accepted:
        raise RuntimeError("every registered ridge candidate failed nested selection")
    winner = min(
        accepted,
        key=lambda row: (
            -float(row["donor_section_type_macro_r2"]),
            -float(row["donor_type_macro_r2"]),
            int(row["parameter_count"]),
            -float(row["alpha"]),
            str(row["family"]),
            str(row["candidate_id"]),
        ),
    )
    selected = next(
        candidate for candidate in candidates if candidate.candidate_id == winner["candidate_id"]
    )
    return RidgeSelectionResult(
        selected=selected,
        candidates=tuple(receipts),
        inner_donors=inner_donors,
        selection_rule=(
            "reject_failed_fold_and_frozen_basis_variance_screens; maximize_"
            "donor_section_type_r2; maximize_donor_type_r2; fewer_parameters; larger_alpha; "
            "lexicographically_smaller_family"
        ),
    )


def _winning_neural_receipt(selection: NeuralSelectionResult) -> Mapping[str, object]:
    return next(
        row
        for row in selection.candidates
        if row["candidate_id"] == selection.selected.candidate_id
    )


def select_neural_arm(
    arm: FeatureArm,
    targets: np.ndarray,
    reference_means: np.ndarray,
    labels: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    observation_ids: np.ndarray,
    technical_covariates: np.ndarray,
    *,
    num_types: int,
    grid: TrainingGrid,
    device: str,
) -> NeuralArmSelection:
    selection = select_neural_hyperparameters(
        arm.features,
        targets,
        reference_means,
        labels,
        donors,
        sections,
        observation_ids,
        technical_covariates,
        num_types=num_types,
        candidates=arm.candidates,
        seeds=grid.seeds,
        max_epochs=grid.maximum_epochs,
        batch_size=grid.batch_size,
        patience=grid.patience,
        learning_rate=grid.learning_rate,
        gradient_clip=grid.gradient_clip,
        minimum_support=grid.minimum_support,
        minimum_variance_ratio=grid.minimum_variance_ratio,
        minimum_basis_ceiling_r2=grid.minimum_basis_ceiling_r2,
        device=device,
        view_dims=arm.view_dims or None,
    )
    return NeuralArmSelection(arm.arm_id, selection, _winning_neural_receipt(selection))


def choose_neural_algorithm(
    selections: Mapping[str, NeuralArmSelection],
    candidate_arms: Sequence[str],
) -> str:
    if set(candidate_arms) - set(selections):
        raise ValueError("algorithm selection is missing a frozen candidate arm")
    return min(
        candidate_arms,
        key=lambda arm_id: (
            -float(selections[arm_id].winning_receipt["donor_section_type_macro_r2"]),
            -float(selections[arm_id].winning_receipt["donor_type_macro_r2"]),
            int(selections[arm_id].winning_receipt["parameter_count"]),
            -float(selections[arm_id].winning_receipt["weight_decay"]),
            str(selections[arm_id].winning_receipt["model_id"]),
            str(selections[arm_id].winning_receipt["candidate_id"]),
            arm_id,
        ),
    )


def _selection_receipt(selection: NeuralArmSelection) -> Mapping[str, object]:
    return {
        "arm_id": selection.arm_id,
        "selected_candidate_id": selection.selection.selected.candidate_id,
        "selected_epoch": selection.selection.selected_epoch,
        "inner_donors": list(selection.selection.inner_donors),
        "seeds": list(selection.selection.seeds),
        "selection_rule": selection.selection.selection_rule,
        "winning_receipt": dict(selection.winning_receipt),
        "candidate_registry_sha256": canonical_sha256(selection.selection.candidates),
        "candidate_count": len(selection.selection.candidates),
    }


def _fitted_neural_arm(
    arm_id: str,
    selection: NeuralArmSelection,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    train_reference: np.ndarray,
    test_reference: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_donors: np.ndarray,
    train_sections: np.ndarray,
    train_ids: np.ndarray,
    train_technical: np.ndarray,
    test_technical: np.ndarray,
    *,
    num_types: int,
    grid: TrainingGrid,
    device: str,
    view_dims: tuple[int, ...] = (),
) -> FittedArm:
    fits = refit_selected_neural_probe(
        selection.selection,
        train_features,
        train_targets,
        train_reference,
        train_labels,
        train_donors,
        train_sections,
        train_ids,
        train_technical,
        num_types=num_types,
        learning_rate=grid.learning_rate,
        batch_size=grid.batch_size,
        gradient_clip=grid.gradient_clip,
        device=device,
        view_dims=view_dims or None,
    )
    target = fits[0].target
    if any(not _same_target(target, fit.target) for fit in fits[1:]):
        raise RuntimeError("outer neural target fit changed across registered seeds")
    coordinates = []
    molecular = []
    for fit in fits:
        local_coordinates, local_molecular = predict_neural_residual_probe(
            fit,
            test_features,
            test_reference,
            test_labels,
            device=device,
            batch_size=grid.batch_size,
        )
        coordinates.append(local_coordinates)
        molecular.append(local_molecular)
    truth, _ = target_coordinates(
        target,
        test_targets,
        test_reference,
        test_technical,
        test_labels,
    )
    lower, upper = _rare_thresholds(
        target,
        train_targets,
        train_reference,
        train_technical,
        train_labels,
        test_labels,
    )
    return FittedArm(
        arm_id=arm_id,
        target=target,
        coordinate_truth=truth,
        coordinate_prediction=np.mean(np.stack(coordinates), axis=0),
        coordinate_ranks=np.full(len(test_labels), target.rank, dtype=np.int64),
        molecular_prediction=np.mean(np.stack(molecular), axis=0),
        corrected_molecular_truth=_corrected_truth(
            target,
            test_targets,
            test_reference,
            test_technical,
            test_labels,
        ),
        rare_lower=lower,
        rare_upper=upper,
        receipt={
            "estimator": "neural",
            "selection": _selection_receipt(selection),
            "outer_refit": [
                {
                    "seed": fit.training.seed,
                    "checkpoint_sha256": fit.checkpoint_sha256,
                    "best_epoch": fit.training.best_epoch,
                    "epochs_run": fit.training.epochs_run,
                    "parameter_count": fit.training.parameter_count,
                    "fit_device": fit.training.fit_device,
                }
                for fit in fits
            ],
        },
    )


def _same_target(left: MolecularTargetFit, right: MolecularTargetFit) -> bool:
    return bool(
        left.rank == right.rank
        and _same_technical_target(left, right)
        and np.allclose(left.residual_means, right.residual_means, rtol=1.0e-10, atol=1.0e-10)
        and np.allclose(left.bases, right.bases, rtol=1.0e-10, atol=1.0e-10)
    )


def _fitted_ridge_arm(
    arm_id: str,
    selection: RidgeSelectionResult,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    train_reference: np.ndarray,
    test_reference: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_donors: np.ndarray,
    train_technical: np.ndarray,
    test_technical: np.ndarray,
    *,
    num_types: int,
    device: str,
) -> FittedArm:
    candidate = selection.selected
    target = fit_molecular_target(
        train_targets,
        train_reference,
        train_labels,
        train_donors,
        train_technical,
        num_types=num_types,
        rank=candidate.rank,
        device=device,
    )
    fit = fit_oracle_ridge_probe(
        train_features,
        train_targets,
        train_reference,
        train_labels,
        train_donors,
        train_technical,
        num_types=num_types,
        rank=candidate.rank,
        alpha=candidate.alpha,
        device=device,
        target_fit=target,
    )
    prediction, molecular = predict_oracle_ridge(fit, test_features, test_reference, test_labels)
    truth, _ = target_coordinates(
        target,
        test_targets,
        test_reference,
        test_technical,
        test_labels,
    )
    lower, upper = _rare_thresholds(
        target,
        train_targets,
        train_reference,
        train_technical,
        train_labels,
        test_labels,
    )
    selected_receipt = next(
        row for row in selection.candidates if row["candidate_id"] == candidate.candidate_id
    )
    return FittedArm(
        arm_id=arm_id,
        target=target,
        coordinate_truth=truth,
        coordinate_prediction=prediction,
        coordinate_ranks=np.full(len(test_labels), target.rank, dtype=np.int64),
        molecular_prediction=molecular,
        corrected_molecular_truth=_corrected_truth(
            target,
            test_targets,
            test_reference,
            test_technical,
            test_labels,
        ),
        rare_lower=lower,
        rare_upper=upper,
        receipt={
            "estimator": "per_type_ridge",
            "selected_candidate": dict(selected_receipt),
            "candidate_registry_sha256": canonical_sha256(selection.candidates),
            "selection_rule": selection.selection_rule,
        },
    )


def project_molecular_prediction(
    target: MolecularTargetFit,
    molecular_prediction: np.ndarray,
    reference_means: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Project any gene-space comparator into the focal arm's target basis."""

    molecular = np.asarray(molecular_prediction, dtype=np.float64)
    reference = np.asarray(reference_means, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    if molecular.shape != reference.shape or label_values.shape != (len(molecular),):
        raise ValueError("focal-basis comparator projection inputs are not aligned")
    coordinates = np.zeros((len(molecular), target.rank), dtype=np.float64)
    for type_index in sorted(set(label_values.tolist())):
        selected = label_values == type_index
        residual = molecular[selected] - reference[selected] - target.residual_means[type_index]
        coordinates[selected] = residual @ target.bases[type_index]
    if not np.isfinite(coordinates).all():
        raise ValueError("focal-basis comparator projection is non-finite")
    return coordinates


def _feature_arms(
    data: HestBiologicalData,
    image_mapping: np.ndarray,
    grid: TrainingGrid,
) -> tuple[Mapping[str, FeatureArm], Mapping[str, np.ndarray]]:
    mapping = np.asarray(image_mapping, dtype=np.int64)
    if (
        mapping.shape != (len(data.observation_ids),)
        or np.any(mapping < 0)
        or np.any(mapping >= len(mapping))
        or len(np.unique(mapping)) != len(mapping)
    ):
        raise ValueError("image mapping must cover every registered source row")
    destination_rows = data.evaluation_indices
    source_rows = mapping[destination_rows]
    if not (
        np.array_equal(data.donor_ids[destination_rows], data.donor_ids[source_rows])
        and np.array_equal(data.section_ids[destination_rows], data.section_ids[source_rows])
        and np.array_equal(data.type_labels[destination_rows], data.type_labels[source_rows])
        and np.array_equal(data.pool_roles[destination_rows], data.pool_roles[source_rows])
    ):
        raise ValueError("image mapping crossed a frozen donor/section/type/pool-role stratum")
    views = data.image_features[source_rows]
    natural = views[:, 0]
    cell = views[:, 1]
    nucleus = views[:, 2]
    removed = views[:, 3]
    nuisance = np.asarray(data.control_families["combined_nonimage"], dtype=np.float32)
    n6 = np.column_stack((natural, nucleus, cell)).astype(np.float32, copy=False)
    n7 = np.column_stack((nuisance, natural, nucleus, cell)).astype(np.float32, copy=False)
    arms = {
        "B3": FeatureArm("B3", natural, (), neural_candidates("B3", grid)),
        "N0": FeatureArm("N0", nuisance, (), neural_candidates("N0", grid)),
        "N1": FeatureArm("N1", natural, (), neural_candidates("N1", grid)),
        "N2": FeatureArm("N2", natural, (), neural_candidates("N2", grid)),
        "N6": FeatureArm(
            "N6",
            n6,
            (natural.shape[1], nucleus.shape[1], cell.shape[1]),
            neural_candidates("N6", grid),
        ),
        "N7": FeatureArm(
            "N7",
            n7,
            (nuisance.shape[1], natural.shape[1], nucleus.shape[1], cell.shape[1]),
            neural_candidates("N7", grid),
        ),
    }
    local_views = {
        "natural": natural,
        "nucleus": nucleus,
        "cell": cell,
        "removed": removed,
        "nuisance": nuisance,
    }
    return arms, local_views


def _blank_features(
    selected_arm: str,
    views: Mapping[str, np.ndarray],
    blank_feature: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    rows = len(views["natural"])
    blank = np.repeat(np.asarray(blank_feature, dtype=np.float32), rows, axis=0)
    width = blank.shape[1]
    if selected_arm in {"N1", "N2"}:
        return blank, ()
    if selected_arm == "N6":
        return np.column_stack((blank, blank, blank)), (width, width, width)
    if selected_arm == "N7":
        nuisance = views["nuisance"]
        return (
            np.column_stack((nuisance, blank, blank, blank)),
            (nuisance.shape[1], width, width, width),
        )
    raise ValueError("blank control has no frozen architecture for selected image arm")


def _select_neural_registry(
    feature_arms: Mapping[str, FeatureArm],
    train: np.ndarray,
    data: HestBiologicalData,
    *,
    grid: TrainingGrid,
    device: str,
) -> Mapping[str, NeuralArmSelection]:
    donors = data.evaluation_donors
    sections = data.evaluation_sections
    labels = data.evaluation_types
    identities = data.evaluation_observation_ids
    return {
        arm_id: select_neural_arm(
            FeatureArm(
                arm_id,
                arm.features[train],
                arm.view_dims,
                arm.candidates,
            ),
            data.molecular_targets[train],
            data.reference_means[train],
            labels[train],
            donors[train],
            sections[train],
            identities[train],
            data.technical_covariates[train],
            num_types=len(data.type_names),
            grid=grid,
            device=device,
        )
        for arm_id, arm in feature_arms.items()
    }


def _fit_neural_from_arm(
    arm_id: str,
    selection: NeuralArmSelection,
    features: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    data: HestBiologicalData,
    *,
    grid: TrainingGrid,
    device: str,
    view_dims: tuple[int, ...] = (),
) -> FittedArm:
    return _fitted_neural_arm(
        arm_id,
        selection,
        features[train],
        features[test],
        data.molecular_targets[train],
        data.molecular_targets[test],
        data.reference_means[train],
        data.reference_means[test],
        data.evaluation_types[train],
        data.evaluation_types[test],
        data.evaluation_donors[train],
        data.evaluation_sections[train],
        data.evaluation_observation_ids[train],
        data.technical_covariates[train],
        data.technical_covariates[test],
        num_types=len(data.type_names),
        grid=grid,
        device=device,
        view_dims=view_dims,
    )


def fit_observed_outer_fold(
    data: HestBiologicalData,
    heldout_donor: str,
    image_mapping: np.ndarray,
    *,
    grid: TrainingGrid = FROZEN_TRAINING_GRID,
    device: str = "cuda",
) -> OuterFoldModels:
    donors = data.evaluation_donors
    train = np.flatnonzero(donors != heldout_donor)
    test = np.flatnonzero(donors == heldout_donor)
    if not len(train) or not len(test) or len(set(donors[train].tolist())) != 14:
        raise ValueError("outer HEST split is not leave-one-of-15-donors-out")
    feature_arms, views = _feature_arms(data, image_mapping, grid)
    natural = views["natural"]
    projection_rng = np.random.default_rng(20_260_720)
    projection = projection_rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(natural.shape[1], 96),
    ) / np.sqrt(96.0)
    ridge_features = {
        "B0": {name: values for name, values in data.control_families.items()},
        "B1": {"B1": natural @ projection},
        "B2": {"B2": natural},
    }
    ridge_selections = {
        arm_id: select_ridge_hyperparameters(
            {name: values[train] for name, values in families.items()},
            data.molecular_targets[train],
            data.reference_means[train],
            data.evaluation_types[train],
            donors[train],
            data.evaluation_sections[train],
            data.evaluation_observation_ids[train],
            data.technical_covariates[train],
            num_types=len(data.type_names),
            grid=grid,
            device=device,
        )
        for arm_id, families in ridge_features.items()
    }
    neural_selections = _select_neural_registry(
        feature_arms,
        train,
        data,
        grid=grid,
        device=device,
    )
    selected_primary = choose_neural_algorithm(neural_selections, PRIMARY_IMAGE_ARMS)
    selected_single = choose_neural_algorithm(neural_selections, SINGLE_IMAGE_ARMS)
    arms: dict[str, FittedArm] = {}
    for arm_id, families in ridge_features.items():
        selection = ridge_selections[arm_id]
        features = families[selection.selected.family]
        arms[arm_id] = _fitted_ridge_arm(
            arm_id,
            selection,
            features[train],
            features[test],
            data.molecular_targets[train],
            data.molecular_targets[test],
            data.reference_means[train],
            data.reference_means[test],
            data.evaluation_types[train],
            data.evaluation_types[test],
            donors[train],
            data.technical_covariates[train],
            data.technical_covariates[test],
            num_types=len(data.type_names),
            device=device,
        )
    for arm_id, arm in feature_arms.items():
        arms[arm_id] = _fit_neural_from_arm(
            arm_id,
            neural_selections[arm_id],
            arm.features,
            train,
            test,
            data,
            grid=grid,
            device=device,
            view_dims=arm.view_dims,
        )
    single_selection = neural_selections[selected_single]
    arms["N1"] = arms[selected_single]
    localization_features = {
        "N3": views["nucleus"],
        "N4": views["cell"],
        "N5": views["removed"],
    }
    for arm_id, features in localization_features.items():
        arms[arm_id] = _fit_neural_from_arm(
            arm_id,
            single_selection,
            features,
            train,
            test,
            data,
            grid=grid,
            device=device,
        )
    primary_blank, primary_blank_dims = _blank_features(
        selected_primary,
        views,
        data.blank_feature,
    )
    arms["C_blank"] = _fit_neural_from_arm(
        "C_blank",
        neural_selections[selected_primary],
        primary_blank,
        train,
        test,
        data,
        grid=grid,
        device=device,
        view_dims=primary_blank_dims,
    )
    single_blank, _ = _blank_features(selected_single, views, data.blank_feature)
    if selected_primary == selected_single:
        arms["C_blank_single"] = arms["C_blank"]
    else:
        arms["C_blank_single"] = _fit_neural_from_arm(
            "C_blank_single",
            single_selection,
            single_blank,
            train,
            test,
            data,
            grid=grid,
            device=device,
        )
    arms["C_target_removed"] = arms["N5"]
    arms["A_star"] = arms[selected_primary]
    focal_truth = arms["A_star"].corrected_molecular_truth
    if any(
        not np.allclose(focal_truth, arm.corrected_molecular_truth, rtol=1.0e-9, atol=1.0e-9)
        for arm in arms.values()
    ):
        raise RuntimeError("outer-fold technical correction changed across model arms")
    registration = {
        comparison_id: RegistrationComparison(
            truth=arms[focal].coordinate_truth,
            focal_prediction=arms[focal].coordinate_prediction,
            comparator_prediction=project_molecular_prediction(
                arms[focal].target,
                arms[comparator].molecular_prediction,
                data.reference_means[test],
                data.evaluation_types[test],
            ),
            coordinate_ranks=arms[focal].coordinate_ranks,
        )
        for comparison_id, (focal, comparator) in REGISTRATION_PAIRS.items()
    }
    return OuterFoldModels(
        heldout_donor=heldout_donor,
        row_indices=test,
        arms=arms,
        registration=registration,
        selected_primary_arm=selected_primary,
        selected_single_arm=selected_single,
        selections={
            "ridge": {
                arm_id: {
                    "selected_candidate_id": selection.selected.candidate_id,
                    "candidate_registry_sha256": canonical_sha256(selection.candidates),
                    "candidate_count": len(selection.candidates),
                    "selection_rule": selection.selection_rule,
                }
                for arm_id, selection in ridge_selections.items()
            },
            "neural": {
                arm_id: _selection_receipt(selection)
                for arm_id, selection in neural_selections.items()
            },
            "A_star": selected_primary,
            "A_single": selected_single,
        },
        corrected_molecular_truth=focal_truth,
        reference_prediction=data.reference_means[test],
    )


def fit_null_outer_fold(
    data: HestBiologicalData,
    heldout_donor: str,
    image_mapping: np.ndarray,
    *,
    grid: TrainingGrid = FROZEN_TRAINING_GRID,
    device: str = "cuda",
) -> Mapping[str, object]:
    donors = data.evaluation_donors
    train = np.flatnonzero(donors != heldout_donor)
    test = np.flatnonzero(donors == heldout_donor)
    feature_arms, _ = _feature_arms(data, image_mapping, grid)
    required = {name: feature_arms[name] for name in ("N0", *PRIMARY_IMAGE_ARMS)}
    selections = _select_neural_registry(required, train, data, grid=grid, device=device)
    selected_primary = choose_neural_algorithm(selections, PRIMARY_IMAGE_ARMS)
    focal = _fit_neural_from_arm(
        "A_star",
        selections[selected_primary],
        required[selected_primary].features,
        train,
        test,
        data,
        grid=grid,
        device=device,
        view_dims=required[selected_primary].view_dims,
    )
    nuisance = _fit_neural_from_arm(
        "N0",
        selections["N0"],
        required["N0"].features,
        train,
        test,
        data,
        grid=grid,
        device=device,
    )
    focal_score, _, focal_donors = macro_r2(
        focal.coordinate_truth,
        focal.coordinate_prediction,
        donors[test],
        data.evaluation_types[test],
        grid.minimum_support,
    )
    nuisance_score, _, nuisance_donors = macro_r2(
        nuisance.coordinate_truth,
        nuisance.coordinate_prediction,
        donors[test],
        data.evaluation_types[test],
        grid.minimum_support,
    )
    return {
        "heldout_donor": heldout_donor,
        "selected_primary_arm": selected_primary,
        "A_star_donor_type_macro_R2": float(focal_score),
        "N0_donor_type_macro_R2": float(nuisance_score),
        "registered_statistic": float(focal_score - nuisance_score),
        "A_star_per_donor": dict(focal_donors),
        "N0_per_donor": dict(nuisance_donors),
        "selection_receipts_sha256": canonical_sha256(
            {name: _selection_receipt(value) for name, value in selections.items()}
        ),
        "outer_refits": {
            "A_star": focal.receipt["outer_refit"],
            "N0": nuisance.receipt["outer_refit"],
        },
        "completed_steps": [
            "feature_preprocessing",
            "technical_correction",
            "molecular_target_fitting",
            "rank_selection",
            "neural_architecture_selection",
            "training",
            "checkpoint_selection",
            "prediction",
            "scoring",
        ],
    }


__all__ = [
    "ARM_IDS",
    "FROZEN_TRAINING_GRID",
    "FeatureArm",
    "FittedArm",
    "OuterFoldModels",
    "PRIMARY_IMAGE_ARMS",
    "REGISTRATION_PAIRS",
    "RIDGE_FAMILIES",
    "RidgeCandidate",
    "RidgeSelectionResult",
    "SINGLE_IMAGE_ARMS",
    "TrainingGrid",
    "choose_neural_algorithm",
    "fit_null_outer_fold",
    "fit_observed_outer_fold",
    "neural_candidates",
    "project_molecular_prediction",
    "select_neural_arm",
    "select_ridge_hyperparameters",
    "validate_frozen_training_grid",
]
