"""Oracle-type ridge probe for the minimum morphology-to-RNA falsification test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from heir.data import MorphologyRidgeDatasetArtifact
from heir.utils import resolve_device

MORPHOLOGY_RIDGE_REPORT_SCHEMA = "heir.morphology_ridge_evaluation.v2"
REGIONAL_UNI2H_TECHNICAL_COVARIATES = ("log1p_library_size",)
REGIONAL_UNI2H_COMPOSITION_FEATURES = (
    "composition_epithelial",
    "composition_immune",
    "composition_stromal",
    "composition_endothelial",
)


def validate_experiment_identity(
    artifact: MorphologyRidgeDatasetArtifact, experiment_role: str
) -> None:
    """Enforce the prespecified frozen encoder and deterministic crop role."""

    expected_encoder = {
        "primary_hoptimus1": "bioptimus/H-optimus-1",
        "replication_h0mini": "bioptimus/H0-mini",
        "confirmation_xenium": "bioptimus/H-optimus-1",
        "regional_hescape_hoptimus1": "bioptimus/H-optimus-1",
        "regional_hescape_uni2h": "MahmoodLab/UNI2-h",
    }.get(experiment_role)
    if experiment_role not in {
        "primary_hoptimus1",
        "replication_h0mini",
        "context_sensitivity",
        "confirmation_xenium",
        "regional_hescape_hoptimus1",
        "regional_hescape_uni2h",
    }:
        raise ValueError("morphology-ridge experiment role is unsupported")
    if expected_encoder is not None and artifact.encoder_name != expected_encoder:
        raise ValueError("%s requires frozen %s features" % (experiment_role, expected_encoder))
    expected_crop = {
        "context_sensitivity": "full_context",
        "confirmation_xenium": "nucleus_centered",
        "regional_hescape_hoptimus1": "full_context",
        "regional_hescape_uni2h": "full_context",
    }.get(experiment_role, "small_cell_centered")
    if artifact.crop_scale != expected_crop:
        raise ValueError("%s requires the %s crop" % (experiment_role, expected_crop))
    if experiment_role in {"primary_hoptimus1", "replication_h0mini"} and (
        artifact.observation_level not in {"cell", "nucleus"}
        or artifact.target_construction != "registered_cell_expression"
    ):
        raise ValueError("the decisive morphology gate requires registered cell-level targets")
    if experiment_role in {"regional_hescape_hoptimus1", "regional_hescape_uni2h"} and (
        artifact.cohort_id != "HESCAPE"
        or artifact.cohort_release != "human-lung-healthy-panel"
        or artifact.observation_level != "pseudo_spot_55um"
        or artifact.target_construction != "sum_pooled_xenium_transcripts"
    ):
        raise ValueError("the HESCAPE role is restricted to the regional pseudo-spot control")
    if experiment_role == "regional_hescape_uni2h":
        if artifact.technical_covariate_names != REGIONAL_UNI2H_TECHNICAL_COVARIATES:
            raise ValueError("the UNI2-h regional role requires the frozen log-library covariate")
        if artifact.composition_feature_names != REGIONAL_UNI2H_COMPOSITION_FEATURES:
            raise ValueError(
                "the UNI2-h regional role requires four frozen RNA-only composition scores"
            )
        if artifact.composition_features.shape != (
            len(artifact.observation_ids),
            len(REGIONAL_UNI2H_COMPOSITION_FEATURES),
        ) or any(
            len(np.unique(artifact.composition_features[:, index])) < 3
            for index in range(len(REGIONAL_UNI2H_COMPOSITION_FEATURES))
        ):
            raise ValueError("the UNI2-h composition controls must be continuous scores")
        if not artifact.stain_feature_names:
            raise ValueError("the UNI2-h regional role requires stain-statistics controls")
    if experiment_role == "confirmation_xenium" and (
        artifact.cohort_id != "HEST" or artifact.assay != "Xenium"
    ):
        raise ValueError("confirmation requires a non-overlapping HEST Xenium artifact")


def donor_type_roi_permutation(
    donor_ids: np.ndarray,
    type_labels: np.ndarray,
    roi_ids: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Derange rows only within donor/type/ROI strata."""

    donors = np.asarray(donor_ids).astype(str)
    labels = np.asarray(type_labels)
    rois = np.asarray(roi_ids).astype(str)
    if donors.ndim != 1 or labels.shape != donors.shape or rois.shape != donors.shape:
        raise ValueError("permutation identities must be aligned vectors")
    rng = np.random.default_rng(seed)
    result = np.arange(len(donors), dtype=np.int64)
    keys = np.column_stack((donors, labels.astype(str), rois))
    for key in sorted(set(map(tuple, keys.tolist()))):
        group = np.flatnonzero(np.all(keys == np.asarray(key), axis=1))
        if len(group) < 2:
            continue
        ordered = group[rng.permutation(len(group))]
        result[ordered] = np.roll(ordered, 1)
    if not (
        np.array_equal(donors, donors[result])
        and np.array_equal(labels, labels[result])
        and np.array_equal(rois, rois[result])
    ):
        raise RuntimeError("preserving permutation crossed a frozen stratum")
    return result


def _stable_basis(values: np.ndarray, rank: int) -> np.ndarray:
    _, _, right = np.linalg.svd(np.asarray(values, dtype=np.float64), full_matrices=False)
    available = min(rank, len(right))
    basis = np.zeros((values.shape[1], rank), dtype=np.float64)
    for component in range(available):
        vector = right[component].copy()
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            vector *= -1.0
        basis[:, component] = vector
    return basis


def _standardization(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    return mean, np.maximum(scale, 1.0e-8)


def _donor_type_weights(donors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=np.float64)
    unique_donors = sorted(set(donors.tolist()))
    for donor in unique_donors:
        donor_mask = donors == donor
        occupied = sorted(set(labels[donor_mask].tolist()))
        for type_index in occupied:
            selected = donor_mask & (labels == type_index)
            weights[selected] = 1.0 / (len(unique_donors) * len(occupied) * int(selected.sum()))
    return weights / weights.mean(dtype=np.float64)


def _ridge(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    if alpha <= 0:
        raise ValueError("ridge alpha must be positive")
    weight_sum = float(weights.sum())
    feature_mean = np.sum(features * weights[:, None], axis=0) / weight_sum
    target_mean = np.sum(targets * weights[:, None], axis=0) / weight_sum
    root = np.sqrt(weights)[:, None]
    x = torch.as_tensor((features - feature_mean) * root, dtype=torch.float64, device=device)
    y = torch.as_tensor((targets - target_mean) * root, dtype=torch.float64, device=device)
    if x.shape[1] <= x.shape[0]:
        gram = x.T @ x
        gram.diagonal().add_(alpha)
        coefficients = torch.linalg.solve(gram, x.T @ y)
    else:
        gram = x @ x.T
        gram.diagonal().add_(alpha)
        coefficients = x.T @ torch.linalg.solve(gram, y)
    coefficient_array = coefficients.cpu().numpy()
    intercept = target_mean - feature_mean @ coefficient_array
    return coefficient_array, intercept


def _fit_technical_effects(
    covariates: np.ndarray, residual: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if covariates.shape[1] == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, residual.shape[1]), dtype=np.float64)
    mean = covariates.mean(axis=0, dtype=np.float64)
    centered = covariates - mean
    coefficients = np.linalg.pinv(centered, rcond=1.0e-10) @ residual
    return mean, coefficients


@dataclass(frozen=True)
class OracleRidgeFit:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    technical_mean: np.ndarray
    technical_coefficients: np.ndarray
    bases: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    rank: int
    alpha: float


def _correct_residuals(
    targets: np.ndarray,
    reference_means: np.ndarray,
    covariates: np.ndarray,
    technical_mean: np.ndarray,
    technical_coefficients: np.ndarray,
) -> np.ndarray:
    residual = targets - reference_means
    if covariates.shape[1]:
        residual = residual - (covariates - technical_mean) @ technical_coefficients
    return residual


def fit_oracle_ridge_probe(
    features: np.ndarray,
    targets: np.ndarray,
    reference_means: np.ndarray,
    labels: np.ndarray,
    donors: np.ndarray,
    technical_covariates: np.ndarray,
    *,
    num_types: int,
    rank: int,
    alpha: float,
    device: str = "auto",
) -> OracleRidgeFit:
    """Fit training-only technical correction, RNA bases, and per-type ridge heads."""

    if rank <= 0 or rank > targets.shape[1]:
        raise ValueError("ridge rank must be within the molecular target width")
    if len(set(donors.tolist())) < 2:
        raise ValueError("oracle ridge fitting requires at least two development donors")
    feature_mean, feature_scale = _standardization(features)
    normalized = (features - feature_mean) / feature_scale
    raw_residual = targets - reference_means
    technical_mean, technical_coefficients = _fit_technical_effects(
        technical_covariates, raw_residual
    )
    residual = _correct_residuals(
        targets,
        reference_means,
        technical_covariates,
        technical_mean,
        technical_coefficients,
    )
    bases = np.zeros((num_types, targets.shape[1], rank), dtype=np.float64)
    coefficients = np.zeros((num_types, features.shape[1], rank), dtype=np.float64)
    intercepts = np.zeros((num_types, rank), dtype=np.float64)
    weights = _donor_type_weights(donors, labels)
    target_device = resolve_device(device)
    for type_index in range(num_types):
        selected = labels == type_index
        if int(selected.sum()) < max(rank + 1, 3):
            raise ValueError("every type needs enough development cells for the selected rank")
        if len(set(donors[selected].tolist())) < 2:
            raise ValueError("every type needs at least two development donors")
        basis = _stable_basis(residual[selected], rank)
        coordinates = residual[selected] @ basis
        coefficient, intercept = _ridge(
            normalized[selected], coordinates, weights[selected], alpha, target_device
        )
        bases[type_index] = basis
        coefficients[type_index] = coefficient
        intercepts[type_index] = intercept
    return OracleRidgeFit(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        technical_mean=technical_mean,
        technical_coefficients=technical_coefficients,
        bases=bases,
        coefficients=coefficients,
        intercepts=intercepts,
        rank=rank,
        alpha=alpha,
    )


def predict_oracle_ridge(
    fit: OracleRidgeFit,
    features: np.ndarray,
    reference_means: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict residual coordinates and reconstruct molecular state using oracle types."""

    normalized = (features - fit.feature_mean) / fit.feature_scale
    coordinates = np.zeros((len(features), fit.rank), dtype=np.float64)
    prediction = np.asarray(reference_means, dtype=np.float64).copy()
    for type_index in sorted(set(labels.tolist())):
        selected = labels == type_index
        local = normalized[selected] @ fit.coefficients[type_index] + fit.intercepts[type_index]
        coordinates[selected] = local
        prediction[selected] += local @ fit.bases[type_index].T
    return coordinates, prediction


def _truth_coordinates(
    fit: OracleRidgeFit,
    targets: np.ndarray,
    reference_means: np.ndarray,
    covariates: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    residual = _correct_residuals(
        targets,
        reference_means,
        covariates,
        fit.technical_mean,
        fit.technical_coefficients,
    )
    coordinates = np.zeros((len(targets), fit.rank), dtype=np.float64)
    projected = reference_means.copy()
    for type_index in sorted(set(labels.tolist())):
        selected = labels == type_index
        local = residual[selected] @ fit.bases[type_index]
        coordinates[selected] = local
        projected[selected] += local @ fit.bases[type_index].T
    return coordinates, projected


def _macro_r2(
    truth: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    labels: np.ndarray,
    minimum_support: int,
) -> Tuple[float, Sequence[Mapping[str, object]], Mapping[str, float]]:
    rows = []
    donor_values: Dict[str, list[float]] = {}
    for donor in sorted(set(donors.tolist())):
        for type_index in sorted(set(labels[donors == donor].tolist())):
            selected = (donors == donor) & (labels == type_index)
            support = int(selected.sum())
            if support < minimum_support:
                continue
            centered = truth[selected] - truth[selected].mean(axis=0, keepdims=True)
            denominator = float(np.square(centered).sum())
            error = float(np.square(prediction[selected] - truth[selected]).sum())
            value = float(1.0 - error / denominator) if denominator > 1.0e-12 else float("nan")
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "donor_id": donor,
                    "type_index": int(type_index),
                    "support": support,
                    "residual_coordinate_r2": value,
                }
            )
            donor_values.setdefault(donor, []).append(value)
    donor_macro = {
        donor: float(np.mean(values)) for donor, values in donor_values.items() if values
    }
    if not donor_macro:
        raise ValueError("no supported locked donor/type stratum is evaluable")
    return float(np.mean(list(donor_macro.values()))), rows, donor_macro


def _fit_and_score(
    development: MorphologyRidgeDatasetArtifact,
    evaluation: MorphologyRidgeDatasetArtifact,
    features: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    rank: int,
    alpha: float,
    minimum_support: int,
    device: str,
    include_composition: bool = False,
) -> Tuple[
    float,
    OracleRidgeFit,
    np.ndarray,
    np.ndarray,
    Sequence[Mapping[str, object]],
    Mapping[str, float],
]:
    development_covariates = _endpoint_covariates(development, include_composition)
    evaluation_covariates = _endpoint_covariates(evaluation, include_composition)
    fit = fit_oracle_ridge_probe(
        features,
        development.molecular_targets,
        development.reference_means,
        development.type_labels,
        development.donor_ids,
        development_covariates,
        num_types=len(development.type_names),
        rank=rank,
        alpha=alpha,
        device=device,
    )
    predicted_coordinates, prediction = predict_oracle_ridge(
        fit, evaluation_features, evaluation.reference_means, evaluation.type_labels
    )
    truth_coordinates, _ = _truth_coordinates(
        fit,
        evaluation.molecular_targets,
        evaluation.reference_means,
        evaluation_covariates,
        evaluation.type_labels,
    )
    macro, rows, donors = _macro_r2(
        truth_coordinates,
        predicted_coordinates,
        evaluation.donor_ids,
        evaluation.type_labels,
        minimum_support,
    )
    return macro, fit, prediction, truth_coordinates, rows, donors


def _endpoint_covariates(
    artifact: MorphologyRidgeDatasetArtifact, include_composition: bool
) -> np.ndarray:
    """Return depth-only or depth-plus-composition covariates without fitting on test rows."""

    if not include_composition:
        return artifact.technical_covariates
    return np.concatenate(
        (artifact.technical_covariates, artifact.composition_features), axis=1
    )


def _select_hyperparameters(
    development: MorphologyRidgeDatasetArtifact,
    *,
    ranks: Sequence[int],
    alphas: Sequence[float],
    minimum_support: int,
    device: str,
    include_composition: bool = False,
) -> Tuple[int, float, Sequence[Mapping[str, object]]]:
    donors = sorted(set(development.donor_ids.tolist()))
    if len(donors) < 3:
        raise ValueError("nested donor validation requires at least three development donors")
    results = []
    for rank in sorted(set(int(value) for value in ranks)):
        for alpha in sorted(set(float(value) for value in alphas)):
            fold_values = []
            for heldout_donor in donors:
                train_mask = development.donor_ids != heldout_donor
                validation_mask = ~train_mask
                train = _subset(development, train_mask, role="development")
                validation = _subset(development, validation_mask, role="locked_test")
                try:
                    value, *_ = _fit_and_score(
                        train,
                        validation,
                        train.frozen_features,
                        validation.frozen_features,
                        rank=rank,
                        alpha=alpha,
                        minimum_support=minimum_support,
                        device=device,
                        include_composition=include_composition,
                    )
                except ValueError:
                    continue
                fold_values.append(value)
            results.append(
                {
                    "rank": rank,
                    "alpha": alpha,
                    "donor_folds": len(fold_values),
                    "macro_r2": float(np.mean(fold_values)) if fold_values else float("-inf"),
                }
            )
    eligible = [row for row in results if row["donor_folds"] == len(donors)]
    if not eligible:
        raise ValueError("no rank/alpha candidate supports every development donor fold")
    selected = sorted(eligible, key=lambda row: (-row["macro_r2"], row["rank"], row["alpha"]))[0]
    return int(selected["rank"]), float(selected["alpha"]), results


def _subset(
    artifact: MorphologyRidgeDatasetArtifact, selected: np.ndarray, *, role: str
) -> MorphologyRidgeDatasetArtifact:
    values = {
        name: getattr(artifact, name)
        for name in artifact.__dataclass_fields__
        if name not in {"role"}
    }
    for name in (
        "observation_ids",
        "donor_ids",
        "block_ids",
        "roi_ids",
        "type_labels",
        "frozen_features",
        "molecular_targets",
        "reference_means",
        "coordinate_features",
        "stain_features",
        "composition_features",
        "technical_covariates",
    ):
        values[name] = values[name][selected]
    values["role"] = role
    return MorphologyRidgeDatasetArtifact(**values)


def evaluate_morphology_ridge_gate(
    development: MorphologyRidgeDatasetArtifact,
    locked_test: MorphologyRidgeDatasetArtifact,
    *,
    ranks: Sequence[int] = (2, 4, 6),
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    permutation_seeds: Sequence[int] = (17, 29, 41),
    permutations_per_seed: int = 100,
    minimum_support: int = 10,
    minimum_development_donors: int = 5,
    minimum_locked_donors: Optional[int] = None,
    minimum_macro_r2: float = 0.05,
    minimum_shuffle_delta: float = 0.03,
    minimum_coordinate_delta: float = 0.01,
    minimum_stain_delta: float = 0.01,
    maximum_permutation_p: float = 0.01,
    minimum_positive_strata_fraction: float = 0.70,
    minimum_expression_error_reduction: float = 0.05,
    minimum_basis_ceiling_r2: float = 0.10,
    device: str = "auto",
) -> Mapping[str, object]:
    """Run the decisive locked-donor ridge probe and every mandatory simple control."""

    development.validate_compatible(locked_test)
    regional = development.observation_level == "pseudo_spot_55um"
    regional_uni2h = regional and development.encoder_name == "MahmoodLab/UNI2-h"
    required_locked_donors = (
        minimum_locked_donors
        if minimum_locked_donors is not None
        else (4 if regional_uni2h else 5)
    )
    development_donors = sorted(set(development.donor_ids.tolist()))
    locked_donors = sorted(set(locked_test.donor_ids.tolist()))
    if len(development_donors) < minimum_development_donors:
        raise ValueError("too few development donors for a morphology decision")
    if len(locked_donors) < required_locked_donors:
        raise ValueError("too few locked donors for the prespecified test")
    if permutations_per_seed < 100 or len(set(permutation_seeds)) < 3:
        raise ValueError("ridge gate requires at least 100 permutations for each of three seeds")
    if (
        not ranks
        or not alphas
        or any(value <= 0 for value in (*ranks, *alphas))
        or any(int(value) > development.molecular_targets.shape[1] for value in ranks)
    ):
        raise ValueError("ridge rank and alpha grids must be positive")

    include_composition = regional and bool(development.composition_feature_names)
    rank, alpha, selection = _select_hyperparameters(
        development,
        ranks=ranks,
        alphas=alphas,
        minimum_support=minimum_support,
        device=device,
        include_composition=include_composition,
    )
    if include_composition:
        raw_matched, *_ = _fit_and_score(
            development,
            locked_test,
            development.frozen_features,
            locked_test.frozen_features,
            rank=rank,
            alpha=alpha,
            minimum_support=minimum_support,
            device=device,
            include_composition=False,
        )
    matched, fit, prediction, truth_coordinates, rows, donor_macro = _fit_and_score(
        development,
        locked_test,
        development.frozen_features,
        locked_test.frozen_features,
        rank=rank,
        alpha=alpha,
        minimum_support=minimum_support,
        device=device,
        include_composition=include_composition,
    )
    if not include_composition:
        raw_matched = matched
    coordinate_macro, *_ = _fit_and_score(
        development,
        locked_test,
        development.coordinate_features,
        locked_test.coordinate_features,
        rank=rank,
        alpha=alpha,
        minimum_support=minimum_support,
        device=device,
        include_composition=include_composition,
    )
    stain_macro: Optional[float] = None
    if development.stain_features.shape[1]:
        stain_macro, *_ = _fit_and_score(
            development,
            locked_test,
            development.stain_features,
            locked_test.stain_features,
            rank=rank,
            alpha=alpha,
            minimum_support=minimum_support,
            device=device,
            include_composition=include_composition,
        )

    _, basis_prediction = _truth_coordinates(
        fit,
        locked_test.molecular_targets,
        locked_test.reference_means,
        _endpoint_covariates(locked_test, include_composition),
        locked_test.type_labels,
    )
    corrected_truth = locked_test.reference_means.copy()
    corrected_truth += _correct_residuals(
        locked_test.molecular_targets,
        locked_test.reference_means,
        _endpoint_covariates(locked_test, include_composition),
        fit.technical_mean,
        fit.technical_coefficients,
    )
    denominator = float(np.square(corrected_truth - locked_test.reference_means).sum())
    ceiling_r2 = float(
        1.0 - np.square(corrected_truth - basis_prediction).sum() / max(denominator, 1.0e-12)
    )
    baseline_rmse = float(
        np.sqrt(np.mean(np.square(corrected_truth - locked_test.reference_means)))
    )
    image_rmse = float(np.sqrt(np.mean(np.square(corrected_truth - prediction))))
    expression_reduction = (baseline_rmse - image_rmse) / max(baseline_rmse, 1.0e-12)

    shuffled_by_seed = []
    all_shuffled = []
    for seed in sorted(set(int(value) for value in permutation_seeds)):
        values = []
        fractions = []
        for permutation_index in range(permutations_per_seed):
            permutation = donor_type_roi_permutation(
                development.donor_ids,
                development.type_labels,
                development.roi_ids,
                seed=seed + permutation_index * 104729,
            )
            fractions.append(float(np.mean(permutation != np.arange(len(permutation)))))
            shuffled, *_ = _fit_and_score(
                development,
                locked_test,
                development.frozen_features[permutation],
                locked_test.frozen_features,
                rank=rank,
                alpha=alpha,
                minimum_support=minimum_support,
                device=device,
                include_composition=include_composition,
            )
            values.append(shuffled)
        array = np.asarray(values, dtype=np.float64)
        empirical_p = float((1 + np.sum(array >= matched)) / (len(array) + 1))
        shuffled_by_seed.append(
            {
                "seed": seed,
                "permutations": len(values),
                "mean_macro_r2": float(array.mean()),
                "matched_minus_shuffle_macro_r2": float(matched - array.mean()),
                "empirical_p": empirical_p,
                "minimum_shuffled_fraction": float(min(fractions)),
                "pass": bool(
                    matched - array.mean() >= minimum_shuffle_delta
                    and empirical_p < maximum_permutation_p
                ),
            }
        )
        all_shuffled.extend(values)

    positive_rows = [row for row in rows if row["residual_coordinate_r2"] > 0]
    positive_fraction = len(positive_rows) / len(rows)
    positive_donors = {name: max(value, 0.0) for name, value in donor_macro.items()}
    total_positive = sum(positive_donors.values())
    largest_donor_share = (
        max(positive_donors.values()) / total_positive if total_positive > 0 else 1.0
    )
    allowed_nonpositive_donors = (
        1 if regional_uni2h and len(donor_macro) >= 4 else (1 if len(donor_macro) >= 10 else 0)
    )
    donor_consistency = (
        sum(value <= 0 for value in donor_macro.values()) <= allowed_nonpositive_donors
    )
    rank_sensitivity = []
    for candidate in sorted(set(int(value) for value in ranks)):
        candidate_macro, *_ = _fit_and_score(
            development,
            locked_test,
            development.frozen_features,
            locked_test.frozen_features,
            rank=candidate,
            alpha=alpha,
            minimum_support=minimum_support,
            device=device,
            include_composition=include_composition,
        )
        rank_sensitivity.append({"rank": candidate, "macro_r2": candidate_macro})
    checks = {
        "matched_macro_r2": matched >= minimum_macro_r2,
        "every_seed_separates_shuffle": all(row["pass"] for row in shuffled_by_seed),
        "permutations_change_training_rows": all(
            row["minimum_shuffled_fraction"] > 0.0 for row in shuffled_by_seed
        ),
        "positive_supported_strata": positive_fraction >= minimum_positive_strata_fraction,
        "donor_consistency": donor_consistency,
        "not_single_donor_driven": largest_donor_share <= 0.5,
        "beats_coordinate_only": matched - coordinate_macro >= minimum_coordinate_delta,
        "expression_relevance": expression_reduction >= minimum_expression_error_reduction,
        "adequate_basis_ceiling": ceiling_r2 >= minimum_basis_ceiling_r2,
        "rank_direction_stable": all(row["macro_r2"] > 0 for row in rank_sensitivity),
    }
    if regional_uni2h:
        checks.update(
            {
                "composition_adjusted_positive": bool(include_composition and matched > 0.0),
                "beats_stain_statistics_only": bool(
                    stain_macro is not None and matched - stain_macro >= minimum_stain_delta
                ),
            }
        )
    component_pass = all(checks.values())
    nucleus_level = development.observation_level in {"cell", "nucleus"}
    regional_endpoints = None
    if regional:
        regional_endpoints = {
            "raw_depth_adjusted": {
                "donor_equal_niche_equal_residual_coordinate_r2": raw_matched,
                "development_fitted_covariates": list(development.technical_covariate_names),
            },
            "composition_adjusted": (
                {
                    "donor_equal_niche_equal_residual_coordinate_r2": matched,
                    "development_fitted_covariates": list(
                        development.technical_covariate_names
                        + development.composition_feature_names
                    ),
                    "coordinate_only_macro_r2": coordinate_macro,
                    "stain_statistics_only_macro_r2": stain_macro,
                    "uni2h_minus_coordinate_macro_r2": matched - coordinate_macro,
                    "uni2h_minus_stain_macro_r2": (
                        matched - stain_macro if stain_macro is not None else None
                    ),
                }
                if include_composition
                else None
            ),
            "correction_coefficients_fit_on_development_only": True,
            "composition_scores_are_continuous_rna_only_controls": bool(
                development.composition_feature_names
            ),
            "composition_score_genes_excluded_from_scored_targets": bool(
                development.composition_feature_names
                and not (set(development.gene_ids) & set(development.type_marker_gene_ids))
            ),
        }
    return {
        "schema_version": MORPHOLOGY_RIDGE_REPORT_SCHEMA,
        "status": "component_pass" if component_pass else "stop_or_pivot",
        "component_pass": component_pass,
        "authorizes_full_heir": False,
        "nucleus_hypothesis_tested": nucleus_level,
        "regional_hypothesis_tested": regional,
        "scientific_scope": (
            "registered_cell_state" if nucleus_level else "regional_pseudospot_exploratory"
        ),
        "reason_full_heir_remains_blocked": (
            "HESCAPE pseudo-spots test regional image-to-expression signal, not one nucleus "
            "paired to that nucleus's RNA"
            if regional
            else (
                "requires small-crop H0-mini replication, non-overlapping cohort confirmation, "
                "and a separate matched-reference specificity gate"
            )
        ),
        "oracle_type_only": True,
        "oracle_label_scope": (
            "rna_only_dominant_regional_niche" if regional else "registered_cell_type"
        ),
        "selected_hyperparameters": {"rank": rank, "alpha": alpha},
        "development_selection": selection,
        "primary_metrics": {
            "donor_equal_type_equal_residual_coordinate_r2": matched,
            "raw_depth_adjusted_regional_macro_r2": raw_matched if regional else None,
            "composition_adjusted_regional_macro_r2": (
                matched if include_composition else None
            ),
            "coordinate_only_macro_r2": coordinate_macro,
            "stain_statistics_only_macro_r2": stain_macro,
            "matched_minus_coordinate_macro_r2": matched - coordinate_macro,
            "basis_ceiling_r2": ceiling_r2,
            "expression_error_reduction_vs_reference_mean": expression_reduction,
            "positive_donor_type_fraction": positive_fraction,
            "largest_donor_positive_improvement_share": largest_donor_share,
        },
        "regional_endpoints": regional_endpoints,
        "donor_type_rows": rows,
        "donor_macro_r2": donor_macro,
        "permutation_control": {
            "training_probe_refit_for_each_permutation": True,
            "preserves_donor_type_roi": True,
            "total_permutations": len(all_shuffled),
            "seeds": shuffled_by_seed,
            "mean_macro_r2": float(np.mean(all_shuffled)),
        },
        "rank_sensitivity": rank_sensitivity,
        "thresholds": {
            "minimum_macro_r2": minimum_macro_r2,
            "minimum_shuffle_delta": minimum_shuffle_delta,
            "minimum_coordinate_delta": minimum_coordinate_delta,
            "minimum_stain_delta": minimum_stain_delta,
            "maximum_permutation_p": maximum_permutation_p,
            "minimum_positive_strata_fraction": minimum_positive_strata_fraction,
            "minimum_expression_error_reduction": minimum_expression_error_reduction,
            "minimum_basis_ceiling_r2": minimum_basis_ceiling_r2,
            "minimum_support": minimum_support,
            "minimum_locked_donors": required_locked_donors,
        },
        "checks": checks,
        "execution": {
            "device": str(resolve_device(device)),
            "development_donors": development_donors,
            "locked_test_donors": locked_donors,
            "excluded_components": [
                "type_classifier",
                "neural_residual_head",
                "uot",
                "graph",
                "unknown_mass",
                "refinement",
            ],
        },
    }


__all__ = [
    "MORPHOLOGY_RIDGE_REPORT_SCHEMA",
    "OracleRidgeFit",
    "donor_type_roi_permutation",
    "evaluate_morphology_ridge_gate",
    "fit_oracle_ridge_probe",
    "predict_oracle_ridge",
    "validate_experiment_identity",
]
