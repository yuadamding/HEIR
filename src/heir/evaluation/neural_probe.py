"""Neural analogue of HEIR's donor-balanced oracle ridge residual probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .hest_nested_ridge import donor_type_row_weights, fit_weighted_standardizer
from .neural_checkpoint import (
    canonical_model_state_sha256,
    load_neural_probe_bundle,
    save_neural_probe_bundle,
)
from .neural_training import (
    NeuralArchitecture,
    NeuralTrainingResult,
    ValidationRows,
    predict_neural_model,
    prepare_neural_execution,
    train_neural_model,
)
from .ridge_probe import MolecularTargetFit, fit_molecular_target, target_coordinates


@dataclass(frozen=True)
class NeuralProbeValidation:
    features: np.ndarray
    molecular_targets: np.ndarray
    reference_means: np.ndarray
    type_labels: np.ndarray
    donor_ids: np.ndarray
    section_ids: np.ndarray
    observation_ids: np.ndarray
    technical_covariates: np.ndarray
    minimum_support: int = 2


@dataclass(frozen=True)
class NeuralResidualFit:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coordinate_mean: np.ndarray
    coordinate_scale: np.ndarray
    target: MolecularTargetFit
    architecture: NeuralArchitecture
    training: NeuralTrainingResult
    weight_decay: float
    learning_rate: float

    @property
    def rank(self) -> int:
        return self.target.rank

    @property
    def checkpoint_sha256(self) -> str:
        return self.training.checkpoint_sha256


def _feature_standardization(
    features: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim not in {2, 3} or not len(values):
        raise ValueError("neural probe features must be a non-empty matrix or view tensor")
    if weights.shape != (len(values),):
        raise ValueError("feature-standardization weights are not row aligned")
    normalized = weights / weights.sum(dtype=np.float64)
    broadcast = normalized.reshape((len(values),) + (1,) * (values.ndim - 1))
    mean = np.sum(values * broadcast, axis=0)
    variance = np.sum(np.square(values - mean) * broadcast, axis=0)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1.0e-8] = 1.0
    return mean, scale


def _normalize_features(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.shape[1:] != mean.shape or mean.shape != scale.shape:
        raise ValueError("feature tensor does not match the fitted neural standardizer")
    normalized = (values - mean) / scale
    if not np.all(np.isfinite(normalized)):
        raise ValueError("standardized neural features are non-finite")
    return normalized.astype(np.float32)


def fit_neural_residual_probe(
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
    rank: int,
    model_id: str,
    type_conditioned: bool,
    weight_decay: float,
    epochs: int,
    seed: int,
    learning_rate: float = 1.0e-3,
    batch_size: int = 256,
    patience: int = 10,
    gradient_clip: float = 1.0,
    device: str = "auto",
    target_fit: Optional[MolecularTargetFit] = None,
    validation: Optional[NeuralProbeValidation] = None,
    view_dims: Optional[tuple[int, ...]] = None,
) -> NeuralResidualFit:
    """Fit a shared residual translator using development rows only."""

    values = np.asarray(features)
    labels_array = np.asarray(labels, dtype=np.int64)
    donor_array = np.asarray(donors).astype(str)
    section_array = np.asarray(sections).astype(str)
    ids = np.asarray(observation_ids).astype(str)
    rows = len(values)
    if any(array.shape != (rows,) for array in (labels_array, donor_array, section_array, ids)):
        raise ValueError("neural probe row identities are not aligned")
    if len(set(donor_array.tolist())) < 2:
        raise ValueError("neural residual fitting requires at least two development donors")
    if len(set(ids.tolist())) != rows or any(not value.strip() for value in ids.tolist()):
        raise ValueError("neural probe observation IDs must be non-empty and unique")
    # Determinism and CUDA workspace validity must be established before the
    # molecular target basis can invoke any torch linear algebra.
    prepare_neural_execution(seed, device)
    # Canonicalize before *every* floating reduction, including technical
    # correction, target-basis fitting, and standardization.  Sorting only in
    # the mini-batch trainer would leave those receipts row-order dependent.
    order = np.argsort(ids, kind="stable")
    values = values[order]
    labels_array = labels_array[order]
    donor_array = donor_array[order]
    section_array = section_array[order]
    ids = ids[order]
    targets = np.asarray(targets)[order]
    reference_means = np.asarray(reference_means)[order]
    technical_covariates = np.asarray(technical_covariates)[order]
    fitted_target = target_fit or fit_molecular_target(
        np.asarray(targets, dtype=np.float64),
        np.asarray(reference_means, dtype=np.float64),
        labels_array,
        donor_array,
        np.asarray(technical_covariates, dtype=np.float64),
        num_types=num_types,
        rank=rank,
        device=device,
    )
    if fitted_target.rank != rank:
        raise ValueError("shared molecular target rank differs from neural probe rank")
    coordinates, _ = target_coordinates(
        fitted_target,
        np.asarray(targets, dtype=np.float64),
        np.asarray(reference_means, dtype=np.float64),
        np.asarray(technical_covariates, dtype=np.float64),
        labels_array,
    )
    weights = donor_type_row_weights(donor_array, labels_array.astype(str))
    feature_mean, feature_scale = _feature_standardization(values, weights)
    normalized_features = _normalize_features(values, feature_mean, feature_scale)
    coordinate_standardizer = fit_weighted_standardizer(coordinates, weights)
    normalized_coordinates = coordinate_standardizer.transform(coordinates).astype(np.float32)
    registered_view_dims: tuple[int, ...] = ()
    if model_id == "late_fusion":
        if values.ndim == 3:
            registered_view_dims = tuple([int(values.shape[2])] * int(values.shape[1]))
            normalized_features = normalized_features.reshape(len(values), -1)
        else:
            registered_view_dims = tuple(view_dims or ())
            if not registered_view_dims or sum(registered_view_dims) != values.shape[1]:
                raise ValueError("flat late-fusion features require registered per-view widths")
        view_count = len(registered_view_dims)
        input_dim = int(sum(registered_view_dims))
    else:
        if values.ndim != 2 or view_dims:
            raise ValueError("single-view neural models require one feature matrix")
        view_count = 1
        input_dim = int(values.shape[1])
    architecture = NeuralArchitecture(
        model_id=model_id,
        input_dim=int(input_dim),
        output_dim=rank,
        num_types=num_types,
        view_count=int(view_count),
        view_dims=registered_view_dims,
        type_conditioned=bool(type_conditioned),
    )
    validation_rows = None
    if validation is not None:
        validation_ids = np.asarray(validation.observation_ids).astype(str)
        validation_donors = np.asarray(validation.donor_ids).astype(str)
        if set(ids.tolist()) & set(validation_ids.tolist()):
            raise ValueError("training and validation observation IDs overlap")
        if set(donor_array.tolist()) & set(validation_donors.tolist()):
            raise ValueError("training and validation donors overlap")
        validation_coordinates, _ = target_coordinates(
            fitted_target,
            np.asarray(validation.molecular_targets, dtype=np.float64),
            np.asarray(validation.reference_means, dtype=np.float64),
            np.asarray(validation.technical_covariates, dtype=np.float64),
            np.asarray(validation.type_labels, dtype=np.int64),
        )
        validation_features = _normalize_features(validation.features, feature_mean, feature_scale)
        if model_id == "late_fusion" and validation_features.ndim == 3:
            validation_features = validation_features.reshape(len(validation_features), -1)
        validation_rows = ValidationRows(
            features=validation_features,
            targets=coordinate_standardizer.transform(validation_coordinates).astype(np.float32),
            type_labels=np.asarray(validation.type_labels, dtype=np.int64),
            donor_ids=np.asarray(validation.donor_ids).astype(str),
            section_ids=np.asarray(validation.section_ids).astype(str),
            observation_ids=np.asarray(validation.observation_ids).astype(str),
            coordinate_mean=coordinate_standardizer.mean.copy(),
            coordinate_scale=coordinate_standardizer.scale.copy(),
            minimum_support=int(validation.minimum_support),
        )
    training = train_neural_model(
        normalized_features,
        normalized_coordinates,
        labels_array,
        ids,
        weights,
        architecture=architecture,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        gradient_clip=gradient_clip,
        seed=seed,
        device=device,
        validation=validation_rows,
    )
    return NeuralResidualFit(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coordinate_mean=coordinate_standardizer.mean,
        coordinate_scale=coordinate_standardizer.scale,
        target=fitted_target,
        architecture=architecture,
        training=training,
        weight_decay=float(weight_decay),
        learning_rate=float(learning_rate),
    )


def predict_neural_residual_probe(
    fit: NeuralResidualFit,
    features: np.ndarray,
    reference_means: np.ndarray,
    labels: np.ndarray,
    *,
    device: str = "auto",
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict latent residual coordinates and reconstruct measured genes."""

    label_values = np.asarray(labels, dtype=np.int64)
    normalized = _normalize_features(features, fit.feature_mean, fit.feature_scale)
    if fit.architecture.model_id == "late_fusion" and normalized.ndim == 3:
        normalized = normalized.reshape(len(normalized), -1)
    standardized_coordinates = predict_neural_model(
        fit.training,
        normalized,
        label_values,
        device=device,
        batch_size=batch_size,
    )
    coordinates = standardized_coordinates * fit.coordinate_scale + fit.coordinate_mean
    prediction = np.asarray(reference_means, dtype=np.float64).copy()
    if prediction.ndim != 2 or len(prediction) != len(coordinates):
        raise ValueError("reference means are not aligned with neural prediction rows")
    for type_index in sorted(set(label_values.tolist())):
        selected = label_values == type_index
        prediction[selected] += fit.target.residual_means[type_index]
        prediction[selected] += coordinates[selected] @ fit.target.bases[type_index].T
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("neural residual reconstruction is non-finite")
    return coordinates, prediction


def save_neural_residual_fit(
    path: Union[str, Path], fit: NeuralResidualFit
) -> dict[str, object]:
    """Persist a complete probe, including target basis and standardizers."""

    arrays = {
        "feature_mean": fit.feature_mean,
        "feature_scale": fit.feature_scale,
        "coordinate_mean": fit.coordinate_mean,
        "coordinate_scale": fit.coordinate_scale,
        "technical_means": fit.target.technical_means,
        "technical_coefficients": fit.target.technical_coefficients,
        "residual_means": fit.target.residual_means,
        "bases": fit.target.bases,
    }
    architecture = fit.architecture
    metadata = {
        "schema": "heir.neural_residual_fit.v1",
        "architecture": {
            "model_id": architecture.model_id,
            "input_dim": architecture.input_dim,
            "output_dim": architecture.output_dim,
            "num_types": architecture.num_types,
            "view_count": architecture.view_count,
            "view_dims": list(architecture.view_dims),
            "type_conditioned": architecture.type_conditioned,
            "type_embedding_width": architecture.type_embedding_width,
            "adapter_rank": architecture.adapter_rank,
        },
        "target_rank": fit.target.rank,
        "weight_decay": fit.weight_decay,
        "learning_rate": fit.learning_rate,
        "training": {
            "best_epoch": fit.training.best_epoch,
            "epochs_run": fit.training.epochs_run,
            "history": list(fit.training.history),
            "checkpoint_sha256": fit.training.checkpoint_sha256,
            "parameter_count": fit.training.parameter_count,
            "fit_device": fit.training.fit_device,
            "seed": fit.training.seed,
        },
    }
    return dict(save_neural_probe_bundle(path, fit.training.state_dict, arrays, metadata))


def load_neural_residual_fit(path: Union[str, Path]) -> NeuralResidualFit:
    """Load and verify a complete neural residual probe bundle."""

    state, arrays, metadata = load_neural_probe_bundle(path)
    if metadata.get("schema") != "heir.neural_residual_fit.v1":
        raise ValueError("neural residual fit schema is unsupported")
    architecture_metadata = metadata.get("architecture")
    training_metadata = metadata.get("training")
    if not isinstance(architecture_metadata, dict) or not isinstance(training_metadata, dict):
        raise ValueError("neural residual fit metadata is incomplete")
    architecture = NeuralArchitecture(
        model_id=str(architecture_metadata["model_id"]),
        input_dim=int(architecture_metadata["input_dim"]),
        output_dim=int(architecture_metadata["output_dim"]),
        num_types=int(architecture_metadata["num_types"]),
        view_count=int(architecture_metadata["view_count"]),
        view_dims=tuple(int(value) for value in architecture_metadata["view_dims"]),
        type_conditioned=bool(architecture_metadata["type_conditioned"]),
        type_embedding_width=int(architecture_metadata["type_embedding_width"]),
        adapter_rank=int(architecture_metadata["adapter_rank"]),
    )
    target = MolecularTargetFit(
        technical_means=arrays["technical_means"],
        technical_coefficients=arrays["technical_coefficients"],
        residual_means=arrays["residual_means"],
        bases=arrays["bases"],
        rank=int(metadata["target_rank"]),
    )
    training = NeuralTrainingResult(
        architecture=architecture,
        state_dict=state,
        best_epoch=int(training_metadata["best_epoch"]),
        epochs_run=int(training_metadata["epochs_run"]),
        history=tuple(training_metadata["history"]),
        checkpoint_sha256=str(training_metadata["checkpoint_sha256"]),
        parameter_count=int(training_metadata["parameter_count"]),
        fit_device=str(training_metadata["fit_device"]),
        seed=int(training_metadata["seed"]),
    )
    if canonical_model_state_sha256(state) != training.checkpoint_sha256:
        raise ValueError("neural residual training hash is inconsistent")
    return NeuralResidualFit(
        feature_mean=arrays["feature_mean"],
        feature_scale=arrays["feature_scale"],
        coordinate_mean=arrays["coordinate_mean"],
        coordinate_scale=arrays["coordinate_scale"],
        target=target,
        architecture=architecture,
        training=training,
        weight_decay=float(metadata["weight_decay"]),
        learning_rate=float(metadata["learning_rate"]),
    )


__all__ = [
    "NeuralProbeValidation",
    "NeuralResidualFit",
    "fit_neural_residual_probe",
    "load_neural_residual_fit",
    "predict_neural_residual_probe",
    "save_neural_residual_fit",
]
