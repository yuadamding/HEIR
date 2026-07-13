"""Compact donor-held-out morphology-to-within-type RNA-state gate.

The module deliberately excludes graphs, transport, molecular abundance priors,
unknown-state heads, and refinement.  Training-only broad-type centroids and
low-rank residual bases define the molecular target; a linear type probe and a
small residual MLP operate on detached, frozen pathology features.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from heir.utils import resolve_device, set_seed, sha256_file

MORPHOLOGY_STATE_CHECKPOINT_SCHEMA = "heir.morphology_state_gate.v1"
MORPHOLOGY_STATE_REPORT_SCHEMA = "heir.morphology_state_evaluation.v1"

ArrayLike = Union[np.ndarray, Tensor, Sequence[int], Sequence[str]]


@dataclass(frozen=True)
class MorphologyStateGateConfig:
    """Architecture and immutable dimensions for :class:`MorphologyStateGate`."""

    feature_dim: int
    latent_dim: int
    num_types: int
    residual_rank: int = 4
    residual_hidden_dim: int = 64
    type_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("feature_dim", "latent_dim", "num_types", "residual_rank"):
            if int(getattr(self, name)) <= 0:
                raise ValueError("%s must be positive" % name)
        if self.residual_rank > self.latent_dim:
            raise ValueError("residual_rank cannot exceed latent_dim")
        if self.residual_hidden_dim <= 0:
            raise ValueError("residual_hidden_dim must be positive")
        names = tuple(str(value) for value in self.type_names)
        if not names:
            names = tuple(str(index) for index in range(self.num_types))
        if len(names) != self.num_types or any(not value for value in names):
            raise ValueError("type_names must contain one non-empty name per type")
        if len(set(names)) != len(names):
            raise ValueError("type_names must be unique")
        object.__setattr__(self, "type_names", names)

    def to_dict(self) -> Dict[str, object]:
        values = asdict(self)
        values["type_names"] = list(self.type_names)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "MorphologyStateGateConfig":
        data = dict(values)
        if "type_names" in data:
            data["type_names"] = tuple(data["type_names"])
        return cls(**data)


@dataclass(frozen=True)
class MorphologyStateOutput:
    """Predicted broad type and type-conditioned molecular latent."""

    type_logits: Tensor
    type_probabilities: Tensor
    predicted_type: Tensor
    selected_type: Tensor
    residual_coordinates: Tensor
    latent_residual: Tensor
    latent: Tensor


def _numpy_float_matrix(value: ArrayLike, name: str, width: int) -> np.ndarray:
    array = np.asarray(value.detach().cpu().numpy() if isinstance(value, Tensor) else value)
    if array.ndim != 2 or array.shape[1] != width or len(array) == 0:
        raise ValueError("%s must be a non-empty matrix with width %d" % (name, width))
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("%s must be finite" % name)
    return array


def _numpy_type_vector(value: ArrayLike, name: str, size: int, num_types: int) -> np.ndarray:
    array = np.asarray(value.detach().cpu().numpy() if isinstance(value, Tensor) else value)
    if array.ndim != 1 or len(array) != size:
        raise ValueError("%s must contain one value per observation" % name)
    if array.dtype.kind not in "iub":
        raise TypeError("%s must contain integer type indices" % name)
    array = array.astype(np.int64, copy=False)
    if np.any(array < 0) or np.any(array >= num_types):
        raise ValueError("%s contains an out-of-range type index" % name)
    return array


def _numpy_string_vector(value: ArrayLike, name: str, size: int) -> np.ndarray:
    array = np.asarray(value.detach().cpu().numpy() if isinstance(value, Tensor) else value)
    if array.ndim != 1 or len(array) != size:
        raise ValueError("%s must contain one value per observation" % name)
    result = array.astype(str)
    if any(not item for item in result.tolist()):
        raise ValueError("%s cannot contain empty values" % name)
    return result


def _stable_basis(residuals: np.ndarray, rank: int) -> np.ndarray:
    """Return a deterministic PCA loading matrix with stable component signs."""

    _, _, right = np.linalg.svd(np.asarray(residuals, dtype=np.float64), full_matrices=False)
    available = min(rank, right.shape[0])
    basis = np.zeros((residuals.shape[1], rank), dtype=np.float32)
    for component in range(available):
        vector = right[component].copy()
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            vector *= -1.0
        basis[:, component] = vector.astype(np.float32)
    return basis


class MorphologyStateGate(nn.Module):
    """Linear broad-type probe plus a type-conditioned low-rank residual MLP."""

    def __init__(self, config: MorphologyStateGateConfig) -> None:
        super().__init__()
        self.config = config
        self.type_classifier = nn.Linear(config.feature_dim, config.num_types)
        self.residual_predictor = nn.Sequential(
            nn.Linear(config.feature_dim, config.residual_hidden_dim),
            nn.GELU(),
            nn.Linear(
                config.residual_hidden_dim,
                config.num_types * config.residual_rank,
            ),
        )
        self.register_buffer("feature_mean", torch.zeros(config.feature_dim))
        self.register_buffer("feature_scale", torch.ones(config.feature_dim))
        self.register_buffer("type_centroids", torch.zeros(config.num_types, config.latent_dim))
        self.register_buffer(
            "type_bases",
            torch.zeros(config.num_types, config.latent_dim, config.residual_rank),
        )
        self.register_buffer("geometry_initialized", torch.tensor(False, dtype=torch.bool))
        self.training_donors: Tuple[str, ...] = ()
        self.training_sample_count = 0

    @classmethod
    def from_training_data(
        cls,
        config: MorphologyStateGateConfig,
        frozen_features: ArrayLike,
        latent_targets: ArrayLike,
        type_labels: ArrayLike,
        donor_ids: ArrayLike,
    ) -> "MorphologyStateGate":
        """Create a gate whose feature scaling and state geometry use training data only."""

        model = cls(config)
        model.initialize_training_geometry(
            frozen_features=frozen_features,
            latent_targets=latent_targets,
            type_labels=type_labels,
            donor_ids=donor_ids,
        )
        return model

    def initialize_training_geometry(
        self,
        *,
        frozen_features: ArrayLike,
        latent_targets: ArrayLike,
        type_labels: ArrayLike,
        donor_ids: ArrayLike,
    ) -> None:
        """Freeze training-only standardization, type centroids, and PCA bases once."""

        if bool(self.geometry_initialized):
            raise RuntimeError("training geometry is already initialized")
        features = _numpy_float_matrix(frozen_features, "frozen_features", self.config.feature_dim)
        targets = _numpy_float_matrix(latent_targets, "latent_targets", self.config.latent_dim)
        if len(targets) != len(features):
            raise ValueError("frozen_features and latent_targets must align")
        labels = _numpy_type_vector(
            type_labels, "type_labels", len(features), self.config.num_types
        )
        donors = _numpy_string_vector(donor_ids, "donor_ids", len(features))
        counts = np.bincount(labels, minlength=self.config.num_types)
        if np.any(counts < 2):
            raise ValueError("every type needs at least two training observations")

        feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        feature_scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
        feature_scale = np.maximum(feature_scale, np.float32(1.0e-6))
        centroids = np.zeros((self.config.num_types, self.config.latent_dim), dtype=np.float32)
        bases = np.zeros(
            (
                self.config.num_types,
                self.config.latent_dim,
                self.config.residual_rank,
            ),
            dtype=np.float32,
        )
        for type_index in range(self.config.num_types):
            selected = targets[labels == type_index]
            centroids[type_index] = selected.mean(axis=0, dtype=np.float64)
            bases[type_index] = _stable_basis(
                selected - centroids[type_index], self.config.residual_rank
            )

        self.feature_mean.copy_(torch.from_numpy(feature_mean))
        self.feature_scale.copy_(torch.from_numpy(feature_scale))
        self.type_centroids.copy_(torch.from_numpy(centroids))
        self.type_bases.copy_(torch.from_numpy(bases))
        self.geometry_initialized.fill_(True)
        self.training_donors = tuple(sorted(set(donors.tolist())))
        self.training_sample_count = int(len(features))

    def _check_ready(self) -> None:
        if not bool(self.geometry_initialized):
            raise RuntimeError("training geometry has not been initialized")

    def normalized_features(self, frozen_features: Tensor) -> Tensor:
        """Standardize detached features using training-only stored statistics."""

        self._check_ready()
        if not torch.is_floating_point(frozen_features):
            raise TypeError("frozen_features must be floating point")
        if frozen_features.ndim != 2 or frozen_features.shape[1] != self.config.feature_dim:
            raise ValueError("frozen_features has the wrong shape")
        if not bool(torch.isfinite(frozen_features).all()):
            raise ValueError("frozen_features must be finite")
        return (frozen_features.detach() - self.feature_mean) / self.feature_scale

    def type_mean_latent(self, type_indices: Tensor) -> Tensor:
        """Return the training-only latent centroid for each requested type."""

        self._check_ready()
        if type_indices.ndim != 1 or type_indices.dtype != torch.long:
            raise TypeError("type_indices must be a one-dimensional int64 tensor")
        if bool(((type_indices < 0) | (type_indices >= self.config.num_types)).any()):
            raise ValueError("type_indices contains an out-of-range type")
        return self.type_centroids[type_indices]

    def forward(
        self,
        frozen_features: Tensor,
        type_indices: Optional[Tensor] = None,
    ) -> MorphologyStateOutput:
        """Predict with either model-selected or caller-supplied oracle broad types."""

        features = self.normalized_features(frozen_features)
        logits = self.type_classifier(features)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_type = probabilities.argmax(dim=-1)
        selected_type = predicted_type if type_indices is None else type_indices
        if selected_type.ndim != 1 or len(selected_type) != len(features):
            raise ValueError("type_indices must contain one value per feature row")
        if selected_type.dtype != torch.long:
            raise TypeError("type_indices must use torch.int64")
        if bool(((selected_type < 0) | (selected_type >= self.config.num_types)).any()):
            raise ValueError("type_indices contains an out-of-range type")

        all_coordinates = self.residual_predictor(features).reshape(
            len(features), self.config.num_types, self.config.residual_rank
        )
        row = torch.arange(len(features), device=features.device)
        coordinates = all_coordinates[row, selected_type]
        basis = self.type_bases[selected_type]
        residual = torch.bmm(basis, coordinates.unsqueeze(-1)).squeeze(-1)
        latent = self.type_centroids[selected_type] + residual
        return MorphologyStateOutput(
            type_logits=logits,
            type_probabilities=probabilities,
            predicted_type=predicted_type,
            selected_type=selected_type,
            residual_coordinates=coordinates,
            latent_residual=residual,
            latent=latent,
        )

    def checkpoint(self) -> Dict[str, object]:
        """Return a portable, weights-only-safe checkpoint payload."""

        self._check_ready()
        state = {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}
        return {
            "schema_version": MORPHOLOGY_STATE_CHECKPOINT_SCHEMA,
            "config": self.config.to_dict(),
            "state_dict": state,
            "metadata": {
                "training_donors": list(self.training_donors),
                "training_sample_count": self.training_sample_count,
                "training_only_type_centroids": True,
                "training_only_type_residual_bases": True,
                "frozen_input_features": True,
                "graph_used": False,
                "uot_used": False,
                "refinement_used": False,
            },
        }

    def save_checkpoint(self, path: Union[str, os.PathLike[str]]) -> Path:
        """Atomically save a self-describing morphology-state checkpoint."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=destination.name + ".",
            suffix=".pt.tmp",
            dir=str(destination.parent),
        )
        os.close(descriptor)
        try:
            torch.save(self.checkpoint(), temporary)
            with Path(temporary).open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return destination

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Mapping[str, Any], *, strict: bool = True
    ) -> "MorphologyStateGate":
        """Reconstruct a model and its training-only geometry from a payload."""

        if checkpoint.get("schema_version") != MORPHOLOGY_STATE_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported morphology-state checkpoint schema")
        config = checkpoint.get("config")
        state = checkpoint.get("state_dict")
        metadata = checkpoint.get("metadata")
        if not isinstance(config, Mapping) or not isinstance(state, Mapping):
            raise ValueError("morphology-state checkpoint lacks config or state_dict")
        if not isinstance(metadata, Mapping):
            raise ValueError("morphology-state checkpoint lacks metadata")
        model = cls(MorphologyStateGateConfig.from_dict(config))
        model.load_state_dict(dict(state), strict=strict)
        if not bool(model.geometry_initialized):
            raise ValueError("morphology-state checkpoint geometry is not initialized")
        donors = metadata.get("training_donors")
        sample_count = metadata.get("training_sample_count")
        if not isinstance(donors, list) or not donors or any(not str(value) for value in donors):
            raise ValueError("morphology-state checkpoint training donors are missing")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
            raise ValueError("morphology-state checkpoint training sample count is invalid")
        required_flags = (
            "training_only_type_centroids",
            "training_only_type_residual_bases",
            "frozen_input_features",
        )
        if any(metadata.get(name) is not True for name in required_flags):
            raise ValueError("morphology-state checkpoint violates training-only geometry")
        if any(
            metadata.get(name) is not False
            for name in ("graph_used", "uot_used", "refinement_used")
        ):
            raise ValueError("morphology-state checkpoint contains an excluded method component")
        model.training_donors = tuple(sorted(set(str(value) for value in donors)))
        model.training_sample_count = sample_count
        return model

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, os.PathLike[str]],
        *,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = True,
    ) -> "MorphologyStateGate":
        """Safely load a morphology-state checkpoint from disk."""

        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("morphology-state checkpoint root must be an object")
        return cls.from_checkpoint(payload, strict=strict)


def fit_morphology_state_gate(
    model: MorphologyStateGate,
    frozen_features: ArrayLike,
    latent_targets: ArrayLike,
    type_labels: ArrayLike,
    *,
    epochs: int = 100,
    batch_size: int = 1024,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    type_loss_weight: float = 1.0,
    residual_loss_weight: float = 1.0,
    seed: int = 17,
    device: str = "auto",
) -> Dict[str, object]:
    """Fit only the compact type and low-rank residual heads on frozen features."""

    model._check_ready()
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if type_loss_weight < 0 or residual_loss_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if type_loss_weight == 0 and residual_loss_weight == 0:
        raise ValueError("at least one loss weight must be positive")

    features = _numpy_float_matrix(frozen_features, "frozen_features", model.config.feature_dim)
    targets = _numpy_float_matrix(latent_targets, "latent_targets", model.config.latent_dim)
    if len(features) != len(targets):
        raise ValueError("frozen_features and latent_targets must align")
    labels = _numpy_type_vector(type_labels, "type_labels", len(features), model.config.num_types)
    counts = np.bincount(labels, minlength=model.config.num_types)
    if np.any(counts == 0):
        raise ValueError("training labels must include every configured type")

    set_seed(seed)
    target_device = resolve_device(device)
    model.to(target_device)
    x = torch.from_numpy(features).to(target_device)
    z = torch.from_numpy(targets).to(target_device)
    y = torch.from_numpy(labels).to(target_device)
    centroids = model.type_centroids[y]
    bases = model.type_bases[y]
    target_coordinates = torch.bmm(bases.transpose(1, 2), (z - centroids).unsqueeze(-1)).squeeze(-1)
    class_weights = torch.as_tensor(
        len(labels) / (model.config.num_types * counts),
        dtype=torch.float32,
        device=target_device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(len(x), generator=generator)
        total = 0.0
        type_total = 0.0
        residual_total = 0.0
        observations = 0
        for start in range(0, len(x), batch_size):
            index = permutation[start : start + batch_size].to(target_device)
            output = model(x[index], y[index])
            type_loss = F.cross_entropy(output.type_logits, y[index], weight=class_weights)
            residual_loss = F.mse_loss(output.residual_coordinates, target_coordinates[index])
            loss = type_loss_weight * type_loss + residual_loss_weight * residual_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(len(index))
            observations += count
            total += float(loss.detach()) * count
            type_total += float(type_loss.detach()) * count
            residual_total += float(residual_loss.detach()) * count
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total / observations,
                "type_loss": type_total / observations,
                "residual_loss": residual_total / observations,
            }
        )
    model.eval()
    return {
        "schema_version": "heir.morphology_state_training.v1",
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(target_device),
        "frozen_input_features": True,
        "graph_used": False,
        "excluded_components": [
            "uot",
            "prototype_abundance",
            "unknown_head",
            "refinement",
            "weak_sample_losses",
        ],
        "first_epoch": history[0],
        "final_epoch": history[-1],
    }


def donor_type_preserving_permutation(
    donor_ids: ArrayLike,
    type_labels: ArrayLike,
    *,
    roi_ids: Optional[ArrayLike] = None,
    seed: int = 17,
) -> np.ndarray:
    """Derange observations within donor/type/(optional ROI) strata where possible."""

    donors_raw = np.asarray(
        donor_ids.detach().cpu().numpy() if isinstance(donor_ids, Tensor) else donor_ids
    )
    if donors_raw.ndim != 1 or len(donors_raw) == 0:
        raise ValueError("donor_ids must be a non-empty vector")
    donors = _numpy_string_vector(donors_raw, "donor_ids", len(donors_raw))
    raw_types = np.asarray(
        type_labels.detach().cpu().numpy() if isinstance(type_labels, Tensor) else type_labels
    )
    if raw_types.ndim != 1 or len(raw_types) != len(donors):
        raise ValueError("type_labels must contain one value per donor_id")
    types = raw_types.astype(str)
    rois = (
        np.repeat("__all_rois__", len(donors))
        if roi_ids is None
        else _numpy_string_vector(roi_ids, "roi_ids", len(donors))
    )
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(donors), dtype=np.int64)
    keys = np.column_stack((donors, types, rois))
    for key in sorted(set(map(tuple, keys.tolist()))):
        group = np.flatnonzero(np.all(keys == np.asarray(key), axis=1))
        if len(group) < 2:
            continue
        ordered = group[rng.permutation(len(group))]
        permutation[ordered] = np.roll(ordered, 1)
    if len(set(permutation.tolist())) != len(permutation):
        raise RuntimeError("preserving permutation is not one-to-one")
    if not (
        np.array_equal(donors, donors[permutation])
        and np.array_equal(types, types[permutation])
        and np.array_equal(rois, rois[permutation])
    ):
        raise RuntimeError("preserving permutation crossed a declared stratum")
    return permutation


def _mean_cosine(prediction: np.ndarray, truth: np.ndarray) -> Tuple[float, int]:
    numerator = np.sum(prediction * truth, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1)
    valid = denominator > 1.0e-12
    values = numerator[valid] / denominator[valid]
    return (float(np.mean(values)) if len(values) else float("nan"), int(valid.sum()))


def _latent_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    type_mean: np.ndarray,
) -> Dict[str, object]:
    residual_prediction = prediction - type_mean
    residual_truth = truth - type_mean
    denominator = float(np.square(residual_truth).sum())
    error = float(np.square(prediction - truth).sum())
    baseline_rmse = float(np.sqrt(np.mean(np.square(type_mean - truth))))
    rmse = float(np.sqrt(np.mean(np.square(prediction - truth))))
    cosine, valid = _mean_cosine(residual_prediction, residual_truth)
    within_type_r2 = float(1.0 - error / denominator) if denominator > 1.0e-12 else float("nan")
    return {
        "within_type_r2": within_type_r2,
        # Explicit scientific name retained beside the shorter compatibility
        # name: type identity is partialled out by the training-only centroid.
        "within_type_partial_r2": within_type_r2,
        "within_type_cosine": cosine,
        "within_type_cosine_valid_cells": valid,
        "rmse": rmse,
        "rmse_delta_vs_type_mean": baseline_rmse - rmse,
        "within_type_rmse_delta_vs_type_mean": baseline_rmse - rmse,
        "type_mean_rmse": baseline_rmse,
    }


def _decoded_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    type_mean_prediction: np.ndarray,
) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean(np.square(prediction - truth))))
    baseline_rmse = float(np.sqrt(np.mean(np.square(type_mean_prediction - truth))))
    cosine, _ = _mean_cosine(prediction, truth)
    return {
        "rmse": rmse,
        "cosine": cosine,
        "type_mean_rmse": baseline_rmse,
        "rmse_delta_vs_type_mean": baseline_rmse - rmse,
    }


def _retrieval_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    true_types: np.ndarray,
    selected_types: np.ndarray,
    centroids: np.ndarray,
) -> Dict[str, float]:
    reciprocal_ranks = []
    top1 = []
    candidate_counts = []
    uniform_top1 = []
    uniform_mrr = []
    for index in range(len(prediction)):
        selected = int(selected_types[index])
        candidates = np.flatnonzero(true_types == selected)
        candidate_counts.append(len(candidates))
        if len(candidates) == 0 or int(true_types[index]) != selected:
            reciprocal_ranks.append(0.0)
            top1.append(0.0)
            uniform_top1.append(0.0)
            uniform_mrr.append(0.0)
            continue
        query = prediction[index] - centroids[selected]
        bank = truth[candidates] - centroids[selected]
        distances = np.square(bank - query[None, :]).sum(axis=1)
        correct_location = int(np.flatnonzero(candidates == index)[0])
        correct = float(distances[correct_location])
        tolerance = max(1.0e-12, abs(correct) * 1.0e-10)
        less = int(np.count_nonzero(distances < correct - tolerance))
        tied = int(np.count_nonzero(np.abs(distances - correct) <= tolerance))
        average_rank = less + (tied + 1.0) / 2.0
        reciprocal_ranks.append(1.0 / average_rank)
        top1.append(float(less == 0 and tied == 1))
        count = len(candidates)
        uniform_top1.append(1.0 / count)
        uniform_mrr.append(sum(1.0 / rank for rank in range(1, count + 1)) / count)
    return {
        "top1": float(np.mean(top1)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "mean_candidates": float(np.mean(candidate_counts)),
        "uniform_expected_top1": float(np.mean(uniform_top1)),
        "uniform_expected_mrr": float(np.mean(uniform_mrr)),
    }


def _endpoint(
    latent: np.ndarray,
    *,
    truth_latent: np.ndarray,
    type_mean_latent: np.ndarray,
    true_types: np.ndarray,
    selected_types: np.ndarray,
    centroids: np.ndarray,
    decoded: np.ndarray,
    truth_expression: np.ndarray,
    decoded_type_mean: np.ndarray,
) -> Dict[str, object]:
    return {
        "latent": _latent_metrics(latent, truth_latent, type_mean_latent),
        "decoded_expression": _decoded_metrics(decoded, truth_expression, decoded_type_mean),
        "state_retrieval": _retrieval_metrics(
            latent, truth_latent, true_types, selected_types, centroids
        ),
    }


def _endpoint_delta(
    endpoint: Mapping[str, object], reference: Mapping[str, object]
) -> Dict[str, float]:
    latent = endpoint["latent"]
    reference_latent = reference["latent"]
    decoded = endpoint["decoded_expression"]
    reference_decoded = reference["decoded_expression"]
    retrieval = endpoint["state_retrieval"]
    reference_retrieval = reference["state_retrieval"]
    assert isinstance(latent, Mapping) and isinstance(reference_latent, Mapping)
    assert isinstance(decoded, Mapping) and isinstance(reference_decoded, Mapping)
    assert isinstance(retrieval, Mapping) and isinstance(reference_retrieval, Mapping)
    return {
        "within_type_r2_delta": float(latent["within_type_r2"])
        - float(reference_latent["within_type_r2"]),
        "within_type_cosine_delta": float(latent["within_type_cosine"])
        - float(reference_latent["within_type_cosine"]),
        "within_type_rmse_improvement": float(reference_latent["rmse"]) - float(latent["rmse"]),
        "decoded_expression_rmse_improvement": float(reference_decoded["rmse"])
        - float(decoded["rmse"]),
        "retrieval_top1_delta": float(retrieval["top1"]) - float(reference_retrieval["top1"]),
        "retrieval_mrr_delta": float(retrieval["mrr"]) - float(reference_retrieval["mrr"]),
    }


def _donor_bootstrap_lower_bound(
    values: Sequence[float],
    *,
    seed: int,
    iterations: int,
    confidence: float,
) -> float:
    """Return a deterministic donor-equal bootstrap lower confidence bound."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("donor bootstrap values must be a non-empty finite vector")
    if iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_confidence must lie in (0, 1)")
    if len(array) == 1:
        return float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(iterations, len(array)))
    draws = array[indices].mean(axis=1)
    return float(np.quantile(draws, (1.0 - confidence) / 2.0))


def _wrong_donor_bank_contrasts(
    prediction: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    donors: np.ndarray,
    centroids: np.ndarray,
) -> Sequence[Mapping[str, object]]:
    """Compare exact matched state error with every same-type wrong-donor bank."""

    rows = []
    for donor in sorted(set(donors.tolist())):
        query_indices = np.flatnonzero(donors == donor)
        for wrong_donor in sorted(set(donors.tolist()) - {donor}):
            wrong_indices = np.flatnonzero(donors == wrong_donor)
            matched_squared = []
            wrong_squared = []
            supported = 0
            for index in query_indices:
                selected_type = int(labels[index])
                candidates = wrong_indices[labels[wrong_indices] == selected_type]
                if not len(candidates):
                    continue
                query = prediction[index] - centroids[selected_type]
                matched = truth[index] - centroids[selected_type]
                bank = truth[candidates] - centroids[selected_type]
                matched_squared.append(float(np.square(query - matched).sum()))
                wrong_squared.append(float(np.square(bank - query[None, :]).sum(axis=1).min()))
                supported += 1
            if supported:
                matched_rmse = float(np.sqrt(np.mean(matched_squared)))
                wrong_rmse = float(np.sqrt(np.mean(wrong_squared)))
                rows.append(
                    {
                        "donor_id": donor,
                        "wrong_donor_id": wrong_donor,
                        "supported_cells": supported,
                        "matched_state_rmse": matched_rmse,
                        "nearest_wrong_bank_state_rmse": wrong_rmse,
                        "matched_vs_wrong_bank_rmse_margin": wrong_rmse - matched_rmse,
                    }
                )
    return rows


def evaluate_morphology_state_checkpoint(
    checkpoint_path: Union[str, os.PathLike[str]],
    frozen_features: ArrayLike,
    latent_targets: ArrayLike,
    type_labels: ArrayLike,
    donor_ids: ArrayLike,
    *,
    decoder: nn.Module,
    expression_targets: ArrayLike,
    roi_ids: Optional[ArrayLike] = None,
    seed: int = 17,
    device: str = "auto",
    minimum_within_type_r2: float = 0.05,
    bootstrap_iterations: int = 2000,
    bootstrap_confidence: float = 0.95,
    require_wrong_donor_banks: bool = True,
) -> Dict[str, object]:
    """Load a checkpoint and regenerate all held-out morphology-state endpoints."""

    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    model = MorphologyStateGate.load_checkpoint(source)
    features = _numpy_float_matrix(frozen_features, "frozen_features", model.config.feature_dim)
    latents = _numpy_float_matrix(latent_targets, "latent_targets", model.config.latent_dim)
    if len(features) != len(latents):
        raise ValueError("frozen_features and latent_targets must align")
    labels = _numpy_type_vector(type_labels, "type_labels", len(features), model.config.num_types)
    donors = _numpy_string_vector(donor_ids, "donor_ids", len(features))
    heldout_donors = tuple(sorted(set(donors.tolist())))
    overlap = sorted(set(heldout_donors) & set(model.training_donors))
    if overlap:
        raise ValueError("held-out evaluation overlaps training donors: %s" % ", ".join(overlap))
    rois = None if roi_ids is None else _numpy_string_vector(roi_ids, "roi_ids", len(features))
    expression_array = np.asarray(
        expression_targets.detach().cpu().numpy()
        if isinstance(expression_targets, Tensor)
        else expression_targets,
        dtype=np.float32,
    )
    if expression_array.ndim != 2 or len(expression_array) != len(features):
        raise ValueError("expression_targets must be a matrix aligned to frozen_features")
    if not np.isfinite(expression_array).all():
        raise ValueError("expression_targets must be finite")
    if not np.isfinite(minimum_within_type_r2) or minimum_within_type_r2 < 0.0:
        raise ValueError("minimum_within_type_r2 must be finite and non-negative")
    if not isinstance(require_wrong_donor_banks, bool):
        raise TypeError("require_wrong_donor_banks must be boolean")

    target_device = resolve_device(device)
    model.to(target_device).eval()
    decoder.to(target_device).eval()
    x = torch.from_numpy(features).to(target_device)
    y = torch.from_numpy(labels).to(target_device)
    with torch.no_grad():
        oracle_output = model(x, y)
        predicted_output = model(x)
        type_mean_tensor = model.type_mean_latent(y)
        decoder_ceiling = decoder(torch.from_numpy(latents).to(target_device))
        decoded_type_mean = decoder(type_mean_tensor)
        decoded_oracle = decoder(oracle_output.latent)
        decoded_predicted = decoder(predicted_output.latent)

    centroids = model.type_centroids.detach().cpu().numpy().astype(np.float64)
    type_mean = type_mean_tensor.detach().cpu().numpy().astype(np.float64)
    oracle_latent = oracle_output.latent.detach().cpu().numpy().astype(np.float64)
    predicted_latent = predicted_output.latent.detach().cpu().numpy().astype(np.float64)
    predicted_types = predicted_output.predicted_type.detach().cpu().numpy().astype(np.int64)
    truth_latent = latents.astype(np.float64)
    truth_expression = expression_array.astype(np.float64)
    decoded_mean_array = decoded_type_mean.detach().cpu().numpy().astype(np.float64)

    endpoints: Dict[str, Dict[str, object]] = {}
    endpoints["decoder_ceiling"] = {
        "decoded_expression": _decoded_metrics(
            decoder_ceiling.detach().cpu().numpy().astype(np.float64),
            truth_expression,
            decoded_mean_array,
        )
    }
    endpoints["oracle_type_mean"] = _endpoint(
        type_mean,
        truth_latent=truth_latent,
        type_mean_latent=type_mean,
        true_types=labels,
        selected_types=labels,
        centroids=centroids,
        decoded=decoded_mean_array,
        truth_expression=truth_expression,
        decoded_type_mean=decoded_mean_array,
    )
    endpoints["oracle_type_image_residual"] = _endpoint(
        oracle_latent,
        truth_latent=truth_latent,
        type_mean_latent=type_mean,
        true_types=labels,
        selected_types=labels,
        centroids=centroids,
        decoded=decoded_oracle.detach().cpu().numpy().astype(np.float64),
        truth_expression=truth_expression,
        decoded_type_mean=decoded_mean_array,
    )
    endpoints["predicted_type_image_residual"] = _endpoint(
        predicted_latent,
        truth_latent=truth_latent,
        type_mean_latent=type_mean,
        true_types=labels,
        selected_types=predicted_types,
        centroids=centroids,
        decoded=decoded_predicted.detach().cpu().numpy().astype(np.float64),
        truth_expression=truth_expression,
        decoded_type_mean=decoded_mean_array,
    )

    controls: Dict[str, object] = {}
    shuffle_specs = {"donor_type_shuffle": None}
    if rois is not None:
        shuffle_specs["donor_type_roi_shuffle"] = rois
    for name, grouping_rois in shuffle_specs.items():
        permutation = donor_type_preserving_permutation(
            donors, labels, roi_ids=grouping_rois, seed=seed
        )
        shuffled_x = torch.from_numpy(features[permutation]).to(target_device)
        with torch.no_grad():
            shuffled_output = model(shuffled_x, y)
            shuffled_decoded = decoder(shuffled_output.latent)
        endpoints[name] = _endpoint(
            shuffled_output.latent.detach().cpu().numpy().astype(np.float64),
            truth_latent=truth_latent,
            type_mean_latent=type_mean,
            true_types=labels,
            selected_types=labels,
            centroids=centroids,
            decoded=shuffled_decoded.detach().cpu().numpy().astype(np.float64),
            truth_expression=truth_expression,
            decoded_type_mean=decoded_mean_array,
        )
        controls[name] = {
            "permutation_sha256": hashlib.sha256(permutation.astype("<i8").tobytes()).hexdigest(),
            "shuffled_cells": int(np.count_nonzero(permutation != np.arange(len(features)))),
            "shuffled_fraction": float(np.mean(permutation != np.arange(len(features)))),
            "preserves_donor": True,
            "preserves_type": True,
            "preserves_roi": grouping_rois is not None,
        }

    deltas = {
        "oracle_image_residual_vs_type_mean": _endpoint_delta(
            endpoints["oracle_type_image_residual"], endpoints["oracle_type_mean"]
        ),
        "predicted_image_residual_vs_type_mean": _endpoint_delta(
            endpoints["predicted_type_image_residual"], endpoints["oracle_type_mean"]
        ),
        "oracle_image_residual_vs_donor_type_shuffle": _endpoint_delta(
            endpoints["oracle_type_image_residual"], endpoints["donor_type_shuffle"]
        ),
    }
    if "donor_type_roi_shuffle" in endpoints:
        deltas["oracle_image_residual_vs_donor_type_roi_shuffle"] = _endpoint_delta(
            endpoints["oracle_type_image_residual"], endpoints["donor_type_roi_shuffle"]
        )

    donor_metrics = []
    permutation = donor_type_preserving_permutation(donors, labels, seed=seed)
    with torch.no_grad():
        donor_shuffle_output = model(torch.from_numpy(features[permutation]).to(target_device), y)
        donor_shuffle_decoded = decoder(donor_shuffle_output.latent)
    donor_shuffle_latent = donor_shuffle_output.latent.detach().cpu().numpy().astype(np.float64)
    donor_shuffle_expression = donor_shuffle_decoded.detach().cpu().numpy().astype(np.float64)
    decoded_oracle_array = decoded_oracle.detach().cpu().numpy().astype(np.float64)
    for donor in heldout_donors:
        selected = donors == donor
        donor_oracle = _endpoint(
            oracle_latent[selected],
            truth_latent=truth_latent[selected],
            type_mean_latent=type_mean[selected],
            true_types=labels[selected],
            selected_types=labels[selected],
            centroids=centroids,
            decoded=decoded_oracle_array[selected],
            truth_expression=truth_expression[selected],
            decoded_type_mean=decoded_mean_array[selected],
        )
        donor_shuffle = _endpoint(
            donor_shuffle_latent[selected],
            truth_latent=truth_latent[selected],
            type_mean_latent=type_mean[selected],
            true_types=labels[selected],
            selected_types=labels[selected],
            centroids=centroids,
            decoded=donor_shuffle_expression[selected],
            truth_expression=truth_expression[selected],
            decoded_type_mean=decoded_mean_array[selected],
        )
        donor_metrics.append(
            {
                "donor_id": donor,
                "observations": int(selected.sum()),
                "oracle_type_image_residual": donor_oracle,
                "donor_type_shuffle": donor_shuffle,
                "oracle_vs_shuffle": _endpoint_delta(donor_oracle, donor_shuffle),
            }
        )

    wrong_bank_rows = _wrong_donor_bank_contrasts(
        oracle_latent,
        truth_latent,
        labels,
        donors,
        centroids,
    )
    donor_r2 = [
        float(row["oracle_type_image_residual"]["latent"]["within_type_r2"])
        for row in donor_metrics
    ]
    donor_decoded_delta = [
        float(row["oracle_type_image_residual"]["decoded_expression"]["rmse_delta_vs_type_mean"])
        for row in donor_metrics
    ]
    bootstrap = {
        "confidence": bootstrap_confidence,
        "iterations": bootstrap_iterations,
        "seed": seed,
        "within_type_r2_lower": _donor_bootstrap_lower_bound(
            donor_r2,
            seed=seed,
            iterations=bootstrap_iterations,
            confidence=bootstrap_confidence,
        ),
        "decoded_expression_delta_vs_type_mean_lower": _donor_bootstrap_lower_bound(
            donor_decoded_delta,
            seed=seed + 1,
            iterations=bootstrap_iterations,
            confidence=bootstrap_confidence,
        ),
    }
    oracle_latent_metrics = endpoints["oracle_type_image_residual"]["latent"]
    oracle_decoded_metrics = endpoints["oracle_type_image_residual"]["decoded_expression"]
    oracle_retrieval = endpoints["oracle_type_image_residual"]["state_retrieval"]
    shuffle_delta = deltas["oracle_image_residual_vs_donor_type_shuffle"]
    checks = {
        "within_type_partial_r2": min(donor_r2) >= minimum_within_type_r2,
        "within_type_shuffle_r2_delta": float(shuffle_delta["within_type_r2_delta"]) > 0.0,
        "within_type_shuffle_cosine_delta": (
            float(shuffle_delta["within_type_cosine_delta"]) > 0.0
        ),
        "within_type_shuffle_rmse_delta": (
            float(shuffle_delta["within_type_rmse_improvement"]) > 0.0
        ),
        "decoded_expression_delta_vs_type_mean": (
            float(oracle_decoded_metrics["rmse_delta_vs_type_mean"]) > 0.0
        ),
        "state_retrieval_top1": (
            float(oracle_retrieval["top1"]) > float(oracle_retrieval["uniform_expected_top1"])
        ),
        "state_retrieval_mrr": (
            float(oracle_retrieval["mrr"]) > float(oracle_retrieval["uniform_expected_mrr"])
        ),
        "matched_better_than_every_wrong_bank": bool(wrong_bank_rows)
        and all(float(row["matched_vs_wrong_bank_rmse_margin"]) > 0.0 for row in wrong_bank_rows),
        "no_type_collapse": len(np.unique(predicted_types)) == model.config.num_types,
        "donor_bootstrap_within_type_r2_lower_above_zero": (
            float(bootstrap["within_type_r2_lower"]) > 0.0
        ),
        "donor_bootstrap_decoded_delta_lower_above_zero": (
            float(bootstrap["decoded_expression_delta_vs_type_mean_lower"]) > 0.0
        ),
    }
    if not require_wrong_donor_banks:
        checks["matched_better_than_every_wrong_bank"] = True
    missing_controls = []
    if require_wrong_donor_banks and not wrong_bank_rows:
        missing_controls.append("wrong_donor_state_bank")
    passed = not missing_controls and all(checks.values())

    primary_metrics = {
        "within_type_partial_r2": float(oracle_latent_metrics["within_type_partial_r2"]),
        "within_type_cosine": float(oracle_latent_metrics["within_type_cosine"]),
        "within_type_rmse_delta_vs_type_mean": float(
            oracle_latent_metrics["within_type_rmse_delta_vs_type_mean"]
        ),
        "within_type_shuffle_cosine_delta": float(shuffle_delta["within_type_cosine_delta"]),
        "within_type_shuffle_rmse_delta": float(shuffle_delta["within_type_rmse_improvement"]),
        "state_retrieval_top1": float(oracle_retrieval["top1"]),
        "state_retrieval_mrr": float(oracle_retrieval["mrr"]),
        "decoded_expression_delta_vs_type_mean": float(
            oracle_decoded_metrics["rmse_delta_vs_type_mean"]
        ),
    }

    return {
        "schema_version": MORPHOLOGY_STATE_REPORT_SCHEMA,
        "status": "pass" if passed else ("blocked_controls" if missing_controls else "fail"),
        "pass": passed,
        "missing_controls": missing_controls,
        "thresholds": {"minimum_within_type_r2": minimum_within_type_r2},
        "checks": checks,
        "primary_metrics": primary_metrics,
        "donor_metrics": donor_metrics,
        "donor_bootstrap": bootstrap,
        "wrong_donor_bank_contrasts": wrong_bank_rows,
        "checkpoint": {
            "path": str(source),
            "sha256": sha256_file(source),
            "schema_version": MORPHOLOGY_STATE_CHECKPOINT_SCHEMA,
            "training_donors": list(model.training_donors),
            "training_sample_count": model.training_sample_count,
        },
        "heldout": {
            "donors": list(heldout_donors),
            "observations": int(len(features)),
            "donor_disjoint_from_training": True,
            "types": list(model.config.type_names),
        },
        "execution": {
            "checkpoint_executed": True,
            "frozen_input_features": True,
            "frozen_decoder": True,
            "graph_used": False,
            "uot_used": False,
            "refinement_used": False,
            "device": str(target_device),
            "seed": seed,
        },
        "type_classification": {
            "accuracy": float(np.mean(predicted_types == labels)),
            "predicted_occupancy_fraction": float(
                len(np.unique(predicted_types)) / model.config.num_types
            ),
        },
        "controls": controls,
        "endpoints": endpoints,
        "deltas": deltas,
    }


__all__ = [
    "MORPHOLOGY_STATE_CHECKPOINT_SCHEMA",
    "MORPHOLOGY_STATE_REPORT_SCHEMA",
    "MorphologyStateGateConfig",
    "MorphologyStateOutput",
    "MorphologyStateGate",
    "fit_morphology_state_gate",
    "donor_type_preserving_permutation",
    "evaluate_morphology_state_checkpoint",
]
