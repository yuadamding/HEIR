"""Deterministic, bounded training for retrospective neural residual probes.

This module contains only the estimator mechanics.  It does not load a study,
select an architecture, or decide whether a scientific gate passed.  Callers
must supply explicit donor-held-out rows and stable observation identifiers.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Mapping, Optional, Union

import numpy as np
import torch
from torch import nn

from .hierarchical_metrics import donor_section_type_macro_r2
from .neural_checkpoint import canonical_model_state_sha256

SUPPORTED_ARCHITECTURES = ("shared_linear", "mlp_tiny", "mlp_small", "late_fusion")


@dataclass(frozen=True)
class NeuralArchitecture:
    """A fully specified residual-network architecture."""

    model_id: str
    input_dim: int
    output_dim: int
    num_types: int
    view_count: int = 1
    view_dims: tuple[int, ...] = ()
    type_conditioned: bool = False
    type_embedding_width: int = 16
    adapter_rank: int = 8

    def __post_init__(self) -> None:
        if self.model_id not in SUPPORTED_ARCHITECTURES:
            raise ValueError("unsupported neural residual architecture")
        if min(self.input_dim, self.output_dim, self.num_types, self.view_count) < 1:
            raise ValueError("neural architecture dimensions must be positive")
        if self.model_id != "late_fusion" and (self.view_count != 1 or self.view_dims):
            raise ValueError("only late_fusion accepts multiple feature views")
        if self.model_id == "late_fusion":
            if len(self.view_dims) != self.view_count or any(width < 1 for width in self.view_dims):
                raise ValueError("late_fusion requires one positive width per view")
            if sum(self.view_dims) != self.input_dim:
                raise ValueError("late-fusion view widths must sum to the flat input width")
        if self.type_conditioned and min(self.type_embedding_width, self.adapter_rank) < 1:
            raise ValueError("type-adapter dimensions must be positive")


@dataclass(frozen=True)
class ValidationRows:
    """Explicit held-out rows used only for epoch selection."""

    features: np.ndarray
    targets: np.ndarray
    type_labels: np.ndarray
    donor_ids: np.ndarray
    section_ids: np.ndarray
    observation_ids: np.ndarray
    coordinate_mean: np.ndarray
    coordinate_scale: np.ndarray
    minimum_support: int = 2


@dataclass(frozen=True)
class NeuralTrainingResult:
    """In-memory deterministic checkpoint and complete bounded history."""

    architecture: NeuralArchitecture
    state_dict: Mapping[str, torch.Tensor]
    best_epoch: int
    epochs_run: int
    history: tuple[Mapping[str, float], ...]
    checkpoint_sha256: str
    parameter_count: int
    fit_device: str
    seed: int


class _TypeFiLM(nn.Module):
    def __init__(self, num_types: int, embedding_width: int, rank: int, hidden: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_types, embedding_width)
        self.down = nn.Linear(embedding_width, rank, bias=False)
        self.up = nn.Linear(rank, hidden * 2)

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scale, shift = self.up(self.down(self.embedding(labels))).chunk(2, dim=1)
        return hidden * (1.0 + scale) + shift


class _TypeLinearAdapter(nn.Module):
    """Low-rank type-specific linear correction to a shared linear map."""

    def __init__(self, num_types: int, input_dim: int, rank: int, output_dim: int) -> None:
        super().__init__()
        self.down = nn.Parameter(torch.empty(num_types, input_dim, rank))
        self.up = nn.Parameter(torch.empty(num_types, rank, output_dim))
        nn.init.xavier_uniform_(self.down)
        nn.init.zeros_(self.up)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        down = self.down[labels]
        up = self.up[labels]
        latent = torch.einsum("bi,bir->br", features, down)
        return torch.einsum("br,bro->bo", latent, up)


class ResidualNetwork(nn.Module):
    """Small shared image-to-residual translator with optional type FiLM."""

    def __init__(self, architecture: NeuralArchitecture) -> None:
        super().__init__()
        self.architecture = architecture
        model_id = architecture.model_id
        if model_id == "late_fusion":
            self.view_norms = nn.ModuleList(
                [nn.LayerNorm(width) for width in architecture.view_dims]
            )
            self.view_projections = nn.ModuleList(
                [nn.Linear(width, 64) for width in architecture.view_dims]
            )
            self.fusion = nn.Linear(64 * architecture.view_count, 64)
            self.output = nn.Linear(64, architecture.output_dim)
            first_hidden = 64
        else:
            if model_id == "shared_linear":
                self.output = nn.Linear(architecture.input_dim, architecture.output_dim)
                self.linear_adapter = (
                    _TypeLinearAdapter(
                        architecture.num_types,
                        architecture.input_dim,
                        architecture.adapter_rank,
                        architecture.output_dim,
                    )
                    if architecture.type_conditioned
                    else None
                )
                first_hidden = 0
            elif model_id == "mlp_tiny":
                self.input_norm = nn.LayerNorm(architecture.input_dim)
                self.first = nn.Linear(architecture.input_dim, 64)
                self.dropout = nn.Dropout(0.0)
                self.output = nn.Linear(64, architecture.output_dim)
                first_hidden = 64
            else:
                self.input_norm = nn.LayerNorm(architecture.input_dim)
                self.first = nn.Linear(architecture.input_dim, 256)
                self.dropout = nn.Dropout(0.2)
                self.second = nn.Linear(256, 64)
                self.output = nn.Linear(64, architecture.output_dim)
                first_hidden = 256
        self.film = (
            _TypeFiLM(
                architecture.num_types,
                architecture.type_embedding_width,
                architecture.adapter_rank,
                first_hidden,
            )
            if architecture.type_conditioned and model_id != "shared_linear"
            else None
        )

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        model_id = self.architecture.model_id
        if model_id == "late_fusion":
            if features.ndim != 2 or features.shape[1] != self.architecture.input_dim:
                raise ValueError("late-fusion features do not match the registered flat width")
            split_views = torch.split(features, self.architecture.view_dims, dim=1)
            views = [
                torch.nn.functional.gelu(projection(norm(view)))
                for view, norm, projection in zip(
                    split_views, self.view_norms, self.view_projections
                )
            ]
            hidden = self.fusion(torch.cat(views, dim=1))
            if self.film is not None:
                hidden = self.film(hidden, labels)
            return self.output(torch.nn.functional.gelu(hidden))

        if features.ndim != 2:
            raise ValueError("single-view residual networks require a feature matrix")
        if model_id == "shared_linear":
            result = self.output(features)
            if self.linear_adapter is not None:
                result = result + self.linear_adapter(features, labels)
            return result
        normalized = self.input_norm(features)
        hidden = self.first(normalized)
        if self.film is not None:
            hidden = self.film(hidden, labels)
        hidden = self.dropout(torch.nn.functional.gelu(hidden))
        if model_id == "mlp_small":
            hidden = torch.nn.functional.gelu(self.second(hidden))
        return self.output(hidden)


def _seed_everything(seed: int, *, deterministic: bool, use_cuda: bool) -> None:
    if seed < 0:
        raise ValueError("training seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False


def _device(requested: str) -> torch.device:
    target = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "auto"
        else torch.device(requested)
    )
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("only CPU and CUDA neural training is supported")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if target.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise RuntimeError(
            "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG before process start"
        )
    return target


def prepare_neural_execution(
    seed: int,
    device: str,
    *,
    deterministic: bool = True,
) -> torch.device:
    """Resolve and seed execution before any target-basis CUDA operation."""

    target = _device(device)
    _seed_everything(seed, deterministic=deterministic, use_cuda=target.type == "cuda")
    return target


def _validated_arrays(
    features: object,
    targets: object,
    labels: object,
    observation_ids: object,
    architecture: NeuralArchitecture,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    types = np.asarray(labels, dtype=np.int64)
    identities = np.asarray(observation_ids).astype(str)
    expected_shape = (len(y), architecture.input_dim)
    if y.ndim != 2 or y.shape[1] != architecture.output_dim or not len(y):
        raise ValueError("training targets do not match the registered output width")
    if x.shape != expected_shape:
        raise ValueError("training features do not match the registered architecture")
    if types.shape != (len(y),) or identities.shape != (len(y),):
        raise ValueError("training labels and observation IDs must be row aligned")
    if len(set(identities.tolist())) != len(identities) or any(
        not value.strip() for value in identities.tolist()
    ):
        raise ValueError("observation IDs must be non-empty and unique")
    if np.any(types < 0) or np.any(types >= architecture.num_types):
        raise ValueError("training type labels are outside the registered ontology")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("neural training arrays must be finite")
    order = np.argsort(identities, kind="stable")
    return x[order], y[order], types[order], identities[order]


def _validation_score(
    model: nn.Module,
    rows: ValidationRows,
    architecture: NeuralArchitecture,
    device: torch.device,
    batch_size: int,
) -> float:
    identities = np.asarray(rows.observation_ids).astype(str)
    if identities.shape != (len(rows.targets),) or len(set(identities.tolist())) != len(identities):
        raise ValueError("validation observation IDs must be unique and row aligned")
    order = np.argsort(identities, kind="stable")
    prediction = predict_neural_model(
        model,
        np.asarray(rows.features)[order],
        np.asarray(rows.type_labels)[order],
        architecture=architecture,
        device=str(device),
        batch_size=batch_size,
    )
    mean = np.asarray(rows.coordinate_mean, dtype=np.float64)
    scale = np.asarray(rows.coordinate_scale, dtype=np.float64)
    if mean.shape != (architecture.output_dim,) or scale.shape != mean.shape:
        raise ValueError("validation coordinate standardizer is malformed")
    prediction = prediction * scale + mean
    truth = np.asarray(rows.targets, dtype=np.float64)[order] * scale + mean
    score, _, _, _ = donor_section_type_macro_r2(
        truth,
        prediction,
        np.asarray(rows.donor_ids).astype(str)[order],
        np.asarray(rows.section_ids).astype(str)[order],
        np.asarray(rows.type_labels, dtype=np.int64)[order],
        int(rows.minimum_support),
    )
    return float(score)


def train_neural_model(
    features: np.ndarray,
    targets: np.ndarray,
    type_labels: np.ndarray,
    observation_ids: np.ndarray,
    sample_weight: np.ndarray,
    *,
    architecture: NeuralArchitecture,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    max_epochs: int = 100,
    batch_size: int = 256,
    patience: int = 10,
    gradient_clip: float = 1.0,
    seed: int = 17,
    device: str = "auto",
    validation: Optional[ValidationRows] = None,
    deterministic: bool = True,
) -> NeuralTrainingResult:
    """Fit one bounded model; optional validation rows select the epoch only."""

    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if min(max_epochs, batch_size, patience) < 1 or gradient_clip <= 0.0:
        raise ValueError("training bounds must be positive")
    x, y, labels, identities = _validated_arrays(
        features, targets, type_labels, observation_ids, architecture
    )
    original_ids = np.asarray(observation_ids).astype(str)
    weight = np.asarray(sample_weight, dtype=np.float64)
    if (
        weight.shape != (len(original_ids),)
        or not np.all(np.isfinite(weight))
        or np.any(weight <= 0)
    ):
        raise ValueError("sample weights must be finite, positive, and row aligned")
    weight_lookup = dict(zip(original_ids.tolist(), weight.tolist()))
    ordered_weight = np.asarray([weight_lookup[value] for value in identities], dtype=np.float32)
    ordered_weight /= ordered_weight.mean(dtype=np.float64)

    target_device = prepare_neural_execution(seed, device, deterministic=deterministic)
    model = ResidualNetwork(architecture).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    weight_tensor = torch.as_tensor(ordered_weight, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_score = -float("inf")
    best_epoch = max_epochs
    best_state: Optional[dict[str, torch.Tensor]] = None
    stale = 0
    history: list[Mapping[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_tensor), generator=generator)
        loss_sum = 0.0
        row_sum = 0
        for start in range(0, len(permutation), batch_size):
            index = permutation[start : start + batch_size]
            batch_x = x_tensor[index].to(target_device)
            batch_y = y_tensor[index].to(target_device)
            batch_labels = label_tensor[index].to(target_device)
            batch_weight = weight_tensor[index].to(target_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x, batch_labels)
            row_loss = torch.mean(torch.square(prediction - batch_y), dim=1)
            # Weights have global mean one.  Averaging weighted row losses
            # makes one complete epoch exactly the registered donor/type-
            # weighted MSE, independent of batch boundaries.
            loss = torch.mean(row_loss * batch_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(index)
            row_sum += len(index)
        row: dict[str, float] = {"epoch": float(epoch), "training_loss": loss_sum / row_sum}
        if validation is not None:
            model.eval()
            validation_score = _validation_score(
                model, validation, architecture, target_device, batch_size
            )
            row["validation_donor_section_type_r2"] = validation_score
            if validation_score > best_score + 1.0e-12:
                best_score = validation_score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= patience:
            break

    if validation is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = max_epochs
    if best_state is None:
        raise RuntimeError("no finite donor-balanced validation checkpoint was selected")
    model.load_state_dict(best_state)
    canonical = {name: tensor.detach().cpu().contiguous() for name, tensor in best_state.items()}
    return NeuralTrainingResult(
        architecture=architecture,
        state_dict=canonical,
        best_epoch=int(best_epoch),
        epochs_run=len(history),
        history=tuple(history),
        checkpoint_sha256=canonical_model_state_sha256(canonical),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        fit_device=str(target_device),
        seed=int(seed),
    )


def predict_neural_model(
    model_or_result: Union[nn.Module, NeuralTrainingResult],
    features: np.ndarray,
    type_labels: np.ndarray,
    *,
    architecture: Optional[NeuralArchitecture] = None,
    device: str = "auto",
    batch_size: int = 1024,
) -> np.ndarray:
    """Predict residual coordinates without changing model state."""

    if batch_size < 1:
        raise ValueError("prediction batch size must be positive")
    if isinstance(model_or_result, NeuralTrainingResult):
        architecture = model_or_result.architecture
        model = ResidualNetwork(architecture)
        model.load_state_dict(model_or_result.state_dict)
    else:
        if architecture is None:
            raise ValueError("architecture is required for an in-memory model")
        model = model_or_result
    x = np.asarray(features, dtype=np.float32)
    labels = np.asarray(type_labels, dtype=np.int64)
    expected = (len(labels), architecture.input_dim)
    if x.shape != expected or np.any(labels < 0) or np.any(labels >= architecture.num_types):
        raise ValueError("prediction rows do not match the registered architecture")
    if not np.all(np.isfinite(x)):
        raise ValueError("prediction features must be finite")
    target_device = _device(device)
    model = model.to(target_device)
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            batch = torch.as_tensor(x[start:stop], dtype=torch.float32, device=target_device)
            batch_labels = torch.as_tensor(
                labels[start:stop], dtype=torch.long, device=target_device
            )
            output.append(model(batch, batch_labels).cpu().numpy())
    result = np.concatenate(output, axis=0).astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("neural residual prediction is non-finite")
    return result


__all__ = [
    "NeuralArchitecture",
    "NeuralTrainingResult",
    "prepare_neural_execution",
    "ResidualNetwork",
    "SUPPORTED_ARCHITECTURES",
    "ValidationRows",
    "predict_neural_model",
    "train_neural_model",
]
