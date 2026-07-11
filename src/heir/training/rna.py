"""A compact VAE fallback when a full scVI environment is unavailable."""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..models.rna import RNAVAE
from ..utils import resolve_device, set_seed


@dataclass(frozen=True)
class RNATrainingResult:
    best_epoch: int
    train_loss: float
    validation_loss: float
    history: Tuple[Dict[str, float], ...]


def _loss(model: RNAVAE, expression: Tensor, beta: float) -> Tensor:
    output = model(expression)
    reconstruction = F.huber_loss(output.reconstruction, expression)
    kl = model.kl_divergence(output.mu, output.logvar)
    return reconstruction + beta * kl


def train_rna_vae(
    model: RNAVAE,
    expression: np.ndarray,
    donor_ids: Sequence[object],
    validation_donors: Sequence[object],
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1.0e-3,
    beta: float = 1.0e-3,
    patience: int = 15,
    seed: int = 17,
    device: str = "auto",
) -> RNATrainingResult:
    """Train with donor-held-out validation, never a random cell split."""

    values = np.asarray(expression, dtype=np.float32)
    donors = np.asarray([str(value) for value in donor_ids])
    held_out = set(str(value) for value in validation_donors)
    if values.ndim != 2 or values.shape[0] != len(donors):
        raise ValueError("expression and donor_ids are misaligned")
    if values.shape[1] != model.config.input_dim:
        raise ValueError("expression width differs from the RNA model")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("RNA VAE expects finite non-negative normalized expression")
    validation_mask = np.asarray([donor in held_out for donor in donors])
    if not validation_mask.any() or validation_mask.all():
        raise ValueError("validation_donors must define a non-empty donor-held-out split")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or beta < 0 or patience <= 0:
        raise ValueError("invalid training hyperparameters")

    set_seed(seed)
    target_device = resolve_device(device)
    model.to(target_device)
    generator = torch.Generator().manual_seed(seed)
    training_tensor = torch.from_numpy(values[~validation_mask])
    validation_tensor = torch.from_numpy(values[validation_mask]).to(target_device)
    loader = DataLoader(
        TensorDataset(training_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    best_state = None
    best_validation = float("inf")
    best_epoch = -1
    history = []
    stale = 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for (batch,) in loader:
            batch = batch.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = _loss(model, batch, beta)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(batch_loss.detach()) * len(batch)
            count += len(batch)
        model.eval()
        with torch.no_grad():
            validation_loss = float(_loss(model, validation_tensor, beta).detach())
        train_loss = total / max(count, 1)
        history.append(
            {"epoch": float(epoch), "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_validation - 1.0e-8:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return RNATrainingResult(
        best_epoch=best_epoch,
        train_loss=float(history[best_epoch]["train_loss"]),
        validation_loss=best_validation,
        history=tuple(history),
    )
