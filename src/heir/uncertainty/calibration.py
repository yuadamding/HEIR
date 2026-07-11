"""Post-hoc calibration fitted on development donors only."""

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def expected_calibration_error(
    probabilities: Tensor,
    labels: Tensor,
    num_bins: int = 15,
    ignore_index: int = -1,
) -> Tensor:
    """Top-label ECE with equal-width confidence bins."""

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape (items, classes>=2)")
    if labels.shape != (probabilities.shape[0],) or labels.dtype != torch.long:
        raise ValueError("labels must be a long vector aligned to probabilities")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if probabilities.numel() == 0:
        return probabilities.sum() * 0.0
    valid = labels != ignore_index
    if not bool(valid.any()):
        return probabilities.sum() * 0.0
    selected = probabilities[valid]
    selected_labels = labels[valid]
    confidence, predictions = selected.max(dim=1)
    correct = (predictions == selected_labels).to(selected.dtype)
    result = selected.new_zeros(())
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=selected.device)
    for index in range(num_bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        if bool(mask.any()):
            fraction = mask.to(selected.dtype).mean()
            result = result + fraction * (confidence[mask].mean() - correct[mask].mean()).abs()
    return result


class TemperatureScaler(nn.Module):
    """A scalar log-temperature with a guarded development-only fit API."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())
        self.fitted = False
        self.provenance: Dict[str, str] = {}

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must have shape (items, classes)")
        return logits / self.temperature

    def probabilities(self, logits: Tensor) -> Tensor:
        return F.softmax(self(logits), dim=-1)

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        analysis_role: str,
        donor_ids: Optional[Tensor] = None,
        max_iter: int = 100,
    ) -> "TemperatureScaler":
        """Fit NLL and reject locked-test roles by construction."""

        role = analysis_role.strip().lower()
        if role not in {"development", "calibration", "inner_validation"}:
            raise ValueError("temperature scaling requires a development/calibration split")
        if logits.ndim != 2 or labels.shape != (logits.shape[0],):
            raise ValueError("logits and labels are misaligned")
        if labels.dtype != torch.long:
            raise TypeError("labels must have dtype torch.long")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        detached_logits = logits.detach()
        detached_labels = labels.detach()
        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=0.1,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(self(detached_logits), detached_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            lower = self.log_temperature.new_tensor(0.05).log()
            upper = self.log_temperature.new_tensor(20.0).log()
            self.log_temperature.clamp_(lower, upper)
        self.fitted = True
        self.provenance = {
            "analysis_role": role,
            "num_items": str(logits.shape[0]),
            "num_donors": str(len(torch.unique(donor_ids))) if donor_ids is not None else "unknown",
        }
        return self

    def to_payload(self) -> Dict[str, object]:
        return {
            "temperature": float(self.temperature.detach().cpu()),
            "fitted": self.fitted,
            "provenance": dict(self.provenance),
        }


def brier_score(probabilities: Tensor, labels: Tensor, ignore_index: int = -1) -> Tensor:
    """Multiclass Brier score used by the blueprint's calibration endpoint."""

    if probabilities.ndim != 2 or labels.shape != (probabilities.shape[0],):
        raise ValueError("probabilities and labels are misaligned")
    valid = labels != ignore_index
    if not bool(valid.any()):
        return probabilities.sum() * 0.0
    target = F.one_hot(labels[valid], probabilities.shape[1]).to(probabilities.dtype)
    return (probabilities[valid] - target).square().sum(dim=1).mean()
