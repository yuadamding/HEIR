"""Sparse cell-to-assay aggregation on the nonnegative linear expression scale."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .schema import Prediction, identifiers


@dataclass(frozen=True)
class ObservationOperator:
    cell_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    rows: np.ndarray
    columns: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        identifiers(self.cell_ids, "observation cell_ids")
        identifiers(self.observation_ids, "observation_ids")
        if np.ndim(self.weights) != 1:
            raise ValueError("observation weights must be a vector")
        for name, size in (("rows", len(self.observation_ids)), ("columns", len(self.cell_ids))):
            index = np.asarray(getattr(self, name))
            if index.shape != np.shape(self.weights) or index.dtype.kind not in "iu":
                raise ValueError(f"observation {name} must be integer indices matching weights")
            if np.any(index < 0) or np.any(index >= size):
                raise ValueError(f"observation {name} index out of bounds")
        if not np.isfinite(self.weights).all() or np.any(self.weights < 0):
            raise ValueError("observation weights must be finite and nonnegative")
        coverage = np.bincount(self.rows, weights=self.weights, minlength=len(self.observation_ids))
        if np.any(coverage <= 0):
            raise ValueError(
                "every observation needs positive cell coverage; exclude uncovered rows"
            )

    def aggregate(self, prediction: Prediction, *, component: str = "expression") -> torch.Tensor:
        if set(prediction.cell_ids) != set(self.cell_ids):
            raise ValueError("prediction and observation cell identifiers differ")
        if component not in {"expression", "transfer", "expression_adjustment"}:
            raise ValueError("only linear expression components can be aggregated")
        expression = getattr(prediction, component)
        lookup = {key: i for i, key in enumerate(prediction.cell_ids)}
        order = torch.tensor([lookup[key] for key in self.cell_ids], device=expression.device)
        cells = expression.index_select(0, order)
        columns = torch.as_tensor(self.columns, dtype=torch.long, device=cells.device)
        rows = torch.as_tensor(self.rows, dtype=torch.long, device=cells.device)
        weights = torch.as_tensor(self.weights, dtype=cells.dtype, device=cells.device)
        result = cells.new_zeros((len(self.observation_ids), cells.shape[1]))
        return result.index_add(0, rows, cells[columns] * weights[:, None])

    @classmethod
    def load(cls, path: str | Path) -> ObservationOperator:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                cell_ids=tuple(data["cell_ids"].tolist()),
                observation_ids=tuple(data["observation_ids"].tolist()),
                rows=data["rows"].copy(),
                columns=data["columns"].copy(),
                weights=data["weights"].copy(),
            )


def normalized_log_expression(
    values: torch.Tensor,
    observed: torch.Tensor,
    *,
    library_size: float = 1e4,
) -> torch.Tensor:
    """Normalize within the declared assay panel *after* aggregation, then log1p.

    The same availability mask is applied to targets and predictions; missing values,
    including NaN under the mask, cannot affect the denominator.
    """
    if values.shape != observed.shape or observed.dtype != torch.bool:
        raise ValueError("normalization requires a matching boolean availability mask")
    if not np.isfinite(library_size) or library_size <= 0:
        raise ValueError("library_size must be finite and positive")
    present = values[observed]
    if not torch.isfinite(present).all() or torch.any(present < 0):
        raise ValueError("normalization requires nonnegative finite linear values")
    masked = torch.where(observed, values, torch.zeros_like(values))
    denominator = masked.sum(dim=1, keepdim=True)
    return torch.log1p(masked / denominator.clamp_min(1e-12) * library_size)
