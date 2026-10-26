"""Sparse cell-to-assay aggregation on the nonnegative linear expression scale."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    normalization_indices: Sequence[int],
    library_size: float = 1e4,
) -> torch.Tensor:
    """Normalize on a fixed complete panel after aggregation, then apply log1p.

    ``normalization_indices`` names the denominator independently of outcome
    selection. Every normalization gene must be observed at every retained region;
    incomplete or zero-exposure regions must be qualified/excluded upstream. The
    outcome mask never changes the denominator. Missing outcomes are filled only
    after the fixed denominator has been computed, and must be masked in scoring.
    """
    if values.ndim != 2 or values.shape != observed.shape or observed.dtype != torch.bool:
        raise ValueError("normalization requires a matching boolean availability mask")
    indices = tuple(normalization_indices)
    if len(indices) < 2 or len(set(indices)) != len(indices):
        raise ValueError("normalization requires at least two distinct fixed-panel genes")
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in indices):
        raise ValueError("normalization indices must be integers")
    if min(indices) < 0 or max(indices) >= values.shape[1]:
        raise ValueError("normalization index outside the named gene axis")
    if not np.isfinite(library_size) or library_size <= 0:
        raise ValueError("library_size must be finite and positive")
    if not observed[:, indices].all():
        raise ValueError("incomplete normalization coverage; use a common complete fixed panel")
    present = values[observed]
    if not torch.isfinite(present).all() or torch.any(present < 0):
        raise ValueError("normalization requires nonnegative finite linear values")
    denominator = values[:, indices].sum(dim=1, keepdim=True)
    if not torch.isfinite(denominator).all() or torch.any(denominator <= 0):
        raise ValueError("normalization denominator must be finite and positive in every region")
    masked = torch.where(observed, values, torch.zeros_like(values))
    return torch.log1p(masked / denominator * library_size)
