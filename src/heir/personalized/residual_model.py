"""Frozen-image baseline plus a bounded, directly scored normalized-log residual.

The RNA mapper supplies features and a separately inspectable linear transfer.
It is optimized through the residual objective, not a state-placement objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .joint_model import JointModel, JointModelConfig
from .observation import normalized_log_expression
from .schema import Prediction


@dataclass(frozen=True)
class ResidualPrediction:
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    coordinates_um: torch.Tensor
    patient_id: str
    specimen_id: str
    prediction_unit: str
    coordinate_units: str
    normalized_log_expression: torch.Tensor
    image_baseline: torch.Tensor
    residual: torch.Tensor
    linear_reference_transfer: Prediction | None
    diagnostics: dict


class ResidualModel(nn.Module):
    """Reuse joint encoders and hierarchical mapper without its expression decoder."""

    family = "joint_residual_normalized_log"

    def __init__(self, config: JointModelConfig, *, residual_bound=1.0):
        super().__init__()
        if (
            "H" not in config.modalities
            or config.transfer_only
            or config.assignment_mode != "learned"
            or set(config.gene_ids) != set(config.supervised_gene_ids)
        ):
            raise ValueError(
                "residual family requires H, learned mapping and fully supervised outputs"
            )
        if not np.isfinite(residual_bound) or residual_bound <= 0:
            raise ValueError("normalized-log residual bound must be finite and positive")
        self.config, self.residual_bound = config, float(residual_bound)
        self.mapper = JointModel(config)
        # Legacy direct and multiplicative decoders have no role in this family.
        for module in (self.mapper.direct_head, self.mapper.correction):
            if module is not None:
                module.requires_grad_(False)
        if self.mapper.correction_gene is not None:
            self.mapper.correction_gene.requires_grad_(False)
        if "R" not in config.modalities:
            self.mapper.fusion.requires_grad_(False)
        width = config.hidden_dim
        self.residual_head = nn.Sequential(
            nn.Linear(4 * width + len(config.gene_ids), width),
            nn.Tanh(),
            nn.Linear(width, len(config.gene_ids)),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, inputs, baseline_values, *, residual_enabled=True):
        if not isinstance(residual_enabled, bool):
            raise ValueError("residual_enabled must be boolean")
        prediction, diagnostics = self.mapper._forward(inputs, correction_enabled=False)
        h, p = diagnostics["image_embedding"], diagnostics["protein_embedding"]
        n, g = len(inputs.cell_ids), len(self.config.gene_ids)
        baseline = torch.as_tensor(baseline_values, device=h.device)
        if (
            baseline.dtype != torch.float64
            or baseline.shape != (n, g)
            or not torch.isfinite(baseline).all()
        ):
            raise ValueError(
                "frozen baseline must be finite float64 on the region/output gene axes"
            )
        transfer = None
        u, global_rna = torch.zeros_like(h), torch.zeros_like(h)
        normalized_transfer = h.new_zeros((n, g))
        if "R" in self.config.modalities:
            transfer = prediction
            u = diagnostics["assigned_rna_embedding"]
            global_rna = diagnostics["specimen_rna_embedding"]
            indices = [
                self.config.gene_ids.index(gene) for gene in self.config.normalization_gene_ids
            ]
            normalized_transfer = normalized_log_expression(
                prediction.transfer,
                torch.ones_like(prediction.transfer, dtype=torch.bool),
                normalization_indices=indices,
            )
        features = torch.cat((h, p, u, normalized_transfer, global_rna), dim=1)
        delta = (
            self.residual_bound * self.residual_head(features).tanh()
            if residual_enabled
            else h.new_zeros((n, g))
        ).to(torch.float64)
        final = baseline + delta
        diagnostics.update(
            diagnostic_schema="heir.joint_residual.diagnostics.v1",
            residual_input=features,
            normalized_reference_transfer=normalized_transfer,
            image_baseline=baseline,
            normalized_log_residual=delta,
            normalized_log_expression=final,
            residual_enabled=residual_enabled,
        )
        return ResidualPrediction(
            inputs.cell_ids,
            self.config.gene_ids,
            torch.as_tensor(np.asarray(inputs.coordinates_um), device=h.device),
            inputs.patient_id,
            inputs.specimen_id,
            inputs.prediction_unit,
            inputs.coordinate_units,
            final,
            baseline,
            delta,
            transfer,
            diagnostics,
        )

    def active_trainable_parameter_count(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        width = self.config.hidden_dim
        # Structurally absent feature columns are excluded from effective capacity.
        if "R" not in self.config.modalities:
            count -= (2 * width + len(self.config.gene_ids)) * width
        if "P" not in self.config.modalities:
            count -= width * width
            if "R" in self.config.modalities:
                count -= 2 * width * width  # fusion's p and h*p columns
        if self.config.protein_signal == "coverage_only":
            count -= 2 * len(self.config.marker_ids) * width
        return count
