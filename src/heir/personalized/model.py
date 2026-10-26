"""One prediction-only pathway for the personalized development comparisons.

This small implementation is an engineering baseline, not a qualified biological
model. Unsupported mass and entropy are diagnostics, not calibrated error bars.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .graph import LRInteraction, compute_messages, neighborhood_average
from .schema import Prediction, PredictionInputs, SpatialGraph

VARIANTS = (
    "he",
    "he_rna",
    "he_rna_protein",
    "generic_fixed",
    "lr_fixed",
    "generic_refined",
    "lr_refined",
)


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int
    state_dim: int
    gene_ids: tuple[str, ...]
    supervised_gene_ids: tuple[str, ...]
    state_gene_ids: tuple[str, ...] = ()
    hidden_dim: int = 32
    residual_rank: int = 16
    variant: str = "he_rna"
    residual_bound: float = 1.0
    refinement_steps: int = 1
    refinement_bound: float = 0.25
    transfer_only: bool = False
    lr_interactions: tuple[LRInteraction, ...] = ()

    def __post_init__(self) -> None:
        for name in ("gene_ids", "supervised_gene_ids", "state_gene_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        interactions = tuple(
            LRInteraction(**item) if isinstance(item, dict) else item
            for item in self.lr_interactions
        )
        object.__setattr__(self, "lr_interactions", interactions)
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if not self.gene_ids or len(set(self.gene_ids)) != len(self.gene_ids):
            raise ValueError("model gene_ids must be nonempty and unique")
        if any(not isinstance(gene, str) or not gene.strip() for gene in self.gene_ids):
            raise ValueError("model genes require nonempty stable IDs")
        if len(set(self.supervised_gene_ids)) != len(self.supervised_gene_ids):
            raise ValueError("supervised gene IDs must be unique")
        if not set(self.supervised_gene_ids) <= set(self.gene_ids):
            raise ValueError("supervised genes must belong to the output gene axis")
        for name in ("feature_dim", "state_dim", "hidden_dim", "residual_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.variant != "he" or self.state_gene_ids:
            if len(self.state_gene_ids) != self.state_dim:
                raise ValueError("state_gene_ids must identify every RNA embedding feature")
            if len(set(self.state_gene_ids)) != len(self.state_gene_ids) or any(
                not isinstance(gene, str) or not gene.strip() for gene in self.state_gene_ids
            ):
                raise ValueError("state_gene_ids must contain unique nonempty gene IDs")
        if self.refinement_steps not in {1, 2}:
            raise ValueError("the initial refinement implementation supports one or two steps")
        for name in ("residual_bound", "refinement_bound"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.variant.startswith(("generic_", "lr_")) and not self.lr_interactions:
            raise ValueError("both context branches require the same explicit molecular programs")
        if self.variant == "he" and self.transfer_only:
            raise ValueError("H&E-only has no reference transfer diagnostic")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        return cls(**dict(values))


def _local_transpose(values: torch.Tensor, graph: SpatialGraph) -> torch.Tensor:
    """Apply the transpose of the include-self incoming average operator."""

    senders = torch.as_tensor(graph.senders, device=values.device, dtype=torch.long)
    receivers = torch.as_tensor(graph.receivers, device=values.device, dtype=torch.long)
    weights = torch.as_tensor(graph.weights, device=values.device, dtype=values.dtype)
    total = torch.ones_like(values[:, :1])
    total.index_add_(0, receivers, weights[:, None])
    normalized = values / total
    output = normalized.clone()
    output.index_add_(0, senders, normalized[receivers] * weights[:, None])
    return output


class PersonalizedModel(nn.Module):
    """Trainable morphology compatibility, optional context, and bounded decoder.

    The state network operates on molecular features, never numeric state IDs.
    Reference and output gene axes are joined by ID on every forward pass.
    Context ``*_fixed`` means no assignment refinement; training that comparison
    must additionally call ``initialize_fixed_mapper`` with the fitted protein
    baseline. The training entrypoint enforces that staged requirement.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.image_mapper = nn.Sequential(nn.Linear(config.feature_dim, h), nn.Tanh())
        self.state_mapper = nn.Sequential(nn.Linear(config.state_dim, h), nn.Tanh())
        self.unsupported_head = nn.Linear(h, 1)
        self.protein_log_strength = nn.Parameter(torch.tensor(0.0))
        self.decoder_image = nn.Sequential(nn.Linear(config.feature_dim, h), nn.Tanh())
        self.baseline_head = nn.Linear(h, len(config.gene_ids))
        message_dim = 2 * len(config.lr_interactions)
        self.residual_cell = nn.Linear(h + message_dim, config.residual_rank)
        self.residual_gene = nn.Parameter(torch.empty(len(config.gene_ids), config.residual_rank))
        nn.init.normal_(self.residual_gene, std=0.02)
        self.refinement_message = nn.Linear(message_dim, h, bias=False) if message_dim else None
        self.fixed_mapper_initialized = False
        self.register_buffer(
            "supervised_mask",
            torch.tensor([gene in config.supervised_gene_ids for gene in config.gene_ids]),
        )

    def initialize_fixed_mapper(self, baseline: "PersonalizedModel") -> None:
        """Copy and freeze the fitted protein mapper for equal-input context arms."""

        if self.config.variant not in {
            "generic_fixed",
            "lr_fixed",
            "generic_refined",
            "lr_refined",
        }:
            raise ValueError("staged mapper initialization is only for context configurations")
        if baseline.config.variant != "he_rna_protein":
            raise ValueError("context initialization requires a fitted he_rna_protein baseline")
        for name in ("feature_dim", "state_dim", "state_gene_ids", "hidden_dim"):
            if getattr(self.config, name) != getattr(baseline.config, name):
                raise ValueError(f"baseline {name} differs")
        for name in ("image_mapper", "state_mapper", "unsupported_head"):
            module = getattr(self, name)
            module.load_state_dict(getattr(baseline, name).state_dict())
        with torch.no_grad():
            self.protein_log_strength.copy_(baseline.protein_log_strength)
        self.freeze_mapper()
        self.fixed_mapper_initialized = True

    def freeze_mapper(self) -> None:
        """Restore the stage boundary after loading already initialized weights."""

        for name in ("image_mapper", "state_mapper", "unsupported_head"):
            for parameter in getattr(self, name).parameters():
                parameter.requires_grad_(False)
        self.protein_log_strength.requires_grad_(False)

    @property
    def uses_proteins(self) -> bool:
        return self.config.variant not in {"he", "he_rna"}

    def _tensor(self, value: Any) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)

    def _protein_arrays(self, inputs: PredictionInputs):
        protein = inputs.proteins
        if protein is None:
            return None
        mask = self._tensor(protein.observed)
        # Fill before subtraction: NaN * 0 is still NaN.
        values = self._tensor(np.where(protein.observed, protein.values, 0.0))
        if protein.quality is not None:
            mask = mask * self._tensor(protein.quality)
        return values, mask, self._tensor(protein.state_values)

    def _protein_logits(self, logits: torch.Tensor, inputs: PredictionInputs) -> torch.Tensor:
        arrays = self._protein_arrays(inputs)
        if arrays is None or not self.uses_proteins:
            return logits
        values, mask, states = arrays
        strength = self.protein_log_strength.exp().clamp(max=10.0)
        if inputs.proteins.correspondence == "same_cell":
            # Expand the quadratic algebraically, avoiding an N x K x M tensor.
            cost = (values.square() * mask).sum(dim=1, keepdim=True)
            cost = cost + mask @ states.square().T - 2 * (values * mask) @ states.T
            cost = cost.clamp_min(0) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            adjustment = -(strength * cost).clamp(max=5.0)
        else:
            if inputs.graph is None:
                raise ValueError("soft/serial/region protein evidence requires a spatial graph")
            q = logits.softmax(dim=1)[:, 1:]
            mixture = neighborhood_average(q @ states, inputs.graph, include_self=True)
            discrepancy = (mixture - values) * mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            cost = _local_transpose(discrepancy, inputs.graph) @ states.T
            adjustment = (-strength * cost).clamp(-1.0, 1.0)
        # Unsupported has no fabricated state marker template.
        return logits + torch.cat((torch.zeros_like(logits[:, :1]), adjustment), dim=1)

    def protein_loss(self, inputs: PredictionInputs, prediction: Prediction) -> torch.Tensor:
        """Masked local-mixture protein discrepancy; RNA-to-protein scale is upstream."""

        arrays = self._protein_arrays(inputs)
        if arrays is None or not self.uses_proteins:
            return prediction.expression.sum() * 0.0
        values, mask, states = arrays
        mixture = prediction.assignment[:, 1:] @ states
        if inputs.proteins.correspondence != "same_cell":
            if inputs.graph is None:
                raise ValueError("soft/serial/region protein evidence requires a spatial graph")
            mixture = neighborhood_average(mixture, inputs.graph, include_self=True)
        return ((mixture - values).square() * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, inputs: PredictionInputs) -> Prediction:
        """No target-ST argument is accepted during either training or inference."""

        config = self.config
        x = self._tensor(inputs.image_features)
        if x.shape[1] != config.feature_dim:
            raise ValueError("cached image feature dimension differs from model bundle")
        n = len(inputs.cell_ids)
        registration = (
            self._tensor(inputs.registration_sensitivity)
            if inputs.registration_sensitivity is not None
            else x.new_full((n,), float("nan"))
        )
        decoder_features = self.decoder_image(x)
        if config.variant == "he":
            # Neither RNA values, state IDs, nor proteins enter the H&E baseline.
            assignment = x.new_ones((n, 1))
            transfer = x.new_zeros((n, len(config.gene_ids)))
            delta = torch.zeros_like(transfer)
            expression = F.softplus(self.baseline_head(decoder_features)) * self.supervised_mask
            state_ids = ("unsupported",)
        else:
            reference = inputs.reference
            embedding_index = {
                gene: index for index, gene in enumerate(reference.embedding_gene_ids)
            }
            missing_embedding = set(config.state_gene_ids) - set(embedding_index)
            if missing_embedding:
                raise ValueError(
                    "RNA embedding panel differs from model bundle; missing genes: "
                    f"{sorted(missing_embedding)}"
                )
            gene_index = {gene: index for index, gene in enumerate(reference.gene_ids)}
            missing = set(config.gene_ids) - set(gene_index)
            if missing:
                raise ValueError(
                    f"requested output genes missing from reference: {sorted(missing)}"
                )
            ordered_embedding = self._tensor(reference.embedding)[
                :, [embedding_index[gene] for gene in config.state_gene_ids]
            ]
            state_features = self.state_mapper(ordered_embedding)
            image_features = self.image_mapper(x)
            compatible = image_features @ state_features.T / config.hidden_dim**0.5
            logits = torch.cat((self.unsupported_head(image_features), compatible), dim=1)
            logits = self._protein_logits(logits, inputs)
            assignment = logits.softmax(dim=1)
            context = config.variant.startswith(("generic_", "lr_"))
            messages = x.new_zeros((n, 2 * len(config.lr_interactions)))
            if context:
                if inputs.graph is None:
                    raise ValueError("context configurations require a spatial graph")
                mode = "lr" if config.variant.startswith("lr_") else "generic"
                if config.variant.endswith("_refined"):
                    for _ in range(config.refinement_steps):
                        messages = compute_messages(
                            inputs.graph,
                            assignment[:, 1:],
                            reference,
                            config.lr_interactions,
                            mode=mode,
                        )
                        # Log1p bounds large expression products before the projection.
                        update_features = self.refinement_message(torch.log1p(messages))
                        update = update_features @ state_features.T / config.hidden_dim**0.5
                        update = config.refinement_bound * torch.tanh(update)
                        logits = logits + torch.cat(
                            (torch.zeros_like(logits[:, :1]), update), dim=1
                        )
                        assignment = logits.softmax(dim=1)
                messages = compute_messages(
                    inputs.graph,
                    assignment[:, 1:],
                    reference,
                    config.lr_interactions,
                    mode=mode,
                )
            ordered_expression = self._tensor(reference.expression)[
                :, [gene_index[gene] for gene in config.gene_ids]
            ]
            transfer = assignment[:, 1:] @ ordered_expression
            factors = self.residual_cell(
                torch.cat((decoder_features, torch.log1p(messages)), dim=1)
            )
            delta = (factors @ self.residual_gene.T).clamp(
                -config.residual_bound, config.residual_bound
            )
            delta = delta * self.supervised_mask
            if config.transfer_only:
                delta = torch.zeros_like(delta)
            # Exact transfer for unsupervised genes and the transfer-only diagnostic.
            # Epsilon is added only for corrected genes, never to untrained outputs.
            corrected = (transfer + 1e-8) * torch.exp(delta)
            expression = torch.where(self.supervised_mask, corrected, transfer)
            if config.transfer_only:
                expression = transfer
            state_ids = ("unsupported", *reference.state_ids)
        entropy = -(assignment * assignment.clamp_min(1e-12).log()).sum(dim=1)
        return Prediction(
            cell_ids=inputs.cell_ids,
            gene_ids=config.gene_ids,
            state_ids=state_ids,
            coordinates_um=self._tensor(inputs.coordinates_um),
            assignment=assignment,
            transfer=transfer,
            log_correction=delta,
            expression=expression,
            assignment_entropy=entropy,
            reference_support=1 - assignment[:, 0],
            registration_sensitivity=registration,
            supervised_genes=self.supervised_mask.clone(),
            independently_evaluated_genes=torch.zeros_like(self.supervised_mask),
            specimen_id=inputs.specimen_id,
            patient_id=inputs.patient_id,
        )
