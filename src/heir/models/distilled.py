"""H&E-only student kept separate from personalized HEIR reporting."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .graph import GraphContextConfig, GraphContextEncoder
from .rna import RNADecoder, RNAVAEConfig


@dataclass(frozen=True)
class DistilledConfig:
    morphology_dim: int
    num_cell_types: int
    expression_dim: int
    latent_dim: int = 32
    hidden_dim: int = 128
    graph_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if (
            min(
                self.morphology_dim,
                self.num_cell_types,
                self.expression_dim,
                self.latent_dim,
                self.hidden_dim,
                self.graph_layers,
            )
            <= 0
        ):
            raise ValueError("distilled dimensions must be positive")
        if self.num_cell_types < 2 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("distilled model requires >=2 types and dropout in [0, 1)")


@dataclass
class DistilledOutput:
    type_logits: Tensor
    type_probabilities: Tensor
    latent_mean: Tensor
    latent_logvar: Tensor
    expression: Tensor
    unknown_probability: Tensor


class DistilledHEIR(nn.Module):
    """Graph student that does not receive sample-specific RNA prototypes."""

    def __init__(self, config: DistilledConfig) -> None:
        super().__init__()
        self.config = config
        graph_config = GraphContextConfig(
            input_dim=config.morphology_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.graph_layers,
            dropout=config.dropout,
        )
        self.graph = GraphContextEncoder(graph_config)
        self.trunk = nn.Sequential(
            nn.Linear(config.morphology_dim + config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.type_head = nn.Linear(config.hidden_dim, config.num_cell_types)
        self.latent_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.latent_logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        self.unknown_head = nn.Linear(config.hidden_dim, 1)
        self.decoder = RNADecoder(
            RNAVAEConfig(
                input_dim=config.expression_dim,
                latent_dim=config.latent_dim,
                hidden_dims=(config.hidden_dim,),
                decoder_hidden_dims=(config.hidden_dim,),
                dropout=config.dropout,
            )
        )

    def forward(
        self,
        morphology: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        sample_latent: Optional[bool] = None,
    ) -> DistilledOutput:
        if morphology.ndim != 2 or morphology.shape[1] != self.config.morphology_dim:
            raise ValueError("morphology has the wrong shape")
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=morphology.device)
        context = self.graph(morphology, edge_index, edge_weight)
        hidden = self.trunk(torch.cat((morphology, context), dim=-1))
        logits = self.type_head(hidden)
        mu = self.latent_mu(hidden)
        logvar = self.latent_logvar(hidden).clamp(-12.0, 8.0)
        sample = self.training if sample_latent is None else sample_latent
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return DistilledOutput(
            type_logits=logits,
            type_probabilities=torch.softmax(logits, dim=-1),
            latent_mean=mu,
            latent_logvar=logvar,
            expression=self.decoder(latent),
            unknown_probability=torch.sigmoid(self.unknown_head(hidden)).squeeze(-1),
        )

    def checkpoint(self) -> Dict[str, Any]:
        return {"config": asdict(self.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, values: Mapping[str, Any]) -> "DistilledHEIR":
        model = cls(DistilledConfig(**values["config"]))
        model.load_state_dict(values["state_dict"])
        return model


def uncertainty_weighted_distillation_loss(
    student: DistilledOutput,
    teacher_type_probabilities: Tensor,
    teacher_latent_mean: Tensor,
    teacher_expression: Tensor,
    teacher_uncertainty: Tensor,
    teacher_unknown: Optional[Tensor] = None,
    temperature: float = 2.0,
) -> Dict[str, Tensor]:
    """Distill soft states while downweighting unreliable personalized calls."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    count = student.type_logits.shape[0]
    if teacher_type_probabilities.shape != student.type_probabilities.shape:
        raise ValueError("teacher type probabilities are misaligned")
    if teacher_latent_mean.shape != student.latent_mean.shape:
        raise ValueError("teacher latent means are misaligned")
    if teacher_expression.shape != student.expression.shape:
        raise ValueError("teacher expression is misaligned")
    if teacher_uncertainty.shape != (count,):
        raise ValueError("teacher uncertainty must have one value per cell")
    reliability = (1.0 - teacher_uncertainty).clamp(0.0, 1.0).detach()
    target = teacher_type_probabilities.clamp_min(1.0e-8)
    type_per_cell = F.kl_div(
        F.log_softmax(student.type_logits / temperature, dim=-1),
        target,
        reduction="none",
    ).sum(dim=-1) * (temperature**2)
    latent_per_cell = (student.latent_mean - teacher_latent_mean).square().mean(dim=-1)
    expression_per_cell = F.huber_loss(
        student.expression,
        teacher_expression,
        reduction="none",
    ).mean(dim=-1)
    denominator = reliability.sum().clamp_min(1.0e-8)
    losses = {
        "type": (type_per_cell * reliability).sum() / denominator,
        "latent": (latent_per_cell * reliability).sum() / denominator,
        "expression": (expression_per_cell * reliability).sum() / denominator,
    }
    if teacher_unknown is not None:
        if teacher_unknown.shape != (count,):
            raise ValueError("teacher_unknown must have one value per cell")
        losses["unknown"] = F.binary_cross_entropy(
            student.unknown_probability,
            teacher_unknown.detach(),
        )
    losses["total"] = sum(losses.values())
    return losses
