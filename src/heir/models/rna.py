"""RNA variational prior and transferable decoder for HEIR."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_dims(values: Sequence[int], name: str) -> Tuple[int, ...]:
    dims = tuple(int(value) for value in values)
    if not dims or any(value <= 0 for value in dims):
        raise ValueError("%s must contain positive layer widths" % name)
    return dims


def _hidden_stack(input_dim: int, widths: Sequence[int], dropout: float) -> nn.Sequential:
    layers = []
    previous = input_dim
    for width in widths:
        layers.extend(
            [
                nn.Linear(previous, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        previous = width
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class RNAVAEConfig:
    """Checkpoint-safe RNA VAE architecture configuration."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, ...] = (256, 128)
    decoder_hidden_dims: Optional[Tuple[int, ...]] = None
    dropout: float = 0.1
    logvar_min: float = -12.0
    logvar_max: float = 8.0
    nonnegative_output: bool = False

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        object.__setattr__(self, "hidden_dims", _positive_dims(self.hidden_dims, "hidden_dims"))
        if self.decoder_hidden_dims is not None:
            object.__setattr__(
                self,
                "decoder_hidden_dims",
                _positive_dims(self.decoder_hidden_dims, "decoder_hidden_dims"),
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.logvar_min >= self.logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")

    def to_dict(self) -> Dict[str, Any]:
        """Return metadata containing only standard Python types."""

        result = asdict(self)
        result["hidden_dims"] = list(self.hidden_dims)
        if self.decoder_hidden_dims is not None:
            result["decoder_hidden_dims"] = list(self.decoder_hidden_dims)
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RNAVAEConfig":
        """Reconstruct a config from checkpoint metadata."""

        data = dict(values)
        if "hidden_dims" in data:
            data["hidden_dims"] = tuple(data["hidden_dims"])
        if data.get("decoder_hidden_dims") is not None:
            data["decoder_hidden_dims"] = tuple(data["decoder_hidden_dims"])
        return cls(**data)


class RNAEncoder(nn.Module):
    """Encode normalized expression into a diagonal Gaussian posterior."""

    def __init__(self, config: RNAVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _hidden_stack(config.input_dim, config.hidden_dims, config.dropout)
        self.mu_head = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dims[-1], config.latent_dim)

    def forward(self, expression: Tensor) -> Tuple[Tensor, Tensor]:
        """Return posterior mean and bounded log variance."""

        if not torch.is_floating_point(expression):
            raise TypeError("expression must be floating point")
        if expression.ndim < 2 or expression.shape[-1] != self.config.input_dim:
            raise ValueError("expression has the wrong final dimension")
        hidden = self.backbone(expression)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden).clamp(
            min=self.config.logvar_min,
            max=self.config.logvar_max,
        )
        return mu, logvar


class RNADecoder(nn.Module):
    """Decode an RNA latent into genes or gene-program scores."""

    def __init__(self, config: RNAVAEConfig) -> None:
        super().__init__()
        self.config = config
        widths = config.decoder_hidden_dims or tuple(reversed(config.hidden_dims))
        self.backbone = _hidden_stack(config.latent_dim, widths, config.dropout)
        self.output_layer = nn.Linear(widths[-1], config.input_dim)

    def forward(self, latent: Tensor) -> Tensor:
        """Decode a tensor whose final dimension is ``latent_dim``."""

        if not torch.is_floating_point(latent):
            raise TypeError("latent must be floating point")
        if latent.ndim < 2 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError("latent has the wrong final dimension")
        output = self.output_layer(self.backbone(latent))
        return F.softplus(output) if self.config.nonnegative_output else output


@dataclass
class RNAVAEOutput:
    """RNA VAE reconstruction and posterior tensors."""

    reconstruction: Tensor
    latent: Tensor
    mu: Tensor
    logvar: Tensor


class RNAVAE(nn.Module):
    """Compact RNA VAE used to construct a sample-specific molecular manifold."""

    def __init__(self, config: RNAVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = RNAEncoder(config)
        self.decoder = RNADecoder(config)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor, sample: bool = True) -> Tensor:
        """Sample a diagonal Gaussian with the reparameterization trick."""

        if mu.shape != logvar.shape:
            raise ValueError("mu and logvar must have identical shapes")
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(self, expression: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode expression into posterior parameters."""

        return self.encoder(expression)

    def decode(self, latent: Tensor) -> Tensor:
        """Decode an RNA latent."""

        return self.decoder(latent)

    def forward(self, expression: Tensor, sample: Optional[bool] = None) -> RNAVAEOutput:
        """Reconstruct expression, sampling only during training by default."""

        mu, logvar = self.encode(expression)
        latent = self.reparameterize(mu, logvar, self.training if sample is None else sample)
        return RNAVAEOutput(self.decode(latent), latent, mu, logvar)

    @staticmethod
    def kl_divergence(mu: Tensor, logvar: Tensor, reduction: str = "mean") -> Tensor:
        """KL divergence to a unit Gaussian."""

        if mu.shape != logvar.shape:
            raise ValueError("mu and logvar must have identical shapes")
        values = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1)
        if reduction == "none":
            return values
        if reduction == "sum":
            return values.sum()
        if reduction == "mean":
            return values.mean() if values.numel() else mu.sum() * 0.0
        raise ValueError("reduction must be none, sum, or mean")

    def freeze_decoder(self, freeze: bool = True) -> None:
        """Freeze or unfreeze decoder parameters."""

        for parameter in self.decoder.parameters():
            parameter.requires_grad_(not freeze)

    def checkpoint(self) -> Dict[str, Any]:
        """Create a self-describing in-memory checkpoint."""

        return {"config": self.config.to_dict(), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        strict: bool = True,
    ) -> "RNAVAE":
        """Reconstruct a VAE from :meth:`checkpoint` output."""

        if "config" not in checkpoint or "state_dict" not in checkpoint:
            raise KeyError("checkpoint must contain config and state_dict")
        model = cls(RNAVAEConfig.from_dict(checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        return model


RNAConfig = RNAVAEConfig


__all__ = [
    "RNAConfig",
    "RNAVAEConfig",
    "RNAEncoder",
    "RNADecoder",
    "RNAVAEOutput",
    "RNAVAE",
]
