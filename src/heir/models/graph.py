"""Native-PyTorch neighborhood context for segmented histology cells."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor, nn


def validate_edge_index(edge_index: Tensor, num_nodes: int, device: torch.device) -> None:
    """Validate a directed COO ``[source, target]`` edge tensor."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long")
    if edge_index.device != device:
        raise ValueError("edge_index and node features must share a device")
    if edge_index.numel() == 0:
        return
    if num_nodes <= 0 or bool((edge_index < 0).any()) or int(edge_index.max()) >= num_nodes:
        raise ValueError("edge_index contains an invalid node index")


def validate_edge_weight(edge_weight: Optional[Tensor], edge_index: Tensor) -> None:
    """Validate optional nonnegative edge weights."""

    if edge_weight is None:
        return
    if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weight must have one value per edge")
    if edge_weight.device != edge_index.device:
        raise ValueError("edge_weight and edge_index must share a device")
    if not torch.is_floating_point(edge_weight):
        raise TypeError("edge_weight must be floating point")
    if not torch.isfinite(edge_weight).all() or bool((edge_weight < 0).any()):
        raise ValueError("edge_weight must contain finite nonnegative values")


@dataclass(frozen=True)
class GraphContextConfig:
    """Configuration for :class:`GraphContextEncoder`."""

    input_dim: int
    hidden_dim: int = 128
    output_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    normalize_messages: bool = True
    residual: bool = True

    def __post_init__(self) -> None:
        dimensions = (self.input_dim, self.hidden_dim, self.output_dim, self.num_layers)
        if any(value <= 0 for value in dimensions):
            raise ValueError("graph dimensions and num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        """Return checkpoint-safe metadata."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GraphContextConfig":
        """Reconstruct a config from metadata."""

        return cls(**dict(values))


class GraphMessageLayer(nn.Module):
    """Aggregate incoming messages with ``index_add`` and a residual update."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        normalize_messages: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_messages = normalize_messages
        self.use_residual = residual
        self.self_projection = nn.Linear(input_dim, output_dim)
        self.neighbor_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.normalization = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if residual and input_dim != output_dim:
            self.residual_projection: nn.Module = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.residual_projection = nn.Identity()

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Return one updated row for every input node."""

        if node_features.ndim != 2 or node_features.shape[1] != self.input_dim:
            raise ValueError("node_features has the wrong shape")
        if not torch.is_floating_point(node_features):
            raise TypeError("node_features must be floating point")
        validate_edge_index(edge_index, node_features.shape[0], node_features.device)
        validate_edge_weight(edge_weight, edge_index)
        source, target = edge_index
        # The projection is linear and bias-free, so project each nucleus once
        # before expanding rows over edges.  On WSI graphs this avoids applying
        # the same 1546->128 projection roughly one time per neighbor.
        projected_neighbors = self.neighbor_projection(node_features)
        messages = projected_neighbors.index_select(0, source)
        if edge_weight is None:
            weights = node_features.new_ones(edge_index.shape[1])
        else:
            weights = edge_weight.to(dtype=node_features.dtype)
            messages = messages * weights.unsqueeze(-1)
        aggregate = node_features.new_zeros((node_features.shape[0], self.output_dim))
        aggregate = aggregate.index_add(0, target, messages)
        if self.normalize_messages:
            degree = node_features.new_zeros(node_features.shape[0])
            degree = degree.index_add(0, target, weights)
            aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
        output = self.self_projection(node_features) + aggregate
        output = self.dropout(self.activation(self.normalization(output)))
        if self.use_residual:
            output = output + self.residual_projection(node_features)
        return output


class GraphContextEncoder(nn.Module):
    """Stack graph message layers without a torch-geometric dependency."""

    def __init__(self, config: GraphContextConfig) -> None:
        super().__init__()
        self.config = config
        dimensions = [config.input_dim]
        dimensions.extend([config.hidden_dim] * max(config.num_layers - 1, 0))
        dimensions.append(config.output_dim)
        self.layers = nn.ModuleList(
            GraphMessageLayer(
                dimensions[index],
                dimensions[index + 1],
                config.dropout,
                config.normalize_messages,
                config.residual,
            )
            for index in range(config.num_layers)
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode local cell context."""

        output = node_features
        for layer in self.layers:
            output = layer(output, edge_index, edge_weight)
        return output


GraphContext = GraphContextEncoder


__all__ = [
    "validate_edge_index",
    "validate_edge_weight",
    "GraphContextConfig",
    "GraphMessageLayer",
    "GraphContextEncoder",
    "GraphContext",
]
