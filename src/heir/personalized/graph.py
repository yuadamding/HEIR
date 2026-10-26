"""Sparse physical neighborhoods and candidate ligand/receptor compatibility.

Messages are compatibility features, never measurements of signaling activity.
Both generic and LR branches use the same edges, program genes, and normalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from .schema import RNAReference, SpatialGraph


@dataclass(frozen=True)
class LRInteraction:
    """A directed, complex-aware candidate from a separately recorded resource."""

    ligand: tuple[str, ...]
    receptor: tuple[str, ...]
    mode: str = "diffusible"
    radius_um: float = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ligand", tuple(self.ligand))
        object.__setattr__(self, "receptor", tuple(self.receptor))
        if not self.ligand or not self.receptor:
            raise ValueError("ligand and receptor complexes must contain gene IDs")
        for members in (self.ligand, self.receptor):
            if len(members) != len(set(members)) or any(not gene for gene in members):
                raise ValueError("complex subunits must have unique nonempty gene IDs")
        if self.mode not in {"contact", "diffusible"}:
            raise ValueError("interaction mode must be contact or diffusible")
        if not np.isfinite(self.radius_um) or self.radius_um <= 0:
            raise ValueError("interaction radius_um must be finite and positive")


def build_graph(
    coordinates_um: np.ndarray,
    radius_um: float,
    *,
    tissue_component_ids: Sequence[str],
    max_neighbors: int = 32,
    edge_allowed: Callable[[int, int], bool] | None = None,
) -> SpatialGraph:
    """Build directed radius/k-nearest edges with bounded degree using grid buckets.

    Components prohibit edges across disconnected tissue. For gaps or boundaries
    *within* a component, supply an ``edge_allowed(sender, receiver)`` predicate
    derived from the tissue mask; coordinates alone cannot establish tissue support.
    Equal-distance ties use stable input indices. No dense all-pairs matrix is built.
    """

    xy = np.asarray(coordinates_um, dtype=float)
    components = tuple(tissue_component_ids)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("coordinates_um must be a finite N x 2 matrix")
    if len(components) != len(xy) or any(not str(item).strip() for item in components):
        raise ValueError("one nonempty tissue component ID is required for each nucleus")
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("radius_um must be finite and positive")
    if isinstance(max_neighbors, bool) or int(max_neighbors) != max_neighbors or max_neighbors < 1:
        raise ValueError("max_neighbors must be a positive integer")
    buckets: dict[tuple[str, int, int], list[int]] = {}
    grid = np.floor(xy / radius_um).astype(np.int64)
    for index, (x, y) in enumerate(grid):
        buckets.setdefault((components[index], int(x), int(y)), []).append(index)
    senders, receivers, distances, weights = [], [], [], []
    for receiver, (x, y) in enumerate(grid):
        candidates: list[tuple[float, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for sender in buckets.get((components[receiver], int(x + dx), int(y + dy)), ()):
                    if sender == receiver:
                        continue
                    distance = float(np.linalg.norm(xy[sender] - xy[receiver]))
                    if distance <= radius_um and (
                        edge_allowed is None or edge_allowed(sender, receiver)
                    ):
                        candidates.append((distance, sender))
        for distance, sender in sorted(candidates)[:max_neighbors]:
            senders.append(sender)
            receivers.append(receiver)
            distances.append(distance)
            weights.append(math.exp(-distance / radius_um))
    return SpatialGraph(
        senders=np.asarray(senders, dtype=np.int64),
        receivers=np.asarray(receivers, dtype=np.int64),
        distances_um=np.asarray(distances, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
        n_cells=len(xy),
    )


def neighborhood_average(
    values: torch.Tensor,
    graph: SpatialGraph,
    *,
    include_self: bool = False,
    edge_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Incoming edge-weighted mean; zero for a receiver with no permitted edges."""

    if values.ndim != 2 or values.shape[0] != graph.n_cells:
        raise ValueError("graph and value rows differ")
    senders = torch.as_tensor(graph.senders, device=values.device, dtype=torch.long)
    receivers = torch.as_tensor(graph.receivers, device=values.device, dtype=torch.long)
    weights = torch.as_tensor(graph.weights, device=values.device, dtype=values.dtype)
    if edge_mask is not None:
        weights = weights * edge_mask.to(device=values.device, dtype=values.dtype)
    output = values.clone() if include_self else torch.zeros_like(values)
    total = torch.ones_like(values[:, :1]) if include_self else torch.zeros_like(values[:, :1])
    output.index_add_(0, receivers, values[senders] * weights[:, None])
    total.index_add_(0, receivers, weights[:, None])
    return output / total.clamp_min(torch.finfo(values.dtype).tiny)


def _complex_support(
    expression: torch.Tensor, gene_index: dict[str, int], genes: tuple[str, ...]
) -> torch.Tensor:
    # A missing subunit invalidates the complete complex. Never silently retain
    # a partial complex or replace a missing subunit by an observed one.
    if any(gene not in gene_index for gene in genes):
        return expression.new_zeros(expression.shape[0])
    return expression[:, [gene_index[gene] for gene in genes]].amin(dim=1)


def compute_messages(
    graph: SpatialGraph,
    assignment: torch.Tensor,
    reference: RNAReference,
    interactions: Sequence[LRInteraction],
    *,
    mode: str = "lr",
) -> torch.Tensor:
    """Return N x 2P messages, with identical input features in both branches.

    ``assignment`` excludes the unsupported component. For each program, the
    generic branch has (mean incoming ligand, receiver receptor); the LR branch
    replaces the first feature by their directed product. Means use the same
    candidate-specific radius and edge weights. Contact candidates must carry
    a suitable explicitly configured contact radius; proximity is not a claim
    of verified membrane contact. Missing genes contribute zero complex support.
    """

    if mode not in {"lr", "generic"}:
        raise ValueError("message mode must be lr or generic")
    if assignment.shape != (graph.n_cells, len(reference.state_ids)):
        raise ValueError("assignment and graph/reference dimensions differ")
    if not interactions:
        return assignment.new_zeros((graph.n_cells, 0))
    expression = torch.as_tensor(
        reference.expression, device=assignment.device, dtype=assignment.dtype
    )
    gene_index = {gene: index for index, gene in enumerate(reference.gene_ids)}
    distances = torch.as_tensor(graph.distances_um, device=assignment.device)
    outputs = []
    for interaction in interactions:
        ligand = assignment @ _complex_support(expression, gene_index, interaction.ligand)
        receptor = assignment @ _complex_support(expression, gene_index, interaction.receptor)
        incoming = neighborhood_average(
            ligand[:, None], graph, edge_mask=distances <= interaction.radius_um
        )[:, 0]
        first = incoming * receptor if mode == "lr" else incoming
        outputs.extend((first, receptor))
    return torch.stack(outputs, dim=1)
