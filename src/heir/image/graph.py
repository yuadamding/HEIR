"""Sparse spatial graphs for nucleus-level morphology models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

EDGE_FEATURE_NAMES = ("distance", "dx", "dy", "angle")


def _coordinates(values: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (nodes, 2)")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must contain only finite values")
    return coordinates


def _ckdtree():
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "spatial graph construction requires scipy; install the spatial dependencies"
        ) from error
    return cKDTree


@dataclass(frozen=True)
class SpatialGraph:
    """COO graph with directed ``source -> target`` edges and geometry."""

    num_nodes: int
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_weight: np.ndarray
    feature_names: Tuple[str, ...] = EDGE_FEATURE_NAMES

    def __post_init__(self) -> None:
        edge_index = np.asarray(self.edge_index, dtype=np.int64)
        edge_features = np.asarray(self.edge_features, dtype=np.float32)
        edge_weight = np.asarray(self.edge_weight, dtype=np.float32)
        if self.num_nodes < 0:
            raise ValueError("num_nodes cannot be negative")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, edges)")
        if edge_features.shape != (edge_index.shape[1], len(self.feature_names)):
            raise ValueError("edge_features must have one row per edge")
        if edge_weight.shape != (edge_index.shape[1],):
            raise ValueError("edge_weight must have shape (edges,)")
        if edge_index.size:
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= self.num_nodes:
                raise ValueError("edge_index contains an invalid node")
        if not np.isfinite(edge_features).all() or not np.isfinite(edge_weight).all():
            raise ValueError("edge features and weights must be finite")
        if bool((edge_weight < 0.0).any()):
            raise ValueError("edge weights cannot be negative")
        edge_index.setflags(write=False)
        edge_features.setflags(write=False)
        edge_weight.setflags(write=False)
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "edge_features", edge_features)
        object.__setattr__(self, "edge_weight", edge_weight)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def distance(self) -> np.ndarray:
        return self.edge_features[:, 0]

    @property
    def dx(self) -> np.ndarray:
        return self.edge_features[:, 1]

    @property
    def dy(self) -> np.ndarray:
        return self.edge_features[:, 2]

    @property
    def angle(self) -> np.ndarray:
        return self.edge_features[:, 3]

    def in_degree(self) -> np.ndarray:
        if self.num_edges == 0:
            return np.zeros(self.num_nodes, dtype=np.int64)
        return np.bincount(self.edge_index[1], minlength=self.num_nodes)

    def out_degree(self) -> np.ndarray:
        if self.num_edges == 0:
            return np.zeros(self.num_nodes, dtype=np.int64)
        return np.bincount(self.edge_index[0], minlength=self.num_nodes)


def _validate_node_values(
    values: Optional[Sequence[float]],
    num_nodes: int,
    name: str,
    maximum: Optional[float] = None,
) -> Optional[np.ndarray]:
    if values is None:
        return None
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (num_nodes,):
        raise ValueError("%s must have shape (nodes,)" % name)
    if not np.isfinite(result).all() or bool((result < 0.0).any()):
        raise ValueError("%s must contain finite nonnegative values" % name)
    if maximum is not None and bool((result > maximum).any()):
        raise ValueError("%s values cannot exceed %s" % (name, maximum))
    return result


def boundary_aware_edge_weights(
    distances: np.ndarray,
    edge_index: np.ndarray,
    num_nodes: int,
    distance_scale: Optional[float] = None,
    boundary_weights: Optional[Sequence[float]] = None,
    boundary_distance: Optional[Sequence[float]] = None,
    boundary_scale: Optional[float] = None,
) -> np.ndarray:
    """Combine a Gaussian distance kernel with optional boundary reliability.

    ``boundary_weights`` are explicit per-node reliabilities in ``[0, 1]``.
    ``boundary_distance`` denotes distance inside the valid tissue/crop; nodes
    closer than ``boundary_scale`` to the boundary are linearly downweighted.
    Edge reliability is the geometric mean of its endpoint reliabilities.
    """

    distance = np.asarray(distances, dtype=np.float64)
    index = np.asarray(edge_index, dtype=np.int64)
    if distance.ndim != 1 or index.shape != (2, distance.shape[0]):
        raise ValueError("distances and edge_index have incompatible shapes")
    if not np.isfinite(distance).all() or bool((distance < 0.0).any()):
        raise ValueError("distances must be finite and nonnegative")
    if distance_scale is None:
        positive = distance[distance > 0.0]
        distance_scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(distance_scale) or distance_scale <= 0.0:
        raise ValueError("distance_scale must be finite and positive")
    weights = np.exp(-0.5 * np.square(distance / float(distance_scale)))

    node_reliability = np.ones(num_nodes, dtype=np.float64)
    explicit = _validate_node_values(boundary_weights, num_nodes, "boundary_weights", 1.0)
    if explicit is not None:
        node_reliability *= explicit
    depth = _validate_node_values(boundary_distance, num_nodes, "boundary_distance")
    if depth is not None:
        if boundary_scale is None:
            positive = depth[depth > 0.0]
            boundary_scale = float(np.median(positive)) if positive.size else 1.0
        if not np.isfinite(boundary_scale) or boundary_scale <= 0.0:
            raise ValueError("boundary_scale must be finite and positive")
        node_reliability *= np.clip(depth / float(boundary_scale), 0.0, 1.0)
    if distance.size:
        source, target = index
        weights *= np.sqrt(node_reliability[source] * node_reliability[target])
    return weights.astype(np.float32, copy=False)


def _neighbors_for_target(
    tree: object,
    coordinates: np.ndarray,
    target: int,
    k: Optional[int],
    radius: Optional[float],
) -> list:
    if k is not None:
        query_count = min(coordinates.shape[0], k + 1)
        upper_bound = np.inf if radius is None else radius
        distance, index = tree.query(  # type: ignore[attr-defined]
            coordinates[target],
            k=query_count,
            distance_upper_bound=upper_bound,
        )
        distances = np.atleast_1d(distance)
        indices = np.atleast_1d(index)
        result = [
            (float(dist), int(node))
            for dist, node in zip(distances, indices)
            if int(node) < coordinates.shape[0] and int(node) != target and np.isfinite(dist)
        ]
    else:
        candidates = tree.query_ball_point(coordinates[target], radius)  # type: ignore[attr-defined]
        result = []
        for node in candidates:
            node = int(node)
            if node == target:
                continue
            dist = float(np.linalg.norm(coordinates[node] - coordinates[target]))
            result.append((dist, node))
    result.sort(key=lambda item: (item[0], item[1]))
    return result[:k] if k is not None else result


def build_spatial_graph(
    coordinates: np.ndarray,
    k: Optional[int] = 8,
    radius: Optional[float] = None,
    max_degree: Optional[int] = None,
    symmetric: bool = True,
    distance_scale: Optional[float] = None,
    boundary_weights: Optional[Sequence[float]] = None,
    boundary_distance: Optional[Sequence[float]] = None,
    boundary_scale: Optional[float] = None,
) -> SpatialGraph:
    """Build a sparse kNN/radius graph without a dense distance matrix.

    For each target nucleus, the closest ``k`` source neighbors are retained,
    optionally filtered by ``radius``.  Set ``k=None`` for a pure radius graph.
    ``max_degree`` is a hard cap on target in-degree after optional
    symmetrization.  Edge displacement is ``target - source``.
    """

    points = _coordinates(coordinates)
    if k is not None and (int(k) != k or int(k) <= 0):
        raise ValueError("k must be a positive integer or None")
    k = None if k is None else int(k)
    if radius is not None and (not np.isfinite(radius) or radius <= 0.0):
        raise ValueError("radius must be finite and positive")
    if k is None and radius is None:
        raise ValueError("at least one of k or radius is required")
    if max_degree is not None and (int(max_degree) != max_degree or int(max_degree) <= 0):
        raise ValueError("max_degree must be a positive integer or None")
    max_degree = None if max_degree is None else int(max_degree)

    num_nodes = points.shape[0]
    if num_nodes < 2:
        return SpatialGraph(
            num_nodes,
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    tree = _ckdtree()(points)
    by_target = []
    for target in range(num_nodes):
        candidate_neighbors = _neighbors_for_target(tree, points, target, k, radius)
        by_target.append({source: distance for distance, source in candidate_neighbors})

    if symmetric:
        original = [dict(neighbor_map) for neighbor_map in by_target]
        for target, neighbor_map in enumerate(original):
            for source, distance in neighbor_map.items():
                previous = by_target[source].get(target)
                if previous is None or distance < previous:
                    by_target[source][target] = distance

    sources = []
    targets = []
    for target, neighbor_map in enumerate(by_target):
        ordered = sorted(neighbor_map.items(), key=lambda item: (item[1], item[0]))
        if max_degree is not None:
            ordered = ordered[:max_degree]
        for source, _ in ordered:
            sources.append(source)
            targets.append(target)
    edge_index = np.asarray((sources, targets), dtype=np.int64)
    if edge_index.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        features = np.empty((0, 4), dtype=np.float32)
        weights = np.empty(0, dtype=np.float32)
    else:
        displacement = points[edge_index[1]] - points[edge_index[0]]
        distances = np.linalg.norm(displacement, axis=1)
        angles = np.arctan2(displacement[:, 1], displacement[:, 0])
        features = np.column_stack((distances, displacement, angles)).astype(np.float32)
        weights = boundary_aware_edge_weights(
            distances,
            edge_index,
            num_nodes,
            distance_scale=distance_scale,
            boundary_weights=boundary_weights,
            boundary_distance=boundary_distance,
            boundary_scale=boundary_scale,
        )
    return SpatialGraph(num_nodes, edge_index, features, weights)


# Concise alias retained for configuration-driven pipelines.
build_nucleus_graph = build_spatial_graph


__all__ = [
    "EDGE_FEATURE_NAMES",
    "SpatialGraph",
    "boundary_aware_edge_weights",
    "build_spatial_graph",
    "build_nucleus_graph",
]
