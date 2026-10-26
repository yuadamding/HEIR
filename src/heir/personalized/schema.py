"""Explicit prediction-only inputs, separate spatial targets, and named output axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def identifiers(values: tuple[str, ...], name: str) -> None:
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{name} must contain nonempty string identifiers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate identifiers")


def array(value: np.ndarray, shape: tuple[int, ...], name: str, *, nonnegative=False):
    if np.shape(value) != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    if nonnegative and np.any(np.asarray(value) < 0):
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class RNAReference:
    state_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    expression: np.ndarray
    embedding: np.ndarray
    proportions: np.ndarray
    state_types: tuple[str, ...]
    variance: np.ndarray
    cell_counts: np.ndarray
    embedding_gene_ids: tuple[str, ...]

    def __post_init__(self):
        identifiers(self.state_ids, "state_ids")
        if "unsupported" in self.state_ids:
            raise ValueError("'unsupported' is reserved for the unsupported-reference component")
        identifiers(self.gene_ids, "gene_ids")
        k, g = len(self.state_ids), len(self.gene_ids)
        array(self.expression, (k, g), "reference expression", nonnegative=True)
        array(self.variance, (k, g), "reference variance", nonnegative=True)
        if np.ndim(self.embedding) != 2 or np.shape(self.embedding)[1] == 0:
            raise ValueError("reference embedding must have a nonempty feature axis")
        array(self.embedding, (k, np.shape(self.embedding)[1]), "reference embedding")
        identifiers(self.embedding_gene_ids, "reference embedding_gene_ids")
        if len(self.embedding_gene_ids) != self.embedding.shape[1]:
            raise ValueError("reference embedding gene IDs must name every feature column")
        if not set(self.embedding_gene_ids) <= set(self.gene_ids):
            raise ValueError("reference embedding genes must belong to the measured RNA gene axis")
        array(self.proportions, (k,), "reference proportions", nonnegative=True)
        array(self.cell_counts, (k,), "reference cell_counts", nonnegative=True)
        if not np.isclose(np.sum(self.proportions), 1):
            raise ValueError("reference proportions must sum to one")
        if np.any(np.asarray(self.cell_counts) <= 0):
            raise ValueError("every reference state must have supporting RNA cells")
        if len(self.state_types) != k or any(not str(t).strip() for t in self.state_types):
            raise ValueError("reference state_types must identify each state's broad type")


@dataclass(frozen=True)
class ProteinEvidence:
    marker_ids: tuple[str, ...]
    values: np.ndarray
    observed: np.ndarray
    state_values: np.ndarray
    correspondence: str
    quality: Optional[np.ndarray] = None

    def __post_init__(self):
        identifiers(self.marker_ids, "marker_ids")
        if np.ndim(self.values) != 2 or np.shape(self.values)[1] != len(self.marker_ids):
            raise ValueError("protein values must be locations by markers")
        mask = np.asarray(self.observed)
        if mask.dtype != bool or mask.shape != np.shape(self.values):
            raise ValueError("protein observed must be a boolean mask matching values")
        if not np.isfinite(np.asarray(self.values)[mask]).all():
            raise ValueError("observed protein values must be finite")
        if np.any(np.asarray(self.values)[mask] < 0):
            raise ValueError("observed protein values must be nonnegative")
        if np.ndim(self.state_values) != 2:
            raise ValueError("protein state_values must be states by markers")
        array(
            self.state_values,
            (len(self.state_values), len(self.marker_ids)),
            "protein state_values",
            nonnegative=True,
        )
        if self.correspondence not in {
            "same_cell",
            "same_section_soft",
            "serial_section",
            "region",
        }:
            raise ValueError("unknown protein correspondence")
        if self.quality is not None:
            array(self.quality, mask.shape, "protein quality", nonnegative=True)
            if np.any(self.quality > 1):
                raise ValueError("protein quality must be in [0, 1]")


@dataclass(frozen=True)
class SpatialGraph:
    senders: np.ndarray
    receivers: np.ndarray
    distances_um: np.ndarray
    weights: np.ndarray
    n_cells: int

    def __post_init__(self):
        e = len(self.senders)
        if self.n_cells < 1:
            raise ValueError("graph n_cells must be positive")
        for name in ("senders", "receivers"):
            values = np.asarray(getattr(self, name))
            if values.shape != (e,) or values.dtype.kind not in "iu":
                raise ValueError(f"graph {name} must be a one-dimensional integer array")
            if np.any(values < 0) or np.any(values >= self.n_cells):
                raise ValueError("graph node index out of bounds")
        array(self.distances_um, (e,), "graph distances_um", nonnegative=True)
        array(self.weights, (e,), "graph weights", nonnegative=True)


@dataclass(frozen=True)
class PredictionInputs:
    """Allowed evidence only. Spatial targets/labels have no field in this interface."""

    specimen_id: str
    patient_id: str
    cell_ids: tuple[str, ...]
    coordinates_um: np.ndarray
    image_features: np.ndarray
    reference: RNAReference
    proteins: Optional[ProteinEvidence] = None
    graph: Optional[SpatialGraph] = None
    registration_sensitivity: Optional[np.ndarray] = None

    def __post_init__(self):
        identifiers(self.cell_ids, "cell_ids")
        if not self.specimen_id or not self.patient_id:
            raise ValueError("specimen_id and patient_id are required")
        n = len(self.cell_ids)
        array(self.coordinates_um, (n, 2), "coordinates_um")
        if np.ndim(self.image_features) != 2 or np.shape(self.image_features)[1] == 0:
            raise ValueError("image_features must have a nonempty feature axis")
        array(self.image_features, (n, np.shape(self.image_features)[1]), "image_features")
        if self.proteins is not None:
            if self.proteins.values.shape[0] != n:
                raise ValueError("protein evidence must be aligned to H&E locations")
            if self.proteins.state_values.shape[0] != len(self.reference.state_ids):
                raise ValueError("protein state evidence must be aligned to reference states")
        if self.graph is not None and self.graph.n_cells != n:
            raise ValueError("graph must use the H&E cell axis")
        if self.registration_sensitivity is not None:
            # NaN denotes not assessed; it must not be silently advertised as zero sensitivity.
            if np.shape(self.registration_sensitivity) != (n,):
                raise ValueError("registration_sensitivity must use the H&E cell axis")
            known = np.asarray(self.registration_sensitivity)
            if np.isinf(known).any() or np.any(known[np.isfinite(known)] < 0):
                raise ValueError("registration_sensitivity must be nonnegative or NaN")


@dataclass(frozen=True)
class SpatialTargets:
    observation_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    expression: np.ndarray
    observed: np.ndarray
    patient_id: str
    specimen_id: str

    def __post_init__(self):
        identifiers(self.observation_ids, "observation_ids")
        identifiers(self.gene_ids, "target gene_ids")
        shape = (len(self.observation_ids), len(self.gene_ids))
        mask = np.asarray(self.observed)
        if mask.shape != shape or mask.dtype != bool or np.shape(self.expression) != shape:
            raise ValueError(
                "targets and boolean availability must align to observations and genes"
            )
        values = np.asarray(self.expression)[mask]
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("observed targets must be finite and nonnegative linear expression")
        if not self.patient_id or not self.specimen_id:
            raise ValueError("targets require patient and specimen identity")


@dataclass
class Prediction:
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    state_ids: tuple[str, ...]
    coordinates_um: torch.Tensor
    assignment: torch.Tensor
    transfer: torch.Tensor
    log_correction: torch.Tensor
    expression: torch.Tensor
    assignment_entropy: torch.Tensor
    reference_support: torch.Tensor
    registration_sensitivity: torch.Tensor
    supervised_genes: torch.Tensor
    independently_evaluated_genes: torch.Tensor
    expression_scale: str = "relative"
    specimen_id: str = ""
    patient_id: str = ""

    @property
    def expression_adjustment(self) -> torch.Tensor:
        """Additional linear expression, distinct from the bounded log correction."""
        return self.expression - self.transfer
