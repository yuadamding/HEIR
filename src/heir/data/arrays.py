"""Validated, immutable NumPy/NPZ contracts used between HEIR stages.

The contracts intentionally contain arrays rather than framework tensors.
Segmentation, RNA preparation, prototype construction, and model training can
therefore run in separate environments while sharing deterministic artifacts.
NPZ loading never enables pickle.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import sparse

PathLike = Union[str, os.PathLike]
CONTRACT_VERSION = 3


def _readonly_array(
    value: Any,
    dtype: Optional[np.dtype] = None,
    ndim: Optional[int] = None,
    name: str = "array",
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    if ndim is not None and array.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))
    array.setflags(write=False)
    return array


def _readonly_strings(value: Any, name: str) -> np.ndarray:
    array = _readonly_array(value, dtype=np.dtype("U"), ndim=1, name=name)
    if any(not item.strip() for item in array.tolist()):
        raise ValueError("%s cannot contain empty identifiers" % name)
    return array


def _check_unique(array: np.ndarray, name: str) -> None:
    if len(set(array.tolist())) != array.shape[0]:
        raise ValueError("%s must be unique" % name)


def _check_finite(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError("%s must contain only finite values" % name)


def _readonly_csr(value: Any, name: str) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(value, dtype=np.float32, copy=True)
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise ValueError("%s must contain only finite values" % name)
    matrix.data.setflags(write=False)
    matrix.indices.setflags(write=False)
    matrix.indptr.setflags(write=False)
    return matrix


def _atomic_npz(path: PathLike, compressed: bool, payload: Dict[str, np.ndarray]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".npz.tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            saver = np.savez_compressed if compressed else np.savez
            saver(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _contract_payload(name: str) -> Dict[str, np.ndarray]:
    return {
        "__contract__": np.asarray(name, dtype=np.dtype("U")),
        "__version__": np.asarray(CONTRACT_VERSION, dtype=np.int64),
    }


def _check_contract(archive: Any, expected: str) -> None:
    if "__contract__" not in archive or "__version__" not in archive:
        raise ValueError("NPZ artifact has no HEIR contract metadata")
    contract = str(np.asarray(archive["__contract__"]).item())
    version = int(np.asarray(archive["__version__"]).item())
    if contract != expected:
        raise ValueError("expected %s NPZ contract, found %s" % (expected, contract))
    if version not in {1, 2, CONTRACT_VERSION}:
        raise ValueError("unsupported %s contract version %d" % (expected, version))


@dataclass(frozen=True)
class HistologyBag:
    """Cell-indexed image features and physical coordinates for one section."""

    slide_id: str
    nucleus_ids: np.ndarray
    features: np.ndarray
    coordinates_um: np.ndarray
    morphology: Optional[np.ndarray] = None
    segmentation_confidence: Optional[np.ndarray] = None
    artifact_probability: Optional[np.ndarray] = None
    edge_index: Optional[np.ndarray] = None
    edge_weight: Optional[np.ndarray] = None
    sample_id: str = ""
    donor_id: str = ""
    block_id: str = ""
    feature_space_id: str = ""
    histology_source_sha256: str = ""
    nuclei_source_sha256: str = ""
    feature_source_sha256: str = ""

    CONTRACT = "heir.histology_bag"

    def __post_init__(self) -> None:
        if not self.slide_id.strip():
            raise ValueError("slide_id cannot be empty")
        for name in ("sample_id", "donor_id", "block_id", "feature_space_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError("%s must be a string" % name)
            if value and not value.strip():
                raise ValueError("%s cannot contain only whitespace" % name)
        for name in (
            "histology_source_sha256",
            "nuclei_source_sha256",
            "feature_source_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError("%s must be a string" % name)
            if value and (
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("%s must be a lowercase SHA-256 digest" % name)
        ids = _readonly_strings(self.nucleus_ids, "nucleus_ids")
        _check_unique(ids, "nucleus_ids")
        count = ids.shape[0]
        if count == 0:
            raise ValueError("HistologyBag must contain at least one nucleus")
        features = _readonly_array(self.features, dtype=np.float32, ndim=2, name="features")
        coordinates = _readonly_array(
            self.coordinates_um, dtype=np.float64, ndim=2, name="coordinates_um"
        )
        if features.shape[0] != count:
            raise ValueError("features rows must match nucleus_ids")
        if features.shape[1] == 0:
            raise ValueError("features must contain at least one feature")
        if coordinates.shape != (count, 2):
            raise ValueError("coordinates_um must have shape (n_nuclei, 2)")
        morphology_value = (
            np.empty((count, 0), dtype=np.float32) if self.morphology is None else self.morphology
        )
        morphology = _readonly_array(morphology_value, dtype=np.float32, ndim=2, name="morphology")
        if morphology.shape[0] != count:
            raise ValueError("morphology rows must match nucleus_ids")
        confidence_value = (
            np.ones(count, dtype=np.float32)
            if self.segmentation_confidence is None
            else self.segmentation_confidence
        )
        confidence = _readonly_array(
            confidence_value,
            dtype=np.float32,
            ndim=1,
            name="segmentation_confidence",
        )
        artifact_value = (
            np.zeros(count, dtype=np.float32)
            if self.artifact_probability is None
            else self.artifact_probability
        )
        artifact = _readonly_array(
            artifact_value, dtype=np.float32, ndim=1, name="artifact_probability"
        )
        if confidence.shape != (count,) or artifact.shape != (count,):
            raise ValueError("confidence arrays must have one value per nucleus")
        if ((confidence < 0) | (confidence > 1)).any():
            raise ValueError("segmentation_confidence must be in [0, 1]")
        if ((artifact < 0) | (artifact > 1)).any():
            raise ValueError("artifact_probability must be in [0, 1]")
        if self.edge_index is None:
            edge_value = np.empty((2, 0), dtype=np.int64)
        else:
            edge_value = self.edge_index
        edges = _readonly_array(edge_value, dtype=np.int64, ndim=2, name="edge_index")
        if edges.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, n_edges)")
        if edges.size and ((edges < 0).any() or (edges >= count).any()):
            raise ValueError("edge_index contains an out-of-range nucleus index")
        edge_weight_value = (
            np.ones(edges.shape[1], dtype=np.float32)
            if self.edge_weight is None
            else self.edge_weight
        )
        edge_weights = _readonly_array(
            edge_weight_value,
            dtype=np.float32,
            ndim=1,
            name="edge_weight",
        )
        if edge_weights.shape != (edges.shape[1],):
            raise ValueError("edge_weight must contain one value per edge")
        if (edge_weights < 0).any():
            raise ValueError("edge_weight must be non-negative")
        for array, name in (
            (features, "features"),
            (coordinates, "coordinates_um"),
            (morphology, "morphology"),
            (confidence, "segmentation_confidence"),
            (artifact, "artifact_probability"),
            (edge_weights, "edge_weight"),
        ):
            _check_finite(array, name)
        object.__setattr__(self, "nucleus_ids", ids)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "coordinates_um", coordinates)
        object.__setattr__(self, "morphology", morphology)
        object.__setattr__(self, "segmentation_confidence", confidence)
        object.__setattr__(self, "artifact_probability", artifact)
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "edge_weight", edge_weights)

    @property
    def image_features(self) -> np.ndarray:
        return self.features

    @property
    def morphology_features(self) -> np.ndarray:
        assert self.morphology is not None
        return self.morphology

    @property
    def n_nuclei(self) -> int:
        return int(self.nucleus_ids.shape[0])

    def save_npz(self, path: PathLike, compressed: bool = True) -> None:
        payload = _contract_payload(self.CONTRACT)
        payload.update(
            {
                "slide_id": np.asarray(self.slide_id, dtype=np.dtype("U")),
                "nucleus_ids": self.nucleus_ids,
                "features": self.features,
                "coordinates_um": self.coordinates_um,
                "morphology": self.morphology_features,
                "segmentation_confidence": self.segmentation_confidence,
                "artifact_probability": self.artifact_probability,
                "edge_index": self.edge_index,
                "edge_weight": self.edge_weight,
                "sample_id": np.asarray(self.sample_id, dtype=np.dtype("U")),
                "donor_id": np.asarray(self.donor_id, dtype=np.dtype("U")),
                "block_id": np.asarray(self.block_id, dtype=np.dtype("U")),
                "feature_space_id": np.asarray(self.feature_space_id, dtype=np.dtype("U")),
                "histology_source_sha256": np.asarray(
                    self.histology_source_sha256, dtype=np.dtype("U")
                ),
                "nuclei_source_sha256": np.asarray(self.nuclei_source_sha256, dtype=np.dtype("U")),
                "feature_source_sha256": np.asarray(
                    self.feature_source_sha256, dtype=np.dtype("U")
                ),
            }
        )
        _atomic_npz(path, compressed, payload)

    @classmethod
    def load_npz(cls, path: PathLike) -> "HistologyBag":
        with np.load(path, allow_pickle=False) as archive:
            _check_contract(archive, cls.CONTRACT)
            return cls(
                slide_id=str(np.asarray(archive["slide_id"]).item()),
                nucleus_ids=archive["nucleus_ids"],
                features=archive["features"],
                coordinates_um=archive["coordinates_um"],
                morphology=archive["morphology"],
                segmentation_confidence=archive["segmentation_confidence"],
                artifact_probability=archive["artifact_probability"],
                edge_index=archive["edge_index"],
                edge_weight=(archive["edge_weight"] if "edge_weight" in archive else None),
                sample_id=(
                    str(np.asarray(archive["sample_id"]).item()) if "sample_id" in archive else ""
                ),
                donor_id=(
                    str(np.asarray(archive["donor_id"]).item()) if "donor_id" in archive else ""
                ),
                block_id=(
                    str(np.asarray(archive["block_id"]).item()) if "block_id" in archive else ""
                ),
                feature_space_id=(
                    str(np.asarray(archive["feature_space_id"]).item())
                    if "feature_space_id" in archive
                    else ""
                ),
                histology_source_sha256=(
                    str(np.asarray(archive["histology_source_sha256"]).item())
                    if "histology_source_sha256" in archive
                    else ""
                ),
                nuclei_source_sha256=(
                    str(np.asarray(archive["nuclei_source_sha256"]).item())
                    if "nuclei_source_sha256" in archive
                    else ""
                ),
                feature_source_sha256=(
                    str(np.asarray(archive["feature_source_sha256"]).item())
                    if "feature_source_sha256" in archive
                    else ""
                ),
            )


@dataclass(frozen=True)
class RNAReference:
    """Sparse raw/corrected counts plus cell-indexed RNA annotations."""

    sample_id: str
    cell_ids: np.ndarray
    gene_ids: np.ndarray
    counts: Any
    library_sizes: Optional[np.ndarray] = None
    latent: Optional[np.ndarray] = None
    cell_type_labels: Optional[np.ndarray] = None
    donor_ids: Optional[np.ndarray] = None
    sample_ids: Optional[np.ndarray] = None
    program_scores: Optional[np.ndarray] = None
    latent_space_id: str = ""
    block_id: str = ""
    source_count_sha256: str = ""
    latent_training_donors: Tuple[str, ...] = ()
    latent_transform_sha256: str = ""

    CONTRACT = "heir.rna_reference"

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id cannot be empty")
        if self.latent_space_id and not self.latent_space_id.strip():
            raise ValueError("latent_space_id cannot contain only whitespace")
        if self.block_id and not self.block_id.strip():
            raise ValueError("block_id cannot contain only whitespace")
        if self.source_count_sha256 and (
            len(self.source_count_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_count_sha256)
        ):
            raise ValueError("source_count_sha256 must be a lowercase SHA-256 digest")
        latent_donors = tuple(str(value).strip() for value in self.latent_training_donors)
        if any(not value for value in latent_donors) or len(set(latent_donors)) != len(
            latent_donors
        ):
            raise ValueError("latent_training_donors must be unique and non-empty")
        if self.latent_transform_sha256 and (
            len(self.latent_transform_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.latent_transform_sha256)
        ):
            raise ValueError("latent_transform_sha256 must be a lowercase SHA-256 digest")
        cells = _readonly_strings(self.cell_ids, "cell_ids")
        genes = _readonly_strings(self.gene_ids, "gene_ids")
        _check_unique(cells, "cell_ids")
        _check_unique(genes, "gene_ids")
        n_cells, n_genes = cells.shape[0], genes.shape[0]
        if n_cells == 0 or n_genes == 0:
            raise ValueError("RNAReference must contain cells and genes")
        counts = _readonly_csr(self.counts, "counts")
        if counts.shape != (n_cells, n_genes):
            raise ValueError("counts must have shape (n_cells, n_genes)")
        if counts.data.size and (counts.data < 0).any():
            raise ValueError("counts must be non-negative")
        selected_library_sizes = np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)
        library_value = selected_library_sizes if self.library_sizes is None else self.library_sizes
        library_sizes = _readonly_array(
            library_value,
            dtype=np.float64,
            ndim=1,
            name="library_sizes",
        )
        if library_sizes.shape != (n_cells,):
            raise ValueError("library_sizes must have one value per cell")
        _check_finite(library_sizes, "library_sizes")
        if (library_sizes < 0).any():
            raise ValueError("library_sizes must be non-negative")
        tolerance = np.maximum(1.0e-5, selected_library_sizes * 1.0e-6)
        if np.any(library_sizes + tolerance < selected_library_sizes):
            raise ValueError("library_sizes cannot be smaller than selected-gene counts")
        latent_value = (
            np.empty((n_cells, 0), dtype=np.float32) if self.latent is None else self.latent
        )
        latent = _readonly_array(latent_value, dtype=np.float32, ndim=2, name="latent")
        if latent.shape[0] != n_cells:
            raise ValueError("latent rows must match cell_ids")
        labels_value = (
            np.full(n_cells, "unknown", dtype=np.dtype("<U7"))
            if self.cell_type_labels is None
            else self.cell_type_labels
        )
        labels = _readonly_strings(labels_value, "cell_type_labels")
        donor_value = (
            np.full(
                n_cells,
                self.sample_id,
                dtype=np.dtype("<U%d" % max(1, len(self.sample_id))),
            )
            if self.donor_ids is None
            else self.donor_ids
        )
        donors = _readonly_strings(donor_value, "donor_ids")
        sample_value = (
            np.full(
                n_cells,
                self.sample_id,
                dtype=np.dtype("<U%d" % max(1, len(self.sample_id))),
            )
            if self.sample_ids is None
            else self.sample_ids
        )
        samples = _readonly_strings(sample_value, "sample_ids")
        programs_value = (
            np.empty((n_cells, 0), dtype=np.float32)
            if self.program_scores is None
            else self.program_scores
        )
        programs = _readonly_array(programs_value, dtype=np.float32, ndim=2, name="program_scores")
        if labels.shape != (n_cells,) or donors.shape != (n_cells,) or samples.shape != (n_cells,):
            raise ValueError("cell metadata must have one value per cell")
        if programs.shape[0] != n_cells:
            raise ValueError("program_scores rows must match cell_ids")
        _check_finite(latent, "latent")
        _check_finite(programs, "program_scores")
        object.__setattr__(self, "cell_ids", cells)
        object.__setattr__(self, "gene_ids", genes)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "library_sizes", library_sizes)
        object.__setattr__(self, "latent", latent)
        object.__setattr__(self, "cell_type_labels", labels)
        object.__setattr__(self, "donor_ids", donors)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "program_scores", programs)
        object.__setattr__(self, "latent_training_donors", latent_donors)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.counts.shape

    def save_npz(self, path: PathLike, compressed: bool = True) -> None:
        matrix = self.counts
        payload = _contract_payload(self.CONTRACT)
        payload.update(
            {
                "sample_id": np.asarray(self.sample_id, dtype=np.dtype("U")),
                "cell_ids": self.cell_ids,
                "gene_ids": self.gene_ids,
                "counts_format": np.asarray("csr", dtype=np.dtype("U")),
                "counts_data": matrix.data,
                "counts_indices": matrix.indices,
                "counts_indptr": matrix.indptr,
                "counts_shape": np.asarray(matrix.shape, dtype=np.int64),
                "library_sizes": self.library_sizes,
                "latent": self.latent,
                "cell_type_labels": self.cell_type_labels,
                "donor_ids": self.donor_ids,
                "sample_ids": self.sample_ids,
                "program_scores": self.program_scores,
                "latent_space_id": np.asarray(self.latent_space_id, dtype=np.dtype("U")),
                "block_id": np.asarray(self.block_id, dtype=np.dtype("U")),
                "source_count_sha256": np.asarray(self.source_count_sha256, dtype=np.dtype("U")),
                "latent_training_donors": np.asarray(
                    self.latent_training_donors, dtype=np.dtype("U")
                ),
                "latent_transform_sha256": np.asarray(
                    self.latent_transform_sha256, dtype=np.dtype("U")
                ),
            }
        )
        _atomic_npz(path, compressed, payload)

    @classmethod
    def load_npz(cls, path: PathLike) -> "RNAReference":
        with np.load(path, allow_pickle=False) as archive:
            _check_contract(archive, cls.CONTRACT)
            if str(np.asarray(archive["counts_format"]).item()) != "csr":
                raise ValueError("unsupported RNAReference matrix encoding")
            shape = tuple(int(value) for value in archive["counts_shape"].tolist())
            counts = sparse.csr_matrix(
                (
                    archive["counts_data"],
                    archive["counts_indices"],
                    archive["counts_indptr"],
                ),
                shape=shape,
            )
            return cls(
                sample_id=str(np.asarray(archive["sample_id"]).item()),
                cell_ids=archive["cell_ids"],
                gene_ids=archive["gene_ids"],
                counts=counts,
                library_sizes=(
                    archive["library_sizes"]
                    if "library_sizes" in archive
                    else np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)
                ),
                latent=archive["latent"],
                cell_type_labels=archive["cell_type_labels"],
                donor_ids=archive["donor_ids"],
                sample_ids=archive["sample_ids"],
                program_scores=archive["program_scores"],
                latent_space_id=(
                    str(np.asarray(archive["latent_space_id"]).item())
                    if "latent_space_id" in archive
                    else ""
                ),
                block_id=(
                    str(np.asarray(archive["block_id"]).item()) if "block_id" in archive else ""
                ),
                source_count_sha256=(
                    str(np.asarray(archive["source_count_sha256"]).item())
                    if "source_count_sha256" in archive
                    else ""
                ),
                latent_training_donors=(
                    tuple(str(value) for value in archive["latent_training_donors"].tolist())
                    if "latent_training_donors" in archive
                    else ()
                ),
                latent_transform_sha256=(
                    str(np.asarray(archive["latent_transform_sha256"]).item())
                    if "latent_transform_sha256" in archive
                    else ""
                ),
            )


@dataclass(frozen=True)
class PrototypeSet:
    """Sample-specific RNA latent prototypes and empirical frequencies."""

    prototype_ids: np.ndarray
    sample_ids: np.ndarray
    cell_type_labels: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    n_cells: Optional[np.ndarray] = None
    latent_space_id: str = ""
    donor_id: str = ""
    block_id: str = ""
    source_reference_sha256: str = ""
    latent_training_donors: Tuple[str, ...] = ()
    latent_transform_sha256: str = ""

    CONTRACT = "heir.prototype_set"

    def __post_init__(self) -> None:
        prototype_ids = _readonly_strings(self.prototype_ids, "prototype_ids")
        if self.latent_space_id and not self.latent_space_id.strip():
            raise ValueError("latent_space_id cannot contain only whitespace")
        if self.donor_id and not self.donor_id.strip():
            raise ValueError("donor_id cannot contain only whitespace")
        if self.block_id and not self.block_id.strip():
            raise ValueError("block_id cannot contain only whitespace")
        if self.source_reference_sha256 and (
            len(self.source_reference_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.source_reference_sha256)
        ):
            raise ValueError("source_reference_sha256 must be a lowercase SHA-256 digest")
        latent_donors = tuple(str(value).strip() for value in self.latent_training_donors)
        if any(not value for value in latent_donors) or len(set(latent_donors)) != len(
            latent_donors
        ):
            raise ValueError("latent_training_donors must be unique and non-empty")
        if self.latent_transform_sha256 and (
            len(self.latent_transform_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.latent_transform_sha256)
        ):
            raise ValueError("latent_transform_sha256 must be a lowercase SHA-256 digest")
        _check_unique(prototype_ids, "prototype_ids")
        count = prototype_ids.shape[0]
        if count == 0:
            raise ValueError("PrototypeSet must contain at least one prototype")
        sample_ids = _readonly_strings(self.sample_ids, "sample_ids")
        labels = _readonly_strings(self.cell_type_labels, "cell_type_labels")
        means = _readonly_array(self.means, dtype=np.float32, ndim=2, name="means")
        variances = _readonly_array(self.variances, dtype=np.float32, ndim=2, name="variances")
        weights = _readonly_array(self.weights, dtype=np.float64, ndim=1, name="weights")
        cell_value = np.ones(count, dtype=np.int64) if self.n_cells is None else self.n_cells
        n_cells = _readonly_array(cell_value, dtype=np.int64, ndim=1, name="n_cells")
        if sample_ids.shape != (count,) or labels.shape != (count,):
            raise ValueError("prototype metadata must have one value per prototype")
        if means.shape[0] != count or means.shape[1] == 0:
            raise ValueError("means must have shape (n_prototypes, latent_dim)")
        if variances.shape != means.shape:
            raise ValueError("variances must have the same shape as means")
        if weights.shape != (count,) or n_cells.shape != (count,):
            raise ValueError("weights and n_cells must have one value per prototype")
        _check_finite(means, "means")
        _check_finite(variances, "variances")
        _check_finite(weights, "weights")
        if (variances <= 0).any():
            raise ValueError("prototype variances must be positive")
        if (weights < 0).any() or (n_cells <= 0).any():
            raise ValueError("prototype weights must be non-negative and n_cells positive")
        for sample in sorted(set(sample_ids.tolist())):
            total = float(weights[np.asarray(sample_ids == sample)].sum())
            if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-8):
                raise ValueError("prototype weights for sample %s must sum to one" % sample)
        object.__setattr__(self, "prototype_ids", prototype_ids)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "cell_type_labels", labels)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "variances", variances)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "n_cells", n_cells)
        object.__setattr__(self, "latent_training_donors", latent_donors)

    @property
    def covariance_diagonal(self) -> np.ndarray:
        return self.variances

    @property
    def frequencies(self) -> np.ndarray:
        return self.weights

    def save_npz(self, path: PathLike, compressed: bool = True) -> None:
        payload = _contract_payload(self.CONTRACT)
        payload.update(
            {
                "prototype_ids": self.prototype_ids,
                "sample_ids": self.sample_ids,
                "cell_type_labels": self.cell_type_labels,
                "means": self.means,
                "variances": self.variances,
                "weights": self.weights,
                "n_cells": self.n_cells,
                "latent_space_id": np.asarray(self.latent_space_id, dtype=np.dtype("U")),
                "donor_id": np.asarray(self.donor_id, dtype=np.dtype("U")),
                "block_id": np.asarray(self.block_id, dtype=np.dtype("U")),
                "source_reference_sha256": np.asarray(
                    self.source_reference_sha256, dtype=np.dtype("U")
                ),
                "latent_training_donors": np.asarray(
                    self.latent_training_donors, dtype=np.dtype("U")
                ),
                "latent_transform_sha256": np.asarray(
                    self.latent_transform_sha256, dtype=np.dtype("U")
                ),
            }
        )
        _atomic_npz(path, compressed, payload)

    @classmethod
    def load_npz(cls, path: PathLike) -> "PrototypeSet":
        with np.load(path, allow_pickle=False) as archive:
            _check_contract(archive, cls.CONTRACT)
            return cls(
                prototype_ids=archive["prototype_ids"],
                sample_ids=archive["sample_ids"],
                cell_type_labels=archive["cell_type_labels"],
                means=archive["means"],
                variances=archive["variances"],
                weights=archive["weights"],
                n_cells=archive["n_cells"],
                latent_space_id=(
                    str(np.asarray(archive["latent_space_id"]).item())
                    if "latent_space_id" in archive
                    else ""
                ),
                donor_id=(
                    str(np.asarray(archive["donor_id"]).item()) if "donor_id" in archive else ""
                ),
                source_reference_sha256=(
                    str(np.asarray(archive["source_reference_sha256"]).item())
                    if "source_reference_sha256" in archive
                    else ""
                ),
                latent_training_donors=(
                    tuple(str(value) for value in archive["latent_training_donors"].tolist())
                    if "latent_training_donors" in archive
                    else ()
                ),
                latent_transform_sha256=(
                    str(np.asarray(archive["latent_transform_sha256"]).item())
                    if "latent_transform_sha256" in archive
                    else ""
                ),
                block_id=(
                    str(np.asarray(archive["block_id"]).item()) if "block_id" in archive else ""
                ),
            )


# Descriptive aliases retained for downstream modules and configuration files.
PrototypeArray = PrototypeSet
PrototypeArrays = PrototypeSet
RNAPrototypes = PrototypeSet
