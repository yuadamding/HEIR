"""Backed, sparse H5AD preparation for matched RNA references.

Only observation/variable metadata is read during selection.  Expression is
streamed in row chunks and converted to CSR, so filtering one donor from the
18,063-gene public files never creates a dense cell-by-gene matrix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

from .arrays import RNAReference

PathLike = Union[str, os.PathLike]


def _require_anndata() -> Any:
    try:
        import anndata
    except ImportError as exc:
        raise ImportError(
            "anndata is required for H5AD preparation; install heir-spatial[h5ad]"
        ) from exc
    return anndata


def _readonly(value: Any, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    if array.ndim != 1:
        raise ValueError("%s must be one-dimensional" % name)
    array.setflags(write=False)
    return array


def _values(value: Optional[Union[str, Sequence[str]]]) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    result = tuple(str(item) for item in value)
    if not result:
        raise ValueError("filter values cannot be empty")
    return result


def _close_backed(adata: Any) -> None:
    file_manager = getattr(adata, "file", None)
    if file_manager is not None:
        close = getattr(file_manager, "close", None)
        if close is not None:
            close()


@dataclass(frozen=True)
class H5ADSelection:
    """Immutable metadata and indices for a backed H5AD slice."""

    path: str
    obs_indices: np.ndarray
    var_indices: np.ndarray
    cell_ids: np.ndarray
    gene_ids: np.ndarray
    donor_ids: np.ndarray
    sample_ids: np.ndarray
    cell_type_labels: np.ndarray
    sample_id: str
    matrix_source: str = "X"

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path cannot be empty")
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        obs = _readonly(self.obs_indices, np.dtype(np.int64), "obs_indices")
        var = _readonly(self.var_indices, np.dtype(np.int64), "var_indices")
        cells = _readonly(self.cell_ids, np.dtype("U"), "cell_ids")
        genes = _readonly(self.gene_ids, np.dtype("U"), "gene_ids")
        donors = _readonly(self.donor_ids, np.dtype("U"), "donor_ids")
        samples = _readonly(self.sample_ids, np.dtype("U"), "sample_ids")
        labels = _readonly(self.cell_type_labels, np.dtype("U"), "cell_type_labels")
        if obs.size == 0 or var.size == 0:
            raise ValueError("H5AD selection must contain cells and genes")
        if np.any(np.diff(obs) <= 0):
            raise ValueError("H5AD observation indices must be strictly increasing")
        if (var < 0).any() or len(set(var.tolist())) != var.size:
            raise ValueError("H5AD variable indices must be unique and non-negative")
        if cells.shape != obs.shape:
            raise ValueError("cell_ids must match obs_indices")
        if genes.shape != var.shape:
            raise ValueError("gene_ids must match var_indices")
        if donors.shape != obs.shape or samples.shape != obs.shape or labels.shape != obs.shape:
            raise ValueError("observation metadata must match obs_indices")
        if len(set(cells.tolist())) != cells.size:
            raise ValueError("selected cell_ids must be unique")
        if len(set(genes.tolist())) != genes.size:
            raise ValueError("selected gene_ids must be unique")
        if self.matrix_source != "X" and not self.matrix_source.strip():
            raise ValueError("matrix_source cannot be blank")
        object.__setattr__(self, "obs_indices", obs)
        object.__setattr__(self, "var_indices", var)
        object.__setattr__(self, "cell_ids", cells)
        object.__setattr__(self, "gene_ids", genes)
        object.__setattr__(self, "donor_ids", donors)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "cell_type_labels", labels)

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.obs_indices.size), int(self.var_indices.size)


def _column_strings(frame: Any, key: str, length: int, default: str) -> np.ndarray:
    if key and key in frame.columns:
        return np.asarray(frame[key].astype(str).to_numpy(), dtype=np.dtype("U"))
    return np.full(length, default, dtype=np.dtype("U%d" % max(1, len(default))))


def _filter_mask(frame: Any, filters: Mapping[str, Tuple[str, ...]]) -> np.ndarray:
    mask = np.ones(frame.shape[0], dtype=bool)
    for key, accepted in filters.items():
        if key not in frame.columns:
            raise KeyError("H5AD obs has no column %s" % key)
        values = frame[key].astype(str).to_numpy()
        current = np.isin(values, np.asarray(accepted, dtype=np.dtype("U")))
        mask &= current
    return mask


def prepare_h5ad(
    path: PathLike,
    donor_filter: Optional[Union[str, Sequence[str]]] = None,
    sample_filter: Optional[Union[str, Sequence[str]]] = None,
    donor_key: str = "donor_id",
    sample_key: str = "sample_id",
    cell_type_key: str = "cell_type",
    genes: Optional[Sequence[str]] = None,
    gene_key: Optional[str] = None,
    layer: Optional[str] = None,
    filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
    sample_id: Optional[str] = None,
) -> H5ADSelection:
    """Inspect a backed H5AD and prepare a sparse, filtered selection.

    ``donor_filter`` and ``sample_filter`` are intersections.  Additional
    observation filters may be passed through ``filters``.  Gene order follows
    the explicit ``genes`` argument; otherwise all variables remain in file
    order.  No expression values are read by this function.
    """

    source = Path(path).expanduser().resolve()
    anndata = _require_anndata()
    adata = anndata.read_h5ad(str(source), backed="r")
    try:
        requested_filters: Dict[str, Tuple[str, ...]] = {}
        donor_values = _values(donor_filter)
        sample_values = _values(sample_filter)
        if donor_values:
            requested_filters[donor_key] = donor_values
        if sample_values:
            requested_filters[sample_key] = sample_values
        if filters:
            for key, value in filters.items():
                if key in requested_filters:
                    raise ValueError("duplicate H5AD filter for %s" % key)
                requested_filters[str(key)] = _values(value)
        mask = _filter_mask(adata.obs, requested_filters)
        obs_indices = np.flatnonzero(mask).astype(np.int64, copy=False)
        if obs_indices.size == 0:
            description = ", ".join(
                "%s=%s" % (key, "|".join(value)) for key, value in requested_filters.items()
            )
            raise ValueError("H5AD filters selected no cells: %s" % description)

        if gene_key is None:
            available_genes = np.asarray(adata.var_names.astype(str), dtype=np.dtype("U"))
        else:
            if gene_key not in adata.var.columns:
                raise KeyError("H5AD var has no column %s" % gene_key)
            available_genes = np.asarray(
                adata.var[gene_key].astype(str).to_numpy(), dtype=np.dtype("U")
            )
        if genes is None:
            var_indices = np.arange(adata.n_vars, dtype=np.int64)
            gene_ids = available_genes
        else:
            requested_genes = tuple(str(gene) for gene in genes)
            if len(set(requested_genes)) != len(requested_genes):
                raise ValueError("genes contains duplicates")
            lookup: Dict[str, int] = {}
            duplicates = set()
            for index, gene in enumerate(available_genes.tolist()):
                if gene in lookup:
                    duplicates.add(gene)
                else:
                    lookup[gene] = index
            ambiguous = sorted(gene for gene in requested_genes if gene in duplicates)
            if ambiguous:
                raise ValueError("requested genes are ambiguous: %s" % ", ".join(ambiguous))
            missing = sorted(gene for gene in requested_genes if gene not in lookup)
            if missing:
                raise KeyError("requested genes are absent: %s" % ", ".join(missing))
            requested_indices = np.asarray(
                [lookup[gene] for gene in requested_genes], dtype=np.int64
            )
            var_indices = requested_indices
            gene_ids = np.asarray(requested_genes, dtype=np.dtype("U"))

        matrix_source = "X" if layer is None or layer == "X" else str(layer)
        if matrix_source != "X" and matrix_source not in adata.layers:
            raise KeyError("H5AD has no layer %s" % matrix_source)
        obs = adata.obs.iloc[obs_indices]
        cell_ids = np.asarray(adata.obs_names[obs_indices].astype(str), dtype=np.dtype("U"))
        donors = _column_strings(obs, donor_key, obs_indices.size, sample_id or "unknown")
        samples = _column_strings(obs, sample_key, obs_indices.size, sample_id or "unknown")
        labels = _column_strings(obs, cell_type_key, obs_indices.size, "unknown")
        resolved_sample_id = sample_id
        if not resolved_sample_id:
            unique_samples = sorted(set(samples.tolist()))
            resolved_sample_id = unique_samples[0] if len(unique_samples) == 1 else source.stem
        return H5ADSelection(
            path=str(source),
            obs_indices=obs_indices,
            var_indices=var_indices,
            cell_ids=cell_ids,
            gene_ids=gene_ids,
            donor_ids=donors,
            sample_ids=samples,
            cell_type_labels=labels,
            sample_id=resolved_sample_id,
            matrix_source=matrix_source,
        )
    finally:
        _close_backed(adata)


def iter_h5ad_chunks(
    selection: H5ADSelection,
    chunk_size: int = 1024,
) -> Iterator[sparse.csr_matrix]:
    """Yield selected expression as bounded-size CSR row chunks."""

    for block, _ in iter_h5ad_chunks_with_library_sizes(selection, chunk_size=chunk_size):
        yield block


def iter_h5ad_chunks_with_library_sizes(
    selection: H5ADSelection,
    chunk_size: int = 1024,
) -> Iterator[Tuple[sparse.csr_matrix, np.ndarray]]:
    """Yield panel counts and full-transcriptome library sizes together.

    Library sizes are computed before selecting the frozen gene panel.  This
    keeps reference normalization in the same expression space as spatial
    truth, where CPM denominators are the complete measured library.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    anndata = _require_anndata()
    adata = anndata.read_h5ad(selection.path, backed="r")
    try:
        matrix = (
            adata.X if selection.matrix_source == "X" else adata.layers[selection.matrix_source]
        )
        column_order = np.argsort(selection.var_indices)
        sorted_columns = selection.var_indices[column_order]
        restore_order = np.argsort(column_order)
        for start in range(0, selection.obs_indices.size, chunk_size):
            rows = selection.obs_indices[start : start + chunk_size]
            block = matrix[rows]
            to_memory = getattr(block, "to_memory", None)
            if to_memory is not None:
                block = to_memory()
            library_sizes = np.asarray(block.sum(axis=1), dtype=np.float64).reshape(-1)
            # Backed sparse stores require monotonically increasing indices.
            # Reordering happens only after the bounded row block is in memory,
            # preserving the caller's requested gene-panel order.
            block = block[:, sorted_columns]
            block = block[:, restore_order]
            if sparse.issparse(block):
                selected = sparse.csr_matrix(block, dtype=np.float32, copy=True)
            else:
                # Dense-backed inputs are converted one bounded chunk at a
                # time.  The assembled reference remains sparse.
                selected = sparse.csr_matrix(np.asarray(block, dtype=np.float32))
            yield selected, library_sizes
    finally:
        _close_backed(adata)


def selection_to_rna_reference(
    selection: H5ADSelection,
    chunk_size: int = 1024,
) -> RNAReference:
    """Materialize a selection as CSR without a dense full-gene intermediate."""

    pairs = list(iter_h5ad_chunks_with_library_sizes(selection, chunk_size=chunk_size))
    chunks = [block for block, _ in pairs]
    library_sizes = np.concatenate([values for _, values in pairs])
    counts = sparse.vstack(chunks, format="csr", dtype=np.float32)
    return RNAReference(
        sample_id=selection.sample_id,
        cell_ids=selection.cell_ids,
        gene_ids=selection.gene_ids,
        counts=counts,
        library_sizes=library_sizes,
        cell_type_labels=selection.cell_type_labels,
        donor_ids=selection.donor_ids,
        sample_ids=selection.sample_ids,
    )


def load_h5ad_reference(
    path: PathLike,
    chunk_size: int = 1024,
    **selection_kwargs: Any,
) -> RNAReference:
    """Convenience wrapper combining :func:`prepare_h5ad` and CSR loading."""

    selection = prepare_h5ad(path, **selection_kwargs)
    return selection_to_rna_reference(selection, chunk_size=chunk_size)


# Explicit name used by workflow code.
prepare_h5ad_reference = load_h5ad_reference
