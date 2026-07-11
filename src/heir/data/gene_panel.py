"""Leakage-safe gene-panel selection for the locked snPATHO benchmark.

The selector deliberately separates two kinds of input:

* one manifest-bound NatCommun B1 development reference supplies expression;
* snPATHO references supply gene-name availability metadata only.

No snPATHO expression matrix is accessed by this module.  Candidate statistics
are computed in full-library ``log1p(CPM)`` space before any gene subset is
applied.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from heir.data.manifest import ManifestRecord, load_manifest, split_filter_values
from heir.utils import atomic_json_dump, sha256_file

PANEL_ALGORITHM = "balanced-level1-markers-plus-hvg-v2"
PANEL_SCHEMA_VERSION = 2
EXPRESSION_SPACE_ID = "log1p-cpm-10000-v1"
DEVELOPMENT_COHORT = "mosaic_natcommun_2025"
DEVELOPMENT_DONOR = "B1"
DEVELOPMENT_SECTION = "B1_4"


def _require_anndata() -> Any:
    try:
        import anndata
    except ImportError as exc:
        raise ImportError("anndata is required to build the snPATHO gene panel") from exc
    return anndata


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to read 10x feature names") from exc
    return h5py


def _canonical_gene_hash(genes: Sequence[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for gene in genes:
        digest.update(str(gene).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _close_backed(adata: Any) -> None:
    manager = getattr(adata, "file", None)
    close = getattr(manager, "close", None)
    if close is not None:
        close()


def read_h5ad_gene_names_only(path: Path, gene_key: Optional[str] = None) -> Tuple[str, ...]:
    """Read only variable metadata from an H5AD availability source.

    The function opens the HDF5 ``var`` group directly and never instantiates
    AnnData or obtains ``X``/``layers``.  This keeps locked-cohort access at the
    exact metadata field allowed for compatibility filtering.
    """

    source = Path(path).expanduser().resolve()
    h5py = _require_h5py()
    with h5py.File(source, "r") as handle:
        if "var" not in handle:
            raise KeyError("H5AD has no var metadata group")
        variables = handle["var"]
        key = gene_key
        if key is None:
            key = variables.attrs.get("_index", "_index")
            if isinstance(key, bytes):
                key = key.decode("utf-8")
        if key not in variables:
            raise KeyError("H5AD var has no column %s" % key)
        encoded = variables[key]
        if isinstance(encoded, h5py.Dataset):
            raw = encoded[:]
        elif isinstance(encoded, h5py.Group) and {"codes", "categories"}.issubset(encoded):
            codes = np.asarray(encoded["codes"][:], dtype=np.int64)
            categories = encoded["categories"][:]
            if np.any(codes < 0):
                raise ValueError("H5AD gene-name metadata contains missing values")
            raw = categories[codes]
        else:
            raise ValueError("unsupported H5AD encoding for var/%s" % key)
    genes = tuple(
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in raw
    )
    if not genes:
        raise ValueError("availability H5AD has no genes: %s" % source)
    if len(set(genes)) != len(genes):
        raise ValueError("availability H5AD gene names are not unique: %s" % source)
    return genes


def read_10x_gene_names_only(path: Path) -> Tuple[str, ...]:
    """Read the feature-name dataset of a filtered 10x HDF5 file only."""

    source = Path(path).expanduser().resolve()
    h5py = _require_h5py()
    with h5py.File(source, "r") as handle:
        if "matrix/features/name" not in handle:
            raise KeyError("10x HDF5 has no matrix/features/name dataset")
        raw = handle["matrix/features/name"][:]
    genes = tuple(
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in raw
    )
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("10x feature names must be non-empty and unique: %s" % source)
    return genes


def load_curated_genes(path: Path, expected_count: Optional[int] = 70) -> Tuple[str, ...]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        genes = tuple(
            line.strip().split("\t")[0]
            for line in handle
            if line.strip() and not line.startswith("#")
        )
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("curated genes must be non-empty and unique")
    if expected_count is not None and len(genes) != expected_count:
        raise ValueError("expected %d curated genes, found %d" % (expected_count, len(genes)))
    return genes


def resolve_development_record(manifest_path: Path) -> ManifestRecord:
    """Resolve and validate the sole permitted expression source."""

    manifest = load_manifest(manifest_path, require_folds=True)
    matches = [record for record in manifest if record.section_id == DEVELOPMENT_SECTION]
    if len(matches) != 1:
        raise ValueError(
            "development section %s matched %d manifest records"
            % (DEVELOPMENT_SECTION, len(matches))
        )
    record = matches[0]
    if (
        record.cohort_id != DEVELOPMENT_COHORT
        or record.donor_id != DEVELOPMENT_DONOR
        or record.specimen_id != DEVELOPMENT_DONOR
        or record.block_id != DEVELOPMENT_DONOR
    ):
        raise ValueError("gene-panel expression source must be NatCommun donor/block B1")
    if record.analysis_role != "development" or not record.included:
        raise ValueError("gene-panel expression source must be an included development record")
    if record.spatial_count_matrix_file or record.spatial_coordinate_file:
        raise ValueError("gene-panel development record cannot expose spatial target data")
    if not record.count_matrix_file:
        raise ValueError("gene-panel development record has no count matrix")
    return record


@dataclass(frozen=True)
class DevelopmentExpression:
    genes: Tuple[str, ...]
    labels: np.ndarray
    counts: sparse.csr_matrix
    selected_cell_count: int
    full_gene_count: int


def read_development_expression(
    record: ManifestRecord,
    gene_key: str = "feature_name",
    cell_type_key: str = "Level1",
    chunk_size: int = 512,
) -> DevelopmentExpression:
    """Read the manifest-filtered B1 counts and labels in bounded row chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source = Path(record.count_matrix_file).expanduser().resolve()
    adata = _require_anndata().read_h5ad(str(source), backed="r")
    try:
        if gene_key not in adata.var.columns:
            raise KeyError("development H5AD var has no column %s" % gene_key)
        genes = tuple(str(value) for value in adata.var[gene_key].astype(str).to_numpy())
        if len(set(genes)) != len(genes):
            raise ValueError("development gene symbols must be unique")
        if cell_type_key not in adata.obs.columns:
            raise KeyError("development H5AD obs has no column %s" % cell_type_key)

        mask = np.ones(adata.n_obs, dtype=bool)
        encoded_filters: Dict[str, Tuple[str, ...]] = {}
        if record.donor_filter_key:
            encoded_filters[record.donor_filter_key] = split_filter_values(
                record.donor_filter_value
            )
        if record.sample_filter_key:
            encoded_filters[record.sample_filter_key] = split_filter_values(
                record.sample_filter_value
            )
        if not encoded_filters:
            raise ValueError("development record must bind expression with observation filters")
        for key, accepted in encoded_filters.items():
            if key not in adata.obs.columns:
                raise KeyError("development H5AD obs has no filter column %s" % key)
            mask &= np.isin(
                adata.obs[key].astype(str).to_numpy(),
                np.asarray(accepted, dtype=np.dtype("U")),
            )
        rows = np.flatnonzero(mask).astype(np.int64, copy=False)
        if rows.size == 0:
            raise ValueError("development manifest filters selected no cells")
        labels = np.asarray(
            adata.obs.iloc[rows][cell_type_key].astype(str).to_numpy(), dtype=np.dtype("U")
        )
        if np.any(np.isin(labels, np.asarray(["", "nan", "None"], dtype=np.dtype("U")))):
            raise ValueError("development labels contain missing values")

        matrix = adata.X
        blocks: List[sparse.csr_matrix] = []
        for start in range(0, rows.size, chunk_size):
            block = matrix[rows[start : start + chunk_size]]
            to_memory = getattr(block, "to_memory", None)
            if to_memory is not None:
                block = to_memory()
            if sparse.issparse(block):
                converted = sparse.csr_matrix(block, dtype=np.float32, copy=True)
            else:
                converted = sparse.csr_matrix(np.asarray(block, dtype=np.float32))
            blocks.append(converted)
        counts = sparse.vstack(blocks, format="csr", dtype=np.float32)
    finally:
        _close_backed(adata)

    if counts.nnz and (not np.isfinite(counts.data).all() or np.min(counts.data) < 0):
        raise ValueError("development expression must contain finite non-negative counts")
    return DevelopmentExpression(
        genes=genes,
        labels=labels,
        counts=counts,
        selected_cell_count=int(rows.size),
        full_gene_count=len(genes),
    )


def log1p_cpm(counts: sparse.spmatrix, scale: float = 10_000.0) -> sparse.csr_matrix:
    """Normalize with denominators from every measured gene, then apply log1p."""

    if scale <= 0:
        raise ValueError("CPM scale must be positive")
    matrix = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    if matrix.nnz and (not np.isfinite(matrix.data).all() or np.min(matrix.data) < 0):
        raise ValueError("counts must be finite and non-negative")
    library_sizes = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
    factors = np.zeros_like(library_sizes)
    positive = library_sizes > 0
    factors[positive] = scale / library_sizes[positive]
    normalized = sparse.diags(factors).dot(matrix).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


_NOISE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("mitochondrial", re.compile(r"^MT-", re.IGNORECASE)),
    ("ribosomal", re.compile(r"^(?:RPS|RPL|MRPS|MRPL)\d", re.IGNORECASE)),
    (
        "obvious_pseudogene",
        re.compile(
            r"^(?:LOC\d+|(?:RPS|RPL|MRPS|MRPL)[A-Z0-9-]*P\d*|MT[A-Z0-9-]*P\d+)$",
            re.IGNORECASE,
        ),
    ),
    ("uncharacterized_locus", re.compile(r"^(?:AC|AL|AP)\d{5,}", re.IGNORECASE)),
)


def candidate_exclusion_reason(gene: str) -> Optional[str]:
    for name, pattern in _NOISE_PATTERNS:
        if pattern.search(gene):
            return name
    return None


@dataclass(frozen=True)
class PanelGene:
    gene: str
    selection_category: str
    cell_type: str
    score: Optional[float]


@dataclass(frozen=True)
class PanelSelection:
    genes: Tuple[PanelGene, ...]
    type_counts: Mapping[str, int]
    candidate_counts: Mapping[str, int]


def _column_mean(matrix: sparse.spmatrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0), dtype=np.float64).reshape(-1)


def _hvg_scores(expression: sparse.csr_matrix) -> np.ndarray:
    means = _column_mean(expression)
    squared_means = _column_mean(expression.power(2))
    variances = np.maximum(squared_means - means * means, 0.0)
    dispersions = variances / np.maximum(means, 1e-12)
    return np.log1p(dispersions)


def _bin_normalize_scores(
    raw_scores: np.ndarray,
    means: np.ndarray,
    candidate_indices: np.ndarray,
    genes: Sequence[str],
    bins: int = 20,
) -> np.ndarray:
    result = np.full(raw_scores.shape, -np.inf, dtype=np.float64)
    ordered = sorted(candidate_indices.tolist(), key=lambda index: (means[index], genes[index]))
    for block in np.array_split(np.asarray(ordered, dtype=np.int64), min(bins, len(ordered))):
        if block.size == 0:
            continue
        values = raw_scores[block]
        standard_deviation = float(np.std(values))
        if standard_deviation <= 1e-12:
            result[block] = 0.0
        else:
            result[block] = (values - float(np.mean(values))) / standard_deviation
    return result


def select_gene_panel(
    counts: sparse.spmatrix,
    genes: Sequence[str],
    labels: Sequence[str],
    curated_genes: Sequence[str],
    available_genes: Iterable[str],
    ranking_genes: Optional[Iterable[str]] = None,
    panel_size: int = 500,
    markers_per_type: int = 40,
    minimum_detection: float = 0.01,
    minimum_type_detection: float = 0.05,
) -> PanelSelection:
    """Select curated genes, balanced Level1 markers, then development HVGs."""

    names = tuple(str(gene) for gene in genes)
    cell_types = np.asarray(labels, dtype=np.dtype("U"))
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    if matrix.shape != (cell_types.size, len(names)):
        raise ValueError("counts shape must match labels and genes")
    if len(set(names)) != len(names):
        raise ValueError("development genes must be unique")
    curated = tuple(str(gene) for gene in curated_genes)
    if not curated or len(set(curated)) != len(curated):
        raise ValueError("curated genes must be non-empty and unique")
    if panel_size <= len(curated):
        raise ValueError("panel_size must exceed the curated gene count")
    if markers_per_type <= 0:
        raise ValueError("markers_per_type must be positive")
    if not 0 <= minimum_detection <= 1 or not 0 <= minimum_type_detection <= 1:
        raise ValueError("detection thresholds must be in [0, 1]")

    type_names = tuple(sorted(set(cell_types.tolist())))
    if len(type_names) < 2:
        raise ValueError("balanced marker selection requires at least two cell types")
    marker_slots = markers_per_type * len(type_names)
    if len(curated) + marker_slots > panel_size:
        raise ValueError("curated and balanced marker quotas exceed panel_size")

    available = set(str(gene) for gene in available_genes)
    ranking_available = (
        available if ranking_genes is None else set(str(gene) for gene in ranking_genes)
    )
    if not available.issubset(ranking_available):
        raise ValueError("evaluable genes must be a subset of the development ranking universe")
    development_lookup = {gene: index for index, gene in enumerate(names)}
    missing_curated = sorted(
        gene for gene in curated if gene not in development_lookup or gene not in available
    )
    if missing_curated:
        raise ValueError(
            "curated genes are not available in every source: %s" % ", ".join(missing_curated)
        )

    normalized = log1p_cpm(matrix)
    means = _column_mean(normalized)
    detection = np.asarray((normalized > 0).mean(axis=0), dtype=np.float64).reshape(-1)
    curated_set = set(curated)
    common_mask = np.asarray([gene in available for gene in names], dtype=bool)
    ranking_common_mask = np.asarray([gene in ranking_available for gene in names], dtype=bool)
    noise_reasons = [candidate_exclusion_reason(gene) for gene in names]
    clean_mask = np.asarray(
        [reason is None or gene in curated_set for gene, reason in zip(names, noise_reasons)],
        dtype=bool,
    )
    detected_mask = detection >= minimum_detection
    eligible_mask = common_mask & clean_mask & detected_mask
    ranking_eligible_mask = ranking_common_mask & clean_mask & detected_mask
    eligible_indices = np.flatnonzero(eligible_mask)
    ranking_eligible_indices = np.flatnonzero(ranking_eligible_mask)

    selected: List[PanelGene] = [
        PanelGene(gene=gene, selection_category="curated", cell_type="", score=None)
        for gene in curated
    ]
    selected_names = set(curated)

    marker_rankings: Dict[str, List[Tuple[int, float]]] = {}
    for type_name in type_names:
        inside = cell_types == type_name
        outside = ~inside
        inside_mean = _column_mean(normalized[inside])
        outside_mean = _column_mean(normalized[outside])
        inside_detection = np.asarray(
            (normalized[inside] > 0).mean(axis=0), dtype=np.float64
        ).reshape(-1)
        marker_score = (inside_mean - outside_mean) * np.sqrt(inside_detection)
        candidates = np.flatnonzero(
            eligible_mask & (inside_detection >= minimum_type_detection) & (marker_score > 0)
        )
        marker_rankings[type_name] = sorted(
            ((int(index), float(marker_score[index])) for index in candidates),
            key=lambda item: (-item[1], names[item[0]]),
        )

    marker_counts = {type_name: 0 for type_name in type_names}
    cursors = {type_name: 0 for type_name in type_names}
    while any(count < markers_per_type for count in marker_counts.values()):
        progress = False
        for type_name in type_names:
            if marker_counts[type_name] >= markers_per_type:
                continue
            ranking = marker_rankings[type_name]
            cursor = cursors[type_name]
            while cursor < len(ranking) and names[ranking[cursor][0]] in selected_names:
                cursor += 1
            cursors[type_name] = cursor
            if cursor >= len(ranking):
                continue
            index, score = ranking[cursor]
            gene = names[index]
            selected.append(
                PanelGene(
                    gene=gene,
                    selection_category="level1_marker",
                    cell_type=type_name,
                    score=score,
                )
            )
            selected_names.add(gene)
            marker_counts[type_name] += 1
            cursors[type_name] += 1
            progress = True
        if not progress:
            shortages = {
                type_name: markers_per_type - count
                for type_name, count in marker_counts.items()
                if count < markers_per_type
            }
            raise ValueError("insufficient unique marker candidates: %r" % shortages)

    raw_hvg = _hvg_scores(normalized)
    hvg_scores = _bin_normalize_scores(raw_hvg, means, ranking_eligible_indices, names)
    hvg_candidates = [
        index for index in eligible_indices.tolist() if names[index] not in selected_names
    ]
    hvg_candidates.sort(key=lambda index: (-hvg_scores[index], names[index]))
    required_hvgs = panel_size - len(selected)
    if len(hvg_candidates) < required_hvgs:
        raise ValueError("insufficient clean, detected HVGs to complete the panel")
    for index in hvg_candidates[:required_hvgs]:
        selected.append(
            PanelGene(
                gene=names[index],
                selection_category="hvg",
                cell_type="",
                score=float(hvg_scores[index]),
            )
        )

    reason_counts: Dict[str, int] = {}
    for index, reason in enumerate(noise_reasons):
        if not common_mask[index] or reason is None or names[index] in curated_set:
            continue
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    candidate_counts: Dict[str, int] = {
        "development_genes": len(names),
        "availability_intersection": int(np.sum(common_mask)),
        "ranking_intersection": int(np.sum(ranking_common_mask)),
        "clean_and_detected": int(np.sum(eligible_mask)),
        "ranking_clean_and_detected": int(np.sum(ranking_eligible_mask)),
        "below_minimum_detection": int(np.sum(common_mask & clean_mask & ~detected_mask)),
        "noise_excluded": int(np.sum(common_mask & ~clean_mask)),
    }
    candidate_counts.update(
        {"noise_%s" % reason: count for reason, count in sorted(reason_counts.items())}
    )
    if len(selected) != panel_size or len({item.gene for item in selected}) != panel_size:
        raise AssertionError("panel selection did not produce the requested unique size")
    return PanelSelection(
        genes=tuple(selected),
        type_counts=marker_counts,
        candidate_counts=candidate_counts,
    )


def _atomic_panel_dump(selection: PanelSelection, path: Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("# gene\tselection_category\tcell_type\tdevelopment_score\n")
            for item in selection.genes:
                score = "" if item.score is None else "%.10g" % item.score
                handle.write(
                    "%s\t%s\t%s\t%s\n" % (item.gene, item.selection_category, item.cell_type, score)
                )
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_snpatho_panel(
    manifest_path: Path,
    curated_path: Path,
    availability_h5ads: Sequence[Path],
    evaluation_h5ads: Sequence[Path],
    output_path: Path,
    provenance_path: Path,
    availability_10x: Sequence[Path] = (),
    panel_size: int = 500,
    markers_per_type: int = 40,
    gene_key: str = "feature_name",
    cell_type_key: str = "Level1",
    minimum_detection: float = 0.01,
    minimum_type_detection: float = 0.05,
    expected_curated_count: Optional[int] = 70,
    chunk_size: int = 512,
) -> Mapping[str, Any]:
    """Build, write, and provenance the frozen snPATHO benchmark panel."""

    manifest_source = Path(manifest_path).expanduser().resolve()
    curated_source = Path(curated_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    provenance_output = Path(provenance_path).expanduser().resolve()
    if not availability_h5ads and not availability_10x:
        raise ValueError("at least one gene-name availability source is required")
    if len(evaluation_h5ads) != 3 or len({Path(path).resolve() for path in evaluation_h5ads}) != 3:
        raise ValueError("exactly three unique snPATHO Visium metadata sources are required")

    record = resolve_development_record(manifest_source)
    development = read_development_expression(
        record,
        gene_key=gene_key,
        cell_type_key=cell_type_key,
        chunk_size=chunk_size,
    )
    curated = load_curated_genes(curated_source, expected_count=expected_curated_count)

    inventories: List[Tuple[str, str, Path, Tuple[str, ...]]] = []
    for raw_path in availability_h5ads:
        path = Path(raw_path).expanduser().resolve()
        inventories.append(
            (
                "personalized_reference_availability",
                "h5ad_var_metadata",
                path,
                read_h5ad_gene_names_only(path, gene_key),
            )
        )
    for raw_path in availability_10x:
        path = Path(raw_path).expanduser().resolve()
        inventories.append(
            (
                "personalized_reference_availability",
                "10x_feature_metadata",
                path,
                read_10x_gene_names_only(path),
            )
        )
    for raw_path in evaluation_h5ads:
        path = Path(raw_path).expanduser().resolve()
        inventories.append(
            (
                "locked_visium_gene_availability",
                "h5ad_var_metadata",
                path,
                read_h5ad_gene_names_only(path, gene_key),
            )
        )
    reference_inventories = [
        inventory
        for cohort_role, _, _, inventory in inventories
        if cohort_role == "personalized_reference_availability"
    ]
    evaluation_inventories = [
        inventory
        for cohort_role, _, _, inventory in inventories
        if cohort_role == "locked_visium_gene_availability"
    ]
    reference_intersection = set(reference_inventories[0])
    for inventory in reference_inventories[1:]:
        reference_intersection.intersection_update(inventory)
    evaluation_intersection = set(evaluation_inventories[0])
    for inventory in evaluation_inventories[1:]:
        evaluation_intersection.intersection_update(inventory)
    availability_intersection = reference_intersection & evaluation_intersection

    selection = select_gene_panel(
        counts=development.counts,
        genes=development.genes,
        labels=development.labels,
        curated_genes=curated,
        available_genes=availability_intersection,
        ranking_genes=reference_intersection,
        panel_size=panel_size,
        markers_per_type=markers_per_type,
        minimum_detection=minimum_detection,
        minimum_type_detection=minimum_type_detection,
    )
    _atomic_panel_dump(selection, output)

    label_values, label_counts = np.unique(development.labels, return_counts=True)
    availability_provenance = [
        {
            "path": str(path),
            "cohort_role": cohort_role,
            "format": source_format,
            "access_scope": "gene_names_only",
            "metadata_field": "var/%s" % gene_key
            if source_format == "h5ad_var_metadata"
            else "matrix/features/name",
            "gene_key": gene_key if source_format == "h5ad_var_metadata" else "features/name",
            "gene_count": len(inventory),
            "gene_names_sha256": _canonical_gene_hash(inventory),
            "expression_matrix_accessed": False,
            "observation_metadata_accessed": False,
            # A whole-file hash would read locked expression bytes.  The
            # canonical name-vector hash binds precisely the metadata used.
            "whole_file_sha256_computed": False,
        }
        for cohort_role, source_format, path, inventory in inventories
    ]
    payload: Dict[str, Any] = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "algorithm": {
            "name": PANEL_ALGORITHM,
            "expression_space_id": EXPRESSION_SPACE_ID,
            "panel_size": panel_size,
            "markers_per_level1_type": markers_per_type,
            "minimum_detection": minimum_detection,
            "minimum_type_detection": minimum_type_detection,
            "hvg_score": "mean-binned z-score of log1p(variance/mean)",
            "marker_score": "(mean_in_type - mean_outside_type) * sqrt(detection_in_type)",
            "tie_breaker": "ascending HGNC symbol",
            "selection_order": ["curated", "round_robin_level1_markers", "hvg"],
            "candidate_availability": (
                "rank in the personalized-reference universe, then gate on the intersection "
                "of all personalized-reference and locked-Visium gene inventories"
            ),
            "target_metadata_affects_ranking": False,
        },
        "leakage_policy": {
            "development_expression_accessed": True,
            "snpatho_availability_access": "gene_names_only",
            "snpatho_expression_accessed": False,
            "target_visium_gene_metadata_accessed": True,
            "target_visium_expression_accessed": False,
            "target_visium_observation_metadata_accessed": False,
        },
        "development_source": {
            "manifest_path": str(manifest_source),
            "manifest_sha256": sha256_file(manifest_source),
            "record_id": record.record_id,
            "cohort_id": record.cohort_id,
            "donor_id": record.donor_id,
            "specimen_id": record.specimen_id,
            "block_id": record.block_id,
            "section_id": record.section_id,
            "analysis_role": record.analysis_role,
            "count_matrix_path": str(Path(record.count_matrix_file).resolve()),
            "count_matrix_sha256": sha256_file(record.count_matrix_file),
            "gene_key": gene_key,
            "cell_type_key": cell_type_key,
            "observation_filters": {
                record.donor_filter_key: list(split_filter_values(record.donor_filter_value)),
                record.sample_filter_key: list(split_filter_values(record.sample_filter_value)),
            },
            "selected_cell_count": development.selected_cell_count,
            "full_library_gene_count": development.full_gene_count,
            "level1_counts": {
                str(label): int(count) for label, count in zip(label_values, label_counts)
            },
        },
        "curated_source": {
            "path": str(curated_source),
            "sha256": sha256_file(curated_source),
            "requested_count": len(curated),
            "retained_count": sum(item.selection_category == "curated" for item in selection.genes),
        },
        "availability_sources": availability_provenance,
        "counts": {
            **selection.candidate_counts,
            "availability_source_count": len(inventories),
            "reference_availability_source_count": sum(
                cohort_role == "personalized_reference_availability"
                for cohort_role, _, _, _ in inventories
            ),
            "visium_availability_source_count": sum(
                cohort_role == "locked_visium_gene_availability"
                for cohort_role, _, _, _ in inventories
            ),
            "availability_gene_intersection": len(availability_intersection),
            "reference_gene_intersection": len(reference_intersection),
            "visium_gene_intersection": len(evaluation_intersection),
            "panel_genes": len(selection.genes),
            "curated": len(curated),
            "level1_markers": sum(selection.type_counts.values()),
            "hvg": sum(item.selection_category == "hvg" for item in selection.genes),
        },
        "balanced_marker_counts": dict(selection.type_counts),
        "panel": {
            "path": str(output),
            "sha256": sha256_file(output),
            "gene_names_sha256": _canonical_gene_hash([item.gene for item in selection.genes]),
        },
        "selection": [asdict(item) for item in selection.genes],
    }
    atomic_json_dump(payload, provenance_output)
    return payload


__all__ = [
    "DEVELOPMENT_SECTION",
    "EXPRESSION_SPACE_ID",
    "PANEL_ALGORITHM",
    "DevelopmentExpression",
    "PanelGene",
    "PanelSelection",
    "build_snpatho_panel",
    "candidate_exclusion_reason",
    "load_curated_genes",
    "log1p_cpm",
    "read_10x_gene_names_only",
    "read_development_expression",
    "read_h5ad_gene_names_only",
    "resolve_development_record",
    "select_gene_panel",
]
