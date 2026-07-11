"""Locked Visium truth preparation for spatial-expression evaluation.

The target Visium assay is deliberately converted into a separate, versioned
artifact.  It contains only values required by :mod:`heir`'s evaluator and
explicitly records its locked analysis role.  Training contracts never load
this artifact.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import io as scipy_io
from scipy import sparse

from heir.expression import EXPRESSION_SPACE_ID, EXPRESSION_TARGET_SUM
from heir.image import assign_nuclei_to_visium_spots

from .h5ad import iter_h5ad_chunks, prepare_h5ad

PathLike = Union[str, os.PathLike]

SPATIAL_TRUTH_CONTRACT = "heir.spatial_truth"
SPATIAL_TRUTH_VERSION = 1
LOCKED_TARGET_ROLES = frozenset(
    {
        "validation",
        "spatial_validation",
        "external_validation",
        "test",
        "locked_test",
        "locked_validation",
    }
)
BARCODE_SUFFIX_POLICIES = ("auto", "exact", "strip-export", "strip-gem")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPORT_SUFFIX = re.compile(r"_[0-9]+$")
_GEM_SUFFIX = re.compile(r"-[0-9]+$")


def _strings(values: Iterable[object]) -> np.ndarray:
    sequence = [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]
    width = max((len(value) for value in sequence), default=1)
    return np.asarray(sequence, dtype="<U%d" % width)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    result.setflags(write=False)
    return result


def _decode(values: np.ndarray) -> np.ndarray:
    return _strings(np.asarray(values).reshape(-1).tolist())


@dataclass(frozen=True)
class VisiumCounts:
    """Sparse spot-by-gene raw counts with stable source identities."""

    barcodes: np.ndarray
    gene_names: np.ndarray
    counts: sparse.csr_matrix
    library_sizes: np.ndarray
    matrix_source: str

    def __post_init__(self) -> None:
        barcodes = _strings(self.barcodes)
        genes = _strings(self.gene_names)
        counts = sparse.csr_matrix(self.counts, dtype=np.float32, copy=True)
        library_sizes = np.asarray(self.library_sizes, dtype=np.float64)
        if counts.shape != (len(barcodes), len(genes)):
            raise ValueError("Visium counts must have shape (barcodes, genes)")
        if not len(barcodes) or not len(genes):
            raise ValueError("Visium counts must contain barcodes and genes")
        if len(set(barcodes.tolist())) != len(barcodes):
            raise ValueError("Visium count barcodes must be unique")
        if len(set(genes.tolist())) != len(genes):
            raise ValueError("selected Visium genes must be unique")
        if counts.data.size and (
            not np.isfinite(counts.data).all() or bool((counts.data < 0).any())
        ):
            raise ValueError("Visium counts must be finite and non-negative")
        if counts.data.size and not np.allclose(counts.data, np.rint(counts.data), atol=1.0e-4):
            raise ValueError("Visium input must contain raw count-like values")
        if library_sizes.shape != (len(barcodes),) or not np.isfinite(library_sizes).all():
            raise ValueError("Visium library_sizes must be finite and align to barcodes")
        if bool((library_sizes <= 0).any()):
            raise ValueError("Visium library_sizes must be positive")
        panel_sizes = np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)
        if np.any(library_sizes + 1.0e-4 < panel_sizes):
            raise ValueError("Visium library_sizes cannot be smaller than panel count mass")
        if not self.matrix_source.strip():
            raise ValueError("matrix_source cannot be blank")
        counts.sort_indices()
        object.__setattr__(self, "barcodes", _readonly(barcodes))
        object.__setattr__(self, "gene_names", _readonly(genes))
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "library_sizes", _readonly(library_sizes))


@dataclass(frozen=True)
class TissuePositions:
    """Full-resolution spot coordinates from a Space Ranger positions file."""

    barcodes: np.ndarray
    coordinates_px: np.ndarray
    in_tissue: np.ndarray

    def __post_init__(self) -> None:
        barcodes = _strings(self.barcodes)
        coordinates = np.asarray(self.coordinates_px, dtype=np.float64)
        in_tissue = np.asarray(self.in_tissue, dtype=bool)
        if not len(barcodes) or coordinates.shape != (len(barcodes), 2):
            raise ValueError("tissue positions must contain one x/y pair per barcode")
        if in_tissue.shape != (len(barcodes),):
            raise ValueError("in_tissue must align to tissue-position barcodes")
        if len(set(barcodes.tolist())) != len(barcodes):
            raise ValueError("tissue-position barcodes must be unique")
        if not np.isfinite(coordinates).all():
            raise ValueError("tissue-position coordinates must be finite")
        object.__setattr__(self, "barcodes", _readonly(barcodes))
        object.__setattr__(self, "coordinates_px", _readonly(coordinates))
        object.__setattr__(self, "in_tissue", _readonly(in_tissue))


@dataclass(frozen=True)
class BarcodeAlignment:
    """One global, collision-free suffix policy joining counts to positions."""

    policy: str
    count_indices: np.ndarray
    position_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.policy not in BARCODE_SUFFIX_POLICIES[1:]:
            raise ValueError("resolved barcode policy is invalid")
        counts = np.asarray(self.count_indices, dtype=np.int64)
        positions = np.asarray(self.position_indices, dtype=np.int64)
        if counts.ndim != 1 or positions.shape != counts.shape or not len(counts):
            raise ValueError("barcode alignment must contain aligned index vectors")
        if bool((counts < 0).any()) or bool((positions < 0).any()):
            raise ValueError("barcode alignment indices cannot be negative")
        if len(set(counts.tolist())) != len(counts) or len(set(positions.tolist())) != len(
            positions
        ):
            raise ValueError("barcode alignment must be one-to-one")
        object.__setattr__(self, "count_indices", _readonly(counts))
        object.__setattr__(self, "position_indices", _readonly(positions))


@dataclass(frozen=True)
class SpatialTruthArtifact:
    """Pickle-free contract consumed by ``heir evaluate-spatial``."""

    observed_expression: np.ndarray
    gene_names: np.ndarray
    spot_ids: np.ndarray
    nucleus_ids: np.ndarray
    nucleus_spot_index: np.ndarray
    spot_library_sizes: np.ndarray
    spot_coordinates_px: np.ndarray
    nucleus_spot_distance_px: np.ndarray
    analysis_role: str
    cohort_id: str
    donor_id: str
    specimen_id: str
    block_id: str
    section_id: str
    outer_fold: str
    inner_fold: str
    barcode_suffix_policy: str
    spot_radius_px: float
    source_artifacts: np.ndarray
    source_sha256: np.ndarray
    source_roles: np.ndarray
    expression_space_id: str = EXPRESSION_SPACE_ID

    def __post_init__(self) -> None:
        expression = np.asarray(self.observed_expression, dtype=np.float32)
        genes = _strings(self.gene_names)
        spots = _strings(self.spot_ids)
        nuclei = _strings(self.nucleus_ids)
        index = np.asarray(self.nucleus_spot_index, dtype=np.int64)
        library_sizes = np.asarray(self.spot_library_sizes, dtype=np.float64)
        coordinates = np.asarray(self.spot_coordinates_px, dtype=np.float64)
        distance = np.asarray(self.nucleus_spot_distance_px, dtype=np.float64)
        role = self.analysis_role.strip().lower()
        artifacts = _strings(self.source_artifacts)
        hashes = _strings(self.source_sha256)
        source_roles = _strings(self.source_roles)
        if role not in LOCKED_TARGET_ROLES:
            raise ValueError("spatial truth requires a locked target analysis role")
        if self.expression_space_id != EXPRESSION_SPACE_ID:
            raise ValueError("spatial truth must use the canonical expression space")
        if expression.shape != (len(spots), len(genes)):
            raise ValueError("observed_expression must have shape (spots, genes)")
        if not np.isfinite(expression).all() or bool((expression < 0).any()):
            raise ValueError("observed_expression must be finite and non-negative")
        if len(set(genes.tolist())) != len(genes) or len(set(spots.tolist())) != len(spots):
            raise ValueError("spatial truth genes and spots must be unique")
        if len(set(nuclei.tolist())) != len(nuclei):
            raise ValueError("spatial truth nucleus IDs must be unique")
        if index.shape != (len(nuclei),) or bool((index < -1).any()):
            raise ValueError("nucleus_spot_index must align to nuclei and use -1 for unassigned")
        if index.size and int(index.max(initial=-1)) >= len(spots):
            raise ValueError("nucleus_spot_index references an unavailable spot")
        if library_sizes.shape != (len(spots),) or not np.isfinite(library_sizes).all():
            raise ValueError("spot_library_sizes must be finite and align to spots")
        if bool((library_sizes <= 0).any()):
            raise ValueError("spot_library_sizes must be positive")
        if coordinates.shape != (len(spots), 2) or not np.isfinite(coordinates).all():
            raise ValueError("spot coordinates must be finite and align to spots")
        if distance.shape != (len(nuclei),):
            raise ValueError("nucleus-spot distances must align to nuclei")
        assigned = index >= 0
        if assigned.any() and (
            not np.isfinite(distance[assigned]).all() or bool((distance[assigned] < 0).any())
        ):
            raise ValueError("assigned nucleus-spot distances must be finite and non-negative")
        if not assigned.any():
            raise ValueError("spatial truth must assign at least one nucleus to a spot")
        if self.barcode_suffix_policy not in BARCODE_SUFFIX_POLICIES[1:]:
            raise ValueError("barcode_suffix_policy must be resolved before artifact creation")
        if not np.isfinite(self.spot_radius_px) or self.spot_radius_px <= 0:
            raise ValueError("spot_radius_px must be finite and positive")
        identifiers = (
            self.cohort_id,
            self.donor_id,
            self.specimen_id,
            self.block_id,
            self.section_id,
        )
        if any(not str(value).strip() for value in identifiers):
            raise ValueError("cohort/donor/specimen/block/section identities are required")
        if not (len(artifacts) == len(hashes) == len(source_roles)) or not len(artifacts):
            raise ValueError("source artifacts, hashes, and roles must align")
        if len(set(artifacts.tolist())) != len(artifacts):
            raise ValueError("source artifacts must be unique")
        if any(not _SHA256.fullmatch(value) for value in hashes.tolist()):
            raise ValueError("source_sha256 entries must be lowercase SHA-256 digests")
        allowed_source_roles = {
            "locked_spatial_counts",
            "locked_spatial_coordinates",
            "locked_spatial_scalefactors",
            "sample_segmentation",
            "canonical_gene_panel",
            "shared_manifest",
            "conversion_provenance",
            "manifest_spatial_source",
        }
        if any(value not in allowed_source_roles for value in source_roles.tolist()):
            raise ValueError("source_roles contains an unsupported provenance role")
        if "locked_spatial_counts" not in source_roles.tolist():
            raise ValueError("spatial truth provenance must identify the locked count artifact")
        if "shared_manifest" not in source_roles.tolist():
            raise ValueError("spatial truth provenance must identify its manifest")
        object.__setattr__(self, "observed_expression", _readonly(expression))
        object.__setattr__(self, "gene_names", _readonly(genes))
        object.__setattr__(self, "spot_ids", _readonly(spots))
        object.__setattr__(self, "nucleus_ids", _readonly(nuclei))
        object.__setattr__(self, "nucleus_spot_index", _readonly(index))
        object.__setattr__(self, "spot_library_sizes", _readonly(library_sizes))
        object.__setattr__(self, "spot_coordinates_px", _readonly(coordinates))
        object.__setattr__(self, "nucleus_spot_distance_px", _readonly(distance))
        object.__setattr__(self, "analysis_role", role)
        object.__setattr__(self, "source_artifacts", _readonly(artifacts))
        object.__setattr__(self, "source_sha256", _readonly(hashes))
        object.__setattr__(self, "source_roles", _readonly(source_roles))

    @property
    def assigned_nuclei(self) -> int:
        return int((self.nucleus_spot_index >= 0).sum())

    @property
    def evaluable_spots(self) -> int:
        assigned = self.nucleus_spot_index[self.nucleus_spot_index >= 0]
        return int(np.unique(assigned).size)

    def save_npz(self, path: PathLike) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".npz.tmp", dir=str(destination.parent)
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    __contract__=np.asarray(SPATIAL_TRUTH_CONTRACT, dtype=np.dtype("U")),
                    __version__=np.asarray(SPATIAL_TRUTH_VERSION, dtype=np.int64),
                    observed_expression=self.observed_expression,
                    gene_names=self.gene_names,
                    spot_ids=self.spot_ids,
                    nucleus_ids=self.nucleus_ids,
                    nucleus_spot_index=self.nucleus_spot_index,
                    spot_library_sizes=self.spot_library_sizes,
                    spot_coordinates_px=self.spot_coordinates_px,
                    nucleus_spot_distance_px=self.nucleus_spot_distance_px,
                    expression_space_id=np.asarray(self.expression_space_id, dtype=np.dtype("U")),
                    analysis_role=np.asarray(self.analysis_role, dtype=np.dtype("U")),
                    cohort_id=np.asarray(self.cohort_id, dtype=np.dtype("U")),
                    donor_id=np.asarray(self.donor_id, dtype=np.dtype("U")),
                    specimen_id=np.asarray(self.specimen_id, dtype=np.dtype("U")),
                    block_id=np.asarray(self.block_id, dtype=np.dtype("U")),
                    section_id=np.asarray(self.section_id, dtype=np.dtype("U")),
                    outer_fold=np.asarray(self.outer_fold, dtype=np.dtype("U")),
                    inner_fold=np.asarray(self.inner_fold, dtype=np.dtype("U")),
                    barcode_suffix_policy=np.asarray(
                        self.barcode_suffix_policy, dtype=np.dtype("U")
                    ),
                    spot_radius_px=np.asarray(self.spot_radius_px, dtype=np.float64),
                    source_artifacts=self.source_artifacts,
                    source_sha256=self.source_sha256,
                    source_roles=self.source_roles,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def from_npz(cls, path: PathLike) -> "SpatialTruthArtifact":
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "__contract__",
                "__version__",
                "observed_expression",
                "gene_names",
                "spot_ids",
                "nucleus_ids",
                "nucleus_spot_index",
                "spot_library_sizes",
                "spot_coordinates_px",
                "nucleus_spot_distance_px",
                "expression_space_id",
                "analysis_role",
                "cohort_id",
                "donor_id",
                "specimen_id",
                "block_id",
                "section_id",
                "outer_fold",
                "inner_fold",
                "barcode_suffix_policy",
                "spot_radius_px",
                "source_artifacts",
                "source_sha256",
                "source_roles",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError("spatial truth artifact is missing: %s" % ", ".join(missing))
            if str(np.asarray(archive["__contract__"]).item()) != SPATIAL_TRUTH_CONTRACT:
                raise ValueError("artifact is not HEIR spatial truth")
            if int(np.asarray(archive["__version__"]).item()) != SPATIAL_TRUTH_VERSION:
                raise ValueError("unsupported HEIR spatial-truth version")
            payload = {name: np.array(archive[name], copy=True) for name in required}
        scalar_strings = {
            name: str(np.asarray(payload[name]).item())
            for name in (
                "expression_space_id",
                "analysis_role",
                "cohort_id",
                "donor_id",
                "specimen_id",
                "block_id",
                "section_id",
                "outer_fold",
                "inner_fold",
                "barcode_suffix_policy",
            )
        }
        return cls(
            observed_expression=payload["observed_expression"],
            gene_names=payload["gene_names"],
            spot_ids=payload["spot_ids"],
            nucleus_ids=payload["nucleus_ids"],
            nucleus_spot_index=payload["nucleus_spot_index"],
            spot_library_sizes=payload["spot_library_sizes"],
            spot_coordinates_px=payload["spot_coordinates_px"],
            nucleus_spot_distance_px=payload["nucleus_spot_distance_px"],
            spot_radius_px=float(np.asarray(payload["spot_radius_px"]).item()),
            source_artifacts=payload["source_artifacts"],
            source_sha256=payload["source_sha256"],
            source_roles=payload["source_roles"],
            **scalar_strings,
        )


def _select_genes(
    matrix: sparse.spmatrix,
    available_genes: Sequence[object],
    requested_genes: Sequence[object],
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    available = _strings(available_genes)
    requested = _strings(requested_genes)
    if not len(requested) or len(set(requested.tolist())) != len(requested):
        raise ValueError("canonical gene panel must be non-empty and unique")
    lookup: Dict[str, int] = {}
    duplicate = set()
    for index, gene in enumerate(available.tolist()):
        if gene in lookup:
            duplicate.add(gene)
        else:
            lookup[gene] = index
    ambiguous = sorted(set(requested.tolist()).intersection(duplicate))
    if ambiguous:
        raise ValueError("requested Visium genes are ambiguous: %s" % ", ".join(ambiguous))
    missing = sorted(set(requested.tolist()) - set(lookup))
    if missing:
        raise KeyError("requested Visium genes are absent: %s" % ", ".join(missing))
    order = np.asarray([lookup[value] for value in requested.tolist()], dtype=np.int64)
    return sparse.csr_matrix(matrix[:, order], dtype=np.float32), requested


def read_visium_counts(
    path: PathLike,
    *,
    genes: Sequence[object],
    layer: Optional[str] = None,
    gene_key: Optional[str] = None,
    chunk_size: int = 1024,
) -> VisiumCounts:
    """Read H5AD, 10x HDF5, or a 10x Matrix Market directory."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))
    if source.suffix.lower() == ".h5ad":
        full_selection = prepare_h5ad(
            source,
            genes=None,
            layer=layer,
            sample_id=source.stem,
        )
        library_sizes = np.concatenate(
            [
                np.asarray(chunk.sum(axis=1), dtype=np.float64).reshape(-1)
                for chunk in iter_h5ad_chunks(full_selection, chunk_size=chunk_size)
            ]
        )
        selection = prepare_h5ad(
            source,
            genes=genes,
            gene_key=gene_key,
            layer=layer,
            sample_id=source.stem,
        )
        matrix = sparse.vstack(
            list(iter_h5ad_chunks(selection, chunk_size=chunk_size)),
            format="csr",
            dtype=np.float32,
        )
        return VisiumCounts(
            barcodes=selection.cell_ids,
            gene_names=selection.gene_ids,
            counts=matrix,
            library_sizes=library_sizes,
            matrix_source="h5ad:%s" % selection.matrix_source,
        )
    if source.is_dir():
        barcodes_path = _one_10x_file(source, ("barcodes.tsv", "barcodes.tsv.gz"))
        features_path = _one_10x_file(
            source,
            ("features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"),
        )
        matrix_path = _one_10x_file(source, ("matrix.mtx", "matrix.mtx.gz"))
        barcodes = _read_text_columns(barcodes_path, column=0)
        feature_rows = _read_text_rows(features_path)
        name_column = 1 if len(feature_rows[0]) > 1 else 0
        available_genes = [row[name_column] for row in feature_rows]
        selected_features = np.ones(len(feature_rows), dtype=bool)
        if len(feature_rows[0]) > 2:
            selected_features = np.asarray(
                [row[2] == "Gene Expression" for row in feature_rows], dtype=bool
            )
        handle = (
            gzip.open(matrix_path, "rb") if matrix_path.suffix == ".gz" else matrix_path.open("rb")
        )
        with handle:
            raw = scipy_io.mmread(handle)
        matrix = sparse.csr_matrix(raw, dtype=np.float32)
        if matrix.shape == (len(feature_rows), len(barcodes)):
            matrix = matrix[selected_features].T.tocsr()
        elif matrix.shape == (len(barcodes), len(feature_rows)):
            matrix = matrix[:, selected_features].tocsr()
        else:
            raise ValueError("10x matrix dimensions do not match barcodes/features")
        available = np.asarray(available_genes, dtype=np.dtype("U"))[selected_features]
        library_sizes = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
        selected, selected_genes = _select_genes(matrix, available, genes)
        return VisiumCounts(barcodes, selected_genes, selected, library_sizes, "10x-mtx")
    if source.suffix.lower() in {".h5", ".hdf5"}:
        return _read_10x_h5(source, genes)
    raise ValueError("Visium counts must be H5AD, 10x HDF5, or a 10x matrix directory")


def _read_10x_h5(path: Path, genes: Sequence[object]) -> VisiumCounts:
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required to read 10x HDF5 counts") from error
    with h5py.File(path, "r") as handle:
        if "matrix" not in handle:
            raise ValueError("10x HDF5 has no matrix group")
        group = handle["matrix"]
        required = {"barcodes", "data", "indices", "indptr", "shape", "features"}
        if not required.issubset(group.keys()):
            raise ValueError("10x HDF5 matrix group is incomplete")
        feature_group = group["features"]
        name_key = "name" if "name" in feature_group else "id"
        available = _decode(feature_group[name_key][...])
        selected_features = np.ones(len(available), dtype=bool)
        if "feature_type" in feature_group:
            feature_types = _decode(feature_group["feature_type"][...])
            selected_features = feature_types == "Gene Expression"
        shape = tuple(int(value) for value in group["shape"][...])
        matrix = sparse.csc_matrix(
            (group["data"][...], group["indices"][...], group["indptr"][...]),
            shape=shape,
        )
        barcodes = _decode(group["barcodes"][...])
    if matrix.shape != (len(available), len(barcodes)):
        raise ValueError("10x HDF5 matrix dimensions do not match barcodes/features")
    spot_by_gene = matrix[selected_features].T.tocsr()
    library_sizes = np.asarray(spot_by_gene.sum(axis=1), dtype=np.float64).reshape(-1)
    selected, selected_genes = _select_genes(spot_by_gene, available[selected_features], genes)
    return VisiumCounts(barcodes, selected_genes, selected, library_sizes, "10x-hdf5")


def _one_10x_file(directory: Path, names: Sequence[str]) -> Path:
    candidates = [directory / name for name in names if (directory / name).is_file()]
    if len(candidates) != 1:
        raise ValueError("%s needs exactly one of: %s" % (directory, ", ".join(names)))
    return candidates[0]


def _read_text_rows(path: Path) -> list:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = [line.rstrip("\r\n").split("\t") for line in handle if line.strip()]
    if not rows:
        raise ValueError("text table is empty: %s" % path)
    return rows


def _read_text_columns(path: Path, column: int) -> np.ndarray:
    rows = _read_text_rows(path)
    if any(len(row) <= column for row in rows):
        raise ValueError("text table lacks column %d: %s" % (column, path))
    return _strings(row[column] for row in rows)


def read_tissue_positions(path: PathLike, *, coordinate_scale: float = 1.0) -> TissuePositions:
    """Read headered or legacy headerless Space Ranger tissue positions."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    scale = float(coordinate_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("coordinate_scale must be finite and positive")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        first = handle.readline()
        if not first:
            raise ValueError("tissue-position table is empty")
        delimiter = "\t" if first.count("\t") > first.count(",") else ","
        handle.seek(0)
        first_fields = [value.strip().lower() for value in first.rstrip("\r\n").split(delimiter)]
        has_header = any(
            value in {"barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"}
            for value in first_fields
        )
        if has_header:
            reader = csv.DictReader(handle, delimiter=delimiter)
            rows = list(reader)
            if not reader.fieldnames:
                raise ValueError("tissue-position table has no header")
            lookup = {value.strip().lower(): value for value in reader.fieldnames}
            barcode_key = _position_column(lookup, ("barcode", "barcodes", "spot_id"))
            in_tissue_key = _position_column(lookup, ("in_tissue", "tissue"), required=False)
            x_key = _position_column(
                lookup,
                ("pxl_col_in_fullres", "imagecol", "pixel_x", "x"),
            )
            y_key = _position_column(
                lookup,
                ("pxl_row_in_fullres", "imagerow", "pixel_y", "y"),
            )
            raw = [
                (
                    row[barcode_key],
                    row[in_tissue_key] if in_tissue_key is not None else "1",
                    row[x_key],
                    row[y_key],
                )
                for row in rows
            ]
        else:
            rows = list(csv.reader(handle, delimiter=delimiter))
            if any(len(row) < 6 for row in rows):
                raise ValueError("legacy tissue positions require six columns")
            raw = [(row[0], row[1], row[5], row[4]) for row in rows]
    barcodes = []
    coordinates = []
    tissue = []
    for row_number, (barcode, in_tissue, x_value, y_value) in enumerate(raw, start=1):
        identifier = str(barcode).strip()
        if not identifier:
            raise ValueError("tissue-position row %d has an empty barcode" % row_number)
        try:
            flag = int(float(in_tissue)) != 0
            coordinates.append((float(x_value) * scale, float(y_value) * scale))
        except (TypeError, ValueError) as error:
            raise ValueError("tissue-position row %d has invalid values" % row_number) from error
        barcodes.append(identifier)
        tissue.append(flag)
    return TissuePositions(_strings(barcodes), np.asarray(coordinates), np.asarray(tissue))


def _position_column(
    lookup: Mapping[str, str], aliases: Sequence[str], *, required: bool = True
) -> Optional[str]:
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    if required:
        raise ValueError("tissue-position table lacks column %s" % "/".join(aliases))
    return None


def read_spot_diameter(path: PathLike, *, coordinate_scale: float = 1.0) -> float:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or "spot_diameter_fullres" not in payload:
        raise ValueError("scalefactors JSON lacks spot_diameter_fullres")
    try:
        diameter = float(payload["spot_diameter_fullres"]) * float(coordinate_scale)
    except (TypeError, ValueError) as error:
        raise ValueError("spot_diameter_fullres must be numeric") from error
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("spot_diameter_fullres must be finite and positive")
    return diameter


def _normalized_barcode(value: object, policy: str) -> str:
    barcode = str(value).strip()
    if policy in {"strip-export", "strip-gem"}:
        barcode = _EXPORT_SUFFIX.sub("", barcode)
    if policy == "strip-gem":
        barcode = _GEM_SUFFIX.sub("", barcode)
    return barcode


def align_visium_barcodes(
    count_barcodes: Sequence[object],
    positions: TissuePositions,
    *,
    policy: str = "auto",
) -> BarcodeAlignment:
    """Align every count barcode using one deterministic suffix policy.

    ``auto`` selects the least destructive policy with the largest complete
    overlap (exact, export ``_N`` suffix removal, then 10x gem ``-N`` removal).
    Any normalization collision or unmatched count barcode fails closed.
    """

    requested = str(policy).strip().lower()
    if requested not in BARCODE_SUFFIX_POLICIES:
        raise ValueError("unsupported barcode suffix policy %s" % requested)
    count_values = _strings(count_barcodes)
    candidates = BARCODE_SUFFIX_POLICIES[1:] if requested == "auto" else (requested,)
    scored = []
    errors = []
    for rank, candidate in enumerate(candidates):
        normalized_counts = [_normalized_barcode(value, candidate) for value in count_values]
        normalized_positions = [
            _normalized_barcode(value, candidate) for value in positions.barcodes
        ]
        if len(set(normalized_counts)) != len(normalized_counts):
            errors.append("%s creates count-barcode collisions" % candidate)
            continue
        if len(set(normalized_positions)) != len(normalized_positions):
            errors.append("%s creates tissue-position collisions" % candidate)
            continue
        lookup = {value: index for index, value in enumerate(normalized_positions)}
        matches = np.asarray([lookup.get(value, -1) for value in normalized_counts], dtype=np.int64)
        matched = int((matches >= 0).sum())
        scored.append((matched, -rank, candidate, matches))
    if not scored:
        raise ValueError("barcode suffix policies are ambiguous: %s" % "; ".join(errors))
    _, _, selected, matched_positions = max(scored, key=lambda item: (item[0], item[1]))
    unmatched = np.flatnonzero(matched_positions < 0)
    if len(unmatched):
        examples = ", ".join(count_values[unmatched[:5]].tolist())
        raise ValueError(
            "%d Visium count barcodes lack tissue positions under %s: %s"
            % (len(unmatched), selected, examples)
        )
    retained = positions.in_tissue[matched_positions]
    count_indices = np.flatnonzero(retained).astype(np.int64)
    if not len(count_indices):
        raise ValueError("no matched Visium count barcodes are in tissue")
    return BarcodeAlignment(selected, count_indices, matched_positions[count_indices])


def normalize_panel_counts(
    counts: sparse.spmatrix,
    library_sizes: Sequence[float],
) -> np.ndarray:
    """Convert panel counts using the full-transcriptome library denominator."""

    matrix = sparse.csr_matrix(counts, dtype=np.float32)
    library = np.asarray(library_sizes, dtype=np.float64)
    if library.shape != (matrix.shape[0],):
        raise ValueError("library_sizes must align to count rows")
    if not np.isfinite(library).all() or bool((library <= 0).any()):
        raise ValueError("every retained spot needs positive full-transcriptome count mass")
    panel_mass = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
    if np.any(library + 1.0e-4 < panel_mass):
        raise ValueError("full-transcriptome library size cannot be smaller than panel mass")
    scale = EXPRESSION_TARGET_SUM / library
    normalized = matrix.multiply(scale[:, None]).toarray().astype(np.float32)
    return np.log1p(normalized).astype(np.float32)


def build_spatial_truth(
    *,
    counts: VisiumCounts,
    positions: TissuePositions,
    nucleus_ids: Sequence[object],
    nucleus_coordinates_px: np.ndarray,
    spot_radius_px: float,
    barcode_suffix_policy: str,
    metadata: Mapping[str, str],
    source_artifacts: Sequence[object],
    source_sha256: Sequence[object],
    source_roles: Sequence[object],
) -> SpatialTruthArtifact:
    """Build a locked truth artifact after barcode and disk assignment checks."""

    alignment = align_visium_barcodes(
        counts.barcodes,
        positions,
        policy=barcode_suffix_policy,
    )
    selected_counts = counts.counts[alignment.count_indices]
    selected_library_sizes = counts.library_sizes[alignment.count_indices]
    position_indices = alignment.position_indices
    spot_ids = positions.barcodes[position_indices]
    spot_coordinates = positions.coordinates_px[position_indices]
    assignment = assign_nuclei_to_visium_spots(
        nucleus_coordinates_px,
        spot_coordinates,
        spot_radius=float(spot_radius_px),
        spot_ids=spot_ids,
    )
    required_metadata = {
        "analysis_role",
        "cohort_id",
        "donor_id",
        "specimen_id",
        "block_id",
        "section_id",
        "outer_fold",
        "inner_fold",
    }
    missing = sorted(required_metadata - set(metadata))
    if missing:
        raise ValueError("spatial truth metadata is missing: %s" % ", ".join(missing))
    return SpatialTruthArtifact(
        observed_expression=normalize_panel_counts(selected_counts, selected_library_sizes),
        gene_names=counts.gene_names,
        spot_ids=spot_ids,
        nucleus_ids=_strings(nucleus_ids),
        nucleus_spot_index=assignment.spot_index,
        spot_library_sizes=selected_library_sizes,
        spot_coordinates_px=spot_coordinates,
        nucleus_spot_distance_px=assignment.distance,
        barcode_suffix_policy=alignment.policy,
        spot_radius_px=float(spot_radius_px),
        source_artifacts=_strings(source_artifacts),
        source_sha256=_strings(source_sha256),
        source_roles=_strings(source_roles),
        **{name: str(metadata[name]) for name in required_metadata},
    )


__all__ = [
    "BARCODE_SUFFIX_POLICIES",
    "LOCKED_TARGET_ROLES",
    "SPATIAL_TRUTH_CONTRACT",
    "SPATIAL_TRUTH_VERSION",
    "BarcodeAlignment",
    "SpatialTruthArtifact",
    "TissuePositions",
    "VisiumCounts",
    "align_visium_barcodes",
    "build_spatial_truth",
    "normalize_panel_counts",
    "read_spot_diameter",
    "read_tissue_positions",
    "read_visium_counts",
]
