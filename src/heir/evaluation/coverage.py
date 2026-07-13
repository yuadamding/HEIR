"""Prespecified full-coverage and selective spatial-expression endpoints.

The helpers in this module deliberately do not infer fallbacks, RNA-mass
weights, coverage, or uncertainty from target expression.  Those inputs must be
frozen before locked truth is opened.  A single truth-derived gene mask can then
be reused for every method and paired comparison.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

COVERAGE_EVALUATION_SCHEMA = "heir.coverage_evaluation.v2"
COVERAGE_ENDPOINT_INPUT_CONTRACT = "heir.coverage_endpoint_input"
COVERAGE_ENDPOINT_INPUT_VERSION = 1
COVERAGE_BENCHMARK_PLAN_SCHEMA = "heir.coverage_benchmark_plan.v1"
TRUTH_GENE_MASK_SCHEMA = "heir.truth_gene_mask.v2"
DEFAULT_TRUTH_VARIANCE_THRESHOLD = 1.0e-12


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float_array_sha256(values: np.ndarray) -> str:
    matrix = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": "float64-le", "shape": list(matrix.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def _ordered_ids_sha256(values: np.ndarray, role: str) -> str:
    return _canonical_sha256(
        {
            "role": role,
            "ordered_ids": [str(value) for value in values.tolist()],
        }
    )


def _selection_sha256(cell_ids: np.ndarray, selected: np.ndarray, role: str) -> str:
    return _canonical_sha256(
        {
            "role": role,
            "cell_ids": [str(value) for value in cell_ids.tolist()],
            "selected": np.asarray(selected, dtype=bool).astype(int).tolist(),
        }
    )


def _integer_vector_sha256(values: np.ndarray) -> str:
    vector = np.asarray(values, dtype=np.int64)
    return _canonical_sha256(
        {
            "length": int(len(vector)),
            "values": vector.tolist(),
        }
    )


def _aligned_array_sha256(
    *,
    role: str,
    ordered_ids_sha256: str,
    array_sha256: str,
    cell_to_spot_mapping_sha256: Optional[str] = None,
) -> str:
    payload = {
        "role": role,
        "ordered_ids_sha256": ordered_ids_sha256,
        "array_sha256": array_sha256,
    }
    if cell_to_spot_mapping_sha256 is not None:
        payload["cell_to_spot_mapping_sha256"] = cell_to_spot_mapping_sha256
    return _canonical_sha256(payload)


def _matrix_identity_sha256(
    *,
    role: str,
    ordered_row_ids_sha256: str,
    ordered_column_ids_sha256: str,
    values: np.ndarray,
) -> str:
    return _canonical_sha256(
        {
            "role": role,
            "ordered_row_ids_sha256": ordered_row_ids_sha256,
            "ordered_column_ids_sha256": ordered_column_ids_sha256,
            "array_sha256": _float_array_sha256(values),
        }
    )


def _coverage_provenance_sha256(
    endpoint: str,
    requested_coverage: float,
    realized_coverage: float,
    metadata: Mapping[str, object],
) -> str:
    payload_metadata = {
        str(key): value
        for key, value in metadata.items()
        if str(key) != "coverage_aggregation_sha256"
    }
    return _canonical_sha256(
        {
            "schema": COVERAGE_EVALUATION_SCHEMA,
            "endpoint": endpoint,
            "requested_coverage": float(requested_coverage),
            "realized_coverage": float(realized_coverage),
            "metadata": payload_metadata,
        }
    )


def _identity_vector(
    values: Sequence[object],
    *,
    name: str,
    length: int,
    unique: bool,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape != (length,):
        raise ValueError("%s must contain one ordered identity per row" % name)
    identifiers = np.asarray([str(value) for value in raw.tolist()])
    listed = identifiers.tolist()
    if any(not value for value in listed) or (unique and len(set(listed)) != len(listed)):
        qualifier = "non-empty and unique" if unique else "non-empty"
        raise ValueError("%s must be %s" % (name, qualifier))
    return identifiers


def _cell_to_spot_mapping_sha256(
    cell_ids: np.ndarray,
    spot_ids: np.ndarray,
    spot_index: np.ndarray,
) -> str:
    assignments = [
        None if int(index) < 0 else str(spot_ids[int(index)]) for index in spot_index.tolist()
    ]
    return _canonical_sha256(
        {
            "role": "ordered_cell_to_spot_assignment",
            "cell_ids": cell_ids.tolist(),
            "spot_ids": spot_ids.tolist(),
            "spot_index": np.asarray(spot_index, dtype=np.int64).tolist(),
            "assigned_spot_ids": assignments,
        }
    )


def _require_digest_match(actual: str, expected: Optional[str], name: str) -> None:
    if expected is None:
        return
    required = _validate_sha256(expected, "expected %s sha256" % name)
    if actual != required:
        raise ValueError("%s differs from its prespecified hash" % name)


def _validate_sha256(value: str, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("%s must be a lowercase SHA-256 digest" % name)
    return digest


@dataclass(frozen=True)
class TruthGeneMask:
    """Immutable truth-only gene-selection policy and identity."""

    gene_names: Tuple[str, ...]
    spot_ids: Tuple[str, ...]
    mask: np.ndarray
    spot_mask: np.ndarray
    sha256: str
    variance_threshold: float
    spots_used: int
    ordered_spot_ids_sha256: str
    truth_spot_mask_sha256: str
    truth_expression_sha256: str
    schema: str = TRUTH_GENE_MASK_SCHEMA

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.gene_names)
        spots = tuple(str(value) for value in self.spot_ids)
        selected = np.asarray(self.mask)
        selected_spots = np.asarray(self.spot_mask)
        if self.schema != TRUTH_GENE_MASK_SCHEMA:
            raise ValueError("unsupported truth gene-mask schema")
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("gene_names must be non-empty and unique")
        if not spots or any(not spot for spot in spots) or len(set(spots)) != len(spots):
            raise ValueError("spot_ids must be non-empty and unique")
        if selected.dtype != np.bool_ or selected.shape != (len(names),):
            raise ValueError("truth gene mask must be a boolean vector aligned to gene_names")
        if not bool(selected.any()):
            raise ValueError("truth gene mask cannot be empty")
        if selected_spots.dtype != np.bool_ or selected_spots.shape != (len(spots),):
            raise ValueError("truth spot mask must be a boolean vector")
        threshold = float(self.variance_threshold)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("variance_threshold must be finite and non-negative")
        if int(self.spots_used) < 2:
            raise ValueError("truth gene mask requires at least two spots")
        if int(selected_spots.sum()) != int(self.spots_used):
            raise ValueError("truth spot mask differs from spots_used")
        spot_identifiers = np.asarray(spots)
        ordered_spot_ids_sha256 = _ordered_ids_sha256(
            spot_identifiers,
            "truth_spot_ids",
        )
        if ordered_spot_ids_sha256 != self.ordered_spot_ids_sha256:
            raise ValueError("ordered spot-ID sha256 does not match its contents")
        if (
            _selection_sha256(
                spot_identifiers,
                selected_spots,
                "truth_spot_mask",
            )
            != self.truth_spot_mask_sha256
        ):
            raise ValueError("truth spot-mask sha256 does not match its contents")
        _validate_sha256(self.ordered_spot_ids_sha256, "ordered spot-ID sha256")
        _validate_sha256(self.truth_spot_mask_sha256, "truth spot-mask sha256")
        _validate_sha256(self.truth_expression_sha256, "truth expression sha256")
        _validate_sha256(self.sha256, "truth gene-mask sha256")
        expected = _canonical_sha256(
            {
                "schema": TRUTH_GENE_MASK_SCHEMA,
                "gene_names": list(names),
                "mask": selected.astype(int).tolist(),
                "policy": "finite_across_truth_and_variance_above_threshold",
                "variance_threshold": threshold,
                "spots_used": int(self.spots_used),
                "ordered_spot_ids_sha256": self.ordered_spot_ids_sha256,
                "truth_spot_mask_sha256": self.truth_spot_mask_sha256,
                "truth_expression_sha256": self.truth_expression_sha256,
            }
        )
        if self.sha256 != expected:
            raise ValueError("truth gene-mask sha256 does not match its contents")
        immutable = selected.copy()
        immutable.setflags(write=False)
        immutable_spots = selected_spots.copy()
        immutable_spots.setflags(write=False)
        object.__setattr__(self, "gene_names", names)
        object.__setattr__(self, "spot_ids", spots)
        object.__setattr__(self, "mask", immutable)
        object.__setattr__(self, "spot_mask", immutable_spots)
        object.__setattr__(self, "variance_threshold", threshold)
        object.__setattr__(self, "spots_used", int(self.spots_used))

    @property
    def selected_gene_names(self) -> Tuple[str, ...]:
        return tuple(name for name, selected in zip(self.gene_names, self.mask) if selected)

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "sha256": self.sha256,
            "genes_total": len(self.gene_names),
            "genes_evaluated": int(self.mask.sum()),
            "selected_gene_names": list(self.selected_gene_names),
            "variance_threshold": self.variance_threshold,
            "spots_used": self.spots_used,
            "ordered_spot_ids_sha256": self.ordered_spot_ids_sha256,
            "truth_spot_mask_sha256": self.truth_spot_mask_sha256,
            "truth_expression_sha256": self.truth_expression_sha256,
            "policy": "finite_across_truth_and_variance_above_threshold",
        }


@dataclass(frozen=True)
class CoverageAggregation:
    """One cell-to-spot aggregation with explicit coverage semantics."""

    endpoint: str
    spot_expression: np.ndarray
    spot_mass: np.ndarray
    selected_cells: np.ndarray
    eligible_cells: np.ndarray
    cell_ids: Tuple[str, ...]
    spot_ids: Tuple[str, ...]
    gene_names: Tuple[str, ...]
    requested_coverage: float
    realized_coverage: float
    metadata: Mapping[str, object]
    schema: str = COVERAGE_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        expression = np.asarray(self.spot_expression)
        mass = np.asarray(self.spot_mass)
        selected = np.asarray(self.selected_cells)
        eligible = np.asarray(self.eligible_cells)
        cell_ids = tuple(str(value) for value in self.cell_ids)
        spot_ids = tuple(str(value) for value in self.spot_ids)
        gene_names = tuple(str(value) for value in self.gene_names)
        metadata = dict(self.metadata)
        if self.schema != COVERAGE_EVALUATION_SCHEMA:
            raise ValueError("unsupported coverage-evaluation schema")
        if self.endpoint not in {"full_coverage_type_mean_fallback", "fixed_coverage_selective"}:
            raise ValueError("unsupported coverage endpoint")
        if expression.ndim != 2 or mass.shape != (expression.shape[0],):
            raise ValueError("spot expression and mass are misaligned")
        if len(spot_ids) != expression.shape[0] or len(gene_names) != expression.shape[1]:
            raise ValueError("coverage identities do not align to spot expression")
        if selected.ndim != 1 or eligible.ndim != 1:
            raise ValueError("selected_cells and eligible_cells must be one-dimensional")
        if len(cell_ids) != len(selected):
            raise ValueError("cell_ids must align to selected cells")
        for name, values in (
            ("cell_ids", cell_ids),
            ("spot_ids", spot_ids),
            ("gene_names", gene_names),
        ):
            if any(not value for value in values) or len(set(values)) != len(values):
                raise ValueError("%s must be non-empty and unique" % name)
        if not np.isfinite(expression).all() or np.any(expression < 0):
            raise ValueError("aggregated spot expression must be finite and non-negative")
        if not np.isfinite(mass).all() or np.any(mass < 0):
            raise ValueError("aggregated spot mass must be finite and non-negative")
        if (
            selected.dtype != np.bool_
            or eligible.dtype != np.bool_
            or selected.shape != eligible.shape
        ):
            raise ValueError("selected_cells and eligible_cells must be aligned boolean vectors")
        if bool(np.any(selected & ~eligible)):
            raise ValueError("selected cells must be a subset of eligible cells")
        if not bool(eligible.any()) or not bool(selected.any()):
            raise ValueError("coverage aggregation requires eligible and selected cells")
        requested = float(self.requested_coverage)
        realized = float(self.realized_coverage)
        if not np.isfinite(requested) or not np.isfinite(realized):
            raise ValueError("coverage values must be finite")
        if not (0 < requested <= 1) or not (0 < realized <= 1):
            raise ValueError("coverage values must be within (0, 1]")
        measured = float(selected.sum() / eligible.sum())
        if not np.isclose(measured, realized, rtol=0.0, atol=1.0e-12):
            raise ValueError("realized coverage differs from the selected-cell mask")
        if self.endpoint == "full_coverage_type_mean_fallback" and (
            requested != 1.0 or realized != 1.0 or not np.array_equal(selected, eligible)
        ):
            raise ValueError("full-coverage endpoint must have coverage one")
        if self.endpoint == "fixed_coverage_selective" and not np.isclose(
            requested,
            realized,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("fixed selective endpoint did not attain requested coverage")
        cell_array = np.asarray(cell_ids)
        spot_array = np.asarray(spot_ids)
        gene_array = np.asarray(gene_names)
        ordered_cell_digest = _ordered_ids_sha256(cell_array, "aggregation_cell_ids")
        ordered_spot_digest = _ordered_ids_sha256(spot_array, "aggregation_spot_ids")
        ordered_gene_digest = _ordered_ids_sha256(gene_array, "aggregation_gene_names")
        required = {
            "ordered_cell_ids_sha256": ordered_cell_digest,
            "ordered_spot_ids_sha256": ordered_spot_digest,
            "ordered_gene_names_sha256": ordered_gene_digest,
            "selected_cells_sha256": _selection_sha256(
                cell_array,
                selected,
                "coverage_selected_cells",
            ),
            "eligible_cells_sha256": _selection_sha256(
                cell_array,
                eligible,
                "coverage_eligible_cells",
            ),
            "aggregated_spot_mass_sha256": _aligned_array_sha256(
                role="aggregated_spot_mass",
                ordered_ids_sha256=ordered_spot_digest,
                array_sha256=_float_array_sha256(mass),
            ),
            "aggregated_spot_expression_sha256": _matrix_identity_sha256(
                role="aggregated_spot_expression",
                ordered_row_ids_sha256=ordered_spot_digest,
                ordered_column_ids_sha256=ordered_gene_digest,
                values=expression,
            ),
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError("coverage aggregation %s is stale" % key)
        expected_provenance = _coverage_provenance_sha256(
            self.endpoint,
            requested,
            realized,
            metadata,
        )
        if metadata.get("coverage_aggregation_sha256") != expected_provenance:
            raise ValueError("coverage aggregation provenance hash is stale")
        immutable_expression = np.array(expression, copy=True)
        immutable_expression.setflags(write=False)
        immutable_mass = np.array(mass, copy=True)
        immutable_mass.setflags(write=False)
        immutable_selected = np.array(selected, copy=True)
        immutable_selected.setflags(write=False)
        immutable_eligible = np.array(eligible, copy=True)
        immutable_eligible.setflags(write=False)
        object.__setattr__(self, "spot_expression", immutable_expression)
        object.__setattr__(self, "spot_mass", immutable_mass)
        object.__setattr__(self, "selected_cells", immutable_selected)
        object.__setattr__(self, "eligible_cells", immutable_eligible)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "spot_ids", spot_ids)
        object.__setattr__(self, "gene_names", gene_names)
        object.__setattr__(self, "requested_coverage", requested)
        object.__setattr__(self, "realized_coverage", realized)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def provenance_dict(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "endpoint": self.endpoint,
            "requested_coverage": self.requested_coverage,
            "realized_coverage": self.realized_coverage,
            "metadata": dict(self.metadata),
        }


def build_truth_gene_mask(
    truth_expression: np.ndarray,
    gene_names: Sequence[object],
    *,
    spot_ids: Sequence[object],
    spot_mask: Optional[np.ndarray] = None,
    variance_threshold: float = DEFAULT_TRUTH_VARIANCE_THRESHOLD,
) -> TruthGeneMask:
    """Create one ordered, truth-only mask to share across every method.

    ``spot_mask`` is part of the prespecified endpoint and must be identical for
    all methods.  Prediction values are never consulted.
    """

    truth = np.asarray(truth_expression, dtype=np.float64)
    names = tuple(str(value) for value in gene_names)
    if truth.ndim != 2 or truth.shape[1] != len(names):
        raise ValueError("truth_expression and gene_names are misaligned")
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("gene_names must be non-empty and unique")
    spot_identifiers = _identity_vector(
        spot_ids,
        name="spot_ids",
        length=truth.shape[0],
        unique=True,
    )
    threshold = float(variance_threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("variance_threshold must be finite and non-negative")
    if spot_mask is None:
        rows = np.ones(truth.shape[0], dtype=bool)
    else:
        raw_rows = np.asarray(spot_mask)
        if raw_rows.dtype != np.bool_ or raw_rows.shape != (truth.shape[0],):
            raise ValueError("spot_mask must be a boolean vector aligned to truth rows")
        rows = raw_rows
    if int(rows.sum()) < 2:
        raise ValueError("truth gene mask requires at least two selected spots")
    selected_truth = truth[rows]
    finite = np.isfinite(selected_truth).all(axis=0)
    variance = np.zeros(truth.shape[1], dtype=np.float64)
    if bool(finite.any()):
        variance[finite] = np.var(selected_truth[:, finite], axis=0)
    mask = finite & (variance > threshold)
    if not bool(mask.any()):
        raise ValueError("truth has no finite genes above the variance threshold")
    ordered_spot_ids_digest = _ordered_ids_sha256(spot_identifiers, "truth_spot_ids")
    spot_mask_digest = _selection_sha256(
        spot_identifiers,
        rows,
        "truth_spot_mask",
    )
    truth_expression_digest = _float_array_sha256(truth)
    payload = {
        "schema": TRUTH_GENE_MASK_SCHEMA,
        "gene_names": list(names),
        "mask": mask.astype(int).tolist(),
        "policy": "finite_across_truth_and_variance_above_threshold",
        "variance_threshold": threshold,
        "spots_used": int(rows.sum()),
        "ordered_spot_ids_sha256": ordered_spot_ids_digest,
        "truth_spot_mask_sha256": spot_mask_digest,
        "truth_expression_sha256": truth_expression_digest,
    }
    return TruthGeneMask(
        gene_names=names,
        spot_ids=tuple(spot_identifiers.tolist()),
        mask=mask,
        spot_mask=rows,
        sha256=_canonical_sha256(payload),
        variance_threshold=threshold,
        spots_used=int(rows.sum()),
        ordered_spot_ids_sha256=ordered_spot_ids_digest,
        truth_spot_mask_sha256=spot_mask_digest,
        truth_expression_sha256=truth_expression_digest,
    )


def _aggregation_inputs(
    cell_log_expression: np.ndarray,
    spot_index: np.ndarray,
    num_spots: int,
    cell_rna_mass: np.ndarray,
    cell_ids: Sequence[object],
    spot_ids: Sequence[object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expression = np.asarray(cell_log_expression, dtype=np.float64)
    indices = np.asarray(spot_index)
    mass = np.asarray(cell_rna_mass, dtype=np.float64)
    if expression.ndim != 2 or not expression.shape[0] or not expression.shape[1]:
        raise ValueError("cell_log_expression must be a non-empty cells-by-genes matrix")
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise ValueError("cell_log_expression must be finite and non-negative")
    if not np.issubdtype(indices.dtype, np.integer) or indices.shape != (expression.shape[0],):
        raise ValueError("spot_index must be one integer per cell")
    indices = indices.astype(np.int64, copy=False)
    resolved_spots = int(num_spots)
    if (
        isinstance(num_spots, (bool, np.bool_))
        or resolved_spots != num_spots
        or resolved_spots <= 0
        or np.any(indices < -1)
        or np.any(indices >= resolved_spots)
    ):
        raise ValueError("num_spots or spot_index is invalid")
    if mass.shape != (expression.shape[0],) or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("cell_rna_mass must be finite, non-negative, and aligned")
    cell_identifiers = _identity_vector(
        cell_ids,
        name="cell_ids",
        length=expression.shape[0],
        unique=True,
    )
    spot_identifiers = _identity_vector(
        spot_ids,
        name="spot_ids",
        length=resolved_spots,
        unique=True,
    )
    assigned = indices >= 0
    if bool(np.any(mass[assigned] <= 0)):
        raise ValueError("every spot-assigned cell requires positive prespecified RNA mass")
    eligible = assigned & (mass > 0)
    if not bool(eligible.any()):
        raise ValueError("no cells are eligible for spot aggregation")
    return expression, indices, mass, eligible, cell_identifiers, spot_identifiers


def _aggregate_selected(
    expression: np.ndarray,
    indices: np.ndarray,
    num_spots: int,
    mass: np.ndarray,
    selected: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    weights = mass * np.asarray(selected, dtype=np.float64)
    contributing = (indices >= 0) & (weights > 0)
    spot_mass = np.bincount(
        indices[contributing],
        weights=weights[contributing],
        minlength=int(num_spots),
    )
    linear = np.expm1(expression)
    if not np.isfinite(linear).all():
        raise ValueError("cell expression overflows when converted to linear space")
    sums = np.zeros((int(num_spots), expression.shape[1]), dtype=np.float64)
    np.add.at(
        sums,
        indices[contributing],
        linear[contributing] * weights[contributing, None],
    )
    aggregated = np.zeros_like(sums)
    nonempty = spot_mass > 0
    aggregated[nonempty] = np.log1p(sums[nonempty] / spot_mass[nonempty, None])
    return aggregated.astype(np.float32), spot_mass


def full_coverage_type_mean_aggregation(
    *,
    cell_log_expression: np.ndarray,
    abstain: np.ndarray,
    frozen_type_index: np.ndarray,
    frozen_type_mean_log_expression: np.ndarray,
    cell_ids: Sequence[object],
    spot_ids: Sequence[object],
    gene_names: Sequence[object],
    type_names: Sequence[object],
    spot_index: np.ndarray,
    num_spots: int,
    cell_rna_mass: np.ndarray,
    expected_fallback_matrix_sha256: Optional[str] = None,
    expected_frozen_type_index_sha256: Optional[str] = None,
    expected_abstain_mask_sha256: Optional[str] = None,
    expected_rna_mass_vector_sha256: Optional[str] = None,
    expected_cell_to_spot_mapping_sha256: Optional[str] = None,
) -> CoverageAggregation:
    """Aggregate every eligible cell, replacing abstentions with frozen type means."""

    expression, indices, mass, eligible, cell_identifiers, spot_identifiers = _aggregation_inputs(
        cell_log_expression,
        spot_index,
        num_spots,
        cell_rna_mass,
        cell_ids,
        spot_ids,
    )
    abstained = np.asarray(abstain)
    type_index = np.asarray(frozen_type_index)
    type_means = np.asarray(frozen_type_mean_log_expression, dtype=np.float64)
    if abstained.dtype != np.bool_ or abstained.shape != (expression.shape[0],):
        raise ValueError("abstain must be a boolean vector aligned to cells")
    if not np.issubdtype(type_index.dtype, np.integer) or type_index.shape != (
        expression.shape[0],
    ):
        raise ValueError("frozen_type_index must be one integer per cell")
    type_index = type_index.astype(np.int64, copy=False)
    if type_means.ndim != 2 or type_means.shape[1] != expression.shape[1] or not len(type_means):
        raise ValueError("frozen type means must have types-by-genes shape")
    if not np.isfinite(type_means).all() or np.any(type_means < 0):
        raise ValueError("frozen type means must be finite and non-negative")
    if np.any(type_index[eligible] < 0) or np.any(type_index[eligible] >= len(type_means)):
        raise ValueError("frozen_type_index contains an unsupported type")
    ordered_cell_ids_digest = _ordered_ids_sha256(cell_identifiers, "aggregation_cell_ids")
    ordered_spot_ids_digest = _ordered_ids_sha256(spot_identifiers, "aggregation_spot_ids")
    gene_identifiers = _identity_vector(
        gene_names,
        name="gene_names",
        length=expression.shape[1],
        unique=True,
    )
    type_identifiers = _identity_vector(
        type_names,
        name="type_names",
        length=len(type_means),
        unique=True,
    )
    ordered_gene_ids_digest = _ordered_ids_sha256(gene_identifiers, "aggregation_gene_names")
    ordered_type_ids_digest = _ordered_ids_sha256(type_identifiers, "fallback_type_names")
    mapping_digest = _cell_to_spot_mapping_sha256(
        cell_identifiers,
        spot_identifiers,
        indices,
    )
    fallback_digest = _matrix_identity_sha256(
        role="frozen_type_mean_log_expression",
        ordered_row_ids_sha256=ordered_type_ids_digest,
        ordered_column_ids_sha256=ordered_gene_ids_digest,
        values=type_means,
    )
    type_index_digest = _aligned_array_sha256(
        role="frozen_type_index",
        ordered_ids_sha256=ordered_cell_ids_digest,
        array_sha256=_integer_vector_sha256(type_index),
    )
    abstain_digest = _selection_sha256(
        cell_identifiers,
        abstained,
        "abstain_mask",
    )
    rna_mass_digest = _aligned_array_sha256(
        role="cell_rna_mass",
        ordered_ids_sha256=ordered_cell_ids_digest,
        array_sha256=_float_array_sha256(mass),
        cell_to_spot_mapping_sha256=mapping_digest,
    )
    expression_digest = _matrix_identity_sha256(
        role="cell_log_expression",
        ordered_row_ids_sha256=ordered_cell_ids_digest,
        ordered_column_ids_sha256=ordered_gene_ids_digest,
        values=expression,
    )
    _require_digest_match(
        fallback_digest,
        expected_fallback_matrix_sha256,
        "frozen type-mean fallback",
    )
    _require_digest_match(
        type_index_digest,
        expected_frozen_type_index_sha256,
        "frozen type index",
    )
    _require_digest_match(abstain_digest, expected_abstain_mask_sha256, "abstain mask")
    _require_digest_match(rna_mass_digest, expected_rna_mass_vector_sha256, "RNA-mass vector")
    _require_digest_match(
        mapping_digest,
        expected_cell_to_spot_mapping_sha256,
        "cell-to-spot mapping",
    )
    substituted = expression.copy()
    fallback_cells = abstained & eligible
    substituted[fallback_cells] = type_means[type_index[fallback_cells]]
    spot_expression, spot_mass = _aggregate_selected(
        substituted,
        indices,
        num_spots,
        mass,
        eligible,
    )
    spot_mass_digest = _aligned_array_sha256(
        role="aggregated_spot_mass",
        ordered_ids_sha256=ordered_spot_ids_digest,
        array_sha256=_float_array_sha256(spot_mass),
    )
    spot_expression_digest = _matrix_identity_sha256(
        role="aggregated_spot_expression",
        ordered_row_ids_sha256=ordered_spot_ids_digest,
        ordered_column_ids_sha256=ordered_gene_ids_digest,
        values=spot_expression,
    )
    metadata: Dict[str, object] = {
        "fallback_policy": "frozen_hard_type_mean_for_abstained_cells",
        "fallback_matrix_sha256": fallback_digest,
        "frozen_type_index_sha256": type_index_digest,
        "abstain_mask_sha256": abstain_digest,
        "ordered_cell_ids_sha256": ordered_cell_ids_digest,
        "ordered_spot_ids_sha256": ordered_spot_ids_digest,
        "ordered_gene_names_sha256": ordered_gene_ids_digest,
        "ordered_type_names_sha256": ordered_type_ids_digest,
        "cell_to_spot_mapping_sha256": mapping_digest,
        "cell_log_expression_sha256": expression_digest,
        "eligible_cells": int(eligible.sum()),
        "fallback_cells": int(fallback_cells.sum()),
        "zero_mass_spots": int((spot_mass <= 0).sum()),
        "rna_mass_policy": "prespecified_external_input",
        "rna_mass_vector_sha256": rna_mass_digest,
        "selected_cells_sha256": _selection_sha256(
            cell_identifiers,
            eligible,
            "coverage_selected_cells",
        ),
        "eligible_cells_sha256": _selection_sha256(
            cell_identifiers,
            eligible,
            "coverage_eligible_cells",
        ),
        "aggregated_spot_mass_sha256": spot_mass_digest,
        "aggregated_spot_expression_sha256": spot_expression_digest,
    }
    metadata["coverage_aggregation_sha256"] = _coverage_provenance_sha256(
        "full_coverage_type_mean_fallback",
        1.0,
        1.0,
        metadata,
    )
    return CoverageAggregation(
        endpoint="full_coverage_type_mean_fallback",
        spot_expression=spot_expression,
        spot_mass=spot_mass,
        selected_cells=eligible.copy(),
        eligible_cells=eligible.copy(),
        cell_ids=tuple(cell_identifiers.tolist()),
        spot_ids=tuple(spot_identifiers.tolist()),
        gene_names=tuple(gene_identifiers.tolist()),
        requested_coverage=1.0,
        realized_coverage=1.0,
        metadata=metadata,
    )


def fixed_coverage_selective_aggregation(
    *,
    cell_log_expression: np.ndarray,
    uncertainty: np.ndarray,
    target_coverage: float,
    cell_ids: Sequence[object],
    spot_ids: Sequence[object],
    gene_names: Sequence[object],
    spot_index: np.ndarray,
    num_spots: int,
    cell_rna_mass: np.ndarray,
    expected_uncertainty_vector_sha256: Optional[str] = None,
    expected_rna_mass_vector_sha256: Optional[str] = None,
    expected_cell_to_spot_mapping_sha256: Optional[str] = None,
) -> CoverageAggregation:
    """Aggregate the least-uncertain cells at an exactly attainable coverage.

    Coverage is defined over spot-assigned cells with positive prespecified RNA
    mass.  The target must map to an integer cell count; the function never
    silently rounds a requested coverage.  Stable cell IDs break uncertainty
    ties deterministically.
    """

    expression, indices, mass, eligible, identifiers, spot_identifiers = _aggregation_inputs(
        cell_log_expression,
        spot_index,
        num_spots,
        cell_rna_mass,
        cell_ids,
        spot_ids,
    )
    scores = np.asarray(uncertainty, dtype=np.float64)
    if scores.shape != (expression.shape[0],) or not np.isfinite(scores).all():
        raise ValueError("uncertainty must be one finite value per cell")
    ordered_cell_ids_digest = _ordered_ids_sha256(identifiers, "aggregation_cell_ids")
    ordered_spot_ids_digest = _ordered_ids_sha256(spot_identifiers, "aggregation_spot_ids")
    gene_identifiers = _identity_vector(
        gene_names,
        name="gene_names",
        length=expression.shape[1],
        unique=True,
    )
    ordered_gene_ids_digest = _ordered_ids_sha256(gene_identifiers, "aggregation_gene_names")
    mapping_digest = _cell_to_spot_mapping_sha256(identifiers, spot_identifiers, indices)
    uncertainty_digest = _aligned_array_sha256(
        role="cell_uncertainty",
        ordered_ids_sha256=ordered_cell_ids_digest,
        array_sha256=_float_array_sha256(scores),
    )
    rna_mass_digest = _aligned_array_sha256(
        role="cell_rna_mass",
        ordered_ids_sha256=ordered_cell_ids_digest,
        array_sha256=_float_array_sha256(mass),
        cell_to_spot_mapping_sha256=mapping_digest,
    )
    expression_digest = _matrix_identity_sha256(
        role="cell_log_expression",
        ordered_row_ids_sha256=ordered_cell_ids_digest,
        ordered_column_ids_sha256=ordered_gene_ids_digest,
        values=expression,
    )
    _require_digest_match(
        uncertainty_digest,
        expected_uncertainty_vector_sha256,
        "uncertainty vector",
    )
    _require_digest_match(rna_mass_digest, expected_rna_mass_vector_sha256, "RNA-mass vector")
    _require_digest_match(
        mapping_digest,
        expected_cell_to_spot_mapping_sha256,
        "cell-to-spot mapping",
    )
    coverage = float(target_coverage)
    if not np.isfinite(coverage) or not (0 < coverage <= 1):
        raise ValueError("target_coverage must be finite and within (0, 1]")
    eligible_indices = np.flatnonzero(eligible)
    requested_count = coverage * len(eligible_indices)
    selected_count = int(round(requested_count))
    if selected_count <= 0 or not np.isclose(
        requested_count,
        selected_count,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(
            "target_coverage is not exactly attainable for %d eligible cells"
            % len(eligible_indices)
        )
    order = np.lexsort((identifiers[eligible_indices], scores[eligible_indices]))
    chosen = eligible_indices[order[:selected_count]]
    selected = np.zeros(expression.shape[0], dtype=bool)
    selected[chosen] = True
    realized = float(selected.sum() / eligible.sum())
    if not np.isclose(realized, coverage, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("internal fixed-coverage selection mismatch")
    spot_expression, spot_mass = _aggregate_selected(
        expression,
        indices,
        num_spots,
        mass,
        selected,
    )
    spot_mass_digest = _aligned_array_sha256(
        role="aggregated_spot_mass",
        ordered_ids_sha256=ordered_spot_ids_digest,
        array_sha256=_float_array_sha256(spot_mass),
    )
    spot_expression_digest = _matrix_identity_sha256(
        role="aggregated_spot_expression",
        ordered_row_ids_sha256=ordered_spot_ids_digest,
        ordered_column_ids_sha256=ordered_gene_ids_digest,
        values=spot_expression,
    )
    cutoff = float(np.max(scores[selected]))
    boundary_ties = int(np.sum(eligible & np.isclose(scores, cutoff, rtol=0.0, atol=0.0)))
    metadata = {
        "selection_policy": "ascending_uncertainty_then_stable_cell_id",
        "eligible_cells": int(eligible.sum()),
        "selected_cells": int(selected.sum()),
        "uncertainty_cutoff": cutoff,
        "boundary_uncertainty_tie_count": boundary_ties,
        "selection_sha256": _selection_sha256(
            identifiers,
            selected,
            "fixed_coverage_selection",
        ),
        "selected_cells_sha256": _selection_sha256(
            identifiers,
            selected,
            "coverage_selected_cells",
        ),
        "eligible_cells_sha256": _selection_sha256(
            identifiers,
            eligible,
            "coverage_eligible_cells",
        ),
        "uncertainty_vector_sha256": uncertainty_digest,
        "ordered_cell_ids_sha256": ordered_cell_ids_digest,
        "ordered_spot_ids_sha256": ordered_spot_ids_digest,
        "ordered_gene_names_sha256": ordered_gene_ids_digest,
        "cell_to_spot_mapping_sha256": mapping_digest,
        "cell_log_expression_sha256": expression_digest,
        "zero_mass_spots": int((spot_mass <= 0).sum()),
        "rna_mass_policy": "prespecified_external_input",
        "rna_mass_vector_sha256": rna_mass_digest,
        "aggregated_spot_mass_sha256": spot_mass_digest,
        "aggregated_spot_expression_sha256": spot_expression_digest,
    }
    metadata["coverage_aggregation_sha256"] = _coverage_provenance_sha256(
        "fixed_coverage_selective",
        coverage,
        realized,
        metadata,
    )
    return CoverageAggregation(
        endpoint="fixed_coverage_selective",
        spot_expression=spot_expression,
        spot_mass=spot_mass,
        selected_cells=selected,
        eligible_cells=eligible.copy(),
        cell_ids=tuple(identifiers.tolist()),
        spot_ids=tuple(spot_identifiers.tolist()),
        gene_names=tuple(gene_identifiers.tolist()),
        requested_coverage=coverage,
        realized_coverage=realized,
        metadata=metadata,
    )


def _correlation(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    rank: bool,
    threshold: float,
) -> float:
    if float(np.var(prediction)) <= threshold:
        return 0.0
    statistic = (
        spearmanr(prediction, truth).statistic if rank else pearsonr(prediction, truth).statistic
    )
    return float(statistic) if np.isfinite(statistic) else 0.0


def evaluate_methods_on_truth_gene_mask(
    *,
    aggregations: Mapping[str, CoverageAggregation],
    truth_expression: np.ndarray,
    gene_mask: TruthGeneMask,
    spot_ids: Sequence[object],
    comparison_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[str, object]:
    """Score all methods and paired per-gene differences on one gene mask.

    Because every gene in ``gene_mask`` is truth-variable, a method-constant
    prediction receives correlation zero instead of being dropped by
    ``nanmedian``.  Consequently every summary and paired delta contains exactly
    the same ordered genes.
    """

    truth = np.asarray(truth_expression, dtype=np.float64)
    if (
        truth.ndim != 2
        or truth.shape[1] != len(gene_mask.gene_names)
        or truth.shape[0] != len(gene_mask.spot_mask)
    ):
        raise ValueError("truth_expression is misaligned with the truth gene mask")
    if not aggregations:
        raise ValueError("aggregations cannot be empty")
    spot_identifiers = _identity_vector(
        spot_ids,
        name="spot_ids",
        length=truth.shape[0],
        unique=True,
    )
    if tuple(spot_identifiers.tolist()) != gene_mask.spot_ids:
        raise ValueError("ordered spot_ids differ from the truth gene-mask artifact")
    if _float_array_sha256(truth) != gene_mask.truth_expression_sha256:
        raise ValueError("truth_expression differs from the truth gene-mask artifact")
    if not np.isfinite(truth[gene_mask.spot_mask][:, gene_mask.mask]).all():
        raise ValueError("selected truth expression must be finite")
    methods: Dict[str, Dict[str, object]] = {}
    selected_truth = truth[gene_mask.spot_mask][:, gene_mask.mask]
    selected_names = gene_mask.selected_gene_names
    for raw_name, aggregation in aggregations.items():
        name = str(raw_name)
        if not name:
            raise ValueError("aggregation method names cannot be empty")
        if name in methods:
            raise ValueError("aggregation method names must remain unique as strings")
        if not isinstance(aggregation, CoverageAggregation):
            raise TypeError("aggregation %s must be a provenance-bound CoverageAggregation" % name)
        if aggregation.spot_ids != gene_mask.spot_ids:
            raise ValueError("aggregation %s has different ordered spot identities" % name)
        if aggregation.gene_names != gene_mask.gene_names:
            raise ValueError("aggregation %s has different ordered gene identities" % name)
        if bool(np.any(aggregation.spot_mass[gene_mask.spot_mask] <= 0)):
            raise ValueError("aggregation %s has zero mass in a truth-scored spot" % name)
        prediction = np.asarray(aggregation.spot_expression, dtype=np.float64)
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError("aggregation %s must be finite and match truth shape" % name)
        selected_prediction = prediction[gene_mask.spot_mask][:, gene_mask.mask]
        per_gene = []
        for column, gene in enumerate(selected_names):
            left = selected_prediction[:, column]
            right = selected_truth[:, column]
            per_gene.append(
                {
                    "gene": gene,
                    "pearson": _correlation(
                        left,
                        right,
                        rank=False,
                        threshold=gene_mask.variance_threshold,
                    ),
                    "spearman": _correlation(
                        left,
                        right,
                        rank=True,
                        threshold=gene_mask.variance_threshold,
                    ),
                    "mse": float(np.mean(np.square(left - right))),
                    "prediction_constant": bool(
                        float(np.var(left)) <= gene_mask.variance_threshold
                    ),
                }
            )
        methods[name] = {
            "truth_gene_mask_sha256": gene_mask.sha256,
            "prediction_sha256": aggregation.metadata["aggregated_spot_expression_sha256"],
            "spot_mass_sha256": aggregation.metadata["aggregated_spot_mass_sha256"],
            "coverage_aggregation_sha256": aggregation.metadata["coverage_aggregation_sha256"],
            "coverage": aggregation.provenance_dict(),
            "genes_evaluated": len(per_gene),
            "constant_prediction_policy": "correlation_scored_zero",
            "summary": {
                "median_gene_pearson": float(np.median([row["pearson"] for row in per_gene])),
                "median_gene_spearman": float(np.median([row["spearman"] for row in per_gene])),
                "median_gene_mse": float(np.median([row["mse"] for row in per_gene])),
                "prediction_constant_count": int(
                    sum(bool(row["prediction_constant"]) for row in per_gene)
                ),
            },
            "per_gene": per_gene,
        }

    method_names = tuple(methods)
    pairs = (
        tuple(comparison_pairs)
        if comparison_pairs is not None
        else tuple(combinations(method_names, 2))
    )
    comparisons = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("every comparison pair must contain two method names")
        left_name, right_name = str(pair[0]), str(pair[1])
        if left_name == right_name or left_name not in methods or right_name not in methods:
            raise ValueError("comparison pair contains an unknown or repeated method")
        left_rows = methods[left_name]["per_gene"]
        right_rows = methods[right_name]["per_gene"]
        if not isinstance(left_rows, list) or not isinstance(right_rows, list):
            raise RuntimeError("internal per-gene score representation is invalid")
        deltas = []
        for left, right in zip(left_rows, right_rows):
            if left["gene"] != right["gene"]:
                raise RuntimeError("internal paired gene order differs")
            deltas.append(
                {
                    "gene": left["gene"],
                    "pearson_delta": float(left["pearson"] - right["pearson"]),
                    "spearman_delta": float(left["spearman"] - right["spearman"]),
                    "mse_delta": float(left["mse"] - right["mse"]),
                }
            )
        comparisons.append(
            {
                "left": left_name,
                "right": right_name,
                "direction": "left_minus_right",
                "left_coverage_aggregation_sha256": methods[left_name][
                    "coverage_aggregation_sha256"
                ],
                "right_coverage_aggregation_sha256": methods[right_name][
                    "coverage_aggregation_sha256"
                ],
                "truth_gene_mask_sha256": gene_mask.sha256,
                "genes_evaluated": len(deltas),
                "summary": {
                    "median_pearson_delta": float(
                        np.median([row["pearson_delta"] for row in deltas])
                    ),
                    "median_spearman_delta": float(
                        np.median([row["spearman_delta"] for row in deltas])
                    ),
                    "median_mse_delta": float(np.median([row["mse_delta"] for row in deltas])),
                },
                "per_gene": deltas,
            }
        )
    return {
        "schema": COVERAGE_EVALUATION_SCHEMA,
        "claim_scope": {
            "endpoint": "coverage_aware_common_truth_mask",
            "historical_report_rewrite": False,
            "prediction_requirement": "provenance_bound_coverage_aggregation",
        },
        "truth_gene_mask": gene_mask.to_dict(),
        "methods": methods,
        "paired_comparisons": comparisons,
    }


__all__ = [
    "COVERAGE_BENCHMARK_PLAN_SCHEMA",
    "COVERAGE_ENDPOINT_INPUT_CONTRACT",
    "COVERAGE_ENDPOINT_INPUT_VERSION",
    "COVERAGE_EVALUATION_SCHEMA",
    "DEFAULT_TRUTH_VARIANCE_THRESHOLD",
    "TRUTH_GENE_MASK_SCHEMA",
    "CoverageAggregation",
    "TruthGeneMask",
    "build_truth_gene_mask",
    "evaluate_methods_on_truth_gene_mask",
    "fixed_coverage_selective_aggregation",
    "full_coverage_type_mean_aggregation",
]
