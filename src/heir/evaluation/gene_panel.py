"""Leakage-safe gene-panel preparation for the NatCommun development study.

The routines in this module deliberately operate on sufficient statistics rather
than a dense spot-by-gene matrix.  A complete broad CSR matrix can therefore be
read once, summarized donor by donor, and released before the next matrix is
opened.  Fold-local panels are then selected from the same statistics by dropping
the held-out donor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

PANEL_SCHEMA = "heir.natcommun_generative_gene_panel.v1"
PROTOCOL_SCHEMA = "heir.natcommun_generative_development_protocol.v2"
TECHNICAL_PREFIXES = ("MT-", "RPL", "RPS", "HBA", "HBB")


def canonical_sha256(value: object) -> str:
    """Return a stable SHA-256 for a JSON-compatible value."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CSRMatrix:
    """Minimal canonical CSR container used without a SciPy dependency."""

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]

    def validate(self, name: str = "CSR") -> "CSRMatrix":
        data = np.asarray(self.data)
        indices = np.asarray(self.indices)
        indptr = np.asarray(self.indptr)
        rows, columns = self.shape
        if rows < 0 or columns <= 0:
            raise ValueError(f"{name} shape is invalid")
        if data.ndim != 1 or indices.shape != data.shape or indptr.shape != (rows + 1,):
            raise ValueError(f"{name} components are not shape-aligned")
        if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
            raise ValueError(f"{name} data must be finite numeric values")
        if np.any(data <= 0):
            raise ValueError(f"{name} stores non-positive entries")
        if not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(
            indptr.dtype, np.integer
        ):
            raise ValueError(f"{name} indices and indptr must be integral")
        if (
            indptr[0] != 0
            or indptr[-1] != len(data)
            or np.any(np.diff(indptr) < 0)
            or np.any(indices < 0)
            or np.any(indices >= columns)
        ):
            raise ValueError(f"{name} CSR pointers or column indices are invalid")
        for row in range(rows):
            local = indices[indptr[row] : indptr[row + 1]]
            if len(local) > 1 and np.any(np.diff(local) <= 0):
                raise ValueError(f"{name} rows must have sorted unique column indices")
        return self


def project_csr_columns(
    matrix: CSRMatrix,
    columns: Sequence[int] | np.ndarray,
    *,
    row_mask: Sequence[bool] | np.ndarray | None = None,
    output_dtype: np.dtype | type = np.int32,
    max_output_bytes: int = 512 * 1024 * 1024,
) -> np.ndarray:
    """Project CSR columns into a bounded dense matrix, preserving requested order.

    ``max_output_bytes`` is checked before allocation.  A boolean ``row_mask`` can
    prevent ineligible source rows from entering a prepared benchmark artifact.
    """

    matrix.validate()
    selected = np.asarray(columns, dtype=np.int64)
    if selected.ndim != 1 or not len(selected):
        raise ValueError("columns must be a non-empty one-dimensional vector")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("projected columns must be unique")
    if np.any(selected < 0) or np.any(selected >= matrix.shape[1]):
        raise ValueError("projected columns are outside the CSR shape")
    if row_mask is None:
        rows = np.arange(matrix.shape[0], dtype=np.int64)
    else:
        mask = np.asarray(row_mask, dtype=bool)
        if mask.shape != (matrix.shape[0],):
            raise ValueError("row_mask must align with CSR rows")
        rows = np.flatnonzero(mask)
    dtype = np.dtype(output_dtype)
    required_bytes = int(len(rows)) * int(len(selected)) * dtype.itemsize
    if required_bytes > int(max_output_bytes):
        raise MemoryError(
            f"dense CSR projection requires {required_bytes} bytes, above "
            f"max_output_bytes={int(max_output_bytes)}"
        )
    output = np.zeros((len(rows), len(selected)), dtype=dtype)
    lookup = np.full(matrix.shape[1], -1, dtype=np.int32)
    lookup[selected] = np.arange(len(selected), dtype=np.int32)
    for output_row, source_row in enumerate(rows.tolist()):
        start = int(matrix.indptr[source_row])
        stop = int(matrix.indptr[source_row + 1])
        local_columns = lookup[matrix.indices[start:stop]]
        retain = local_columns >= 0
        if retain.any():
            values = matrix.data[start:stop][retain]
            if np.issubdtype(dtype, np.integer):
                if not np.equal(values, np.floor(values)).all():
                    raise ValueError("non-integral CSR data cannot be projected to counts")
                limits = np.iinfo(dtype)
                if np.any(values > limits.max):
                    raise OverflowError("CSR count exceeds requested output dtype")
            output[output_row, local_columns[retain]] = values.astype(dtype, copy=False)
    return output


@dataclass(frozen=True)
class GroupMoments:
    """Sparse normalized-count moments grouped by donor or donor/type."""

    group_ids: tuple[str, ...]
    row_counts: np.ndarray
    sums: np.ndarray
    sums_of_squares: np.ndarray
    nonzero_rows: np.ndarray

    def validate(self, genes: int) -> "GroupMoments":
        groups = len(self.group_ids)
        if len(set(self.group_ids)) != groups:
            raise ValueError("moment group IDs must be unique")
        if self.row_counts.shape != (groups,) or np.any(self.row_counts < 0):
            raise ValueError("moment row counts are malformed")
        expected = (groups, genes)
        for name, value in (
            ("sums", self.sums),
            ("sums_of_squares", self.sums_of_squares),
            ("nonzero_rows", self.nonzero_rows),
        ):
            if value.shape != expected or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError(f"moment {name} is malformed")
        if np.any(self.nonzero_rows > self.row_counts[:, None]):
            raise ValueError("moment detection counts exceed group sizes")
        return self


def normalized_group_moments(
    matrix: CSRMatrix,
    library_sizes: Sequence[float] | np.ndarray,
    group_ids: Sequence[str] | np.ndarray,
    *,
    row_mask: Sequence[bool] | np.ndarray | None = None,
    scale: float = 10_000.0,
) -> GroupMoments:
    """Summarize sparse library-normalized counts without global densification.

    Zero-library rows are excluded.  For split halves, pass the *full* library
    size to each half and to the full count matrix.  Then ``full = A + B`` in the
    normalized space, allowing exact split cross-moments to be recovered later
    while each large CSR is held in memory only once.
    """

    matrix.validate()
    libraries = np.asarray(library_sizes, dtype=np.float64)
    groups_raw = np.asarray(group_ids).astype(str)
    if libraries.shape != (matrix.shape[0],) or groups_raw.shape != (matrix.shape[0],):
        raise ValueError("library sizes and group IDs must align with CSR rows")
    if not np.isfinite(libraries).all() or np.any(libraries < 0):
        raise ValueError("library sizes must be finite and nonnegative")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    if row_mask is None:
        retain = np.ones(matrix.shape[0], dtype=bool)
    else:
        retain = np.asarray(row_mask, dtype=bool)
        if retain.shape != (matrix.shape[0],):
            raise ValueError("row_mask must align with CSR rows")
    retain &= libraries > 0
    retained_group_ids = sorted(set(groups_raw[retain].tolist()))
    if not retained_group_ids:
        raise ValueError("no positive-library rows remain for moments")
    group_lookup = {name: index for index, name in enumerate(retained_group_ids)}
    gene_count = matrix.shape[1]
    shape = (len(retained_group_ids), gene_count)
    sums = np.zeros(shape, dtype=np.float64)
    squares = np.zeros(shape, dtype=np.float64)
    nonzero = np.zeros(shape, dtype=np.int64)
    row_counts = np.zeros(len(retained_group_ids), dtype=np.int64)
    for row in np.flatnonzero(retain).tolist():
        group = group_lookup[str(groups_raw[row])]
        row_counts[group] += 1
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        if stop == start:
            continue
        columns = matrix.indices[start:stop]
        values = matrix.data[start:stop].astype(np.float64) * (scale / libraries[row])
        sums[group, columns] += values
        squares[group, columns] += values * values
        nonzero[group, columns] += 1
    return GroupMoments(
        tuple(retained_group_ids), row_counts, sums, squares, nonzero
    ).validate(gene_count)


@dataclass(frozen=True)
class PanelMomentBundle:
    """All sufficient statistics needed for external and fold-local panels."""

    gene_ids: tuple[str, ...]
    st_full: GroupMoments
    st_half_a: GroupMoments
    st_half_b: GroupMoments
    sc_donor_type: GroupMoments

    def validate(self) -> "PanelMomentBundle":
        genes = len(self.gene_ids)
        if not genes or len(set(self.gene_ids)) != genes or any(not gene for gene in self.gene_ids):
            raise ValueError("gene IDs must be non-empty and unique")
        for moments in (self.st_full, self.st_half_a, self.st_half_b, self.sc_donor_type):
            moments.validate(genes)
        if not (
            self.st_full.group_ids
            == self.st_half_a.group_ids
            == self.st_half_b.group_ids
        ):
            raise ValueError("ST full and half moments must use identical donor groups")
        if not np.array_equal(
            self.st_full.row_counts, self.st_half_a.row_counts
        ) or not np.array_equal(self.st_full.row_counts, self.st_half_b.row_counts):
            raise ValueError("ST full and half moments must cover identical rows")
        if not np.allclose(
            self.st_full.sums,
            self.st_half_a.sums + self.st_half_b.sums,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError("normalized ST split sums do not reconstruct full sums")
        return self


@dataclass(frozen=True)
class GenePanelSelection:
    gene_ids: tuple[str, ...]
    broad_column_indices: tuple[int, ...]
    scores: tuple[float, ...]
    training_donor_ids: tuple[str, ...]
    held_out_donor_id: str | None
    eligible_gene_count: int
    retained_program_genes: tuple[str, ...]
    minimum_split_reliability: float
    metrics_by_gene: Mapping[str, Mapping[str, float | int | bool]]


def _subset_moments(moments: GroupMoments, retain_groups: np.ndarray) -> GroupMoments:
    indices = np.flatnonzero(retain_groups)
    return GroupMoments(
        tuple(moments.group_ids[index] for index in indices),
        moments.row_counts[indices],
        moments.sums[indices],
        moments.sums_of_squares[indices],
        moments.nonzero_rows[indices],
    )


def _centered_sum_of_squares(moments: GroupMoments) -> np.ndarray:
    valid = moments.row_counts > 0
    correction = np.sum(
        moments.sums[valid] * moments.sums[valid] / moments.row_counts[valid, None],
        axis=0,
    )
    return np.maximum(np.sum(moments.sums_of_squares[valid], axis=0) - correction, 0.0)


def _split_reliability(
    full: GroupMoments, first: GroupMoments, second: GroupMoments
) -> np.ndarray:
    if full.group_ids != first.group_ids or full.group_ids != second.group_ids:
        raise ValueError("split reliability requires identical groups")
    cross_by_group = (
        full.sums_of_squares - first.sums_of_squares - second.sums_of_squares
    ) / 2.0
    valid = full.row_counts > 0
    cross = np.sum(cross_by_group[valid], axis=0) - np.sum(
        first.sums[valid]
        * second.sums[valid]
        / full.row_counts[valid, None],
        axis=0,
    )
    first_ss = _centered_sum_of_squares(first)
    second_ss = _centered_sum_of_squares(second)
    denominator = np.sqrt(first_ss * second_ss)
    result = np.zeros_like(cross)
    np.divide(cross, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1.0, 1.0)


def _rank_fraction(values: np.ndarray, gene_ids: Sequence[str], eligible: np.ndarray) -> np.ndarray:
    """Deterministic [0, 1] rank with gene-symbol tie breaking."""

    output = np.zeros(len(values), dtype=np.float64)
    candidates = np.flatnonzero(eligible).tolist()
    candidates.sort(key=lambda index: (float(values[index]), str(gene_ids[index])))
    if len(candidates) == 1:
        output[candidates[0]] = 1.0
    elif candidates:
        denominator = float(len(candidates) - 1)
        for rank, index in enumerate(candidates):
            output[index] = rank / denominator
    return output


def is_technical_gene(gene_id: str) -> bool:
    upper = str(gene_id).upper()
    return any(upper.startswith(prefix) for prefix in TECHNICAL_PREFIXES)


def select_gene_panel(
    bundle: PanelMomentBundle,
    *,
    training_donor_ids: Sequence[str],
    program_genes: Sequence[str] = (),
    panel_size: int = 256,
    held_out_donor_id: str | None = None,
    minimum_detection_donors: int = 2,
    minimum_split_reliability: float = -1.0,
) -> GenePanelSelection:
    """Select a deterministic panel using training-donor sufficient statistics only."""

    bundle.validate()
    if panel_size <= 0 or panel_size > len(bundle.gene_ids):
        raise ValueError("panel_size is outside the broad gene universe")
    training = tuple(sorted(set(str(value) for value in training_donor_ids)))
    if not training or any(not value for value in training):
        raise ValueError("training_donor_ids must be non-empty")
    available = set(bundle.st_full.group_ids)
    if not set(training) <= available:
        raise ValueError("training_donor_ids include donors absent from ST moments")
    if held_out_donor_id is not None and str(held_out_donor_id) in set(training):
        raise ValueError("held-out donor cannot be a training donor")
    if not np.isfinite(minimum_split_reliability) or not -1.0 <= minimum_split_reliability <= 1.0:
        raise ValueError("minimum_split_reliability must be finite in [-1, 1]")

    st_keep = np.asarray([group in set(training) for group in bundle.st_full.group_ids])
    st_full = _subset_moments(bundle.st_full, st_keep)
    st_a = _subset_moments(bundle.st_half_a, st_keep)
    st_b = _subset_moments(bundle.st_half_b, st_keep)
    sc_donors = np.asarray([group.split("|", 1)[0] for group in bundle.sc_donor_type.group_ids])
    sc_keep = np.isin(sc_donors, np.asarray(training))
    if not sc_keep.any():
        raise ValueError("no scRNA donor/type moments remain for training donors")
    sc = _subset_moments(bundle.sc_donor_type, sc_keep)

    reliability = _split_reliability(st_full, st_a, st_b)
    st_rows = max(int(np.sum(st_full.row_counts) - len(st_full.group_ids)), 1)
    st_variance = _centered_sum_of_squares(st_full) / st_rows
    sc_rows = max(int(np.sum(sc.row_counts) - len(sc.group_ids)), 1)
    sc_variance = _centered_sum_of_squares(sc) / sc_rows
    st_detection_donors = np.count_nonzero(st_full.nonzero_rows > 0, axis=0)
    sc_detection_by_donor = np.zeros((len(training), len(bundle.gene_ids)), dtype=bool)
    donor_lookup = {donor: index for index, donor in enumerate(training)}
    for group_index, donor in enumerate(sc_donors[sc_keep].tolist()):
        sc_detection_by_donor[donor_lookup[donor]] |= sc.nonzero_rows[group_index] > 0
    sc_detection_donors = np.count_nonzero(sc_detection_by_donor, axis=0)

    required_donors = min(max(int(minimum_detection_donors), 1), len(training))
    technical = np.asarray([is_technical_gene(gene) for gene in bundle.gene_ids])
    eligible = (
        ~technical
        & (st_detection_donors >= required_donors)
        & (sc_detection_donors >= required_donors)
        & (reliability >= float(minimum_split_reliability))
        & np.isfinite(reliability)
        & np.isfinite(st_variance)
        & np.isfinite(sc_variance)
    )
    if np.count_nonzero(eligible) < panel_size:
        raise ValueError(
            f"only {int(np.count_nonzero(eligible))} eligible genes remain for a "
            f"{panel_size}-gene panel"
        )

    score = (
        0.40 * _rank_fraction(np.log1p(st_variance), bundle.gene_ids, eligible)
        + 0.30 * _rank_fraction(np.maximum(reliability, 0.0), bundle.gene_ids, eligible)
        + 0.30 * _rank_fraction(np.log1p(sc_variance), bundle.gene_ids, eligible)
    )
    index_by_gene = {gene: index for index, gene in enumerate(bundle.gene_ids)}
    retained_program = sorted(
        {
            str(gene)
            for gene in program_genes
            if str(gene) in index_by_gene and eligible[index_by_gene[str(gene)]]
        }
    )
    if len(retained_program) > panel_size:
        raise ValueError("eligible program genes exceed panel_size")
    chosen = [index_by_gene[gene] for gene in retained_program]
    chosen_set = set(chosen)
    ranked = np.flatnonzero(eligible).tolist()
    ranked.sort(key=lambda index: (-float(score[index]), str(bundle.gene_ids[index])))
    chosen.extend(index for index in ranked if index not in chosen_set)
    chosen = chosen[:panel_size]
    # Final order is rank-based for decoder stability; program genes are guaranteed
    # inclusion but do not occupy an artificial prefix in the count matrix.
    chosen.sort(key=lambda index: (-float(score[index]), str(bundle.gene_ids[index])))
    metrics = {
        bundle.gene_ids[index]: {
            "score": float(score[index]),
            "st_split_reliability": float(reliability[index]),
            "st_within_donor_variance": float(st_variance[index]),
            "sc_within_donor_type_variance": float(sc_variance[index]),
            "st_detection_donors": int(st_detection_donors[index]),
            "sc_detection_donors": int(sc_detection_donors[index]),
            "prespecified_program_gene": bundle.gene_ids[index] in set(retained_program),
        }
        for index in chosen
    }
    return GenePanelSelection(
        gene_ids=tuple(bundle.gene_ids[index] for index in chosen),
        broad_column_indices=tuple(int(index) for index in chosen),
        scores=tuple(float(score[index]) for index in chosen),
        training_donor_ids=training,
        held_out_donor_id=None if held_out_donor_id is None else str(held_out_donor_id),
        eligible_gene_count=int(np.count_nonzero(eligible)),
        retained_program_genes=tuple(retained_program),
        minimum_split_reliability=float(minimum_split_reliability),
        metrics_by_gene=metrics,
    )


def panel_artifact(
    selection: GenePanelSelection,
    *,
    source_sha256: str,
    source_path: str,
    mode: str,
    program_gene_source: str,
) -> dict[str, object]:
    """Build a hash-bound exposed-development panel receipt."""

    if mode not in {"external_frozen", "lodo_fold_local"}:
        raise ValueError("panel artifact mode is invalid")
    identity = {
        "gene_ids": list(selection.gene_ids),
        "broad_column_indices": list(selection.broad_column_indices),
        "training_donor_ids": list(selection.training_donor_ids),
        "held_out_donor_id": selection.held_out_donor_id,
    }
    artifact: dict[str, object] = {
        "schema": PANEL_SCHEMA,
        "analysis_status": "exposed_development_only_non_confirmatory",
        "scope": "exposed_development_only_non_confirmatory",
        "mode": mode,
        # Top-level identity fields keep the frozen receipt directly consumable
        # by the benchmark runner; the detailed, hash-bound copy remains below.
        "gene_ids": list(selection.gene_ids),
        "broad_column_indices": list(selection.broad_column_indices),
        "source": {"path": str(source_path), "sha256": str(source_sha256)},
        "selection": {
            "panel_size": len(selection.gene_ids),
            **identity,
            "identity_sha256": canonical_sha256(identity),
            "eligible_gene_count": selection.eligible_gene_count,
            "retained_program_genes": list(selection.retained_program_genes),
            "program_gene_source": str(program_gene_source),
            "technical_prefixes_excluded": list(TECHNICAL_PREFIXES),
            "minimum_ST_split_reliability": selection.minimum_split_reliability,
            "ranking": (
                "0.40_ST_within_donor_variance_rank_plus_0.30_ST_split_reliability_rank_"
                "plus_0.30_scRNA_within_donor_type_variance_rank"
            ),
            "metrics_by_gene": dict(selection.metrics_by_gene),
        },
        "leakage_control": {
            "selection_uses_only_training_donor_counts": True,
            "held_out_ST_or_snRNA_used_for_fold_selection": False,
            "external_panel_uses_exposed_NatCommun_development_data": mode == "external_frozen",
        },
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    validate_panel_artifact(artifact, expected_size=len(selection.gene_ids))
    return artifact


def validate_panel_artifact(artifact: Mapping[str, object], *, expected_size: int = 256) -> None:
    if artifact.get("schema") != PANEL_SCHEMA:
        raise ValueError("gene-panel artifact schema is invalid")
    if artifact.get("analysis_status") != "exposed_development_only_non_confirmatory":
        raise ValueError("gene-panel artifact must disclose development-only exposure")
    source = artifact.get("source")
    selection = artifact.get("selection")
    leakage = artifact.get("leakage_control")
    if not isinstance(source, Mapping) or not isinstance(selection, Mapping) or not isinstance(
        leakage, Mapping
    ):
        raise ValueError("gene-panel artifact sections are malformed")
    source_hash = str(source.get("sha256", ""))
    if len(source_hash) != 64 or any(value not in "0123456789abcdef" for value in source_hash):
        raise ValueError("gene-panel source SHA-256 is invalid")
    genes = selection.get("gene_ids")
    columns = selection.get("broad_column_indices")
    training = selection.get("training_donor_ids")
    if (
        not isinstance(genes, list)
        or len(genes) != expected_size
        or len(set(map(str, genes))) != expected_size
        or any(is_technical_gene(str(gene)) for gene in genes)
    ):
        raise ValueError("gene-panel identities are malformed or technically excluded")
    if (
        selection.get("panel_size") != expected_size
        or not isinstance(columns, list)
        or len(columns) != expected_size
        or len(set(columns)) != expected_size
        or any(not isinstance(value, int) or value < 0 for value in columns)
        or not isinstance(training, list)
        or not training
    ):
        raise ValueError("gene-panel dimensions or training donors are malformed")
    if artifact.get("gene_ids") != genes or artifact.get("broad_column_indices") != columns:
        raise ValueError("top-level panel identity does not match the hash-bound selection")
    identity = {
        "gene_ids": genes,
        "broad_column_indices": columns,
        "training_donor_ids": training,
        "held_out_donor_id": selection.get("held_out_donor_id"),
    }
    if selection.get("identity_sha256") != canonical_sha256(identity):
        raise ValueError("gene-panel identity hash is inconsistent")
    payload = dict(artifact)
    reported_hash = payload.pop("artifact_sha256", None)
    if reported_hash != canonical_sha256(payload):
        raise ValueError("gene-panel artifact hash is inconsistent")
    if leakage.get("selection_uses_only_training_donor_counts") is not True:
        raise ValueError("gene-panel artifact does not enforce training-only selection")


def validate_development_protocol(protocol: Mapping[str, object]) -> None:
    """Fail closed on the scientific boundaries required by the latest plan."""

    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("generative-development protocol schema is invalid")
    if protocol.get("analysis_status") != "exposed_development_only_non_confirmatory":
        raise ValueError("protocol must be marked exposed development-only")
    if protocol.get("latent_dimensions") != 20 or protocol.get("gene_panel_size") != 256:
        raise ValueError("protocol must freeze a 20-D latent and 256-gene panel")
    encoders = protocol.get("encoders")
    crops = protocol.get("image_inputs")
    models = protocol.get("model_arms")
    if not isinstance(encoders, Mapping) or encoders.get("UNI2_h") != "forbidden_not_run":
        raise ValueError("protocol must explicitly forbid UNI2-h")
    if (
        not isinstance(crops, Mapping)
        or crops.get("primary") != "natural_registered_112um_H_optimus_1"
        or crops.get("secondary")
        != "unavailable_not_run_registered_source_lacks_55um_H_optimus_1_embeddings"
        or crops.get("secondary_status") != "required_sensitivity_unavailable_not_run"
    ):
        raise ValueError("protocol image input hierarchy is invalid")
    expected_models = {f"M{index}" for index in range(9)} | {
        "M2_supported",
        "M3_supported",
    }
    if not isinstance(models, Mapping) or set(models) != expected_models:
        raise ValueError("protocol must define M0 through M8 and the Gate-3 support pair")
    baseline = protocol.get("retrieval_baseline")
    if (
        not isinstance(baseline, Mapping)
        or baseline.get("family") != "BLEEP_style_contrastive"
        or "then_same_indication_then_global_emergency_fallback"
        not in str(baseline.get("hard_negative_rule", ""))
        or baseline.get("hard_negative_fallback_reporting")
        != "count_and_fraction_saved_per_fold"
    ):
        raise ValueError("protocol lacks the BLEEP-style retrieval baseline")
    gates = protocol.get("ordered_gates")
    gate_ids = [gate.get("id") for gate in gates if isinstance(gate, Mapping)] if isinstance(
        gates, list
    ) else []
    if not isinstance(gates, list) or gate_ids != [
        "gate_1_core_increment",
        "gate_2_image_and_pairing",
        "gate_3_state_beyond_composition",
        "gate_4_natural_reference_bank_separation",
        "gate_5_measurement_headroom",
    ]:
        raise ValueError("protocol ordered gates are incomplete or out of order")
    gate3 = gates[2]
    gate3_tests = gate3.get("tests", ()) if isinstance(gate3, Mapping) else ()
    proxy_test = (
        "training_only_donor_type_proxy_alignment_ratio_less_than_one_and_"
        "less_than_same_seed_lambda_zero_ratio"
    )
    conditional_test = (
        "M3_supported_less_than_M2_supported_on_spots_with_at_least_0.90_"
        "H_composition_mass_over_matched_supported_types"
    )
    if (
        not isinstance(gate3_tests, list)
        or proxy_test not in gate3_tests
        or conditional_test not in gate3_tests
        or gate3.get("minimum_eligible_spots_per_section") != 3
    ):
        raise ValueError("protocol Gate 3 support matching is incomplete")
    gate1_tests = gates[0].get("tests", ()) if isinstance(gates[0], Mapping) else ()
    if "exact_one_sided_donor_sign_flip_p_value_at_most_0.05" not in gate1_tests:
        raise ValueError("protocol Gate 1 exact significance rule is missing")
    outer = protocol.get("outer_validation")
    if (
        not isinstance(outer, Mapping)
        or outer.get("base_seed") != 1729
        or outer.get("epochs") != 80
        or outer.get("batch_size") != 256
    ):
        raise ValueError("protocol training configuration is not frozen")
    resources = protocol.get("resource_limits")
    oom_rule = (
        "fail_closed_no_batch_change_under_current_identity_any_reduced_"
        "batch_run_requires_new_protocol_and_checkpoint_identity"
    )
    if (
        not isinstance(resources, Mapping)
        or resources.get("out_of_memory_action") != oom_rule
    ):
        raise ValueError("protocol OOM identity rule is invalid")
    boundaries = protocol.get("claim_boundaries")
    if (
        not isinstance(boundaries, Mapping)
        or boundaries.get("iterative_refinement") != "prohibited"
        or boundaries.get("cell_level_claims") != "prohibited"
        or boundaries.get("regional_confirmation") != "requires_independent_cohort"
    ):
        raise ValueError("protocol claim boundaries are incomplete")
