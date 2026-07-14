#!/usr/bin/env python3
"""Donor-held-out NatCommun matched-reference regional experiment.

This is a bounded scientific prototype, not a software-development benchmark.
It evaluates the nine prespecified M0--M8 arms on the same underlying held-out
Visium spots; M8 contributes two split-half rows per spot.  All target transforms,
H&E probes, scRNA-to-ST calibration, routing
probes, fusion hyperparameters, and support thresholds are fitted without the
outer held-out donor.  The experiment is intentionally one step: there is no
iterative-refinement code path.

The source archive is produced by ``build_natcommun_regional_source.py``.
Because the released cohort has no barcode-keyed, outcome-independent spot
type annotation, the runner uses a disclosed train-only coarse routing probe
for M2/M3 and marks the confirmatory within-type residual endpoint blocked.
It never derives held-out spot types from held-out ST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from heir.evaluation.hest_nested_ridge import (
    fit_weighted_ridge_grid,
    grouped_donor_folds,
)
from heir.evaluation.hest_scoring import (
    holm_adjust,
    summarize_paired_donor_effects,
)
from heir.evaluation.reference_fusion import (
    PrototypeBank,
    build_reference_prototypes,
    deterministic_group_derangement,
    donor_section_macro_loss,
    fit_reference_calibrator,
    fit_target_basis,
    reference_only_state,
    variance_preservation,
)

SCHEMA = "heir.natcommun_reference_fusion_report.v1"
SOURCE_SCHEMA = "heir.natcommun_regional_source.v2"
SOURCE_RECEIPT_SCHEMA = "heir.natcommun_regional_source_receipt.v2"
SCOPE = "retrospective_regional_non_authorizing"
EXPECTED_PRIMARY_DONORS = 13
EXPECTED_PRIMARY_DONOR_IDS = (
    "B1",
    "B3",
    "B4",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "L1",
    "L2",
    "L3",
    "L4",
)
EXPECTED_ENCODER_REPOSITORY = "bioptimus/H-optimus-1"
SECONDARY_ENCODER_REPOSITORY = "MahmoodLab/UNI2-h"
REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROTOCOL = REPO_ROOT / "configs/natcommun_matched_regional_protocol.json"
FROZEN_SOURCE_BUILDER = REPO_ROOT / "scripts/build_natcommun_regional_source.py"
FROZEN_HEST_RUNNER = REPO_ROOT / "scripts/benchmark_hest_scientific_reanalysis.py"
FROZEN_REFERENCE_FUSION = REPO_ROOT / "src/heir/evaluation/reference_fusion.py"
FROZEN_NESTED_RIDGE = REPO_ROOT / "src/heir/evaluation/hest_nested_ridge.py"
FROZEN_SCORING = REPO_ROOT / "src/heir/evaluation/hest_scoring.py"
FROZEN_HEST_MEASUREMENT = REPO_ROOT / "src/heir/evaluation/hest_measurement.py"
REFERENCE_MINIMUM_QUALIFIED_CELLS = 50
FAILED_REFERENCE_SENSITIVITY_DONOR = "B2"
HOPTIMUS1_REVISION = "3592cb220dec7a150c5d7813fb56e68bd57473b9"
HOPTIMUS1_MANIFEST_SHA256 = (
    "f6852288e1ae146a4865bf19e38ce994c0be9ce1c2bfa09bdf77747043ac8fd9"
)
HEST_REQUIRED_MORPHOLOGY_TARGETS = (
    "nucleus_area_um2",
    "nucleus_perimeter_um",
    "nucleus_circularity",
    "nucleus_solidity",
    "nucleus_gray_mean",
    "nucleus_hematoxylin_od_mean",
    "nucleus_glcm_contrast",
)
DECISIVE_COMPARISONS = (
    "M3_vs_M0_incremental_reference",
    "M3_vs_M1_image_beyond_reference",
    "M3_vs_M2_continuous_state_beyond_type_routing",
    "M3_vs_M4_exact_pairing",
    "M3_vs_M6_matched_specificity",
    "M3_vs_M7_generic_specificity",
)
MODEL_ARMS = {
    "M0": "frozen_H_and_E_only",
    "M1": "matched_scRNA_only",
    "M2": "H_and_E_type_routing_plus_matched_scRNA",
    "M3": "full_H_and_E_query_plus_matched_scRNA_one_step",
    "M4": "within_section_deranged_H_and_E_plus_matched_scRNA",
    "M5": "blank_or_coordinates_plus_matched_scRNA",
    "M6": "H_and_E_plus_each_hard_wrong_same_indication_scRNA",
    "M7": "H_and_E_plus_pooled_same_indication_scRNA_excluding_query",
    "M8": "cross_fitted_full_depth_corrected_ST_split_half_floor",
}
DEFAULT_RIDGE_ALPHAS = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
DEFAULT_FUSION_ALPHAS = (0.0, 0.1, 0.25, 0.5)
DEFAULT_TEMPERATURES = (0.25, 1.0, 4.0)
PROGRAM_RELIABILITY_MINIMUM_SPEARMAN = 0.20
PROGRAM_RELIABILITY_MINIMUM_DONOR_FRACTION = 0.70
PROGRAM_RELIABILITY_MINIMUM_PROGRAMS = 3
_QUALITY_STRATA_CACHE: dict[int, np.ndarray] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _json_scalar(archive: np.lib.npyio.NpzFile, name: str, default: object = None) -> object:
    if name not in archive.files:
        return default
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"{name} must be a scalar")
    raw = value.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _strings(values: object, name: str, rows: int | None = None) -> np.ndarray:
    result = np.asarray(values).astype(str)
    if result.ndim != 1 or (rows is not None and len(result) != rows):
        raise ValueError(f"{name} must be a row-aligned vector")
    if any(not value for value in result.tolist()):
        raise ValueError(f"{name} contains empty values")
    return result


def _matrix(values: object, name: str, rows: int | None = None) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 2 or not len(result) or (rows is not None and len(result) != rows):
        raise ValueError(f"{name} must be a non-empty row-aligned matrix")
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be numeric and finite")
    return result


@dataclass(frozen=True)
class CSRCounts:
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]

    def row_sums(self, *, chunk_rows: int = 4096) -> np.ndarray:
        output = np.zeros(self.shape[0], dtype=np.int64)
        for start in range(0, self.shape[0], chunk_rows):
            stop = min(start + chunk_rows, self.shape[0])
            left = int(self.indptr[start])
            right = int(self.indptr[stop])
            counts = np.diff(self.indptr[start : stop + 1])
            rows = np.repeat(np.arange(stop - start, dtype=np.int64), counts)
            output[start:stop] = np.bincount(
                rows,
                weights=self.data[left:right],
                minlength=stop - start,
            ).astype(np.int64)
        return output

    def take_rows(self, keep: np.ndarray) -> "CSRCounts":
        selected = np.flatnonzero(np.asarray(keep, dtype=bool))
        parts = [self.data[self.indptr[row] : self.indptr[row + 1]] for row in selected]
        index_parts = [self.indices[self.indptr[row] : self.indptr[row + 1]] for row in selected]
        lengths = np.asarray([len(value) for value in parts], dtype=np.int64)
        indptr = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
        return CSRCounts(
            data=np.concatenate(parts) if parts else np.asarray([], dtype=self.data.dtype),
            indices=(
                np.concatenate(index_parts)
                if index_parts
                else np.asarray([], dtype=self.indices.dtype)
            ),
            indptr=indptr,
            shape=(len(selected), self.shape[1]),
        )

    def dense_columns(self, columns: np.ndarray, *, chunk_rows: int = 4096) -> np.ndarray:
        selected = np.asarray(columns, dtype=np.int64)
        if (
            selected.ndim != 1
            or not len(selected)
            or np.any(selected < 0)
            or np.any(selected >= self.shape[1])
        ):
            raise ValueError("selected CSR columns are invalid")
        lookup = np.full(self.shape[1], -1, dtype=np.int64)
        lookup[selected] = np.arange(len(selected), dtype=np.int64)
        output = np.zeros((self.shape[0], len(selected)), dtype=np.float64)
        for start in range(0, self.shape[0], chunk_rows):
            stop = min(start + chunk_rows, self.shape[0])
            left = int(self.indptr[start])
            right = int(self.indptr[stop])
            counts = np.diff(self.indptr[start : stop + 1])
            rows = np.repeat(np.arange(start, stop, dtype=np.int64), counts)
            local_columns = lookup[self.indices[left:right]]
            retained = local_columns >= 0
            np.add.at(
                output,
                (rows[retained], local_columns[retained]),
                self.data[left:right][retained],
            )
        return output

    def weighted_log_variance(
        self,
        row_mask: np.ndarray,
        total_library: np.ndarray,
        row_weight: np.ndarray,
        *,
        chunk_rows: int = 4096,
    ) -> np.ndarray:
        mask = np.asarray(row_mask, dtype=bool)
        totals = np.asarray(total_library, dtype=np.float64)
        weights = np.asarray(row_weight, dtype=np.float64)
        if (
            mask.shape != (self.shape[0],)
            or totals.shape != mask.shape
            or weights.shape != mask.shape
        ):
            raise ValueError("CSR variance inputs must be row aligned")
        normalized_weights = np.zeros_like(weights)
        normalized_weights[mask] = weights[mask] / weights[mask].sum()
        first = np.zeros(self.shape[1], dtype=np.float64)
        second = np.zeros(self.shape[1], dtype=np.float64)
        for start in range(0, self.shape[0], chunk_rows):
            stop = min(start + chunk_rows, self.shape[0])
            left = int(self.indptr[start])
            right = int(self.indptr[stop])
            counts = np.diff(self.indptr[start : stop + 1])
            rows = np.repeat(np.arange(start, stop, dtype=np.int64), counts)
            retained = mask[rows] & (totals[rows] > 0)
            columns = self.indices[left:right][retained]
            values = np.log1p(self.data[left:right][retained] * (10_000.0 / totals[rows[retained]]))
            local_weight = normalized_weights[rows[retained]]
            first += np.bincount(columns, weights=local_weight * values, minlength=self.shape[1])
            second += np.bincount(
                columns, weights=local_weight * np.square(values), minlength=self.shape[1]
            )
        return np.maximum(second - np.square(first), 0.0)

    def covered_columns(self, row_mask: np.ndarray, *, chunk_rows: int = 4096) -> np.ndarray:
        mask = np.asarray(row_mask, dtype=bool)
        covered = np.zeros(self.shape[1], dtype=bool)
        for start in range(0, self.shape[0], chunk_rows):
            stop = min(start + chunk_rows, self.shape[0])
            left = int(self.indptr[start])
            right = int(self.indptr[stop])
            counts = np.diff(self.indptr[start : stop + 1])
            rows = np.repeat(np.arange(start, stop, dtype=np.int64), counts)
            retained = mask[rows]
            covered[self.indices[left:right][retained]] = True
        return covered

    def column_nonzero_counts(self, row_mask: np.ndarray, *, chunk_rows: int = 4096) -> np.ndarray:
        mask = np.asarray(row_mask, dtype=bool)
        counts_by_column = np.zeros(self.shape[1], dtype=np.int64)
        for start in range(0, self.shape[0], chunk_rows):
            stop = min(start + chunk_rows, self.shape[0])
            left = int(self.indptr[start])
            right = int(self.indptr[stop])
            row_counts = np.diff(self.indptr[start : stop + 1])
            rows = np.repeat(np.arange(start, stop, dtype=np.int64), row_counts)
            columns = self.indices[left:right][mask[rows]]
            counts_by_column += np.bincount(columns, minlength=self.shape[1])
        return counts_by_column


def _load_csr(
    archive: np.lib.npyio.NpzFile,
    prefix: str,
    *,
    expected_rows: int,
    expected_columns: int,
) -> CSRCounts:
    required = [f"{prefix}_{suffix}" for suffix in ("data", "indices", "indptr", "shape")]
    if any(name not in archive.files for name in required):
        raise ValueError(f"source lacks broad CSR components for {prefix}")
    data = np.asarray(archive[f"{prefix}_data"])
    raw_indices = np.asarray(archive[f"{prefix}_indices"])
    raw_indptr = np.asarray(archive[f"{prefix}_indptr"])
    raw_shape = np.asarray(archive[f"{prefix}_shape"])
    if (
        data.dtype != np.int32
        or raw_indices.dtype != np.int32
        or raw_indptr.dtype != np.int64
        or raw_shape.dtype != np.int64
    ):
        raise ValueError(f"{prefix} CSR dtypes must be int32/int32/int64/int64")
    indices = raw_indices
    indptr = raw_indptr
    shape_values = raw_shape
    if shape_values.shape != (2,):
        raise ValueError(f"{prefix}_shape must contain two dimensions")
    shape = (int(shape_values[0]), int(shape_values[1]))
    if shape != (expected_rows, expected_columns):
        raise ValueError(f"{prefix} CSR shape is not row/gene aligned")
    if (
        data.ndim != 1
        or indices.shape != data.shape
        or indptr.shape != (expected_rows + 1,)
        or indptr[0] != 0
        or indptr[-1] != len(data)
        or np.any(np.diff(indptr) < 0)
        or np.any(indices < 0)
        or np.any(indices >= expected_columns)
        or np.any(data <= 0)
    ):
        raise ValueError(f"{prefix} CSR components are malformed")
    for row in range(expected_rows):
        local = indices[indptr[row] : indptr[row + 1]]
        if len(local) > 1 and np.any(np.diff(local) <= 0):
            raise ValueError(f"{prefix} CSR indices must be canonical and sorted")
    return CSRCounts(data, indices, indptr, shape)


@dataclass(frozen=True)
class RegionalSource:
    path: Path
    spot_ids: np.ndarray
    donor_ids: np.ndarray
    section_ids: np.ndarray
    indication_ids: np.ndarray
    image_features: np.ndarray
    blank_image_feature: np.ndarray
    coordinate_features: np.ndarray
    gene_ids: np.ndarray
    st_full: np.ndarray
    st_half_a: np.ndarray
    st_half_b: np.ndarray
    st_total_full: np.ndarray
    st_total_half_a: np.ndarray
    st_total_half_b: np.ndarray
    sc_counts: np.ndarray
    sc_total_counts: np.ndarray
    sc_cell_ids: np.ndarray
    sc_donor_ids: np.ndarray
    sc_indication_ids: np.ndarray
    sc_type_ids: np.ndarray
    sc_n_count: np.ndarray
    sc_n_feature: np.ndarray
    sc_percent_mt: np.ndarray
    program_names: np.ndarray
    program_membership: np.ndarray
    broad_gene_ids: np.ndarray
    st_broad_full: CSRCounts
    st_broad_half_a: CSRCounts
    st_broad_half_b: CSRCounts
    sc_broad_counts: CSRCounts
    source_receipt: Mapping[str, object]
    blank_receipt: Mapping[str, object]
    failed_reference_sensitivity: Mapping[str, object]


def load_source(
    path: Path,
    *,
    expected_primary_donors: int = EXPECTED_PRIMARY_DONORS,
    expected_source_sha256: str | None = None,
) -> RegionalSource:
    """Load and strictly validate the builder's primary-eligible source rows."""

    resolved = path.expanduser().resolve()
    if expected_source_sha256 is not None and _sha256(resolved) != expected_source_sha256:
        raise ValueError("source does not match --expected-source-sha256")
    with np.load(resolved, allow_pickle=False) as archive:
        if (
            "schema_version" not in archive.files
            or str(np.asarray(archive["schema_version"]).reshape(-1)[0]) != SOURCE_SCHEMA
        ):
            raise ValueError("source archive schema_version is missing or unexpected")
        receipt = _json_scalar(archive, "source_receipt_json", {})
        if not isinstance(receipt, Mapping):
            raise ValueError("source_receipt_json must contain a JSON object")
        declared_schema = str(receipt.get("schema", SOURCE_RECEIPT_SCHEMA))
        if declared_schema != SOURCE_RECEIPT_SCHEMA:
            raise ValueError(f"unexpected source schema: {declared_schema}")
        encoder = receipt.get("encoder")
        if (
            not isinstance(encoder, Mapping)
            or str(encoder.get("repository")) != EXPECTED_ENCODER_REPOSITORY
            or str(encoder.get("fine_tuning")) not in {"none", "prohibited"}
            or str(encoder.get("device")) != "cuda"
        ):
            raise ValueError(
                "source must contain frozen CUDA bioptimus/H-optimus-1 image features"
            )
        manifest_sha256 = str(encoder.get("manifest_sha256", ""))
        parity = encoder.get("official_local_parity")
        parity_path = (
            Path(str(parity.get("receipt_path", ""))).expanduser().resolve()
            if isinstance(parity, Mapping)
            else Path("/__missing_hoptimus_parity_receipt__")
        )
        parity_sha256 = str(parity.get("receipt_sha256", "")) if isinstance(parity, Mapping) else ""
        if (
            not manifest_sha256
            or not isinstance(parity, Mapping)
            or parity.get("status") != "passed"
            or parity.get("schema") != "heir.hoptimus1_official_local_parity.v1"
            or str(parity.get("encoder_manifest_sha256")) != manifest_sha256
            or len(parity_sha256) != 64
            or any(character not in "0123456789abcdef" for character in parity_sha256)
            or not parity_path.is_file()
            or _sha256(parity_path) != parity_sha256
        ):
            raise ValueError(
                "official-vs-local H-optimus-1 parity must pass for the exact encoder manifest"
            )
        try:
            parity_payload = json.loads(parity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("H-optimus-1 parity receipt cannot be verified") from error
        if (
            not isinstance(parity_payload, Mapping)
            or parity_payload.get("schema") != "heir.hoptimus1_official_local_parity.v1"
            or parity_payload.get("repository") != EXPECTED_ENCODER_REPOSITORY
            or str(parity_payload.get("revision")) != str(encoder.get("revision"))
            or parity_payload.get("encoder_manifest_sha256") != manifest_sha256
            or parity_payload.get("status") != "passed"
            or parity_payload.get("passed") is not True
        ):
            raise ValueError("H-optimus-1 parity receipt payload is inconsistent")
        roles = receipt.get("encoder_roles")
        if (
            not isinstance(roles, Mapping)
            or not isinstance(roles.get("primary"), Mapping)
            or roles["primary"].get("repository") != EXPECTED_ENCODER_REPOSITORY
            or not isinstance(roles.get("secondary_comparator"), Mapping)
            or roles["secondary_comparator"].get("repository") != SECONDARY_ENCODER_REPOSITORY
            or roles["secondary_comparator"].get("status")
            != "prespecified_not_run_in_primary_source"
        ):
            raise ValueError("source encoder roles do not preserve the prespecified comparator")

        spot_ids_all = _strings(archive["spot_ids"], "spot_ids")
        rows_all = len(spot_ids_all)
        primary = np.asarray(archive["spot_primary_eligible"], dtype=bool)
        if primary.shape != (rows_all,):
            raise ValueError("spot_primary_eligible must align with spots")
        if not primary.any():
            raise ValueError("source has no primary-eligible spots")

        donors_all = _strings(archive["donor_ids"], "donor_ids", rows_all)
        sections_all = _strings(archive["section_ids"], "section_ids", rows_all)
        indications_all = _strings(archive["indication_ids"], "indication_ids", rows_all)
        for section in sorted(set(sections_all[primary].tolist())):
            section_rows = primary & (sections_all == section)
            if len(set(donors_all[section_rows].tolist())) != 1:
                raise ValueError("a primary section spans more than one donor")

        image_all = _matrix(archive["image_features"], "image_features", rows_all)
        coordinates_all = _matrix(archive["coordinate_features"], "coordinate_features", rows_all)
        gene_ids = _strings(archive["gene_ids"], "gene_ids")
        genes = len(gene_ids)
        if len(set(gene_ids.tolist())) != genes:
            raise ValueError("gene_ids must be unique")
        full_all = _matrix(archive["st_counts_full"], "st_counts_full", rows_all)
        half_a_all = _matrix(archive["st_counts_half_a"], "st_counts_half_a", rows_all)
        half_b_all = _matrix(archive["st_counts_half_b"], "st_counts_half_b", rows_all)
        if any(matrix.shape[1] != genes for matrix in (full_all, half_a_all, half_b_all)):
            raise ValueError("spatial count matrices do not align to gene_ids")
        if np.any(full_all < 0) or np.any(half_a_all < 0) or np.any(half_b_all < 0):
            raise ValueError("spatial counts cannot be negative")
        if not np.array_equal(full_all, half_a_all + half_b_all):
            raise ValueError("transcript halves do not exactly reconstruct st_counts_full")
        total_full_all = np.asarray(archive["st_total_umi_counts_full"], dtype=np.float64)
        total_half_a_all = np.asarray(archive["st_total_umi_counts_half_a"], dtype=np.float64)
        total_half_b_all = np.asarray(archive["st_total_umi_counts_half_b"], dtype=np.float64)
        if any(
            value.shape != (rows_all,)
            for value in (total_full_all, total_half_a_all, total_half_b_all)
        ):
            raise ValueError("spatial total-library counts must align to spots")
        if not np.array_equal(total_full_all, total_half_a_all + total_half_b_all):
            raise ValueError("total-library transcript halves do not reconstruct full totals")

        sc_counts_all = _matrix(archive["sc_counts"], "sc_counts")
        sc_rows = len(sc_counts_all)
        if sc_counts_all.shape[1] != genes or np.any(sc_counts_all < 0):
            raise ValueError("sc_counts must be nonnegative and align to gene_ids")
        sc_total_all = np.asarray(archive["sc_total_umi_counts"], dtype=np.float64)
        if sc_total_all.shape != (sc_rows,) or np.any(sc_total_all < sc_counts_all.sum(axis=1)):
            raise ValueError("sc_total_umi_counts must be cell aligned and cover selected genes")
        sc_primary = np.asarray(archive["sc_primary_eligible"], dtype=bool)
        if sc_primary.shape != (sc_rows,) or not sc_primary.any():
            raise ValueError("sc_primary_eligible must align and retain cells")
        sc_types_key = next(
            (
                key
                for key in ("sc_level1_type_ids", "sc_level2_type_ids", "sc_level3_type_ids")
                if key in archive.files
            ),
            None,
        )
        if sc_types_key is None:
            raise ValueError("source lacks a coarse scRNA type vector")
        sc_donors_all = _strings(archive["sc_donor_ids"], "sc_donor_ids", sc_rows)
        sc_indications_all = _strings(archive["sc_indication_ids"], "sc_indication_ids", sc_rows)
        sc_cells_all = _strings(archive["sc_cell_ids"], "sc_cell_ids", sc_rows)
        sc_types_all = _strings(archive[sc_types_key], sc_types_key, sc_rows)

        def optional_numeric(name: str, fallback: np.ndarray) -> np.ndarray:
            if name not in archive.files:
                return fallback.astype(np.float64, copy=True)
            value = np.asarray(archive[name], dtype=np.float64)
            if value.shape != (sc_rows,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite and cell aligned")
            return value

        n_count_key = "sc_n_count" if "sc_n_count" in archive.files else "sc_total_umi_counts"
        n_feature_key = "sc_n_feature" if "sc_n_feature" in archive.files else "sc_n_features_rna"
        n_count_all = optional_numeric(n_count_key, sc_counts_all.sum(axis=1))
        n_feature_all = optional_numeric(n_feature_key, np.count_nonzero(sc_counts_all, axis=1))
        percent_mt_all = optional_numeric("sc_percent_mt", np.zeros(sc_rows))

        program_names = _strings(archive["program_names"], "program_names")
        membership = np.asarray(archive["program_gene_membership"], dtype=bool)
        if membership.shape != (len(program_names), genes) or np.any(membership.sum(axis=1) == 0):
            raise ValueError("program_gene_membership must be nonempty and program/gene aligned")
        broad_gene_ids = _strings(archive["broad_gene_ids"], "broad_gene_ids")
        if len(set(broad_gene_ids.tolist())) != len(broad_gene_ids):
            raise ValueError("broad_gene_ids must be unique")
        broad_columns = len(broad_gene_ids)
        st_broad_full_all = _load_csr(
            archive,
            "st_broad_counts_full",
            expected_rows=rows_all,
            expected_columns=broad_columns,
        )
        st_broad_half_a_all = _load_csr(
            archive,
            "st_broad_counts_half_a",
            expected_rows=rows_all,
            expected_columns=broad_columns,
        )
        st_broad_half_b_all = _load_csr(
            archive,
            "st_broad_counts_half_b",
            expected_rows=rows_all,
            expected_columns=broad_columns,
        )
        if not np.array_equal(
            st_broad_full_all.row_sums(),
            st_broad_half_a_all.row_sums() + st_broad_half_b_all.row_sums(),
        ):
            raise ValueError("broad transcript-half row sums do not reconstruct full counts")
        sc_broad_all = _load_csr(
            archive,
            "sc_broad_counts",
            expected_rows=sc_rows,
            expected_columns=broad_columns,
        )

        blank = np.asarray(archive["blank_image_feature_vector"], dtype=np.float64)
        if blank.shape != (image_all.shape[1],) or not np.isfinite(blank).all():
            raise ValueError("blank_image_feature_vector must align to image feature width")
        blank_receipt = _json_scalar(archive, "blank_image_receipt_json", {})
        if not isinstance(blank_receipt, Mapping) or not blank_receipt:
            raise ValueError("blank image control requires a nonempty provenance receipt")

        coverage_donors = _strings(
            archive["reference_coverage_donor_ids"], "reference_coverage_donor_ids"
        )
        coverage_types = _strings(
            archive["reference_coverage_level1_type_ids"],
            "reference_coverage_level1_type_ids",
            len(coverage_donors),
        )
        coverage_counts = np.asarray(archive["reference_coverage_cell_counts"], dtype=np.int64)
        coverage_primary = np.asarray(
            archive["reference_coverage_primary_eligible"], dtype=bool
        )
        if (
            coverage_counts.shape != (len(coverage_donors),)
            or coverage_primary.shape != coverage_counts.shape
            or np.any(coverage_counts <= 0)
            or len(set(zip(coverage_donors.tolist(), coverage_types.tolist())))
            != len(coverage_donors)
        ):
            raise ValueError("reference coverage count audit is malformed")
        observed_coverage = {
            (str(donor), str(type_name)): int(
                np.sum((sc_donors_all == donor) & (sc_types_all == type_name))
            )
            for donor, type_name in zip(coverage_donors, coverage_types)
        }
        declared_coverage = {
            (str(donor), str(type_name)): int(count)
            for donor, type_name, count in zip(
                coverage_donors, coverage_types, coverage_counts
            )
        }
        all_observed_pairs = set(zip(sc_donors_all.tolist(), sc_types_all.tolist()))
        if set(declared_coverage) != all_observed_pairs or declared_coverage != observed_coverage:
            raise ValueError("reference coverage counts do not match the cell-resolved source")
        expected_primary_by_pair = {
            (str(donor), str(type_name)): bool(
                sc_primary[(sc_donors_all == donor) & (sc_types_all == type_name)][0]
            )
            for donor, type_name in all_observed_pairs
        }
        declared_primary_by_pair = {
            (str(donor), str(type_name)): bool(value)
            for donor, type_name, value in zip(
                coverage_donors, coverage_types, coverage_primary
            )
        }
        if declared_primary_by_pair != expected_primary_by_pair:
            raise ValueError("reference coverage eligibility does not match cell eligibility")

        sensitivity_donors = sorted(set(sc_donors_all[~sc_primary].tolist()))
        spot_sensitivity_donors = sorted(set(donors_all[~primary].tolist()))
        if sensitivity_donors != [FAILED_REFERENCE_SENSITIVITY_DONOR] or (
            spot_sensitivity_donors != [FAILED_REFERENCE_SENSITIVITY_DONOR]
        ):
            raise ValueError("B2 must be the only failed-reference sensitivity donor")
        sensitivity_type_counts = {
            type_name: count
            for (donor, type_name), count in sorted(declared_coverage.items())
            if donor == FAILED_REFERENCE_SENSITIVITY_DONOR
        }
        qualified_sensitivity = {
            name: count
            for name, count in sensitivity_type_counts.items()
            if count >= REFERENCE_MINIMUM_QUALIFIED_CELLS
        }
        failed_reference_sensitivity = {
            "donor": FAILED_REFERENCE_SENSITIVITY_DONOR,
            "spot_rows": int(np.sum((donors_all == FAILED_REFERENCE_SENSITIVITY_DONOR) & ~primary)),
            "selected_reference_cells": int(sum(sensitivity_type_counts.values())),
            "selected_type_counts": sensitivity_type_counts,
            "minimum_qualified_cells_per_type": REFERENCE_MINIMUM_QUALIFIED_CELLS,
            "qualified_type_counts": qualified_sensitivity,
            "excluded_subthreshold_type_counts": {
                name: count
                for name, count in sensitivity_type_counts.items()
                if name not in qualified_sensitivity
            },
            "qualified_type_count": len(qualified_sensitivity),
        }

        donors = sorted(set(donors_all[primary].tolist()))
        if len(donors) != expected_primary_donors:
            raise ValueError(
                f"expected {expected_primary_donors} primary donors, found {len(donors)}"
            )
        for donor in donors:
            indication = set(indications_all[primary & (donors_all == donor)].tolist())
            if len(indication) != 1:
                raise ValueError("a primary donor spans multiple indications")
            if not np.any(sc_primary & (sc_donors_all == donor)):
                raise ValueError(f"primary donor lacks matched scRNA cells: {donor}")
        for donor in donors:
            indication = indications_all[np.flatnonzero(primary & (donors_all == donor))[0]]
            wrong = [
                candidate
                for candidate in donors
                if candidate != donor
                and np.any(primary & (donors_all == candidate) & (indications_all == indication))
            ]
            if not wrong:
                raise ValueError(f"primary donor lacks a same-indication wrong bank: {donor}")

        selected_sc = sc_primary & np.isin(sc_donors_all, donors)
        return RegionalSource(
            path=resolved,
            spot_ids=spot_ids_all[primary],
            donor_ids=donors_all[primary],
            section_ids=sections_all[primary],
            indication_ids=indications_all[primary],
            image_features=image_all[primary].astype(np.float64),
            blank_image_feature=blank,
            coordinate_features=coordinates_all[primary].astype(np.float64),
            gene_ids=gene_ids,
            st_full=full_all[primary].astype(np.float64),
            st_half_a=half_a_all[primary].astype(np.float64),
            st_half_b=half_b_all[primary].astype(np.float64),
            st_total_full=total_full_all[primary],
            st_total_half_a=total_half_a_all[primary],
            st_total_half_b=total_half_b_all[primary],
            sc_counts=sc_counts_all[selected_sc].astype(np.float64),
            sc_total_counts=sc_total_all[selected_sc],
            sc_cell_ids=sc_cells_all[selected_sc],
            sc_donor_ids=sc_donors_all[selected_sc],
            sc_indication_ids=sc_indications_all[selected_sc],
            sc_type_ids=sc_types_all[selected_sc],
            sc_n_count=n_count_all[selected_sc],
            sc_n_feature=n_feature_all[selected_sc],
            sc_percent_mt=percent_mt_all[selected_sc],
            program_names=program_names,
            program_membership=membership,
            broad_gene_ids=broad_gene_ids,
            st_broad_full=st_broad_full_all.take_rows(primary),
            st_broad_half_a=st_broad_half_a_all.take_rows(primary),
            st_broad_half_b=st_broad_half_b_all.take_rows(primary),
            sc_broad_counts=sc_broad_all.take_rows(selected_sc),
            source_receipt=dict(receipt),
            blank_receipt=dict(blank_receipt),
            failed_reference_sensitivity=failed_reference_sensitivity,
        )


def _log_normalize(counts: np.ndarray, total_library_counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    library = np.asarray(total_library_counts, dtype=np.float64)
    if library.shape != (len(values),) or np.any(library < values.sum(axis=1)):
        raise ValueError("total library counts must align and cover selected-gene counts")
    result = np.zeros_like(values, dtype=np.float64)
    nonzero = library > 0
    result[nonzero] = np.log1p(values[nonzero] * (10_000.0 / library[nonzero, None]))
    return result


def _program_scores(expression: np.ndarray, membership: np.ndarray) -> np.ndarray:
    weights = membership.astype(np.float64)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.asarray(expression, dtype=np.float64) @ weights.T


def _donor_section_weights(donors: np.ndarray, sections: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(donors), dtype=np.float64)
    unique_donors = sorted(set(donors.tolist()))
    for donor in unique_donors:
        local_sections = sorted(set(sections[donors == donor].tolist()))
        for section in local_sections:
            selected = (donors == donor) & (sections == section)
            weights[selected] = 1.0 / (
                len(unique_donors) * len(local_sections) * int(selected.sum())
            )
    return weights / weights.mean()


def _fit_predict_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    train_donors: np.ndarray,
    train_sections: np.ndarray,
    alpha: float,
    device: str,
) -> np.ndarray:
    fit = fit_weighted_ridge_grid(
        train_x,
        train_y,
        [alpha],
        _donor_section_weights(train_donors, train_sections),
        device=device,
    )
    if device == "cuda" and not str(fit.fit_device).startswith("cuda"):
        raise RuntimeError("CUDA ridge silently fell back to CPU; scientific run aborted")
    return fit.predict(test_x)[0]


def _select_ridge_alpha(
    features: np.ndarray,
    targets: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    alphas: Sequence[float],
    *,
    seed: int,
    device: str,
) -> tuple[float, Mapping[str, float]]:
    unique = sorted(set(donors.tolist()))
    folds = grouped_donor_folds(donors, n_splits=min(5, len(unique)), seed=seed)
    predictions = np.empty((len(alphas), len(targets), targets.shape[1]), dtype=np.float64)
    for train, validation in folds:
        fit = fit_weighted_ridge_grid(
            features[train],
            targets[train],
            alphas,
            _donor_section_weights(donors[train], sections[train]),
            device=device,
        )
        if device == "cuda" and not str(fit.fit_device).startswith("cuda"):
            raise RuntimeError("CUDA ridge silently fell back to CPU; scientific run aborted")
        predictions[:, validation] = fit.predict(features[validation])
    losses = {
        f"{float(alpha):g}": float(
            donor_section_macro_loss(targets, predictions[index], donors, sections)[
                "donor_section_macro_mse"
            ]
        )
        for index, alpha in enumerate(alphas)
    }
    selected = min((loss, float(alpha)) for alpha, loss in losses.items())[1]
    return selected, losses


def _stable_rank(ids: np.ndarray, seed: int) -> np.ndarray:
    values = []
    for index, value in enumerate(ids.tolist()):
        digest = hashlib.blake2b(f"{seed}\0{value}".encode(), digest_size=8, person=b"HEIReq1")
        values.append((int.from_bytes(digest.digest(), "little"), str(value), index))
    return np.asarray([row[2] for row in sorted(values)], dtype=np.int64)


def _quality_strata(source: RegionalSource) -> np.ndarray:
    """Create outcome-free type/depth/QC strata for composition equalization."""

    cached = _QUALITY_STRATA_CACHE.get(id(source))
    if cached is not None:
        return cached
    output = np.empty(len(source.sc_counts), dtype=object)
    for indication in sorted(set(source.sc_indication_ids.tolist())):
        selected = source.sc_indication_ids == indication
        depth_edges = np.quantile(source.sc_n_count[selected], [1 / 3, 2 / 3])
        quality = source.sc_n_feature[selected] - source.sc_percent_mt[selected]
        quality_edge = float(np.median(quality))
        depth_bin = np.digitize(source.sc_n_count[selected], depth_edges, right=True)
        qc_bin = (quality >= quality_edge).astype(int)
        indices = np.flatnonzero(selected)
        for local, index in enumerate(indices):
            output[index] = f"{source.sc_type_ids[index]}|depth{depth_bin[local]}|qc{qc_bin[local]}"
    result = output.astype(str)
    _QUALITY_STRATA_CACHE[id(source)] = result
    return result


def _equalized_indices(
    source: RegionalSource,
    query_donor: str,
    candidate_donors: Sequence[str],
    *,
    pooled: bool,
    seed: int,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Select the same common type/depth/QC quota for every candidate bank."""

    indication = source.indication_ids[source.donor_ids == query_donor][0]
    eligible_donors = sorted(
        donor for donor in set(source.donor_ids[source.indication_ids == indication].tolist())
    )
    strata = _quality_strata(source)
    counts: dict[str, dict[str, int]] = {}
    for donor in eligible_donors:
        local = source.sc_donor_ids == donor
        counts[donor] = {
            stratum: int(np.sum(local & (strata == stratum)))
            for stratum in sorted(set(strata[local].tolist()))
        }
    common = sorted(set.intersection(*(set(value) for value in counts.values())))
    quotas = {
        stratum: min(counts[donor][stratum] for donor in eligible_donors) for stratum in common
    }
    quotas = {key: value for key, value in quotas.items() if value > 0}
    if not quotas:
        raise ValueError(f"composition equalization has no common strata for {indication}")
    candidate_mask = np.isin(source.sc_donor_ids, list(candidate_donors))
    selected: list[int] = []
    for stratum, quota in sorted(quotas.items()):
        indices = np.flatnonzero(candidate_mask & (strata == stratum))
        if not pooled and len(candidate_donors) != 1:
            raise ValueError("a non-pooled equalized bank must identify one donor")
        if len(indices) < quota:
            raise ValueError("candidate bank cannot satisfy the fixed equalization quota")
        order = _stable_rank(source.sc_cell_ids[indices], seed)
        selected.extend(indices[order[:quota]].tolist())
    result = np.asarray(sorted(selected), dtype=np.int64)
    return result, {
        "schema": "heir.reference_bank_equalization.v1",
        "indication": str(indication),
        "candidate_donors": sorted(str(value) for value in candidate_donors),
        "pooled": bool(pooled),
        "strata_definition": (
            "sc_level1_type|within_indication_depth_tertile|within_indication_QC_half"
        ),
        "common_strata": len(quotas),
        "quota_per_stratum": quotas,
        "selected_cells": int(len(result)),
        "depth_metric": "sc_n_count",
        "quality_metric": "sc_n_feature_minus_sc_percent_mt",
        "outcome_used": False,
    }


def _bank_indices(
    source: RegionalSource,
    query_donor: str,
    candidates: Sequence[str],
    bank_mode: str,
    *,
    pooled: bool,
    seed: int,
) -> tuple[np.ndarray, Mapping[str, object]]:
    if bank_mode == "natural":
        indices = np.flatnonzero(np.isin(source.sc_donor_ids, list(candidates)))
        return indices, {
            "schema": "heir.reference_bank_natural.v1",
            "candidate_donors": sorted(str(value) for value in candidates),
            "selected_cells": int(len(indices)),
            "composition_preserved": True,
        }
    if bank_mode != "composition_equalized":
        raise ValueError(f"unknown bank mode: {bank_mode}")
    return _equalized_indices(source, query_donor, candidates, pooled=pooled, seed=seed)


def _make_bank(
    latent: np.ndarray,
    source: RegionalSource,
    indices: np.ndarray,
    *,
    prototypes_per_type: int,
    seed: int,
) -> PrototypeBank:
    if not len(indices):
        raise ValueError("reference bank is empty")
    return build_reference_prototypes(
        latent[indices],
        source.sc_donor_ids[indices],
        source.sc_type_ids[indices],
        source.sc_cell_ids[indices],
        max_prototypes_per_type=prototypes_per_type,
        seed=seed,
    )


def _qualify_reference_indices(
    type_ids: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Apply the frozen 50-cell rule to the exact post-selection bank."""

    selected = np.asarray(indices, dtype=np.int64)
    types = np.asarray(type_ids).astype(str)
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(types)):
        raise ValueError("reference-bank indices are invalid")
    selected_counts = {
        name: int(np.sum(types[selected] == name))
        for name in sorted(set(types[selected].tolist()))
    }
    qualified_counts = {
        name: count
        for name, count in selected_counts.items()
        if count >= REFERENCE_MINIMUM_QUALIFIED_CELLS
    }
    retained = selected[np.isin(types[selected], list(qualified_counts))]
    return retained, {
        "schema": "heir.reference_bank_type_qualification.v1",
        "rule_timing": "after_exact_natural_or_equalized_bank_selection",
        "minimum_qualified_cells_per_type": REFERENCE_MINIMUM_QUALIFIED_CELLS,
        "selected_cells_before_type_qualification": int(len(selected)),
        "selected_type_counts": selected_counts,
        "qualified_cells": int(len(retained)),
        "qualified_type_counts": qualified_counts,
        "qualified_type_count": len(qualified_counts),
        "excluded_subthreshold_cells": int(len(selected) - len(retained)),
        "excluded_subthreshold_type_counts": {
            name: count for name, count in selected_counts.items() if name not in qualified_counts
        },
    }


def _empty_bank(width: int) -> PrototypeBank:
    return PrototypeBank(
        states=np.empty((0, width), dtype=np.float64),
        weights=np.empty(0, dtype=np.float64),
        donor_ids=np.empty(0, dtype=str),
        type_labels=np.empty(0, dtype=str),
        prototype_ids=np.empty(0, dtype=str),
    )


def _type_centroids(bank: PrototypeBank, type_names: Sequence[str]) -> np.ndarray:
    states = []
    for type_name in type_names:
        selected = bank.type_labels == type_name
        if selected.any():
            states.append(np.average(bank.states[selected], axis=0, weights=bank.weights[selected]))
        else:
            states.append(np.full(bank.states.shape[1], np.nan))
    return np.vstack(states)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=1, keepdims=True)


def _derive_training_route_labels(
    target: np.ndarray,
    donors: np.ndarray,
    reference_latent: np.ndarray,
    source: RegionalSource,
    type_names: np.ndarray,
) -> np.ndarray:
    """Generate coarse labels only for training rows; never call on held-out rows."""

    labels = np.empty(len(target), dtype=np.int64)
    for donor in sorted(set(donors.tolist())):
        rows = donors == donor
        local = source.sc_donor_ids == donor
        centroids = []
        available = []
        for type_index, type_name in enumerate(type_names):
            selected = local & (source.sc_type_ids == type_name)
            if selected.any():
                centroids.append(reference_latent[selected].mean(axis=0))
                available.append(type_index)
        states = np.vstack(centroids)
        distance = (
            np.sum(np.square(target[rows]), axis=1)[:, None]
            + np.sum(np.square(states), axis=1)[None, :]
            - 2.0 * target[rows] @ states.T
        )
        labels[rows] = np.asarray(available)[np.argmin(distance, axis=1)]
    return labels


def _route_probabilities(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    train_donors: np.ndarray,
    train_sections: np.ndarray,
    type_count: int,
    alpha: float,
    device: str,
) -> np.ndarray:
    one_hot = np.eye(type_count, dtype=np.float64)[train_labels]
    scores = _fit_predict_ridge(
        train_features,
        one_hot,
        test_features,
        train_donors,
        train_sections,
        alpha,
        device,
    )
    return _softmax(scores)


def _type_routed_state(
    type_probabilities: np.ndarray,
    bank: PrototypeBank,
    type_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centroids = _type_centroids(bank, type_names)
    available = np.isfinite(centroids).all(axis=1)
    probabilities = type_probabilities.copy()
    probabilities[:, ~available] = 0.0
    coverage = probabilities.sum(axis=1)
    normalized = probabilities / np.maximum(coverage[:, None], 1.0e-12)
    filled = np.nan_to_num(centroids, nan=0.0)
    return normalized @ filled, coverage


def _retrieve(
    query: np.ndarray,
    type_probabilities: np.ndarray,
    bank: PrototypeBank,
    type_names: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
    if not len(bank.states):
        return np.zeros_like(query, dtype=np.float64), {
            "support_distance": np.full(len(query), np.inf, dtype=np.float64),
            "type_coverage": np.zeros(len(query), dtype=np.float64),
            "reference_uncertainty": np.ones(len(query), dtype=np.float64),
        }
    type_lookup = {name: index for index, name in enumerate(type_names.tolist())}
    prototype_prior = np.asarray(
        [type_probabilities[:, type_lookup.get(name, 0)] for name in bank.type_labels],
        dtype=np.float64,
    ).T
    unknown = np.asarray([name not in type_lookup for name in bank.type_labels])
    prototype_prior[:, unknown] = 0.0
    available_type_indices = sorted(
        {type_lookup[name] for name in bank.type_labels.tolist() if name in type_lookup}
    )
    type_coverage = (
        type_probabilities[:, available_type_indices].sum(axis=1)
        if available_type_indices
        else np.zeros(len(query), dtype=np.float64)
    )
    distance = (
        np.sum(np.square(query), axis=1)[:, None]
        + np.sum(np.square(bank.states), axis=1)[None, :]
        - 2.0 * query @ bank.states.T
    )
    distance = np.maximum(distance, 0.0)
    prior = prototype_prior * bank.weights[None, :]
    logits = -distance / float(temperature) + np.log(np.maximum(prior, 1.0e-300))
    logits -= logits.max(axis=1, keepdims=True)
    attention = np.exp(logits)
    attention *= prior > 0
    denominator = attention.sum(axis=1, keepdims=True)
    attention /= np.maximum(denominator, 1.0e-300)
    state = attention @ bank.states
    nearest = np.min(np.where(prior > 0, distance, np.inf), axis=1)
    entropy = -np.sum(attention * np.log(np.maximum(attention, 1.0e-300)), axis=1)
    support_count = np.maximum(np.sum(prior > 0, axis=1), 1)
    entropy /= np.log(np.maximum(support_count, 2))
    return state, {
        "support_distance": nearest,
        "type_coverage": np.minimum(type_coverage, 1.0),
        "reference_uncertainty": np.clip(entropy, 0.0, 1.0),
    }


def _adaptive_fusion(
    image: np.ndarray,
    reference: np.ndarray,
    diagnostics: Mapping[str, np.ndarray],
    base_alpha: float,
    support_threshold: float,
) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
    distance = np.asarray(diagnostics["support_distance"], dtype=np.float64)
    coverage = np.asarray(diagnostics["type_coverage"], dtype=np.float64)
    uncertainty = np.asarray(diagnostics["reference_uncertainty"], dtype=np.float64)
    support = np.clip(1.0 - distance / max(float(support_threshold), 1.0e-12), 0.0, 1.0)
    adaptive = float(base_alpha) * support * coverage * (1.0 - uncertainty)
    adaptive = np.clip(adaptive, 0.0, 0.5)
    unsupported = (coverage <= 0.0) | ~np.isfinite(distance)
    adaptive[unsupported] = 0.0
    abstained = (adaptive <= 1.0e-12) | unsupported
    prediction = np.asarray(image, dtype=np.float64).copy()
    supported = ~abstained
    prediction[supported] += adaptive[supported, None] * (
        reference[supported] - image[supported]
    )
    return prediction, {
        **diagnostics,
        "support_score": support,
        "adaptive_alpha": adaptive,
        "abstained_fallback_to_H": abstained,
    }


def _normalized_balanced_loss(
    truth: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    types: np.ndarray,
) -> Mapping[str, object]:
    difference = truth - prediction
    donor_type: dict[str, list[float]] = {}
    donor_section_type: dict[str, dict[str, list[float]]] = {}
    for donor in sorted(set(donors.tolist())):
        for type_name in sorted(set(types[donors == donor].tolist())):
            selected = (donors == donor) & (types == type_name)
            centered = truth[selected] - truth[selected].mean(axis=0)
            denominator = float(np.sum(np.square(centered)))
            if selected.sum() >= 2 and denominator > 1.0e-12:
                donor_type.setdefault(donor, []).append(
                    float(np.sum(np.square(difference[selected])) / denominator)
                )
        for section in sorted(set(sections[donors == donor].tolist())):
            for type_name in sorted(set(types[(donors == donor) & (sections == section)].tolist())):
                selected = (donors == donor) & (sections == section) & (types == type_name)
                centered = truth[selected] - truth[selected].mean(axis=0)
                denominator = float(np.sum(np.square(centered)))
                if selected.sum() >= 2 and denominator > 1.0e-12:
                    donor_section_type.setdefault(donor, {}).setdefault(section, []).append(
                        float(np.sum(np.square(difference[selected])) / denominator)
                    )
    donor_type_loss = {
        donor: float(np.mean(values)) for donor, values in donor_type.items() if values
    }
    donor_section_type_loss = {
        donor: float(np.mean([np.mean(values) for values in local.values() if values]))
        for donor, local in donor_section_type.items()
        if any(local.values())
    }
    return {
        "exact_donor_type_normalized_loss": None,
        "exact_donor_section_type_normalized_loss": None,
        "exact_type_balanced_loss_status": (
            "blocked_no_outcome_independent_barcode_keyed_spot_type_labels"
        ),
        "exploratory_predicted_route_donor_type_normalized_loss": (
            float(np.mean(list(donor_type_loss.values()))) if donor_type_loss else None
        ),
        "exploratory_predicted_route_donor_section_type_normalized_loss": (
            float(np.mean(list(donor_section_type_loss.values())))
            if donor_section_type_loss
            else None
        ),
        "exploratory_predicted_route_donor_type_loss": donor_type_loss,
        "exploratory_predicted_route_donor_section_type_loss": donor_section_type_loss,
        "type_provenance": "outer_heldout_H_and_E_router_prediction_not_ST_derived",
    }


def _calibration(truth: np.ndarray, prediction: np.ndarray) -> Mapping[str, float | None]:
    x = prediction.reshape(-1)
    y = truth.reshape(-1)
    variance = float(np.sum(np.square(x - x.mean())))
    slope = (
        None if variance <= 1.0e-12 else float(np.sum((x - x.mean()) * (y - y.mean())) / variance)
    )
    intercept = None if slope is None else float(y.mean() - slope * x.mean())
    return {
        "slope_truth_on_prediction": slope,
        "intercept_truth_on_prediction": intercept,
        "mean_bias_prediction_minus_truth": float(np.mean(x - y)),
    }


def _donor_section_r2(
    truth: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
) -> Mapping[str, object]:
    donor_values: dict[str, float] = {}
    section_values: dict[str, float] = {}
    for donor in sorted(set(donors.tolist())):
        local = []
        for section in sorted(set(sections[donors == donor].tolist())):
            selected = (donors == donor) & (sections == section)
            centered = truth[selected] - truth[selected].mean(axis=0)
            denominator = float(np.sum(np.square(centered)))
            if selected.sum() >= 2 and denominator > 1.0e-12:
                value = 1.0 - float(
                    np.sum(np.square(truth[selected] - prediction[selected])) / denominator
                )
                section_values[f"{donor}::{section}"] = value
                local.append(value)
        if local:
            donor_values[donor] = float(np.mean(local))
    return {
        "donor_section_macro_R2": (
            float(np.mean(list(donor_values.values()))) if donor_values else None
        ),
        "donor_R2": donor_values,
        "section_R2": section_values,
    }


def _score_model(
    truth: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    types: np.ndarray,
) -> Mapping[str, object]:
    mse = donor_section_macro_loss(truth, prediction, donors, sections)
    normalized = _normalized_balanced_loss(truth, prediction, donors, sections, types)
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = variance_preservation(truth, prediction, sections)
    return {
        **mse,
        **_donor_section_r2(truth, prediction, donors, sections),
        **normalized,
        "exploratory_predicted_route_donor_type_macro_R2": (
            None
            if normalized["exploratory_predicted_route_donor_type_normalized_loss"] is None
            else 1.0 - float(normalized["exploratory_predicted_route_donor_type_normalized_loss"])
        ),
        "exploratory_predicted_route_donor_section_type_macro_R2": (
            None
            if normalized["exploratory_predicted_route_donor_section_type_normalized_loss"] is None
            else 1.0
            - float(normalized["exploratory_predicted_route_donor_section_type_normalized_loss"])
        ),
        "variance_preservation": variance,
        "calibration": _calibration(truth, prediction),
    }


def _coverage_summary(rows: Sequence[Mapping[str, np.ndarray]]) -> Mapping[str, object]:
    if not rows:
        return {"rows": 0}
    joined = {key: np.concatenate([np.asarray(row[key]) for row in rows]) for key in rows[0]}
    finite_distance = joined["support_distance"][np.isfinite(joined["support_distance"])]
    return {
        "rows": int(len(joined["adaptive_alpha"])),
        "median_support_distance": float(np.median(finite_distance))
        if len(finite_distance)
        else None,
        "median_type_coverage": float(np.median(joined["type_coverage"])),
        "median_reference_uncertainty": float(np.median(joined["reference_uncertainty"])),
        "median_adaptive_alpha": float(np.median(joined["adaptive_alpha"])),
        "abstention_fraction": float(np.mean(joined["abstained_fallback_to_H"])),
        "fallback_is_exact_H": True,
    }


def _full_depth_floor_metrics(
    raw: Mapping[str, object],
) -> Mapping[str, object]:
    """Convert two independent half-depth errors to full-depth noise variance.

    If equally thinned halves have independent error variance ``v``, their
    difference has variance ``2v`` while their full-depth mean has variance
    ``v/2``.  The full-depth measurement-error estimate is therefore one
    quarter of the cross-half squared error.  Both quantities are retained.
    """

    donor = {key: float(value) / 4.0 for key, value in raw["donor_mse"].items()}
    section = {key: float(value) / 4.0 for key, value in raw["section_mse"].items()}
    return {
        "donor_section_macro_mse": float(raw["donor_section_macro_mse"]) / 4.0,
        "donor_mse": donor,
        "section_mse": section,
        "raw_cross_half_donor_section_macro_mse": float(raw["donor_section_macro_mse"]),
        "full_depth_correction_factor": 0.25,
        "derivation": "Var(A-B)=2v and Var((A+B)/2)=v/2; ratio=(v/2)/(2v)=1/4",
        "status": "secondary_thinning_variance_approximation_not_a_technical_replicate",
        "assumptions": [
            "independent equal-probability UMI thinning",
            "locally linear error propagation in the frozen molecular target space",
            "cross-fitted mapping bias is small relative to thinning noise",
        ],
        "secondary_floor_only": True,
    }


@dataclass(frozen=True)
class EndpointFold:
    name: str
    target_names: np.ndarray
    full: np.ndarray
    half_a: np.ndarray
    half_b: np.ndarray
    reference: np.ndarray
    basis_receipt: Mapping[str, object]


def _endpoint_fold(
    source: RegionalSource,
    full_expression: np.ndarray,
    half_a_expression: np.ndarray,
    half_b_expression: np.ndarray,
    sc_expression: np.ndarray,
    training_donors: Sequence[str],
) -> EndpointFold:
    weights = _donor_section_weights(source.donor_ids, source.section_ids)
    spatial_raw = _program_scores(full_expression, source.program_membership)
    half_a_raw = _program_scores(half_a_expression, source.program_membership)
    half_b_raw = _program_scores(half_b_expression, source.program_membership)
    reference_raw = _program_scores(sc_expression, source.program_membership)
    basis = fit_target_basis(
        spatial_raw,
        source.donor_ids,
        training_donors,
        n_components=None,
        sample_weight=weights,
    )
    names = source.program_names.copy()
    receipt = {
        "representation": "prespecified_fixed_program_means",
        "fit_donors": list(basis.fit_donors),
        "programs": names.tolist(),
        "genes_per_program": source.program_membership.sum(axis=1).astype(int).tolist(),
    }
    return EndpointFold(
        name="program_total",
        target_names=names,
        full=basis.transform(spatial_raw),
        half_a=basis.transform(half_a_raw),
        half_b=basis.transform(half_b_raw),
        reference=basis.transform(reference_raw),
        basis_receipt=receipt,
    )


def _broad_pca_endpoint_fold(
    source: RegionalSource,
    training_donors: Sequence[str],
    *,
    pca_components: int,
    pca_genes: int,
) -> EndpointFold:
    """Select broad genes and fit PCA using only outer-training donor outcomes."""

    train = np.isin(source.donor_ids, list(training_donors))
    train_sc = np.isin(source.sc_donor_ids, list(training_donors))
    row_weights = _donor_section_weights(source.donor_ids, source.section_ids)
    variance = source.st_broad_full.weighted_log_variance(train, source.st_total_full, row_weights)
    st_nonzero = source.st_broad_full.column_nonzero_counts(train)
    sc_nonzero = source.sc_broad_counts.column_nonzero_counts(train_sc)
    st_coverage = st_nonzero / max(int(train.sum()), 1)
    sc_coverage = sc_nonzero / max(int(train_sc.sum()), 1)
    eligible = (variance > 1.0e-10) & (st_coverage >= 0.01) & (sc_coverage >= 0.01)
    ordered = sorted(
        np.flatnonzero(eligible).tolist(),
        key=lambda index: (-float(variance[index]), source.broad_gene_ids[index]),
    )
    selected = np.asarray(ordered[: min(pca_genes, len(ordered))], dtype=np.int64)
    if len(selected) < pca_components:
        raise ValueError(
            "outer-training broad gene coverage leaves fewer genes than PCA components"
        )

    full_counts = source.st_broad_full.dense_columns(selected)
    half_a_counts = source.st_broad_half_a.dense_columns(selected)
    half_b_counts = source.st_broad_half_b.dense_columns(selected)
    if not np.array_equal(full_counts, half_a_counts + half_b_counts):
        raise ValueError("selected broad PCA counts do not reconstruct from transcript halves")
    full_expression = _log_normalize(full_counts, source.st_total_full)
    half_a_expression = _log_normalize(
        half_a_counts * 2.0,
        source.st_total_half_a * 2.0,
    )
    half_b_expression = _log_normalize(
        half_b_counts * 2.0,
        source.st_total_half_b * 2.0,
    )
    del full_counts, half_a_counts, half_b_counts
    reference_expression = _log_normalize(
        source.sc_broad_counts.dense_columns(selected), source.sc_total_counts
    )
    basis = fit_target_basis(
        full_expression,
        source.donor_ids,
        training_donors,
        n_components=pca_components,
        sample_weight=row_weights,
    )
    return EndpointFold(
        name="pca_total",
        target_names=np.asarray(
            [f"fold_local_PC{index + 1:02d}" for index in range(pca_components)]
        ),
        full=basis.transform(full_expression),
        half_a=basis.transform(half_a_expression),
        half_b=basis.transform(half_b_expression),
        reference=basis.transform(reference_expression),
        basis_receipt={
            "representation": "outer_training_donor_balanced_PCA_broad_common_gene_CSR",
            "fit_donors": list(basis.fit_donors),
            "components": pca_components,
            "broad_source_gene_count": len(source.broad_gene_ids),
            "eligible_training_only_gene_count": int(eligible.sum()),
            "selected_gene_count": int(len(selected)),
            "selected_genes": source.broad_gene_ids[selected].tolist(),
            "selection": {
                "minimum_ST_training_spot_nonzero_fraction": 0.01,
                "minimum_scRNA_training_cell_nonzero_fraction": 0.01,
                "positive_donor_section_weighted_log_expression_variance": True,
                "ordering": "descending_training_variance_then_gene_id",
            },
            "heldout_outcomes_or_scRNA_used_for_gene_selection": False,
            "global_broad_matrix_densified": False,
            "only_selected_columns_densified_within_outer_fold": True,
        },
    )


def _cross_fitted_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    alpha: float,
    *,
    seed: int,
    device: str,
) -> np.ndarray:
    predictions = np.empty_like(targets, dtype=np.float64)
    folds = grouped_donor_folds(donors, n_splits=min(5, len(set(donors.tolist()))), seed=seed)
    for train, validation in folds:
        predictions[validation] = _fit_predict_ridge(
            features[train],
            targets[train],
            features[validation],
            donors[train],
            sections[train],
            alpha,
            device,
        )
    return predictions


def _select_fusion_parameters(
    image: np.ndarray,
    type_probabilities: np.ndarray,
    target: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    banks: Mapping[str, PrototypeBank],
    type_names: np.ndarray,
    temperatures: Sequence[float],
    alphas: Sequence[float],
) -> tuple[float, float, float, Mapping[str, float]]:
    losses: dict[str, float] = {}
    candidates: list[tuple[float, float, float, float]] = []
    for temperature in temperatures:
        reference = np.empty_like(target)
        distance = np.empty(len(target), dtype=np.float64)
        coverage = np.empty(len(target), dtype=np.float64)
        uncertainty = np.empty(len(target), dtype=np.float64)
        for donor in sorted(set(donors.tolist())):
            selected = donors == donor
            state, diagnostics = _retrieve(
                image[selected],
                type_probabilities[selected],
                banks[donor],
                type_names,
                float(temperature),
            )
            reference[selected] = state
            distance[selected] = diagnostics["support_distance"]
            coverage[selected] = diagnostics["type_coverage"]
            uncertainty[selected] = diagnostics["reference_uncertainty"]
        finite = distance[np.isfinite(distance)]
        threshold = float(np.quantile(finite, 0.95)) if len(finite) else 1.0
        diagnostics = {
            "support_distance": distance,
            "type_coverage": coverage,
            "reference_uncertainty": uncertainty,
        }
        for alpha in alphas:
            prediction, _receipt = _adaptive_fusion(
                image, reference, diagnostics, float(alpha), threshold
            )
            loss = float(
                donor_section_macro_loss(target, prediction, donors, sections)[
                    "donor_section_macro_mse"
                ]
            )
            key = f"temperature={float(temperature):g}|alpha={float(alpha):g}"
            losses[key] = loss
            candidates.append((loss, float(alpha), float(temperature), threshold))
    if not candidates:
        raise ValueError("fusion parameter selection has no candidates")
    _loss, alpha, temperature, threshold = min(candidates)
    return alpha, temperature, threshold, losses


def _inner_oof_fusion_inputs(
    source: RegionalSource,
    fold: EndpointFold,
    outer_train: np.ndarray,
    bank_mode: str,
    type_names: np.ndarray,
    *,
    h_alpha: float,
    router_alpha: float,
    prototypes_per_type: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, PrototypeBank], Mapping[str, object]]:
    """Build fusion-development inputs with every inner validation donor excluded.

    Target bases are frozen at the outer-training level, but each inner H&E
    fit, routing fit, and scRNA-to-ST calibrator excludes the complete grouped
    validation-donor fold.  Matched scRNA for a validation donor remains an
    inference input, mirroring the intended held-out use case.
    """

    outer_indices = np.flatnonzero(outer_train)
    donors = source.donor_ids[outer_train]
    sections = source.section_ids[outer_train]
    image = source.image_features[outer_train]
    target = fold.full[outer_train]
    folds = grouped_donor_folds(donors, n_splits=min(5, len(set(donors.tolist()))), seed=seed)
    h_oof = np.empty_like(target)
    route_oof = np.empty((len(target), len(type_names)), dtype=np.float64)
    banks: dict[str, PrototypeBank] = {}
    receipts: dict[str, object] = {}
    for inner_index, (inner_train, validation) in enumerate(folds):
        fit_donors = sorted(set(donors[inner_train].tolist()))
        validation_donors = sorted(set(donors[validation].tolist()))
        calibrator = fit_reference_calibrator(
            fold.reference,
            source.sc_donor_ids,
            fold.full,
            source.donor_ids,
            fit_donors,
            ridge_alpha=1.0,
        )
        calibrated_reference = calibrator.transform(fold.reference)
        routing_labels = _derive_training_route_labels(
            target[inner_train],
            donors[inner_train],
            calibrated_reference,
            source,
            type_names,
        )
        h_oof[validation] = _fit_predict_ridge(
            image[inner_train],
            target[inner_train],
            image[validation],
            donors[inner_train],
            sections[inner_train],
            h_alpha,
            device,
        )
        route_oof[validation] = _route_probabilities(
            image[inner_train],
            routing_labels,
            image[validation],
            donors[inner_train],
            sections[inner_train],
            len(type_names),
            router_alpha,
            device,
        )
        for validation_donor in validation_donors:
            bank, bank_receipt = _fold_bank(
                source,
                calibrated_reference,
                validation_donor,
                [validation_donor],
                bank_mode,
                pooled=False,
                prototypes_per_type=prototypes_per_type,
                seed=seed + inner_index,
                bank_role="inner_matched_primary",
                fail_if_no_qualified_types=True,
            )
            banks[validation_donor] = bank
            receipts[validation_donor] = {
                **bank_receipt,
                "inner_fold": inner_index,
                "calibration_fit_donors": list(calibrator.fit_donors),
                "validation_fold_donors": validation_donors,
                "validation_donor_excluded_from_calibration": (
                    validation_donor not in calibrator.fit_donors
                ),
                "outer_row_indices_are_receipt_only_not_model_features": (
                    outer_indices[validation].astype(int).tolist()
                ),
            }
    if set(banks) != set(donors.tolist()):
        raise RuntimeError("inner grouped CV did not construct every validation donor bank")
    return h_oof, route_oof, banks, receipts


def _fold_bank(
    source: RegionalSource,
    reference_latent: np.ndarray,
    query_donor: str,
    candidates: Sequence[str],
    bank_mode: str,
    *,
    pooled: bool,
    prototypes_per_type: int,
    seed: int,
    bank_role: str,
    fail_if_no_qualified_types: bool,
) -> tuple[PrototypeBank, Mapping[str, object]]:
    indices, receipt = _bank_indices(
        source,
        query_donor,
        candidates,
        bank_mode,
        pooled=pooled,
        seed=seed,
    )
    qualified, qualification = _qualify_reference_indices(source.sc_type_ids, indices)
    merged_receipt = {
        **receipt,
        "bank_role": bank_role,
        "type_qualification": qualification,
        "fail_if_no_qualified_types": fail_if_no_qualified_types,
        "sensitivity_donors_excluded": [FAILED_REFERENCE_SENSITIVITY_DONOR],
        "sensitivity_donor_used_for_fit_selection_or_inference": False,
    }
    if not len(qualified):
        if fail_if_no_qualified_types:
            raise ValueError(
                f"primary reference bank {bank_role} has no type with at least "
                f"{REFERENCE_MINIMUM_QUALIFIED_CELLS} selected cells"
            )
        return _empty_bank(reference_latent.shape[1]), {
            **merged_receipt,
            "status": "unsupported_no_qualified_types_exact_H_only_fallback",
            "prototype_count": 0,
        }
    bank = _make_bank(
        reference_latent,
        source,
        qualified,
        prototypes_per_type=prototypes_per_type,
        seed=seed,
    )
    if not set(bank.type_labels.tolist()) <= set(
        qualification["qualified_type_counts"]
    ):
        raise RuntimeError("a subthreshold reference type reached prototype construction")
    return bank, {
        **merged_receipt,
        "status": "qualified",
        "prototype_count": int(len(bank.states)),
        "prototype_type_counts": {
            name: int(np.sum(bank.type_labels == name))
            for name in sorted(set(bank.type_labels.tolist()))
        },
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 3:
        return None
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    if np.std(first_rank) <= 1.0e-12 or np.std(second_rank) <= 1.0e-12:
        return None
    value = float(np.corrcoef(first_rank, second_rank)[0, 1])
    return value if np.isfinite(value) else None


def _program_reliability_gate(
    half_a: np.ndarray,
    half_b: np.ndarray,
    donors: np.ndarray,
    training_donors: Sequence[str],
    names: np.ndarray,
) -> Mapping[str, object]:
    selected = np.isin(donors, list(training_donors))
    programs = {}
    retained = []
    for index, name in enumerate(names):
        by_donor = {}
        for donor in sorted(set(donors[selected].tolist())):
            local = selected & (donors == donor)
            correlation = _spearman(half_a[local, index], half_b[local, index])
            by_donor[donor] = correlation
        evaluable = [value for value in by_donor.values() if value is not None]
        passing = sum(
            value is not None and value >= PROGRAM_RELIABILITY_MINIMUM_SPEARMAN
            for value in by_donor.values()
        )
        fraction = float(passing / len(by_donor)) if by_donor else 0.0
        qualifies = bool(evaluable and fraction >= PROGRAM_RELIABILITY_MINIMUM_DONOR_FRACTION)
        if qualifies:
            retained.append(str(name))
        programs[str(name)] = {
            "training_donor_spearman": by_donor,
            "evaluable_training_donors": len(evaluable),
            "fraction_at_or_above_threshold": fraction,
            "retained": qualifies,
        }
    feasible = len(retained) >= PROGRAM_RELIABILITY_MINIMUM_PROGRAMS
    return {
        "schema": "heir.outer_training_program_reliability_gate.v1",
        "metric": "within_training_donor_Spearman_between_disjoint_UMI_halves",
        "minimum_spearman": PROGRAM_RELIABILITY_MINIMUM_SPEARMAN,
        "minimum_donor_fraction": PROGRAM_RELIABILITY_MINIMUM_DONOR_FRACTION,
        "minimum_retained_programs": PROGRAM_RELIABILITY_MINIMUM_PROGRAMS,
        "fit_donors": sorted(str(value) for value in training_donors),
        "heldout_donor_outcomes_used": False,
        "programs": programs,
        "retained_programs": retained,
        "retained_program_count": len(retained),
        "status": "feasible" if feasible else "blocked_fewer_than_three_reliable_programs",
        "candidate_program_metrics_role": "secondary_all_candidates",
    }


def _rare_state_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    thresholds: np.ndarray,
    names: np.ndarray,
) -> Mapping[str, object]:
    output = {}
    for index, name in enumerate(names):
        positive = truth[:, index] >= thresholds[:, index]
        called = prediction[:, index] >= thresholds[:, index]
        true_positive = int(np.sum(positive & called))
        output[str(name)] = {
            "truth_positive": int(positive.sum()),
            "predicted_positive": int(called.sum()),
            "recall": float(true_positive / positive.sum()) if positive.any() else None,
            "coverage_ratio": float(called.sum() / positive.sum()) if positive.any() else None,
        }
    return output


def _add_target_metrics(
    score: Mapping[str, object],
    truth: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    target_names: np.ndarray,
) -> Mapping[str, object]:
    result = dict(score)
    result["per_target_donor_section_macro_mse"] = {
        str(name): float(
            donor_section_macro_loss(truth[:, [index]], prediction[:, [index]], donors, sections)[
                "donor_section_macro_mse"
            ]
        )
        for index, name in enumerate(target_names)
    }
    return result


def _paired_inference_from_losses(
    loss_by_model: Mapping[str, Mapping[str, float]],
    comparisons: Mapping[str, tuple[str, str]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> Mapping[str, object]:
    paired: dict[str, object] = {}
    p_values: dict[str, float] = {}
    for offset, (name, (model, control)) in enumerate(comparisons.items()):
        summary = summarize_paired_donor_effects(
            {donor: -float(value) for donor, value in loss_by_model[model].items()},
            {donor: -float(value) for donor, value in loss_by_model[control].items()},
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=seed + offset,
        )
        paired[name] = summary
        p_values[name] = float(summary["exact_sign_flip_p"])
    adjusted = holm_adjust(p_values)
    return {
        name: {
            **summary,
            "effect_definition": "control_loss_minus_M3_loss; positive favors M3",
            "holm_adjusted_p_within_endpoint_bank_mode": adjusted[name],
        }
        for name, summary in paired.items()
    }


def _measurement_floor_inference(
    loss_by_model: Mapping[str, Mapping[str, float]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> Mapping[str, object]:
    summary = summarize_paired_donor_effects(
        {donor: -float(value) for donor, value in loss_by_model["M8"].items()},
        {donor: -float(value) for donor, value in loss_by_model["M3"].items()},
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=seed,
    )
    return {
        **summary,
        "inequality": "L(M8_ST_floor) < L(M3_H_and_E_plus_matched_scRNA)",
        "effect_definition": "M3_loss_minus_M8_loss; positive supports floor_below_M3",
    }


def run_endpoint(
    source: RegionalSource,
    endpoint: str,
    bank_mode: str,
    *,
    ridge_alphas: Sequence[float],
    fusion_alphas: Sequence[float],
    temperatures: Sequence[float],
    pca_components: int,
    pca_genes: int,
    prototypes_per_type: int,
    bootstrap_iterations: int,
    seed: int,
    device: str,
) -> Mapping[str, object]:
    """Evaluate one endpoint/bank mode with strict outer donor isolation."""

    if endpoint == "program_total":
        full_expression = _log_normalize(source.st_full, source.st_total_full)
        half_a_expression = _log_normalize(source.st_half_a * 2.0, source.st_total_half_a * 2.0)
        half_b_expression = _log_normalize(source.st_half_b * 2.0, source.st_total_half_b * 2.0)
        sc_expression = _log_normalize(source.sc_counts, source.sc_total_counts)
    elif endpoint == "pca_total":
        full_expression = half_a_expression = half_b_expression = sc_expression = None
    else:
        raise ValueError(f"unknown endpoint: {endpoint}")
    donors_all = sorted(set(source.donor_ids.tolist()))
    type_names = np.asarray(sorted(set(source.sc_type_ids.tolist())))

    truth_rows: list[np.ndarray] = []
    donor_rows: list[np.ndarray] = []
    section_rows: list[np.ndarray] = []
    type_rows: list[np.ndarray] = []
    threshold_rows: list[np.ndarray] = []
    predictions: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "M0",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5_blank",
            "M5_coordinates",
            "M7",
        )
    }
    floor_truth_rows: list[np.ndarray] = []
    floor_prediction_rows: list[np.ndarray] = []
    floor_donor_rows: list[np.ndarray] = []
    floor_section_rows: list[np.ndarray] = []
    floor_type_rows: list[np.ndarray] = []
    wrong_truth: list[np.ndarray] = []
    wrong_prediction: list[np.ndarray] = []
    wrong_donors: list[np.ndarray] = []
    wrong_sections: list[np.ndarray] = []
    wrong_types: list[np.ndarray] = []
    wrong_conditions: dict[str, dict[str, float]] = {}
    coverage: dict[str, list[Mapping[str, np.ndarray]]] = {
        name: [] for name in ("M3", "M4", "M5_blank", "M5_coordinates", "M6", "M7")
    }
    fold_receipts: dict[str, object] = {}
    program_gate_receipts: dict[str, Mapping[str, object]] = {}
    primary_program_donor_loss: dict[str, dict[str, float]] = {
        name: {}
        for name in (
            "M0",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5_blank",
            "M5_coordinates",
            "M6",
            "M7",
            "M8",
        )
    }
    target_names: np.ndarray | None = None

    for fold_index, held_out in enumerate(donors_all):
        train_donors = [donor for donor in donors_all if donor != held_out]
        train = source.donor_ids != held_out
        test = source.donor_ids == held_out
        if endpoint == "program_total":
            assert full_expression is not None
            assert half_a_expression is not None
            assert half_b_expression is not None
            assert sc_expression is not None
            fold = _endpoint_fold(
                source,
                full_expression,
                half_a_expression,
                half_b_expression,
                sc_expression,
                train_donors,
            )
        else:
            fold = _broad_pca_endpoint_fold(
                source,
                train_donors,
                pca_components=pca_components,
                pca_genes=pca_genes,
            )
        if target_names is None:
            target_names = fold.target_names
        elif len(target_names) != len(fold.target_names):
            raise RuntimeError("endpoint width changed across outer folds")
        program_gate = (
            _program_reliability_gate(
                fold.half_a,
                fold.half_b,
                source.donor_ids,
                train_donors,
                fold.target_names,
            )
            if endpoint == "program_total"
            else None
        )
        if program_gate is not None:
            program_gate_receipts[held_out] = program_gate

        calibrator = fit_reference_calibrator(
            fold.reference,
            source.sc_donor_ids,
            fold.full,
            source.donor_ids,
            train_donors,
            ridge_alpha=1.0,
        )
        calibrated_reference = calibrator.transform(fold.reference)
        route_labels = _derive_training_route_labels(
            fold.full[train],
            source.donor_ids[train],
            calibrated_reference,
            source,
            type_names,
        )

        h_alpha, h_cv = _select_ridge_alpha(
            source.image_features[train],
            fold.full[train],
            source.donor_ids[train],
            source.section_ids[train],
            ridge_alphas,
            seed=seed + fold_index,
            device=device,
        )
        router_alpha, router_cv = _select_ridge_alpha(
            source.image_features[train],
            np.eye(len(type_names))[route_labels],
            source.donor_ids[train],
            source.section_ids[train],
            ridge_alphas,
            seed=seed + 100 + fold_index,
            device=device,
        )
        coordinate_alpha, coordinate_cv = _select_ridge_alpha(
            source.coordinate_features[train],
            fold.full[train],
            source.donor_ids[train],
            source.section_ids[train],
            ridge_alphas,
            seed=seed + 200 + fold_index,
            device=device,
        )
        coordinate_router_alpha, _coordinate_router_cv = _select_ridge_alpha(
            source.coordinate_features[train],
            np.eye(len(type_names))[route_labels],
            source.donor_ids[train],
            source.section_ids[train],
            ridge_alphas,
            seed=seed + 300 + fold_index,
            device=device,
        )

        h_test = _fit_predict_ridge(
            source.image_features[train],
            fold.full[train],
            source.image_features[test],
            source.donor_ids[train],
            source.section_ids[train],
            h_alpha,
            device,
        )
        blank_features = np.repeat(source.blank_image_feature[None, :], int(test.sum()), axis=0)
        blank_test = _fit_predict_ridge(
            source.image_features[train],
            fold.full[train],
            blank_features,
            source.donor_ids[train],
            source.section_ids[train],
            h_alpha,
            device,
        )
        coordinate_test = _fit_predict_ridge(
            source.coordinate_features[train],
            fold.full[train],
            source.coordinate_features[test],
            source.donor_ids[train],
            source.section_ids[train],
            coordinate_alpha,
            device,
        )
        route_test = _route_probabilities(
            source.image_features[train],
            route_labels,
            source.image_features[test],
            source.donor_ids[train],
            source.section_ids[train],
            len(type_names),
            router_alpha,
            device,
        )
        blank_route = _route_probabilities(
            source.image_features[train],
            route_labels,
            blank_features,
            source.donor_ids[train],
            source.section_ids[train],
            len(type_names),
            router_alpha,
            device,
        )
        coordinate_route = _route_probabilities(
            source.coordinate_features[train],
            route_labels,
            source.coordinate_features[test],
            source.donor_ids[train],
            source.section_ids[train],
            len(type_names),
            coordinate_router_alpha,
            device,
        )

        # Inner grouped-donor predictions select the one-step correction only.
        h_oof, route_oof, inner_banks, inner_bank_receipts = _inner_oof_fusion_inputs(
            source,
            fold,
            train,
            bank_mode,
            type_names,
            h_alpha=h_alpha,
            router_alpha=router_alpha,
            prototypes_per_type=prototypes_per_type,
            seed=seed + 400 + fold_index,
            device=device,
        )
        fusion_alpha, temperature, support_threshold, fusion_cv = _select_fusion_parameters(
            h_oof,
            route_oof,
            fold.full[train],
            source.donor_ids[train],
            source.section_ids[train],
            inner_banks,
            type_names,
            temperatures,
            fusion_alphas,
        )

        matched_bank, matched_receipt = _fold_bank(
            source,
            calibrated_reference,
            held_out,
            [held_out],
            bank_mode,
            pooled=False,
            prototypes_per_type=prototypes_per_type,
            seed=seed + fold_index,
            bank_role="outer_matched_primary_M1_through_M5",
            fail_if_no_qualified_types=True,
        )
        m1 = reference_only_state(matched_bank.states, matched_bank.weights, int(test.sum()))
        m2_reference, m2_coverage = _type_routed_state(route_test, matched_bank, type_names)
        m2 = h_test.copy()
        m2_supported = m2_coverage > 0.0
        m2[m2_supported] = m2_reference[m2_supported]
        matched_reference, matched_diagnostics = _retrieve(
            h_test, route_test, matched_bank, type_names, temperature
        )
        m3, m3_receipt = _adaptive_fusion(
            h_test,
            matched_reference,
            matched_diagnostics,
            fusion_alpha,
            support_threshold,
        )

        local_spots = source.spot_ids[test]
        derangement = deterministic_group_derangement(
            source.section_ids[test], local_spots, seed=seed + fold_index
        )
        shuffled_h = h_test[derangement]
        shuffled_route = route_test[derangement]
        shuffled_reference, shuffled_diagnostics = _retrieve(
            shuffled_h, shuffled_route, matched_bank, type_names, temperature
        )
        m4, m4_receipt = _adaptive_fusion(
            shuffled_h,
            shuffled_reference,
            shuffled_diagnostics,
            fusion_alpha,
            support_threshold,
        )
        blank_reference, blank_diagnostics = _retrieve(
            blank_test, blank_route, matched_bank, type_names, temperature
        )
        m5_blank, m5_blank_receipt = _adaptive_fusion(
            blank_test,
            blank_reference,
            blank_diagnostics,
            fusion_alpha,
            support_threshold,
        )
        coordinate_reference, coordinate_diagnostics = _retrieve(
            coordinate_test, coordinate_route, matched_bank, type_names, temperature
        )
        m5_coordinate, m5_coordinate_receipt = _adaptive_fusion(
            coordinate_test,
            coordinate_reference,
            coordinate_diagnostics,
            fusion_alpha,
            support_threshold,
        )

        indication = source.indication_ids[test][0]
        same_indication = sorted(
            donor
            for donor in donors_all
            if donor != held_out
            and source.indication_ids[source.donor_ids == donor][0] == indication
        )
        wrong_conditions[held_out] = {}
        local_truth = fold.full[test]
        local_donors = source.donor_ids[test]
        local_sections = source.section_ids[test]
        local_types_base = type_names[np.argmax(route_test, axis=1)]
        local_types = local_types_base
        local_wrong_predictions: list[np.ndarray] = []
        local_wrong_sections: list[np.ndarray] = []
        wrong_bank_receipts: dict[str, Mapping[str, object]] = {}
        for wrong_index, wrong_donor in enumerate(same_indication):
            wrong_bank, wrong_bank_receipt = _fold_bank(
                source,
                calibrated_reference,
                held_out,
                [wrong_donor],
                bank_mode,
                pooled=False,
                prototypes_per_type=prototypes_per_type,
                seed=seed + 1_000 + fold_index * 20 + wrong_index,
                bank_role=f"hard_wrong_M6::{wrong_donor}",
                fail_if_no_qualified_types=False,
            )
            wrong_bank_receipts[wrong_donor] = wrong_bank_receipt
            wrong_reference, wrong_diagnostics = _retrieve(
                h_test, route_test, wrong_bank, type_names, temperature
            )
            wrong, wrong_receipt = _adaptive_fusion(
                h_test,
                wrong_reference,
                wrong_diagnostics,
                fusion_alpha,
                support_threshold,
            )
            repeated_wrong = wrong
            local_wrong_predictions.append(repeated_wrong)
            local_wrong_sections.append(
                np.asarray([f"{value}::wrong={wrong_donor}" for value in local_sections])
            )
            wrong_truth.append(local_truth)
            wrong_prediction.append(repeated_wrong)
            wrong_donors.append(local_donors)
            wrong_sections.append(local_wrong_sections[-1])
            wrong_types.append(local_types)
            wrong_conditions[held_out][wrong_donor] = float(
                donor_section_macro_loss(
                    local_truth,
                    repeated_wrong,
                    local_donors,
                    local_sections,
                )["donor_section_macro_mse"]
            )
            coverage["M6"].append(wrong_receipt)

        generic_bank, generic_receipt = _fold_bank(
            source,
            calibrated_reference,
            held_out,
            same_indication,
            bank_mode,
            pooled=True,
            prototypes_per_type=prototypes_per_type,
            seed=seed + 2_000 + fold_index,
            bank_role="pooled_wrong_M7",
            fail_if_no_qualified_types=False,
        )
        generic_reference, generic_diagnostics = _retrieve(
            h_test, route_test, generic_bank, type_names, temperature
        )
        m7, m7_receipt = _adaptive_fusion(
            h_test,
            generic_reference,
            generic_diagnostics,
            fusion_alpha,
            support_threshold,
        )

        # M8 maps each disjoint half to the other half without held-out ST.
        # Order the paired mapping so predictions align exactly with the shared
        # [half-A, half-B] truth rows used by every arm: B->A followed by A->B.
        oracle_x_train = np.vstack((fold.half_b[train], fold.half_a[train]))
        oracle_y_train = np.vstack((fold.half_a[train], fold.half_b[train]))
        oracle_donors_train = np.concatenate((source.donor_ids[train], source.donor_ids[train]))
        oracle_sections_train = np.concatenate(
            (source.section_ids[train], source.section_ids[train])
        )
        oracle_alpha, oracle_cv = _select_ridge_alpha(
            oracle_x_train,
            oracle_y_train,
            oracle_donors_train,
            oracle_sections_train,
            ridge_alphas,
            seed=seed + 3_000 + fold_index,
            device=device,
        )
        oracle_x_test = np.vstack((fold.half_b[test], fold.half_a[test]))
        m8 = _fit_predict_ridge(
            oracle_x_train,
            oracle_y_train,
            oracle_x_test,
            oracle_donors_train,
            oracle_sections_train,
            oracle_alpha,
            device,
        )
        oracle_truth = np.vstack((fold.half_a[test], fold.half_b[test]))
        oracle_donors = np.concatenate((source.donor_ids[test], source.donor_ids[test]))
        oracle_sections = np.concatenate((source.section_ids[test], source.section_ids[test]))
        oracle_types = np.concatenate((local_types_base, local_types_base))

        repeated = {
            "M0": h_test,
            "M1": m1,
            "M2": m2,
            "M3": m3,
            "M4": m4,
            "M5_blank": m5_blank,
            "M5_coordinates": m5_coordinate,
            "M7": m7,
        }
        if program_gate is not None and program_gate["status"] == "feasible":
            retained_names = set(program_gate["retained_programs"])
            retained_indices = np.asarray(
                [
                    index
                    for index, name in enumerate(fold.target_names.tolist())
                    if name in retained_names
                ],
                dtype=np.int64,
            )
            for model_name, model_prediction in repeated.items():
                local_score = donor_section_macro_loss(
                    local_truth[:, retained_indices],
                    model_prediction[:, retained_indices],
                    local_donors,
                    local_sections,
                )
                value = float(local_score["donor_section_macro_mse"])
                primary_program_donor_loss[model_name][held_out] = value
            oracle_primary = donor_section_macro_loss(
                oracle_truth[:, retained_indices],
                m8[:, retained_indices],
                oracle_donors,
                oracle_sections,
            )
            primary_program_donor_loss["M8"][held_out] = (
                float(oracle_primary["donor_section_macro_mse"]) / 4.0
            )
            wrong_primary_truth = np.vstack(
                [local_truth[:, retained_indices] for _ in local_wrong_predictions]
            )
            wrong_primary_prediction = np.vstack(
                [value[:, retained_indices] for value in local_wrong_predictions]
            )
            wrong_primary_donors = np.concatenate([local_donors for _ in local_wrong_predictions])
            wrong_primary_sections = np.concatenate(local_wrong_sections)
            primary_program_donor_loss["M6"][held_out] = float(
                donor_section_macro_loss(
                    wrong_primary_truth,
                    wrong_primary_prediction,
                    wrong_primary_donors,
                    wrong_primary_sections,
                )["donor_section_macro_mse"]
            )
        truth_rows.append(local_truth)
        donor_rows.append(local_donors)
        section_rows.append(local_sections)
        type_rows.append(local_types)
        for name, value in repeated.items():
            predictions[name].append(value)
        floor_truth_rows.append(oracle_truth)
        floor_prediction_rows.append(m8)
        floor_donor_rows.append(oracle_donors)
        floor_section_rows.append(oracle_sections)
        floor_type_rows.append(oracle_types)
        if endpoint == "program_total":
            threshold = np.quantile(fold.full[train], 0.90, axis=0)
            threshold_rows.append(np.repeat(threshold[None, :], len(local_truth), axis=0))

        coverage["M3"].append(m3_receipt)
        coverage["M4"].append(m4_receipt)
        coverage["M5_blank"].append(m5_blank_receipt)
        coverage["M5_coordinates"].append(m5_coordinate_receipt)
        coverage["M7"].append(m7_receipt)
        fold_receipts[held_out] = {
            "outer_heldout_donor": held_out,
            "outer_training_donors": train_donors,
            "heldout_sections": sorted(set(source.section_ids[test].tolist())),
            "heldout_rows": int(test.sum()),
            "target_basis": fold.basis_receipt,
            "reference_calibration_fit_donors": list(calibrator.fit_donors),
            "reference_calibration_excludes_heldout": held_out not in calibrator.fit_donors,
            "H_ridge_alpha": h_alpha,
            "H_inner_grouped_donor_cv": h_cv,
            "router_ridge_alpha": router_alpha,
            "router_inner_grouped_donor_cv": router_cv,
            "router_training_label_provenance": (
                "training_donor_ST_to_matched_sc_centroid pseudo-label; heldout ST never used; "
                "regional exploratory routing only"
            ),
            "coordinate_ridge_alpha": coordinate_alpha,
            "coordinate_inner_grouped_donor_cv": coordinate_cv,
            "fusion_alpha": fusion_alpha,
            "fusion_temperature": temperature,
            "support_distance_threshold_training_q95": support_threshold,
            "fusion_inner_grouped_donor_cv": fusion_cv,
            "matched_bank": matched_receipt,
            "inner_bank_receipts": inner_bank_receipts,
            "generic_bank": generic_receipt,
            "wrong_bank_receipts": wrong_bank_receipts,
            "generic_excludes_query_donor": held_out not in same_indication,
            "wrong_donors_all_same_indication": same_indication,
            "shuffle_receipt": {
                "within_section": True,
                "fixed_points": int(np.sum(derangement == np.arange(len(derangement)))),
                "seed": seed + fold_index,
            },
            "blank_image_receipt": source.blank_receipt,
            "oracle_alpha": oracle_alpha,
            "oracle_inner_grouped_donor_cv": oracle_cv,
            "program_reliability_training_only": program_gate,
            "heldout_ST_used_for_fit_selection_or_support_threshold": False,
            (
                "failed_reference_sensitivity_donors_excluded_from_all_primary_"
                "fit_selection_inference"
            ): [FAILED_REFERENCE_SENSITIVITY_DONOR],
        }

    assert target_names is not None
    truth = np.vstack(truth_rows)
    donors = np.concatenate(donor_rows)
    sections = np.concatenate(section_rows)
    types = np.concatenate(type_rows)
    joined_predictions = {name: np.vstack(values) for name, values in predictions.items()}
    scores = {
        name: _add_target_metrics(
            _score_model(truth, value, donors, sections, types),
            truth,
            value,
            donors,
            sections,
            target_names,
        )
        for name, value in joined_predictions.items()
    }
    floor_truth = np.vstack(floor_truth_rows)
    floor_prediction = np.vstack(floor_prediction_rows)
    floor_donors = np.concatenate(floor_donor_rows)
    floor_sections = np.concatenate(floor_section_rows)
    floor_types = np.concatenate(floor_type_rows)
    scores["M8_raw_cross_half"] = _add_target_metrics(
        _score_model(
            floor_truth,
            floor_prediction,
            floor_donors,
            floor_sections,
            floor_types,
        ),
        floor_truth,
        floor_prediction,
        floor_donors,
        floor_sections,
        target_names,
    )
    scores["M8"] = _full_depth_floor_metrics(scores["M8_raw_cross_half"])
    scores["M8"]["adjacent_section_replicate_sensitivity"] = {
        "status": "not_run_registration_mapping_unavailable",
        "candidate_multi_section_donors": ["B1", "L1"],
        "reason": (
            "serial sections are present but no released spot-to-spot registration defines "
            "replicate molecular observations on identical regional units"
        ),
    }

    wrong_truth_array = np.vstack(wrong_truth)
    wrong_prediction_array = np.vstack(wrong_prediction)
    wrong_donor_array = np.concatenate(wrong_donors)
    wrong_section_array = np.concatenate(wrong_sections)
    wrong_type_array = np.concatenate(wrong_types)
    scores["M6"] = _add_target_metrics(
        _score_model(
            wrong_truth_array,
            wrong_prediction_array,
            wrong_donor_array,
            wrong_section_array,
            wrong_type_array,
        ),
        wrong_truth_array,
        wrong_prediction_array,
        wrong_donor_array,
        wrong_section_array,
        target_names,
    )
    scores["M5"] = {
        "blank": scores["M5_blank"],
        "coordinates": scores["M5_coordinates"],
        "purpose": "two prespecified image-content ablations under one M5 arm",
    }

    comparisons = {
        "M3_vs_M0_incremental_reference": ("M3", "M0"),
        "M3_vs_M1_image_beyond_reference": ("M3", "M1"),
        "M3_vs_M2_continuous_state_beyond_type_routing": ("M3", "M2"),
        "M3_vs_M4_exact_pairing": ("M3", "M4"),
        "M3_vs_M5_blank_image_content": ("M3", "M5_blank"),
        "M3_vs_M5_coordinates_image_content": ("M3", "M5_coordinates"),
        "M3_vs_M6_matched_specificity": ("M3", "M6"),
        "M3_vs_M7_generic_specificity": ("M3", "M7"),
    }
    candidate_loss_by_model = {
        model: scores[model]["donor_mse"]
        for model in (
            "M0",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5_blank",
            "M5_coordinates",
            "M6",
            "M7",
            "M8",
        )
    }
    candidate_paired = _paired_inference_from_losses(
        candidate_loss_by_model,
        comparisons,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed + 10_000,
    )
    program_primary_feasible = bool(
        endpoint == "program_total"
        and program_gate_receipts
        and all(value["status"] == "feasible" for value in program_gate_receipts.values())
        and all(len(values) == len(donors_all) for values in primary_program_donor_loss.values())
    )
    primary_program_scores: Mapping[str, object]
    if endpoint == "program_total":
        primary_program_scores = {
            "status": (
                "feasible"
                if program_primary_feasible
                else "blocked_at_least_one_outer_fold_retained_fewer_than_three_programs"
            ),
            "selection_rule": {
                "minimum_spearman": PROGRAM_RELIABILITY_MINIMUM_SPEARMAN,
                "minimum_training_donor_fraction": (PROGRAM_RELIABILITY_MINIMUM_DONOR_FRACTION),
                "minimum_retained_programs_per_outer_fold": (PROGRAM_RELIABILITY_MINIMUM_PROGRAMS),
                "heldout_ST_used": False,
            },
            "fold_gates": program_gate_receipts,
            "scores": {
                model: {
                    "donor_mse": values,
                    "donor_section_macro_mse": (
                        float(np.mean(list(values.values()))) if values else None
                    ),
                }
                for model, values in primary_program_donor_loss.items()
            },
        }
    else:
        primary_program_scores = {"status": "not_applicable"}

    if program_primary_feasible:
        active_loss_by_model = primary_program_donor_loss
        paired = _paired_inference_from_losses(
            active_loss_by_model,
            comparisons,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + 20_000,
        )
        headline_role = "primary_outer_training_reliability_qualified_programs"
    elif endpoint == "program_total":
        active_loss_by_model = candidate_loss_by_model
        paired = {}
        headline_role = "secondary_all_fixed_candidates_primary_endpoint_blocked"
    else:
        active_loss_by_model = candidate_loss_by_model
        paired = candidate_paired
        headline_role = "primary_outer_training_donor_PCA"

    h_loss = float(np.mean(list(active_loss_by_model["M0"].values())))
    fusion_loss = float(np.mean(list(active_loss_by_model["M3"].values())))
    floor_loss = float(np.mean(list(active_loss_by_model["M8"].values())))
    denominator = h_loss - floor_loss
    relative_mse = (h_loss - fusion_loss) / h_loss if h_loss > 0 else None
    relative_rmse = 1.0 - math.sqrt(fusion_loss / h_loss) if h_loss > 0 else None
    gap_closed = (h_loss - fusion_loss) / denominator if denominator > 0 else None
    positive_fraction = float(
        np.mean(
            [
                active_loss_by_model["M3"][donor] < active_loss_by_model["M0"][donor]
                for donor in sorted(active_loss_by_model["M0"])
            ]
        )
    )
    per_donor_headline = {}
    for donor in donors_all:
        if donor not in active_loss_by_model["M0"]:
            continue
        h_donor = float(active_loss_by_model["M0"][donor])
        fusion_donor = float(active_loss_by_model["M3"][donor])
        floor_donor = float(active_loss_by_model["M8"][donor])
        floor_gap = h_donor - floor_donor
        per_donor_headline[donor] = {
            "M0_loss": h_donor,
            "M3_loss": fusion_donor,
            "M8_full_depth_corrected_floor_loss_secondary": floor_donor,
            "relative_MSE_gain_M3_vs_M0": (
                (h_donor - fusion_donor) / h_donor if h_donor > 0 else None
            ),
            "relative_RMSE_gain_M3_vs_M0": (
                1.0 - math.sqrt(fusion_donor / h_donor) if h_donor > 0 else None
            ),
            "gap_closed_secondary": (
                (h_donor - fusion_donor) / floor_gap if floor_gap > 0 else None
            ),
        }

    if endpoint == "program_total":
        thresholds = np.vstack(threshold_rows)
        for model in ("M0", "M1", "M2", "M3", "M4", "M5_blank", "M5_coordinates", "M7"):
            scores[model]["rare_state_recall_coverage"] = _rare_state_metrics(
                truth, joined_predictions[model], thresholds, target_names
            )

    def active_mean(model: str) -> float:
        return float(np.mean(list(active_loss_by_model[model].values())))

    gate_evaluable = endpoint != "program_total" or program_primary_feasible
    floor_inference = (
        _measurement_floor_inference(
            active_loss_by_model,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + 30_000,
        )
        if gate_evaluable
        else None
    )

    return {
        "endpoint": endpoint,
        "bank_mode": bank_mode,
        "target_names": target_names.tolist(),
        "models": {key: MODEL_ARMS.get(key, key) for key in MODEL_ARMS},
        "scores": scores,
        "all_candidate_program_scores_role": (
            "secondary" if endpoint == "program_total" else "not_applicable"
        ),
        "reliability_qualified_program_primary": primary_program_scores,
        "wrong_donor_conditions": wrong_conditions,
        "coverage_uncertainty_abstention": {
            name: _coverage_summary(values) for name, values in coverage.items()
        },
        "paired_inference": paired,
        "measurement_floor_inference": floor_inference,
        "secondary_all_candidate_paired_inference": (
            candidate_paired if endpoint == "program_total" else None
        ),
        "headline": {
            "status": (
                "evaluable" if gate_evaluable else "blocked_program_reliability_feasibility_failed"
            ),
            "loss_role": headline_role,
            "M0_loss": h_loss,
            "M3_loss": fusion_loss,
            "M8_full_depth_corrected_floor_loss_secondary": floor_loss,
            "relative_MSE_gain_M3_vs_M0": relative_mse,
            "relative_RMSE_gain_M3_vs_M0": relative_rmse,
            "positive_donor_fraction_M3_vs_M0": positive_fraction,
            "gap_closed_secondary": gap_closed,
            "per_donor": per_donor_headline,
        },
        "research_prototype_thresholds": {
            "scientific_gate_evaluable": gate_evaluable,
            "at_least_5_percent_relative_MSE_gain": bool(
                gate_evaluable and relative_mse is not None and relative_mse >= 0.05
            ),
            "at_least_70_percent_donors_positive": bool(
                gate_evaluable and positive_fraction >= 0.70
            ),
            "M3_beats_M1": bool(gate_evaluable and active_mean("M3") < active_mean("M1")),
            "M3_beats_M2_key_state_selection_control": bool(
                gate_evaluable and active_mean("M3") < active_mean("M2")
            ),
            "M3_beats_M4": bool(gate_evaluable and active_mean("M3") < active_mean("M4")),
            "M3_beats_all_wrong_mean": bool(
                gate_evaluable and active_mean("M3") < active_mean("M6")
            ),
            "M3_beats_M7": bool(gate_evaluable and active_mean("M3") < active_mean("M7")),
        },
        "within_type_residual_state_endpoint": {
            "status": "blocked_unavailable",
            "reason": (
                "No barcode-keyed outcome-independent registered spot type is released. "
                "Held-out ST-derived types are forbidden; predicted routing types are used "
                "only for balanced total-expression stratification and cannot authorize the "
                "confirmatory residual y_ST-minus-mu[d,type] endpoint."
            ),
            "required_to_unblock": "registered pathology/H&E spot labels with barcode provenance",
        },
        "evaluation_rows": {
            "M0_through_M7": "one prediction row per held-out Visium spot",
            "M8": "two cross-fitted split-half rows per same underlying held-out spot",
            "same_underlying_heldout_spot_identities": True,
            "M8_rows": int(scores["M8_raw_cross_half"]["rows"]),
            "underlying_spots": int(len(source.spot_ids)),
        },
        "iteration": {
            "status": "not_run_prohibited_by_first_experiment_protocol",
            "maximum_rounds": 1,
            "implementation_present": False,
        },
        "fold_receipts": fold_receipts,
    }


def _failed_reference_sensitivity_report(source: RegionalSource) -> Mapping[str, object]:
    audit = dict(source.failed_reference_sensitivity)
    qualified = int(audit["qualified_type_count"])
    if qualified:
        raise ValueError(
            "the frozen B2 failed-reference sensitivity unexpectedly has a qualified type"
        )
    spot_rows = int(audit["spot_rows"])
    return {
        "status": "descriptive_failed_reference_sensitivity_complete",
        "role": "descriptive_only_excluded_from_primary_modeling_and_inference",
        "donor": FAILED_REFERENCE_SENSITIVITY_DONOR,
        "published_exclusion": "Chromium_failure_50_nuclei_no_tumor_cells",
        **audit,
        "used_for_any_primary_fit_selection_support_threshold_or_inference": False,
        "molecular_outcomes_used": False,
        "unsupported_spot_rows": spot_rows,
        "unsupported_spot_fraction": 1.0,
        "adaptive_alpha": {
            "unique_values": [0.0],
            "all_rows_exactly_zero": True,
        },
        "M3_equals_M0": {
            "exact": True,
            "equality": "bitwise_exact_by_construction",
            "reason": "zero qualified reference types forces reference alpha to exactly zero",
        },
        "statistical_testing": "not_run_prohibited_for_failed_reference_sensitivity",
        "outcome_metrics": "not_computed",
    }


def _decisive_hypothesis_decision(
    experiments: Mapping[str, Mapping[str, object]],
    global_holm: Mapping[str, float],
) -> Mapping[str, object]:
    expected_experiments = {
        f"{endpoint}::{bank_mode}"
        for endpoint in ("program_total", "pca_total")
        for bank_mode in ("natural", "composition_equalized")
    }
    rows: dict[str, object] = {}
    missing: list[str] = []
    for experiment_name in sorted(expected_experiments):
        experiment = experiments.get(experiment_name)
        if experiment is None or experiment.get("headline", {}).get("status") != "evaluable":
            missing.extend(
                f"{experiment_name}::{comparison}" for comparison in DECISIVE_COMPARISONS
            )
            continue
        paired = experiment.get("paired_inference")
        if not isinstance(paired, Mapping):
            missing.extend(
                f"{experiment_name}::{comparison}" for comparison in DECISIVE_COMPARISONS
            )
            continue
        for comparison in DECISIVE_COMPARISONS:
            key = f"{experiment_name}::{comparison}"
            summary = paired.get(comparison)
            adjusted = global_holm.get(key)
            if not isinstance(summary, Mapping) or adjusted is None:
                missing.append(key)
                continue
            mean_effect = float(summary["mean_effect"])
            adjusted_value = float(adjusted)
            rows[key] = {
                "effect_definition": "control_loss_minus_M3_loss; positive favors M3",
                "mean_effect": mean_effect,
                "global_holm_adjusted_exact_sign_flip_p": adjusted_value,
                "direction_passed": mean_effect > 0.0,
                "multiplicity_passed": adjusted_value <= 0.05,
                "passed": bool(mean_effect > 0.0 and adjusted_value <= 0.05),
            }
    evaluable = not missing and len(rows) == len(expected_experiments) * len(
        DECISIVE_COMPARISONS
    )
    supported = bool(evaluable and all(bool(row["passed"]) for row in rows.values()))
    decision = (
        "supported"
        if supported
        else ("not_supported" if evaluable else "blocked_indeterminate")
    )
    return {
        "schema": "heir.natcommun_decisive_hypothesis.v1",
        "hypothesis": "L(M3) < min[L(M0), L(M1), L(M2), L(M4), L(M6), L(M7)]",
        "alpha": 0.05,
        "multiplicity_method": "Holm_across_all_evaluable_endpoint_bank_model_comparisons",
        "multiplicity_family_size": len(global_holm),
        "required_experiments": sorted(expected_experiments),
        "required_comparisons": list(DECISIVE_COMPARISONS),
        "comparisons": rows,
        "missing_or_blocked": missing,
        "status": "evaluable" if evaluable else "blocked_fail_closed",
        "decision": decision,
        "supported": supported,
        "fail_closed": True,
    }


def _measurement_floor_decision(
    experiments: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    expected_experiments = {
        f"{endpoint}::{bank_mode}"
        for endpoint in ("program_total", "pca_total")
        for bank_mode in ("natural", "composition_equalized")
    }
    raw_p: dict[str, float] = {}
    summaries: dict[str, Mapping[str, object]] = {}
    missing: list[str] = []
    for name in sorted(expected_experiments):
        experiment = experiments.get(name)
        inference = (
            experiment.get("measurement_floor_inference")
            if isinstance(experiment, Mapping)
            else None
        )
        if (
            not isinstance(experiment, Mapping)
            or experiment.get("headline", {}).get("status") != "evaluable"
            or not isinstance(inference, Mapping)
        ):
            missing.append(name)
            continue
        raw_p[name] = float(inference["exact_sign_flip_p"])
        summaries[name] = inference
    adjusted = holm_adjust(raw_p) if raw_p else {}
    rows = {
        name: {
            "effect_definition": "M3_loss_minus_M8_loss; positive supports floor_below_M3",
            "mean_effect": float(summary["mean_effect"]),
            "exact_sign_flip_p": float(summary["exact_sign_flip_p"]),
            "holm_adjusted_p_within_floor_family": float(adjusted[name]),
            "direction_passed": float(summary["mean_effect"]) > 0.0,
            "multiplicity_passed": float(adjusted[name]) <= 0.05,
            "passed": bool(
                float(summary["mean_effect"]) > 0.0 and float(adjusted[name]) <= 0.05
            ),
        }
        for name, summary in summaries.items()
    }
    evaluable = not missing and len(rows) == len(expected_experiments)
    supported = bool(evaluable and all(bool(row["passed"]) for row in rows.values()))
    decision = (
        "supported"
        if supported
        else ("not_supported" if evaluable else "blocked_indeterminate")
    )
    return {
        "schema": "heir.natcommun_measurement_floor_decision.v1",
        "hypothesis": "L(M8_ST_floor) < L(M3_H_and_E_plus_matched_scRNA)",
        "role": "separate_required_measurement_floor_inequality",
        "alpha": 0.05,
        "multiplicity_method": "Holm_across_four_endpoint_bank_floor_comparisons",
        "required_experiments": sorted(expected_experiments),
        "comparisons": rows,
        "missing_or_blocked": missing,
        "status": "evaluable" if evaluable else "blocked_fail_closed",
        "decision": decision,
        "supported": supported,
        "fail_closed": True,
    }


def _effect_size_consistency_decision(
    experiments: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    expected_experiments = {
        f"{endpoint}::{bank_mode}"
        for endpoint in ("program_total", "pca_total")
        for bank_mode in ("natural", "composition_equalized")
    }
    rows: dict[str, object] = {}
    missing_or_inconsistent: list[str] = []
    for name in sorted(expected_experiments):
        experiment = experiments.get(name)
        headline = experiment.get("headline") if isinstance(experiment, Mapping) else None
        thresholds = (
            experiment.get("research_prototype_thresholds")
            if isinstance(experiment, Mapping)
            else None
        )
        if (
            not isinstance(headline, Mapping)
            or headline.get("status") != "evaluable"
            or not isinstance(thresholds, Mapping)
        ):
            missing_or_inconsistent.append(name)
            continue
        try:
            stored_m0 = float(headline["M0_loss"])
            stored_m3 = float(headline["M3_loss"])
            stored_relative = float(headline["relative_MSE_gain_M3_vs_M0"])
            stored_positive = float(headline["positive_donor_fraction_M3_vs_M0"])
            per_donor = headline["per_donor"]
            expected_donor_set = set(EXPECTED_PRIMARY_DONOR_IDS)
            if (
                not isinstance(per_donor, Mapping)
                or len(per_donor) != EXPECTED_PRIMARY_DONORS
                or set(str(donor) for donor in per_donor) != expected_donor_set
            ):
                raise ValueError("paired donor identity or count differs from the frozen cohort")
            donor_m0 = []
            donor_m3 = []
            for donor in EXPECTED_PRIMARY_DONOR_IDS:
                values = per_donor[donor]
                if not isinstance(values, Mapping):
                    raise ValueError("paired donor headline row is malformed")
                donor_m0.append(float(values["M0_loss"]))
                donor_m3.append(float(values["M3_loss"]))
            if not np.isfinite(donor_m0).all() or not np.isfinite(donor_m3).all():
                raise ValueError("paired donor losses must be finite")
            recomputed_m0 = float(np.mean(donor_m0, dtype=np.float64))
            recomputed_m3 = float(np.mean(donor_m3, dtype=np.float64))
            donor_positive = [m3 < m0 for m0, m3 in zip(donor_m0, donor_m3)]
            recomputed_relative = (recomputed_m0 - recomputed_m3) / recomputed_m0
            recomputed_positive = float(np.mean(donor_positive))
            relative_pass = recomputed_relative >= 0.05
            positive_pass = recomputed_positive >= 0.70
            if (
                not np.isfinite(
                    [stored_m0, stored_m3, stored_relative, stored_positive]
                ).all()
                or recomputed_m0 <= 0.0
                or not np.isclose(stored_m0, recomputed_m0, rtol=0.0, atol=1.0e-12)
                or not np.isclose(stored_m3, recomputed_m3, rtol=0.0, atol=1.0e-12)
                or not np.isclose(stored_relative, recomputed_relative, rtol=0.0, atol=1.0e-12)
                or not np.isclose(stored_positive, recomputed_positive, rtol=0.0, atol=1.0e-12)
                or thresholds.get("at_least_5_percent_relative_MSE_gain")
                is not relative_pass
                or thresholds.get("at_least_70_percent_donors_positive")
                is not positive_pass
            ):
                raise ValueError("stored effect-size gate is inconsistent")
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            missing_or_inconsistent.append(name)
            continue
        rows[name] = {
            "donors": list(EXPECTED_PRIMARY_DONOR_IDS),
            "donor_count": EXPECTED_PRIMARY_DONORS,
            "recomputed_donor_macro_M0_loss": recomputed_m0,
            "recomputed_donor_macro_M3_loss": recomputed_m3,
            "headline_mean_tolerance_absolute": 1.0e-12,
            "relative_MSE_gain_M3_vs_M0": recomputed_relative,
            "minimum_relative_MSE_gain": 0.05,
            "relative_MSE_gain_passed": relative_pass,
            "positive_donor_fraction_M3_vs_M0": recomputed_positive,
            "minimum_positive_donor_fraction": 0.70,
            "positive_donor_fraction_passed": positive_pass,
            "passed": bool(relative_pass and positive_pass),
        }
    evaluable = not missing_or_inconsistent and len(rows) == len(expected_experiments)
    supported = bool(evaluable and all(bool(row["passed"]) for row in rows.values()))
    return {
        "schema": "heir.natcommun_effect_size_consistency_decision.v1",
        "required_experiments": sorted(expected_experiments),
        "criteria": {
            "exact_donor_ids": list(EXPECTED_PRIMARY_DONOR_IDS),
            "exact_donor_count": EXPECTED_PRIMARY_DONORS,
            "headline_losses_must_equal_recomputed_donor_macro_means": True,
            "headline_mean_tolerance_absolute": 1.0e-12,
            "minimum_relative_MSE_gain_M3_vs_M0": 0.05,
            "minimum_positive_donor_fraction_M3_vs_M0": 0.70,
            "both_required_in_every_experiment": True,
        },
        "experiments": rows,
        "missing_or_inconsistent": missing_or_inconsistent,
        "status": "evaluable" if evaluable else "blocked_fail_closed",
        "decision": (
            "supported"
            if supported
            else ("not_supported" if evaluable else "blocked_indeterminate")
        ),
        "supported": supported,
        "fail_closed": True,
    }


def _overall_scientific_decision(
    controls: Mapping[str, object],
    floor: Mapping[str, object],
    effect_size_consistency: Mapping[str, object],
) -> Mapping[str, object]:
    blocked = any(
        value.get("status") != "evaluable"
        for value in (controls, floor, effect_size_consistency)
    )
    supported = bool(
        not blocked
        and controls.get("supported") is True
        and floor.get("supported") is True
        and effect_size_consistency.get("supported") is True
    )
    return {
        "schema": "heir.natcommun_overall_scientific_decision.v1",
        "requires": [
            "multiplicity_aware_M3_vs_prespecified_controls",
            "separate_multiplicity_aware_M8_ST_floor_below_M3",
            "at_least_5_percent_M3_relative_MSE_gain_in_every_experiment",
            "at_least_70_percent_positive_donors_in_every_experiment",
        ],
        "controls_supported": controls.get("supported") is True,
        "measurement_floor_supported": floor.get("supported") is True,
        "effect_size_consistency_supported": effect_size_consistency.get("supported") is True,
        "status": "blocked_fail_closed" if blocked else "evaluable",
        "decision": (
            "blocked_indeterminate"
            if blocked
            else ("supported" if supported else "not_supported")
        ),
        "supported": supported,
        "scientific_authorization_requires_all_components": True,
        "fail_closed": True,
    }


def _validate_sha256(value: str, label: str) -> str:
    raw = str(value)
    normalized = raw.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ) or raw != normalized:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return normalized


def _recompute_hest_positive_control_gate(
    controls: object,
) -> Mapping[str, object]:
    if not isinstance(controls, Mapping):
        raise ValueError("HEST positive_controls must be a complete object")
    classification_rows: dict[str, Mapping[str, object]] = {}
    classification_passes: list[bool] = []
    try:
        for target in ("broad_lineage", "fine_type"):
            row = controls[target]
            model = float(row["model"]["donor_balanced"]["donor_macro_balanced_accuracy"])
            baseline = float(
                row["training_majority_baseline"]["donor_balanced"][
                    "donor_macro_balanced_accuracy"
                ]
            )
            passed = bool(np.isfinite(model) and np.isfinite(baseline) and model > baseline)
            classification_rows[target] = {
                "metric": "donor_macro_balanced_accuracy",
                "model": model,
                "outer_training_majority_baseline": baseline,
                "minimum_required_increment": 0.0,
                "observed_increment": model - baseline,
                "passed": passed,
            }
            classification_passes.append(passed)
        morphology_scores = controls["nuclear_morphology"]["full_context"]["scores"][
            "targets"
        ]
        morphology_rows: dict[str, Mapping[str, object]] = {}
        morphology_passes: list[bool] = []
        for target in HEST_REQUIRED_MORPHOLOGY_TARGETS:
            value = float(
                morphology_scores[target]["donor_type_macro_reference_error_reduction"]
            )
            passed = bool(np.isfinite(value) and value > 0.0)
            morphology_rows[target] = {
                "metric": "donor_type_macro_reference_error_reduction",
                "minimum_required": 0.0,
                "observed": value,
                "passed": passed,
            }
            morphology_passes.append(passed)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("HEST positive_controls are missing a required gate input") from error
    passed = bool(all(classification_passes) and all(morphology_passes))
    return {
        "schema": "heir.hest_visible_positive_control_gate.v1",
        "evaluated_before_molecular_models": True,
        "thresholds_frozen_without_hoptimus_molecular_outcomes": True,
        "natural_unmasked_112um_is_primary": True,
        "classification": classification_rows,
        "full_context_morphology": morphology_rows,
        "nucleus_mask_only_role": "secondary_attribution_not_used_for_gate",
        "passed": passed,
        "molecular_interpretation_allowed": passed,
        "failure_action": (
            None
            if passed
            else "stop_before_molecular_models_and_audit_loading_preprocessing_crop_scale"
        ),
    }


def _load_hest_hoptimus_qualification(args: argparse.Namespace) -> Mapping[str, object]:
    path = args.hest_hoptimus_qualification_report.expanduser().resolve()
    expected_report_sha = _validate_sha256(
        args.expected_hest_hoptimus_qualification_report_sha256,
        "expected HEST H-optimus-1 qualification report hash",
    )
    expected_source_sha = _validate_sha256(
        args.expected_hest_hoptimus_source_sha256,
        "expected HEST H-optimus-1 source hash",
    )
    expected_runner_sha = _validate_sha256(
        args.expected_hest_runner_sha256,
        "expected HEST runner hash",
    )
    if not path.is_file():
        raise ValueError("HEST H-optimus-1 qualification report is missing")
    if _sha256(FROZEN_HEST_RUNNER) != expected_runner_sha:
        raise ValueError("current HEST runner does not match the expected qualification runner")
    try:
        report_bytes = path.read_bytes()
        if hashlib.sha256(report_bytes).hexdigest() != expected_report_sha:
            raise ValueError("HEST H-optimus-1 qualification report hash does not match")
        report = json.loads(report_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("HEST H-optimus-1 qualification report is not valid JSON") from error
    identity = {
        "schema": report.get("schema"),
        "analysis_status": report.get("analysis_status"),
        "study_stage": report.get("study_stage"),
        "requested_phase": report.get("requested_phase"),
        "execution_status": report.get("execution_status"),
        "encoder": report.get("encoder"),
        "encoder_revision": report.get("encoder_revision"),
        "encoder_manifest_sha256": report.get("encoder_manifest_sha256"),
        "encoder_role": report.get("encoder_role"),
        "encoder_feature_width": report.get("encoder_feature_width"),
        "source_sha256": report.get("source_sha256"),
        "source_receipt_expected_sha256": report.get("source_receipt_expected_sha256"),
    }
    expected_identity = {
        "schema": "heir.hest_scientific_reanalysis.v2",
        "analysis_status": "retrospective_exposed_non_authorizing",
        "study_stage": "retrospective_exposed",
        "requested_phase": "full",
        "execution_status": "scientific_reanalysis_complete",
        "encoder": EXPECTED_ENCODER_REPOSITORY,
        "encoder_revision": HOPTIMUS1_REVISION,
        "encoder_manifest_sha256": HOPTIMUS1_MANIFEST_SHA256,
        "encoder_role": "primary_Hoptimus1_qualification",
        "encoder_feature_width": 1536,
        "source_sha256": expected_source_sha,
        "source_receipt_expected_sha256": expected_source_sha,
    }
    if identity != expected_identity:
        raise ValueError("HEST H-optimus-1 qualification identity is incomplete or inconsistent")
    gate = report.get("positive_control_gate")
    recomputed_gate = _recompute_hest_positive_control_gate(report.get("positive_controls"))
    if (
        not isinstance(gate, Mapping)
        or dict(gate) != dict(recomputed_gate)
        or recomputed_gate.get("natural_unmasked_112um_is_primary") is not True
        or recomputed_gate.get("passed") is not True
        or recomputed_gate.get("molecular_interpretation_allowed") is not True
    ):
        raise ValueError(
            "HEST H-optimus-1 visible positive-control gate is incomplete, stale, or failed"
        )
    preflight = report.get("same_runner_uni2_comparator_preflight")
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("schema") != "heir.hest_same_runner_uni2_preflight.v1"
        or preflight.get("passed") is not True
    ):
        raise ValueError("HEST H-optimus-1 same-runner UNI2 comparator preflight did not pass")
    implementation = report.get("implementation_receipt")
    file_hashes = implementation.get("file_sha256") if isinstance(implementation, Mapping) else None
    expected_nested = _validate_sha256(
        args.expected_nested_ridge_sha256, "expected nested-ridge hash"
    )
    expected_scoring = _validate_sha256(
        args.expected_scoring_sha256, "expected scoring hash"
    )
    expected_measurement = _validate_sha256(
        args.expected_hest_measurement_sha256, "expected HEST measurement hash"
    )
    if _sha256(FROZEN_HEST_MEASUREMENT) != expected_measurement:
        raise ValueError("current HEST measurement implementation does not match expected hash")
    if (
        not isinstance(file_hashes, Mapping)
        or file_hashes.get("scripts/benchmark_hest_scientific_reanalysis.py")
        != expected_runner_sha
        or file_hashes.get("src/heir/evaluation/hest_nested_ridge.py") != expected_nested
        or file_hashes.get("src/heir/evaluation/hest_scoring.py") != expected_scoring
        or file_hashes.get("src/heir/evaluation/hest_measurement.py")
        != expected_measurement
    ):
        raise ValueError("HEST qualification implementation receipt is not current")
    hest_command = implementation.get("command") if isinstance(implementation, Mapping) else None
    required_command_options = {
        "--phase": "full",
        "--device": "cuda",
        "--representation-profile": "full",
        "--expected-source-sha256": expected_source_sha,
        "--expected-encoder": EXPECTED_ENCODER_REPOSITORY,
    }
    if not isinstance(hest_command, Sequence) or isinstance(hest_command, (str, bytes)):
        raise ValueError("HEST qualification command receipt is missing")
    command_values = list(hest_command)
    for option, value in required_command_options.items():
        if command_values.count(option) != 1:
            raise ValueError(f"HEST qualification command must contain {option} once")
        index = command_values.index(option)
        if index + 1 >= len(command_values) or command_values[index + 1] != value:
            raise ValueError(f"HEST qualification command has an inconsistent {option}")
    if command_values.count("--comparison-report") != 1 or (
        "--allow-gate-failed-uni2-baseline-only" in command_values
    ):
        raise ValueError("HEST qualification command does not preserve the primary H-optimus role")
    backend = report.get("numeric_backend")
    if (
        not isinstance(backend, Mapping)
        or backend.get("requested_device") != "cuda"
        or backend.get("cuda_available") is not True
        or backend.get("deterministic_algorithms_enabled") is not True
        or backend.get("cublas_workspace_config") != ":4096:8"
        or backend.get("cudnn_deterministic") is not True
        or backend.get("cudnn_benchmark") is not False
        or backend.get("cuda_matmul_allow_tf32") is not False
        or backend.get("cudnn_allow_tf32") is not False
    ):
        raise ValueError("HEST qualification numeric backend is not deterministic CUDA")
    return {
        "schema": "heir.natcommun_hest_hoptimus_prerequisite.v1",
        "status": "passed_before_natcommun_molecular_interpretation",
        "molecular_interpretation_prerequisite_satisfied": True,
        "report": str(path),
        "report_sha256": expected_report_sha,
        "source_sha256": expected_source_sha,
        "hest_runner_sha256": expected_runner_sha,
        "encoder": EXPECTED_ENCODER_REPOSITORY,
        "encoder_revision": HOPTIMUS1_REVISION,
        "encoder_role": "primary_Hoptimus1_qualification",
        "execution_status": "scientific_reanalysis_complete",
        "positive_control_gate_passed": True,
        "molecular_interpretation_allowed": True,
    }


def _implementation_receipt(
    args: argparse.Namespace, source: RegionalSource
) -> Mapping[str, object]:
    paths = {
        "protocol": FROZEN_PROTOCOL,
        "source_builder": FROZEN_SOURCE_BUILDER,
        "benchmark_runner": Path(__file__).resolve(),
        "reference_fusion": FROZEN_REFERENCE_FUSION,
        "nested_ridge": FROZEN_NESTED_RIDGE,
        "scoring": FROZEN_SCORING,
        "hest_measurement": FROZEN_HEST_MEASUREMENT,
    }
    actual = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "protocol": _validate_sha256(args.expected_protocol_sha256, "expected protocol hash"),
        "source_builder": _validate_sha256(
            args.expected_builder_sha256, "expected builder hash"
        ),
        "benchmark_runner": _validate_sha256(
            args.expected_runner_sha256, "expected runner hash"
        ),
        "reference_fusion": _validate_sha256(
            args.expected_reference_fusion_sha256, "expected reference-fusion hash"
        ),
        "nested_ridge": _validate_sha256(
            args.expected_nested_ridge_sha256, "expected nested-ridge hash"
        ),
        "scoring": _validate_sha256(args.expected_scoring_sha256, "expected scoring hash"),
        "hest_measurement": _validate_sha256(
            args.expected_hest_measurement_sha256, "expected HEST measurement hash"
        ),
    }
    if actual != expected:
        raise ValueError("protocol/builder/runner hashes do not match the expected contract")
    if source.source_receipt.get("protocol_sha256") != actual["protocol"] or (
        source.source_receipt.get("builder_implementation_sha256") != actual["source_builder"]
    ):
        raise ValueError("source receipt is not bound to the exact protocol and builder")
    command = getattr(args, "command", None)
    if not isinstance(command, list) or not command:
        raise ValueError("exact benchmark command receipt is unavailable")
    required_options = {
        "--expected-source-sha256": args.expected_source_sha256,
        "--expected-protocol-sha256": args.expected_protocol_sha256,
        "--expected-builder-sha256": args.expected_builder_sha256,
        "--expected-runner-sha256": args.expected_runner_sha256,
        "--expected-reference-fusion-sha256": args.expected_reference_fusion_sha256,
        "--expected-nested-ridge-sha256": args.expected_nested_ridge_sha256,
        "--expected-scoring-sha256": args.expected_scoring_sha256,
        "--expected-hest-measurement-sha256": args.expected_hest_measurement_sha256,
        "--hest-hoptimus-qualification-report": str(
            args.hest_hoptimus_qualification_report
        ),
        "--expected-hest-hoptimus-qualification-report-sha256": (
            args.expected_hest_hoptimus_qualification_report_sha256
        ),
        "--expected-hest-hoptimus-source-sha256": (
            args.expected_hest_hoptimus_source_sha256
        ),
        "--expected-hest-runner-sha256": args.expected_hest_runner_sha256,
        "--device": "cuda",
    }
    for option, value in required_options.items():
        if command.count(option) != 1:
            raise ValueError(f"exact command must contain {option} once")
        index = command.index(option)
        if index + 1 >= len(command) or command[index + 1] != str(value):
            raise ValueError(f"exact command has an inconsistent {option}")
    try:
        git_head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_head = None
        git_dirty = None
    return {
        "expected_source_sha256": args.expected_source_sha256,
        "file_sha256": actual,
        "expected_file_sha256": expected,
        "command": command,
        "git_head": git_head,
        "git_worktree_dirty_at_start": git_dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _configure_numeric_backend(cpu_threads: int, device: str) -> Mapping[str, object]:
    if device != "cuda":
        raise ValueError("the frozen NatCommun benchmark requires --device cuda")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError(
            "CUDA execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before process start"
        )
    thread_variables = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    expected_threads = str(cpu_threads)
    mismatched_threads = {
        name: os.environ.get(name)
        for name in thread_variables
        if os.environ.get(name) != expected_threads
    }
    if mismatched_threads:
        raise ValueError(
            "CPU thread environment must be set to --cpu-threads before process start: "
            f"{mismatched_threads}"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; CPU fallback is prohibited")
    torch.set_num_threads(cpu_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    return {
        "requested_device": device,
        "actual_device": "cuda:0",
        "cpu_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": str(properties.name),
        "gpu_capability": [int(value) for value in torch.cuda.get_device_capability(0)],
        "gpu_total_memory_bytes": int(properties.total_memory),
        "ridge_dtype": "float32",
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cpu_thread_environment": {
            name: os.environ.get(name) for name in thread_variables
        },
        "cpu_fallback_allowed": False,
    }


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# NatCommun matched-reference regional benchmark",
        "",
        (
            f"Scope: `{report['analysis_scope']}`. This is a retrospective regional, "
            "non-authorizing prototype."
        ),
        "",
        (
            "The first experiment evaluates exactly M0–M8 with a frozen H&E ridge and "
            "one bounded, support-adaptive M3 correction. Iteration is prohibited."
        ),
        "",
        "## Headline results",
        "",
        (
            "| Endpoint | Bank | Status / loss role | M0 loss | M3 loss | "
            "Relative MSE gain | Positive donors | Corrected ST floor |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, result in report["experiments"].items():
        headline = result["headline"]
        lines.append(
            (
                "| {endpoint} | {bank} | {status}; {role} | {m0:.4g} | "
                "{m3:.4g} | {gain} | {positive:.1%} | {floor:.4g} |"
            ).format(
                endpoint=result["endpoint"],
                bank=result["bank_mode"],
                status=headline["status"],
                role=headline["loss_role"],
                m0=headline["M0_loss"],
                m3=headline["M3_loss"],
                gain=(
                    "NA"
                    if headline["relative_MSE_gain_M3_vs_M0"] is None
                    else f"{headline['relative_MSE_gain_M3_vs_M0']:.1%}"
                ),
                positive=headline["positive_donor_fraction_M3_vs_M0"],
                floor=headline["M8_full_depth_corrected_floor_loss_secondary"],
            )
        )
    decisive = report["decisive_hypothesis"]
    floor_decision = report["measurement_floor_decision"]
    effect_decision = report["effect_size_consistency_decision"]
    overall_decision = report["overall_scientific_decision"]
    sensitivity = report["failed_reference_sensitivity"]
    type_counts = ", ".join(
        f"{name}={count}" for name, count in sensitivity["selected_type_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Multiplicity-aware decisive decision",
            "",
            f"- Status: `{decisive['status']}`; decision: `{decisive['decision']}`.",
            (
                f"- Holm family: {decisive['multiplicity_family_size']} tests; "
                f"required M3 comparisons: {len(decisive['comparisons'])}."
            ),
            "- Missing or blocked required comparisons: "
            + (", ".join(decisive["missing_or_blocked"]) or "none"),
            "",
            "## Separate ST-floor inequality",
            "",
            (
                f"- Status: `{floor_decision['status']}`; decision: "
                f"`{floor_decision['decision']}` for `L(M8) < L(M3)`."
            ),
            (
                f"- Overall scientific decision: `{overall_decision['decision']}`; "
                "the M3-control family, separate M8-floor family, >=5% relative-MSE gain, "
                "and >=70% positive donors in every experiment are required."
            ),
            f"- Effect-size/donor-consistency decision: `{effect_decision['decision']}`.",
            "",
            "## Failed-reference B2 sensitivity",
            "",
            f"- Actual selected reference counts: {type_counts}.",
            (
                f"- Qualified types at the frozen >=50-cell rule: "
                f"{sensitivity['qualified_type_count']}; unsupported rows: "
                f"{sensitivity['unsupported_spot_rows']} (100%)."
            ),
            (
                "- Adaptive alpha is exactly zero and M3 is bitwise-identical to M0; "
                "outcome metrics and statistical tests were not run."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Scientific limits",
            "",
            "- The released Visium cohort tests a regional hypothesis, not cell-level annotation.",
            (
                "- Primary image features are frozen bioptimus/H-optimus-1 and require a "
                "passed official-vs-local parity receipt; UNI2-h remains a prespecified "
                "separate comparator and is not run here."
            ),
            (
                "- Confirmatory within-type residual state is blocked because no "
                "barcode-keyed, outcome-independent spot type labels are released."
            ),
            (
                "- M2 uses a disclosed training-donor-only ST/sc pseudo-label to fit an "
                "H&E router; held-out ST never supplies routing labels."
            ),
            (
                "- M8 is secondary. Its raw cross-half error is multiplied by 1/4 under "
                "the independent equal-thinning variance derivation to estimate the "
                "full-depth floor. It uses two split-half rows per same underlying spot."
            ),
            "- No result from this single retrospective cohort authorizes full HEIR development.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    numeric_backend = _configure_numeric_backend(args.cpu_threads, args.device)
    source = load_source(
        args.source,
        expected_primary_donors=args.expected_primary_donors,
        expected_source_sha256=args.expected_source_sha256,
    )
    implementation_receipt = _implementation_receipt(args, source)
    hest_hoptimus_prerequisite = _load_hest_hoptimus_qualification(args)
    experiments = {}
    for endpoint in ("program_total", "pca_total"):
        for bank_mode in ("natural", "composition_equalized"):
            key = f"{endpoint}::{bank_mode}"
            print(f"NatCommun reference fusion: {key}", flush=True)
            experiments[key] = run_endpoint(
                source,
                endpoint,
                bank_mode,
                ridge_alphas=args.ridge_alphas,
                fusion_alphas=args.fusion_alphas,
                temperatures=args.temperatures,
                pca_components=args.pca_components,
                pca_genes=args.pca_genes,
                prototypes_per_type=args.prototypes_per_type,
                bootstrap_iterations=args.bootstrap_iterations,
                seed=args.seed,
                device=args.device,
            )
    p_values = {}
    for experiment_name, experiment in experiments.items():
        for comparison_name, summary in experiment["paired_inference"].items():
            p_values[f"{experiment_name}::{comparison_name}"] = summary["exact_sign_flip_p"]
    global_holm = holm_adjust(p_values)
    decisive_hypothesis = _decisive_hypothesis_decision(experiments, global_holm)
    measurement_floor_decision = _measurement_floor_decision(experiments)
    effect_size_consistency = _effect_size_consistency_decision(experiments)
    overall_scientific_decision = _overall_scientific_decision(
        decisive_hypothesis, measurement_floor_decision, effect_size_consistency
    )
    report = {
        "schema": SCHEMA,
        "analysis_scope": SCOPE,
        "status": "complete_retrospective_regional_non_authorizing",
        "source": str(source.path),
        "source_sha256": args.expected_source_sha256,
        "source_receipt": source.source_receipt,
        "implementation_receipt": implementation_receipt,
        "numeric_backend": numeric_backend,
        "prerequisites": {
            "hest_hoptimus_qualification": hest_hoptimus_prerequisite,
        },
        "design": {
            "primary_donors": sorted(set(source.donor_ids.tolist())),
            "primary_donor_count": len(set(source.donor_ids.tolist())),
            "sections_grouped_with_donor": True,
            "outer_evaluation": "leave_one_donor_out",
            "inner_selection": "grouped_training_donor_CV",
            "same_underlying_heldout_spots_for_M0_through_M8": True,
            "M8_rows_per_underlying_spot": 2,
            "M8_row_definition": "cross_fitted_split_half_A_to_B_and_B_to_A",
            "target_transforms_fit_on_outer_training_donors_only": True,
            "sc_to_ST_calibration_fit_on_outer_training_donors_only": True,
            "reference_type_qualification": {
                "minimum_selected_cells_per_type": REFERENCE_MINIMUM_QUALIFIED_CELLS,
                "applied_after_each_natural_or_equalized_bank_selection": True,
                "subthreshold_types_excluded_from_M1_through_M7": True,
                "unsupported_rows_exact_H_only_fallback_with_zero_alpha": True,
                "primary_bank_without_qualified_types": "fail_closed",
            },
            (
                "failed_reference_sensitivity_donors_excluded_from_all_primary_"
                "fit_selection_inference"
            ): [FAILED_REFERENCE_SENSITIVITY_DONOR],
            "bank_modes": ["natural", "composition_equalized"],
            "model_arms": MODEL_ARMS,
            "frozen_image_encoder": EXPECTED_ENCODER_REPOSITORY,
            "encoder_roles": {
                "primary": {
                    "repository": EXPECTED_ENCODER_REPOSITORY,
                    "official_local_parity": "required_passed_exact_manifest",
                },
                "secondary_comparator": {
                    "repository": SECONDARY_ENCODER_REPOSITORY,
                    "status": "prespecified_separate_not_run_by_this_primary_runner",
                    "historical_evidence_removed": False,
                },
            },
            "iteration": "prohibited",
            "resource_limits": {
                "cpu_threads": args.cpu_threads,
                "ridge_device_request": args.device,
                "dense_program_gene_panel_width": len(source.gene_ids),
                "broad_sparse_common_gene_width": len(source.broad_gene_ids),
                "broad_global_densification": False,
                "maximum_outer_fold_selected_PCA_genes": args.pca_genes,
                "reference_prototypes_per_type": args.prototypes_per_type,
            },
        },
        "experiments": experiments,
        "global_holm_adjusted_exact_sign_flip_p": global_holm,
        "decisive_hypothesis": decisive_hypothesis,
        "measurement_floor_decision": measurement_floor_decision,
        "effect_size_consistency_decision": effect_size_consistency,
        "overall_scientific_decision": overall_scientific_decision,
        "failed_reference_sensitivity": _failed_reference_sensitivity_report(source),
        "authorization": {
            "regional_scientific_hypothesis_supported": overall_scientific_decision[
                "supported"
            ],
            (
                "regional_scientific_authorization_requires_controls_floor_"
                "effect_size_and_donor_consistency"
            ): True,
            "full_HEIR_development_authorized": False,
            "reason": (
                "single retrospective regional cohort; confirmatory residual-state endpoint "
                "and independent replication remain unavailable"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "report.json", report)
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(args.output_dir / "report.json", flush=True)
    return 0


def _float_list(value: str, *, positive: bool, bounded: bool = False) -> tuple[float, ...]:
    values = tuple(float(part) for part in value.split(",") if part.strip())
    if not values or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("grid must contain finite comma-separated numbers")
    if positive and any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("grid values must be positive")
    if bounded and any(item < 0 or item > 0.5 for item in values):
        raise argparse.ArgumentTypeError("fusion alphas must lie in [0, 0.5]")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-builder-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-reference-fusion-sha256", required=True)
    parser.add_argument("--expected-nested-ridge-sha256", required=True)
    parser.add_argument("--expected-scoring-sha256", required=True)
    parser.add_argument("--expected-hest-measurement-sha256", required=True)
    parser.add_argument("--hest-hoptimus-qualification-report", type=Path, required=True)
    parser.add_argument(
        "--expected-hest-hoptimus-qualification-report-sha256", required=True
    )
    parser.add_argument("--expected-hest-hoptimus-source-sha256", required=True)
    parser.add_argument("--expected-hest-runner-sha256", required=True)
    parser.add_argument("--expected-primary-donors", type=int, default=EXPECTED_PRIMARY_DONORS)
    parser.add_argument(
        "--ridge-alphas",
        type=lambda value: _float_list(value, positive=True),
        default=DEFAULT_RIDGE_ALPHAS,
    )
    parser.add_argument(
        "--fusion-alphas",
        type=lambda value: _float_list(value, positive=False, bounded=True),
        default=DEFAULT_FUSION_ALPHAS,
    )
    parser.add_argument(
        "--temperatures",
        type=lambda value: _float_list(value, positive=True),
        default=DEFAULT_TEMPERATURES,
    )
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--pca-genes", type=int, default=256)
    parser.add_argument("--prototypes-per-type", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cuda",), required=True)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args.command = [sys.executable, str(Path(__file__).resolve()), *raw_argv]
    for name in (
        "expected_source_sha256",
        "expected_protocol_sha256",
        "expected_builder_sha256",
        "expected_runner_sha256",
        "expected_reference_fusion_sha256",
        "expected_nested_ridge_sha256",
        "expected_scoring_sha256",
        "expected_hest_measurement_sha256",
        "expected_hest_hoptimus_qualification_report_sha256",
        "expected_hest_hoptimus_source_sha256",
        "expected_hest_runner_sha256",
    ):
        try:
            _validate_sha256(getattr(args, name), name.replace("_", " "))
        except ValueError as error:
            parser.error(str(error))
    if args.expected_primary_donors < 3:
        parser.error("--expected-primary-donors must be at least 3")
    if not 20 <= args.pca_components <= 50:
        parser.error("--pca-components must lie in [20, 50]")
    if args.pca_genes < args.pca_components:
        parser.error("--pca-genes cannot be smaller than --pca-components")
    if args.prototypes_per_type <= 0:
        parser.error("--prototypes-per-type must be positive")
    if args.bootstrap_iterations <= 0:
        parser.error("--bootstrap-iterations must be positive")
    if not 1 <= args.cpu_threads <= 8:
        parser.error("--cpu-threads must lie in [1, 8]")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
