#!/usr/bin/env python3
"""Freeze leakage-safe NatCommun generative gene panels and optional counts.

This is preparation for an *exposed development* experiment.  It does not run a
model, inspect an independent cohort, or authorize a biological claim.  Large
CSR matrices are opened and summarized one at a time so the complete broad
spot-by-gene matrix is never densified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import tempfile
from pathlib import Path
from typing import Mapping

# Bound native thread pools before importing NumPy.  The statistics below are
# sparse scans, so larger BLAS pools add risk without useful throughput.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "4")

import numpy as np  # noqa: E402

from heir.evaluation.gene_panel import (  # noqa: E402
    CSRMatrix,
    PanelMomentBundle,
    canonical_sha256,
    normalized_group_moments,
    panel_artifact,
    project_csr_columns,
    select_gene_panel,
    validate_panel_artifact,
)

SOURCE_SCHEMA = "heir.natcommun_regional_source.v2"
PANEL_SET_SCHEMA = "heir.natcommun_generative_gene_panel_set.v1"
PROJECTED_SCHEMA = "heir.natcommun_generative_projected_counts.v1"
EXPECTED_SOURCE_SHA256 = "ec37d5717a9b737dfac226ae9267258fb728ee024496a7655bb69a913aa3cf20"
EXPECTED_HOPTIMUS_REPOSITORY = "bioptimus/H-optimus-1"
EXPECTED_HOPTIMUS_REVISION = "3592cb220dec7a150c5d7813fb56e68bd57473b9"
DEFAULT_SOURCE = Path("/mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz")
DEFAULT_OUTPUT = Path("configs/natcommun_generative_gene_panel.json")
PANEL_SIZE = 256
MINIMUM_SPLIT_RELIABILITY = 0.05


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


def _json_scalar(archive: np.lib.npyio.NpzFile, name: str) -> Mapping[str, object]:
    if name not in archive.files:
        raise ValueError(f"source lacks {name}")
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"{name} must be a JSON scalar")
    raw = value.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    parsed = json.loads(str(raw))
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{name} must decode to an object")
    return parsed


def _strings(archive: np.lib.npyio.NpzFile, name: str, rows: int | None = None) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"source lacks {name}")
    values = np.asarray(archive[name]).astype(str)
    if values.ndim != 1 or (rows is not None and len(values) != rows):
        raise ValueError(f"{name} must be a row-aligned vector")
    if any(not value for value in values.tolist()):
        raise ValueError(f"{name} contains empty values")
    return values


def _load_csr(archive: np.lib.npyio.NpzFile, prefix: str, rows: int, columns: int) -> CSRMatrix:
    required = [f"{prefix}_{suffix}" for suffix in ("data", "indices", "indptr", "shape")]
    if any(name not in archive.files for name in required):
        raise ValueError(f"source lacks CSR components for {prefix}")
    shape = np.asarray(archive[f"{prefix}_shape"], dtype=np.int64)
    if shape.shape != (2,) or tuple(shape.tolist()) != (rows, columns):
        raise ValueError(f"{prefix} shape does not align with source metadata")
    matrix = CSRMatrix(
        np.asarray(archive[f"{prefix}_data"]),
        np.asarray(archive[f"{prefix}_indices"]),
        np.asarray(archive[f"{prefix}_indptr"]),
        (rows, columns),
    )
    return matrix.validate(prefix)


def _rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.  This repository's supported execution
    # environment is Linux/CUDA; fail closed rather than guessing elsewhere.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _check_rss(max_rss_bytes: int, stage: str) -> None:
    observed = _rss_bytes()
    if observed > max_rss_bytes:
        raise MemoryError(
            f"peak RSS {observed} bytes exceeded limit {max_rss_bytes} after {stage}"
        )


def _program_genes(archive: np.lib.npyio.NpzFile, broad_genes: np.ndarray) -> tuple[str, ...]:
    fixed = _strings(archive, "gene_ids")
    membership = np.asarray(archive["program_gene_membership"], dtype=bool)
    if membership.ndim != 2 or membership.shape[1] != len(fixed):
        raise ValueError("program_gene_membership does not align with gene_ids")
    genes = tuple(sorted(set(fixed[np.any(membership, axis=0)].tolist())))
    missing = sorted(set(genes) - set(broad_genes.tolist()))
    if missing:
        raise ValueError(f"prespecified program genes are absent from broad genes: {missing[:5]}")
    return genes


def build_moment_bundle(
    source: Path,
    *,
    max_rss_bytes: int,
) -> tuple[PanelMomentBundle, tuple[str, ...], dict[str, object]]:
    """Read each large source CSR once and return compact sufficient statistics."""

    with np.load(source, allow_pickle=False) as archive:
        schema = str(np.asarray(archive["schema_version"]).reshape(-1)[0])
        if schema != SOURCE_SCHEMA:
            raise ValueError(f"source schema is invalid: {schema}")
        receipt = _json_scalar(archive, "source_receipt_json")
        if receipt.get("schema") != "heir.natcommun_regional_source_receipt.v2":
            raise ValueError("source receipt schema is invalid")
        roles = receipt.get("encoder_roles")
        if (
            not isinstance(roles, Mapping)
            or not isinstance(roles.get("primary"), Mapping)
            or roles["primary"].get("repository") != EXPECTED_HOPTIMUS_REPOSITORY
            or roles["primary"].get("revision") != EXPECTED_HOPTIMUS_REVISION
        ):
            raise ValueError("source does not identify the frozen H-optimus-1 primary encoder")
        broad_genes = _strings(archive, "broad_gene_ids")
        if len(set(broad_genes.tolist())) != len(broad_genes) or len(broad_genes) < PANEL_SIZE:
            raise ValueError("source broad gene universe is malformed")
        spot_ids = _strings(archive, "spot_ids")
        st_rows = len(spot_ids)
        st_donors = _strings(archive, "donor_ids", st_rows)
        st_primary = np.asarray(archive["spot_primary_eligible"], dtype=bool)
        st_library = np.asarray(archive["st_total_umi_counts_full"], dtype=np.float64)
        if st_primary.shape != (st_rows,) or st_library.shape != (st_rows,):
            raise ValueError("source ST eligibility or library size is not row aligned")

        sc_cells = _strings(archive, "sc_cell_ids")
        sc_rows = len(sc_cells)
        sc_donors = _strings(archive, "sc_donor_ids", sc_rows)
        type_key = next(
            (
                name
                for name in ("sc_level1_type_ids", "sc_level2_type_ids", "sc_level3_type_ids")
                if name in archive.files
            ),
            None,
        )
        if type_key is None:
            raise ValueError("source lacks a scRNA type vector")
        sc_types = _strings(archive, type_key, sc_rows)
        if np.any(np.char.find(sc_types, "|") >= 0) or np.any(np.char.find(sc_donors, "|") >= 0):
            raise ValueError("scRNA donor/type IDs cannot contain the group delimiter")
        sc_groups = np.char.add(np.char.add(sc_donors, "|"), sc_types)
        sc_primary = np.asarray(archive["sc_primary_eligible"], dtype=bool)
        sc_library = np.asarray(archive["sc_total_umi_counts"], dtype=np.float64)
        if sc_primary.shape != (sc_rows,) or sc_library.shape != (sc_rows,):
            raise ValueError("source scRNA eligibility or library size is not row aligned")
        program_genes = _program_genes(archive, broad_genes)

        moments = {}
        for prefix, destination in (
            ("st_broad_counts_full", "full"),
            ("st_broad_counts_half_a", "half_a"),
            ("st_broad_counts_half_b", "half_b"),
        ):
            matrix = _load_csr(archive, prefix, st_rows, len(broad_genes))
            moments[destination] = normalized_group_moments(
                matrix,
                st_library,
                st_donors,
                row_mask=st_primary,
            )
            del matrix
            gc.collect()
            _check_rss(max_rss_bytes, prefix)
        matrix = _load_csr(archive, "sc_broad_counts", sc_rows, len(broad_genes))
        sc_moments = normalized_group_moments(
            matrix,
            sc_library,
            sc_groups,
            row_mask=sc_primary,
        )
        del matrix
        gc.collect()
        _check_rss(max_rss_bytes, "sc_broad_counts")

    bundle = PanelMomentBundle(
        tuple(broad_genes.tolist()),
        moments["full"],
        moments["half_a"],
        moments["half_b"],
        sc_moments,
    ).validate()
    metadata: dict[str, object] = {
        "st_rows": st_rows,
        "sc_rows": sc_rows,
        "broad_gene_count": len(broad_genes),
        "primary_st_donors": sorted(set(st_donors[st_primary].tolist())),
        "primary_sc_donors": sorted(set(sc_donors[sc_primary].tolist())),
        "zero_full_library_ST_rows_excluded": int(np.count_nonzero(st_primary & (st_library <= 0))),
        "source_receipt_sha256": canonical_sha256(receipt),
    }
    return bundle, program_genes, metadata


def _panel_set(
    source: Path,
    source_hash: str,
    bundle: PanelMomentBundle,
    program_genes: tuple[str, ...],
    *,
    mode: str,
    metadata: Mapping[str, object],
) -> tuple[dict[str, object], object]:
    donors = tuple(bundle.st_full.group_ids)
    external = None
    external_selection = None
    if mode in {"external", "both"}:
        external_selection = select_gene_panel(
            bundle,
            training_donor_ids=donors,
            program_genes=program_genes,
            panel_size=PANEL_SIZE,
            minimum_split_reliability=MINIMUM_SPLIT_RELIABILITY,
        )
        external = panel_artifact(
            external_selection,
            source_sha256=source_hash,
            source_path=str(source),
            mode="external_frozen",
            program_gene_source="source.gene_ids_plus_program_gene_membership",
        )
    folds: dict[str, object] = {}
    if mode in {"lodo", "both"}:
        for held_out in donors:
            training = tuple(donor for donor in donors if donor != held_out)
            selection = select_gene_panel(
                bundle,
                training_donor_ids=training,
                program_genes=program_genes,
                panel_size=PANEL_SIZE,
                held_out_donor_id=held_out,
                minimum_split_reliability=MINIMUM_SPLIT_RELIABILITY,
            )
            folds[held_out] = panel_artifact(
                selection,
                source_sha256=source_hash,
                source_path=str(source),
                mode="lodo_fold_local",
                program_gene_source="source.gene_ids_plus_program_gene_membership",
            )
    if mode == "external":
        assert external is not None
        return external, external_selection
    payload: dict[str, object] = {
        "schema": PANEL_SET_SCHEMA,
        "analysis_status": "exposed_development_only_non_confirmatory",
        "scope": "exposed_development_only_non_confirmatory",
        "source": {"path": str(source), "sha256": source_hash},
        "panel_size": PANEL_SIZE,
        "external_frozen_panel": external,
        "lodo_fold_local_panels": folds,
        "source_summary": dict(metadata),
        "scientific_use": {
            "external_frozen_panel": (
                "freeze_before_any_future_independent_external_validation"
                if external is not None
                else "not_generated"
            ),
            "fold_local_panels": "NatCommun_LODO_development_sensitivity_only",
            "confirmation_authorized": False,
        },
    }
    if external_selection is not None:
        payload["gene_ids"] = list(external_selection.gene_ids)
        payload["broad_column_indices"] = list(external_selection.broad_column_indices)
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload, external_selection


def _project_external_counts(
    source: Path,
    output: Path,
    *,
    selection: object,
    source_hash: str,
    max_output_bytes: int,
    max_rss_bytes: int,
) -> None:
    if selection is None:
        raise ValueError("--projected-output requires an external panel")
    columns = np.asarray(selection.broad_column_indices, dtype=np.int64)
    estimated = 0
    projected: dict[str, np.ndarray] = {}
    with np.load(source, allow_pickle=False) as archive:
        spot_ids = _strings(archive, "spot_ids")
        st_primary = np.asarray(archive["spot_primary_eligible"], dtype=bool)
        sc_cells = _strings(archive, "sc_cell_ids")
        sc_primary = np.asarray(archive["sc_primary_eligible"], dtype=bool)
        broad = _strings(archive, "broad_gene_ids")
        projected_shapes = (
            (int(np.count_nonzero(st_primary)), 3),
            (int(np.count_nonzero(sc_primary)), 1),
        )
        for rows, repetitions in projected_shapes:
            estimated += rows * PANEL_SIZE * np.dtype(np.int32).itemsize * repetitions
        if estimated > max_output_bytes:
            raise MemoryError(
                f"projected count matrices require {estimated} bytes, above limit "
                f"{max_output_bytes}"
            )
        for prefix, destination in (
            ("st_broad_counts_full", "st_counts_full"),
            ("st_broad_counts_half_a", "st_counts_half_a"),
            ("st_broad_counts_half_b", "st_counts_half_b"),
        ):
            matrix = _load_csr(archive, prefix, len(spot_ids), len(broad))
            projected[destination] = project_csr_columns(
                matrix,
                columns,
                row_mask=st_primary,
                max_output_bytes=max_output_bytes,
            )
            del matrix
            gc.collect()
            _check_rss(max_rss_bytes, f"projection {prefix}")
        if not np.array_equal(
            projected["st_counts_full"].astype(np.int64),
            projected["st_counts_half_a"].astype(np.int64)
            + projected["st_counts_half_b"].astype(np.int64),
        ):
            raise ValueError("projected ST halves do not reconstruct full counts")
        st_total_full = np.asarray(archive["st_total_umi_counts_full"])
        st_total_half_a = np.asarray(archive["st_total_umi_counts_half_a"])
        st_total_half_b = np.asarray(archive["st_total_umi_counts_half_b"])
        if any(
            value.shape != (len(spot_ids),)
            for value in (st_total_full, st_total_half_a, st_total_half_b)
        ):
            raise ValueError("ST full and half library totals are not spot aligned")
        if not np.array_equal(
            st_total_full.astype(np.int64),
            st_total_half_a.astype(np.int64) + st_total_half_b.astype(np.int64),
        ):
            raise ValueError("ST half library totals do not reconstruct full totals")
        matrix = _load_csr(archive, "sc_broad_counts", len(sc_cells), len(broad))
        projected["sc_counts"] = project_csr_columns(
            matrix,
            columns,
            row_mask=sc_primary,
            max_output_bytes=max_output_bytes,
        )
        del matrix
        gc.collect()
        _check_rss(max_rss_bytes, "projection sc_broad_counts")

        type_key = next(
            name
            for name in ("sc_level1_type_ids", "sc_level2_type_ids", "sc_level3_type_ids")
            if name in archive.files
        )
        image_features = np.asarray(archive["image_features"])
        coordinates = np.asarray(archive["coordinate_features"])
        blank_image = np.asarray(archive["blank_image_feature_vector"])
        source_program_genes = _strings(archive, "gene_ids")
        program_names = _strings(archive, "program_names")
        source_program_membership = np.asarray(archive["program_gene_membership"], dtype=bool)
        if source_program_membership.shape != (len(program_names), len(source_program_genes)):
            raise ValueError("source program membership is malformed")
        source_program_index = {
            gene: index for index, gene in enumerate(source_program_genes.tolist())
        }
        selected_program_membership = np.zeros(
            (len(program_names), len(selection.gene_ids)), dtype=bool
        )
        for selected_index, gene in enumerate(selection.gene_ids):
            source_index = source_program_index.get(gene)
            if source_index is not None:
                selected_program_membership[:, selected_index] = source_program_membership[
                    :, source_index
                ]
        if (
            image_features.ndim != 2
            or image_features.shape[0] != len(spot_ids)
            or coordinates.ndim != 2
            or coordinates.shape[0] != len(spot_ids)
            or blank_image.shape != (image_features.shape[1],)
            or not np.isfinite(image_features).all()
            or not np.isfinite(coordinates).all()
            or not np.isfinite(blank_image).all()
        ):
            raise ValueError("H&E, coordinate, or blank-image features are malformed")
        sc_total = np.asarray(archive["sc_total_umi_counts"])
        if sc_total.shape != (len(sc_cells),) or np.any(sc_total < 0):
            raise ValueError("scRNA library totals are not cell aligned")
        payload: dict[str, object] = {
            **projected,
            "gene_ids": np.asarray(selection.gene_ids),
            "broad_column_indices": columns,
            "spot_ids": np.asarray(archive["spot_ids"])[st_primary],
            "donor_ids": np.asarray(archive["donor_ids"])[st_primary],
            "section_ids": np.asarray(archive["section_ids"])[st_primary],
            "indication_ids": np.asarray(archive["indication_ids"])[st_primary],
            "image_features": image_features[st_primary],
            "coordinate_features": coordinates[st_primary],
            "blank_image_feature_vector": blank_image,
            "program_names": program_names,
            "program_gene_membership": selected_program_membership,
            "st_total_umi_counts_full": st_total_full[st_primary],
            "st_total_umi_counts_half_a": st_total_half_a[st_primary],
            "st_total_umi_counts_half_b": st_total_half_b[st_primary],
            "sc_cell_ids": np.asarray(archive["sc_cell_ids"])[sc_primary],
            "sc_donor_ids": np.asarray(archive["sc_donor_ids"])[sc_primary],
            "sc_indication_ids": np.asarray(archive["sc_indication_ids"])[sc_primary],
            "sc_level1_type_ids": np.asarray(archive[type_key])[sc_primary],
            "sc_total_umi_counts": sc_total[sc_primary],
            "metadata_json": np.asarray(
                json.dumps(
                    {
                        "schema": PROJECTED_SCHEMA,
                        "analysis_status": "exposed_development_only_non_confirmatory",
                        "source_sha256": source_hash,
                        "panel_identity_sha256": canonical_sha256(
                            {
                                "gene_ids": list(selection.gene_ids),
                                "broad_column_indices": list(selection.broad_column_indices),
                                "training_donor_ids": list(selection.training_donor_ids),
                                "held_out_donor_id": None,
                            }
                        ),
                        "global_broad_matrix_densified": False,
                        "zero_full_library_policy": (
                            "retained_in_artifact_excluded_before_model_fit"
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        }
        # Unlike the early count-only guard, this bound includes image features,
        # the blank control, coordinates, identities, totals, and metadata.
        total_payload_bytes = sum(np.asarray(value).nbytes for value in payload.values())
        if total_payload_bytes > max_output_bytes:
            raise MemoryError(
                f"complete projected artifact requires {total_payload_bytes} uncompressed bytes, "
                f"above limit {max_output_bytes}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name, suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("external", "lodo", "both"), default="external")
    parser.add_argument("--projected-output", type=Path)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument("--max-rss-gib", type=float, default=6.0)
    parser.add_argument("--max-projection-gib", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.threads < 1 or args.threads > 4:
        raise ValueError("--threads must remain between 1 and 4")
    if args.max_rss_gib <= 0 or args.max_projection_gib <= 0:
        raise ValueError("memory limits must be positive")
    source_hash = _sha256(args.source)
    if source_hash != args.expected_source_sha256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {args.expected_source_sha256}, got {source_hash}"
        )
    max_rss_bytes = int(args.max_rss_gib * 1024**3)
    bundle, programs, metadata = build_moment_bundle(
        args.source,
        max_rss_bytes=max_rss_bytes,
    )
    artifact, external_selection = _panel_set(
        args.source,
        source_hash,
        bundle,
        programs,
        mode=args.mode,
        metadata=metadata,
    )
    if artifact.get("schema") == "heir.natcommun_generative_gene_panel.v1":
        validate_panel_artifact(artifact)
    _atomic_json(args.output, artifact)
    if args.projected_output is not None:
        _project_external_counts(
            args.source,
            args.projected_output,
            selection=external_selection,
            source_hash=source_hash,
            max_output_bytes=int(args.max_projection_gib * 1024**3),
            max_rss_bytes=max_rss_bytes,
        )
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
