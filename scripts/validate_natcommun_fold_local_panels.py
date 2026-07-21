#!/usr/bin/env python3
"""Run the frozen NatCommun model with donor-local, training-only gene panels.

The sensitivity is staged deliberately.  ``prepare`` is the only stage that
opens the registered broad source.  It projects the 358-gene union once, then
materializes one exact 256-gene public fit bundle and a separate score target
for each held-out donor.  Fit and score authorities are written to physically
separate manifests.  ``fit-predict`` calls the frozen model runner exactly once
for each complete donor fold and cannot discover either the union projection
or a score target.  ``score`` validates every prediction before opening the
score-only manifest, then repeats validation immediately before each target.

This is an outcome-exposed gene-selection sensitivity, not confirmation.  It
does not change the H-optimus-1 encoder, architecture, 20-D latent, loss,
training schedule, fusion rule, or thresholds.  UNI2-h is prohibited.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "4"

# isort: off
import numpy as np  # noqa: E402

from heir.evaluation.gene_panel import (  # noqa: E402
    canonical_sha256,
    validate_panel_artifact,
)
# isort: on


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/benchmark_natcommun_generative_development.py"
CORE_PATH = ROOT / "src/heir/evaluation/generative_fusion.py"
VALIDATION_PROTOCOL_PATH = ROOT / "configs/natcommun_frozen_validation_protocol.json"
DEVELOPMENT_PROTOCOL_PATH = ROOT / "configs/natcommun_generative_development_protocol.json"
DEFAULT_SOURCE = Path("/mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz")
DEFAULT_RECEIPTS = Path(
    "/mnt/seagate/HEIR_runs/natcommun_generative_development/"
    "gene_panel_external_and_lodo_receipts.json"
)
DEFAULT_BASELINE_REPORT = Path(
    "/mnt/seagate/HEIR_runs/natcommun_generative_development/report.json"
)
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/natcommun_fold_local_panel_sensitivity")

PREPARED_SCHEMA = "heir.natcommun_fold_local_panel_fit_prepared.v2"
PREPARE_RECEIPT_SCHEMA = "heir.natcommun_fold_local_panel_fit_prepare_receipt.v1"
SCORE_TARGET_MANIFEST_SCHEMA = "heir.natcommun_fold_local_panel_score_targets.v1"
SCORE_TARGET_RECEIPT_SCHEMA = "heir.natcommun_fold_local_panel_score_target_receipt.v1"
PREDICTION_MANIFEST_SCHEMA = "heir.natcommun_fold_local_panel_predictions.v1"
PREDICTION_RECEIPT_SCHEMA = "heir.natcommun_fold_local_panel_prediction_receipt.v1"
UNION_SCHEMA = "heir.natcommun_fold_local_panel_union_projection.v1"
UNION_RECEIPT_SCHEMA = "heir.natcommun_fold_local_panel_union_receipt.v1"
REPORT_SCHEMA = "heir.natcommun_fold_local_panel_sensitivity.v1"

EXPECTED_DONORS = (
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
EXPECTED = {
    "source_sha256": "ec37d5717a9b737dfac226ae9267258fb728ee024496a7655bb69a913aa3cf20",
    "development_protocol_sha256": (
        "2cb92b22b6870488a06e64b213e37ffbbdfe3044f1da8fc7442f506915e78197"
    ),
    "runner_sha256": "cf27504e25dfd8cd7e8bfe2894efc8b4a8f79306b47bc492d0e61406d20668ce",
    "core_sha256": "55a63f1360e8cc76267e4b00ba8e2167f36259789e9bfdf2aa929c8cadd83b17",
    "receipts_file_sha256": ("7dcd5bbd6fe6f1a18625afd1fddd36bbd1a9ccad284612cab5248cd21a09f982"),
    "receipts_identity_sha256": (
        "3faa42ff97906e2f3040de6a3b74e0a0942e85ac8bf98fec151a6ae5398406ff"
    ),
    "baseline_report_sha256": ("bf3144cf22405752488509dbb1a65b573967fe1b14110881020787187828cf29"),
}
ARMS = ("M0", "M1", "M2", "M3")
ORIGINAL_EXTERNAL_PANEL_RELATIVE_GAIN = 0.08109603393465813
INACTIVE_PROGRAM_THRESHOLD_POLICY = "inactive_zero_gene_program_not_scored_zero_sentinel"
INACTIVE_PROGRAM_THRESHOLD_SENTINEL = np.float32(0.0)


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


def _load_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _canonical_path(value: object, expected: Path, *, label: str) -> Path:
    """Require an artifact authority to be the exact canonical registered path."""

    canonical = expected.resolve()
    if str(value) != str(canonical):
        raise ValueError(f"{label} is not the canonical registered path")
    return canonical


def _assert_fit_manifest_target_free(manifest: Mapping[str, object]) -> None:
    """Fail closed if source, union, or score-target authority enters fit state."""

    forbidden = {
        "source_path",
        "source_sha256",
        "union_path",
        "union_projection",
        "score_target_path",
        "score_target_file_sha256",
        "score_target_semantic_sha256",
        "score_target_manifest_path",
    }

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            names = {str(name) for name in value}
            overlap = forbidden & names
            overlap.update(
                name
                for name in names
                if (
                    (name.startswith("source_") and name != "contains_source_authority")
                    or (name.startswith("union_") and name != "contains_union_authority")
                    or (
                        name.startswith("score_target_")
                        and name != "contains_score_target_authority"
                    )
                )
            )
            if overlap:
                raise ValueError(
                    f"fit-prepared manifest exposes prohibited authority: {sorted(overlap)}"
                )
            for item in value.values():
                walk(item)
        elif isinstance(value, str) and any(
            token in value
            for token in (
                "source.npz",
                "score_target.npz",
                "lodo_union_projected_counts.npz",
            )
        ):
            raise ValueError("fit-prepared manifest contains a prohibited artifact path")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                walk(item)

    walk(manifest)


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("heir_frozen_natcommun_fold_panel", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen NatCommun runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_array_fields(runner: Any) -> tuple[str, ...]:
    return tuple(
        field.name
        for field in dataclasses.fields(runner.SourceArrays)
        if field.name != "source_receipt"
    )


def _validate_panel_set(
    payload: Mapping[str, object],
    *,
    expected_source_sha256: str,
    expected_donors: Sequence[str],
    expected_panel_size: int,
    expected_union_size: int | None,
) -> tuple[Mapping[str, Mapping[str, object]], tuple[str, ...], tuple[int, ...]]:
    """Validate all LODO receipts and return a stable union projection contract."""

    if payload.get("schema") != "heir.natcommun_generative_gene_panel_set.v1":
        raise ValueError("LODO panel-set schema is invalid")
    if payload.get("analysis_status") != "exposed_development_only_non_confirmatory":
        raise ValueError("LODO panels must disclose outcome-exposed development status")
    source = payload.get("source")
    if not isinstance(source, Mapping) or source.get("sha256") != expected_source_sha256:
        raise ValueError("LODO panels are not bound to the registered source")
    top = dict(payload)
    reported_identity = top.pop("artifact_sha256", None)
    if reported_identity != canonical_sha256(top):
        raise ValueError("LODO panel-set semantic identity is inconsistent")

    raw_panels = payload.get("lodo_fold_local_panels")
    donors = tuple(sorted(str(donor) for donor in expected_donors))
    if not isinstance(raw_panels, Mapping) or tuple(sorted(map(str, raw_panels))) != donors:
        raise ValueError("LODO panel receipts do not exactly cover the registered donors")

    panels: dict[str, Mapping[str, object]] = {}
    gene_to_column: dict[str, int] = {}
    donor_set = set(donors)
    for donor in donors:
        raw = raw_panels[donor]
        if not isinstance(raw, Mapping):
            raise ValueError(f"LODO panel receipt is malformed for {donor}")
        validate_panel_artifact(raw, expected_size=expected_panel_size)
        selection = raw.get("selection")
        leakage = raw.get("leakage_control")
        if not isinstance(selection, Mapping) or not isinstance(leakage, Mapping):
            raise ValueError(f"LODO panel contract is incomplete for {donor}")
        training = set(map(str, selection.get("training_donor_ids", ())))
        if selection.get("held_out_donor_id") != donor or training != donor_set - {donor}:
            raise ValueError(f"LODO panel for {donor} is not selected from the other donors only")
        if (
            leakage.get("selection_uses_only_training_donor_counts") is not True
            or leakage.get("held_out_ST_or_snRNA_used_for_fold_selection") is not False
        ):
            raise ValueError(f"LODO panel leakage receipt is invalid for {donor}")
        genes = tuple(map(str, raw.get("gene_ids", ())))
        columns = tuple(int(value) for value in raw.get("broad_column_indices", ()))
        for gene, column in zip(genes, columns):
            previous = gene_to_column.setdefault(gene, column)
            if previous != column:
                raise ValueError(f"broad source column differs across panel receipts for {gene}")
        panels[donor] = raw

    union_genes = tuple(sorted(gene_to_column))
    if expected_union_size is not None and len(union_genes) != expected_union_size:
        raise ValueError(
            f"LODO panel union has {len(union_genes)} genes; expected {expected_union_size}"
        )
    union_columns = tuple(gene_to_column[gene] for gene in union_genes)
    if len(set(union_columns)) != len(union_columns):
        raise ValueError("LODO union maps distinct genes to duplicate broad-source columns")
    return panels, union_genes, union_columns


def _validate_contracts(
    args: argparse.Namespace,
    runner: Any,
    *,
    verify_source: bool,
    verify_baseline: bool,
) -> Mapping[str, object]:
    if args.smoke:
        receipts = _load_json(args.panel_receipts)
        donors = tuple(sorted(map(str, receipts["lodo_fold_local_panels"])))
        panels, union_genes, union_columns = _validate_panel_set(
            receipts,
            expected_source_sha256=str(receipts["source"]["sha256"]),
            expected_donors=donors,
            expected_panel_size=int(receipts["panel_size"]),
            expected_union_size=None,
        )
        return {
            "receipts": receipts,
            "panels": panels,
            "donors": donors,
            "union_genes": union_genes,
            "union_columns": union_columns,
            "source_sha256": str(receipts["source"]["sha256"]),
            "runner_sha256": _sha256(RUNNER_PATH),
            "core_sha256": _sha256(CORE_PATH),
            "validation_protocol_sha256": _sha256(args.validation_protocol),
            "development_protocol_sha256": _sha256(args.development_protocol),
            "receipts_file_sha256": _sha256(args.panel_receipts),
            "receipts_identity_sha256": str(receipts["artifact_sha256"]),
            "baseline_report_sha256": None,
        }

    observed_fixed = {
        "runner_sha256": _sha256(RUNNER_PATH),
        "core_sha256": _sha256(CORE_PATH),
        "development_protocol_sha256": _sha256(args.development_protocol),
        "receipts_file_sha256": _sha256(args.panel_receipts),
    }
    mismatched = [name for name, value in observed_fixed.items() if value != EXPECTED[name]]
    if mismatched:
        raise ValueError(f"frozen fold-local sensitivity inputs changed: {mismatched}")
    if verify_source and _sha256(args.source) != EXPECTED["source_sha256"]:
        raise ValueError("registered source changed before union projection")
    if verify_baseline and _sha256(args.baseline_report) != EXPECTED["baseline_report_sha256"]:
        raise ValueError("baseline report changed before sensitivity comparison")

    validation = _load_json(args.validation_protocol)
    development = _load_json(args.development_protocol)
    receipts = _load_json(args.panel_receipts)
    if validation.get("schema") != "heir.natcommun_frozen_validation_protocol.v1":
        raise ValueError("frozen-validation protocol schema is invalid")
    frozen = validation.get("model_freeze")
    artifacts = validation.get("frozen_artifacts")
    panel_plan = validation.get("fold_local_panel_sensitivity")
    resources = validation.get("resource_limits")
    if not all(isinstance(value, Mapping) for value in (frozen, artifacts, panel_plan, resources)):
        raise ValueError("frozen-validation protocol lacks the LODO scientific contract")
    if (
        frozen.get("image_encoder") != "bioptimus/H-optimus-1"
        or frozen.get("gene_count") != 256
        or frozen.get("latent_dimension") != 20
        or frozen.get("epochs") != 80
        or frozen.get("batch_size") != 256
        or frozen.get("base_seed") != 1729
        or frozen.get("iterative_refinement") != "prohibited"
        or frozen.get("UNI2_h") != "prohibited_not_run"
    ):
        raise ValueError("frozen model identity changed for the LODO sensitivity")
    if (
        artifacts.get("source_sha256") != EXPECTED["source_sha256"]
        or artifacts.get("development_protocol_sha256") != EXPECTED["development_protocol_sha256"]
        or artifacts.get("development_runner_sha256") != EXPECTED["runner_sha256"]
        or artifacts.get("generative_core_sha256") != EXPECTED["core_sha256"]
        or artifacts.get("development_report_sha256") != EXPECTED["baseline_report_sha256"]
    ):
        raise ValueError("validation protocol is not bound to the frozen model artifacts")
    if (
        Path(str(panel_plan.get("panel_receipts", ""))).resolve() != args.panel_receipts.resolve()
        or panel_plan.get("architecture_or_hyperparameter_change") is not False
    ):
        raise ValueError("validation protocol LODO panel plan changed")
    if (
        int(resources.get("maximum_CPU_threads", -1)) > 4
        or float(resources.get("maximum_GPU_memory_fraction", 1.0)) > 0.60
        or resources.get("outer_folds_serial") is not True
        or resources.get("swap_permitted") is not False
    ):
        raise ValueError("validation resource limits changed")

    panel_contract = development.get("gene_panel")
    if not isinstance(panel_contract, Mapping):
        raise ValueError("development protocol panel contract is missing")
    if (
        panel_contract.get("external_and_lodo_receipts_sha256") != EXPECTED["receipts_file_sha256"]
        or panel_contract.get("external_and_lodo_receipts_identity_sha256")
        != EXPECTED["receipts_identity_sha256"]
    ):
        raise ValueError("development protocol LODO receipt identity changed")
    if receipts.get("artifact_sha256") != EXPECTED["receipts_identity_sha256"]:
        raise ValueError("LODO receipt semantic identity changed")

    panels, union_genes, union_columns = _validate_panel_set(
        receipts,
        expected_source_sha256=EXPECTED["source_sha256"],
        expected_donors=EXPECTED_DONORS,
        expected_panel_size=256,
        expected_union_size=358,
    )
    return {
        "receipts": receipts,
        "panels": panels,
        "donors": EXPECTED_DONORS,
        "union_genes": union_genes,
        "union_columns": union_columns,
        "source_sha256": EXPECTED["source_sha256"],
        **observed_fixed,
        "validation_protocol_sha256": _sha256(args.validation_protocol),
        "receipts_identity_sha256": str(receipts["artifact_sha256"]),
        "baseline_report_sha256": (EXPECTED["baseline_report_sha256"] if verify_baseline else None),
    }


def _union_metadata(contracts: Mapping[str, object], *, smoke: bool) -> Mapping[str, object]:
    identity = {
        "source_sha256": contracts["source_sha256"],
        "panel_receipts_identity_sha256": contracts["receipts_identity_sha256"],
        "union_gene_ids": list(contracts["union_genes"]),
        "union_broad_column_indices": list(contracts["union_columns"]),
    }
    return {
        "schema": UNION_SCHEMA,
        "analysis_status": "outcome_exposed_gene_panel_sensitivity_non_confirmatory",
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "image_encoder": "bioptimus/H-optimus-1",
        "uni2_h_run": False,
        "smoke": bool(smoke),
        "contains_sealed_targets": True,
        "permitted_open_stage": "prepare_only",
    }


def _union_arrays(runner: Any, source: Any, metadata: Mapping[str, object]) -> Mapping[str, object]:
    arrays = {name: np.asarray(getattr(source, name)) for name in _source_array_fields(runner)}
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    arrays["source_receipt_json"] = np.asarray(
        json.dumps(source.source_receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    return arrays


def _source_from_union(
    runner: Any,
    arrays: Mapping[str, np.ndarray],
    expected_metadata: Mapping[str, object],
) -> Any:
    metadata = json.loads(runner._scalar_text(arrays["metadata_json"]))
    if metadata != expected_metadata:
        raise ValueError("union projection metadata differs from the sealed LODO identity")
    receipt = json.loads(runner._scalar_text(arrays["source_receipt_json"]))
    source = runner.SourceArrays(
        **{name: np.asarray(arrays[name]) for name in _source_array_fields(runner)},
        source_receipt=receipt,
    )
    genes = np.asarray(source.gene_ids).astype(str)
    expected_genes = list(expected_metadata["identity"]["union_gene_ids"])
    if genes.tolist() != expected_genes:
        raise ValueError("union projection gene order differs from its identity")
    if (
        source.st_counts.shape != (len(source.spot_ids), len(genes))
        or source.st_half_a.shape != source.st_counts.shape
        or source.st_half_b.shape != source.st_counts.shape
        or source.sc_counts.shape != (len(source.sc_cell_ids), len(genes))
        or source.program_gene_membership.shape[1] != len(genes)
        or not np.array_equal(source.st_counts, source.st_half_a + source.st_half_b)
    ):
        raise ValueError("union projection arrays are malformed")
    return source


def _load_or_create_union(
    args: argparse.Namespace,
    runner: Any,
    contracts: Mapping[str, object],
) -> tuple[Any, Mapping[str, object]]:
    union_path = args.output / "lodo_union_projected_counts.npz"
    receipt_path = args.output / "lodo_union_projection_receipt.json"
    metadata = _union_metadata(contracts, smoke=bool(args.smoke))
    if union_path.exists() != receipt_path.exists():
        raise ValueError("incomplete union projection; use a fresh output directory")
    if union_path.is_file():
        receipt = _load_json(receipt_path)
        if (
            receipt.get("schema") != UNION_RECEIPT_SCHEMA
            or receipt.get("union_path") != str(union_path.resolve())
            or receipt.get("union_file_sha256") != _sha256(union_path)
            or receipt.get("union_identity_sha256") != metadata["identity_sha256"]
            or receipt.get("orchestrator_sha256") != _sha256(Path(__file__).resolve())
        ):
            raise ValueError("existing union projection is stale; use a fresh output directory")
        arrays = runner._load_arrays(union_path)
        if receipt.get("union_semantic_sha256") != runner._semantic_array_hash(arrays):
            raise ValueError("existing union projection semantic hash changed")
        return _source_from_union(runner, arrays, metadata), receipt

    source = runner.load_selected_source(
        args.source,
        contracts["union_genes"],
        expected_sha256=(None if args.smoke else str(contracts["source_sha256"])),
        smoke=args.smoke,
    )
    if np.asarray(source.gene_ids).astype(str).tolist() != list(contracts["union_genes"]):
        raise ValueError("broad-source union projection changed gene order")
    arrays = _union_arrays(runner, source, metadata)
    runner._atomic_npz(union_path, arrays)
    receipt = {
        "schema": UNION_RECEIPT_SCHEMA,
        "union_path": str(union_path.resolve()),
        "union_file_sha256": _sha256(union_path),
        "union_semantic_sha256": runner._semantic_array_hash(arrays),
        "union_identity_sha256": metadata["identity_sha256"],
        "source_sha256": contracts["source_sha256"],
        "panel_receipts_identity_sha256": contracts["receipts_identity_sha256"],
        "gene_count": len(contracts["union_genes"]),
        "projected_once": True,
        "smoke": bool(args.smoke),
        "permitted_open_stage": "prepare_only",
        "orchestrator_sha256": _sha256(Path(__file__).resolve()),
    }
    _atomic_json(receipt_path, receipt)
    return source, receipt


def _slice_source(runner: Any, source: Any, genes: Sequence[str]) -> Any:
    union = np.asarray(source.gene_ids).astype(str)
    lookup = {gene: index for index, gene in enumerate(union.tolist())}
    missing = [gene for gene in genes if gene not in lookup]
    if missing:
        raise ValueError(f"fold panel genes are absent from the union: {missing[:5]}")
    columns = np.asarray([lookup[str(gene)] for gene in genes], dtype=np.int64)
    values = {name: getattr(source, name) for name in _source_array_fields(runner)}
    for name in ("st_counts", "st_half_a", "st_half_b", "sc_counts"):
        values[name] = np.asarray(values[name])[:, columns]
    values["gene_ids"] = np.asarray(genes)
    values["program_gene_membership"] = np.asarray(source.program_gene_membership)[:, columns]
    return runner.SourceArrays(**values, source_receipt=source.source_receipt)


def _filter_positive_training_rows(public: dict[str, np.ndarray]) -> None:
    positive = np.asarray(public["train_st_library"], dtype=float) > 0
    for name in (
        "train_spot_ids",
        "train_donor_ids",
        "train_section_ids",
        "train_indication_ids",
        "train_image",
        "train_coordinates",
        "train_st_counts",
        "train_st_library",
        "train_st_half_a",
        "train_st_half_b",
        "train_st_library_half_a",
        "train_st_library_half_b",
    ):
        public[name] = np.asarray(public[name])[positive]


def prepare(args: argparse.Namespace, runner: Any) -> Mapping[str, object]:
    manifest_path = args.output / "prepared_manifest.json"
    target_manifest_path = args.output / "score_target_manifest.json"
    if manifest_path.exists() != target_manifest_path.exists():
        raise ValueError("incomplete fit/score preparation; use a fresh output directory")
    if manifest_path.is_file():
        prepared = _read_prepared(args, runner)
        _read_score_target_manifest(args, runner, prepared, allow_smoke=True)
        return prepared
    contracts = _validate_contracts(args, runner, verify_source=True, verify_baseline=False)
    source, union_receipt = _load_or_create_union(args, runner, contracts)
    folds: dict[str, object] = {}
    target_folds: dict[str, object] = {}
    for donor in contracts["donors"]:
        donor = str(donor)
        panel = contracts["panels"][donor]
        genes = tuple(map(str, panel["gene_ids"]))
        fold_source = _slice_source(runner, source, genes)
        seed = runner._fold_seed(args.seed, donor)
        public = runner._fold_public_arrays(fold_source, donor, seed)
        _filter_positive_training_rows(public)
        runner.validate_public_fold(public)
        if np.asarray(public["gene_ids"]).astype(str).tolist() != list(genes):
            raise ValueError(f"public fold panel order changed for {donor}")
        secret = dict(runner._fold_secret_arrays(fold_source, donor))
        secret["gene_ids"] = np.asarray(public["gene_ids"]).copy()
        fold_dir = args.output / "folds" / donor
        public_path = fold_dir / "fit_predict_input.npz"
        secret_path = fold_dir / "score_target.npz"
        prepare_receipt_path = fold_dir / "prepare_receipt.json"
        target_receipt_path = fold_dir / "score_target_receipt.json"
        runner._atomic_npz(public_path, public)
        runner._atomic_npz(secret_path, secret)
        fit_receipt = {
            "schema": PREPARE_RECEIPT_SCHEMA,
            "receipt_path": str(prepare_receipt_path.resolve()),
            "heldout_donor": donor,
            "seed": seed,
            "smoke": bool(args.smoke),
            "panel_artifact_sha256": panel["artifact_sha256"],
            "panel_identity_sha256": panel["selection"]["identity_sha256"],
            "panel_gene_ids": list(genes),
            "panel_gene_count": len(genes),
            "training_panel_donors": list(panel["selection"]["training_donor_ids"]),
            "public_path": str(public_path.resolve()),
            "public_file_sha256": _sha256(public_path),
            "public_semantic_sha256": runner._semantic_array_hash(public),
            "fit_predict_has_heldout_ST": False,
            "train_spots": len(public["train_spot_ids"]),
            "query_spots": len(public["query_spot_ids"]),
        }
        _atomic_json(prepare_receipt_path, fit_receipt)
        folds[donor] = {
            **fit_receipt,
            "prepare_receipt_file_sha256": _sha256(prepare_receipt_path),
        }
        target_receipt = {
            "schema": SCORE_TARGET_RECEIPT_SCHEMA,
            "receipt_path": str(target_receipt_path.resolve()),
            "heldout_donor": donor,
            "seed": seed,
            "smoke": bool(args.smoke),
            "panel_identity_sha256": panel["selection"]["identity_sha256"],
            "public_semantic_sha256": runner._semantic_array_hash(public),
            "gene_ids": list(genes),
            "score_target_path": str(secret_path.resolve()),
            "score_target_file_sha256": _sha256(secret_path),
            "score_target_semantic_sha256": runner._semantic_array_hash(secret),
            "permitted_open_stage": "score_only_after_global_prediction_preflight",
            "zero_depth_score_rows_excluded": int(secret["zero_depth_excluded_count"]),
        }
        _atomic_json(target_receipt_path, target_receipt)
        target_folds[donor] = {
            **target_receipt,
            "score_target_receipt_file_sha256": _sha256(target_receipt_path),
        }

    manifest = {
        "schema": PREPARED_SCHEMA,
        "manifest_path": str(manifest_path.resolve()),
        "analysis_scope": "outcome_exposed_gene_panel_sensitivity_non_confirmatory",
        "scientific_authorization": "none",
        "smoke": bool(args.smoke),
        "image_encoder": "bioptimus/H-optimus-1",
        "image_scale_um": 112,
        "uni2_h_run": False,
        "iterative_refinement_run": False,
        "panel_receipts_file_sha256": contracts["receipts_file_sha256"],
        "panel_receipts_identity_sha256": contracts["receipts_identity_sha256"],
        "validation_protocol_sha256": contracts["validation_protocol_sha256"],
        "development_protocol_sha256": contracts["development_protocol_sha256"],
        "frozen_runner_sha256": contracts["runner_sha256"],
        "frozen_core_sha256": contracts["core_sha256"],
        "orchestrator_sha256": _sha256(Path(__file__).resolve()),
        "panel_union_gene_count": len(contracts["union_genes"]),
        "common_gene_count": len(
            set.intersection(
                *(
                    set(map(str, contracts["panels"][donor]["gene_ids"]))
                    for donor in contracts["donors"]
                )
            )
        ),
        "fold_panel_gene_count": int(args.panel_size),
        "donors": list(contracts["donors"]),
        "leave_one_donor_out": True,
        "folds_serial": True,
        "frozen_fit_calls_required": len(contracts["donors"]),
        "frozen_fit_calls_per_donor": 1,
        "training_configuration": {
            "base_seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "device": args.device,
            "cpu_threads": args.cpu_threads,
            "gpu_memory_fraction": args.gpu_memory_fraction,
        },
        "target_boundary": {
            "contains_source_authority": False,
            "contains_union_authority": False,
            "contains_score_target_authority": False,
            "global_prediction_preflight_before_any_target": True,
            "immediate_prediction_revalidation_before_each_target": True,
        },
        "folds": folds,
    }
    _assert_fit_manifest_target_free(manifest)
    _atomic_json(manifest_path, manifest)
    union_identity = {
        name: union_receipt[name]
        for name in (
            "schema",
            "union_file_sha256",
            "union_semantic_sha256",
            "union_identity_sha256",
            "source_sha256",
            "panel_receipts_identity_sha256",
            "gene_count",
            "projected_once",
            "smoke",
            "orchestrator_sha256",
        )
    }
    target_manifest = {
        "schema": SCORE_TARGET_MANIFEST_SCHEMA,
        "manifest_path": str(target_manifest_path.resolve()),
        "prepared_manifest_path": str(manifest_path.resolve()),
        "prepared_manifest_sha256": _sha256(manifest_path),
        "analysis_scope": "outcome_exposed_gene_panel_sensitivity_non_confirmatory",
        "smoke": bool(args.smoke),
        "scientific_score_permitted": not bool(args.smoke),
        "source_sha256": contracts["source_sha256"],
        "union_projection_identity": union_identity,
        "orchestrator_sha256": _sha256(Path(__file__).resolve()),
        "donors": list(contracts["donors"]),
        "folds": target_folds,
    }
    _atomic_json(target_manifest_path, target_manifest)
    return manifest


def _validate_training_args(args: argparse.Namespace) -> None:
    if not 1 <= args.cpu_threads <= 4:
        raise ValueError("CPU threads must be in [1, 4]")
    if not 0 < args.gpu_memory_fraction <= 0.60:
        raise ValueError("GPU memory fraction must be in (0, 0.60]")
    if not args.smoke:
        observed = (args.seed, args.epochs, args.batch_size, args.latent_dim)
        if observed != (1729, 80, 256, 20):
            raise ValueError("training arguments differ from the frozen model identity")
        if args.panel_size != 256:
            raise ValueError("the fold-local sensitivity requires exact 256-gene panels")
        if not args.device.startswith("cuda"):
            raise ValueError("the real LODO sensitivity requires bounded CUDA execution")
    runner_text = str(args.device).casefold().replace("-", "").replace("_", "")
    if "uni2" in runner_text:
        raise ValueError("UNI2-h is prohibited")


def _read_prepared(args: argparse.Namespace, runner: Any) -> Mapping[str, object]:
    manifest_path = args.output / "prepared_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != PREPARED_SCHEMA:
        raise ValueError("fold-local prepared manifest schema is invalid")
    _canonical_path(manifest.get("manifest_path"), manifest_path, label="fit-prepared manifest")
    _assert_fit_manifest_target_free(manifest)
    if manifest.get("smoke") is not bool(args.smoke):
        raise ValueError("smoke/real preparation identity changed")
    donors = tuple(map(str, manifest.get("donors", ())))
    if (
        not donors
        or len(set(donors)) != len(donors)
        or (not args.smoke and donors != EXPECTED_DONORS)
    ):
        raise ValueError("prepared donor identity changed")
    required = {
        "image_encoder": "bioptimus/H-optimus-1",
        "uni2_h_run": False,
        "iterative_refinement_run": False,
        "validation_protocol_sha256": _sha256(args.validation_protocol),
        "development_protocol_sha256": _sha256(args.development_protocol),
        "frozen_runner_sha256": _sha256(RUNNER_PATH),
        "frozen_core_sha256": _sha256(CORE_PATH),
        "orchestrator_sha256": _sha256(Path(__file__).resolve()),
        "donors": list(donors),
        "frozen_fit_calls_required": len(donors),
        "frozen_fit_calls_per_donor": 1,
        "leave_one_donor_out": True,
        "folds_serial": True,
        "target_boundary": {
            "contains_source_authority": False,
            "contains_union_authority": False,
            "contains_score_target_authority": False,
            "global_prediction_preflight_before_any_target": True,
            "immediate_prediction_revalidation_before_each_target": True,
        },
    }
    if not args.smoke:
        required.update(
            {
                "panel_receipts_file_sha256": EXPECTED["receipts_file_sha256"],
                "panel_receipts_identity_sha256": EXPECTED["receipts_identity_sha256"],
            }
        )
    mismatched = [name for name, expected in required.items() if manifest.get(name) != expected]
    if mismatched:
        raise ValueError(f"fold-local prepared manifest is stale: {mismatched}")
    expected_training = {
        "base_seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "device": args.device,
        "cpu_threads": args.cpu_threads,
        "gpu_memory_fraction": args.gpu_memory_fraction,
    }
    if manifest.get("training_configuration") != expected_training:
        raise ValueError("current arguments differ from the prepared training identity")
    folds = manifest.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != set(donors):
        raise ValueError("prepared folds do not exactly cover the registered donors")
    for donor in donors:
        donor = str(donor)
        fold = folds[donor]
        if not isinstance(fold, Mapping):
            raise ValueError(f"prepared fold receipt is malformed for {donor}")
        public_path = args.output / "folds" / donor / "fit_predict_input.npz"
        receipt_path = args.output / "folds" / donor / "prepare_receipt.json"
        _canonical_path(fold.get("public_path"), public_path, label=f"public fold {donor}")
        _canonical_path(fold.get("receipt_path"), receipt_path, label=f"prepare receipt {donor}")
        genes = tuple(map(str, fold.get("panel_gene_ids", ())))
        training = tuple(map(str, fold.get("training_panel_donors", ())))
        if (
            fold.get("heldout_donor") != donor
            or int(fold.get("seed", -1)) != runner._fold_seed(args.seed, donor)
            or fold.get("smoke") is not bool(args.smoke)
            or len(genes) != args.panel_size
            or len(set(genes)) != len(genes)
            or int(fold.get("panel_gene_count", -1)) != len(genes)
            or set(training) != set(donors) - {donor}
            or not isinstance(fold.get("panel_artifact_sha256"), str)
            or not isinstance(fold.get("panel_identity_sha256"), str)
            or fold.get("fit_predict_has_heldout_ST") is not False
        ):
            raise ValueError(f"prepared fold identity changed for {donor}")
        if _sha256(receipt_path) != fold.get("prepare_receipt_file_sha256"):
            raise ValueError(f"prepare receipt file identity changed for {donor}")
        receipt = _load_json(receipt_path)
        expected_receipt = {
            name: value for name, value in fold.items() if name != "prepare_receipt_file_sha256"
        }
        if receipt != expected_receipt or receipt.get("schema") != PREPARE_RECEIPT_SCHEMA:
            raise ValueError(f"prepare receipt content changed for {donor}")
    return manifest


def _read_score_target_manifest(
    args: argparse.Namespace,
    runner: Any,
    prepared: Mapping[str, object],
    *,
    allow_smoke: bool = False,
) -> Mapping[str, object]:
    """Read score authority only after the target-free prediction preflight."""

    path = args.output / "score_target_manifest.json"
    value = _load_json(path)
    _canonical_path(value.get("manifest_path"), path, label="score-target manifest")
    prepared_path = args.output / "prepared_manifest.json"
    _canonical_path(
        value.get("prepared_manifest_path"),
        prepared_path,
        label="score-target prepared manifest",
    )
    if (
        value.get("schema") != SCORE_TARGET_MANIFEST_SCHEMA
        or value.get("prepared_manifest_sha256") != _sha256(prepared_path)
        or value.get("orchestrator_sha256") != _sha256(Path(__file__).resolve())
        or value.get("smoke") is not bool(prepared["smoke"])
        or value.get("scientific_score_permitted") is not (not bool(prepared["smoke"]))
        or value.get("donors") != prepared["donors"]
    ):
        raise ValueError("score-target manifest identity changed")
    union = value.get("union_projection_identity")
    if (
        not isinstance(union, Mapping)
        or union.get("schema") != UNION_RECEIPT_SCHEMA
        or union.get("smoke") is not bool(prepared["smoke"])
        or union.get("projected_once") is not True
        or union.get("gene_count") != prepared["panel_union_gene_count"]
        or union.get("source_sha256") != value.get("source_sha256")
        or union.get("panel_receipts_identity_sha256") != prepared["panel_receipts_identity_sha256"]
        or union.get("orchestrator_sha256") != _sha256(Path(__file__).resolve())
        or (not bool(prepared["smoke"]) and value.get("source_sha256") != EXPECTED["source_sha256"])
    ):
        raise ValueError("score-only source/union preparation identity changed")
    if bool(prepared["smoke"]) and not allow_smoke:
        raise ValueError("smoke artifacts are prohibited from scientific scoring")
    folds = value.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != set(prepared["donors"]):
        raise ValueError("score-target folds do not exactly cover prepared donors")
    for donor_value in prepared["donors"]:
        donor = str(donor_value)
        fold = folds[donor]
        if not isinstance(fold, Mapping):
            raise ValueError(f"score-target receipt is malformed for {donor}")
        target_path = args.output / "folds" / donor / "score_target.npz"
        receipt_path = args.output / "folds" / donor / "score_target_receipt.json"
        _canonical_path(fold.get("score_target_path"), target_path, label=f"score target {donor}")
        _canonical_path(
            fold.get("receipt_path"), receipt_path, label=f"score-target receipt {donor}"
        )
        prepared_fold = prepared["folds"][donor]
        if (
            fold.get("schema") != SCORE_TARGET_RECEIPT_SCHEMA
            or fold.get("heldout_donor") != donor
            or int(fold.get("seed", -1)) != int(prepared_fold["seed"])
            or fold.get("smoke") is not bool(prepared["smoke"])
            or fold.get("panel_identity_sha256") != prepared_fold["panel_identity_sha256"]
            or fold.get("public_semantic_sha256") != prepared_fold["public_semantic_sha256"]
            or fold.get("gene_ids") != prepared_fold["panel_gene_ids"]
            or fold.get("permitted_open_stage") != "score_only_after_global_prediction_preflight"
        ):
            raise ValueError(f"score-target identity changed for {donor}")
        if _sha256(receipt_path) != fold.get("score_target_receipt_file_sha256"):
            raise ValueError(f"score-target receipt file identity changed for {donor}")
        receipt = _load_json(receipt_path)
        expected_receipt = {
            name: item for name, item in fold.items() if name != "score_target_receipt_file_sha256"
        }
        if receipt != expected_receipt:
            raise ValueError(f"score-target receipt content changed for {donor}")
    return value


def _verify_npz(
    runner: Any,
    path: Path,
    *,
    file_sha256: str,
    semantic_sha256: str,
) -> dict[str, np.ndarray]:
    if _sha256(path) != file_sha256:
        raise ValueError(f"NPZ file hash changed: {path}")
    arrays = runner._load_arrays(path)
    if runner._semantic_array_hash(arrays) != semantic_sha256:
        raise ValueError(f"NPZ semantic hash changed: {path}")
    return arrays


def _prediction_identity(
    manifest: Mapping[str, object],
    fold: Mapping[str, object],
    donor: str,
) -> str:
    return canonical_sha256(
        {
            "prepared_schema": manifest["schema"],
            "prepared_manifest_path": manifest["manifest_path"],
            "smoke": manifest["smoke"],
            "heldout_donor": donor,
            "fold_seed": fold["seed"],
            "public_file_sha256": fold["public_file_sha256"],
            "public_semantic_sha256": fold["public_semantic_sha256"],
            "panel_artifact_sha256": fold["panel_artifact_sha256"],
            "panel_identity_sha256": fold["panel_identity_sha256"],
            "panel_gene_ids": fold["panel_gene_ids"],
            "training_configuration": manifest["training_configuration"],
            "frozen_runner_sha256": manifest["frozen_runner_sha256"],
            "frozen_core_sha256": manifest["frozen_core_sha256"],
            "development_protocol_sha256": manifest["development_protocol_sha256"],
            "validation_protocol_sha256": manifest["validation_protocol_sha256"],
            "orchestrator_sha256": manifest["orchestrator_sha256"],
        }
    )


def _prediction_receipt_required(
    args: argparse.Namespace,
    manifest: Mapping[str, object],
    fold: Mapping[str, object],
    donor: str,
) -> Mapping[str, object]:
    fold_dir = args.output / "folds" / donor
    return {
        "schema": PREDICTION_RECEIPT_SCHEMA,
        "receipt_path": str((fold_dir / "fit_predict_receipt.json").resolve()),
        "heldout_donor": donor,
        "checkpoint_identity": _prediction_identity(manifest, fold, donor),
        "panel_identity_sha256": fold["panel_identity_sha256"],
        "public_semantic_sha256": fold["public_semantic_sha256"],
        "prediction_path": str((fold_dir / "predictions.npz").resolve()),
        "heldout_ST_opened": False,
        "frozen_fit_function": "fit_predict_one_fold",
        "frozen_fit_calls_for_this_donor": 1,
        "image_encoder": "bioptimus/H-optimus-1",
        "uni2_h_run": False,
        "smoke": bool(manifest["smoke"]),
        "device": args.device,
        "cpu_threads": args.cpu_threads,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
    }


def _receipt_matches_required(
    receipt: Mapping[str, object], required: Mapping[str, object]
) -> bool:
    return all(receipt.get(name) == value for name, value in required.items()) and all(
        isinstance(receipt.get(name), str) and len(str(receipt[name])) == 64
        for name in ("prediction_file_sha256", "prediction_semantic_sha256")
    )


def _normalize_inactive_program_thresholds(
    predictions: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Make the frozen diagnostic receipt finite only for programs with zero panel genes.

    The frozen trainer represents an unestimable threshold for an inactive, zero-gene
    program as NaN, while its artifact validator requires every threshold to be finite.
    Downstream quality scoring already excludes inactive programs.  This wrapper therefore
    records a finite sentinel only on those unused entries and leaves every active threshold
    and every model prediction unchanged.
    """

    normalized = dict(predictions)
    membership = np.asarray(predictions["diagnostic_program_membership"], dtype=bool)
    active = np.asarray(predictions["diagnostic_program_active"], dtype=bool)
    thresholds = np.asarray(predictions["diagnostic_rare_program_thresholds"])
    programs = membership.shape[0] if membership.ndim == 2 else 0
    if (
        programs < 1
        or active.shape != (programs,)
        or thresholds.shape != (programs,)
        or not np.array_equal(active, membership.any(axis=1))
        or not np.isfinite(thresholds[active]).all()
    ):
        raise ValueError("fold-local diagnostic program receipt is malformed")
    inactive = ~active
    repaired = thresholds.copy()
    repaired[inactive] = INACTIVE_PROGRAM_THRESHOLD_SENTINEL
    normalized["diagnostic_rare_program_thresholds"] = repaired
    normalized["fold_local_inactive_program_threshold_policy"] = np.asarray(
        INACTIVE_PROGRAM_THRESHOLD_POLICY
    )
    normalized["fold_local_inactive_program_threshold_sentinel"] = np.asarray(
        INACTIVE_PROGRAM_THRESHOLD_SENTINEL, dtype=np.float32
    )
    normalized["fold_local_inactive_program_threshold_count"] = np.asarray(
        int(inactive.sum()), dtype=np.int64
    )
    return normalized


def _validate_fold_local_prediction(
    runner: Any,
    predictions: Mapping[str, np.ndarray],
    public: Mapping[str, np.ndarray],
    *,
    donor: str,
    epochs: int,
) -> None:
    membership = np.asarray(predictions["diagnostic_program_membership"], dtype=bool)
    active = np.asarray(predictions["diagnostic_program_active"], dtype=bool)
    thresholds = np.asarray(predictions["diagnostic_rare_program_thresholds"])
    programs = membership.shape[0] if membership.ndim == 2 else 0
    policy = np.asarray(predictions.get("fold_local_inactive_program_threshold_policy"))
    sentinel = np.asarray(predictions.get("fold_local_inactive_program_threshold_sentinel"))
    count = np.asarray(predictions.get("fold_local_inactive_program_threshold_count"))
    if (
        programs < 1
        or active.shape != (programs,)
        or thresholds.shape != (programs,)
        or not np.array_equal(active, membership.any(axis=1))
        or policy.shape != ()
        or str(policy.item()) != INACTIVE_PROGRAM_THRESHOLD_POLICY
        or sentinel.shape != ()
        or not np.isclose(float(sentinel), float(INACTIVE_PROGRAM_THRESHOLD_SENTINEL))
        or count.shape != ()
        or int(count) != int((~active).sum())
        or not np.isfinite(thresholds[active]).all()
        or not np.array_equal(
            thresholds[~active],
            np.full(int((~active).sum()), INACTIVE_PROGRAM_THRESHOLD_SENTINEL),
        )
    ):
        raise ValueError(f"fold-local inactive-program threshold receipt differs for {donor}")
    runner._validate_prediction_artifact(predictions, public, donor=donor, epochs=epochs)


def fit_predict(args: argparse.Namespace, runner: Any) -> Mapping[str, object]:
    manifest = _read_prepared(args, runner)
    receipts: dict[str, object] = {}
    new_calls = 0
    for donor in manifest["donors"]:
        donor = str(donor)
        fold = manifest["folds"][donor]
        public = _verify_npz(
            runner,
            Path(str(fold["public_path"])),
            file_sha256=str(fold["public_file_sha256"]),
            semantic_sha256=str(fold["public_semantic_sha256"]),
        )
        runner.validate_public_fold(public)
        identity = _prediction_identity(manifest, fold, donor)
        fold_dir = args.output / "folds" / donor
        prediction_path = fold_dir / "predictions.npz"
        receipt_path = fold_dir / "fit_predict_receipt.json"
        required_receipt = _prediction_receipt_required(args, manifest, fold, donor)
        if required_receipt["checkpoint_identity"] != identity:
            raise RuntimeError("prediction receipt/checkpoint construction is inconsistent")
        if args.resume and prediction_path.is_file() and receipt_path.is_file():
            old = _load_json(receipt_path)
            if _receipt_matches_required(old, required_receipt):
                predictions = _verify_npz(
                    runner,
                    prediction_path,
                    file_sha256=str(old["prediction_file_sha256"]),
                    semantic_sha256=str(old["prediction_semantic_sha256"]),
                )
                _validate_fold_local_prediction(
                    runner, predictions, public, donor=donor, epochs=args.epochs
                )
                receipts[donor] = {
                    **old,
                    "receipt_file_sha256": _sha256(receipt_path),
                }
                continue
        predictions = runner.fit_predict_one_fold(
            public,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            seed=int(fold["seed"]),
        )
        predictions = _normalize_inactive_program_thresholds(predictions)
        new_calls += 1
        _validate_fold_local_prediction(
            runner, predictions, public, donor=donor, epochs=args.epochs
        )
        runner._atomic_npz(prediction_path, predictions)
        receipt = {
            **required_receipt,
            "prediction_file_sha256": _sha256(prediction_path),
            "prediction_semantic_sha256": runner._semantic_array_hash(predictions),
        }
        _atomic_json(receipt_path, receipt)
        receipts[donor] = {
            **receipt,
            "receipt_file_sha256": _sha256(receipt_path),
        }
        if args.device.startswith("cuda"):
            runner.torch.cuda.empty_cache()
    aggregate = {
        "schema": PREDICTION_MANIFEST_SCHEMA,
        "manifest_path": str((args.output / "fit_predict_manifest.json").resolve()),
        "prepared_manifest_path": str((args.output / "prepared_manifest.json").resolve()),
        "analysis_scope": "outcome_exposed_gene_panel_sensitivity_non_confirmatory",
        "smoke": bool(manifest["smoke"]),
        "orchestrator_sha256": _sha256(Path(__file__).resolve()),
        "all_folds_complete": len(receipts) == len(manifest["donors"]),
        "required_frozen_fit_calls_total": len(manifest["donors"]),
        "frozen_fit_calls_per_donor": 1,
        "new_frozen_fit_calls_this_invocation": new_calls,
        "heldout_ST_opened": False,
        "prepared_manifest_sha256": _sha256(args.output / "prepared_manifest.json"),
        "folds": receipts,
    }
    _atomic_json(args.output / "fit_predict_manifest.json", aggregate)
    return aggregate


def _read_prediction_manifest(
    args: argparse.Namespace, prepared: Mapping[str, object]
) -> Mapping[str, object]:
    path = args.output / "fit_predict_manifest.json"
    value = _load_json(path)
    _canonical_path(value.get("manifest_path"), path, label="prediction manifest")
    prepared_path = args.output / "prepared_manifest.json"
    _canonical_path(
        value.get("prepared_manifest_path"),
        prepared_path,
        label="prediction prepared manifest",
    )
    if (
        value.get("schema") != PREDICTION_MANIFEST_SCHEMA
        or value.get("smoke") is not bool(prepared["smoke"])
        or value.get("orchestrator_sha256") != _sha256(Path(__file__).resolve())
        or value.get("all_folds_complete") is not True
        or value.get("heldout_ST_opened") is not False
        or value.get("prepared_manifest_sha256") != _sha256(args.output / "prepared_manifest.json")
        or set(value.get("folds", {})) != set(prepared["donors"])
    ):
        raise ValueError("fold-local prediction manifest is incomplete")
    return value


def _preflight_predictions(
    args: argparse.Namespace,
    runner: Any,
    prepared: Mapping[str, object],
    predicted: Mapping[str, object],
) -> None:
    folds = predicted.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != set(prepared["donors"]):
        raise ValueError("prediction folds do not exactly cover prepared donors")
    for donor in prepared["donors"]:
        donor = str(donor)
        fold = prepared["folds"][donor]
        receipt = folds[donor]
        if not isinstance(receipt, Mapping):
            raise ValueError(f"prediction receipt is malformed for {donor}")
        required = _prediction_receipt_required(args, prepared, fold, donor)
        if any(receipt.get(name) != expected for name, expected in required.items()):
            raise ValueError(f"prediction receipt identity changed for {donor}")
        receipt_path = args.output / "folds" / donor / "fit_predict_receipt.json"
        prediction_path = args.output / "folds" / donor / "predictions.npz"
        _canonical_path(
            receipt.get("receipt_path"), receipt_path, label=f"prediction receipt {donor}"
        )
        _canonical_path(
            receipt.get("prediction_path"),
            prediction_path,
            label=f"prediction artifact {donor}",
        )
        if _sha256(receipt_path) != receipt.get("receipt_file_sha256"):
            raise ValueError(f"prediction receipt file identity changed for {donor}")
        receipt_file = _load_json(receipt_path)
        expected_file = {
            name: item for name, item in receipt.items() if name != "receipt_file_sha256"
        }
        if receipt_file != expected_file or not _receipt_matches_required(receipt_file, required):
            raise ValueError(f"prediction receipt content changed for {donor}")
        public = _verify_npz(
            runner,
            Path(str(fold["public_path"])),
            file_sha256=str(fold["public_file_sha256"]),
            semantic_sha256=str(fold["public_semantic_sha256"]),
        )
        predictions = _verify_npz(
            runner,
            prediction_path,
            file_sha256=str(receipt["prediction_file_sha256"]),
            semantic_sha256=str(receipt["prediction_semantic_sha256"]),
        )
        _validate_fold_local_prediction(
            runner, predictions, public, donor=donor, epochs=args.epochs
        )


def _score_m0_m3(
    runner: Any,
    core: Any,
    secret: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
) -> Mapping[str, object]:
    if runner._scalar_text(secret["heldout_donor"]) != runner._scalar_text(
        predictions["heldout_donor"]
    ):
        raise ValueError("prediction and target donor identities differ")
    if not np.array_equal(secret["heldout_spot_ids"], predictions["query_spot_ids"]):
        raise ValueError("prediction rows do not align with the sealed target")
    keep = np.asarray(secret["primary_score_eligible"], dtype=bool)
    counts = np.asarray(secret["heldout_st_counts"], dtype=np.float32)[keep]
    library = np.asarray(secret["heldout_st_library"], dtype=np.float32)[keep]
    sections = np.asarray(secret["heldout_section_ids"]).astype(str)[keep]
    indications = np.asarray(secret["heldout_indication_ids"]).astype(str)[keep]
    theta = np.asarray(predictions["training_only_dispersion"], dtype=np.float32)
    if len(set(indications.tolist())) != 1:
        raise ValueError("held-out donor spans indications")
    losses: dict[str, float] = {}
    log_likelihoods: dict[str, float] = {}
    for arm in ARMS:
        rate = np.asarray(predictions[f"rate_{arm}"], dtype=np.float32)[keep]
        mean = rate * library[:, None]
        losses[arm] = runner._section_macro(
            runner._nb_deviance_rows(core, counts, mean, theta), sections
        )
        log_likelihoods[arm] = runner._section_macro(
            runner._nb_log_likelihood_rows(core, counts, mean, theta), sections
        )
    return {
        "heldout_donor": runner._scalar_text(secret["heldout_donor"]),
        "indication": str(indications[0]),
        "scored_spots": int(keep.sum()),
        "scored_sections": len(set(sections.tolist())),
        "mean_nb_deviance": losses,
        "heldout_nb_log_likelihood": log_likelihoods,
    }


def _validate_score_target_identity(
    runner: Any,
    secret: Mapping[str, np.ndarray],
    public: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    donor: str,
) -> None:
    """Bind the newly prepared fold-local target immediately after opening it."""

    rows = len(np.asarray(public["query_spot_ids"]))
    genes = len(np.asarray(public["gene_ids"]))
    if (
        runner._scalar_text(secret.get("schema")) != runner.PREPARED_SCHEMA
        or runner._scalar_text(secret.get("heldout_donor")) != donor
        or runner._scalar_text(public.get("heldout_donor")) != donor
        or runner._scalar_text(predictions.get("heldout_donor")) != donor
    ):
        raise ValueError(f"fold-local target scalar identity differs for {donor}")
    for target_field, public_field in (
        ("heldout_spot_ids", "query_spot_ids"),
        ("heldout_section_ids", "query_section_ids"),
        ("heldout_indication_ids", "query_indication_ids"),
    ):
        if not np.array_equal(
            np.asarray(secret[target_field]).astype(str),
            np.asarray(public[public_field]).astype(str),
        ):
            raise ValueError(f"fold-local target {target_field} differs for {donor}")
    if not np.array_equal(secret["heldout_spot_ids"], predictions["query_spot_ids"]):
        raise ValueError(f"fold-local prediction rows differ from target for {donor}")
    target_genes = np.asarray(secret.get("gene_ids"))
    public_genes = np.asarray(public["gene_ids"])
    prediction_genes = np.asarray(predictions["gene_ids"])
    if (
        target_genes.shape != (genes,)
        or not np.array_equal(target_genes, public_genes)
        or not np.array_equal(target_genes, prediction_genes)
    ):
        raise ValueError(f"fold-local target/public/prediction gene axis differs for {donor}")

    for field in ("heldout_st_counts", "heldout_st_half_a", "heldout_st_half_b"):
        value = np.asarray(secret[field])
        if value.shape != (rows, genes) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"fold-local target {field} is malformed for {donor}")
    for field in (
        "heldout_st_library",
        "heldout_st_library_half_a",
        "heldout_st_library_half_b",
        "primary_score_eligible",
    ):
        value = np.asarray(secret[field])
        if value.shape != (rows,):
            raise ValueError(f"fold-local target {field} is malformed for {donor}")
        if field != "primary_score_eligible" and (
            not np.isfinite(value).all() or np.any(value < 0)
        ):
            raise ValueError(f"fold-local target {field} values are invalid for {donor}")
    zero_depth = np.asarray(secret.get("zero_depth_excluded_count"))
    full = np.asarray(secret["heldout_st_counts"])
    half_a = np.asarray(secret["heldout_st_half_a"])
    half_b = np.asarray(secret["heldout_st_half_b"])
    library = np.asarray(secret["heldout_st_library"])
    library_a = np.asarray(secret["heldout_st_library_half_a"])
    library_b = np.asarray(secret["heldout_st_library_half_b"])
    eligible = np.asarray(secret["primary_score_eligible"], dtype=bool)
    if (
        zero_depth.shape != ()
        or not np.isfinite(zero_depth)
        or not np.array_equal(full, half_a + half_b)
        or not np.array_equal(library, library_a + library_b)
        or not np.array_equal(eligible, library > 0)
        or int(zero_depth) != int(np.count_nonzero(~eligible))
        or np.any(full.sum(axis=1) > library)
        or np.any(half_a.sum(axis=1) > library_a)
        or np.any(half_b.sum(axis=1) > library_b)
    ):
        raise ValueError(f"fold-local target count/exposure contract differs for {donor}")


def _comparison_summary(
    runner: Any,
    core: Any,
    per_donor: Mapping[str, Mapping[str, float]],
    *,
    baseline_arm: str,
    model_arm: str,
    seed: int,
) -> Mapping[str, object]:
    donors = tuple(sorted(per_donor))
    effects = np.asarray(
        [per_donor[donor][baseline_arm] - per_donor[donor][model_arm] for donor in donors],
        dtype=np.float64,
    )
    return {
        "effect_definition": f"{baseline_arm}_loss_minus_{model_arm}_loss",
        "heldout_donor_order": list(donors),
        "donor_improvement": effects.tolist(),
        "mean_improvement": float(effects.mean()),
        "median_improvement": float(np.median(effects)),
        "donor_fraction_improved": float(np.mean(effects > 0)),
        "paired_donor_bootstrap_confidence_interval": list(
            runner._donor_bootstrap_interval(effects, seed=seed)
        ),
        "exact_one_sided_sign_flip": runner._sign_flip(core, effects),
    }


def _relative_reduction_summary(
    runner: Any,
    core: Any,
    per_donor: Mapping[str, Mapping[str, float]],
    *,
    seed: int,
    original_relative_gain: float,
) -> Mapping[str, object]:
    """Summarize within-donor relative gains without pooling unlike gene axes."""

    donors = tuple(sorted(per_donor))
    baseline = np.asarray([per_donor[donor]["M0"] for donor in donors], dtype=np.float64)
    fusion = np.asarray([per_donor[donor]["M3"] for donor in donors], dtype=np.float64)
    if (
        not np.isfinite(baseline).all()
        or not np.isfinite(fusion).all()
        or np.any(baseline <= 0)
        or not np.isfinite(original_relative_gain)
        or original_relative_gain <= 0
    ):
        raise ValueError("relative reduction requires finite losses and a positive baseline")
    effects = (baseline - fusion) / baseline
    mean = float(np.mean(effects))
    median = float(np.median(effects))
    return {
        "effect_definition": "within_donor_(M0_loss_minus_M3_loss)_divided_by_M0_loss",
        "gene_axis_handling": (
            "relative_reduction_computed_within_each_donor_before_donor_summary_"
            "because_fold_panels_differ"
        ),
        "heldout_donor_order": list(donors),
        "donor_relative_reduction": effects.tolist(),
        "mean_relative_reduction": mean,
        "median_relative_reduction": median,
        "donor_fraction_improved": float(np.mean(effects > 0)),
        "paired_donor_bootstrap_confidence_interval": list(
            runner._donor_bootstrap_interval(effects, seed=seed)
        ),
        "exact_one_sided_sign_flip": runner._sign_flip(core, effects),
        "original_external_panel_relative_gain": float(original_relative_gain),
        "original_external_panel_relative_gain_percent": float(100.0 * original_relative_gain),
        "mean_gain_retained_vs_original": mean / float(original_relative_gain),
        "median_gain_retained_vs_original": median / float(original_relative_gain),
    }


def score(args: argparse.Namespace, runner: Any) -> Mapping[str, object]:
    prepared = _read_prepared(args, runner)
    if bool(prepared["smoke"]):
        raise ValueError("smoke artifacts are prohibited from scientific scoring")
    predicted = _read_prediction_manifest(args, prepared)
    # Global target-free preflight must finish before the first target is opened.
    _preflight_predictions(args, runner, prepared, predicted)
    targets = _read_score_target_manifest(args, runner, prepared)
    contracts = _validate_contracts(args, runner, verify_source=False, verify_baseline=True)
    core = runner._import_core()
    fold_reports: dict[str, object] = {}
    donor_losses: dict[str, Mapping[str, float]] = {}
    for donor in prepared["donors"]:
        donor = str(donor)
        fold = prepared["folds"][donor]
        target_fold = targets["folds"][donor]
        receipt = predicted["folds"][donor]
        public = _verify_npz(
            runner,
            Path(str(fold["public_path"])),
            file_sha256=str(fold["public_file_sha256"]),
            semantic_sha256=str(fold["public_semantic_sha256"]),
        )
        predictions = _verify_npz(
            runner,
            Path(str(receipt["prediction_path"])),
            file_sha256=str(receipt["prediction_file_sha256"]),
            semantic_sha256=str(receipt["prediction_semantic_sha256"]),
        )
        _validate_fold_local_prediction(
            runner, predictions, public, donor=donor, epochs=args.epochs
        )
        secret = _verify_npz(
            runner,
            Path(str(target_fold["score_target_path"])),
            file_sha256=str(target_fold["score_target_file_sha256"]),
            semantic_sha256=str(target_fold["score_target_semantic_sha256"]),
        )
        _validate_score_target_identity(runner, secret, public, predictions, donor=donor)
        fold_report = _score_m0_m3(runner, core, secret, predictions)
        fold_reports[donor] = fold_report
        donor_losses[donor] = fold_report["mean_nb_deviance"]
        _atomic_json(args.output / "folds" / donor / "score_report_M0_M3.json", fold_report)

    donor_balanced = {
        arm: float(np.mean([donor_losses[donor][arm] for donor in prepared["donors"]]))
        for arm in ARMS
    }
    comparison_m0 = _comparison_summary(
        runner,
        core,
        donor_losses,
        baseline_arm="M0",
        model_arm="M3",
        seed=args.seed + 301,
    )
    comparison_m1 = _comparison_summary(
        runner,
        core,
        donor_losses,
        baseline_arm="M1",
        model_arm="M3",
        seed=args.seed + 302,
    )
    baseline = _load_json(args.baseline_report)
    original_relative_gain = float(baseline["ordered_gates"]["decision"]["gate_1"]["relative_gain"])
    if not np.isclose(
        original_relative_gain,
        ORIGINAL_EXTERNAL_PANEL_RELATIVE_GAIN,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("frozen external-panel relative gain changed")
    donor_relative = _relative_reduction_summary(
        runner,
        core,
        donor_losses,
        seed=args.seed + 303,
        original_relative_gain=original_relative_gain,
    )
    fold_local_relative_gain = float(
        (donor_balanced["M0"] - donor_balanced["M3"]) / donor_balanced["M0"]
    )
    retention = fold_local_relative_gain / original_relative_gain
    positive_fraction = float(comparison_m0["donor_fraction_improved"])
    report = {
        "schema": REPORT_SCHEMA,
        "analysis_scope": "outcome_exposed_gene_panel_sensitivity_non_confirmatory",
        "scientific_authorization": "none",
        "purpose": "test_stability_to_training_only_fold_local_gene_selection",
        "image_encoder": "bioptimus/H-optimus-1",
        "uni2_h_run": False,
        "iterative_refinement_run": False,
        "frozen_architecture_or_hyperparameter_change": False,
        "panel_design": {
            "selection_unit": "other_12_training_donors_only",
            "fold_panel_gene_count": prepared["fold_panel_gene_count"],
            "union_projection_gene_count": prepared["panel_union_gene_count"],
            "genes_common_to_all_folds": prepared["common_gene_count"],
            "frozen_fit_calls_total": len(prepared["donors"]),
            "frozen_fit_calls_per_donor": 1,
        },
        "donor_balanced_mean_nb_deviance": donor_balanced,
        "aggregate_absolute_loss_scope": ("descriptive_only_because_fold_local_gene_axes_differ"),
        "comparisons": {
            "M3_vs_M0": comparison_m0,
            "M3_vs_M1": comparison_m1,
            "M3_vs_M0_within_donor_relative_reduction": donor_relative,
        },
        "original_external_panel_comparison": {
            "relative_M3_vs_M0_gain": original_relative_gain,
            "relative_M3_vs_M0_gain_percent": 100.0 * original_relative_gain,
            "report_path": str(args.baseline_report.resolve()),
            "report_sha256": contracts["baseline_report_sha256"],
        },
        "fold_local_stability": {
            "relative_M3_vs_M0_gain": fold_local_relative_gain,
            "fraction_of_original_relative_gain_retained": retention,
            "retention_is_reported_without_a_new_tuned_threshold": True,
            "M3_vs_M0_positive_donor_fraction": positive_fraction,
            "M3_vs_M0_at_least_70_percent_positive_donors": positive_fraction >= 0.70,
            "M3_vs_M0_direction_stable": float(comparison_m0["mean_improvement"]) > 0,
            "M3_vs_M1_direction_favors_fusion": float(comparison_m1["mean_improvement"]) > 0,
            "donor_wise_relative_M3_vs_M0": donor_relative,
        },
        "interpretation_boundary": (
            "gene_selection_sensitivity_only; panels_and_architecture_were_exposed_to_the_"
            "NatCommun_development_cohort; this_result_cannot_confirm_the_hypothesis"
        ),
        "folds": fold_reports,
        "artifact_identities": {
            "source_sha256": contracts["source_sha256"],
            "panel_receipts_file_sha256": prepared["panel_receipts_file_sha256"],
            "panel_receipts_identity_sha256": prepared["panel_receipts_identity_sha256"],
            "development_protocol_sha256": prepared["development_protocol_sha256"],
            "validation_protocol_sha256": prepared["validation_protocol_sha256"],
            "frozen_runner_sha256": prepared["frozen_runner_sha256"],
            "frozen_core_sha256": prepared["frozen_core_sha256"],
            "orchestrator_sha256": prepared["orchestrator_sha256"],
            "prepared_manifest_sha256": _sha256(args.output / "prepared_manifest.json"),
            "fit_predict_manifest_sha256": _sha256(args.output / "fit_predict_manifest.json"),
            "score_target_manifest_sha256": _sha256(args.output / "score_target_manifest.json"),
        },
    }
    _atomic_json(args.output / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "fit-predict", "score"), default="prepare")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--panel-receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--validation-protocol", type=Path, default=VALIDATION_PROTOCOL_PATH)
    parser.add_argument("--development-protocol", type=Path, default=DEVELOPMENT_PROTOCOL_PATH)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.60)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_training_args(args)
    runner = _load_runner()
    runner.configure_resources(
        cpu_threads=args.cpu_threads,
        gpu_memory_fraction=args.gpu_memory_fraction,
        device=args.device,
    )
    runner.seed_everything(args.seed)
    if args.stage == "prepare":
        prepare(args, runner)
    elif args.stage == "fit-predict":
        fit_predict(args, runner)
    else:
        score(args, runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
