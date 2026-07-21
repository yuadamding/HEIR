#!/usr/bin/env python3
"""Preflight and smoke the blocked HEST nonlinear qualification v1.

The registered v1 protocol currently fails closed because its source lacks a
receipt-bound blank-patch embedding and its best-registration subset has no
donor/type stratum at the primary support threshold.  This runner therefore
supports metadata preflight and a synthetic implementation smoke, but it will
not fit exposed biological outcomes under ``--phase full`` while those frozen
blockers remain.
"""

# ruff: noqa: E402 -- native thread limits must be set before NumPy/Torch imports.

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

# Bound native thread pools before NumPy and PyTorch are imported. A smaller
# externally requested value is preserved; an absent or excessive value is
# capped at the registered four-thread maximum.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        _thread_value = int(os.environ.get(_thread_variable, "4"))
    except ValueError:
        _thread_value = 4
    os.environ[_thread_variable] = str(min(max(_thread_value, 1), 4))

import numpy as np
import torch

from heir.evaluation.hierarchical_metrics import donor_section_type_macro_r2
from heir.evaluation.neural_checkpoint import canonical_array_registry_sha256
from heir.evaluation.neural_model_selection import (
    NeuralCandidate,
    refit_selected_neural_probe,
    select_neural_hyperparameters,
)
from heir.evaluation.neural_nulls import (
    COMPLETE_REFIT_STEPS,
    build_neural_null_design,
    run_refitted_neural_null,
)
from heir.evaluation.neural_probe import (
    load_neural_residual_fit,
    predict_neural_residual_probe,
    save_neural_residual_fit,
)
from heir.evaluation.nonlinear_qualification_contract import (
    ANALYSIS_STATUS,
    REPORT_SCHEMA,
    canonical_sha256,
    file_sha256,
    non_authorizing_report_fields,
    validate_protocol,
    validate_retrospective_manifest,
)
from heir.evaluation.ridge_probe import target_coordinates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/mnt/seagate/HEIR_runs/hest_hoptimus1_qualification/source.npz")
DEFAULT_PROTOCOL = ROOT / "configs/hest_nonlinear_qualification_v1.json"
DEFAULT_MANIFEST = ROOT / "manifests/studies/hest_nonlinear_qualification.retrospective.json"
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v1/report.json")
DEFAULT_MARKDOWN = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v1/report.md")
DEFAULT_CHECKPOINTS = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v1/checkpoints")
SMOKE_SCHEMA = "heir.hest_nonlinear_qualification_smoke.v1"
EXPECTED_SOURCE_SCHEMA = "heir.registered_observations_retrospective.v1"
EXPECTED_ENCODER = "bioptimus/H-optimus-1"
EXPECTED_CROPS = (
    "crop_112um",
    "cell_mask_only",
    "nucleus_mask_only",
    "target_cell_removed_112um",
)


def _npz_shape(path: Path, name: str) -> tuple[int, ...]:
    """Read an NPY member header without inflating its array payload."""

    member = name + ".npy"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise ValueError(f"registered source is missing {name}")
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(handle)
            elif version in {(2, 0), (3, 0)}:
                shape, _, _ = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"unsupported NPY header version for {name}")
    return tuple(int(value) for value in shape)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise ValueError(f"registered source field {name} must be scalar")
    return value.item()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _runtime_provenance(requested_device: str) -> Mapping[str, object]:
    """Capture execution identity without mutating the worktree."""

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_lines = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        git_error = None
    except (OSError, subprocess.CalledProcessError) as error:
        head = None
        status_lines = []
        git_error = f"{type(error).__name__}: {error}"
    return {
        "git_commit": head,
        "git_worktree_dirty": bool(status_lines),
        "git_status_entry_count": len(status_lines),
        "git_probe_error": git_error,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "requested_device": requested_device,
        "cuda_available": torch.cuda.is_available(),
        "cpu_thread_limit": torch.get_num_threads(),
        "native_thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _type_mapping(raw_labels: np.ndarray, type_names: np.ndarray) -> Mapping[str, object]:
    raw = np.asarray(raw_labels, dtype=np.int64)
    names = np.asarray(type_names).astype(str)
    observed = tuple(sorted(set(raw.tolist())))
    if not observed or min(observed) < 0 or max(observed) >= len(names):
        raise ValueError("registered fine-type labels do not match the source ontology")
    rows = [
        {"raw_label": int(label), "contiguous_label": index, "type_name": names[label]}
        for index, label in enumerate(observed)
    ]
    return {
        "rule": "sorted_observed_raw_label_in_source_ontology_order",
        "raw_label_count": len(names),
        "observed_type_count": len(rows),
        "rows": rows,
        "sha256": canonical_sha256(rows),
    }


def inspect_registered_source(
    source: Path,
    protocol: Mapping[str, object],
    *,
    verify_hash: bool,
) -> Mapping[str, object]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    registered = protocol["registered_source"]
    if not isinstance(registered, Mapping):
        raise ValueError("protocol registered_source is malformed")
    actual_sha = file_sha256(source) if verify_hash else None
    if verify_hash and actual_sha != registered["sha256"]:
        raise ValueError("registered HEST source hash differs from the protocol")
    with np.load(source, allow_pickle=False) as archive:
        schema = str(_scalar(archive, "schema_version"))
        status = str(_scalar(archive, "analysis_status"))
        encoder = str(_scalar(archive, "encoder_name"))
        authorizations = {
            name: bool(_scalar(archive, name))
            for name in ("authorizes_h_cell", "authorizes_h_intrinsic", "authorizes_full_heir")
        }
        crop_ids = tuple(np.asarray(archive["crop_ids"]).astype(str).tolist())
        donor_ids = np.asarray(archive["donor_ids"]).astype(str)
        section_ids = np.asarray(archive["section_ids"]).astype(str)
        raw_labels = np.asarray(archive["type_labels"], dtype=np.int64)
        type_names = np.asarray(archive["type_names"]).astype(str)
        roles = np.asarray(archive["pool_roles"]).astype(str)
        registration = np.asarray(archive["registration_quality_strata"]).astype(str)
        blank_available = "blank_patch_features" in archive.files or "blank_patch" in crop_ids
    image_shape = _npz_shape(source, "image_features")
    target_shape = _npz_shape(source, "nucleus_molecular_targets")
    if schema != EXPECTED_SOURCE_SCHEMA or status != ANALYSIS_STATUS:
        raise ValueError("source is not the registered exposed non-authorizing HEST artifact")
    if encoder != EXPECTED_ENCODER or crop_ids != EXPECTED_CROPS:
        raise ValueError("source encoder or crop registry differs from H-optimus v1")
    if any(authorizations.values()):
        raise ValueError("registered exposed source unexpectedly carries authorization")
    if image_shape != (len(donor_ids), 4, 1536) or target_shape != (len(donor_ids), 260):
        raise ValueError("registered source feature or target shape is malformed")
    if len(donor_ids) != 36121 or len(set(donor_ids.tolist())) != 15:
        raise ValueError("registered HEST source row or donor count differs from v1")
    if len(set(section_ids.tolist())) != 20:
        raise ValueError("registered HEST source section count differs from v1")
    mapping = _type_mapping(raw_labels, type_names)
    compact_lookup = {
        int(row["raw_label"]): int(row["contiguous_label"]) for row in mapping["rows"]
    }
    compact = np.asarray([compact_lookup[int(value)] for value in raw_labels], dtype=np.int64)
    evaluation = np.char.startswith(roles, "evaluation")
    best = evaluation & (registration == "best")
    supported = 0
    for donor in sorted(set(donor_ids[best].tolist())):
        for type_index in sorted(set(compact[best & (donor_ids == donor)].tolist())):
            if np.count_nonzero(best & (donor_ids == donor) & (compact == type_index)) >= 20:
                supported += 1
    blockers = []
    if not blank_available:
        blockers.append("registered_source_missing_receipt_bound_blank_patch_embedding")
    if supported == 0:
        blockers.append("best_registration_subset_has_zero_donor_type_strata_at_primary_support_20")
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": actual_sha,
        "hash_verified": bool(verify_hash),
        "schema": schema,
        "analysis_status": status,
        "encoder": encoder,
        "crop_ids": list(crop_ids),
        "rows": len(donor_ids),
        "donors": len(set(donor_ids.tolist())),
        "sections": len(set(section_ids.tolist())),
        "image_shape": list(image_shape),
        "target_shape": list(target_shape),
        "type_mapping": mapping,
        "blank_patch_embedding_available": blank_available,
        "best_registration_evaluation_rows": int(best.sum()),
        "best_registration_supported_donor_type_strata_at_20": supported,
        "blockers": blockers,
    }


def preflight(
    source: Path,
    protocol_path: Path,
    manifest_path: Path,
    *,
    verify_source_hash: bool,
) -> Mapping[str, object]:
    protocol = validate_protocol(_json(protocol_path))
    protocol_sha = file_sha256(protocol_path)
    manifest = validate_retrospective_manifest(
        _json(manifest_path), protocol, protocol_file_sha256=protocol_sha
    )
    source_receipt = inspect_registered_source(source, protocol, verify_hash=verify_source_hash)
    blockers = list(source_receipt["blockers"])
    if blockers != list(manifest["execution"]["blockers"]):
        raise ValueError("observed preflight blockers differ from the frozen manifest")
    return {
        "schema": "heir.hest_nonlinear_qualification_preflight.v1",
        "analysis_status": ANALYSIS_STATUS,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "source": source_receipt,
        "execution_authorized": not blockers,
        "blockers": blockers,
        **non_authorizing_report_fields(False),
    }


def _synthetic_smoke(checkpoint_dir: Path, *, device: str) -> Mapping[str, object]:
    rng = np.random.default_rng(20260720)
    donors = np.repeat(["D1", "D2", "D3", "D4"], 24)
    labels = np.tile(np.repeat([0, 1], 12), 4)
    sections = np.asarray([f"{donor}-S" for donor in donors])
    blocks = np.tile(np.repeat(["a", "b"], 6), 8)
    identities = np.asarray([f"synthetic-{index:04d}" for index in range(len(donors))])
    features = rng.normal(size=(len(donors), 8)).astype(np.float32)
    technical = rng.normal(size=(len(donors), 1))
    reference = np.column_stack((labels * 0.1, labels * -0.1, labels * 0.05))
    signal = 3.0 * features[:, 0] + labels * features[:, 1]
    targets = reference + np.column_stack((signal, -0.5 * signal, 0.25 * signal))
    targets += technical @ np.asarray([[0.05, -0.02, 0.01]])

    candidate = NeuralCandidate("mlp_tiny", True, 1.0e-4, 1)

    def nested_fit_predict(
        local_features: np.ndarray,
    ) -> tuple[tuple[object, ...], float, Mapping[str, object]]:
        selection = select_neural_hyperparameters(
            local_features,
            targets,
            reference,
            labels,
            donors,
            sections,
            identities,
            technical,
            num_types=2,
            candidates=(candidate,),
            seeds=(17,),
            max_epochs=20,
            batch_size=32,
            patience=10,
            minimum_support=5,
            minimum_variance_ratio=0.0,
            device=device,
        )
        fits = refit_selected_neural_probe(
            selection,
            local_features,
            targets,
            reference,
            labels,
            donors,
            sections,
            identities,
            technical,
            num_types=2,
            batch_size=32,
            device=device,
        )
        coordinate_predictions = []
        for fit in fits:
            coordinates, _ = predict_neural_residual_probe(
                fit, local_features, reference, labels, device=device
            )
            coordinate_predictions.append(coordinates)
        truth, _ = target_coordinates(
            fits[0].target,
            targets,
            reference,
            technical,
            labels,
        )
        ensemble = np.mean(np.stack(coordinate_predictions, axis=0), axis=0)
        score, _, _, _ = donor_section_type_macro_r2(
            truth,
            ensemble,
            donors,
            sections,
            labels,
            5,
        )
        target_sha256 = canonical_array_registry_sha256(
            {
                "technical_means": fits[0].target.technical_means,
                "technical_coefficients": fits[0].target.technical_coefficients,
                "residual_means": fits[0].target.residual_means,
                "bases": fits[0].target.bases,
            }
        )
        return (
            fits,
            float(score),
            {
                "selected_candidate_id": selection.selected.candidate_id,
                "selected_epoch": selection.selected_epoch,
                "inner_donors": list(selection.inner_donors),
                "selection_rule": selection.selection_rule,
                "candidate_receipt": selection.candidates[0],
                "refit_checkpoint_sha256": [fit.checkpoint_sha256 for fit in fits],
                "target_fit_sha256": target_sha256,
            },
        )

    fits, observed, observed_selection = nested_fit_predict(features)
    fit = fits[0]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "synthetic_smoke_probe.npz"
    checkpoint_receipt = save_neural_residual_fit(checkpoint_path, fit)
    loaded = load_neural_residual_fit(checkpoint_path)
    _, first_prediction = predict_neural_residual_probe(
        fit, features, reference, labels, device=device
    )
    _, loaded_prediction = predict_neural_residual_probe(
        loaded, features, reference, labels, device=device
    )
    if not np.allclose(first_prediction, loaded_prediction, rtol=0.0, atol=1.0e-7):
        raise RuntimeError("synthetic checkpoint replay changed predictions")

    null_results = {}
    for null_index, kind in enumerate(
        ("within_section_type_derangement", "different_spatial_block_reassignment")
    ):
        design = build_neural_null_design(
            kind,
            20,
            donors,
            sections,
            labels,
            blocks,
            identities,
            seed=20260720 + null_index * 1_000_003,
        )

        def refit(
            mapped: np.ndarray,
            mapping: np.ndarray,
            receipt: Mapping[str, object],
        ) -> Mapping[str, object]:
            del mapping
            _, score, local = nested_fit_predict(mapped)
            return {
                "mapping_sha256": receipt["mapping_sha256"],
                "completed_steps": list(COMPLETE_REFIT_STEPS),
                "score": score,
                **local,
            }

        null_results[kind] = run_refitted_neural_null(observed, features, design, refit)
    return {
        "schema": SMOKE_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "scope": "synthetic_implementation_smoke_no_biological_rows",
        "biological_data_fit": False,
        "device": device,
        "checkpoint_receipt": checkpoint_receipt,
        "checkpoint_replay_identical": True,
        "observed_selection": observed_selection,
        "selection_smoke_scope": {
            "candidate_count": 1,
            "seed_count": 1,
            "maximum_epochs": 20,
            "minimum_variance_ratio": 0.0,
            "purpose": "exercise_complete_nested_refit_path_not_estimate_biological_performance",
        },
        "complete_refit_steps_exercised": list(COMPLETE_REFIT_STEPS),
        "nulls": null_results,
        "smoke_pass": True,
        **non_authorizing_report_fields(False),
    }


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# HEST nonlinear qualification v1",
        "",
        f"- Analysis status: `{report['analysis_status']}`",
        f"- Requested phase: `{report['requested_phase']}`",
        f"- Execution status: `{report['execution_status']}`",
        "- Biological authorization: **false for every HEIR hypothesis**",
        f"- Git commit: `{report['runtime']['git_commit']}`",
        f"- Dirty worktree at execution: `{report['runtime']['git_worktree_dirty']}`",
        "",
    ]
    blockers = report.get("blockers", [])
    if blockers:
        lines.extend(["## Registered blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
        lines.append("")
    if report.get("smoke"):
        lines.extend(
            [
                "## Implementation smoke",
                "",
                "The smoke used synthetic rows only and exercised checkpoint replay plus "
                "exactly 20 fully refitted mappings for each registered null family. It is "
                "not a biological experiment.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The full exposed-HEST qualification was not run because doing so would violate "
            "the frozen v1 completeness rule. Resolving the blockers requires a new "
            "receipt-bound input supplement or a prospectively registered protocol revision; "
            "neither may be chosen after examining model outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    source: Path,
    protocol: Path,
    manifest: Path,
    output: Path,
    markdown_output: Path,
    checkpoint_dir: Path,
    phase: str,
    device: str,
    verify_source_hash: bool,
) -> Mapping[str, object]:
    if phase not in {"preflight", "smoke", "full"}:
        raise ValueError("qualification phase must be preflight, smoke, or full")
    preflight_receipt = preflight(source, protocol, manifest, verify_source_hash=verify_source_hash)
    smoke = _synthetic_smoke(checkpoint_dir, device=device) if phase == "smoke" else None
    authorized = bool(preflight_receipt["execution_authorized"])
    if phase == "full" and authorized:
        raise NotImplementedError(
            "full fitting requires a newly frozen unblocked protocol and source supplement"
        )
    status = (
        "synthetic_smoke_complete_full_biological_run_blocked"
        if phase == "smoke"
        else "blocked_before_biological_fit"
        if phase == "full"
        else "preflight_complete_execution_blocked"
    )
    report = {
        "schema": REPORT_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "requested_phase": phase,
        "execution_status": status,
        "runtime": _runtime_provenance(device),
        "preflight": preflight_receipt,
        "smoke": smoke,
        "blockers": list(preflight_receipt["blockers"]),
        "biological_experiment_run": False,
        "engineering_decision_available": False,
        **non_authorizing_report_fields(False),
    }
    _atomic_json(output, report)
    _atomic_text(markdown_output, _markdown(report))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cpu")
    parser.add_argument("--phase", choices=("preflight", "smoke", "full"), default="preflight")
    parser.add_argument("--verify-source-hash", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.cpu_threads <= 4:
        raise ValueError("cpu-threads must be between one and four")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    if str(args.device).startswith("cuda"):
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "set CUBLAS_WORKSPACE_CONFIG before launching deterministic CUDA smoke"
            )
        torch.cuda.set_per_process_memory_fraction(0.60, 0)
    report = run(
        source=args.source.expanduser().resolve(),
        protocol=args.protocol.expanduser().resolve(),
        manifest=args.manifest.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        markdown_output=args.markdown_output.expanduser().resolve(),
        checkpoint_dir=args.checkpoint_dir.expanduser().resolve(),
        phase=args.phase,
        device=str(args.device),
        verify_source_hash=bool(args.verify_source_hash),
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "execution_status": report["execution_status"],
                "report": str(args.output.expanduser().resolve()),
                "markdown_report": str(args.markdown_output.expanduser().resolve()),
                "blockers": report["blockers"],
                "synthetic_smoke_pass": (
                    bool(report["smoke"]["smoke_pass"]) if report["smoke"] else None
                ),
                "biological_experiment_run": report["biological_experiment_run"],
                "engineering_decision_available": report["engineering_decision_available"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 2 if args.phase == "full" and report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
