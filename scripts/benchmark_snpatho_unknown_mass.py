#!/usr/bin/env python3
"""Evaluate the seed-17 snPATHO fixed unknown-mass sensitivity grid.

The five sensitivity values are intentionally fixed here rather than accepted
from the command line.  Each canonical hardened-runner output is validated
against its prediction telemetry and complete refinement lineage before it is
scored against the frozen Visium truth.  Missing or invalid cases leave the
stability conclusion blocked and make the command return a non-zero status.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from heir.data import PrototypeSet
from heir.inference import PredictionBundle
from heir.utils import reject_output_input_collisions, sha256_file


def _load_matrix_helpers() -> Any:
    """Load the sibling evaluator regardless of the caller's import path."""

    module_name = "_heir_snpatho_refinement_matrix_helpers"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    path = Path(__file__).resolve().with_name("benchmark_snpatho_refinement_matrix.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load refinement-matrix helpers from %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_MATRIX = _load_matrix_helpers()
HARD_BASELINE = _MATRIX.HARD_BASELINE
METHOD = _MATRIX.METHOD
SOFT_BASELINE = _MATRIX.SOFT_BASELINE
ArtifactRequest = _MATRIX.ArtifactRequest
SampleInputs = _MATRIX.SampleInputs
_json_object = _MATRIX._json_object
load_prediction = _MATRIX.load_prediction
load_sample_inputs = _MATRIX.load_sample_inputs
score_prediction = _MATRIX.score_prediction
_paired_spearman_delta = _MATRIX._paired_spearman_delta
_practical_delta_status = _MATRIX._practical_delta_status
_raw_sign_status = _MATRIX._raw_sign_status


def _load_runner_helpers() -> Any:
    """Load the canonical runner so the scorer cannot redefine its run grid."""

    module_name = "_heir_snpatho_refinement_runner_helpers"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    path = Path(__file__).resolve().with_name("run_snpatho_refinement_benchmark.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load refinement runner helpers from %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_RUNNER = _load_runner_helpers()


REPORT_SCHEMA = "heir.snpatho_unknown_mass_sensitivity.v1"
DEFAULT_SAMPLES = ("4066", "4399", "4411")
SEED = 17
UNKNOWN_MASSES = (0.0, 0.01, 0.05, 0.10, 0.20)
COMPACT_METRICS = (
    "median_gene_spearman",
    "median_gene_pearson",
    "median_gene_mse",
    "median_gene_mae",
    "mean_location_cosine",
    "median_location_spearman",
    "morans_i_spearman",
)
CONTRASTS = (
    "refined_minus_round0",
    "heir_minus_hard_baseline",
    "heir_minus_soft_baseline",
)
_DIRECTION_EPSILON = 1.0e-12
DEFAULT_PRACTICAL_DELTA_THRESHOLD = _MATRIX.DEFAULT_PRACTICAL_DELTA_THRESHOLD


def _scorer_source_identity() -> Dict[str, Any]:
    """Bind the report to this scorer and both dynamically loaded helpers."""

    relative_paths = (
        "scripts/benchmark_snpatho_unknown_mass.py",
        "scripts/benchmark_snpatho_refinement_matrix.py",
        "scripts/run_snpatho_refinement_benchmark.py",
    )
    source_repository = Path(__file__).resolve().parents[1]
    return dict(
        _RUNNER._source_identity(
            source_repository,
            _RUNNER._runtime_source_files(source_repository, relative_paths),
        )
    )


def _mass_label(value: float) -> str:
    """Return the exact label emitted by the hardened benchmark runner."""

    return ("%.2f" % value).replace(".", "p")


def _directories(artifact_root: Path, sample: str, mass: float) -> Tuple[Path, Path]:
    prefix = "model_refinement_r1_v1_seed17_unknown_mass_%s" % _mass_label(mass)
    root = artifact_root / sample
    return root / (prefix + "_round0"), root / (prefix + "_refined")


def _resolved_path(value: object, name: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError("%s path is blank" % name)
    return Path(text).expanduser().resolve()


def _require_file(
    path: Path,
    name: str,
    validated_hashes: Optional[Mapping[Path, str]] = None,
) -> str:
    if not path.is_file():
        raise FileNotFoundError("%s is absent: %s" % (name, path))
    resolved = path.resolve()
    if validated_hashes is not None and resolved in validated_hashes:
        return validated_hashes[resolved]
    return sha256_file(path)


def _validate_run_manifest(
    path: Path,
    *,
    repository: Path,
    artifact_root: Path,
    samples: Sequence[str],
    molecular_generation: str,
) -> Dict[str, Any]:
    """Validate the exact canonical grid, commands, source, and all output hashes."""

    path = path.expanduser().resolve()
    manifest_sha256 = _require_file(path, "unknown-mass run manifest")
    payload = _json_object(path, "unknown-mass run manifest")
    if payload.get("schema") != _RUNNER.UNKNOWN_MASS_MANIFEST_SCHEMA:
        raise ValueError("unknown-mass run manifest schema is invalid")
    samples = tuple(str(sample) for sample in samples)
    if payload.get("samples") != list(samples):
        raise ValueError("unknown-mass run manifest samples differ from the scoring request")
    if payload.get("seed") != SEED:
        raise ValueError("unknown-mass run manifest seed is not 17")
    if payload.get("molecular_generation", "r1") != molecular_generation:
        raise ValueError("unknown-mass run manifest molecular generation is stale")
    if payload.get("unknown_masses") != list(UNKNOWN_MASSES):
        raise ValueError("unknown-mass run manifest does not contain the exact fixed mass grid")
    if payload.get("stage_names") != list(_RUNNER.UNKNOWN_MASS_STAGE_NAMES):
        raise ValueError("unknown-mass run manifest stage names are non-canonical")

    plan_stages = _RUNNER.build_plan(
        repository,
        samples=samples,
        seeds=(SEED,),
        unknown_mass_sensitivity=True,
        artifact_root=artifact_root,
        molecular_generation=molecular_generation,
    )
    plan = _RUNNER.unknown_mass_plan_payload(
        plan_stages,
        samples=samples,
        molecular_generation=molecular_generation,
    )
    if payload.get("plan_sha256") != _RUNNER._canonical_sha256(plan):
        raise ValueError("unknown-mass run manifest plan SHA-256 is stale")
    if payload.get("validation_recipe_source_identity") != (
        _RUNNER.unknown_mass_source_identity(repository)
    ):
        raise ValueError("unknown-mass run manifest validation-recipe identity is stale")

    rows = payload.get("stages")
    if not isinstance(rows, list) or len(rows) != len(plan["stages"]):
        raise ValueError("unknown-mass run manifest stage grid is incomplete or duplicated")
    if payload.get("stage_count") != len(rows):
        raise ValueError("unknown-mass run manifest stage_count is stale")
    if payload.get("stage_time_artifact_identities_complete") is not True:
        raise ValueError("unknown-mass run manifest lacks stage-time artifact identities")
    validated_hashes: Dict[Path, str] = {}
    stage_lookup: Dict[Tuple[str, float, str], Mapping[str, Any]] = {}
    identity_fields = (
        "stage_index",
        "sample",
        "seed",
        "unknown_mass",
        "stage",
        "command",
    )
    for expected, row in zip(plan["stages"], rows):
        if not isinstance(row, Mapping):
            raise ValueError("unknown-mass run manifest stage row is not an object")
        for field in identity_fields:
            if row.get(field) != expected[field]:
                raise ValueError(
                    "unknown-mass run manifest stage %s differs from the canonical plan" % field
                )
        if row.get("status") not in {"completed", "skipped_valid"}:
            raise ValueError("unknown-mass run manifest contains an unvalidated stage")
        if row.get("artifact_identity_capture") != _RUNNER.STAGE_ARTIFACT_IDENTITY_CAPTURE:
            raise ValueError("unknown-mass run manifest stage identity capture is incomplete")
        inputs = row.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != len(expected["inputs"]):
            raise ValueError("unknown-mass run manifest stage inputs are incomplete")
        for expected_input, observed_input in zip(expected["inputs"], inputs):
            if not isinstance(observed_input, Mapping):
                raise ValueError("unknown-mass run manifest input row is not an object")
            input_path = _resolved_path(observed_input.get("path"), "manifest stage input")
            if (
                observed_input.get("role") != expected_input["role"]
                or input_path != Path(expected_input["path"]).resolve()
            ):
                raise ValueError("unknown-mass run manifest input identity is non-canonical")
            input_hash = _require_file(input_path, "manifested stage input")
            if observed_input.get("sha256") != input_hash:
                raise ValueError("unknown-mass run manifest input SHA-256 is stale")
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != len(expected["outputs"]):
            raise ValueError("unknown-mass run manifest stage outputs are incomplete")
        for expected_path, output in zip(expected["outputs"], outputs):
            if not isinstance(output, Mapping):
                raise ValueError("unknown-mass run manifest output row is not an object")
            output_path = _resolved_path(output.get("path"), "manifest stage output")
            if output_path != Path(expected_path).resolve():
                raise ValueError("unknown-mass run manifest output path is non-canonical")
            output_hash = _require_file(output_path, "manifested stage output")
            if output.get("sha256") != output_hash:
                raise ValueError("unknown-mass run manifest output SHA-256 is stale")
            validated_hashes[output_path] = output_hash
        key = (str(row["sample"]), float(row["unknown_mass"]), str(row["stage"]))
        if key in stage_lookup:
            raise ValueError("unknown-mass run manifest contains a duplicate stage identity")
        stage_lookup[key] = row
    return {
        "path": str(path),
        "sha256": manifest_sha256,
        "payload": payload,
        "validated_output_hashes": validated_hashes,
        "stage_lookup": stage_lookup,
    }


def _checkpoint_metadata(path: Path) -> Mapping[str, Any]:
    """Read only the safe tensor/checkpoint types needed for lineage validation."""

    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("invalid HEIR checkpoint %s: %s" % (path, error)) from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("metadata"), Mapping):
        raise ValueError("HEIR checkpoint lacks metadata: %s" % path)
    return payload["metadata"]


def _validate_round_audit(
    audit: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> int:
    """Validate the fixed-round safety trajectory and return its selected round."""

    try:
        selected = int(audit["selected_round"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("refinement audit has an invalid selected_round") from error
    if selected not in range(5):
        raise ValueError("refinement audit selected_round must be in 0..4")
    rounds = audit.get("rounds")
    if (
        not isinstance(rounds, list)
        or not rounds
        or any(not isinstance(row, Mapping) for row in rounds)
    ):
        raise ValueError("refinement audit must contain a non-empty round trajectory")
    if len(rounds) > 4 or [row.get("round_id") for row in rounds] != list(
        range(1, len(rounds) + 1)
    ):
        raise ValueError("refinement audit rounds must be consecutive and limited to four")
    try:
        candidates = [(float(audit["round_zero_validation_loss"]), 0)]
        candidates.extend(
            (float(row["validation_loss"]), int(row["round_id"]))
            for row in rounds
            if row.get("committed") is True
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("refinement audit has invalid validation losses") from error
    if any(not math.isfinite(loss) for loss, _ in candidates):
        raise ValueError("refinement audit validation losses must be finite")
    expected_selected = min(candidates)[1]
    if selected != expected_selected:
        raise ValueError("selected round is not the lowest-loss safe round with round-0 fallback")
    round_checkpoints = audit.get("round_checkpoints")
    if not isinstance(round_checkpoints, Mapping):
        raise ValueError("refinement audit round_checkpoints must be an object")
    if metadata.get("schema") != "heir.refined_model.v1":
        raise ValueError("refined checkpoint schema is invalid")
    if int(metadata.get("seed", -1)) != SEED:
        raise ValueError("refined checkpoint seed is not 17")
    if int(metadata.get("refinement_round", -1)) != selected:
        raise ValueError("refined checkpoint and audit selected rounds differ")
    if int(metadata.get("refinement_rounds_executed", -1)) != len(rounds):
        raise ValueError("refined checkpoint has a stale executed-round count")
    if metadata.get("refinement_rounds") != rounds:
        raise ValueError("refined checkpoint and audit trajectories differ")
    if metadata.get("refinement_stopped_reason") != audit.get("stopped_reason"):
        raise ValueError("refined checkpoint and audit stop reasons differ")
    try:
        losses_match = np.isclose(
            float(metadata["refinement_round_zero_validation_loss"]),
            float(audit["round_zero_validation_loss"]),
            rtol=1.0e-9,
            atol=1.0e-9,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("refinement audit has invalid round-zero loss metadata") from error
    if not bool(losses_match):
        raise ValueError("refined checkpoint and audit round-zero losses differ")
    return selected


def _unknown_mass_metadata_binding(
    metadata: Mapping[str, Any],
    mass: float,
    *,
    label: str,
) -> str:
    """Require the checkpoint itself to serialize the fixed unknown-mass recipe."""

    has_mass = "uot_unknown_mass" in metadata
    has_mode = "uot_unknown_mass_mode" in metadata
    if not has_mass and not has_mode:
        raise ValueError(
            "%s checkpoint lacks serialized unknown-mass metadata; a post-hoc run "
            "manifest cannot prove the mass used to produce this endpoint" % label
        )
    if not has_mass or not has_mode:
        raise ValueError("%s checkpoint has partial unknown-mass metadata" % label)
    try:
        matches = np.isclose(
            float(metadata["uot_unknown_mass"]),
            float(mass),
            rtol=0.0,
            atol=1.0e-12,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("%s checkpoint has invalid unknown-mass metadata" % label) from error
    if not bool(matches) or metadata["uot_unknown_mass_mode"] != "fixed":
        raise ValueError(
            "%s checkpoint unknown-mass metadata differs from its run manifest" % label
        )
    return "checkpoint_and_manifest_bound"


def _validate_lineage(
    *,
    sample: str,
    mass: float,
    artifact_root: Path,
    repository: Path,
    sample_inputs: SampleInputs,
    run_binding: Mapping[str, Any],
    molecular_generation: str,
) -> Tuple[
    PredictionBundle,
    Dict[str, Any],
    ArtifactRequest,
    PredictionBundle,
    Dict[str, Any],
    ArtifactRequest,
]:
    """Load and bind round-zero and refined endpoints to one manifested run case."""

    round0, refined = _directories(artifact_root, sample, mass)
    refined_prediction_path = refined / "predictions.npz"
    refined_telemetry_path = refined / "prediction.telemetry.json"
    round0_prediction_path = round0 / "predictions.npz"
    round0_telemetry_path = round0 / "prediction.telemetry.json"
    audit_path = refined / "refinement.json"
    checkpoint_path = refined / "heir_refined.pt"
    parent_path = round0 / "heir.pt"
    view_path = round0 / "refinement_views.npz"
    prototype_path = refined / "prototypes" / ("%s__%s.npz" % (sample, sample))
    native_prototype_path = (
        repository
        / "artifacts"
        / "snpatho"
        / (molecular_generation + "_scanvi")
        / sample
        / "prototypes_rare_complete.npz"
    )
    histology_path = repository / "artifacts" / "snpatho" / sample / "histology_full.npz"
    ood_path = repository / "artifacts" / "snpatho" / sample / "ood_target_calibrated.npz"
    validated_hashes = run_binding["validated_output_hashes"]

    file_hashes = {
        "refined_prediction": _require_file(
            refined_prediction_path, "refined prediction", validated_hashes
        ),
        "refined_telemetry": _require_file(
            refined_telemetry_path, "refined prediction telemetry", validated_hashes
        ),
        "round0_prediction": _require_file(
            round0_prediction_path, "round-zero prediction", validated_hashes
        ),
        "round0_telemetry": _require_file(
            round0_telemetry_path, "round-zero prediction telemetry", validated_hashes
        ),
        "refinement_audit": _require_file(audit_path, "refinement audit", validated_hashes),
        "checkpoint": _require_file(checkpoint_path, "refined checkpoint", validated_hashes),
        "parent_checkpoint": _require_file(parent_path, "round-zero checkpoint", validated_hashes),
        "refinement_views": _require_file(view_path, "refinement views", validated_hashes),
        "prototype": _require_file(prototype_path, "refined prototype", validated_hashes),
        "native_prototype": _require_file(native_prototype_path, "native prototype"),
        "histology": _require_file(histology_path, "histology input"),
        "ood": _require_file(ood_path, "OOD input"),
    }
    audit = _json_object(audit_path, "refinement audit")
    metadata = _checkpoint_metadata(checkpoint_path)
    parent_metadata = _checkpoint_metadata(parent_path)
    if parent_metadata.get("schema") != "heir.trained_model.v1":
        raise ValueError("round-zero checkpoint schema is invalid")
    selected_round = _validate_round_audit(audit, metadata)
    metadata_binding = {
        "round0": _unknown_mass_metadata_binding(parent_metadata, mass, label="round-zero"),
        "refined": _unknown_mass_metadata_binding(metadata, mass, label="refined"),
    }

    label = _mass_label(mass)
    refined_request = ArtifactRequest(
        sample=sample,
        seed=SEED,
        variant="unknown_mass_%s_refined" % label,
        family="unknown_mass_sensitivity",
        prediction=refined_prediction_path,
        telemetry=refined_telemetry_path,
        expected_round=selected_round,
    )
    round0_request = ArtifactRequest(
        sample=sample,
        seed=SEED,
        variant="unknown_mass_%s_round0" % label,
        family="unknown_mass_sensitivity",
        prediction=round0_prediction_path,
        telemetry=round0_telemetry_path,
        expected_round=0,
    )
    refined_prediction, refined_provenance = load_prediction(
        refined_request,
        sample_inputs,
        wrong_donor_source="",
    )
    round0_prediction, round0_provenance = load_prediction(
        round0_request,
        sample_inputs,
        wrong_donor_source="",
    )
    for endpoint, prediction, telemetry_path, prediction_path in (
        ("refined", refined_prediction, refined_telemetry_path, refined_prediction_path),
        ("round-zero", round0_prediction, round0_telemetry_path, round0_prediction_path),
    ):
        telemetry = _json_object(telemetry_path, "%s prediction telemetry" % endpoint)
        if _resolved_path(telemetry.get("prediction_path"), "telemetry prediction") != (
            prediction_path.resolve()
        ):
            raise ValueError("%s prediction telemetry points to a different output" % endpoint)
        if int(telemetry.get("genes", -1)) != len(prediction.gene_names):
            raise ValueError("%s prediction telemetry gene count is stale" % endpoint)

    prototype_rows = audit.get("prototype_artifacts")
    prototype_key = "%s::%s" % (sample, sample)
    if not isinstance(prototype_rows, Mapping) or prototype_key not in prototype_rows:
        raise ValueError("refinement audit lacks its specimen prototype")
    if _resolved_path(prototype_rows[prototype_key], "audit prototype") != (
        prototype_path.resolve()
    ):
        raise ValueError("refinement audit points to a different prototype")
    prototype = PrototypeSet.load_npz(prototype_path)
    if prototype.donor_id != sample or set(prototype.sample_ids.tolist()) != {sample}:
        raise ValueError("refined prototype specimen provenance is stale")
    if prototype.latent_space_id != sample_inputs.latent_space_id:
        raise ValueError("refined prototype latent space differs from native scANVI")

    if _resolved_path(metadata.get("parent_checkpoint"), "parent checkpoint") != (
        parent_path.resolve()
    ):
        raise ValueError("refined checkpoint points to a different parent")
    if metadata.get("parent_checkpoint_sha256") != file_hashes["parent_checkpoint"]:
        raise ValueError("refined checkpoint parent SHA-256 is stale")
    view_rows = metadata.get("refinement_view_artifacts")
    if (
        not isinstance(view_rows, list)
        or len(view_rows) != 1
        or not isinstance(view_rows[0], Mapping)
    ):
        raise ValueError("refined checkpoint must bind exactly one refinement view")
    if _resolved_path(view_rows[0].get("path"), "refinement view") != view_path.resolve():
        raise ValueError("refined checkpoint points to a different refinement view")
    if view_rows[0].get("sha256") != file_hashes["refinement_views"]:
        raise ValueError("refinement-view SHA-256 is stale")

    endpoint_hashes = {
        "refined": {
            "checkpoint_sha256": file_hashes["checkpoint"],
            "prototype_sha256": file_hashes["prototype"],
        },
        "round-zero": {
            "checkpoint_sha256": file_hashes["parent_checkpoint"],
            "prototype_sha256": file_hashes["native_prototype"],
        },
    }
    for endpoint, prediction in (
        ("refined", refined_prediction),
        ("round-zero", round0_prediction),
    ):
        expected_prediction_hashes = {
            **endpoint_hashes[endpoint],
            "histology_sha256": file_hashes["histology"],
            "ood_sha256": file_hashes["ood"],
        }
        for field, expected in expected_prediction_hashes.items():
            if getattr(prediction, field) != expected:
                raise ValueError(
                    "%s PredictionBundle %s does not bind the current artifact" % (endpoint, field)
                )
    if tuple(str(value) for value in metadata.get("gene_names", ())) != tuple(
        refined_prediction.gene_names.tolist()
    ):
        raise ValueError("refined checkpoint and prediction gene orders differ")
    if metadata.get("latent_space_id") != refined_prediction.latent_space_id:
        raise ValueError("refined checkpoint and prediction latent spaces differ")
    if metadata.get("expression_space_id") != refined_prediction.expression_space_id:
        raise ValueError("refined checkpoint and prediction expression spaces differ")

    stage_lookup = run_binding["stage_lookup"]
    endpoint_stage_rows = {
        "round0": stage_lookup[(sample, mass, "predict_round0")],
        "refined": stage_lookup[(sample, mass, "predict_refined")],
    }
    manifest_provenance = {
        endpoint: {
            "run_manifest_stage_bound": True,
            "run_manifest_stage_index": int(row["stage_index"]),
            "run_manifest_stage_status": str(row["status"]),
            "run_manifest_command_sha256": _RUNNER._canonical_sha256(row["command"]),
        }
        for endpoint, row in endpoint_stage_rows.items()
    }
    shared_provenance = {
        "run_manifest_path": run_binding["path"],
        "run_manifest_sha256": run_binding["sha256"],
        "unknown_mass": float(mass),
        "unknown_mass_metadata_binding": metadata_binding,
        "histology_path": str(histology_path.resolve()),
        "histology_sha256": file_hashes["histology"],
        "ood_path": str(ood_path.resolve()),
        "ood_sha256": file_hashes["ood"],
    }
    refined_provenance = {
        **dict(refined_provenance),
        **shared_provenance,
        **manifest_provenance["refined"],
        "refinement_audit_path": str(audit_path.resolve()),
        "refinement_audit_sha256": file_hashes["refinement_audit"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "parent_checkpoint_path": str(parent_path.resolve()),
        "parent_checkpoint_sha256": file_hashes["parent_checkpoint"],
        "refinement_views_path": str(view_path.resolve()),
        "refinement_views_sha256": file_hashes["refinement_views"],
        "prototype_path": str(prototype_path.resolve()),
        "selected_round": selected_round,
        "rounds_executed": len(audit["rounds"]),
        "stopped_reason": audit.get("stopped_reason"),
    }
    round0_provenance = {
        **dict(round0_provenance),
        **shared_provenance,
        **manifest_provenance["round0"],
        "checkpoint_path": str(parent_path.resolve()),
        "prototype_path": str(native_prototype_path.resolve()),
    }
    return (
        refined_prediction,
        refined_provenance,
        refined_request,
        round0_prediction,
        round0_provenance,
        round0_request,
    )


def _endpoint_diagnostics(prediction: PredictionBundle) -> Dict[str, float]:
    public = prediction.public_cell_expression_mean
    available = np.isfinite(public).all(axis=1)
    return {
        "abstention_fraction": float(np.mean(np.asarray(prediction.abstain, dtype=bool))),
        "public_expression_coverage": float(np.mean(available)),
        "mean_unknown_probability": float(
            np.mean(np.asarray(prediction.unknown_probability, dtype=np.float64))
        ),
    }


def _compact_case(
    refined_scored: Mapping[str, Any],
    round0_scored: Mapping[str, Any],
    refined_prediction: PredictionBundle,
    round0_prediction: PredictionBundle,
    mass: float,
    practical_delta_threshold: float,
) -> Dict[str, Any]:
    """Compact two fully scored endpoints after constructing paired per-gene deltas."""

    methods = {
        method: {
            metric: refined_scored["methods"][method]["summary"].get(metric)
            for metric in COMPACT_METRICS
        }
        for method in (METHOD, HARD_BASELINE, SOFT_BASELINE)
    }
    round0_metrics = {
        metric: round0_scored["methods"][METHOD]["summary"].get(metric)
        for metric in COMPACT_METRICS
    }
    refined_minus_round0 = _paired_spearman_delta(
        refined_scored["methods"][METHOD],
        round0_scored["methods"][METHOD],
    )
    deltas = {
        "refined_minus_round0": dict(refined_minus_round0["summary"]),
        "heir_minus_hard_baseline": dict(
            refined_scored["paired_gene_spearman_deltas"]["heir_minus_hard_baseline"]["summary"]
        ),
        "heir_minus_soft_baseline": dict(
            refined_scored["paired_gene_spearman_deltas"]["heir_minus_soft_baseline"]["summary"]
        ),
    }
    if methods[METHOD]["median_gene_spearman"] is None or any(
        deltas[name]["median_delta"] is None for name in CONTRASTS
    ):
        raise ValueError("case lacks an evaluable median gene-Spearman conclusion")
    for delta in deltas.values():
        value = float(delta["median_delta"])
        delta["practical_status"] = _practical_delta_status(
            value,
            practical_delta_threshold,
        )
        delta["raw_sign_status"] = _raw_sign_status(value)
        delta["practical_delta_threshold"] = practical_delta_threshold
    return {
        "case_id": refined_scored["case_id"],
        "sample": refined_scored["sample"],
        "seed": refined_scored["seed"],
        "unknown_mass": float(mass),
        "unknown_mass_label": _mass_label(mass),
        "refinement_round": refined_scored["refinement_round"],
        "prediction": refined_scored["prediction"],
        "aggregation": refined_scored["aggregation"],
        "endpoints": {
            "round0": {
                "case_id": round0_scored["case_id"],
                "refinement_round": round0_scored["refinement_round"],
                "prediction": round0_scored["prediction"],
                "metrics": round0_metrics,
                "diagnostics": _endpoint_diagnostics(round0_prediction),
            },
            "refined": {
                "case_id": refined_scored["case_id"],
                "refinement_round": refined_scored["refinement_round"],
                "prediction": refined_scored["prediction"],
                "metrics": methods[METHOD],
                "diagnostics": _endpoint_diagnostics(refined_prediction),
            },
        },
        "metrics": methods,
        "paired_gene_spearman_deltas": deltas,
        "practical_delta_threshold": practical_delta_threshold,
    }


def _direction(value: float) -> str:
    if value > _DIRECTION_EPSILON:
        return "positive"
    if value < -_DIRECTION_EPSILON:
        return "negative"
    return "tie"


def _stability(
    cases: Sequence[Mapping[str, Any]],
    samples: Sequence[str],
    practical_delta_threshold: float = DEFAULT_PRACTICAL_DELTA_THRESHOLD,
) -> Dict[str, Any]:
    expected = len(samples) * len(UNKNOWN_MASSES)
    if len(cases) != expected:
        return {
            "status": "blocked",
            "practical_status_stable_across_masses": None,
            "direction_stable_across_masses": None,
            "refined_beats_round0_at_every_mass": None,
            "heir_beats_both_baselines_at_every_mass": None,
            "refined_beats_all_comparators_at_every_mass": None,
            "per_sample": {},
            "policy": (
                "blocked unless all 15 canonical cases validate; otherwise compare the signs "
                "and practical pass/tie/fail classifications of paired median per-gene "
                "Spearman deltas at all five masses"
            ),
            "practical_delta_threshold": practical_delta_threshold,
        }
    per_sample: Dict[str, Any] = {}
    for sample in samples:
        rows = sorted(
            (row for row in cases if row["sample"] == sample),
            key=lambda row: float(row["unknown_mass"]),
        )
        if [float(row["unknown_mass"]) for row in rows] != list(UNKNOWN_MASSES):
            raise ValueError("validated case grid is incomplete or duplicated for %s" % sample)
        directions: Dict[str, Dict[str, str]] = {}
        practical_statuses: Dict[str, Dict[str, str]] = {}
        contrast_stability: Dict[str, bool] = {}
        practical_contrast_stability: Dict[str, bool] = {}
        for name in CONTRASTS:
            values = {
                row["unknown_mass_label"]: _direction(
                    float(row["paired_gene_spearman_deltas"][name]["median_delta"])
                )
                for row in rows
            }
            directions[name] = values
            contrast_stability[name] = len(set(values.values())) == 1
            practical_values = {
                row["unknown_mass_label"]: _practical_delta_status(
                    float(row["paired_gene_spearman_deltas"][name]["median_delta"]),
                    practical_delta_threshold,
                )
                for row in rows
            }
            practical_statuses[name] = practical_values
            practical_contrast_stability[name] = len(set(practical_values.values())) == 1
        refined_beats_round0 = all(
            _practical_delta_status(
                float(row["paired_gene_spearman_deltas"]["refined_minus_round0"]["median_delta"]),
                practical_delta_threshold,
            )
            == "pass"
            for row in rows
        )
        beats_both = all(
            all(
                _practical_delta_status(
                    float(row["paired_gene_spearman_deltas"][name]["median_delta"]),
                    practical_delta_threshold,
                )
                == "pass"
                for name in ("heir_minus_hard_baseline", "heir_minus_soft_baseline")
            )
            for row in rows
        )
        per_sample[sample] = {
            "direction_by_mass": directions,
            "practical_status_by_mass": practical_statuses,
            "contrast_direction_stable": contrast_stability,
            "contrast_practical_status_stable": practical_contrast_stability,
            "direction_stable_across_masses": all(contrast_stability.values()),
            "practical_status_stable_across_masses": all(practical_contrast_stability.values()),
            "refined_beats_round0_at_every_mass": refined_beats_round0,
            "heir_beats_both_baselines_at_every_mass": beats_both,
            "refined_beats_all_comparators_at_every_mass": (refined_beats_round0 and beats_both),
            "selected_rounds_by_mass": {
                row["unknown_mass_label"]: int(row["refinement_round"]) for row in rows
            },
        }
    raw_stable = all(row["direction_stable_across_masses"] for row in per_sample.values())
    stable = all(row["practical_status_stable_across_masses"] for row in per_sample.values())
    refined_beats_round0 = all(
        row["refined_beats_round0_at_every_mass"] for row in per_sample.values()
    )
    beats_both = all(row["heir_beats_both_baselines_at_every_mass"] for row in per_sample.values())
    return {
        "status": "stable" if stable else "unstable",
        "practical_status_stable_across_masses": stable,
        "direction_stable_across_masses": raw_stable,
        "refined_beats_round0_at_every_mass": refined_beats_round0,
        "heir_beats_both_baselines_at_every_mass": beats_both,
        "refined_beats_all_comparators_at_every_mass": refined_beats_round0 and beats_both,
        "per_sample": per_sample,
        "direction_epsilon": _DIRECTION_EPSILON,
        "practical_delta_threshold": practical_delta_threshold,
        "policy": (
            "For each specimen and comparator (round zero, hard baseline, and soft "
            "baseline), practical pass/tie/fail status under the prespecified delta margin "
            "must be identical at 0, 0.01, 0.05, 0.10, and 0.20. Raw sign stability is "
            "reported separately and does not imply benefit; benefit requires practical-pass "
            "status for refined-minus-round0 and both refined-minus-baseline deltas at every "
            "mass."
        ),
    }


def _blocker(sample: str, mass: float, error: Exception) -> Dict[str, Any]:
    return {
        "code": (
            "missing_unknown_mass_artifact"
            if isinstance(error, FileNotFoundError)
            else "invalid_unknown_mass_artifact"
        ),
        "sample": sample,
        "seed": SEED,
        "unknown_mass": float(mass),
        "unknown_mass_label": _mass_label(mass),
        "message": str(error),
    }


def evaluate_unknown_mass(
    *,
    repository: Path,
    artifact_root: Path,
    run_manifest_path: Path,
    truth_manifest_path: Path,
    native_manifest_path: Path,
    samples: Sequence[str] = DEFAULT_SAMPLES,
    molecular_generation: str = "r1",
    minimum_nuclei: int = 3,
    practical_delta_threshold: float = DEFAULT_PRACTICAL_DELTA_THRESHOLD,
) -> Dict[str, Any]:
    """Validate and score the complete fixed unknown-mass grid."""

    repository = repository.expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()
    run_manifest_path = run_manifest_path.expanduser().resolve()
    truth_manifest_path = truth_manifest_path.expanduser().resolve()
    native_manifest_path = native_manifest_path.expanduser().resolve()
    samples = tuple(dict.fromkeys(str(sample) for sample in samples))
    if not samples:
        raise ValueError("at least one sample is required")
    if molecular_generation not in _RUNNER.MOLECULAR_GENERATIONS:
        raise ValueError("molecular_generation must be r1 or r2")
    practical_delta_threshold = float(practical_delta_threshold)
    if not np.isfinite(practical_delta_threshold) or practical_delta_threshold < 0:
        raise ValueError("practical_delta_threshold must be finite and non-negative")
    run_binding = _validate_run_manifest(
        run_manifest_path,
        repository=repository,
        artifact_root=artifact_root,
        samples=samples,
        molecular_generation=molecular_generation,
    )
    truth_manifest = _json_object(truth_manifest_path, "frozen truth manifest")
    native_manifest = _json_object(native_manifest_path, "native scANVI manifest")
    if _MATRIX._native_molecular_generation(native_manifest) != molecular_generation:
        raise ValueError("native scANVI manifest differs from requested molecular generation")

    blockers = []
    sample_inputs: Dict[str, SampleInputs] = {}
    for sample in samples:
        try:
            sample_inputs[sample] = load_sample_inputs(
                sample=sample,
                truth_manifest_path=truth_manifest_path,
                truth_manifest=truth_manifest,
                native_manifest_path=native_manifest_path,
                native_manifest=native_manifest,
                repository=repository,
            )
        except (OSError, TypeError, ValueError) as error:
            blockers.append(
                {
                    "code": "invalid_sample_inputs",
                    "sample": sample,
                    "seed": SEED,
                    "unknown_mass": None,
                    "unknown_mass_label": None,
                    "message": str(error),
                }
            )

    cases = []
    for sample in samples:
        inputs = sample_inputs.get(sample)
        if inputs is None:
            continue
        for mass in UNKNOWN_MASSES:
            try:
                (
                    refined_prediction,
                    refined_provenance,
                    refined_request,
                    round0_prediction,
                    round0_provenance,
                    round0_request,
                ) = _validate_lineage(
                    sample=sample,
                    mass=mass,
                    artifact_root=artifact_root,
                    repository=repository,
                    sample_inputs=inputs,
                    run_binding=run_binding,
                    molecular_generation=molecular_generation,
                )
                refined_scored = score_prediction(
                    refined_request,
                    refined_prediction,
                    inputs,
                    refined_provenance,
                    minimum_nuclei=minimum_nuclei,
                )
                round0_scored = score_prediction(
                    round0_request,
                    round0_prediction,
                    inputs,
                    round0_provenance,
                    minimum_nuclei=minimum_nuclei,
                )
                cases.append(
                    _compact_case(
                        refined_scored,
                        round0_scored,
                        refined_prediction,
                        round0_prediction,
                        mass,
                        practical_delta_threshold,
                    )
                )
            except (OSError, TypeError, ValueError) as error:
                blockers.append(_blocker(sample, mass, error))

    stability = (
        _stability(cases, samples, practical_delta_threshold)
        if not blockers
        else _stability((), samples, practical_delta_threshold)
    )
    return {
        "schema": REPORT_SCHEMA,
        "requirement": "unknown_mass_sweep",
        "contract": {
            "requirement": "unknown_mass_sweep",
            "samples": list(samples),
            "seed": SEED,
            "molecular_generation": molecular_generation,
            "unknown_masses": list(UNKNOWN_MASSES),
            "practical_delta_threshold": practical_delta_threshold,
            "expected_case_count": len(samples) * len(UNKNOWN_MASSES),
            "expected_prediction_count": len(samples) * len(UNKNOWN_MASSES) * 2,
        },
        "status": "blocked" if blockers else "complete",
        "analysis_role": "native_scanvi_published_integrated_annotation_sensitivity",
        "molecular_generation": molecular_generation,
        "request": {
            "samples": list(samples),
            "seed": SEED,
            "molecular_generation": molecular_generation,
            "artifact_root": str(artifact_root),
            "unknown_masses": list(UNKNOWN_MASSES),
            "practical_delta_threshold": practical_delta_threshold,
            "minimum_nuclei": minimum_nuclei,
            "expected_case_count": len(samples) * len(UNKNOWN_MASSES),
            "expected_prediction_count": len(samples) * len(UNKNOWN_MASSES) * 2,
        },
        "manifests": {
            "unknown_mass_run": {
                "path": run_binding["path"],
                "sha256": run_binding["sha256"],
                "schema": run_binding["payload"]["schema"],
                "plan_sha256": run_binding["payload"]["plan_sha256"],
                "execution_mode": run_binding["payload"]["execution_mode"],
                "validated_stage_count": len(run_binding["payload"]["stages"]),
            },
            "frozen_truth": {
                "path": str(truth_manifest_path),
                "sha256": sha256_file(truth_manifest_path),
            },
            "native_scanvi": {
                "path": str(native_manifest_path),
                "sha256": sha256_file(native_manifest_path),
                "molecular_generation": molecular_generation,
            },
        },
        "annotation_provenance": native_manifest.get("annotation_provenance"),
        "scorer_source_identity": _scorer_source_identity(),
        "mass_binding": (
            "each round-zero and refined checkpoint must serialize the exact fixed "
            "uot_unknown_mass and uot_unknown_mass_mode=fixed; the canonical run manifest "
            "additionally binds the recipe commands and output SHA-256 values"
        ),
        "scored_case_count": len(cases),
        "scored_prediction_count": len(cases) * 2,
        "practical_delta_threshold": practical_delta_threshold,
        "blockers": blockers,
        "cases": cases,
        "stability": stability,
    }


def _tsv(report: Mapping[str, Any]) -> str:
    columns = (
        "sample",
        "seed",
        "unknown_mass",
        "unknown_mass_label",
        "selected_round",
        "method",
        "median_gene_spearman",
        "paired_median_gene_spearman_delta",
        "direction",
        "raw_sign_status",
        "practical_status",
        "practical_delta_threshold",
        "report_status",
    )
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for case in report["cases"]:
        for method, contrast, metric_source, selected_round in (
            (
                "heir_round0",
                "refined_minus_round0",
                case["endpoints"]["round0"]["metrics"],
                0,
            ),
            (
                "heir_refined",
                None,
                case["endpoints"]["refined"]["metrics"],
                case["refinement_round"],
            ),
            (HARD_BASELINE, "heir_minus_hard_baseline", None, None),
            (SOFT_BASELINE, "heir_minus_soft_baseline", None, None),
        ):
            if method in {HARD_BASELINE, SOFT_BASELINE}:
                metric_source = case["metrics"][method]
                selected_round = case["refinement_round"]
            delta = (
                None
                if contrast is None
                else case["paired_gene_spearman_deltas"][contrast]["median_delta"]
            )
            writer.writerow(
                {
                    "sample": case["sample"],
                    "seed": case["seed"],
                    "unknown_mass": case["unknown_mass"],
                    "unknown_mass_label": case["unknown_mass_label"],
                    "selected_round": selected_round,
                    "method": method,
                    "median_gene_spearman": metric_source["median_gene_spearman"],
                    "paired_median_gene_spearman_delta": "" if delta is None else delta,
                    "direction": "" if delta is None else _direction(float(delta)),
                    "raw_sign_status": (
                        ""
                        if delta is None
                        else case["paired_gene_spearman_deltas"][contrast]["raw_sign_status"]
                    ),
                    "practical_status": (
                        ""
                        if delta is None
                        else case["paired_gene_spearman_deltas"][contrast]["practical_status"]
                    ),
                    "practical_delta_threshold": report["practical_delta_threshold"],
                    "report_status": report["status"],
                }
            )
    return handle.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    stability = report["stability"]
    lines = [
        "# snPATHO unknown-mass sensitivity",
        "",
        "Evaluation status: **%s**. Stability conclusion: **%s**."
        % (report["status"], stability["status"]),
        "",
        "Scored %d of %d required seed-17 cases."
        % (report["scored_case_count"], report["request"]["expected_case_count"]),
        "",
        "Practical paired-delta threshold: **%.6f**; raw signs are retained separately."
        % report["practical_delta_threshold"],
        "",
    ]
    if report["blockers"]:
        lines.extend(("## Blockers", ""))
        lines.extend(
            "- `%s` / mass `%s`: %s" % (row["sample"], row["unknown_mass_label"], row["message"])
            for row in report["blockers"]
        )
        lines.append("")
    lines.extend(
        (
            "## Compact results",
            "",
            "| Sample | Mass | Round | Round0 rho | Refined rho | "
            "Refined-round0 | Status | Refined-hard | Status | Refined-soft | Status |",
            "|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|",
        )
    )
    for case in report["cases"]:
        lines.append(
            "| %s | %.2f | %d | %.6f | %.6f | %.6f | %s | %.6f | %s | %.6f | %s |"
            % (
                case["sample"],
                case["unknown_mass"],
                case["refinement_round"],
                case["endpoints"]["round0"]["metrics"]["median_gene_spearman"],
                case["metrics"][METHOD]["median_gene_spearman"],
                case["paired_gene_spearman_deltas"]["refined_minus_round0"]["median_delta"],
                case["paired_gene_spearman_deltas"]["refined_minus_round0"]["practical_status"],
                case["paired_gene_spearman_deltas"]["heir_minus_hard_baseline"]["median_delta"],
                case["paired_gene_spearman_deltas"]["heir_minus_hard_baseline"]["practical_status"],
                case["paired_gene_spearman_deltas"]["heir_minus_soft_baseline"]["median_delta"],
                case["paired_gene_spearman_deltas"]["heir_minus_soft_baseline"]["practical_status"],
            )
        )
    lines.extend(
        (
            "",
            "## Endpoint behavior",
            "",
            "| Sample | Mass | R0 abstain | Refined abstain | R0 public expr. | "
            "Refined public expr. | R0 mean unknown | Refined mean unknown |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for case in report["cases"]:
        round0 = case["endpoints"]["round0"]["diagnostics"]
        refined = case["endpoints"]["refined"]["diagnostics"]
        lines.append(
            "| %s | %.2f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |"
            % (
                case["sample"],
                case["unknown_mass"],
                round0["abstention_fraction"],
                refined["abstention_fraction"],
                round0["public_expression_coverage"],
                refined["public_expression_coverage"],
                round0["mean_unknown_probability"],
                refined["mean_unknown_probability"],
            )
        )
    lines.extend(("", "## Stability interpretation", "", stability["policy"], ""))
    if stability["status"] != "blocked":
        lines.append(
            "Practical status stable across masses: **%s**; raw direction stable across "
            "masses: **%s**; refined beats round zero by the practical margin at every "
            "mass: **%s**; refined beats both baselines at every mass: **%s**."
            % (
                str(stability["practical_status_stable_across_masses"]).lower(),
                str(stability["direction_stable_across_masses"]).lower(),
                str(stability["refined_beats_round0_at_every_mass"]).lower(),
                str(stability["heir_beats_both_baselines_at_every_mass"]).lower(),
            )
        )
        lines.append("")
    return "\n".join(lines)


def _atomic_write_texts(outputs: Mapping[Path, str]) -> None:
    temporary: Dict[Path, str] = {}
    try:
        for destination, content in outputs.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=destination.name + ".",
                suffix=".tmp",
                dir=str(destination.parent),
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary[destination] = temporary_path
        for destination, temporary_path in temporary.items():
            os.replace(temporary_path, destination)
    finally:
        for temporary_path in temporary.values():
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def write_report(
    report: Mapping[str, Any],
    *,
    json_output: Path,
    tsv_output: Path,
    markdown_output: Path,
    input_paths: Sequence[Path],
) -> None:
    destinations = tuple(
        path.expanduser().resolve() for path in (json_output, tsv_output, markdown_output)
    )
    reject_output_input_collisions(
        destinations,
        input_paths,
        label="snPATHO unknown-mass report",
    )
    _atomic_write_texts(
        {
            destinations[0]: json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            destinations[1]: _tsv(report),
            destinations[2]: _markdown(report),
        }
    )


def _arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument(
        "--molecular-generation",
        choices=_RUNNER.MOLECULAR_GENERATIONS,
        default="r2",
        help="R2 preserves specimen biology; use r1 only for historical reproduction",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--truth-manifest",
        type=Path,
        default=repository
        / "artifacts"
        / "snpatho"
        / "orchestration_v0_2"
        / "benchmark_plan.all.json",
    )
    parser.add_argument(
        "--native-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=None,
        help="required hash-bound manifest emitted by the hardened sensitivity runner",
    )
    parser.add_argument("--minimum-nuclei", type=int, default=3)
    parser.add_argument(
        "--practical-delta-threshold",
        type=float,
        default=DEFAULT_PRACTICAL_DELTA_THRESHOLD,
        help="Prespecified |paired median gene-Spearman delta| practical margin",
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _arguments(argv)
    repository = args.repository.expanduser().resolve()
    artifact_root = (
        repository / "artifacts" / "snpatho" / (args.molecular_generation + "_scanvi")
        if args.artifact_root is None
        else args.artifact_root
    )
    native_manifest = (
        repository
        / (
            "reports/snpatho_scanvi_r1_manifest.json"
            if args.molecular_generation == "r1"
            else "artifacts/snpatho/r2_scanvi/native_manifest.json"
        )
        if args.native_manifest is None
        else args.native_manifest
    )
    run_manifest = (
        repository
        / "artifacts"
        / "snpatho"
        / (
            "unknown_mass_sensitivity_v1"
            if args.molecular_generation == "r1"
            else "unknown_mass_sensitivity_r2_v1"
        )
        / "run_manifest.json"
        if args.run_manifest is None
        else args.run_manifest
    )
    report = evaluate_unknown_mass(
        repository=repository,
        artifact_root=artifact_root,
        run_manifest_path=run_manifest,
        truth_manifest_path=args.truth_manifest,
        native_manifest_path=native_manifest,
        samples=DEFAULT_SAMPLES,
        molecular_generation=args.molecular_generation,
        minimum_nuclei=args.minimum_nuclei,
        practical_delta_threshold=args.practical_delta_threshold,
    )
    write_report(
        report,
        json_output=args.json_output,
        tsv_output=args.tsv_output,
        markdown_output=args.markdown_output,
        input_paths=_MATRIX._bound_input_paths(
            report,
            repository=repository,
            manifest_paths=(run_manifest, args.truth_manifest, native_manifest),
        ),
    )
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
