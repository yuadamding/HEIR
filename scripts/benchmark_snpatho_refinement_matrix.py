#!/usr/bin/env python3
"""Score the native-scANVI snPATHO refinement and negative-control matrix.

This evaluator is deliberately separate from the historical DeepBench report.  It
discovers the canonical native refinement outputs, validates every prediction
against its inference telemetry checksum, validates the frozen Visium truth and
native scANVI reference against their manifests, and then evaluates every artifact
that is present.  Requested artifacts that are absent or invalid are recorded as
blockers and can never produce a passing strict-ordering result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from heir.data import PrototypeSet, RNAReference, SpatialTruthArtifact
from heir.evaluation.deepbench import (
    _cell_rna_mass,
    _hard_assigned_cell_rna_mass,
    _soft_type_mean_cells,
    _type_mean_cells,
    aggregate_cells_to_spots,
    deepbench_expression_metrics,
)
from heir.inference import PredictionBundle, validate_wrong_donor_prototype_filter
from heir.utils import reject_output_input_collisions, sha256_file

REPORT_SCHEMA = "heir.snpatho_refinement_matrix.v1"
REFINEMENT_RUN_MANIFEST_SCHEMA = "heir.snpatho_refinement_run_manifest.v2"
TRUTH_MANIFEST_SCHEMA = "heir.snpatho_benchmark_plan.v1"
NATIVE_MANIFEST_SCHEMAS = {
    "r1": "heir.snpatho_scanvi_r1_manifest.v1",
    "r2": "heir.snpatho_scanvi_r2_manifest.v1",
}
NATIVE_MANIFEST_SCHEMA = NATIVE_MANIFEST_SCHEMAS["r1"]
DEFAULT_SAMPLES = ("4066", "4399", "4411")
DEFAULT_SAMPLE_SITES = {
    "4066": "primary_breast",
    "4399": "liver_metastasis",
    "4411": "liver_metastasis",
}
DEFAULT_SEEDS = (17, 41, 89, 131, 197)
DEFAULT_CONTROL_SEEDS = (17, 41, 89)
DEFAULT_PRACTICAL_DELTA_THRESHOLD = 0.002
DEFAULT_CONTROLS = (
    "round0_prototype_only",
    "refined_prototype_only",
    "image_shuffle",
    "graph_shuffle",
    "no_graph",
    "wrong_prototype_bank",
)
LEGACY_CONTROL_ALIASES = {
    "prototype_only": "refined_prototype_only",
    "wrong_donor": "wrong_prototype_bank",
}
CONTROL_TELEMETRY_KEYS = {
    "round0_prototype_only": "prototype_only",
    "refined_prototype_only": "prototype_only",
    "image_shuffle": "image_feature_shuffle",
    "graph_shuffle": "graph_node_shuffle",
    "no_graph": "no_graph",
    "wrong_prototype_bank": "wrong_donor",
}
METHOD = "heir_native_scanvi_library_size_weighted"
HARD_BASELINE = "matched_native_scanvi_hard_type_mean_hard_assigned_mass"
SOFT_BASELINE = "matched_native_scanvi_soft_type_mean_expected_mass"
SUMMARY_METRIC = "median_gene_spearman"
EVIDENCE_MANIFEST_SCHEMA = "heir.snpatho_refinement_matrix_evidence.v1"
EVIDENCE_ARTIFACT_SCHEMAS = {
    "generic_atlas": "heir.snpatho_generic_atlas_control.v1",
    "label_permutation": "heir.snpatho_label_permutation_control.v1",
    "state_omission": "heir.snpatho_state_omission_sensitivity.v1",
    "reference_downsampling": "heir.snpatho_reference_downsampling_sensitivity.v1",
    "unknown_mass_sweep": "heir.snpatho_unknown_mass_sensitivity.v1",
    "clean_independent_reannotation": "heir.snpatho_clean_independent_reannotation.v1",
    "untouched_external_cohort": "heir.snpatho_untouched_external_cohort.v1",
}
EVIDENCE_REQUIREMENTS = {
    "generic_atlas": (
        "The generic-atlas RNA control requested by the benchmark plan is unavailable."
    ),
    "label_permutation": (
        "The label-permutation negative control requested by the benchmark plan is unavailable."
    ),
    "state_omission": (
        "The state-omission sensitivity requested by the benchmark plan is unavailable."
    ),
    "reference_downsampling": (
        "The 1,000/2,500/5,000/all-cell reference-downsampling sensitivity is unavailable."
    ),
    "unknown_mass_sweep": (
        "The prespecified 0/0.01/0.05/0.10/0.20 unknown-mass sweep is unavailable."
    ),
    "clean_independent_reannotation": (
        "Native scANVI still uses published integrated annotations rather than an independent "
        "clean reannotation."
    ),
    "untouched_external_cohort": (
        "No untouched external cohort is available for confirmatory validation."
    ),
}
UNKNOWN_MASS_EVIDENCE_SAMPLES = ("4066", "4399", "4411")
UNKNOWN_MASS_EVIDENCE_SEED = 17
UNKNOWN_MASS_EVIDENCE_VALUES = (0.0, 0.01, 0.05, 0.10, 0.20)
FOLLOWUP_EVIDENCE_SEEDS = DEFAULT_SEEDS
REFERENCE_DOWNSAMPLING_SIZES = (1000, 2500, 5000, "all")


@dataclass(frozen=True)
class ArtifactRequest:
    """One canonical prediction artifact that the matrix requests."""

    sample: str
    seed: int
    variant: str
    family: str
    prediction: Path
    telemetry: Path
    expected_round: int
    control: Optional[str] = None
    prototype_donor_id: Optional[str] = None
    prototype_source: Optional[Path] = None

    @property
    def case_id(self) -> str:
        return "%s/seed%d/%s" % (self.sample, self.seed, self.variant)


@dataclass(frozen=True)
class SampleInputs:
    """Hash-validated truth and native scANVI reference for one specimen."""

    sample: str
    truth_path: Path
    truth_sha256: str
    truth: SpatialTruthArtifact
    reference_path: Path
    reference_sha256: str
    reference: RNAReference
    latent_space_id: str
    expression_space_id: str


@dataclass(frozen=True)
class RunManifestValidation:
    """Validated execution/adoption lineage for the requested score matrix."""

    path: Path
    sha256: str
    execution_provenance_verified: bool
    execution_transform_hash_verified: bool
    original_execution_source_verified: bool
    execution_mode: str
    manifest_role: str
    stage_count: int
    request_stages: Mapping[str, Mapping[str, Any]]


def _json_object(
    path: Path,
    name: str,
    *,
    schema: Optional[str] = None,
) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError("%s is absent: %s" % (name, path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("%s is not valid JSON: %s" % (name, path)) from error
    if not isinstance(payload, Mapping):
        raise ValueError("%s must contain a JSON object" % name)
    if schema is not None and payload.get("schema") != schema:
        raise ValueError("%s schema is not %s" % (name, schema))
    return payload


def _require_digest(value: object, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("%s must be a lowercase SHA-256 digest" % name)
    return digest


def _native_molecular_generation(payload: Mapping[str, Any]) -> str:
    schema = str(payload.get("schema", ""))
    for generation, expected in NATIVE_MANIFEST_SCHEMAS.items():
        if schema == expected:
            declared = str(payload.get("molecular_generation", generation))
            if declared != generation:
                raise ValueError("native scANVI manifest generation disagrees with its schema")
            return generation
    raise ValueError("native scANVI manifest schema is invalid")


def _resolve_manifest_path(raw: object, manifest: Path, repository: Path) -> Path:
    value = str(raw).strip()
    if not value:
        raise ValueError("manifest artifact path cannot be blank")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = ((manifest.parent / path).resolve(), (repository / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _validated_file(path: Path, expected: object, name: str) -> str:
    digest = _require_digest(expected, name + "_sha256")
    if not path.is_file():
        raise ValueError("%s is absent: %s" % (name, path))
    observed = sha256_file(path)
    if observed != digest:
        raise ValueError("%s SHA-256 mismatch: expected %s, observed %s" % (name, digest, observed))
    return observed


def _truth_rows(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    if payload.get("schema_version") != TRUTH_MANIFEST_SCHEMA:
        raise ValueError("truth manifest schema is not %s" % TRUTH_MANIFEST_SCHEMA)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("truth manifest cases must be a list")
    rows: Dict[str, Mapping[str, Any]] = {}
    for row in cases:
        if not isinstance(row, Mapping):
            raise ValueError("truth manifest case must be an object")
        sample = str(row.get("section_id", "")).strip()
        if not sample or sample in rows:
            raise ValueError("truth manifest section IDs must be non-empty and unique")
        rows[sample] = row
    return rows


def load_sample_inputs(
    *,
    sample: str,
    truth_manifest_path: Path,
    truth_manifest: Mapping[str, Any],
    native_manifest_path: Path,
    native_manifest: Mapping[str, Any],
    repository: Path,
) -> SampleInputs:
    """Load and hash-bind one frozen truth/native-reference pair."""

    truth_row = _truth_rows(truth_manifest).get(sample)
    if truth_row is None:
        raise ValueError("truth manifest has no case for %s" % sample)
    specimens = native_manifest.get("specimens")
    molecular_generation = _native_molecular_generation(native_manifest)
    if not isinstance(specimens, Mapping):
        raise ValueError("native manifest schema/specimens are invalid")
    specimen = specimens.get(sample)
    if not isinstance(specimen, Mapping):
        raise ValueError("native manifest has no specimen for %s" % sample)

    truth_path = _resolve_manifest_path(truth_row.get("truth"), truth_manifest_path, repository)
    truth_digest = _validated_file(
        truth_path,
        truth_row.get("truth_sha256"),
        "frozen truth %s" % sample,
    )
    reference_path = _resolve_manifest_path(
        specimen.get("latent_reference"), native_manifest_path, repository
    )
    reference_digest = _validated_file(
        reference_path,
        specimen.get("latent_reference_sha256"),
        "native %s reference %s" % (molecular_generation.upper(), sample),
    )
    truth = SpatialTruthArtifact.from_npz(truth_path)
    reference = RNAReference.load_npz(reference_path)
    latent_space_id = str(native_manifest.get("latent_space_id", ""))
    expression_space_id = str(native_manifest.get("expression_space_id", ""))
    if not latent_space_id or reference.latent_space_id != latent_space_id:
        raise ValueError("native scANVI reference latent space differs from native manifest")
    if truth.expression_space_id != expression_space_id:
        raise ValueError("frozen truth expression space differs from native manifest")
    if truth.section_id != sample or truth.specimen_id != sample:
        raise ValueError("frozen truth identity differs from requested specimen")
    if reference.sample_id != sample:
        raise ValueError("native scANVI reference identity differs from requested specimen")
    if set(np.asarray(reference.sample_ids).astype(str).tolist()) != {sample}:
        raise ValueError("native scANVI reference sample IDs differ from requested specimen")
    if set(np.asarray(reference.donor_ids).astype(str).tolist()) != {sample}:
        raise ValueError("native scANVI reference donor IDs differ from requested specimen")
    if tuple(np.asarray(reference.gene_ids).astype(str)) != tuple(
        np.asarray(truth.gene_names).astype(str)
    ):
        raise ValueError("native scANVI reference and frozen truth gene orders differ")
    return SampleInputs(
        sample=sample,
        truth_path=truth_path,
        truth_sha256=truth_digest,
        truth=truth,
        reference_path=reference_path,
        reference_sha256=reference_digest,
        reference=reference,
        latent_space_id=latent_space_id,
        expression_space_id=expression_space_id,
    )


def build_requests(
    *,
    artifact_root: Path,
    samples: Sequence[str],
    seeds: Sequence[int],
    trajectory_seed: int,
    controls: Sequence[str],
    control_seeds: Sequence[int],
    wrong_donor_pairings: Optional[Sequence[Tuple[str, str]]] = None,
    wrong_donor_target: Optional[str] = None,
    wrong_donor_source: Optional[str] = None,
) -> Tuple[ArtifactRequest, ...]:
    """Build the prespecified primary, trajectory, and control matrix."""

    controls = tuple(LEGACY_CONTROL_ALIASES.get(control, control) for control in controls)
    if len(set(controls)) != len(controls):
        raise ValueError("controls contain duplicate canonical cases after legacy alias expansion")
    unsupported = sorted(set(controls) - set(DEFAULT_CONTROLS))
    if unsupported:
        raise ValueError("unsupported controls: %s" % ", ".join(unsupported))
    if wrong_donor_pairings is not None and (
        wrong_donor_target is not None or wrong_donor_source is not None
    ):
        raise ValueError(
            "supply wrong-prototype-bank pairings or a legacy single pairing, not both"
        )
    if (wrong_donor_target is None) != (wrong_donor_source is None):
        raise ValueError("legacy wrong-donor target/source aliases must be supplied together")
    if wrong_donor_pairings is None:
        if wrong_donor_target is not None and wrong_donor_source is not None:
            pairings = ((str(wrong_donor_target), str(wrong_donor_source)),)
        else:
            pairings = tuple(
                (target, source) for target in samples for source in samples if source != target
            )
    else:
        pairings = tuple((str(target), str(source)) for target, source in wrong_donor_pairings)
    if len(set(pairings)) != len(pairings):
        raise ValueError("wrong-prototype-bank pairings must be unique")
    sample_set = set(samples)
    if any(target not in sample_set or target == source for target, source in pairings):
        raise ValueError(
            "wrong-prototype-bank pairings require requested targets and non-self sources"
        )
    donor_sources = {
        target: tuple(source for candidate, source in pairings if candidate == target)
        for target in samples
    }
    requests = []
    for sample in samples:
        root = artifact_root / sample
        for seed in seeds:
            round0 = root / ("model_refinement_r1_v1_seed%d_round0" % seed)
            refined = root / ("model_refinement_r1_v1_seed%d_refined" % seed)
            requests.extend(
                (
                    ArtifactRequest(
                        sample,
                        seed,
                        "round0",
                        "primary",
                        round0 / "predictions.npz",
                        round0 / "prediction.telemetry.json",
                        0,
                    ),
                    ArtifactRequest(
                        sample,
                        seed,
                        "refined",
                        "primary",
                        refined / "predictions.npz",
                        refined / "prediction.telemetry.json",
                        4,
                    ),
                )
            )
        refined = root / ("model_refinement_r1_v1_seed%d_refined" % trajectory_seed)
        for round_id in (1, 2, 3):
            directory = refined / ("round_%d" % round_id)
            requests.append(
                ArtifactRequest(
                    sample,
                    trajectory_seed,
                    "round%d" % round_id,
                    "trajectory",
                    directory / "predictions.npz",
                    directory / "prediction.telemetry.json",
                    round_id,
                )
            )
        for seed in control_seeds:
            refined = root / ("model_refinement_r1_v1_seed%d_refined" % seed)
            for control in controls:
                if control == "wrong_prototype_bank":
                    for source in donor_sources[sample]:
                        directory = refined / ("control_wrong_donor_%s" % source)
                        requests.append(
                            ArtifactRequest(
                                sample,
                                seed,
                                "wrong_prototype_bank_" + source,
                                "control",
                                directory / "predictions.npz",
                                directory / "prediction.telemetry.json",
                                4,
                                control,
                                source,
                                artifact_root
                                / source
                                / ("model_refinement_r1_v1_seed%d_refined" % seed)
                                / "prototypes"
                                / ("%s__%s.npz" % (source, source)),
                            )
                        )
                    continue
                if control == "round0_prototype_only":
                    directory = root / ("model_refinement_r1_v1_seed%d_round0" % seed)
                    directory = directory / "control_prototype_only"
                    expected_round = 0
                elif control == "refined_prototype_only":
                    directory = refined / "control_prototype_only"
                    expected_round = 4
                else:
                    directory = refined / ("control_" + control)
                    expected_round = 4
                requests.append(
                    ArtifactRequest(
                        sample,
                        seed,
                        control,
                        "control",
                        directory / "predictions.npz",
                        directory / "prediction.telemetry.json",
                        expected_round,
                        control,
                    )
                )
    identifiers = [request.case_id for request in requests]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("matrix request contains duplicate case identifiers")
    return tuple(requests)


def _validate_telemetry(
    request: ArtifactRequest,
    prediction_sha256: str,
    prediction: PredictionBundle,
) -> Tuple[Mapping[str, Any], str]:
    telemetry = _json_object(request.telemetry, "prediction telemetry")
    telemetry_digest = sha256_file(request.telemetry)
    if telemetry.get("schema") != "heir.inference_telemetry.v1":
        raise ValueError("prediction telemetry schema is invalid")
    if telemetry.get("prediction_sha256") != prediction_sha256:
        raise ValueError("prediction SHA-256 does not match inference telemetry")
    if int(telemetry.get("nuclei", -1)) != len(prediction.nucleus_ids):
        raise ValueError("prediction telemetry nucleus count is misaligned")
    negative = telemetry.get("negative_control")
    if not isinstance(negative, Mapping):
        raise ValueError("prediction telemetry has no negative-control audit")
    expected_control = request.control
    expected_telemetry_key = (
        None if expected_control is None else CONTROL_TELEMETRY_KEYS[expected_control]
    )
    for key in set(CONTROL_TELEMETRY_KEYS.values()):
        expected = key == expected_telemetry_key
        if negative.get(key) is not expected:
            raise ValueError(
                "prediction telemetry negative-control flag %s is not %s" % (key, expected)
            )
    if int(negative.get("seed", -1)) != request.seed:
        raise ValueError("prediction telemetry seed differs from requested seed")
    prototype_donor = str(negative.get("prototype_donor_id", ""))
    if request.control == "wrong_prototype_bank":
        if (
            request.prototype_donor_id is None
            or prototype_donor != request.prototype_donor_id
            or prototype_donor == request.sample
        ):
            raise ValueError(
                "wrong-prototype-bank telemetry does not identify the requested source"
            )
        if request.prototype_source is None:
            raise ValueError("wrong-prototype-bank request has no resolvable source prototype bank")
        source_digest = sha256_file(request.prototype_source)
        if prediction.prototype_sha256 != source_digest:
            raise ValueError("wrong-prototype-bank prediction is not bound to its full source bank")
        source_prototypes = PrototypeSet.load_npz(request.prototype_source)
        if not source_prototypes.donor_id:
            raise ValueError("wrong-prototype-bank source lacks donor provenance")
        if source_prototypes.donor_id != request.prototype_donor_id:
            raise ValueError("wrong-prototype-bank source identity is stale")
        validate_wrong_donor_prototype_filter(
            source_prototypes,
            prediction.type_names.tolist(),
            prediction.prototype_ids.tolist(),
            negative.get("prototype_filter"),
            source_sha256=source_digest,
        )
    elif prototype_donor != request.sample:
        raise ValueError("matched prediction telemetry has a non-matched prototype donor")
    elif negative.get("prototype_filter") is not None:
        raise ValueError("matched prediction telemetry unexpectedly reports prototype filtering")
    return telemetry, telemetry_digest


def _load_current_runner(repository: Path) -> Any:
    runner_path = repository / "scripts" / "run_snpatho_refinement_benchmark.py"
    if not runner_path.is_file():
        raise ValueError("refinement runner source is absent: %s" % runner_path)
    module_name = "_heir_refinement_runner_manifest_validation"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the current refinement runner source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_true_loo_molecular_folds(
    repository: Path,
    specifications: Sequence[str],
    *,
    required_samples: Sequence[str],
) -> Mapping[str, Any]:
    """Load the runner's target-specific fold map and recheck scorer identities."""

    repository = repository.expanduser().resolve()
    runner = _load_current_runner(repository)
    folds = runner.load_true_loo_molecular_folds(
        repository,
        specifications,
        required_samples=required_samples,
    )
    for sample, fold in folds.items():
        native = _json_object(
            fold.native_manifest,
            "%s native true-LOO manifest" % sample,
            schema=NATIVE_MANIFEST_SCHEMAS["r2"],
        )
        if _native_molecular_generation(native) != "r2":
            raise ValueError("true-LOO native manifest must use molecular generation r2")
        if native.get("latent_space_id") != fold.latent_space_id:
            raise ValueError("true-LOO scorer latent identity differs for %s" % sample)
        decoder = native.get("distilled_decoder")
        if not isinstance(decoder, Mapping) or decoder.get("sha256") != fold.decoder_sha256:
            raise ValueError("true-LOO scorer decoder identity differs for %s" % sample)
        decoder_name = Path(str(decoder.get("external_path", ""))).expanduser()
        decoder_path = (
            decoder_name.resolve()
            if decoder_name.is_absolute()
            else (repository / decoder_name).resolve()
        )
        if decoder_path != fold.decoder or sha256_file(decoder_path) != fold.decoder_sha256:
            raise ValueError("true-LOO scorer decoder file differs for %s" % sample)
    return folds


def _manifest_artifact_rows(
    value: object,
    *,
    manifest_path: Path,
    expected: Sequence[Tuple[str, Path]],
    label: str,
) -> Dict[str, Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != len(expected):
        raise ValueError("%s artifact inventory does not match the current exact plan" % label)
    result: Dict[str, Mapping[str, Any]] = {}
    for index, ((expected_role, expected_path), raw) in enumerate(zip(expected, value)):
        if not isinstance(raw, Mapping):
            raise ValueError("%s artifact %d must be an object" % (label, index))
        role = str(raw.get("role", ""))
        if role != expected_role or role in result:
            raise ValueError("%s artifact roles differ from the current exact plan" % label)
        path = _resolve_manifest_path(raw.get("path"), manifest_path, manifest_path.parent)
        expected_path = expected_path.expanduser().resolve()
        if path != expected_path:
            raise ValueError("%s %s path differs from the current exact plan" % (label, role))
        digest = _validated_file(path, raw.get("sha256"), "%s %s" % (label, role))
        result[role] = {"path": str(path), "sha256": digest}
    return result


def _prediction_stage_name(request: ArtifactRequest) -> str:
    if request.variant == "round0":
        return "predict_round0"
    if request.variant == "refined":
        return "predict_refined"
    if request.variant in {"round1", "round2", "round3"}:
        return "predict_" + request.variant
    if request.control == "wrong_prototype_bank":
        if request.prototype_donor_id is None:
            raise ValueError("wrong-prototype-bank request has no prototype source")
        return "wrong_prototype_bank_" + request.prototype_donor_id
    return request.variant


def validate_refinement_run_manifest(
    path: Path,
    *,
    repository: Path,
    native_manifest_path: Optional[Path],
    native_manifest: Optional[Mapping[str, Any]],
    requests: Sequence[ArtifactRequest],
    artifact_root: Optional[Path] = None,
    molecular_folds: Optional[Mapping[str, Any]] = None,
) -> RunManifestValidation:
    """Validate the exact run plan, every artifact hash, and recursive stage lineage."""

    repository = repository.expanduser().resolve()
    path = path.expanduser().resolve()
    payload = _json_object(path, "refinement run/adoption manifest")
    if payload.get("schema") != REFINEMENT_RUN_MANIFEST_SCHEMA:
        raise ValueError(
            "refinement run manifest schema is not %s" % REFINEMENT_RUN_MANIFEST_SCHEMA
        )
    runner = _load_current_runner(repository)
    fold_lookup = {} if molecular_folds is None else dict(molecular_folds)
    molecular_generation = str(payload.get("molecular_generation", "r1"))
    fold_native: Dict[str, Mapping[str, Any]] = {}
    if fold_lookup:
        if native_manifest_path is not None or native_manifest is not None:
            raise ValueError("true-LOO scoring cannot also use one shared native manifest")
        if molecular_generation != "r2" or set(fold_lookup) != set(runner.SAMPLES):
            raise ValueError("refinement run true-LOO fold coverage is invalid")
        for sample, fold in fold_lookup.items():
            fold_native[sample] = _json_object(
                fold.native_manifest,
                "%s native true-LOO manifest" % sample,
                schema=NATIVE_MANIFEST_SCHEMAS["r2"],
            )
        expected_bundle_sha256 = runner._canonical_sha256(
            {sample: fold_lookup[sample].native_manifest_sha256 for sample in runner.SAMPLES}
        )
        if (
            payload.get("native_scanvi_manifest_sha256") is not None
            or payload.get("native_scanvi_fold_bundle_sha256") != expected_bundle_sha256
            or payload.get("latent_space_id") is not None
            or payload.get("latent_space_id_by_sample")
            != {sample: fold_lookup[sample].latent_space_id for sample in runner.SAMPLES}
        ):
            raise ValueError("refinement run is not bound to the exact true-LOO fold map")
        recorded_folds = payload.get("native_scanvi_fold_manifests")
        if not isinstance(recorded_folds, Mapping) or set(recorded_folds) != set(fold_lookup):
            raise ValueError("refinement run true-LOO fold manifest inventory is incomplete")
        for sample, fold in fold_lookup.items():
            recorded = recorded_folds[sample]
            if not isinstance(recorded, Mapping) or any(
                (
                    recorded.get("held_out_sample") != sample,
                    recorded.get("training_donors") != list(fold.training_donors),
                    recorded.get("latent_space_id") != fold.latent_space_id,
                    recorded.get("preparation_manifest_sha256") != fold.preparation_manifest_sha256,
                    recorded.get("native_manifest_sha256") != fold.native_manifest_sha256,
                    recorded.get("decoder_sha256") != fold.decoder_sha256,
                )
            ):
                raise ValueError("refinement run true-LOO fold identity differs for %s" % sample)
        claim_scope = payload.get("claim_scope")
        required_reasons = {
            "uninitialized_morphology_negative_control",
            "live_student_e_step_negative_control",
        }
        if (
            payload.get("negative_control") is not True
            or not isinstance(claim_scope, Mapping)
            or claim_scope.get("eligible_for_primary_performance_claims") is not False
            or not required_reasons.issubset(set(claim_scope.get("reasons", ())))
        ):
            raise ValueError("true-LOO run manifest does not preserve negative-control claim scope")
    else:
        if native_manifest_path is None or native_manifest is None:
            raise ValueError("shared native-manifest scoring inputs are absent")
        if payload.get("native_scanvi_manifest_sha256") != sha256_file(native_manifest_path):
            raise ValueError("refinement run manifest is not bound to the native scANVI manifest")
        if molecular_generation != _native_molecular_generation(native_manifest):
            raise ValueError("refinement run molecular generation differs from native scANVI")

    current_source = runner.refinement_run_source_identity(repository)
    recorded_source = payload.get("validation_recipe_source_identity")
    if not isinstance(recorded_source, Mapping) or recorded_source != current_source:
        raise ValueError(
            "refinement run validation-recipe source identity differs from current source"
        )
    current_cli_source_binding = runner._heir_source_binding(
        repository,
        require_sources=True,
    )
    validation_cli_source_binding = payload.get(
        "validation_cli_source_binding",
        payload.get("cli_source_binding"),
    )
    if (
        not isinstance(validation_cli_source_binding, Mapping)
        or validation_cli_source_binding != current_cli_source_binding
    ):
        raise ValueError("refinement run validation CLI identity differs from current source")
    execution_source_identity = payload.get("execution_source_identity")
    execution_cli_source_binding = payload.get("cli_source_binding")
    execution_source_captured = bool(
        isinstance(execution_source_identity, Mapping)
        and isinstance(execution_cli_source_binding, Mapping)
    )
    execution_source_continuous = bool(
        execution_source_captured
        and execution_source_identity == recorded_source
        and execution_cli_source_binding == validation_cli_source_binding
    )
    expected_stages = runner.build_plan(
        repository,
        samples=runner.SAMPLES,
        seeds=runner.SEEDS,
        controls=True,
        artifact_root=artifact_root,
        molecular_generation=molecular_generation,
        molecular_folds=fold_lookup or None,
    )
    expected_plan = runner.full_matrix_plan_payload(
        expected_stages,
        molecular_generation=molecular_generation,
        molecular_folds=fold_lookup or None,
    )
    if payload.get("plan_sha256") != runner._canonical_sha256(expected_plan):
        raise ValueError("refinement run manifest plan SHA-256 differs from the current exact plan")
    expected_plan_header = {key: value for key, value in expected_plan.items() if key != "stages"}
    if payload.get("plan") != expected_plan_header:
        raise ValueError("refinement run manifest plan header differs from the current exact plan")
    rows = payload.get("stages")
    if not isinstance(rows, list) or len(rows) != len(expected_stages):
        raise ValueError("refinement run manifest stage coverage is incomplete")
    if payload.get("stage_count") != len(rows):
        raise ValueError("refinement run manifest stage_count is inconsistent")

    stages_by_key: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    completed = 0
    adopted = 0
    control_transform_verified = []
    for expected, planned, raw in zip(expected_stages, expected_plan["stages"], rows):
        if not isinstance(raw, Mapping):
            raise ValueError("refinement run manifest stage must be an object")
        for name in (
            "stage_index",
            "stage_id",
            "sample",
            "seed",
            "stage",
            "control",
            "prototype_donor_id",
            "command",
            "command_sha256",
            "deterministic_transform_recipe",
        ):
            if raw.get(name) != planned[name]:
                raise ValueError(
                    "refinement run stage %s %s differs from current exact plan"
                    % (planned["stage_id"], name)
                )
        input_rows = _manifest_artifact_rows(
            raw.get("inputs"),
            manifest_path=path,
            expected=expected.inputs,
            label=planned["stage_id"] + " input",
        )
        output_rows = _manifest_artifact_rows(
            raw.get("outputs"),
            manifest_path=path,
            expected=tuple(zip(expected.output_roles, expected.outputs)),
            label=planned["stage_id"] + " output",
        )
        try:
            expected.validate()
        except Exception as error:
            raise ValueError(
                "current recursive validation failed for stage %s: %s"
                % (planned["stage_id"], error)
            ) from error
        runner_status = raw.get("runner_status")
        if runner_status == "completed":
            completed += 1
            expected_status = "completed_current_invocation"
            expected_original = execution_source_continuous
        elif runner_status == "skipped_valid":
            adopted += 1
            expected_status = "adopted_existing_output_after_current_validation"
            expected_original = False
        else:
            raise ValueError("refinement run stage has an invalid runner_status")
        if (
            raw.get("status") != expected_status
            or raw.get("current_recipe_validation") != "passed"
            or raw.get("original_execution_source_verified") is not expected_original
            or raw.get("artifact_identity_capture") != runner.STAGE_ARTIFACT_IDENTITY_CAPTURE
        ):
            raise ValueError("refinement run stage execution semantics are inconsistent")

        recipe = planned["deterministic_transform_recipe"]
        transform_verified = True
        if expected.control in {"image_shuffle", "graph_shuffle"}:
            telemetry = _json_object(
                Path(output_rows["telemetry"]["path"]),
                "refinement run control telemetry",
                schema="heir.inference_telemetry.v1",
            )
            negative = telemetry.get("negative_control")
            transform = negative.get("transform") if isinstance(negative, Mapping) else None
            transform_verified = bool(
                isinstance(transform, Mapping)
                and transform.get("recipe_sha256") == recipe["recipe_sha256"]
                and transform.get("map_sha256") == recipe["expected_transform_map_sha256"]
            )
        if raw.get("execution_transform_hash_verified") is not transform_verified:
            raise ValueError("refinement run stage transform-hash claim is inconsistent")
        if recipe is not None:
            control_transform_verified.append(transform_verified)

        stage_native = fold_native.get(expected.sample, native_manifest)
        if not isinstance(stage_native, Mapping):
            raise ValueError("native scANVI manifest is unavailable for %s" % expected.sample)
        native_specimens = stage_native.get("specimens")
        decoder = stage_native.get("distilled_decoder")
        if not isinstance(native_specimens, Mapping) or not isinstance(decoder, Mapping):
            raise ValueError("native scANVI manifest lacks recursive lineage artifacts")
        specimen = native_specimens.get(expected.sample)
        if not isinstance(specimen, Mapping):
            raise ValueError("native scANVI manifest lacks specimen %s" % expected.sample)
        if expected.name == "train_round0":
            if input_rows["rna_decoder"]["sha256"] != decoder.get("sha256"):
                raise ValueError("round-zero decoder lineage differs from native scANVI")
            if input_rows["residual_geometry"]["sha256"] != specimen.get(
                "residual_geometry_sha256"
            ):
                raise ValueError("round-zero residual geometry differs from native scANVI")
        if expected.name == "predict_round0" and input_rows["prototype"]["sha256"] != specimen.get(
            "rare_complete_prototypes_sha256"
        ):
            raise ValueError("round-zero native prototype differs from native scANVI")
        key = (expected.sample, expected.seed, expected.name)
        if key in stages_by_key:
            raise ValueError("refinement run manifest has duplicate stage identity")
        stages_by_key[key] = {
            "stage_id": planned["stage_id"],
            "inputs": input_rows,
            "outputs": output_rows,
            "runner_status": runner_status,
            "control": expected.control,
        }

    execution = payload.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("refinement run manifest execution summary must be an object")
    transform_verified = all(control_transform_verified) if control_transform_verified else True
    original_verified = completed == len(rows) and execution_source_continuous
    execution_verified = original_verified and transform_verified
    if execution.get("stage_status_counts") != {
        "completed": completed,
        "skipped_valid": adopted,
    }:
        raise ValueError("refinement run manifest execution counts are inconsistent")
    if (
        execution.get("posthoc_adoption_present") is not (adopted > 0)
        or execution.get("execution_source_identity_captured_before_stage_1")
        is not execution_source_captured
        or execution.get("execution_source_identity_unchanged") is not execution_source_continuous
        or execution.get("execution_cli_source_binding_unchanged")
        is not execution_source_continuous
        or execution.get("original_execution_source_verified") is not original_verified
        or execution.get("execution_transform_hash_verified") is not transform_verified
        or execution.get("stage_time_artifact_identities_complete") is not True
        or execution.get("execution_provenance_verified") is not execution_verified
        or execution.get("current_recipe_validation_complete") is not True
    ):
        raise ValueError("refinement run manifest aggregate provenance claims are inconsistent")

    request_stages = {}
    for request in requests:
        stage_name = _prediction_stage_name(request)
        stage = stages_by_key.get((request.sample, request.seed, stage_name))
        if stage is None:
            raise ValueError("refinement run manifest does not bind %s" % request.case_id)
        prediction = stage["outputs"]["prediction"]
        telemetry = stage["outputs"]["telemetry"]
        if (
            Path(prediction["path"]) != request.prediction.resolve()
            or Path(telemetry["path"]) != request.telemetry.resolve()
        ):
            raise ValueError("refinement run stage paths differ for %s" % request.case_id)
        request_stages[request.case_id] = stage
    return RunManifestValidation(
        path=path,
        sha256=sha256_file(path),
        execution_provenance_verified=execution_verified,
        execution_transform_hash_verified=transform_verified,
        original_execution_source_verified=original_verified,
        execution_mode=str(execution.get("execution_mode", "")),
        manifest_role=str(payload.get("manifest_role", "")),
        stage_count=len(rows),
        request_stages=request_stages,
    )


def load_prediction(
    request: ArtifactRequest,
    sample_inputs: SampleInputs,
    *,
    wrong_donor_source: Optional[str] = None,
    run_stage: Optional[Mapping[str, Any]] = None,
) -> Tuple[PredictionBundle, Dict[str, Any]]:
    """Load one prediction and enforce checksum, identity, and control provenance."""

    if not request.prediction.is_file() or not request.telemetry.is_file():
        missing = [
            str(path) for path in (request.prediction, request.telemetry) if not path.is_file()
        ]
        raise FileNotFoundError("missing requested artifact(s): %s" % ", ".join(missing))
    if request.control == "wrong_prototype_bank" and request.prototype_source is None:
        raise ValueError("wrong-prototype-bank request has no resolvable source bank")
    if request.prototype_source is not None and not request.prototype_source.is_file():
        raise FileNotFoundError(
            "missing requested wrong-prototype-bank source prototype: %s" % request.prototype_source
        )
    prediction_digest = sha256_file(request.prediction)
    prediction = PredictionBundle.from_npz(request.prediction)
    del wrong_donor_source
    telemetry, telemetry_digest = _validate_telemetry(request, prediction_digest, prediction)
    if run_stage is not None:
        outputs = run_stage.get("outputs")
        inputs = run_stage.get("inputs")
        if not isinstance(outputs, Mapping) or not isinstance(inputs, Mapping):
            raise ValueError("run-manifest predict-stage binding is malformed")
        if outputs.get("prediction", {}).get("sha256") != prediction_digest:
            raise ValueError("prediction SHA-256 differs from its run-manifest stage")
        if outputs.get("telemetry", {}).get("sha256") != telemetry_digest:
            raise ValueError("telemetry SHA-256 differs from its run-manifest stage")
        for field, role in (
            ("checkpoint_sha256", "checkpoint"),
            ("prototype_sha256", "prototype"),
            ("histology_sha256", "histology"),
            ("ood_sha256", "ood"),
        ):
            row = inputs.get(role)
            if not isinstance(row, Mapping) or getattr(prediction, field) != row.get("sha256"):
                raise ValueError(
                    "prediction %s differs from its run-manifest %s input" % (field, role)
                )
    if prediction.sample_id != request.sample or prediction.donor_id != request.sample:
        raise ValueError("prediction sample/donor identity differs from requested specimen")
    if int(prediction.inference_seed) != request.seed:
        raise ValueError("prediction inference seed differs from requested seed")
    if prediction.refinement_round != request.expected_round:
        raise ValueError("prediction refinement round differs from its matrix position")
    if prediction.latent_space_id != sample_inputs.latent_space_id:
        raise ValueError("prediction latent space differs from native scANVI reference")
    if prediction.expression_space_id != sample_inputs.expression_space_id:
        raise ValueError("prediction expression space differs from frozen truth")
    if not np.array_equal(prediction.nucleus_ids, sample_inputs.truth.nucleus_ids):
        raise ValueError("prediction nuclei/order differ from frozen truth")
    if not np.array_equal(prediction.gene_names, sample_inputs.truth.gene_names):
        raise ValueError("prediction genes/order differ from frozen truth")
    return prediction, {
        "path": str(request.prediction.resolve()),
        "sha256": prediction_digest,
        "telemetry_path": str(request.telemetry.resolve()),
        "telemetry_sha256": telemetry_digest,
        "telemetry_prediction_sha256_match": True,
        "checkpoint_sha256": prediction.checkpoint_sha256,
        "prototype_sha256": prediction.prototype_sha256,
        "negative_control": dict(telemetry["negative_control"]),
        "run_manifest_stage_id": (
            None if run_stage is None else str(run_stage.get("stage_id", ""))
        ),
        "run_manifest_stage_bound": run_stage is not None,
    }


def _primary_spots(truth: SpatialTruthArtifact, minimum_nuclei: int) -> np.ndarray:
    if minimum_nuclei <= 0:
        raise ValueError("minimum_nuclei must be positive")
    assigned = truth.nucleus_spot_index[truth.nucleus_spot_index >= 0]
    counts = np.bincount(assigned, minlength=len(truth.spot_ids))
    selected = counts >= minimum_nuclei
    if int(selected.sum()) < 3:
        raise ValueError("fewer than three frozen Visium spots meet the nucleus threshold")
    return selected


def _paired_spearman_delta(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    left_gene = left["per_gene"]
    right_gene = right["per_gene"]
    if left_gene["gene_names"] != right_gene["gene_names"]:
        raise ValueError("paired metrics have different gene orders")
    per_gene = []
    deltas = []
    for gene, left_value, right_value in zip(
        left_gene["gene_names"],
        left_gene["spearman"],
        right_gene["spearman"],
    ):
        delta = (
            None
            if left_value is None or right_value is None
            else float(left_value) - float(right_value)
        )
        per_gene.append(
            {
                "gene": str(gene),
                "left_spearman": left_value,
                "right_spearman": right_value,
                "delta": delta,
            }
        )
        if delta is not None:
            deltas.append(delta)
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "summary": {
            "evaluable_genes": int(len(values)),
            "median_delta": float(np.median(values)) if len(values) else None,
            "mean_delta": float(np.mean(values)) if len(values) else None,
            "positive_fraction": float(np.mean(values > 0)) if len(values) else None,
            "nonnegative_fraction": float(np.mean(values >= 0)) if len(values) else None,
        },
        "per_gene": per_gene,
    }


def score_prediction(
    request: ArtifactRequest,
    prediction: PredictionBundle,
    sample_inputs: SampleInputs,
    provenance: Mapping[str, Any],
    *,
    minimum_nuclei: int,
) -> Dict[str, Any]:
    """Score HEIR and two type-map-matched native scANVI baselines."""

    truth = sample_inputs.truth
    reference = sample_inputs.reference
    selected = _primary_spots(truth, minimum_nuclei)
    expected_mass = _cell_rna_mass(reference, prediction)
    hard_mass = _hard_assigned_cell_rna_mass(reference, prediction)
    hard_cells = _type_mean_cells(reference, prediction)
    soft_cells = _soft_type_mean_cells(reference, prediction)
    heir_spots, heir_mass = aggregate_cells_to_spots(
        prediction.internal_aggregate_expression_mean,
        truth.nucleus_spot_index,
        len(truth.spot_ids),
        expected_mass,
    )
    hard_spots, hard_spot_mass = aggregate_cells_to_spots(
        hard_cells,
        truth.nucleus_spot_index,
        len(truth.spot_ids),
        hard_mass,
    )
    soft_spots, soft_spot_mass = aggregate_cells_to_spots(
        soft_cells,
        truth.nucleus_spot_index,
        len(truth.spot_ids),
        expected_mass,
    )
    for name, mass in (
        (METHOD, heir_mass),
        (HARD_BASELINE, hard_spot_mass),
        (SOFT_BASELINE, soft_spot_mass),
    ):
        if np.any(mass[selected] <= 0):
            raise ValueError("%s has zero aggregation mass in an evaluated spot" % name)
    observed = truth.observed_expression[selected]
    coordinates = truth.spot_coordinates_px[selected]
    genes = truth.gene_names.tolist()
    metrics = {
        METHOD: deepbench_expression_metrics(heir_spots[selected], observed, coordinates, genes),
        HARD_BASELINE: deepbench_expression_metrics(
            hard_spots[selected], observed, coordinates, genes
        ),
        SOFT_BASELINE: deepbench_expression_metrics(
            soft_spots[selected], observed, coordinates, genes
        ),
    }
    return {
        "case_id": request.case_id,
        "sample": request.sample,
        "seed": request.seed,
        "variant": request.variant,
        "family": request.family,
        "refinement_round": request.expected_round,
        "control": request.control,
        "prototype_donor_id": request.prototype_donor_id,
        "prediction": dict(provenance),
        "aggregation": {
            "method": "native_scanvi_expected_type_median_library_size_mass",
            "hard_baseline": "native_scanvi_hard_type_median_library_size_mass",
            "soft_baseline": "native_scanvi_expected_type_median_library_size_mass",
            "type_map_source": "this_prediction_artifact",
            "minimum_nuclei_per_spot": minimum_nuclei,
            "spots_total": int(len(truth.spot_ids)),
            "spots_evaluated": int(selected.sum()),
            "nuclei_total": int(len(prediction.nucleus_ids)),
            "type_names": [str(value) for value in prediction.type_names.tolist()],
        },
        "methods": metrics,
        "paired_gene_spearman_deltas": {
            "heir_minus_hard_baseline": _paired_spearman_delta(
                metrics[METHOD], metrics[HARD_BASELINE]
            ),
            "heir_minus_soft_baseline": _paired_spearman_delta(
                metrics[METHOD], metrics[SOFT_BASELINE]
            ),
        },
    }


def _blocker(
    code: str,
    message: str,
    *,
    request: Optional[ArtifactRequest] = None,
    sample: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "case_id": None if request is None else request.case_id,
        "sample": sample if request is None else request.sample,
        "seed": None if request is None else request.seed,
        "variant": None if request is None else request.variant,
        "path": None if request is None else str(request.prediction.resolve()),
    }


def _require_evidence_sequence(
    value: object,
    expected: Sequence[object],
    name: str,
) -> None:
    if not isinstance(value, list) or value != list(expected):
        raise ValueError("%s must be exactly %s" % (name, list(expected)))


def _require_nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    return value.strip()


def _require_finite_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
    ):
        raise ValueError("%s must be a finite number" % name)
    return float(value)


def _canonical_evidence_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _validated_evidence_json(
    reference: object,
    name: str,
    *,
    report_path: Optional[Path],
    repository: Optional[Path],
    schema: str,
) -> Tuple[Mapping[str, Any], str, Path]:
    if report_path is None or repository is None:
        raise ValueError("%s requires evidence-report path context" % name)
    if not isinstance(reference, Mapping):
        raise ValueError("%s reference must be an object" % name)
    path = _resolve_manifest_path(reference.get("path"), report_path, repository)
    digest = _validated_file(path, reference.get("sha256"), name)
    payload = _json_object(path, name, schema=schema)
    return payload, digest, path


def _require_unique_text_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("%s must be a non-empty list" % name)
    values = [_require_nonempty_text(item, name + " entry") for item in value]
    if len(set(values)) != len(values):
        raise ValueError("%s entries must be unique" % name)
    return values


def _require_identity(
    payload: Mapping[str, Any],
    expected: Mapping[str, object],
    name: str,
) -> None:
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError("%s %s differs from its evidence case" % (name, field))


def _validate_followup_sample_seed_contract(
    contract: Mapping[str, Any],
    requirement: str,
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    _require_evidence_sequence(
        contract.get("samples"),
        UNKNOWN_MASS_EVIDENCE_SAMPLES,
        "%s contract samples" % requirement,
    )
    _require_evidence_sequence(
        contract.get("seeds"),
        FOLLOWUP_EVIDENCE_SEEDS,
        "%s contract seeds" % requirement,
    )
    return UNKNOWN_MASS_EVIDENCE_SAMPLES, FOLLOWUP_EVIDENCE_SEEDS


def _validate_evidence_case_grid(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    requirement: str,
    fields: Sequence[str],
    expected_grid: set,
) -> int:
    expected_count = len(expected_grid)
    if contract.get("expected_case_count") != expected_count:
        raise ValueError(
            "%s contract expected_case_count must be exactly %d" % (requirement, expected_count)
        )
    if payload.get("scored_case_count") != expected_count:
        raise ValueError("%s scored_case_count must be exactly %d" % (requirement, expected_count))
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_count:
        raise ValueError(
            "%s report must contain exactly %d covered cases" % (requirement, expected_count)
        )
    observed = set()
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError("%s case %d must be an object" % (requirement, index))
        identity = tuple(case.get(field) for field in fields)
        if identity not in expected_grid or identity in observed:
            raise ValueError(
                "%s case coverage is incomplete, duplicated, or unexpected" % requirement
            )
        observed.add(identity)
    if observed != expected_grid:
        raise ValueError("%s case coverage is incomplete" % requirement)
    return expected_count


def _validate_generic_atlas_evidence(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    report_path: Optional[Path],
    repository: Optional[Path],
) -> Dict[str, Any]:
    samples, seeds = _validate_followup_sample_seed_contract(contract, "generic_atlas")
    references = contract.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("generic_atlas contract references must be a non-empty list")
    reference_ids = []
    bindings: Dict[str, Dict[str, Any]] = {}
    validated_artifacts = 0
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise ValueError("generic_atlas reference %d must be an object" % index)
        reference_id = _require_nonempty_text(
            reference.get("reference_id"),
            "generic_atlas reference %d reference_id" % index,
        )
        if reference_id in reference_ids:
            raise ValueError("generic_atlas reference IDs must be unique")
        reference_ids.append(reference_id)
        reference_payload, reference_digest, _ = _validated_evidence_json(
            reference.get("reference_artifact"),
            "generic_atlas reference %s" % reference_id,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.generic_atlas_reference.v1",
        )
        _require_identity(
            reference_payload,
            {"reference_id": reference_id},
            "generic_atlas reference",
        )
        donor_ids = _require_unique_text_list(
            reference_payload.get("donor_ids"),
            "generic_atlas reference donor_ids",
        )
        if set(donor_ids) & set(samples):
            raise ValueError("generic_atlas reference donors overlap target donors")
        cell_count = reference_payload.get("cell_count")
        if isinstance(cell_count, bool) or not isinstance(cell_count, int) or cell_count <= 0:
            raise ValueError("generic_atlas reference cell_count must be positive")

        ontology_payload, ontology_digest, _ = _validated_evidence_json(
            reference.get("ontology_artifact"),
            "generic_atlas ontology %s" % reference_id,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.generic_atlas_ontology.v1",
        )
        _require_identity(
            ontology_payload,
            {"reference_id": reference_id},
            "generic_atlas ontology",
        )
        type_names = _require_unique_text_list(
            ontology_payload.get("type_names"),
            "generic_atlas ontology type_names",
        )

        prototype_payload, prototype_digest, _ = _validated_evidence_json(
            reference.get("prototype_artifact"),
            "generic_atlas prototype %s" % reference_id,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.generic_atlas_prototype.v1",
        )
        _require_identity(
            prototype_payload,
            {"reference_id": reference_id},
            "generic_atlas prototype",
        )
        if prototype_payload.get("donor_ids") != donor_ids:
            raise ValueError("generic_atlas prototype donor IDs differ from its reference")
        if prototype_payload.get("type_names") != type_names:
            raise ValueError("generic_atlas prototype type names differ from its ontology")
        prototype_count = prototype_payload.get("prototype_count")
        if (
            isinstance(prototype_count, bool)
            or not isinstance(prototype_count, int)
            or prototype_count < len(type_names)
        ):
            raise ValueError("generic_atlas prototype_count is incomplete")
        bindings[reference_id] = {
            "reference_sha256": reference_digest,
            "ontology_sha256": ontology_digest,
            "prototype_sha256": prototype_digest,
        }
        validated_artifacts += 3
    grid = {
        (sample, seed, reference_id)
        for sample in samples
        for seed in seeds
        for reference_id in reference_ids
    }
    case_count = _validate_evidence_case_grid(
        payload,
        contract,
        "generic_atlas",
        ("sample", "seed", "reference_id"),
        grid,
    )
    for index, case in enumerate(payload["cases"]):
        identity = {
            "sample": case["sample"],
            "seed": case["seed"],
            "reference_id": case["reference_id"],
        }
        binding = bindings[str(case["reference_id"])]
        prediction, prediction_digest, _ = _validated_evidence_json(
            case.get("prediction_artifact"),
            "generic_atlas case %d prediction" % index,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.generic_atlas_prediction.v1",
        )
        _require_identity(prediction, identity, "generic_atlas prediction")
        if prediction.get("status") != "complete":
            raise ValueError("generic_atlas prediction status must be complete")
        for field, digest in binding.items():
            if prediction.get(field) != digest:
                raise ValueError("generic_atlas prediction %s is stale" % field)
        score, _, _ = _validated_evidence_json(
            case.get("score_artifact"),
            "generic_atlas case %d score" % index,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.generic_atlas_score.v1",
        )
        _require_identity(score, identity, "generic_atlas score")
        if score.get("status") != "complete":
            raise ValueError("generic_atlas score status must be complete")
        if score.get("prediction_sha256") != prediction_digest:
            raise ValueError("generic_atlas score is not bound to its prediction")
        _require_nonempty_text(score.get("metric"), "generic_atlas score metric")
        _require_finite_number(score.get("statistic"), "generic_atlas score statistic")
        validated_artifacts += 2
    return {
        "reference_count": len(reference_ids),
        "validated_case_count": case_count,
        "validated_supporting_artifact_count": validated_artifacts,
    }


def _validate_label_permutation_evidence(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    report_path: Optional[Path],
    repository: Optional[Path],
) -> Dict[str, Any]:
    samples, seeds = _validate_followup_sample_seed_contract(contract, "label_permutation")
    permutations = contract.get("permutation_count")
    if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 100:
        raise ValueError("label_permutation permutation_count must be at least 100")
    if contract.get("draw_seed_scheme") != "sha256_label_permutation_v1":
        raise ValueError("label_permutation draw_seed_scheme is not prespecified")
    grid = {(sample, seed) for sample in samples for seed in seeds}
    case_count = _validate_evidence_case_grid(
        payload,
        contract,
        "label_permutation",
        ("sample", "seed"),
        grid,
    )
    observed_values = []
    null_values = []
    metric: Optional[str] = None
    validated_artifacts = 0
    for index, case in enumerate(payload["cases"]):
        identity = {"sample": case["sample"], "seed": case["seed"]}
        observed, _, _ = _validated_evidence_json(
            case.get("observed_score_artifact"),
            "label_permutation case %d observed score" % index,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.label_permutation_observed.v1",
        )
        _require_identity(observed, identity, "label_permutation observed score")
        observed_metric = _require_nonempty_text(
            observed.get("metric"),
            "label_permutation observed metric",
        )
        observed_values.append(
            _require_finite_number(
                observed.get("statistic"),
                "label_permutation observed statistic",
            )
        )
        if metric is None:
            metric = observed_metric
        elif metric != observed_metric:
            raise ValueError("label_permutation observed metrics are inconsistent")

        draws, _, _ = _validated_evidence_json(
            case.get("draw_manifest_artifact"),
            "label_permutation case %d draw manifest" % index,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.label_permutation_draws.v1",
        )
        _require_identity(draws, identity, "label_permutation draw manifest")
        if draws.get("metric") != metric:
            raise ValueError("label_permutation draw metric differs from observed metric")
        rows = draws.get("draws")
        if not isinstance(rows, list) or len(rows) != permutations:
            raise ValueError("label_permutation draw manifest has an incomplete draw count")
        draw_seeds = set()
        for draw_index, draw in enumerate(rows):
            if not isinstance(draw, Mapping) or draw.get("draw_index") != draw_index:
                raise ValueError("label_permutation draw indices must be complete and ordered")
            material = "%s|%s|%d|label_permutation_v1" % (
                case["sample"],
                case["seed"],
                draw_index,
            )
            expected_seed = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)
            draw_seed = draw.get("draw_seed")
            if draw_seed != expected_seed or draw_seed in draw_seeds:
                raise ValueError("label_permutation draw seed is stale or duplicated")
            draw_seeds.add(draw_seed)
            source = _require_unique_text_list(
                draw.get("source_labels"),
                "label_permutation source_labels",
            )
            permuted = _require_unique_text_list(
                draw.get("permuted_labels"),
                "label_permutation permuted_labels",
            )
            if len(source) < 2 or set(source) != set(permuted) or source == permuted:
                raise ValueError(
                    "label_permutation map must be a bijective nonidentity permutation"
                )
            map_payload = {"source_labels": source, "permuted_labels": permuted}
            if draw.get("map_sha256") != _canonical_evidence_sha256(map_payload):
                raise ValueError("label_permutation map SHA-256 is stale")
            null_values.append(
                _require_finite_number(
                    draw.get("statistic"),
                    "label_permutation draw statistic",
                )
            )
        validated_artifacts += 2

    summary = payload.get("null_result_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("label_permutation null_result_summary must be an object")
    if summary.get("permutation_count") != permutations:
        raise ValueError("label_permutation null summary permutation count is stale")
    if summary.get("case_count") != case_count or summary.get("draw_count") != len(null_values):
        raise ValueError("label_permutation null summary coverage is stale")
    if summary.get("metric") != metric:
        raise ValueError("label_permutation null summary metric is stale")
    reported_observed = _require_finite_number(
        summary.get("observed_statistic"),
        "label_permutation null summary observed_statistic",
    )
    reported_mean = _require_finite_number(
        summary.get("null_mean"),
        "label_permutation null summary null_mean",
    )
    reported_sd = _require_finite_number(
        summary.get("null_standard_deviation"),
        "label_permutation null summary null_standard_deviation",
    )
    if reported_sd < 0:
        raise ValueError("label_permutation null_standard_deviation cannot be negative")
    reported_p = _require_finite_number(
        summary.get("empirical_p_value"),
        "label_permutation null summary empirical_p_value",
    )
    if reported_p < 0 or reported_p > 1:
        raise ValueError("label_permutation empirical_p_value must be in [0, 1]")
    null_array = np.asarray(null_values, dtype=np.float64)
    observed_mean = float(np.mean(observed_values))
    null_mean = float(np.mean(null_array))
    null_sd = float(np.std(null_array))
    empirical_p = float(
        (1 + np.sum(np.abs(null_array) >= abs(observed_mean))) / (len(null_array) + 1)
    )
    for reported, expected, name in (
        (reported_observed, observed_mean, "observed_statistic"),
        (reported_mean, null_mean, "null_mean"),
        (reported_sd, null_sd, "null_standard_deviation"),
        (reported_p, empirical_p, "empirical_p_value"),
    ):
        if not np.isclose(reported, expected, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("label_permutation null summary %s is not reproduced" % name)
    return {
        "permutation_count": permutations,
        "null_metric": metric,
        "validated_case_count": case_count,
        "validated_draw_count": len(null_values),
        "validated_supporting_artifact_count": validated_artifacts,
    }


def _validate_state_omission_evidence(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    report_path: Optional[Path],
    repository: Optional[Path],
) -> Dict[str, Any]:
    samples, seeds = _validate_followup_sample_seed_contract(contract, "state_omission")
    raw_states = contract.get("omitted_states")
    if not isinstance(raw_states, list) or not raw_states:
        raise ValueError("state_omission omitted_states must be a non-empty list")
    states = [_require_nonempty_text(value, "state_omission omitted state") for value in raw_states]
    if len(set(states)) != len(states):
        raise ValueError("state_omission omitted_states must be unique")
    grid = {(sample, seed, state) for sample in samples for seed in seeds for state in states}
    case_count = _validate_evidence_case_grid(
        payload,
        contract,
        "state_omission",
        ("sample", "seed", "omitted_state"),
        grid,
    )
    validated_artifacts = 0
    for index, case in enumerate(payload["cases"]):
        identity = {
            "sample": case["sample"],
            "seed": case["seed"],
            "omitted_state": case["omitted_state"],
        }
        reference, reference_digest, _ = _validated_evidence_json(
            case.get("reference_artifact"),
            "state_omission case %d reference" % index,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.state_omission_reference.v1",
        )
        _require_identity(reference, identity, "state_omission reference")
        source_states = _require_unique_text_list(
            reference.get("source_states"),
            "state_omission source_states",
        )
        retained_states = _require_unique_text_list(
            reference.get("retained_states"),
            "state_omission retained_states",
        )
        omitted = str(case["omitted_state"])
        if omitted not in source_states or omitted in retained_states:
            raise ValueError("state_omission reference does not remove the omitted state")
        if set(retained_states) != set(source_states) - {omitted}:
            raise ValueError("state_omission reference retained-state set is incomplete")

        prototype, prototype_digest, _ = _validated_evidence_json(
            case.get("prototype_artifact"),
            "state_omission case %d prototype" % index,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.state_omission_prototype.v1",
        )
        _require_identity(prototype, identity, "state_omission prototype")
        if prototype.get("reference_sha256") != reference_digest:
            raise ValueError("state_omission prototype is not bound to its reference")
        if prototype.get("type_names") != retained_states or omitted in prototype.get(
            "type_names", []
        ):
            raise ValueError("state_omission prototype still contains the omitted state")

        prediction, prediction_digest, _ = _validated_evidence_json(
            case.get("prediction_artifact"),
            "state_omission case %d prediction" % index,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.state_omission_prediction.v1",
        )
        _require_identity(prediction, identity, "state_omission prediction")
        if prediction.get("status") != "complete":
            raise ValueError("state_omission prediction status must be complete")
        if (
            prediction.get("reference_sha256") != reference_digest
            or prediction.get("prototype_sha256") != prototype_digest
            or prediction.get("type_names") != retained_states
            or omitted in prediction.get("type_names", [])
        ):
            raise ValueError("state_omission prediction lineage/state set is invalid")

        risk, _, _ = _validated_evidence_json(
            case.get("risk_coverage_artifact"),
            "state_omission case %d risk coverage" % index,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.state_omission_risk_coverage.v1",
        )
        _require_identity(risk, identity, "state_omission risk coverage")
        if risk.get("prediction_sha256") != prediction_digest:
            raise ValueError("state_omission risk coverage is not bound to its prediction")
        coverage = risk.get("coverage")
        risks = risk.get("risk")
        if (
            not isinstance(coverage, list)
            or not isinstance(risks, list)
            or len(coverage) < 2
            or len(coverage) != len(risks)
        ):
            raise ValueError("state_omission risk-coverage arrays are incomplete")
        coverage_values = [
            _require_finite_number(value, "state_omission coverage") for value in coverage
        ]
        for value in risks:
            _require_finite_number(value, "state_omission risk")
        if coverage_values != sorted(coverage_values) or any(
            value < 0 or value > 1 for value in coverage_values
        ):
            raise ValueError("state_omission coverage must be ordered within [0, 1]")
        validated_artifacts += 4
    return {
        "omitted_state_count": len(states),
        "validated_case_count": case_count,
        "validated_supporting_artifact_count": validated_artifacts,
    }


def _validate_reference_downsampling_evidence(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    report_path: Optional[Path],
    repository: Optional[Path],
) -> Dict[str, Any]:
    samples, seeds = _validate_followup_sample_seed_contract(
        contract,
        "reference_downsampling",
    )
    _require_evidence_sequence(
        contract.get("reference_sizes"),
        REFERENCE_DOWNSAMPLING_SIZES,
        "reference_downsampling contract reference_sizes",
    )
    if contract.get("draw_seed_scheme") != "sha256_reference_downsampling_v1":
        raise ValueError("reference_downsampling draw_seed_scheme is not prespecified")
    grid = {
        (sample, seed, size)
        for sample in samples
        for seed in seeds
        for size in REFERENCE_DOWNSAMPLING_SIZES
    }
    case_count = _validate_evidence_case_grid(
        payload,
        contract,
        "reference_downsampling",
        ("sample", "seed", "reference_size"),
        grid,
    )
    validated_artifacts = 0
    for index, case in enumerate(payload["cases"]):
        identity = {
            "sample": case["sample"],
            "seed": case["seed"],
            "reference_size": case["reference_size"],
        }
        draw, draw_digest, _ = _validated_evidence_json(
            case.get("cell_id_draw_artifact"),
            "reference_downsampling case %d cell-ID draw" % index,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.reference_downsampling_draw.v1",
        )
        _require_identity(draw, identity, "reference_downsampling draw")
        material = "%s|%s|%s|reference_downsampling_v1" % (
            case["sample"],
            case["seed"],
            case["reference_size"],
        )
        expected_seed = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)
        if draw.get("draw_seed") != expected_seed:
            raise ValueError("reference_downsampling draw seed is stale")
        cell_ids = _require_unique_text_list(
            draw.get("cell_ids"),
            "reference_downsampling cell_ids",
        )
        source_count = draw.get("source_cell_count")
        if (
            isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or source_count < len(cell_ids)
        ):
            raise ValueError("reference_downsampling source_cell_count is invalid")
        size = case["reference_size"]
        if size == "all":
            if draw.get("is_full_reference") is not True or len(cell_ids) != source_count:
                raise ValueError("reference_downsampling all-cell draw is not complete")
        elif len(cell_ids) != size or draw.get("is_full_reference") is not False:
            raise ValueError("reference_downsampling numeric draw has the wrong cell count")
        cell_ids_sha256 = _canonical_evidence_sha256(cell_ids)
        if draw.get("cell_ids_sha256") != cell_ids_sha256:
            raise ValueError("reference_downsampling cell-ID SHA-256 is stale")

        prediction, prediction_digest, _ = _validated_evidence_json(
            case.get("prediction_artifact"),
            "reference_downsampling case %d prediction" % index,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.reference_downsampling_prediction.v1",
        )
        _require_identity(prediction, identity, "reference_downsampling prediction")
        if prediction.get("status") != "complete":
            raise ValueError("reference_downsampling prediction status must be complete")
        if (
            prediction.get("draw_manifest_sha256") != draw_digest
            or prediction.get("cell_ids_sha256") != cell_ids_sha256
        ):
            raise ValueError("reference_downsampling prediction draw lineage is stale")
        metric, _, _ = _validated_evidence_json(
            case.get("metric_artifact"),
            "reference_downsampling case %d metric" % index,
            report_path=report_path,
            repository=repository,
            schema="heir.evidence.reference_downsampling_metric.v1",
        )
        _require_identity(metric, identity, "reference_downsampling metric")
        if metric.get("prediction_sha256") != prediction_digest:
            raise ValueError("reference_downsampling metric is not bound to its prediction")
        _require_nonempty_text(metric.get("metric"), "reference_downsampling metric name")
        _require_finite_number(metric.get("statistic"), "reference_downsampling statistic")
        validated_artifacts += 3
    return {
        "reference_sizes": list(REFERENCE_DOWNSAMPLING_SIZES),
        "validated_case_count": case_count,
        "validated_supporting_artifact_count": validated_artifacts,
    }


def _validate_unknown_mass_evidence(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Require the complete fixed three-specimen, five-mass paired grid."""

    _require_evidence_sequence(
        contract.get("samples"),
        UNKNOWN_MASS_EVIDENCE_SAMPLES,
        "unknown-mass contract samples",
    )
    if contract.get("seed") != UNKNOWN_MASS_EVIDENCE_SEED:
        raise ValueError("unknown-mass contract seed must be exactly 17")
    _require_evidence_sequence(
        contract.get("unknown_masses"),
        UNKNOWN_MASS_EVIDENCE_VALUES,
        "unknown-mass contract values",
    )
    expected_cases = len(UNKNOWN_MASS_EVIDENCE_SAMPLES) * len(UNKNOWN_MASS_EVIDENCE_VALUES)
    expected_endpoints = expected_cases * 2
    if contract.get("expected_case_count") != expected_cases:
        raise ValueError("unknown-mass contract expected_case_count must be exactly 15")
    if contract.get("expected_prediction_count") != expected_endpoints:
        raise ValueError("unknown-mass contract expected_prediction_count must be exactly 30")

    request = payload.get("request")
    if not isinstance(request, Mapping):
        raise ValueError("unknown-mass report request must be an object")
    _require_evidence_sequence(
        request.get("samples"),
        UNKNOWN_MASS_EVIDENCE_SAMPLES,
        "unknown-mass request samples",
    )
    if request.get("seed") != UNKNOWN_MASS_EVIDENCE_SEED:
        raise ValueError("unknown-mass request seed must be exactly 17")
    _require_evidence_sequence(
        request.get("unknown_masses"),
        UNKNOWN_MASS_EVIDENCE_VALUES,
        "unknown-mass request values",
    )
    if request.get("expected_case_count") != expected_cases:
        raise ValueError("unknown-mass request expected_case_count must be exactly 15")
    if request.get("expected_prediction_count") != expected_endpoints:
        raise ValueError("unknown-mass request expected_prediction_count must be exactly 30")
    if payload.get("scored_case_count") != expected_cases:
        raise ValueError("unknown-mass scored_case_count must be exactly 15")
    if payload.get("scored_prediction_count") != expected_endpoints:
        raise ValueError("unknown-mass scored_prediction_count must be exactly 30")

    blockers = payload.get("blockers")
    if not isinstance(blockers, list) or blockers:
        raise ValueError("unknown-mass report blockers must be an empty list")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise ValueError("unknown-mass report must contain exactly 15 paired cases")
    expected_grid = {
        (sample, mass)
        for sample in UNKNOWN_MASS_EVIDENCE_SAMPLES
        for mass in UNKNOWN_MASS_EVIDENCE_VALUES
    }
    observed_grid = set()
    endpoint_count = 0
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError("unknown-mass case %d must be an object" % index)
        sample = str(case.get("sample", ""))
        seed = case.get("seed")
        mass = case.get("unknown_mass")
        if seed != UNKNOWN_MASS_EVIDENCE_SEED or isinstance(seed, bool):
            raise ValueError("unknown-mass case %d seed must be exactly 17" % index)
        if isinstance(mass, bool) or not isinstance(mass, (int, float)):
            raise ValueError("unknown-mass case %d has an invalid mass" % index)
        identity = (sample, float(mass))
        if identity not in expected_grid or identity in observed_grid:
            raise ValueError("unknown-mass case grid is incomplete, duplicated, or unexpected")
        observed_grid.add(identity)
        expected_label = ("%.2f" % float(mass)).replace(".", "p")
        if case.get("unknown_mass_label") != expected_label:
            raise ValueError("unknown-mass case %d has a non-canonical mass label" % index)
        endpoints = case.get("endpoints")
        if not isinstance(endpoints, Mapping) or set(endpoints) != {"round0", "refined"}:
            raise ValueError(
                "unknown-mass case %d must contain exactly round0 and refined endpoints" % index
            )
        for endpoint_name in ("round0", "refined"):
            endpoint = endpoints[endpoint_name]
            if not isinstance(endpoint, Mapping):
                raise ValueError(
                    "unknown-mass case %d %s endpoint must be an object" % (index, endpoint_name)
                )
            _require_nonempty_text(
                endpoint.get("case_id"),
                "unknown-mass case %d %s endpoint case_id" % (index, endpoint_name),
            )
            prediction = endpoint.get("prediction")
            if not isinstance(prediction, Mapping):
                raise ValueError(
                    "unknown-mass case %d %s endpoint prediction must be an object"
                    % (index, endpoint_name)
                )
            if not np.isclose(
                _require_finite_number(
                    prediction.get("unknown_mass"),
                    "unknown-mass case %d %s endpoint mass" % (index, endpoint_name),
                ),
                float(mass),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "unknown-mass case %d %s endpoint mass differs from its case"
                    % (index, endpoint_name)
                )
            metadata_binding = prediction.get("unknown_mass_metadata_binding")
            if not isinstance(metadata_binding, Mapping) or metadata_binding != {
                "round0": "checkpoint_and_manifest_bound",
                "refined": "checkpoint_and_manifest_bound",
            }:
                raise ValueError(
                    "unknown-mass case %d %s endpoint is not checkpoint-and-manifest bound"
                    % (index, endpoint_name)
                )
            if not isinstance(endpoint.get("metrics"), Mapping):
                raise ValueError(
                    "unknown-mass case %d %s endpoint metrics must be an object"
                    % (index, endpoint_name)
                )
            refinement_round = endpoint.get("refinement_round")
            if endpoint_name == "round0" and refinement_round != 0:
                raise ValueError("unknown-mass round0 endpoint must have refinement_round=0")
            if endpoint_name == "refined" and (
                isinstance(refinement_round, bool)
                or not isinstance(refinement_round, int)
                or refinement_round not in range(5)
            ):
                raise ValueError("unknown-mass refined endpoint round must be in 0..4")
            endpoint_count += 1
        contrasts = case.get("paired_gene_spearman_deltas")
        expected_contrasts = {
            "refined_minus_round0",
            "heir_minus_hard_baseline",
            "heir_minus_soft_baseline",
        }
        if not isinstance(contrasts, Mapping) or set(contrasts) != expected_contrasts:
            raise ValueError("unknown-mass case %d has an invalid paired contrast set" % index)
        for name, contrast in contrasts.items():
            if not isinstance(contrast, Mapping):
                raise ValueError("unknown-mass paired contrast %s must be an object" % name)
            median_delta = contrast.get("median_delta")
            if (
                isinstance(median_delta, bool)
                or not isinstance(median_delta, (int, float))
                or not np.isfinite(float(median_delta))
            ):
                raise ValueError(
                    "unknown-mass paired contrast %s requires a finite median_delta" % name
                )
    if observed_grid != expected_grid or endpoint_count != expected_endpoints:
        raise ValueError("unknown-mass report must validate 15 paired cases and 30 endpoints")

    stability = payload.get("stability")
    if not isinstance(stability, Mapping) or stability.get("status") not in {
        "stable",
        "unstable",
    }:
        raise ValueError("unknown-mass stability must be present and nonblocked")
    direction_stable = stability.get("direction_stable_across_masses")
    if not isinstance(direction_stable, bool):
        raise ValueError("unknown-mass stability conclusion must be boolean")
    if (stability.get("status") == "stable") != direction_stable:
        raise ValueError("unknown-mass stability status and conclusion disagree")
    return {
        "paired_case_count": expected_cases,
        "validated_endpoint_count": endpoint_count,
        "stability_status": stability["status"],
    }


def _validate_clean_reannotation_evidence(
    contract: Mapping[str, Any],
    *,
    report_path: Optional[Path],
    repository: Optional[Path],
) -> Dict[str, Any]:
    provenance = contract.get("annotation_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("clean-reannotation contract requires annotation_provenance")
    if provenance.get("independent") is not True:
        raise ValueError("clean-reannotation provenance must declare independent=true")
    if provenance.get("published_integrated_labels_used") is not False:
        raise ValueError(
            "clean-reannotation provenance must declare published_integrated_labels_used=false"
        )
    _require_nonempty_text(provenance.get("method"), "clean-reannotation provenance method")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("clean-reannotation contract artifacts must be an object")
    cell_manifest, _, _ = _validated_evidence_json(
        artifacts.get("cell_ids"),
        "clean-reannotation cell-ID manifest",
        report_path=report_path,
        repository=repository,
        schema="heir.evidence.clean_reannotation_cell_ids.v1",
    )
    cell_ids = _require_unique_text_list(
        cell_manifest.get("cell_ids"),
        "clean-reannotation cell IDs",
    )
    expected_count = contract.get("annotation_cell_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(cell_ids)
    ):
        raise ValueError("clean-reannotation annotation_cell_count is stale")

    annotation, annotation_digest, _ = _validated_evidence_json(
        artifacts.get("annotation_table"),
        "clean-reannotation annotation table",
        report_path=report_path,
        repository=repository,
        schema="heir.evidence.clean_reannotation_annotation_table.v1",
    )
    if (
        annotation.get("independent") is not True
        or annotation.get("published_integrated_labels_used") is not False
    ):
        raise ValueError("clean-reannotation table is not independently derived")
    if annotation.get("cell_ids") != cell_ids:
        raise ValueError("clean-reannotation table is not cell-ID aligned")
    labels = annotation.get("labels")
    if not isinstance(labels, list) or len(labels) != len(cell_ids):
        raise ValueError("clean-reannotation table label count is misaligned")
    labels = [_require_nonempty_text(value, "clean-reannotation label") for value in labels]

    ontology, ontology_digest, _ = _validated_evidence_json(
        artifacts.get("ontology"),
        "clean-reannotation ontology",
        report_path=report_path,
        repository=repository,
        schema="heir.evidence.clean_reannotation_ontology.v1",
    )
    type_names = _require_unique_text_list(
        ontology.get("type_names"),
        "clean-reannotation ontology type_names",
    )
    if not set(labels).issubset(set(type_names)):
        raise ValueError("clean-reannotation table contains labels outside its ontology")

    markers, _, _ = _validated_evidence_json(
        artifacts.get("marker_evidence"),
        "clean-reannotation marker evidence",
        report_path=report_path,
        repository=repository,
        schema="heir.evidence.clean_reannotation_markers.v1",
    )
    supported_types = _require_unique_text_list(
        markers.get("supported_types"),
        "clean-reannotation marker supported_types",
    )
    if markers.get("status") != "complete" or not set(type_names).issubset(set(supported_types)):
        raise ValueError("clean-reannotation marker evidence is incomplete")

    qc, _, _ = _validated_evidence_json(
        artifacts.get("qc"),
        "clean-reannotation QC",
        report_path=report_path,
        repository=repository,
        schema="heir.evidence.clean_reannotation_qc.v1",
    )
    if (
        qc.get("status") != "pass"
        or qc.get("cell_count") != len(cell_ids)
        or qc.get("aligned_cell_count") != len(cell_ids)
        or qc.get("annotation_sha256") != annotation_digest
    ):
        raise ValueError("clean-reannotation QC is incomplete or stale")

    adjudication, _, _ = _validated_evidence_json(
        artifacts.get("adjudication"),
        "clean-reannotation adjudication",
        report_path=report_path,
        repository=repository,
        schema="heir.evidence.clean_reannotation_adjudication.v1",
    )
    if (
        adjudication.get("status") != "complete"
        or adjudication.get("cell_count") != len(cell_ids)
        or adjudication.get("unresolved_count") != 0
        or adjudication.get("annotation_sha256") != annotation_digest
        or adjudication.get("ontology_sha256") != ontology_digest
    ):
        raise ValueError("clean-reannotation adjudication is incomplete or stale")
    return {
        "independent_annotation_provenance_validated": True,
        "validated_annotation_cell_count": len(cell_ids),
        "validated_supporting_artifact_count": 6,
    }


def _validate_untouched_cohort_evidence(
    contract: Mapping[str, Any],
    *,
    report_path: Optional[Path],
    repository: Optional[Path],
) -> Dict[str, Any]:
    cohort_id = _require_nonempty_text(
        contract.get("cohort_id"),
        "untouched-cohort contract cohort_id",
    )
    if contract.get("analysis_role") not in {
        "untouched_locked_validation",
        "untouched_locked_confirmatory_validation",
    }:
        raise ValueError("untouched-cohort analysis_role must explicitly be untouched and locked")
    if contract.get("untouched") is not True or contract.get("locked") is not True:
        raise ValueError("untouched-cohort contract must declare untouched=true and locked=true")
    if contract.get("freeze_before_truth_access") is not True:
        raise ValueError("untouched-cohort contract must declare freeze_before_truth_access=true")
    development_donors = _require_unique_text_list(
        contract.get("development_donor_ids"),
        "untouched-cohort development_donor_ids",
    )
    target_donors = _require_unique_text_list(
        contract.get("target_donor_ids"),
        "untouched-cohort target_donor_ids",
    )
    if set(development_donors) & set(target_donors):
        raise ValueError("untouched-cohort target donors overlap development donors")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("untouched-cohort contract artifacts must be an object")
    freeze, freeze_digest, _ = _validated_evidence_json(
        artifacts.get("freeze_manifest"),
        "untouched-cohort freeze manifest",
        report_path=report_path,
        repository=repository,
        schema="heir.evidence.untouched_cohort_freeze.v1",
    )
    if (
        freeze.get("cohort_id") != cohort_id
        or freeze.get("analysis_role") != contract.get("analysis_role")
        or freeze.get("freeze_before_truth_access") is not True
        or freeze.get("development_donor_ids") != development_donors
        or freeze.get("target_donor_ids") != target_donors
    ):
        raise ValueError("untouched-cohort freeze manifest contract is stale")
    _require_nonempty_text(freeze.get("frozen_at"), "untouched-cohort frozen_at")

    prediction, prediction_digest, _ = _validated_evidence_json(
        artifacts.get("prediction"),
        "untouched-cohort frozen prediction",
        report_path=report_path,
        repository=repository,
        schema="heir.evidence.untouched_cohort_prediction.v1",
    )
    if (
        prediction.get("cohort_id") != cohort_id
        or prediction.get("donor_ids") != target_donors
        or prediction.get("status") != "complete"
        or prediction.get("frozen") is not True
        or prediction.get("created_before_truth_access") is not True
        or prediction.get("freeze_manifest_sha256") != freeze_digest
    ):
        raise ValueError("untouched-cohort prediction is not frozen before truth access")

    truth, truth_digest, _ = _validated_evidence_json(
        artifacts.get("truth"),
        "untouched-cohort locked truth",
        report_path=report_path,
        repository=repository,
        schema="heir.evidence.untouched_cohort_truth.v1",
    )
    if (
        truth.get("cohort_id") != cohort_id
        or truth.get("donor_ids") != target_donors
        or truth.get("locked") is not True
        or truth.get("opened_after_prediction_freeze") is not True
        or truth.get("freeze_manifest_sha256") != freeze_digest
    ):
        raise ValueError("untouched-cohort truth locking/freeze lineage is invalid")

    evaluation, _, _ = _validated_evidence_json(
        artifacts.get("evaluation"),
        "untouched-cohort evaluation",
        report_path=report_path,
        repository=repository,
        schema="heir.evidence.untouched_cohort_evaluation.v1",
    )
    if (
        evaluation.get("cohort_id") != cohort_id
        or evaluation.get("donor_ids") != target_donors
        or evaluation.get("status") != "complete"
        or evaluation.get("analysis_role") != contract.get("analysis_role")
        or evaluation.get("prediction_sha256") != prediction_digest
        or evaluation.get("truth_sha256") != truth_digest
    ):
        raise ValueError("untouched-cohort evaluation lineage/status is invalid")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("untouched-cohort evaluation metrics must be non-empty")
    for name, value in metrics.items():
        _require_nonempty_text(name, "untouched-cohort metric name")
        _require_finite_number(value, "untouched-cohort metric %s" % name)
    return {
        "untouched_locked_freeze_contract_validated": True,
        "validated_target_donor_count": len(target_donors),
        "validated_supporting_artifact_count": 4,
    }


def _validate_evidence_artifact(
    requirement: str,
    payload: Mapping[str, Any],
    *,
    report_path: Optional[Path] = None,
    repository: Optional[Path] = None,
) -> Dict[str, Any]:
    expected_schema = EVIDENCE_ARTIFACT_SCHEMAS[requirement]
    if payload.get("schema") != expected_schema:
        raise ValueError("evidence %s schema must be %s" % (requirement, expected_schema))
    if payload.get("requirement") != requirement:
        raise ValueError("evidence artifact requirement identity is not %s" % requirement)
    if payload.get("status") != "complete":
        raise ValueError("evidence artifact status must be complete")
    blockers = payload.get("blockers")
    if not isinstance(blockers, list) or blockers:
        raise ValueError("evidence artifact blockers must be an empty list")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("evidence artifact contract must be an object")
    if contract.get("requirement") != requirement:
        raise ValueError("evidence contract requirement identity is not %s" % requirement)

    validation: Dict[str, Any] = {}
    if requirement == "generic_atlas":
        validation = _validate_generic_atlas_evidence(
            payload,
            contract,
            report_path=report_path,
            repository=repository,
        )
    elif requirement == "label_permutation":
        validation = _validate_label_permutation_evidence(
            payload,
            contract,
            report_path=report_path,
            repository=repository,
        )
    elif requirement == "state_omission":
        validation = _validate_state_omission_evidence(
            payload,
            contract,
            report_path=report_path,
            repository=repository,
        )
    elif requirement == "reference_downsampling":
        validation = _validate_reference_downsampling_evidence(
            payload,
            contract,
            report_path=report_path,
            repository=repository,
        )
    elif requirement == "unknown_mass_sweep":
        validation = _validate_unknown_mass_evidence(payload, contract)
    elif requirement == "clean_independent_reannotation":
        validation = _validate_clean_reannotation_evidence(
            contract,
            report_path=report_path,
            repository=repository,
        )
    elif requirement == "untouched_external_cohort":
        validation = _validate_untouched_cohort_evidence(
            contract,
            report_path=report_path,
            repository=repository,
        )
    return {
        "schema": expected_schema,
        "requirement": requirement,
        **validation,
    }


def _evidence_status(
    evidence_manifest_path: Optional[Path],
    repository: Path,
) -> Tuple[list, Dict[str, Any], Optional[Dict[str, str]]]:
    """Validate optional follow-up evidence without inferring it from matrix outputs."""

    ready: Dict[str, Any] = {}
    blockers = []
    manifest_provenance: Optional[Dict[str, str]] = None
    payload: Mapping[str, Any] = {}
    artifacts: Mapping[str, Any] = {}
    if evidence_manifest_path is not None:
        evidence_manifest_path = evidence_manifest_path.expanduser().resolve()
        manifest_provenance = {
            "path": str(evidence_manifest_path),
            "sha256": sha256_file(evidence_manifest_path),
        }
        payload = _json_object(evidence_manifest_path, "refinement-matrix evidence manifest")
        if payload.get("schema") != EVIDENCE_MANIFEST_SCHEMA:
            raise ValueError("refinement-matrix evidence manifest schema is invalid")
        candidate = payload.get("artifacts")
        if not isinstance(candidate, Mapping):
            raise ValueError("refinement-matrix evidence manifest artifacts must be an object")
        artifacts = candidate
    for requirement, message in EVIDENCE_REQUIREMENTS.items():
        row = artifacts.get(requirement)
        path: Optional[Path] = None
        if not isinstance(row, Mapping):
            blockers.append(
                {
                    "code": "missing_evidence_" + requirement,
                    "requirement": requirement,
                    "message": message,
                    "path": None,
                }
            )
            continue
        if row.get("status") != "complete":
            message = row.get("message")
            if not isinstance(message, str) or not message.strip():
                message = "evidence manifest row status must be complete"
            try:
                assert evidence_manifest_path is not None
                path = _resolve_manifest_path(row.get("path"), evidence_manifest_path, repository)
                _validated_file(path, row.get("sha256"), "blocked evidence " + requirement)
            except (OSError, TypeError, ValueError) as error:
                message = "%s; blocked evidence artifact is invalid: %s" % (message, error)
            blockers.append(
                {
                    "code": "invalid_evidence_" + requirement,
                    "requirement": requirement,
                    "message": message,
                    "path": None if path is None else str(path),
                }
            )
            continue
        try:
            assert evidence_manifest_path is not None
            path = _resolve_manifest_path(row.get("path"), evidence_manifest_path, repository)
            digest = _validated_file(path, row.get("sha256"), "evidence " + requirement)
            artifact = _json_object(path, "evidence " + requirement)
            validation = _validate_evidence_artifact(
                requirement,
                artifact,
                report_path=path,
                repository=repository,
            )
        except (OSError, TypeError, ValueError) as error:
            blockers.append(
                {
                    "code": "invalid_evidence_" + requirement,
                    "requirement": requirement,
                    "message": str(error),
                    "path": None if path is None else str(path),
                }
            )
            continue
        ready[requirement] = {
            "status": "contract_validated_complete",
            "path": str(path),
            "sha256": digest,
            **validation,
        }
    return blockers, ready, manifest_provenance


def _summary_value(case: Mapping[str, Any], method: str = METHOD) -> Optional[float]:
    value = case["methods"][method]["summary"].get(SUMMARY_METRIC)
    return None if value is None else float(value)


def _cross_case_contrast(
    name: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    paired = _paired_spearman_delta(left["methods"][METHOD], right["methods"][METHOD])
    return {
        "name": name,
        "sample": left["sample"],
        "seed": left["seed"],
        "left_case_id": left["case_id"],
        "right_case_id": right["case_id"],
        **paired,
    }


def _practical_delta_status(value: float, threshold: float) -> str:
    """Classify a paired delta using a prespecified practical equivalence interval."""

    if value >= threshold:
        return "pass"
    if value <= -threshold:
        return "fail"
    return "tie"


def _raw_sign_status(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _order_check(
    name: str,
    sample: str,
    seed: int,
    left: Optional[Mapping[str, Any]],
    right: Optional[Mapping[str, Any]],
    *,
    left_method: str = METHOD,
    right_method: str = METHOD,
    practical_delta_threshold: float = DEFAULT_PRACTICAL_DELTA_THRESHOLD,
) -> Dict[str, Any]:
    if left is None or right is None:
        return {
            "name": name,
            "sample": sample,
            "seed": seed,
            "status": "blocked",
            "reason": "one or both requested comparison artifacts are unavailable",
            "left_case_id": None if left is None else left["case_id"],
            "right_case_id": None if right is None else right["case_id"],
            "left_method": left_method,
            "right_method": right_method,
            "left_value": None,
            "right_value": None,
            "raw_sign_status": "blocked",
            "practical_delta_threshold": practical_delta_threshold,
        }
    left_value = _summary_value(left, left_method)
    right_value = _summary_value(right, right_method)
    paired = _paired_spearman_delta(
        left["methods"][left_method],
        right["methods"][right_method],
    )["summary"]
    paired_median_delta = paired["median_delta"]
    if left_value is None or right_value is None or paired_median_delta is None:
        status = "blocked"
        reason = "paired per-gene Spearman delta is not evaluable"
        raw_sign = "blocked"
    else:
        paired_median_delta = float(paired_median_delta)
        status = _practical_delta_status(paired_median_delta, practical_delta_threshold)
        raw_sign = _raw_sign_status(paired_median_delta)
        reason = {
            "pass": "",
            "tie": "paired delta falls inside the prespecified practical equivalence interval",
            "fail": "paired delta is at or below the negative practical margin",
        }[status]
    return {
        "name": name,
        "sample": sample,
        "seed": seed,
        "status": status,
        "reason": reason,
        "left_case_id": left["case_id"],
        "right_case_id": right["case_id"],
        "left_method": left_method,
        "right_method": right_method,
        "left_value": left_value,
        "right_value": right_value,
        "paired_median_per_gene_spearman_delta": paired_median_delta,
        "raw_sign_status": raw_sign,
        "raw_sign_pass": None if raw_sign == "blocked" else raw_sign == "positive",
        "practical_delta_threshold": practical_delta_threshold,
    }


def _method_macro(cases: Sequence[Mapping[str, Any]], method: str) -> Dict[str, Any]:
    summaries = [case["methods"][method]["summary"] for case in cases]
    names = sorted({name for summary in summaries for name in summary})
    metrics: Dict[str, Any] = {}
    for name in names:
        values = [
            float(summary[name])
            for summary in summaries
            if summary.get(name) is not None
            and isinstance(summary.get(name), (int, float))
            and not isinstance(summary.get(name), bool)
        ]
        metrics[name] = {
            "evaluable_cases": len(values),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
        }
    return {"case_count": len(cases), "summary_metrics": metrics}


def _macro_summaries(
    cases: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    variants: Dict[str, Any] = {}
    for variant in sorted({str(case["variant"]) for case in cases}):
        selected = [case for case in cases if case["variant"] == variant]
        variants[variant] = {
            method: _method_macro(selected, method)
            for method in (METHOD, HARD_BASELINE, SOFT_BASELINE)
        }
    contrast_macros: Dict[str, Any] = {}
    for name in sorted({str(contrast["name"]) for contrast in contrasts}):
        selected = [contrast for contrast in contrasts if contrast["name"] == name]
        case_medians = [
            float(contrast["summary"]["median_delta"])
            for contrast in selected
            if contrast["summary"]["median_delta"] is not None
        ]
        pooled = [
            float(row["delta"])
            for contrast in selected
            for row in contrast["per_gene"]
            if row["delta"] is not None
        ]
        contrast_macros[name] = {
            "case_count": len(selected),
            "evaluable_case_count": len(case_medians),
            "median_of_case_median_deltas": (
                float(np.median(case_medians)) if case_medians else None
            ),
            "mean_of_case_median_deltas": float(np.mean(case_medians)) if case_medians else None,
            "pooled_gene_case_median_delta": float(np.median(pooled)) if pooled else None,
            "pooled_gene_case_positive_fraction": (
                float(np.mean(np.asarray(pooled) > 0)) if pooled else None
            ),
        }
    return {"variants": variants, "contrasts": contrast_macros}


def _residual_routing_decomposition(
    contrasts: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    components = (
        "round0_residual_effect",
        "refined_residual_effect",
        "routing_refinement_effect",
        "total_refinement_effect",
    )
    lookup = {
        (str(row["sample"]), int(row["seed"]), str(row["name"])): row
        for row in contrasts
        if row.get("name") in components
    }
    identities = sorted({(sample, seed) for sample, seed, _ in lookup})
    cases = []
    for sample, seed in identities:
        rows: Dict[str, Any] = {}
        for component in components:
            contrast = lookup.get((sample, seed, component))
            rows[component] = (
                None
                if contrast is None
                else {
                    "left_case_id": contrast["left_case_id"],
                    "right_case_id": contrast["right_case_id"],
                    "summary": contrast["summary"],
                }
            )
        cases.append({"sample": sample, "seed": seed, "components": rows})
    return {
        "component_definitions": {
            "round0_residual_effect": "round0 residual-on minus round0 residual-off",
            "refined_residual_effect": "refined residual-on minus refined residual-off",
            "routing_refinement_effect": ("refined residual-off minus round0 residual-off"),
            "total_refinement_effect": "refined residual-on minus round0 residual-on",
        },
        "case_count": len(cases),
        "cases": cases,
    }


def _build_comparisons(
    *,
    cases: Sequence[Mapping[str, Any]],
    samples: Sequence[str],
    seeds: Sequence[int],
    controls: Sequence[str],
    control_seeds: Sequence[int],
    trajectory_seed: int,
    wrong_donor_pairings: Sequence[Tuple[str, str]],
    practical_delta_threshold: float,
) -> Tuple[list, list]:
    lookup = {
        (str(case["sample"]), int(case["seed"]), str(case["variant"])): case for case in cases
    }
    contrasts = []
    checks = []

    def order_check(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return _order_check(
            *args,
            **kwargs,
            practical_delta_threshold=practical_delta_threshold,
        )

    for sample in samples:
        trajectory_cases = [
            lookup.get((sample, trajectory_seed, variant))
            for variant in ("round0", "round1", "round2", "round3", "refined")
        ]
        for round_id, (left, right) in enumerate(
            zip(trajectory_cases[1:], trajectory_cases[:-1]),
            start=1,
        ):
            if left is not None and right is not None:
                contrasts.append(
                    _cross_case_contrast(
                        "trajectory_round%d_minus_round%d" % (round_id, round_id - 1),
                        left,
                        right,
                    )
                )
        for seed in seeds:
            refined = lookup.get((sample, seed, "refined"))
            round0 = lookup.get((sample, seed, "round0"))
            if refined is not None and round0 is not None:
                contrasts.append(_cross_case_contrast("refined_minus_round0", refined, round0))
            checks.extend(
                (
                    order_check("refined_gt_round0", sample, seed, refined, round0),
                    order_check(
                        "refined_gt_hard_baseline",
                        sample,
                        seed,
                        refined,
                        refined,
                        right_method=HARD_BASELINE,
                    ),
                    order_check(
                        "refined_gt_soft_baseline",
                        sample,
                        seed,
                        refined,
                        refined,
                        right_method=SOFT_BASELINE,
                    ),
                )
            )
        for seed in control_seeds:
            refined = lookup.get((sample, seed, "refined"))
            round0 = lookup.get((sample, seed, "round0"))
            round0_prototype = lookup.get((sample, seed, "round0_prototype_only"))
            refined_prototype = lookup.get((sample, seed, "refined_prototype_only"))
            if "round0_prototype_only" in controls:
                if round0 is not None and round0_prototype is not None:
                    contrasts.append(
                        _cross_case_contrast(
                            "round0_residual_effect",
                            round0,
                            round0_prototype,
                        )
                    )
                checks.append(
                    order_check(
                        "round0_residual_on_gt_off",
                        sample,
                        seed,
                        round0,
                        round0_prototype,
                    )
                )
            if "refined_prototype_only" in controls:
                if refined is not None and refined_prototype is not None:
                    contrasts.append(
                        _cross_case_contrast(
                            "refined_residual_effect",
                            refined,
                            refined_prototype,
                        )
                    )
                checks.append(
                    order_check(
                        "refined_residual_on_gt_off",
                        sample,
                        seed,
                        refined,
                        refined_prototype,
                    )
                )
            if {
                "round0_prototype_only",
                "refined_prototype_only",
            }.issubset(controls):
                if round0_prototype is not None and refined_prototype is not None:
                    contrasts.append(
                        _cross_case_contrast(
                            "routing_refinement_effect",
                            refined_prototype,
                            round0_prototype,
                        )
                    )
                checks.append(
                    order_check(
                        "routing_refinement_gt_zero",
                        sample,
                        seed,
                        refined_prototype,
                        round0_prototype,
                    )
                )
            if refined is not None and round0 is not None:
                contrasts.append(
                    _cross_case_contrast(
                        "total_refinement_effect",
                        refined,
                        round0,
                    )
                )
            for control in controls:
                if control in {"round0_prototype_only", "refined_prototype_only"}:
                    continue
                if control == "wrong_prototype_bank":
                    for target, source in wrong_donor_pairings:
                        if target != sample:
                            continue
                        controlled = lookup.get((sample, seed, "wrong_prototype_bank_" + source))
                        if refined is not None:
                            if controlled is not None:
                                contrasts.append(
                                    _cross_case_contrast(
                                        "refined_minus_wrong_prototype_bank",
                                        refined,
                                        controlled,
                                    )
                                )
                        checks.append(
                            order_check(
                                "refined_gt_wrong_prototype_bank",
                                sample,
                                seed,
                                refined,
                                controlled,
                            )
                        )
                    continue
                controlled = lookup.get((sample, seed, control))
                if refined is not None and controlled is not None:
                    contrasts.append(
                        _cross_case_contrast("refined_minus_" + control, refined, controlled)
                    )
                checks.append(
                    order_check("refined_gt_" + control, sample, seed, refined, controlled)
                )
    return contrasts, checks


def _wrong_donor_summary(
    cases: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    *,
    samples: Sequence[str],
    control_seeds: Sequence[int],
    sample_sites: Mapping[str, str],
    wrong_donor_pairings: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    """Summarize wrong-prototype-bank contrasts without hiding the worst source."""

    expected_pairings = tuple(wrong_donor_pairings)
    wrong_cases = {
        (str(case["sample"]), int(case["seed"]), str(case.get("prototype_donor_id"))): case
        for case in cases
        if case.get("control") == "wrong_prototype_bank"
    }
    check_lookup = {
        (str(check["sample"]), int(check["seed"]), str(check.get("right_case_id"))): check
        for check in checks
        if check.get("name") == "refined_gt_wrong_prototype_bank"
    }
    rows = []
    for target, source in expected_pairings:
        for seed in control_seeds:
            case = wrong_cases.get((target, seed, source))
            case_id = None if case is None else str(case["case_id"])
            check = check_lookup.get((target, seed, str(case_id)))
            rows.append(
                {
                    "target": target,
                    "source": source,
                    "seed": seed,
                    "site_matched": bool(
                        sample_sites.get(target) is not None
                        and sample_sites.get(target) == sample_sites.get(source)
                    ),
                    "case_id": case_id,
                    "status": "blocked" if check is None else check["status"],
                    "paired_median_per_gene_spearman_delta": (
                        None
                        if check is None
                        else check.get("paired_median_per_gene_spearman_delta")
                    ),
                }
            )

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        values = [
            float(row["paired_median_per_gene_spearman_delta"])
            for row in selected
            if row.get("paired_median_per_gene_spearman_delta") is not None
        ]
        statuses = [str(row.get("status", "blocked")) for row in selected]
        raw_all_positive = bool(len(values) == len(selected) and all(value > 0 for value in values))
        return {
            "expected_case_count": len(selected),
            "evaluable_case_count": len(values),
            "mean_paired_median_delta": float(np.mean(values)) if values else None,
            "worst_paired_median_delta": float(np.min(values)) if values else None,
            "practical_status_counts": {
                status: statuses.count(status) for status in ("pass", "tie", "fail", "blocked")
            },
            "all_practical_pass": bool(statuses and all(status == "pass" for status in statuses)),
            "all_positive_raw_sign": raw_all_positive,
            "all_positive": raw_all_positive,
        }

    by_target_seed = []
    for target in samples:
        for seed in control_seeds:
            selected = [row for row in rows if row["target"] == target and row["seed"] == seed]
            by_target_seed.append({"target": target, "seed": seed, **aggregate(selected)})
    site_matched = [row for row in rows if row["site_matched"]]
    return {
        "required_pairings": [
            {
                "target": target,
                "source": source,
                "site_matched": bool(
                    sample_sites.get(target) is not None
                    and sample_sites.get(target) == sample_sites.get(source)
                ),
            }
            for target, source in expected_pairings
        ],
        "required_pairing_count_per_control_seed": len(expected_pairings),
        "required_case_count": len(expected_pairings) * len(control_seeds),
        "observed_case_count": len(wrong_cases),
        "coverage_complete": len(wrong_cases) == len(expected_pairings) * len(control_seeds),
        "all_directed": aggregate(rows),
        "site_matched": aggregate(site_matched),
        "by_target_seed": by_target_seed,
        "cases": rows,
    }


def evaluate_matrix(
    *,
    repository: Path,
    artifact_root: Path,
    truth_manifest_path: Path,
    native_manifest_path: Optional[Path],
    evidence_manifest_path: Optional[Path] = None,
    samples: Sequence[str],
    seeds: Sequence[int],
    trajectory_seed: int = 17,
    controls: Sequence[str] = DEFAULT_CONTROLS,
    control_seeds: Sequence[int] = DEFAULT_CONTROL_SEEDS,
    wrong_donor_pairings: Optional[Sequence[Tuple[str, str]]] = None,
    wrong_donor_target: Optional[str] = None,
    wrong_donor_source: Optional[str] = None,
    sample_sites: Optional[Mapping[str, str]] = None,
    minimum_nuclei: int = 3,
    run_manifest_path: Optional[Path] = None,
    practical_delta_threshold: float = DEFAULT_PRACTICAL_DELTA_THRESHOLD,
    molecular_folds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate every available requested artifact and retain all blockers."""

    samples = tuple(dict.fromkeys(str(value) for value in samples))
    seeds = tuple(dict.fromkeys(int(value) for value in seeds))
    requested_controls = tuple(
        dict.fromkeys(LEGACY_CONTROL_ALIASES.get(str(value), str(value)) for value in controls)
    )
    fold_lookup = {} if molecular_folds is None else dict(molecular_folds)
    unsupported_true_loo_controls = (
        ("wrong_prototype_bank",)
        if fold_lookup and "wrong_prototype_bank" in requested_controls
        else ()
    )
    controls = tuple(
        control for control in requested_controls if control not in unsupported_true_loo_controls
    )
    control_seeds = tuple(dict.fromkeys(int(value) for value in control_seeds))
    if sample_sites is None:
        sample_sites = {
            sample: DEFAULT_SAMPLE_SITES[sample]
            for sample in samples
            if sample in DEFAULT_SAMPLE_SITES
        }
    else:
        sample_sites = {str(sample): str(site) for sample, site in sample_sites.items()}
    if not samples or not seeds:
        raise ValueError("at least one sample and seed are required")
    practical_delta_threshold = float(practical_delta_threshold)
    if not np.isfinite(practical_delta_threshold) or practical_delta_threshold < 0:
        raise ValueError("practical_delta_threshold must be finite and non-negative")
    if any(seed not in seeds for seed in control_seeds):
        raise ValueError("control seeds must be included in the primary seed matrix")
    if trajectory_seed not in seeds:
        raise ValueError("trajectory seed must be included in the primary seed matrix")
    repository = repository.expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()
    truth_manifest_path = truth_manifest_path.expanduser().resolve()
    truth_manifest = _json_object(truth_manifest_path, "frozen truth manifest")
    native_manifest: Optional[Mapping[str, Any]] = None
    native_by_sample: Dict[str, tuple[Path, Mapping[str, Any]]] = {}
    if fold_lookup:
        if native_manifest_path is not None:
            raise ValueError("true-LOO fold scoring cannot also use --native-manifest")
        if set(fold_lookup) != set(samples):
            raise ValueError("true-LOO scorer fold map must exactly cover requested samples")
        for sample in samples:
            fold = fold_lookup[sample]
            if fold.sample != sample or sample in set(fold.training_donors):
                raise ValueError("true-LOO scorer fold donor scope is invalid for %s" % sample)
            native_path = fold.native_manifest.expanduser().resolve()
            if sha256_file(native_path) != fold.native_manifest_sha256:
                raise ValueError("true-LOO native manifest hash differs for %s" % sample)
            native = _json_object(
                native_path,
                "%s native true-LOO manifest" % sample,
                schema=NATIVE_MANIFEST_SCHEMAS["r2"],
            )
            if native.get("latent_space_id") != fold.latent_space_id:
                raise ValueError("true-LOO native latent identity differs for %s" % sample)
            decoder = native.get("distilled_decoder")
            if not isinstance(decoder, Mapping) or decoder.get("sha256") != fold.decoder_sha256:
                raise ValueError("true-LOO native decoder identity differs for %s" % sample)
            decoder_name = Path(str(decoder.get("external_path", ""))).expanduser()
            decoder_path = (
                decoder_name.resolve()
                if decoder_name.is_absolute()
                else (repository / decoder_name).resolve()
            )
            if decoder_path != fold.decoder or sha256_file(decoder_path) != fold.decoder_sha256:
                raise ValueError("true-LOO native decoder file differs for %s" % sample)
            native_by_sample[sample] = (native_path, native)
        molecular_generation = "r2"
    else:
        if native_manifest_path is None:
            raise ValueError("native scANVI manifest is required without true-LOO folds")
        native_manifest_path = native_manifest_path.expanduser().resolve()
        native_manifest = _json_object(native_manifest_path, "native scANVI manifest")
        molecular_generation = _native_molecular_generation(native_manifest)
        native_by_sample = {sample: (native_manifest_path, native_manifest) for sample in samples}
    blockers = []
    if unsupported_true_loo_controls:
        blockers.append(
            _blocker(
                "requested_control_unavailable_cross_latent_space",
                "wrong-prototype-bank controls were requested, but independent true-LOO "
                "folds have distinct latent spaces and no target-fold-compatible wrong banks "
                "were produced; the remaining matrix is scored but the requested matrix is "
                "incomplete",
            )
        )
    sample_inputs: Dict[str, SampleInputs] = {}
    for sample in samples:
        try:
            sample_native_path, sample_native = native_by_sample[sample]
            sample_inputs[sample] = load_sample_inputs(
                sample=sample,
                truth_manifest_path=truth_manifest_path,
                truth_manifest=truth_manifest,
                native_manifest_path=sample_native_path,
                native_manifest=sample_native,
                repository=repository,
            )
        except (OSError, TypeError, ValueError) as error:
            blockers.append(_blocker("invalid_sample_inputs", str(error), sample=sample))
    requests = build_requests(
        artifact_root=artifact_root,
        samples=samples,
        seeds=seeds,
        trajectory_seed=trajectory_seed,
        controls=controls,
        control_seeds=control_seeds,
        wrong_donor_pairings=wrong_donor_pairings,
        wrong_donor_target=wrong_donor_target,
        wrong_donor_source=wrong_donor_source,
    )
    expected_wrong_donor_pairings = tuple(
        (target, source) for target in samples for source in samples if source != target
    )
    requested_wrong_donor_pairings = (
        tuple(
            (request.sample, str(request.prototype_donor_id))
            for request in requests
            if request.control == "wrong_prototype_bank" and request.seed == control_seeds[0]
        )
        if control_seeds
        else ()
    )
    if (
        "wrong_prototype_bank" in controls
        and len(samples) > 1
        and (set(requested_wrong_donor_pairings) != set(expected_wrong_donor_pairings))
    ):
        blockers.append(
            _blocker(
                "incomplete_wrong_prototype_bank_pairing_plan",
                "wrong-prototype-bank controls must cover both alternative sources for every "
                "specimen",
            )
        )
    comparison_wrong_donor_pairings = (
        expected_wrong_donor_pairings if len(samples) > 1 else requested_wrong_donor_pairings
    )
    execution_provenance_blockers = []
    run_manifest_validation: Optional[RunManifestValidation] = None
    if run_manifest_path is None:
        execution_provenance_blockers.append(
            _blocker(
                "missing_refinement_run_manifest",
                "No exact-plan refinement run/adoption manifest was supplied; scores remain "
                "development evidence only.",
            )
        )
    else:
        try:
            run_manifest_validation = validate_refinement_run_manifest(
                run_manifest_path,
                repository=repository,
                native_manifest_path=native_manifest_path,
                native_manifest=native_manifest,
                requests=requests,
                artifact_root=artifact_root,
                molecular_folds=fold_lookup or None,
            )
        except (OSError, TypeError, ValueError) as error:
            execution_provenance_blockers.append(
                _blocker("invalid_refinement_run_manifest", str(error))
            )
        else:
            if not run_manifest_validation.original_execution_source_verified:
                execution_provenance_blockers.append(
                    _blocker(
                        "posthoc_adoption_not_original_execution_proof",
                        "The exact-plan manifest validates current artifacts but adopted one or "
                        "more pre-existing stages, so it cannot prove their original execution "
                        "source revision.",
                    )
                )
            if not run_manifest_validation.execution_transform_hash_verified:
                execution_provenance_blockers.append(
                    _blocker(
                        "control_transform_hash_unverified",
                        "Legacy image/graph shuffle telemetry lacks realized permutation-map "
                        "hashes; deterministic recipes are recorded but execution maps are not "
                        "cryptographically verified.",
                    )
                )
    cases = []
    for request in requests:
        inputs = sample_inputs.get(request.sample)
        if inputs is None:
            blockers.append(
                _blocker(
                    "unscorable_sample_inputs",
                    "sample truth/reference did not pass validation",
                    request=request,
                )
            )
            continue
        try:
            prediction, provenance = load_prediction(
                request,
                inputs,
                run_stage=(
                    None
                    if run_manifest_validation is None
                    else run_manifest_validation.request_stages.get(request.case_id)
                ),
            )
            cases.append(
                score_prediction(
                    request,
                    prediction,
                    inputs,
                    provenance,
                    minimum_nuclei=minimum_nuclei,
                )
            )
        except FileNotFoundError as error:
            blockers.append(_blocker("missing_requested_artifact", str(error), request=request))
        except (OSError, TypeError, ValueError) as error:
            blockers.append(_blocker("invalid_requested_artifact", str(error), request=request))
    contrasts, checks = _build_comparisons(
        cases=cases,
        samples=samples,
        seeds=seeds,
        controls=controls,
        control_seeds=control_seeds,
        trajectory_seed=trajectory_seed,
        wrong_donor_pairings=comparison_wrong_donor_pairings,
        practical_delta_threshold=practical_delta_threshold,
    )
    wrong_donor_summary = _wrong_donor_summary(
        cases,
        checks,
        samples=samples,
        control_seeds=control_seeds,
        sample_sites=sample_sites,
        wrong_donor_pairings=comparison_wrong_donor_pairings,
    )
    if fold_lookup:
        wrong_donor_summary = {
            **wrong_donor_summary,
            "coverage_complete": False,
            "availability": "unavailable_cross_latent_space",
            "unavailable_reason": (
                "Independent true-LOO folds have distinct latent spaces. No wrong-prototype "
                "bank is valid unless it is regenerated under the target fold transform."
            ),
        }
    check_statuses = [str(check["status"]) for check in checks]
    if blockers or "blocked" in check_statuses:
        strict_status = "blocked"
    elif any(status in {"tie", "fail"} for status in check_statuses):
        strict_status = "fail"
    else:
        strict_status = "pass"
    trajectory = {}
    lookup = {
        (str(case["sample"]), int(case["seed"]), str(case["variant"])): case for case in cases
    }
    for sample in samples:
        rows = []
        for round_id, variant in ((1, "round1"), (2, "round2"), (3, "round3"), (4, "refined")):
            case = lookup.get((sample, trajectory_seed, variant))
            rows.append(
                {
                    "round": round_id,
                    "case_id": None if case is None else case["case_id"],
                    SUMMARY_METRIC: None if case is None else _summary_value(case),
                    "status": "blocked" if case is None else "scored",
                }
            )
        trajectory[sample] = rows
    input_rows = {
        sample: {
            "truth": {
                "path": str(value.truth_path),
                "sha256": value.truth_sha256,
                "hash_validation": "matched_frozen_truth_manifest",
            },
            "native_scanvi_reference": {
                "path": str(value.reference_path),
                "sha256": value.reference_sha256,
                "hash_validation": (
                    "matched_target_specific_true_loo_native_manifest"
                    if fold_lookup
                    else "matched_native_scanvi_manifest"
                ),
            },
        }
        for sample, value in sample_inputs.items()
    }
    evidence_blockers, evidence_ready, evidence_manifest_provenance = _evidence_status(
        evidence_manifest_path,
        repository,
    )
    matrix_status = "blocked" if blockers else "complete"
    execution_provenance_verified = bool(
        run_manifest_validation is not None
        and run_manifest_validation.execution_provenance_verified
    )
    execution_transform_hash_verified = bool(
        run_manifest_validation is not None
        and run_manifest_validation.execution_transform_hash_verified
    )
    primary_evidence_status = (
        "blocked" if evidence_blockers or execution_provenance_blockers else "complete"
    )
    overall_status = (
        "blocked_matrix"
        if matrix_status == "blocked"
        else (
            "blocked_evidence"
            if primary_evidence_status == "blocked"
            else ("complete_ordering_failed" if strict_status == "fail" else "complete")
        )
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": overall_status,
        "matrix_status": matrix_status,
        "primary_evidence_status": primary_evidence_status,
        "execution_provenance_verified": execution_provenance_verified,
        "execution_transform_hash_verified": execution_transform_hash_verified,
        "strict_ordering_status": strict_status,
        "practical_delta_threshold": practical_delta_threshold,
        "analysis_role": (
            "true_leave_one_donor_out_uninitialized_live_e_step_negative_control"
            if fold_lookup
            else "native_scanvi_published_integrated_annotation_sensitivity"
        ),
        "negative_control": bool(fold_lookup),
        "claim_scope": (
            {
                "eligible_for_primary_performance_claims": False,
                "reasons": [
                    "uninitialized_morphology_negative_control",
                    "live_student_e_step_negative_control",
                ],
            }
            if fold_lookup
            else None
        ),
        "molecular_generation": molecular_generation,
        "annotation_provenance": (
            {sample: native_by_sample[sample][1].get("annotation_provenance") for sample in samples}
            if fold_lookup
            else native_manifest.get("annotation_provenance")
        ),
        "request": {
            "samples": list(samples),
            "seeds": list(seeds),
            "trajectory_seed": trajectory_seed,
            "controls": list(controls),
            "requested_controls": list(requested_controls),
            "unsupported_controls": list(unsupported_true_loo_controls),
            "control_coverage": {
                "wrong_prototype_bank": {
                    "available": not fold_lookup,
                    "coverage_complete": not fold_lookup,
                    "reason": (
                        "target-fold-compatible wrong banks were not produced"
                        if fold_lookup
                        else None
                    ),
                }
            },
            "control_seeds": list(control_seeds),
            "practical_delta_threshold": practical_delta_threshold,
            "wrong_prototype_bank_pairings": [
                {"target": target, "source": source}
                for target, source in requested_wrong_donor_pairings
            ],
            "requested_wrong_prototype_bank_pairings": [
                {"target": target, "source": source}
                for target, source in requested_wrong_donor_pairings
            ],
            "expected_in_cohort_wrong_prototype_bank_pairings": [
                {"target": target, "source": source}
                for target, source in expected_wrong_donor_pairings
            ],
            # Deprecated aliases retained for existing report consumers and CLI users.
            "wrong_donor_pairings": [
                {"target": target, "source": source}
                for target, source in requested_wrong_donor_pairings
            ],
            "requested_wrong_donor_pairings": [
                {"target": target, "source": source}
                for target, source in requested_wrong_donor_pairings
            ],
            "expected_in_cohort_wrong_donor_pairings": [
                {"target": target, "source": source}
                for target, source in expected_wrong_donor_pairings
            ],
            "sample_sites": dict(sample_sites),
            "minimum_nuclei": minimum_nuclei,
        },
        "manifests": {
            "frozen_truth": {
                "path": str(truth_manifest_path),
                "sha256": sha256_file(truth_manifest_path),
            },
            "native_scanvi": {
                "path": None if fold_lookup else str(native_manifest_path),
                "sha256": None if fold_lookup else sha256_file(native_manifest_path),
                "molecular_generation": molecular_generation,
            },
            "native_scanvi_folds": (
                {
                    sample: {
                        "path": str(fold_lookup[sample].native_manifest),
                        "sha256": fold_lookup[sample].native_manifest_sha256,
                        "decoder": str(fold_lookup[sample].decoder),
                        "decoder_sha256": fold_lookup[sample].decoder_sha256,
                        "latent_space_id": fold_lookup[sample].latent_space_id,
                    }
                    for sample in samples
                }
                if fold_lookup
                else None
            ),
            "additional_evidence": evidence_manifest_provenance,
            "refinement_run": (
                None
                if run_manifest_validation is None
                else {
                    "path": str(run_manifest_validation.path),
                    "sha256": run_manifest_validation.sha256,
                    "schema": REFINEMENT_RUN_MANIFEST_SCHEMA,
                    "manifest_role": run_manifest_validation.manifest_role,
                    "execution_mode": run_manifest_validation.execution_mode,
                    "stage_count": run_manifest_validation.stage_count,
                    "original_execution_source_verified": (
                        run_manifest_validation.original_execution_source_verified
                    ),
                    "execution_transform_hash_verified": (
                        run_manifest_validation.execution_transform_hash_verified
                    ),
                    "execution_provenance_verified": (
                        run_manifest_validation.execution_provenance_verified
                    ),
                }
            ),
        },
        "inputs": input_rows,
        "requested_artifact_count": len(requests),
        "scored_artifact_count": len(cases),
        "blockers": blockers + execution_provenance_blockers + evidence_blockers,
        "matrix_blockers": blockers,
        "execution_provenance_blockers": execution_provenance_blockers,
        "evidence_blockers": evidence_blockers,
        "evidence_ready": evidence_ready,
        "cases": cases,
        "trajectory": trajectory,
        "paired_gene_spearman_contrasts": contrasts,
        "residual_routing_decomposition": _residual_routing_decomposition(contrasts),
        "wrong_prototype_bank_contrasts": wrong_donor_summary,
        # Deprecated report-key alias.
        "wrong_donor_contrasts": wrong_donor_summary,
        "macro_summaries": _macro_summaries(cases, contrasts),
        "strict_ordering_checks": checks,
        "strict_ordering_summary": {
            "status": strict_status,
            "pass_count": check_statuses.count("pass"),
            "tie_count": check_statuses.count("tie"),
            "fail_count": check_statuses.count("fail"),
            "blocked_count": check_statuses.count("blocked"),
            "practical_delta_threshold": practical_delta_threshold,
            "raw_sign_diagnostics": {
                "positive_count": sum(
                    check.get("raw_sign_status") == "positive" for check in checks
                ),
                "zero_count": sum(check.get("raw_sign_status") == "zero" for check in checks),
                "negative_count": sum(
                    check.get("raw_sign_status") == "negative" for check in checks
                ),
                "blocked_count": sum(check.get("raw_sign_status") == "blocked" for check in checks),
            },
            "required_policy": (
                "Every required paired median per-gene Spearman delta must meet the "
                "prespecified positive practical margin; nested round0/refined residual-on/off "
                "controls decompose residual, routing, and total refinement effects"
            ),
        },
    }
    return report


def _tsv(report: Mapping[str, Any]) -> str:
    columns = (
        "row_type",
        "sample",
        "seed",
        "variant",
        "method_or_contrast",
        "metric",
        "value",
        "status",
        "raw_sign_status",
        "practical_delta_threshold",
        "case_id",
    )
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for case in report["cases"]:
        for method, metrics in case["methods"].items():
            for metric, value in metrics["summary"].items():
                writer.writerow(
                    {
                        "row_type": "method_summary",
                        "sample": case["sample"],
                        "seed": case["seed"],
                        "variant": case["variant"],
                        "method_or_contrast": method,
                        "metric": metric,
                        "value": "" if value is None else value,
                        "status": "scored",
                        "case_id": case["case_id"],
                    }
                )
        for name, contrast in case["paired_gene_spearman_deltas"].items():
            for metric, value in contrast["summary"].items():
                writer.writerow(
                    {
                        "row_type": "paired_delta_summary",
                        "sample": case["sample"],
                        "seed": case["seed"],
                        "variant": case["variant"],
                        "method_or_contrast": name,
                        "metric": metric,
                        "value": "" if value is None else value,
                        "status": "scored",
                        "case_id": case["case_id"],
                    }
                )
    for check in report["strict_ordering_checks"]:
        writer.writerow(
            {
                "row_type": "strict_ordering",
                "sample": check["sample"],
                "seed": check["seed"],
                "variant": "",
                "method_or_contrast": check["name"],
                "metric": "paired_median_per_gene_spearman_delta",
                "value": check.get("paired_median_per_gene_spearman_delta", ""),
                "status": check["status"],
                "raw_sign_status": check.get("raw_sign_status", ""),
                "practical_delta_threshold": check.get("practical_delta_threshold", ""),
                "case_id": check["left_case_id"] or "",
            }
        )
    wrong_donor = report["wrong_prototype_bank_contrasts"]
    for scope in ("all_directed", "site_matched"):
        row = wrong_donor[scope]
        for metric in ("mean_paired_median_delta", "worst_paired_median_delta"):
            writer.writerow(
                {
                    "row_type": "wrong_prototype_bank_aggregate",
                    "sample": scope,
                    "seed": "",
                    "variant": "wrong_prototype_bank",
                    "method_or_contrast": "refined_minus_wrong_prototype_bank",
                    "metric": metric,
                    "value": "" if row[metric] is None else row[metric],
                    "status": "complete" if wrong_donor["coverage_complete"] else "blocked",
                    "case_id": "",
                }
            )
    return handle.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Native snPATHO refinement matrix",
        "",
        ("Status: **%s**; matrix: **%s**; strict ordering: **%s**; full primary evidence: **%s**.")
        % (
            report["status"],
            report["matrix_status"],
            report["strict_ordering_status"],
            report["primary_evidence_status"],
        ),
        "",
        "Execution provenance verified: **%s**; control transform hashes verified: **%s**."
        % (
            str(bool(report["execution_provenance_verified"])).lower(),
            str(bool(report["execution_transform_hash_verified"])).lower(),
        ),
        "",
        "Scored %d of %d requested artifacts."
        % (report["scored_artifact_count"], report["requested_artifact_count"]),
        "",
        "Practical paired-delta threshold: **%.6f**. Values inside (-threshold, "
        "+threshold) are practical ties; raw signs remain separate diagnostics."
        % report["practical_delta_threshold"],
        "",
    ]
    blockers = report["blockers"]
    lines.extend(("## Blockers", ""))
    if blockers:
        lines.extend(
            "- `%s`: %s%s"
            % (
                row["code"],
                row["message"],
                " (`%s`)" % row.get("case_id") if row.get("case_id") else "",
            )
            for row in blockers
        )
    else:
        lines.append("None.")
    lines.extend(
        (
            "",
            "## Wrong-prototype-bank coverage and aggregates",
            "",
            "Coverage: **%d/%d** directed target/source/seed cases; complete: **%s**."
            % (
                report["wrong_prototype_bank_contrasts"]["observed_case_count"],
                report["wrong_prototype_bank_contrasts"]["required_case_count"],
                str(bool(report["wrong_prototype_bank_contrasts"]["coverage_complete"])).lower(),
            ),
            "",
            "| Scope | Evaluable / expected | Mean paired delta | "
            "Worst paired delta | All practical pass | Raw all-positive |",
            "|---|---:|---:|---:|---|---|",
        )
    )
    for scope in ("all_directed", "site_matched"):
        row = report["wrong_prototype_bank_contrasts"][scope]
        lines.append(
            "| %s | %d / %d | %s | %s | %s | %s |"
            % (
                scope,
                row["evaluable_case_count"],
                row["expected_case_count"],
                "NA"
                if row["mean_paired_median_delta"] is None
                else "%.6f" % row["mean_paired_median_delta"],
                "NA"
                if row["worst_paired_median_delta"] is None
                else "%.6f" % row["worst_paired_median_delta"],
                str(bool(row["all_practical_pass"])).lower(),
                str(bool(row["all_positive_raw_sign"])).lower(),
            )
        )
    lines.extend(
        (
            "",
            "## HEIR median gene Spearman",
            "",
            "| Sample | Seed | Variant | Value |",
            "|---|---:|---|---:|",
        )
    )
    for case in report["cases"]:
        value = _summary_value(case)
        lines.append(
            "| %s | %d | %s | %s |"
            % (
                case["sample"],
                case["seed"],
                case["variant"],
                "NA" if value is None else "%.6f" % value,
            )
        )
    lines.extend(
        (
            "",
            "## Strict ordering checks",
            "",
            "Pass/tie/fail uses the practical threshold; raw sign is reported independently. "
            "Both use the paired median across per-gene Spearman differences.",
            "",
            "| Check | Sample | Seed | Practical status | Raw sign | Paired delta |",
            "|---|---|---:|---|---|---:|",
        )
    )
    for check in report["strict_ordering_checks"]:
        delta = check.get("paired_median_per_gene_spearman_delta")
        lines.append(
            "| %s | %s | %d | %s | %s | %s |"
            % (
                check["name"],
                check["sample"],
                check["seed"],
                check["status"],
                check.get("raw_sign_status", "blocked"),
                "NA" if delta is None else "%.6f" % delta,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_write_texts(outputs: Mapping[Path, str]) -> None:
    """Write each report through a same-directory fsynced temporary file."""

    temporary: Dict[Path, str] = {}
    try:
        for destination, content in outputs.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, path = tempfile.mkstemp(
                prefix=destination.name + ".",
                suffix=".tmp",
                dir=str(destination.parent),
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary[destination] = path
        for destination, path in temporary.items():
            os.replace(path, destination)
    finally:
        for path in temporary.values():
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def _declared_path_values(value: object, *, bases: Sequence[Path]) -> Tuple[Path, ...]:
    """Collect path-valued fields from a validated report or one of its manifests."""

    paths = []
    semantic_path_keys = {
        "cli_source",
        "decoder",
        "gene_panel",
        "latent_reference",
        "module_entrypoint",
        "molecular_producer",
        "native_scanvi_manifest",
        "panel_reference",
        "predictions",
        "prototypes",
        "python_executable",
        "rare_complete_prototypes",
        "refined_prototype",
        "refinement_audit",
        "residual_geometry",
        "source_root",
        "telemetry",
        "truth",
    }

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = str(raw_key)
                if (
                    (key == "path" or key.endswith("_path") or key in semantic_path_keys)
                    and isinstance(child, str)
                    and child.strip()
                ):
                    candidate = Path(child).expanduser()
                    if candidate.is_absolute():
                        paths.append(candidate.resolve())
                    else:
                        paths.extend((base / candidate).resolve() for base in bases)
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return tuple(dict.fromkeys(paths))


def _bound_input_paths(
    report: Mapping[str, Any],
    *,
    repository: Path,
    manifest_paths: Sequence[Path],
) -> Tuple[Path, ...]:
    """Inventory every report path plus paths declared by its primary manifests."""

    repository = repository.expanduser().resolve()
    inputs = list(_declared_path_values(report, bases=(repository,)))
    for raw_path in manifest_paths:
        path = raw_path.expanduser().resolve()
        inputs.append(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        inputs.extend(_declared_path_values(payload, bases=(path.parent, repository)))
    return tuple(dict.fromkeys(inputs))


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
        label="snPATHO refinement-matrix report",
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
        choices=tuple(NATIVE_MANIFEST_SCHEMAS),
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
        "--molecular-fold-preparation-manifest",
        action="append",
        default=[],
        metavar="TARGET=PATH",
        help=(
            "repeat once per requested target to score a true-LOO runner manifest with "
            "target-specific native manifests, decoders, and latent spaces"
        ),
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=None,
        help="Exact-plan v2 execution/adoption manifest produced by the refinement runner",
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        help=(
            "Optional hash-bound manifest for additional controls, clean reannotation, and "
            "untouched-cohort evidence"
        ),
    )
    parser.add_argument("--sample", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--trajectory-seed", type=int, default=17)
    parser.add_argument(
        "--control",
        action="append",
        choices=(*DEFAULT_CONTROLS, *LEGACY_CONTROL_ALIASES),
    )
    parser.add_argument("--control-seed", action="append", type=int)
    parser.add_argument(
        "--wrong-donor-target",
        help=(
            "Legacy CLI alias for one wrong-prototype-bank target; the default evaluates "
            "every directed pairing"
        ),
    )
    parser.add_argument(
        "--wrong-donor-source",
        help=(
            "Legacy CLI alias for one wrong-prototype-bank source; the default evaluates "
            "every directed pairing"
        ),
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
    run_manifest = (
        repository
        / (
            "reports/snpatho_refinement_v1_five_seed_manifest.json"
            if args.molecular_generation == "r1"
            else "artifacts/snpatho/r2_scanvi/refinement_run_manifest.json"
        )
        if args.run_manifest is None
        else args.run_manifest
    )
    samples = tuple(args.sample) if args.sample else DEFAULT_SAMPLES
    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    molecular_folds = (
        load_true_loo_molecular_folds(
            repository,
            args.molecular_fold_preparation_manifest,
            required_samples=samples,
        )
        if args.molecular_fold_preparation_manifest
        else None
    )
    if molecular_folds is not None and args.molecular_generation != "r2":
        raise ValueError("true-LOO molecular fold manifests require --molecular-generation r2")
    if molecular_folds is not None and args.native_manifest is not None:
        raise ValueError("true-LOO molecular folds cannot be combined with --native-manifest")
    native_manifest = (
        None
        if molecular_folds is not None
        else (
            repository
            / (
                "reports/snpatho_scanvi_r1_manifest.json"
                if args.molecular_generation == "r1"
                else "artifacts/snpatho/r2_scanvi/native_manifest.json"
            )
            if args.native_manifest is None
            else args.native_manifest
        )
    )
    controls = tuple(args.control) if args.control else DEFAULT_CONTROLS
    control_seeds = (
        tuple(args.control_seed)
        if args.control_seed
        else tuple(seed for seed in DEFAULT_CONTROL_SEEDS if seed in seeds)
    )
    report = evaluate_matrix(
        repository=repository,
        artifact_root=artifact_root,
        truth_manifest_path=args.truth_manifest,
        native_manifest_path=native_manifest,
        evidence_manifest_path=args.evidence_manifest,
        samples=samples,
        seeds=seeds,
        trajectory_seed=args.trajectory_seed,
        controls=controls,
        control_seeds=control_seeds,
        wrong_donor_target=args.wrong_donor_target,
        wrong_donor_source=args.wrong_donor_source,
        minimum_nuclei=args.minimum_nuclei,
        run_manifest_path=run_manifest,
        practical_delta_threshold=args.practical_delta_threshold,
        molecular_folds=molecular_folds,
    )
    manifest_paths = [args.truth_manifest, run_manifest]
    if native_manifest is not None:
        manifest_paths.append(native_manifest)
    if args.evidence_manifest is not None:
        manifest_paths.append(args.evidence_manifest)
    if molecular_folds is not None:
        for fold in molecular_folds.values():
            manifest_paths.extend((fold.preparation_manifest, fold.native_manifest))
    guard_controls = tuple(
        control
        for control in controls
        if not (
            molecular_folds is not None
            and LEGACY_CONTROL_ALIASES.get(control, control) == "wrong_prototype_bank"
        )
    )
    requested_input_paths = []
    for request in build_requests(
        artifact_root=artifact_root,
        samples=samples,
        seeds=seeds,
        trajectory_seed=args.trajectory_seed,
        controls=guard_controls,
        control_seeds=control_seeds,
        wrong_donor_target=args.wrong_donor_target,
        wrong_donor_source=args.wrong_donor_source,
    ):
        requested_input_paths.extend((request.prediction, request.telemetry))
        if request.prototype_source is not None:
            requested_input_paths.append(request.prototype_source)
    write_report(
        report,
        json_output=args.json_output,
        tsv_output=args.tsv_output,
        markdown_output=args.markdown_output,
        input_paths=(
            *_bound_input_paths(
                report,
                repository=repository,
                manifest_paths=manifest_paths,
            ),
            *requested_input_paths,
        ),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "strict_ordering_status": report["strict_ordering_status"],
                "scored_artifacts": report["scored_artifact_count"],
                "requested_artifacts": report["requested_artifact_count"],
                "blockers": len(report["blockers"]),
            },
            sort_keys=True,
        )
    )
    return 2 if report["strict_ordering_status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
