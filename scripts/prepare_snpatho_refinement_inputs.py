#!/usr/bin/env python3
"""Prepare hash-bound native-scANVI inputs for the snPATHO refiner.

The script owns only the four derived artifacts consumed by the refinement
benchmark.  It validates the native scANVI provenance, frozen spatial split,
and histology-only OOD calibration before invoking the public ``heir`` CLI.
Per-stage receipts make interrupted runs resumable; untracked, partial, stale,
or hash-mismatched outputs are rejected instead of overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from heir.data import HistologyBag, PrototypeSet, RNAReference
from heir.expression import EXPRESSION_SPACE_ID
from heir.prior import RNAResidualGeometry
from heir.prior.scvi_adapter import SCVI_EXPRESSION_NORMALIZATION_CONTRACT
from heir.training import HEIRTrainingBatch, TrainingStage
from heir.uncertainty import MahalanobisOOD

SAMPLES = ("4066", "4399", "4411")
SCHEMA = "heir.snpatho_refinement_input_preparation.v1"
RECEIPT_SCHEMA = "heir.snpatho_refinement_input_stage.v1"
TARGET_OOD_SCHEMA = "heir.target_histology_ood_calibration.v2"
TARGET_OOD_CONTRACT = "heir.target_histology_ood_calibration"
TARGET_OOD_CONTRACT_VERSION = 2
TARGET_OOD_INPUT_MODALITY = "development_threshold_plus_target_histology_telemetry"
TARGET_OOD_THRESHOLD_SOURCE = "development_detector"
RECIPE = {
    "prototype": {
        "include_rare_types": True,
        "max_per_type": 10,
        "minimum_cells": 50,
        "seed": 17,
        "shrinkage_kappa": 50.0,
    },
    "residual_geometry": {
        "bound_fraction": 0.5,
        "calibration_quantile": 0.9,
        "minimum_bound": 0.001,
        "minimum_calibration_cells": 3,
        "rank": 4,
    },
    "batch": {
        "analysis_role": "development_retrospective",
        "artifact_threshold": 0.5,
        "markers_per_type": 25,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_string_sha256(values: Sequence[object]) -> str:
    normalized = [str(value) for value in np.asarray(values).reshape(-1).tolist()]
    encoded = json.dumps(
        {"dtype": "string", "shape": [len(normalized)], "values": normalized},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    sources = sorted(item for item in path.rglob("*") if item.is_file())
    if not sources:
        raise ValueError("native scANVI checkpoint directory is empty: %s" % path)
    for source in sources:
        digest.update(str(source.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("JSON root must be an object: %s" % path)
    return payload


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _path_name(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.resolve()))
    except ValueError:
        return str(path.resolve())


def _record(path: Path, repository: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _path_name(path, repository), "sha256": _sha256(path)}


def _declared_file(record: object, expected: Path, label: str) -> Mapping[str, object]:
    if not isinstance(record, Mapping):
        raise ValueError("%s provenance record is missing" % label)
    declared_path = record.get("path")
    declared_hash = record.get("sha256")
    if not isinstance(declared_path, str) or Path(declared_path).expanduser().resolve() != expected:
        raise ValueError("%s provenance path differs from the expected artifact" % label)
    if declared_hash != _sha256(expected):
        raise ValueError("%s provenance SHA-256 is stale" % label)
    return record


def _decoder_metadata(path: Path, label: str) -> Mapping[str, object]:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("%s cannot be parsed" % label) from error
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("metadata"), Mapping):
        raise ValueError("%s lacks checkpoint metadata" % label)
    return checkpoint["metadata"]


def _validate_decoder_metadata(
    metadata: Mapping[str, object],
    *,
    label: str,
    genes: Sequence[str],
    latent_space_id: str,
    correction_mode: object,
    transform_batch: object,
    training_donors: Sequence[str],
) -> None:
    if metadata.get("schema") != "heir.scvi_distilled_decoder.v3":
        raise ValueError("%s does not use the R2 decoder schema" % label)
    expected = {
        "gene_names": list(genes),
        "latent_space_id": latent_space_id,
        "batch_correction_mode": correction_mode,
        "transform_batch": transform_batch,
        "training_donors": sorted(training_donors),
        "decoder_only": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError("%s %s differs from R2 provenance" % (label, key))
    samples = metadata.get("posterior_samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 1:
        raise ValueError("%s lacks MC-averaged expression targets" % label)
    for key in (
        "distillation_latent_sha256",
        "distillation_target_sha256",
        "validation_mask_sha256",
    ):
        value = metadata.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("%s lacks %s" % (label, key))


@dataclass(frozen=True)
class R2TrainingScope:
    """Validated molecular fit scope for one downstream preparation family."""

    training_donors: Tuple[str, ...]
    held_out_sample: Optional[str]
    active_samples: Tuple[str, ...]
    true_leave_one_donor_out: bool


def _r2_training_scope(provenance: Mapping[str, object]) -> R2TrainingScope:
    def donors(value: object) -> Tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(str(item) for item in value)

    status = provenance.get("status")
    partition = provenance.get("training_partition")
    if status == "native_scanvi_true_leave_one_donor_out":
        if provenance.get("analysis_role") != "leave_one_donor_out_molecular_audit":
            raise ValueError("true-LOO scANVI provenance has the wrong analysis role")
        if not isinstance(partition, Mapping) or partition.get("mode") != "leave_one_donor_out":
            raise ValueError("true-LOO scANVI provenance lacks its training partition")
        held_out = str(partition.get("held_out_sample", ""))
        if held_out not in SAMPLES:
            raise ValueError("true-LOO scANVI provenance names an invalid held-out donor")
        expected = tuple(sample for sample in SAMPLES if sample != held_out)
        label_ontology_value = partition.get("label_ontology")
        label_ontology = (
            tuple(str(value) for value in label_ontology_value)
            if isinstance(label_ontology_value, (list, tuple))
            else ()
        )
        if (
            not label_ontology
            or tuple(sorted(set(label_ontology))) != label_ontology
            or partition.get("label_ontology_sha256") != _ordered_string_sha256(label_ontology)
        ):
            raise ValueError("true-LOO training-donor label ontology is invalid")
        if (
            donors(partition.get("backbone_training_donors")) != expected
            or donors(partition.get("decoder_training_donors")) != expected
            or partition.get("all_donor_behavior_role") != "not_applicable"
        ):
            raise ValueError("true-LOO backbone/decoder training donors are invalid")
        mapping = partition.get("held_out_mapping")
        if (
            not isinstance(mapping, Mapping)
            or mapping.get("method") != "SCANVI.load_query_data_without_query_training"
            or any(
                mapping.get(key) is not expected_value
                for key, expected_value in {
                    "labels_available_to_query_model": False,
                    "query_train_called": False,
                    "query_parameters_frozen_before_inference": True,
                    "inference_guard_enabled_without_optimization": True,
                    "held_out_expression_used_for_fitting": False,
                    "held_out_annotation_used_for_label_mapping": False,
                }.items()
            )
            or mapping.get("label_mapping_method") != "frozen_training_donor_SCANVI_classifier"
            or donors(mapping.get("label_training_donors")) != expected
        ):
            raise ValueError("true-LOO held-out query mapping is not frozen and label-free")
        runtime = mapping.get("runtime_audit")
        if (
            not isinstance(runtime, Mapping)
            or any(
                runtime.get(key) is not expected_value
                for key, expected_value in {
                    "labels_removed_before_registry_transfer": True,
                    "query_train_called": False,
                    "parameters_frozen_before_inference": True,
                    "inference_guard_enabled_without_optimization": True,
                    "label_predictions_generated_without_target_annotation": True,
                }.items()
            )
            or runtime.get("label_prediction_rule") != "SCANVI.predict(soft=False)"
        ):
            raise ValueError("true-LOO provenance lacks its runtime frozen-query audit")
        if (
            int(runtime.get("frozen_parameter_count", 0)) <= 0
            or int(runtime.get("cells_mapped", 0)) <= 0
        ):
            raise ValueError("true-LOO runtime frozen-query audit has invalid counts")
        fit_counts = partition.get("fit_cell_counts")
        if (
            not isinstance(fit_counts, Mapping)
            or set(fit_counts) != set(expected)
            or any(
                isinstance(fit_counts.get(sample), bool)
                or not isinstance(fit_counts.get(sample), int)
                or int(fit_counts[sample]) <= 0
                for sample in expected
            )
        ):
            raise ValueError("true-LOO provenance has invalid molecular fit-cell counts")
        return R2TrainingScope(
            training_donors=expected,
            held_out_sample=held_out,
            active_samples=(held_out,),
            true_leave_one_donor_out=True,
        )

    if status not in {
        "native_scanvi_with_specimen_biology_preserved",
        "native_scanvi_specimen_batch_sensitivity",
    }:
        raise ValueError("native scANVI v2 status is unsupported")
    if provenance.get("analysis_role") not in {None, "historical_all_donor_negative_control"}:
        raise ValueError("historical all-donor scANVI provenance has the wrong analysis role")
    if partition is not None:
        if not isinstance(partition, Mapping) or (
            partition.get("mode") != "historical_all_donor_negative_control"
            or partition.get("held_out_sample") is not None
            or donors(partition.get("backbone_training_donors")) != SAMPLES
        ):
            raise ValueError("historical all-donor scANVI training partition is invalid")
    return R2TrainingScope(
        training_donors=SAMPLES,
        held_out_sample=None,
        active_samples=SAMPLES,
        true_leave_one_donor_out=False,
    )


def _validate_r2_count_binding(
    *,
    provenance: Mapping[str, object],
    scope: R2TrainingScope,
    sample: str,
    observed_cells: int,
    declared_latent: Mapping[str, object],
) -> None:
    """Cross-check declared fit/query counts against the loaded RNA artifact."""

    if isinstance(observed_cells, bool) or int(observed_cells) <= 0:
        raise ValueError("observed RNA cell count must be positive")
    observed = int(observed_cells)
    declared = declared_latent.get("cells")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared != observed:
        raise ValueError("%s latent cell count differs from its RNA reference" % sample)
    if not scope.true_leave_one_donor_out:
        return
    partition = provenance.get("training_partition")
    if not isinstance(partition, Mapping):
        raise ValueError("true-LOO provenance lacks its training partition")
    if sample in scope.training_donors:
        fit_counts = partition.get("fit_cell_counts")
        if not isinstance(fit_counts, Mapping) or fit_counts.get(sample) != observed:
            raise ValueError("%s fit-cell count differs from its RNA reference" % sample)
    elif sample == scope.held_out_sample:
        mapping = partition.get("held_out_mapping")
        runtime = mapping.get("runtime_audit") if isinstance(mapping, Mapping) else None
        if not isinstance(runtime, Mapping) or runtime.get("cells_mapped") != observed:
            raise ValueError("held-out mapped-cell count differs from its RNA reference")
    else:
        raise ValueError("RNA sample is outside the true-LOO fold")


def _validate_r2_decoder_family(
    provenance: Mapping[str, object],
    *,
    decoder: Path,
    genes: Sequence[str],
) -> Tuple[Mapping[str, object], Tuple[Mapping[str, object], ...]]:
    design = provenance["molecular_design"]
    validation = provenance["decoder_validation"]
    assert isinstance(design, Mapping) and isinstance(validation, Mapping)
    policy = str(validation["policy"])
    scope = _r2_training_scope(provenance)
    correction_mode = design.get("batch_correction_mode")
    transform_batch = design.get("transform_batch")
    if policy == "true_leave_one_donor_out_training_donor_stratified":
        if not scope.true_leave_one_donor_out:
            raise ValueError("true-LOO decoder policy requires a held-out molecular fit")
        canonical_donors = scope.training_donors
    elif policy == "donor_rotated_audit_plus_stratified_deployment_split":
        canonical_donors = SAMPLES
    else:
        canonical_donors = tuple(
            sample for sample in SAMPLES if sample != str(validation.get("single_donor_sample"))
        )
    deployed = _decoder_metadata(decoder, "deployed R2 decoder")
    _validate_decoder_metadata(
        deployed,
        label="deployed R2 decoder",
        genes=genes,
        latent_space_id=str(provenance["latent_space_id"]),
        correction_mode=correction_mode,
        transform_batch=transform_batch,
        training_donors=canonical_donors,
    )
    declared_contract = provenance.get("decoder_contract")
    if not isinstance(declared_contract, Mapping):
        raise ValueError("native scANVI R2 provenance lacks its embedded decoder contract")
    contract_keys = [
        "schema",
        "batch_correction_mode",
        "posterior_samples",
        "distillation_latent_sha256",
        "distillation_target_sha256",
        "validation_mask_sha256",
    ]
    if scope.true_leave_one_donor_out or "training_donors" in declared_contract:
        contract_keys.append("training_donors")
    for key in contract_keys:
        if declared_contract.get(key) != deployed.get(key):
            raise ValueError("native scANVI R2 decoder contract is stale for %s" % key)

    raw_rotations = validation.get("rotations")
    rotations: List[Mapping[str, object]] = []
    if policy.startswith("donor_rotated"):
        if not isinstance(raw_rotations, list) or len(raw_rotations) != len(SAMPLES):
            raise ValueError("native scANVI donor-rotation audit is incomplete")
        seen_paths = set()
        for row in raw_rotations:
            if not isinstance(row, Mapping):
                raise ValueError("native scANVI donor rotation record is invalid")
            held_out = str(row.get("held_out_sample", ""))
            path = Path(str(row.get("path", ""))).expanduser().resolve()
            if held_out not in SAMPLES or path in seen_paths or not path.is_file():
                raise ValueError("native scANVI donor rotation identity is invalid")
            seen_paths.add(path)
            if row.get("sha256") != _sha256(path):
                raise ValueError("native scANVI donor rotation SHA-256 is stale")
            metadata = _decoder_metadata(path, "%s donor-rotation decoder" % held_out)
            _validate_decoder_metadata(
                metadata,
                label="%s donor-rotation decoder" % held_out,
                genes=genes,
                latent_space_id=str(provenance["latent_space_id"]),
                correction_mode=correction_mode,
                transform_batch=transform_batch,
                training_donors=[sample for sample in SAMPLES if sample != held_out],
            )
            for key in ("distillation_latent_sha256", "distillation_target_sha256"):
                if metadata.get(key) != deployed.get(key):
                    raise ValueError("donor rotations do not share one frozen %s" % key)
            metrics = row.get("validation")
            if not isinstance(metrics, Mapping) or int(metrics.get("cells", 0)) <= 0:
                raise ValueError("native scANVI donor rotation metrics are incomplete")
            if any(
                not np.isfinite(float(metrics.get(key, np.nan)))
                for key in ("smooth_l1", "median_absolute_error")
            ):
                raise ValueError("native scANVI donor rotation metrics are non-finite")
            rotations.append(
                {"held_out_sample": held_out, "path": str(path), "sha256": _sha256(path)}
            )
        if {str(row["held_out_sample"]) for row in rotations} != set(SAMPLES):
            raise ValueError("native scANVI donor-rotation audit is incomplete")
    elif raw_rotations not in ([], None):
        raise ValueError("single-donor decoder sensitivity cannot declare rotation artifacts")
    return deployed, tuple(rotations)


@dataclass(frozen=True)
class SamplePaths:
    sample: str
    source: Path
    scanvi: Path
    scanvi_input: Path
    molecular_generation: str = "r2"

    @property
    def reference(self) -> Path:
        return self.scanvi / "reference500_scanvi.npz"

    @property
    def source_h5ad(self) -> Path:
        return self.scanvi_input / "reference.h5ad"

    @property
    def prototypes(self) -> Path:
        return self.scanvi / "prototypes_rare_complete.npz"

    @property
    def geometry(self) -> Path:
        # Keep v2 geometry beside, rather than overwriting, the historical v1
        # artifact used by the reported R1 benchmark.
        return self.scanvi / "residual_geometry_rare_complete_v2.npz"

    def histology(self, role: str) -> Path:
        return self.source / ("histology_%s.npz" % role)

    @property
    def split(self) -> Path:
        return self.source / "histology_split.json"

    @property
    def ood(self) -> Path:
        return self.source / "ood_target_calibrated.npz"

    @property
    def ood_provenance(self) -> Path:
        return self.source / "ood_target_calibrated.provenance.json"

    def batch(self, role: str) -> Path:
        return self.scanvi / ("batch_%s_rare_complete.npz" % role)


def _validate_target_ood_calibration(
    *,
    paths: SamplePaths,
    sample: str,
    full: HistologyBag,
) -> Path:
    """Validate that target scores are telemetry around a frozen development detector."""

    detector = MahalanobisOOD.from_npz(paths.ood)
    provenance = _json(paths.ood_provenance)
    if (
        provenance.get("schema") != TARGET_OOD_SCHEMA
        or provenance.get("sample_id") != sample
        or provenance.get("target_expression_accessed") is not False
        or provenance.get("calibration_input_modality") != TARGET_OOD_INPUT_MODALITY
        or provenance.get("threshold_source") != TARGET_OOD_THRESHOLD_SOURCE
    ):
        raise ValueError("%s OOD calibration provenance is invalid" % sample)

    output_record = _declared_file(
        provenance.get("output"),
        paths.ood,
        "%s calibrated OOD" % sample,
    )
    if (
        output_record.get("contract") != MahalanobisOOD.CONTRACT
        or output_record.get("contract_version") != MahalanobisOOD.CONTRACT_VERSION
    ):
        raise ValueError("%s calibrated OOD output contract is invalid" % sample)

    inputs = provenance.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("%s OOD calibration input provenance is missing" % sample)
    histology_record = _declared_file(
        inputs.get("histology"),
        paths.histology("full"),
        "%s OOD histology" % sample,
    )
    if (
        histology_record.get("sample_id") != sample
        or histology_record.get("nuclei") != full.n_nuclei
        or histology_record.get("feature_width") != int(full.features.shape[1])
        or histology_record.get("feature_space_id") != full.feature_space_id
    ):
        raise ValueError("%s OOD histology provenance is stale" % sample)

    base_record = inputs.get("base_ood")
    if not isinstance(base_record, Mapping) or not isinstance(base_record.get("path"), str):
        raise ValueError("%s base OOD provenance is missing" % sample)
    base_path = Path(str(base_record["path"])).expanduser().resolve()
    _declared_file(base_record, base_path, "%s base OOD" % sample)
    base = MahalanobisOOD.from_npz(base_path)
    if base.threshold is None:
        raise ValueError("%s base OOD detector lacks a development threshold" % sample)

    copied = provenance.get("copied_training_provenance")
    if not isinstance(copied, Mapping):
        raise ValueError("%s OOD training provenance is missing" % sample)
    if (
        detector.feature_space_id != full.feature_space_id
        or base.feature_space_id != full.feature_space_id
        or detector.training_donors != ("B1",)
        or detector.training_donors != base.training_donors
        or detector.source_sha256 != base.source_sha256
        or not np.array_equal(detector.mean, base.mean)
        or not np.array_equal(detector.precision, base.precision)
        or detector.threshold != base.threshold
        or detector.quantile != base.quantile
        or tuple(copied.get("training_donors", ())) != detector.training_donors
        or tuple(copied.get("source_sha256", ())) != detector.source_sha256
        or copied.get("feature_space_id") != detector.feature_space_id
        or provenance.get("threshold") != base.threshold
        or base_record.get("threshold") != base.threshold
        or base_record.get("quantile") != base.quantile
    ):
        raise ValueError("%s calibrated OOD detector differs from its development source" % sample)

    try:
        descriptive_quantile = float(provenance["descriptive_target_quantile"])
        descriptive_value = float(provenance["descriptive_target_quantile_value"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("%s OOD target-score telemetry is malformed" % sample) from error
    if not 0.0 < descriptive_quantile < 1.0 or not np.isfinite(descriptive_value):
        raise ValueError("%s OOD target-score telemetry is malformed" % sample)
    target_scores = np.empty(full.n_nuclei, dtype=np.float32)
    for start in range(0, full.n_nuclei, 2048):
        stop = min(start + 2048, full.n_nuclei)
        target_scores[start:stop] = base.score(full.features[start:stop])
    scores = target_scores.astype(np.float64)
    expected_stats = {
        "count": int(scores.size),
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
        "mean": float(scores.mean()),
        "standard_deviation": float(scores.std()),
        "median": float(np.quantile(scores, 0.5)),
        "descriptive_target_quantile": descriptive_quantile,
        "descriptive_target_quantile_value": float(np.quantile(scores, descriptive_quantile)),
    }
    score_stats = provenance.get("score_stats")
    if not isinstance(score_stats, Mapping):
        raise ValueError("%s OOD target-score telemetry is missing" % sample)
    for name, expected in expected_stats.items():
        observed = score_stats.get(name)
        if name == "count":
            matches = observed == expected
        else:
            try:
                matches = bool(np.isclose(float(observed), expected, rtol=0.0, atol=1.0e-12))
            except (TypeError, ValueError):
                matches = False
        if not matches:
            raise ValueError("%s OOD target-score telemetry is stale" % sample)
    if not np.isclose(
        descriptive_value,
        expected_stats["descriptive_target_quantile_value"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("%s OOD target-score telemetry is stale" % sample)

    with np.load(paths.ood, allow_pickle=False) as archive:
        required = {
            "calibration_contract",
            "calibration_version",
            "base_ood_sha256",
            "histology_sha256",
            "sample_id",
            "threshold_source",
            "target_score_quantile",
            "target_score_quantile_value",
            "target_expression_accessed",
        }
        if not required.issubset(archive.files):
            raise ValueError("%s calibrated OOD artifact lacks v2 telemetry" % sample)
        embedded = {
            "contract": str(np.asarray(archive["calibration_contract"]).item()),
            "version": int(np.asarray(archive["calibration_version"]).item()),
            "base_sha256": str(np.asarray(archive["base_ood_sha256"]).item()),
            "histology_sha256": str(np.asarray(archive["histology_sha256"]).item()),
            "sample_id": str(np.asarray(archive["sample_id"]).item()),
            "threshold_source": str(np.asarray(archive["threshold_source"]).item()),
            "quantile": float(np.asarray(archive["target_score_quantile"]).item()),
            "quantile_value": float(np.asarray(archive["target_score_quantile_value"]).item()),
            "target_expression_accessed": bool(
                np.asarray(archive["target_expression_accessed"]).item()
            ),
        }
    if embedded != {
        "contract": TARGET_OOD_CONTRACT,
        "version": TARGET_OOD_CONTRACT_VERSION,
        "base_sha256": _sha256(base_path),
        "histology_sha256": _sha256(paths.histology("full")),
        "sample_id": sample,
        "threshold_source": TARGET_OOD_THRESHOLD_SOURCE,
        "quantile": descriptive_quantile,
        "quantile_value": descriptive_value,
        "target_expression_accessed": False,
    }:
        raise ValueError("%s calibrated OOD artifact telemetry differs from provenance" % sample)
    return base_path


def _native_r2_manifest(
    *,
    repository: Path,
    provenance: Mapping[str, object],
    samples: Mapping[str, SamplePaths],
    gene_panel: Path,
) -> Dict[str, object]:
    """Build the scorer-facing R2 manifest after every derived input validates."""

    if provenance.get("schema") != "heir.snpatho_scanvi_r2.v1":
        raise ValueError("R2 native manifest requires R2 scANVI provenance")
    latent_outputs = provenance.get("latent_outputs")
    if not isinstance(latent_outputs, Mapping):
        raise ValueError("R2 scANVI provenance lacks latent outputs")
    decoder_contract = provenance.get("decoder_contract")
    if not isinstance(decoder_contract, Mapping):
        raise ValueError("R2 scANVI provenance lacks its decoder contract")
    native_model = Path(str(provenance.get("native_model", ""))).expanduser().resolve()
    decoder = Path(str(provenance.get("decoder", ""))).expanduser().resolve()
    specimen_rows: Dict[str, object] = {}
    for sample in SAMPLES:
        if sample not in samples:
            continue
        paths = samples[sample]
        latent = _record(paths.reference, repository)
        prototypes = _record(paths.prototypes, repository)
        geometry = _record(paths.geometry, repository)
        declared = latent_outputs.get(sample)
        if not isinstance(declared, Mapping) or int(declared.get("cells", 0)) <= 0:
            raise ValueError("R2 scANVI provenance lacks %s cell count" % sample)
        specimen_rows[sample] = {
            "cells": int(declared["cells"]),
            "latent_reference": latent["path"],
            "latent_reference_sha256": latent["sha256"],
            "rare_complete_prototypes": prototypes["path"],
            "rare_complete_prototypes_sha256": prototypes["sha256"],
            "residual_geometry": geometry["path"],
            "residual_geometry_sha256": geometry["sha256"],
        }
    return {
        "schema": "heir.snpatho_scanvi_r2_manifest.v1",
        "molecular_generation": "r2",
        "status": provenance.get("status"),
        "workflow_filter": provenance.get("workflow_filter"),
        "annotation_provenance": provenance.get("annotation_provenance"),
        "expression_space_id": EXPRESSION_SPACE_ID,
        "expression_transform": SCVI_EXPRESSION_NORMALIZATION_CONTRACT,
        "gene_panel": _path_name(gene_panel, repository),
        "gene_panel_sha256": _sha256(gene_panel),
        "latent_space_id": provenance.get("latent_space_id"),
        "native_model": {
            "external_path": _path_name(native_model, repository),
            "sha256": provenance.get("native_model_sha256"),
            "scvi_tools_version": provenance.get("scvi_tools_version"),
            "scvi_epochs": provenance.get("scvi_epochs"),
            "scanvi_epochs": provenance.get("scanvi_epochs"),
            "latent_dim": provenance.get("latent_dim"),
            "cuda": provenance.get("cuda"),
            "molecular_design": provenance.get("molecular_design"),
        },
        "distilled_decoder": {
            "external_path": _path_name(decoder, repository),
            "sha256": provenance.get("decoder_sha256"),
            "distillation_epochs": provenance.get("decoder_epochs"),
            "posterior_samples": provenance.get("decoder_posterior_samples"),
            "contract": dict(decoder_contract),
            "validation": provenance.get("decoder_validation"),
        },
        "molecular_design": provenance.get("molecular_design"),
        "decoder_validation": provenance.get("decoder_validation"),
        "training_partition": provenance.get("training_partition"),
        "molecular_producer": provenance.get("producer"),
        "specimens": specimen_rows,
    }


@dataclass(frozen=True)
class Stage:
    sample: str
    name: str
    inputs: Tuple[Tuple[str, Path], ...]
    output: Path
    command: Callable[[Path], Tuple[str, ...]]
    validate: Callable[[Path], None]


def _validate_upstream(
    *,
    repository: Path,
    source_root: Path,
    scanvi_root: Path,
    scanvi_input_root: Path,
    provenance_path: Path,
    expected_molecular_generation: str,
) -> Tuple[Mapping[str, object], Dict[str, SamplePaths], Dict[str, object]]:
    provenance = _json(provenance_path)
    provenance_schema = provenance.get("schema")
    if provenance_schema not in {
        "heir.snpatho_scanvi_r1.v1",
        "heir.snpatho_scanvi_r2.v1",
    }:
        raise ValueError("native scANVI provenance schema is invalid")
    expected_schema = {
        "r1": "heir.snpatho_scanvi_r1.v1",
        "r2": "heir.snpatho_scanvi_r2.v1",
    }.get(expected_molecular_generation)
    if expected_schema is None:
        raise ValueError("expected molecular generation must be r1 or r2")
    if provenance_schema != expected_schema:
        raise ValueError(
            "native scANVI provenance is %s but --molecular-generation requested %s"
            % (provenance_schema, expected_molecular_generation)
        )
    if tuple(provenance.get("samples", ())) != SAMPLES:
        raise ValueError("native scANVI provenance must contain the three frozen samples")
    if provenance.get("seed") != 17 or provenance.get("cuda") is not True:
        raise ValueError("native scANVI provenance differs from the frozen CUDA/seed recipe")
    if provenance.get("latent_dim") != 32:
        raise ValueError("native scANVI provenance latent width is not 32")
    r2_scope = (
        _r2_training_scope(provenance) if provenance_schema == "heir.snpatho_scanvi_r2.v1" else None
    )
    if r2_scope is not None and r2_scope.true_leave_one_donor_out:
        _declared_file(
            provenance.get("producer"),
            (repository / "scripts" / "train_snpatho_scanvi.py").resolve(),
            "true-LOO molecular producer",
        )
    if provenance_schema == "heir.snpatho_scanvi_r2.v1":
        assert r2_scope is not None
        molecular_design = provenance.get("molecular_design")
        decoder_validation = provenance.get("decoder_validation")
        if not isinstance(molecular_design, Mapping):
            raise ValueError("native scANVI v2 provenance lacks its molecular design")
        design_name = molecular_design.get("name")
        correction_mode = molecular_design.get("batch_correction_mode")
        transform_batch = molecular_design.get("transform_batch")
        if design_name == "no_specimen_correction":
            if (
                molecular_design.get("model_batch_key") is not None
                or correction_mode != "none"
                or transform_batch != []
            ):
                raise ValueError("no-specimen-correction provenance is internally inconsistent")
        elif design_name == "technical_batch_only":
            technical_key = molecular_design.get("technical_batch_key")
            contingency = molecular_design.get("technical_batch_contingency")
            if (
                not isinstance(technical_key, str)
                or not technical_key
                or technical_key == "section_id"
                or molecular_design.get("model_batch_key") != technical_key
                or correction_mode != "reference_batch_marginalization"
                or not isinstance(transform_batch, list)
                or not transform_batch
                or not isinstance(contingency, Mapping)
            ):
                raise ValueError("technical-batch-only provenance is internally inconsistent")
            try:
                crossed = all(
                    all(int(contingency[sample][level]) > 0 for level in transform_batch)
                    for sample in r2_scope.training_donors
                )
            except (KeyError, TypeError, ValueError):
                crossed = False
            if not crossed:
                raise ValueError(
                    "technical-batch contingency is not crossed across molecular training donors"
                )
            query_contract = molecular_design.get("technical_batch_query_contract")
            if r2_scope.true_leave_one_donor_out:
                if not isinstance(query_contract, Mapping):
                    raise ValueError("true-LOO technical design lacks its frozen-query contract")
                held_out_levels = query_contract.get("held_out_levels")
                if (
                    query_contract.get("held_out_sample") != r2_scope.held_out_sample
                    or query_contract.get("reference_levels") != transform_batch
                    or not isinstance(held_out_levels, list)
                    or not held_out_levels
                    or not set(held_out_levels).issubset(set(transform_batch))
                    or query_contract.get("novel_levels") != []
                    or query_contract.get("category_extension_allowed") is not False
                ):
                    raise ValueError(
                        "true-LOO technical query categories are not locked to reference levels"
                    )
                partition = provenance.get("training_partition")
                assert isinstance(partition, Mapping)
                mapping = partition.get("held_out_mapping")
                assert isinstance(mapping, Mapping)
                runtime = mapping.get("runtime_audit")
                assert isinstance(runtime, Mapping)
                runtime_categories = runtime.get("technical_batch_categories")
                if not isinstance(runtime_categories, Mapping) or any(
                    runtime_categories.get(key) != expected
                    for key, expected in {
                        "key": technical_key,
                        "reference_levels": transform_batch,
                        "held_out_levels": held_out_levels,
                        "novel_levels": [],
                        "category_extension_allowed": False,
                    }.items()
                ):
                    raise ValueError(
                        "true-LOO technical design differs from its runtime query registry"
                    )
            elif query_contract is not None:
                raise ValueError("all-donor technical design cannot declare a held-out query")
        elif design_name == "specimen_batch_sensitivity":
            if r2_scope.true_leave_one_donor_out:
                raise ValueError("specimen-batch sensitivity is invalid for true LOO")
            if (
                molecular_design.get("model_batch_key") != "section_id"
                or correction_mode != "reference_batch_marginalization"
                or transform_batch != list(SAMPLES)
            ):
                raise ValueError("specimen-batch sensitivity provenance is internally inconsistent")
        else:
            raise ValueError("native scANVI v2 provenance names an unsupported molecular design")
        specimen_sensitivity_status = (
            provenance.get("status") == "native_scanvi_specimen_batch_sensitivity"
        )
        if specimen_sensitivity_status != (design_name == "specimen_batch_sensitivity"):
            raise ValueError("native scANVI status and molecular design disagree")
        if molecular_design.get("specimen_identity_is_biological") is not True:
            raise ValueError("native scANVI v2 provenance must identify specimen as biological")
        if not isinstance(decoder_validation, Mapping):
            raise ValueError("native scANVI v2 provenance lacks decoder validation")
        validation_policy = decoder_validation.get("policy")
        if validation_policy not in {
            "donor_rotated_audit_plus_stratified_deployment_split",
            "single_donor_sensitivity",
            "true_leave_one_donor_out_training_donor_stratified",
        }:
            raise ValueError("native scANVI v2 decoder validation policy is unsupported")
        if r2_scope.true_leave_one_donor_out != (
            validation_policy == "true_leave_one_donor_out_training_donor_stratified"
        ):
            raise ValueError("native scANVI status and decoder validation policy disagree")
        rotations = decoder_validation.get("rotations")
        if validation_policy.startswith("donor_rotated"):
            if not isinstance(rotations, list) or {
                row.get("held_out_sample") for row in rotations if isinstance(row, Mapping)
            } != set(SAMPLES):
                raise ValueError("native scANVI donor-rotation audit is incomplete")
        if r2_scope.true_leave_one_donor_out:
            evaluation = decoder_validation.get("held_out_evaluation")
            if not isinstance(evaluation, Mapping) or int(evaluation.get("cells", 0)) <= 0:
                raise ValueError("true-LOO decoder lacks held-out evaluation metrics")
            if any(
                not np.isfinite(float(evaluation.get(key, np.nan)))
                for key in ("smooth_l1", "median_absolute_error")
            ):
                raise ValueError("true-LOO held-out decoder metrics are non-finite")

    panel = (repository / "manifests" / "gene_panel_snpatho_500.tsv").resolve()
    if provenance.get("gene_panel_sha256") != _sha256(panel):
        raise ValueError("native scANVI provenance uses a different gene panel")
    native_model = Path(str(provenance.get("native_model", ""))).expanduser().resolve()
    decoder = Path(str(provenance.get("decoder", ""))).expanduser().resolve()
    model_hash = _directory_sha256(native_model)
    if provenance.get("native_model_sha256") != model_hash:
        raise ValueError("native scANVI checkpoint hash differs from its provenance")
    if provenance.get("latent_space_id") != "sha256:" + model_hash:
        raise ValueError("native scANVI latent identity differs from the checkpoint hash")
    if provenance.get("decoder_sha256") != _sha256(decoder):
        raise ValueError("distilled native scANVI decoder hash differs from its provenance")

    latent_outputs = provenance.get("latent_outputs")
    h5ad_hashes = provenance.get("input_h5ad_sha256")
    if not isinstance(latent_outputs, Mapping) or not isinstance(h5ad_hashes, Mapping):
        raise ValueError("native scANVI provenance lacks reference hashes")

    samples: Dict[str, SamplePaths] = {}
    input_records: Dict[str, object] = {
        "scanvi_provenance": _record(provenance_path, repository),
        "gene_panel": _record(panel, repository),
        "native_model": {
            "path": _path_name(native_model, repository),
            "sha256": model_hash,
        },
        "decoder": _record(decoder, repository),
    }
    genes = tuple(
        line.split("\t", 1)[0].strip()
        for line in panel.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    if len(genes) != len(set(genes)) or not genes:
        raise ValueError("frozen gene panel is empty or contains duplicates")
    if provenance_schema == "heir.snpatho_scanvi_r2.v1":
        decoder_metadata, decoder_rotations = _validate_r2_decoder_family(
            provenance,
            decoder=decoder,
            genes=genes,
        )
        input_records["decoder_contract"] = {
            "schema": decoder_metadata["schema"],
            "batch_correction_mode": decoder_metadata["batch_correction_mode"],
            "posterior_samples": decoder_metadata["posterior_samples"],
            "distillation_latent_sha256": decoder_metadata["distillation_latent_sha256"],
            "distillation_target_sha256": decoder_metadata["distillation_target_sha256"],
        }
        input_records["decoder_rotations"] = list(decoder_rotations)

    for sample in SAMPLES:
        paths = SamplePaths(
            sample=sample,
            source=(source_root / sample).resolve(),
            scanvi=(scanvi_root / sample).resolve(),
            scanvi_input=(scanvi_input_root / sample).resolve(),
            molecular_generation=(
                "r2" if provenance_schema == "heir.snpatho_scanvi_r2.v1" else "r1"
            ),
        )
        samples[sample] = paths
        declared_latent = _declared_file(
            latent_outputs.get(sample),
            paths.reference,
            "%s latent reference" % sample,
        )
        if h5ad_hashes.get(sample) != _sha256(paths.source_h5ad):
            raise ValueError("%s source H5AD hash differs from scANVI provenance" % sample)

        reference = RNAReference.load_npz(paths.reference)
        if paths.molecular_generation == "r2" and r2_scope is not None:
            _validate_r2_count_binding(
                provenance=provenance,
                scope=r2_scope,
                sample=sample,
                observed_cells=reference.shape[0],
                declared_latent=declared_latent,
            )
        if (
            reference.sample_id != sample
            or set(reference.donor_ids.tolist()) != {sample}
            or reference.block_id != sample + "_FFPE"
        ):
            raise ValueError("%s native RNA reference identity is invalid" % sample)
        if reference.latent.shape != (reference.shape[0], 32):
            raise ValueError("%s native RNA reference has the wrong latent shape" % sample)
        if reference.latent_space_id != provenance["latent_space_id"]:
            raise ValueError("%s native RNA reference has a foreign latent identity" % sample)
        expected_latent_donors = (
            r2_scope.training_donors
            if paths.molecular_generation == "r2" and r2_scope is not None
            else ()
        )
        expected_latent_transform = model_hash if paths.molecular_generation == "r2" else ""
        if (
            reference.latent_training_donors != expected_latent_donors
            or reference.latent_transform_sha256 != expected_latent_transform
        ):
            raise ValueError("%s native RNA latent-training provenance is invalid" % sample)
        fit_scope_fields = {
            "inference_role",
            "latent_training_donors",
            "sample_expression_used_for_model_fitting",
            "sample_labels_used_for_model_fitting",
            "cell_type_label_source",
            "cell_type_label_training_donors",
            "sample_annotation_used_for_cell_type_labels",
            "cell_type_labels_sha256",
        }
        if (
            paths.molecular_generation == "r2"
            and r2_scope is not None
            and (r2_scope.true_leave_one_donor_out or fit_scope_fields & set(declared_latent))
        ):
            expected_role = (
                "frozen_query_encoder"
                if sample == r2_scope.held_out_sample
                else "reference_encoder"
            )
            sample_was_fit = sample in r2_scope.training_donors
            expected_label_source = (
                "frozen_training_donor_SCANVI_classifier"
                if sample == r2_scope.held_out_sample
                else "published_training_donor_annotation"
            )
            if (
                declared_latent.get("inference_role") != expected_role
                or tuple(declared_latent.get("latent_training_donors", ()))
                != r2_scope.training_donors
                or declared_latent.get("sample_expression_used_for_model_fitting")
                is not sample_was_fit
                or declared_latent.get("sample_labels_used_for_model_fitting") is not sample_was_fit
                or declared_latent.get("cell_type_label_source") != expected_label_source
                or tuple(declared_latent.get("cell_type_label_training_donors", ()))
                != r2_scope.training_donors
                or declared_latent.get("sample_annotation_used_for_cell_type_labels")
                is not sample_was_fit
                or declared_latent.get("cell_type_labels_sha256")
                != _ordered_string_sha256(reference.cell_type_labels)
            ):
                raise ValueError("%s emitted latent fit-scope provenance is invalid" % sample)
            if sample == r2_scope.held_out_sample and sample in set(
                reference.latent_training_donors
            ):
                raise ValueError("held-out target donor leaked into latent_training_donors")
            if sample == r2_scope.held_out_sample:
                partition = provenance.get("training_partition")
                mapping = (
                    partition.get("held_out_mapping") if isinstance(partition, Mapping) else None
                )
                runtime = mapping.get("runtime_audit") if isinstance(mapping, Mapping) else None
                label_counts = {
                    str(label): int(count)
                    for label, count in zip(
                        *np.unique(reference.cell_type_labels.astype(str), return_counts=True)
                    )
                }
                label_ontology = (
                    set(str(value) for value in partition.get("label_ontology", ()))
                    if isinstance(partition, Mapping)
                    else set()
                )
                if (
                    not isinstance(runtime, Mapping)
                    or runtime.get("predicted_label_order") != "RNAReference.cell_ids"
                    or runtime.get("predicted_label_sha256")
                    != declared_latent.get("cell_type_labels_sha256")
                    or runtime.get("predicted_label_counts") != label_counts
                    or not set(label_counts).issubset(label_ontology)
                ):
                    raise ValueError(
                        "held-out frozen classifier label provenance differs from its RNA reference"
                    )
        if reference.source_count_sha256 != h5ad_hashes[sample]:
            raise ValueError("%s native RNA reference is bound to a different H5AD" % sample)
        if tuple(reference.gene_ids.tolist()) != genes:
            raise ValueError("%s native RNA reference differs from the frozen gene panel" % sample)

        full = HistologyBag.load_npz(paths.histology("full"))
        train = HistologyBag.load_npz(paths.histology("train"))
        validation = HistologyBag.load_npz(paths.histology("validation"))
        for role, bag in (("full", full), ("train", train), ("validation", validation)):
            if (bag.sample_id, bag.donor_id, bag.block_id) != (
                sample,
                sample,
                sample + "_FFPE",
            ):
                raise ValueError("%s %s HistologyBag identity is invalid" % (sample, role))
        if len({full.feature_space_id, train.feature_space_id, validation.feature_space_id}) != 1:
            raise ValueError("%s spatial split uses different feature spaces" % sample)
        train_ids = set(train.nucleus_ids.tolist())
        validation_ids = set(validation.nucleus_ids.tolist())
        full_ids = set(full.nucleus_ids.tolist())
        if train_ids & validation_ids or not (train_ids | validation_ids).issubset(full_ids):
            raise ValueError("%s spatial train/validation split is not disjoint" % sample)

        split = _json(paths.split)
        declared_split_paths = {
            "input": paths.histology("full"),
            "train_output": paths.histology("train"),
            "validation_output": paths.histology("validation"),
        }
        if any(
            Path(str(split.get(key, ""))).expanduser().resolve() != expected
            for key, expected in declared_split_paths.items()
        ):
            raise ValueError("%s split summary names different HistologyBags" % sample)
        if (
            split.get("seed") != 17
            or split.get("nucleus_overlap") != 0
            or split.get("total_nuclei") != full.n_nuclei
            or split.get("train_nuclei") != train.n_nuclei
            or split.get("validation_nuclei") != validation.n_nuclei
        ):
            raise ValueError("%s spatial split summary is stale" % sample)

        base_path = _validate_target_ood_calibration(paths=paths, sample=sample, full=full)

        input_records[sample] = {
            "source_h5ad": _record(paths.source_h5ad, repository),
            "reference": _record(paths.reference, repository),
            "histology_full": _record(paths.histology("full"), repository),
            "histology_train": _record(paths.histology("train"), repository),
            "histology_validation": _record(paths.histology("validation"), repository),
            "histology_split": _record(paths.split, repository),
            "calibrated_ood": _record(paths.ood, repository),
            "calibrated_ood_provenance": _record(paths.ood_provenance, repository),
            "base_ood": _record(base_path, repository),
        }
    if r2_scope is not None:
        samples = {sample: samples[sample] for sample in r2_scope.active_samples}
    return provenance, samples, input_records


def _validate_prototypes(output: Path, paths: SamplePaths) -> None:
    reference = RNAReference.load_npz(paths.reference)
    prototypes = PrototypeSet.load_npz(output)
    expected_types = set(reference.cell_type_labels.tolist())
    if set(prototypes.cell_type_labels.tolist()) != expected_types:
        raise ValueError("rare-complete prototypes omit or add an RNA cell type")
    if (
        set(prototypes.sample_ids.tolist()) != {paths.sample}
        or prototypes.donor_id != paths.sample
        or prototypes.block_id != paths.sample + "_FFPE"
        or prototypes.source_reference_sha256 != _sha256(paths.reference)
        or prototypes.latent_space_id != reference.latent_space_id
        or prototypes.latent_training_donors != reference.latent_training_donors
        or prototypes.latent_transform_sha256 != reference.latent_transform_sha256
    ):
        raise ValueError("prototype provenance differs from the native RNA reference")


def _validate_geometry(output: Path, paths: SamplePaths) -> None:
    reference = RNAReference.load_npz(paths.reference)
    geometry = RNAResidualGeometry.from_npz(output)
    expected_types = tuple(sorted(set(reference.cell_type_labels.tolist())))
    if (
        tuple(geometry.type_names.tolist()) != expected_types
        or geometry.rank != RECIPE["residual_geometry"]["rank"]
        or geometry.latent_space_id != reference.latent_space_id
        or geometry.source_reference_sha256 != _sha256(paths.reference)
        or geometry.training_donors
        != tuple(sorted(set((paths.sample, *reference.latent_training_donors))))
        or geometry.latent_transform_sha256 != reference.latent_transform_sha256
    ):
        raise ValueError("RNA residual geometry provenance or ontology is invalid")


def _validate_batch(output: Path, paths: SamplePaths, role: str) -> None:
    reference = RNAReference.load_npz(paths.reference)
    prototypes = PrototypeSet.load_npz(paths.prototypes)
    histology = HistologyBag.load_npz(paths.histology(role))
    detector = MahalanobisOOD.from_npz(paths.ood)
    batch = HEIRTrainingBatch.load_npz(output)
    batch.validate(TrainingStage.PERSONALIZED)
    expected_sources = (
        paths.histology(role),
        paths.prototypes,
        paths.reference,
        paths.ood,
    )
    if tuple(batch.source_artifacts) != tuple(str(path.resolve()) for path in expected_sources):
        raise ValueError("training batch names different source artifacts")
    if tuple(batch.source_sha256) != tuple(_sha256(path) for path in expected_sources):
        raise ValueError("training batch source hashes are stale")
    if batch.source_roles != (
        "sample_assay",
        "sample_assay",
        "sample_assay",
        "shared_teacher",
    ):
        raise ValueError("training batch source roles are invalid")
    if (
        (batch.sample_id, batch.bag_id, batch.donor_id, batch.block_id)
        != (paths.sample, "%s_%s" % (paths.sample, role), paths.sample, paths.sample + "_FFPE")
        or batch.analysis_role != RECIPE["batch"]["analysis_role"]
        or batch.latent_space_id != reference.latent_space_id
        or batch.feature_space_id != histology.feature_space_id
        or tuple(batch.nucleus_ids) != tuple(histology.nucleus_ids.tolist())
        or tuple(batch.type_names) != tuple(sorted(set(reference.cell_type_labels.tolist())))
        or tuple(batch.gene_names) != tuple(reference.gene_ids.tolist())
        or tuple(batch.prototype_ids) != tuple(prototypes.prototype_ids.tolist())
        or batch.molecular_training_donors
        != tuple(sorted(set(prototypes.latent_training_donors) | set(detector.training_donors)))
    ):
        raise ValueError("training batch identity, ontology, or feature provenance is invalid")


def build_stages(
    *,
    samples: Mapping[str, SamplePaths],
    heir_command: str,
    unsafe_allow_legacy_latent: bool = False,
) -> Tuple[Stage, ...]:
    stages = []
    unknown_samples = sorted(set(samples) - set(SAMPLES))
    if unknown_samples:
        raise ValueError("stage plan contains unknown snPATHO samples: %s" % unknown_samples)
    for sample in SAMPLES:
        if sample not in samples:
            continue
        paths = samples[sample]

        def prototype_command(output: Path, p: SamplePaths = paths) -> Tuple[str, ...]:
            command = (
                heir_command,
                "build-prototypes",
                "--reference",
                str(p.reference),
                "--output",
                str(output),
                "--max-per-type",
                "10",
                "--minimum-cells",
                "50",
                "--shrinkage-kappa",
                "50",
                "--include-rare-types",
                "--seed",
                "17",
            )
            if unsafe_allow_legacy_latent:
                command += ("--unsafe-allow-legacy-latent-transform",)
            return command

        stages.append(
            Stage(
                sample,
                "prototypes",
                (("reference", paths.reference),),
                paths.prototypes,
                prototype_command,
                lambda output, p=paths: _validate_prototypes(output, p),
            )
        )

        def geometry_command(output: Path, p: SamplePaths = paths) -> Tuple[str, ...]:
            return (
                heir_command,
                "fit-residual-geometry",
                "--reference",
                str(p.reference),
                "--prototypes",
                str(p.prototypes),
                "--output",
                str(output),
                "--rank",
                "4",
                "--calibration-quantile",
                "0.90",
                "--bound-fraction",
                "0.50",
                "--minimum-bound",
                "0.001",
                "--minimum-calibration-cells",
                "3",
            )

        stages.append(
            Stage(
                sample,
                "residual_geometry_v2",
                (("reference", paths.reference), ("prototypes", paths.prototypes)),
                paths.geometry,
                geometry_command,
                lambda output, p=paths: _validate_geometry(output, p),
            )
        )
        for role in ("train", "validation"):

            def batch_command(
                output: Path, p: SamplePaths = paths, r: str = role
            ) -> Tuple[str, ...]:
                return (
                    heir_command,
                    "assemble-batch",
                    "--histology",
                    str(p.histology(r)),
                    "--prototypes",
                    str(p.prototypes),
                    "--reference",
                    str(p.reference),
                    "--ood-artifact",
                    str(p.ood),
                    "--output",
                    str(output),
                    "--sample-id",
                    p.sample,
                    "--bag-id",
                    "%s_%s" % (p.sample, r),
                    "--donor-id",
                    p.sample,
                    "--block-id",
                    p.sample + "_FFPE",
                    "--analysis-role",
                    str(RECIPE["batch"]["analysis_role"]),
                    "--artifact-threshold",
                    "0.50",
                    "--markers-per-type",
                    "25",
                )

            stages.append(
                Stage(
                    sample,
                    "batch_" + role,
                    (
                        ("histology", paths.histology(role)),
                        ("prototypes", paths.prototypes),
                        ("reference", paths.reference),
                        ("ood", paths.ood),
                    ),
                    paths.batch(role),
                    batch_command,
                    lambda output, p=paths, r=role: _validate_batch(output, p, r),
                )
            )
    return tuple(stages)


def _stage_recipe(stage: Stage, repository: Path) -> Dict[str, object]:
    return {
        "schema": RECEIPT_SCHEMA,
        "sample": stage.sample,
        "stage": stage.name,
        "inputs": {name: _record(path, repository) for name, path in stage.inputs},
        "output": _path_name(stage.output, repository),
        "recipe": RECIPE,
    }


def _complete_receipt(
    receipt_path: Path, recipe: Mapping[str, object], output: Path, repository: Path
) -> None:
    payload = dict(recipe)
    payload.update({"state": "complete", "output_artifact": _record(output, repository)})
    _atomic_json(receipt_path, payload)


def _run_stage(
    stage: Stage,
    *,
    repository: Path,
    receipt_root: Path,
    execute: bool,
    adopt_existing: bool,
) -> str:
    recipe = _stage_recipe(stage, repository)
    receipt_path = receipt_root / stage.sample / (stage.name + ".json")
    pending = receipt_root / stage.sample / (stage.name + ".pending.npz")
    receipt = _json(receipt_path) if receipt_path.is_file() else None
    if receipt is not None:
        observed_recipe = {
            key: value
            for key, value in receipt.items()
            if key
            not in {
                "state",
                "output_artifact",
            }
        }
        if observed_recipe != recipe:
            raise RuntimeError("stale stage receipt: %s" % receipt_path)
        state = receipt.get("state")
        if state not in {"planned", "complete"}:
            raise RuntimeError("invalid stage receipt state: %s" % receipt_path)
        if state == "complete":
            if pending.exists() or not stage.output.is_file():
                raise RuntimeError("partial completed stage: %s" % stage.name)
            stage.validate(stage.output)
            output_record = receipt.get("output_artifact")
            if not isinstance(output_record, Mapping) or output_record != _record(
                stage.output, repository
            ):
                raise RuntimeError("stage output hash differs from receipt: %s" % stage.name)
            return "skipped_valid"
        if stage.output.exists() and pending.exists():
            raise RuntimeError("stage has both final and pending outputs: %s" % stage.name)
        if stage.output.is_file():
            stage.validate(stage.output)
            _complete_receipt(receipt_path, recipe, stage.output, repository)
            return "recovered_final"
        if pending.is_file():
            stage.validate(pending)
            stage.output.parent.mkdir(parents=True, exist_ok=True)
            os.replace(pending, stage.output)
            _complete_receipt(receipt_path, recipe, stage.output, repository)
            return "recovered_pending"
    else:
        if stage.output.exists():
            if not stage.output.is_file():
                raise RuntimeError("stage output is not a regular file: %s" % stage.output)
            if not adopt_existing:
                raise RuntimeError(
                    "untracked stage output exists; pass --adopt-existing to reproduce and "
                    "hash-verify it: %s" % stage.output
                )
            if not execute:
                print("VERIFY " + shlex.join(stage.command(pending)))
                return "planned_adoption"
            if not pending.exists():
                pending.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(stage.command(pending), cwd=repository, check=True)
            if not pending.is_file():
                raise RuntimeError("adoption command did not emit %s" % pending)
            stage.validate(stage.output)
            stage.validate(pending)
            if _sha256(pending) != _sha256(stage.output):
                raise RuntimeError(
                    "existing output is not byte-identical to the frozen CLI recipe: %s"
                    % stage.output
                )
            pending.unlink()
            _complete_receipt(receipt_path, recipe, stage.output, repository)
            return "adopted_reproduced"
        if pending.exists():
            raise RuntimeError("untracked pending stage output exists: %s" % pending)

    command = stage.command(pending)
    if not execute:
        print(shlex.join(stage.command(stage.output)))
        return "planned"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    planned = dict(recipe)
    planned["state"] = "planned"
    _atomic_json(receipt_path, planned)
    subprocess.run(command, cwd=repository, check=True)
    if not pending.is_file():
        raise RuntimeError("stage command did not emit %s" % pending)
    stage.validate(pending)
    stage.output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(pending, stage.output)
    _complete_receipt(receipt_path, recipe, stage.output, repository)
    return "completed"


def _parser(repository: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=repository / "artifacts/snpatho")
    parser.add_argument(
        "--molecular-generation",
        choices=("r1", "r2"),
        default="r2",
        help="R2 preserves specimen biology; use r1 only for historical reproduction",
    )
    parser.add_argument("--scanvi-root", type=Path, default=None)
    parser.add_argument("--scanvi-input-root", type=Path, default=None)
    parser.add_argument("--scanvi-provenance", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--native-manifest-output", type=Path, default=None)
    parser.add_argument("--heir-command", default="heir")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--adopt-existing",
        action="store_true",
        help="re-run the frozen CLI recipe and adopt only byte-identical legacy outputs",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    args = _parser(repository).parse_args(argv)
    source_root = args.source_root.expanduser().resolve()
    scanvi_root = (
        (source_root / ("%s_scanvi" % args.molecular_generation)).resolve()
        if args.scanvi_root is None
        else args.scanvi_root.expanduser().resolve()
    )
    scanvi_input_root = (
        (source_root / "r1_ffpe").resolve()
        if args.scanvi_input_root is None
        else args.scanvi_input_root.expanduser().resolve()
    )
    provenance_path = (
        scanvi_root / "provenance.json"
        if args.scanvi_provenance is None
        else args.scanvi_provenance.expanduser().resolve()
    )
    manifest_output = (
        scanvi_root / "preparation_manifest.json"
        if args.manifest_output is None
        else args.manifest_output.expanduser().resolve()
    )
    native_manifest_output = (
        scanvi_root / "native_manifest.json"
        if args.native_manifest_output is None
        else args.native_manifest_output.expanduser().resolve()
    )
    provenance, samples, input_records = _validate_upstream(
        repository=repository,
        source_root=source_root,
        scanvi_root=scanvi_root,
        scanvi_input_root=scanvi_input_root,
        provenance_path=provenance_path,
        expected_molecular_generation=args.molecular_generation,
    )
    validated_generation = "r2" if provenance["schema"] == "heir.snpatho_scanvi_r2.v1" else "r1"
    stages = build_stages(
        samples=samples,
        heir_command=args.heir_command,
        unsafe_allow_legacy_latent=validated_generation == "r1",
    )
    receipt_root = scanvi_root / ".preparation_receipts"
    statuses = []
    for stage in stages:
        status = _run_stage(
            stage,
            repository=repository,
            receipt_root=receipt_root,
            execute=args.execute,
            adopt_existing=args.adopt_existing,
        )
        statuses.append({"sample": stage.sample, "stage": stage.name, "status": status})

    if args.execute:
        outputs = {
            sample: {
                "prototypes": _record(paths.prototypes, repository),
                "residual_geometry": _record(paths.geometry, repository),
                "batch_train": _record(paths.batch("train"), repository),
                "batch_validation": _record(paths.batch("validation"), repository),
            }
            for sample, paths in samples.items()
        }
        native_manifest_record = None
        if validated_generation == "r2":
            gene_panel = (repository / "manifests" / "gene_panel_snpatho_500.tsv").resolve()
            native_manifest = _native_r2_manifest(
                repository=repository,
                provenance=provenance,
                samples=samples,
                gene_panel=gene_panel,
            )
            if native_manifest_output.exists():
                if _json(native_manifest_output) != native_manifest:
                    raise RuntimeError(
                        "existing native R2 manifest is stale: %s" % native_manifest_output
                    )
            else:
                _atomic_json(native_manifest_output, native_manifest)
            native_manifest_record = _record(native_manifest_output, repository)
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "producer": _record(Path(__file__).resolve(), repository),
            "molecular_generation": validated_generation,
            "latent_space_id": provenance["latent_space_id"],
            "recipe": RECIPE,
            "inputs": input_records,
            "outputs": outputs,
            "native_manifest": native_manifest_record,
        }
        if manifest_output.exists():
            if _json(manifest_output) != manifest:
                raise RuntimeError("existing preparation manifest is stale: %s" % manifest_output)
        else:
            _atomic_json(manifest_output, manifest)
    print(json.dumps({"execute": args.execute, "stages": statuses}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
