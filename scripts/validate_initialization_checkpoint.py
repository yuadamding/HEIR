#!/usr/bin/env python3
"""Validate a frozen morphology/cross-modal HEIR initializer on held-out evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from heir.models import HEIRModel
from heir.training import (
    ordered_identity_sha256,
    recompute_initialization_validation,
    validate_primary_claim_exclusions,
)
from heir.utils import atomic_json_dump, reject_output_input_collisions

PLAN_SCHEMA = "heir.initialization_validation_plan.v1"
REPORT_SCHEMA = "heir.initialization_validation_evidence.v1"
PLAN_KEYS = {
    "schema",
    "status",
    "checkpoint",
    "evaluation_artifact",
    "label_source",
    "latent_target_source",
    "held_out_donors",
    "seeds",
    "thresholds",
}
THRESHOLD_KEYS = {
    "minimum_macro_f1",
    "minimum_image_shuffle_macro_f1_delta",
    "minimum_latent_cosine",
    "minimum_image_shuffle_latent_cosine_delta",
    "maximum_latent_rmse",
    "maximum_ece",
    "maximum_brier",
    "minimum_predicted_class_occupancy_fraction",
    "minimum_per_type_support",
}
MINIMUM_ALLOWED_THRESHOLDS = {
    "minimum_macro_f1": 0.65,
    "minimum_image_shuffle_macro_f1_delta": 0.05,
    "minimum_latent_cosine": 0.0,
    "minimum_image_shuffle_latent_cosine_delta": 0.01,
    "minimum_predicted_class_occupancy_fraction": 0.75,
    "minimum_per_type_support": 2.0,
}
MAXIMUM_ALLOWED_THRESHOLDS = {"maximum_ece": 0.10, "maximum_brier": 0.25}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError("initializer checkpoint must contain a mapping")
    return value


def _scalar_string(archive: Mapping[str, np.ndarray], name: str) -> str:
    if name not in archive:
        raise ValueError("initialization evidence is missing %s" % name)
    value = np.asarray(archive[name])
    if value.ndim != 0 or not str(value.item()).strip():
        raise ValueError("initialization evidence %s must be a non-empty scalar" % name)
    return str(value.item())


def _scalar_bool(archive: Mapping[str, np.ndarray], name: str) -> bool:
    if name not in archive:
        raise ValueError("initialization evidence is missing %s" % name)
    value = np.asarray(archive[name])
    if value.ndim != 0 or value.dtype != np.bool_:
        raise ValueError("initialization evidence %s must be a boolean scalar" % name)
    return bool(value.item())


def _string_vector(archive: Mapping[str, np.ndarray], name: str) -> Tuple[str, ...]:
    if name not in archive:
        raise ValueError("initialization evidence is missing %s" % name)
    value = np.asarray(archive[name])
    if value.ndim != 1:
        raise ValueError("initialization evidence %s must be one-dimensional" % name)
    result = tuple(str(item) for item in value.tolist())
    if any(not item.strip() for item in result):
        raise ValueError("initialization evidence %s contains empty values" % name)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    plan_path = args.plan.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    reject_output_input_collisions(
        (output_path,),
        (plan_path,),
        label="initialization validation",
    )
    plan_sha256 = _sha256(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if _sha256(plan_path) != plan_sha256:
        raise ValueError("initialization validation plan changed while it was being loaded")
    if not isinstance(plan, Mapping) or set(plan) != PLAN_KEYS or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("initialization validation plan schema is invalid")
    if plan.get("status") != "ready":
        raise ValueError("initialization validation plan must have status=ready")
    base = plan_path.parent
    checkpoint_spec = plan.get("checkpoint")
    evidence_spec = plan.get("evaluation_artifact")
    label_spec = plan.get("label_source")
    latent_target_spec = plan.get("latent_target_source")
    thresholds = plan.get("thresholds")
    if not all(
        isinstance(item, Mapping)
        for item in (
            checkpoint_spec,
            evidence_spec,
            label_spec,
            latent_target_spec,
            thresholds,
        )
    ):
        raise ValueError("plan artifact bindings and thresholds must be mappings")
    assert isinstance(checkpoint_spec, Mapping)
    assert isinstance(evidence_spec, Mapping)
    assert isinstance(label_spec, Mapping)
    assert isinstance(latent_target_spec, Mapping)
    assert isinstance(thresholds, Mapping)
    for name, specification in (
        ("checkpoint", checkpoint_spec),
        ("evaluation_artifact", evidence_spec),
        ("label_source", label_spec),
        ("latent_target_source", latent_target_spec),
    ):
        if (
            set(specification) != {"path", "sha256"}
            or not isinstance(specification.get("path"), str)
            or not str(specification["path"]).strip()
            or not isinstance(specification.get("sha256"), str)
            or len(str(specification["sha256"])) != 64
            or any(character not in "0123456789abcdef" for character in specification["sha256"])
        ):
            raise ValueError("initialization plan %s binding is malformed" % name)
    raw_holdout = plan["held_out_donors"]
    if (
        not isinstance(raw_holdout, list)
        or not raw_holdout
        or any(not isinstance(value, str) or not value.strip() for value in raw_holdout)
        or len(set(raw_holdout)) != len(raw_holdout)
    ):
        raise ValueError("initialization plan held_out_donors are malformed")
    raw_seeds = plan["seeds"]
    if (
        not isinstance(raw_seeds, list)
        or len(raw_seeds) < 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_seeds
        )
        or len(set(raw_seeds)) != len(raw_seeds)
    ):
        raise ValueError("initialization validation plan seeds are malformed")
    if set(thresholds) != THRESHOLD_KEYS or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in thresholds.values()
    ):
        raise ValueError("initialization plan thresholds are malformed")
    minimum_support = thresholds["minimum_per_type_support"]
    if isinstance(minimum_support, bool) or not isinstance(minimum_support, int):
        raise ValueError("minimum_per_type_support must be an integer")
    threshold_values = {name: float(value) for name, value in thresholds.items()}
    if (
        not all(np.isfinite(value) for value in threshold_values.values())
        or not 0.65 <= threshold_values["minimum_macro_f1"] <= 1.0
        or not 0.05 <= threshold_values["minimum_image_shuffle_macro_f1_delta"] <= 1.0
        or not 0.0 <= threshold_values["minimum_latent_cosine"] <= 1.0
        or not 0.01 <= threshold_values["minimum_image_shuffle_latent_cosine_delta"] <= 2.0
        or threshold_values["maximum_latent_rmse"] <= 0.0
        or not 0.0 <= threshold_values["maximum_ece"] <= 0.10
        or not 0.0 <= threshold_values["maximum_brier"] <= 0.25
        or not 0.75 <= threshold_values["minimum_predicted_class_occupancy_fraction"] <= 1.0
        or minimum_support < 2
    ):
        raise ValueError("initialization plan thresholds violate fail-closed bounds")
    checkpoint_path = (base / str(checkpoint_spec.get("path", ""))).resolve()
    evidence_path = (base / str(evidence_spec.get("path", ""))).resolve()
    label_source_path = (base / str(label_spec.get("path", ""))).resolve()
    latent_target_source_path = (base / str(latent_target_spec.get("path", ""))).resolve()
    reject_output_input_collisions(
        (output_path,),
        (
            plan_path,
            checkpoint_path,
            evidence_path,
            label_source_path,
            latent_target_source_path,
        ),
        label="initialization validation",
    )
    checkpoint_sha256 = _sha256(checkpoint_path)
    evidence_sha256 = _sha256(evidence_path)
    label_source_sha256 = _sha256(label_source_path)
    latent_target_source_sha256 = _sha256(latent_target_source_path)
    if checkpoint_sha256 != checkpoint_spec.get("sha256"):
        raise ValueError("initializer checkpoint hash differs from the prespecified plan")
    if evidence_sha256 != evidence_spec.get("sha256"):
        raise ValueError("initialization evidence hash differs from the prespecified plan")
    if label_source_sha256 != label_spec.get("sha256"):
        raise ValueError("independent label-source hash differs from the prespecified plan")
    if latent_target_source_sha256 != latent_target_spec.get("sha256"):
        raise ValueError("registered latent-target hash differs from the prespecified plan")
    if checkpoint_sha256 in {label_source_sha256, latent_target_source_sha256}:
        raise ValueError("initializer checkpoint cannot be its own validation source")

    checkpoint = _load_checkpoint(checkpoint_path)
    model = HEIRModel.from_checkpoint(checkpoint)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("initializer checkpoint lacks provenance metadata")
    validate_primary_claim_exclusions(metadata, artifact="initializer checkpoint")
    type_names = tuple(str(value) for value in metadata.get("type_names", ()))
    training_donors = tuple(sorted(str(value) for value in metadata.get("training_donors", ())))
    if not type_names or not training_donors:
        raise ValueError("initializer checkpoint lacks type/training-donor provenance")
    feature_space_id = str(metadata.get("feature_space_id", ""))
    latent_space_id = str(metadata.get("latent_space_id", ""))
    if not feature_space_id or not latent_space_id:
        raise ValueError("initializer checkpoint lacks feature/latent-space provenance")

    with np.load(evidence_path, allow_pickle=False) as archive:
        required = {
            "morphology",
            "edge_index",
            "nucleus_ids",
            "donor_ids",
            "type_labels",
            "type_names",
            "target_latent",
            "feature_space_id",
            "latent_space_id",
            "label_source_sha256",
            "latent_target_source_sha256",
            "labels_independent_of_checkpoint",
            "latent_targets_independent_of_checkpoint",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("initialization evidence is missing: %s" % ", ".join(missing))
        morphology = np.asarray(archive["morphology"], dtype=np.float32)
        raw_edge_index = np.asarray(archive["edge_index"])
        if not np.issubdtype(raw_edge_index.dtype, np.integer):
            raise ValueError("evidence edge_index must use an integer dtype")
        edge_index = raw_edge_index.astype(np.int64, copy=False)
        edge_weight = (
            np.asarray(archive["edge_weight"], dtype=np.float32)
            if "edge_weight" in archive
            else None
        )
        nucleus_ids = _string_vector(archive, "nucleus_ids")
        donor_ids = _string_vector(archive, "donor_ids")
        raw_labels = np.asarray(archive["type_labels"])
        if not np.issubdtype(raw_labels.dtype, np.integer):
            raise ValueError("evidence type_labels must use an integer dtype")
        labels = raw_labels.astype(np.int64, copy=False)
        evidence_types = _string_vector(archive, "type_names")
        target_latent = np.asarray(archive["target_latent"], dtype=np.float32)
        evidence_feature_space = _scalar_string(archive, "feature_space_id")
        evidence_latent_space = _scalar_string(archive, "latent_space_id")
        recorded_label_sha256 = _scalar_string(archive, "label_source_sha256")
        recorded_latent_target_sha256 = _scalar_string(archive, "latent_target_source_sha256")
        labels_independent = _scalar_bool(archive, "labels_independent_of_checkpoint")
        latent_targets_independent = _scalar_bool(
            archive, "latent_targets_independent_of_checkpoint"
        )
    cells = len(morphology)
    if morphology.ndim != 2 or morphology.shape[1] != model.config.morphology_dim:
        raise ValueError("evidence morphology dimensions differ from the checkpoint")
    if not np.isfinite(morphology).all():
        raise ValueError("evidence morphology must be finite")
    if target_latent.shape != (cells, model.config.latent_dim):
        raise ValueError("evidence target_latent dimensions differ from the checkpoint")
    if not np.isfinite(target_latent).all():
        raise ValueError("evidence target_latent must be finite")
    if labels.shape != (cells,) or len(nucleus_ids) != cells or len(donor_ids) != cells:
        raise ValueError("evidence cell-level arrays do not align")
    if len(set(nucleus_ids)) != len(nucleus_ids):
        raise ValueError("evidence nucleus_ids must be unique")
    if evidence_types != type_names:
        raise ValueError("evidence type ontology differs from the checkpoint")
    if evidence_feature_space != feature_space_id or evidence_latent_space != latent_space_id:
        raise ValueError("evidence feature/latent spaces differ from the checkpoint")
    if (
        recorded_label_sha256 != label_source_sha256
        or recorded_latent_target_sha256 != latent_target_source_sha256
        or not labels_independent
        or not latent_targets_independent
    ):
        raise ValueError("initialization evidence lacks independent source provenance")
    with np.load(label_source_path, allow_pickle=False) as label_archive:
        required_labels = {
            "schema",
            "nucleus_ids",
            "donor_ids",
            "type_labels",
            "type_names",
            "independent_of_checkpoint",
        }
        missing = sorted(required_labels - set(label_archive.files))
        if missing:
            raise ValueError("independent label source is missing: %s" % ", ".join(missing))
        if _scalar_string(label_archive, "schema") != "heir.independent_initialization_labels.v1":
            raise ValueError("independent label source schema is invalid")
        if not _scalar_bool(label_archive, "independent_of_checkpoint"):
            raise ValueError("label source is not independently reviewed")
        if _string_vector(label_archive, "nucleus_ids") != nucleus_ids:
            raise ValueError("independent label-source nucleus order differs")
        if _string_vector(label_archive, "donor_ids") != donor_ids:
            raise ValueError("independent label-source donor order differs")
        if _string_vector(label_archive, "type_names") != type_names:
            raise ValueError("independent label-source ontology differs")
        source_labels = np.asarray(label_archive["type_labels"])
        if source_labels.dtype.kind not in "iu" or not np.array_equal(source_labels, labels):
            raise ValueError("evidence type labels differ from the independent label source")
    with np.load(latent_target_source_path, allow_pickle=False) as latent_archive:
        required_latent = {
            "schema",
            "nucleus_ids",
            "target_latent",
            "latent_space_id",
            "independent_of_checkpoint",
        }
        missing = sorted(required_latent - set(latent_archive.files))
        if missing:
            raise ValueError("registered latent source is missing: %s" % ", ".join(missing))
        if _scalar_string(latent_archive, "schema") != "heir.registered_image_latent_targets.v1":
            raise ValueError("registered latent-target source schema is invalid")
        if not _scalar_bool(latent_archive, "independent_of_checkpoint"):
            raise ValueError("registered latent targets are not checkpoint-independent")
        if _string_vector(latent_archive, "nucleus_ids") != nucleus_ids:
            raise ValueError("registered latent-target nucleus order differs")
        if _scalar_string(latent_archive, "latent_space_id") != latent_space_id:
            raise ValueError("registered latent-target space differs")
        source_latent = np.asarray(latent_archive["target_latent"], dtype=np.float32)
        if not np.array_equal(source_latent, target_latent):
            raise ValueError("evidence targets differ from the registered latent source")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("evidence edge_index must have shape (2, edges)")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= cells):
        raise ValueError("evidence edge_index contains out-of-range indices")
    if edge_index.size:
        donor_array = np.asarray(donor_ids)
        if np.any(donor_array[edge_index[0]] != donor_array[edge_index[1]]):
            raise ValueError("held-out evidence graph cannot connect different donors")
    if edge_weight is not None and edge_weight.shape != (edge_index.shape[1],):
        raise ValueError("evidence edge_weight must align to edge_index")
    if edge_weight is not None and (not np.isfinite(edge_weight).all() or np.any(edge_weight < 0)):
        raise ValueError("evidence edge_weight must be finite and non-negative")
    if np.any(labels < 0) or np.any(labels >= len(type_names)):
        raise ValueError("evidence type labels fall outside the checkpoint ontology")
    if set(labels.tolist()) != set(range(len(type_names))):
        raise ValueError("held-out evidence must contain every checkpoint cell type")
    held_out_donors = tuple(sorted(set(donor_ids)))
    planned_holdout = tuple(sorted(str(value) for value in plan.get("held_out_donors", ())))
    if held_out_donors != planned_holdout:
        raise ValueError("evidence donors differ from the prespecified held-out donors")
    overlap = sorted(set(training_donors) & set(held_out_donors))
    if overlap:
        raise ValueError("initializer trained on held-out evidence donors: %s" % ", ".join(overlap))

    seeds = tuple(raw_seeds)
    if args.device not in {"auto", "cpu"}:
        raise ValueError("initializer validation replay is fixed to deterministic CPU float32")
    replay = recompute_initialization_validation(
        checkpoint=checkpoint,
        morphology=morphology,
        edge_index=edge_index,
        edge_weight=edge_weight,
        labels=labels,
        target_latent=target_latent,
        donor_ids=donor_ids,
        seeds=seeds,
    )
    real_metrics = replay["metrics"]
    donor_metrics = replay["donor_metrics"]
    controls = replay["shuffle_controls"]
    if not isinstance(real_metrics, Mapping):
        raise RuntimeError("initializer validation replay returned malformed pooled metrics")

    for name, floor in MINIMUM_ALLOWED_THRESHOLDS.items():
        value = float(thresholds[name])
        if not np.isfinite(value) or value < floor:
            raise ValueError("initialization threshold %s is below the fail-closed floor" % name)
    for name, ceiling in MAXIMUM_ALLOWED_THRESHOLDS.items():
        value = float(thresholds[name])
        if not np.isfinite(value) or value > ceiling or value < 0:
            raise ValueError("initialization threshold %s exceeds the fail-closed ceiling" % name)
    maximum_rmse = float(thresholds["maximum_latent_rmse"])
    if not np.isfinite(maximum_rmse) or maximum_rmse <= 0:
        raise ValueError("maximum_latent_rmse must be finite and positive")
    checks = {
        "macro_f1": min(row["macro_f1"] for row in donor_metrics)
        >= float(thresholds["minimum_macro_f1"]),
        "image_shuffle_macro_f1_delta": min(
            row["real_minus_image_shuffle_macro_f1"] for row in controls
        )
        >= float(thresholds["minimum_image_shuffle_macro_f1_delta"]),
        "latent_cosine": min(row["latent_cosine"] for row in donor_metrics)
        >= float(thresholds["minimum_latent_cosine"]),
        "image_shuffle_latent_cosine_delta": min(
            row["real_minus_image_shuffle_latent_cosine"] for row in controls
        )
        >= float(thresholds["minimum_image_shuffle_latent_cosine_delta"]),
        "latent_rmse": max(row["latent_rmse"] for row in donor_metrics)
        <= float(thresholds["maximum_latent_rmse"]),
        "ece": max(row["ece"] for row in donor_metrics) <= float(thresholds["maximum_ece"]),
        "brier": max(row["brier"] for row in donor_metrics) <= float(thresholds["maximum_brier"]),
        "predicted_class_occupancy": min(
            row["predicted_class_occupancy_fraction"] for row in donor_metrics
        )
        >= float(thresholds["minimum_predicted_class_occupancy_fraction"]),
        "per_type_support": min(row["minimum_per_type_support"] for row in donor_metrics)
        >= float(thresholds["minimum_per_type_support"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "pass": bool(all(checks.values())),
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256},
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "evidence_artifact": {"path": str(evidence_path), "sha256": evidence_sha256},
        "label_source": {"path": str(label_source_path), "sha256": label_source_sha256},
        "latent_target_source": {
            "path": str(latent_target_source_path),
            "sha256": latent_target_source_sha256,
        },
        "feature_space_id": feature_space_id,
        "latent_space_id": latent_space_id,
        "type_ontology_sha256": ordered_identity_sha256(type_names),
        "training_donors": list(training_donors),
        "held_out_donors": list(held_out_donors),
        "capabilities": {"broad_type": True, "image_to_latent": True},
        "thresholds": {name: float(thresholds[name]) for name in sorted(THRESHOLD_KEYS)},
        "metrics": dict(real_metrics),
        "donor_metrics": donor_metrics,
        "shuffle_controls": controls,
        "checks": checks,
        "execution": {"device": "cpu-float32", "seeds": list(seeds)},
    }
    for path, expected_sha256 in (
        (plan_path, plan_sha256),
        (checkpoint_path, checkpoint_sha256),
        (evidence_path, evidence_sha256),
        (label_source_path, label_source_sha256),
        (latent_target_source_path, latent_target_source_sha256),
    ):
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise ValueError("initialization validation input changed during evaluation: %s" % path)
    reject_output_input_collisions(
        (output_path,),
        (
            plan_path,
            checkpoint_path,
            evidence_path,
            label_source_path,
            latent_target_source_path,
        ),
        label="initialization validation",
    )
    atomic_json_dump(report, output_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
