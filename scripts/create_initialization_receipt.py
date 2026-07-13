#!/usr/bin/env python3
"""Create a fail-closed initializer receipt from a passing validation report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

from heir.training import (
    ValidatedInitializationReceipt,
    ordered_identity_sha256,
    validate_primary_claim_exclusions,
)
from heir.utils import atomic_json_dump, reject_output_input_collisions

EVIDENCE_SCHEMA = "heir.initialization_validation_evidence.v1"


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


def _bound_path(specification: Mapping[str, object], *, base: Path) -> Path:
    path = Path(str(specification.get("path", ""))).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _validate_bound_file(
    specification: object,
    label: str,
    *,
    base: Path,
) -> tuple[Path, str]:
    if not isinstance(specification, Mapping):
        raise ValueError("validation report %s binding is malformed" % label)
    path = _bound_path(specification, base=base)
    expected_sha256 = str(specification.get("sha256", ""))
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("validation report %s hash binding is stale" % label)
    return path, expected_sha256


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    report_path = args.evidence_report.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    reject_output_input_collisions(
        (output_path,),
        (checkpoint_path, report_path),
        label="initialization receipt",
    )
    report_sha256 = _sha256(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if _sha256(report_path) != report_sha256:
        raise ValueError("initializer evidence report changed while it was being loaded")
    if not isinstance(report, Mapping) or report.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("initializer evidence report schema is invalid")
    if report.get("status") != "complete" or report.get("pass") is not True:
        raise ValueError("initializer evidence report is not a completed passing result")
    checks = report.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("initializer evidence report does not pass every required check")
    capabilities = report.get("capabilities")
    if not isinstance(capabilities, Mapping) or not all(
        capabilities.get(name) is True for name in ("broad_type", "image_to_latent")
    ):
        raise ValueError("initializer evidence report lacks required capabilities")
    bound_specs = (
        (report.get("plan"), "plan"),
        (report.get("evidence_artifact"), "evidence artifact"),
        (report.get("label_source"), "label source"),
        (report.get("latent_target_source"), "latent-target source"),
    )
    bound_paths = []
    for specification, label in bound_specs:
        if not isinstance(specification, Mapping):
            raise ValueError("validation report %s binding is malformed" % label)
        bound_paths.append(_bound_path(specification, base=report_path.parent))
    reject_output_input_collisions(
        (output_path,),
        (checkpoint_path, report_path, *bound_paths),
        label="initialization receipt",
    )
    bound_records = [
        _validate_bound_file(specification, label, base=report_path.parent)
        for specification, label in bound_specs
    ]

    checkpoint_sha256 = _sha256(checkpoint_path)
    bound_checkpoint = report.get("checkpoint")
    if not isinstance(bound_checkpoint, Mapping):
        raise ValueError("initializer evidence checkpoint binding is malformed")
    if (
        _bound_path(bound_checkpoint, base=report_path.parent) != checkpoint_path
        or bound_checkpoint.get("sha256") != checkpoint_sha256
    ):
        raise ValueError("initializer evidence is bound to a different checkpoint")
    checkpoint = _load_checkpoint(checkpoint_path)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("initializer checkpoint lacks provenance metadata")
    validate_primary_claim_exclusions(metadata, artifact="initializer checkpoint")
    type_names = tuple(str(value) for value in metadata.get("type_names", ()))
    training_donors = tuple(sorted(str(value) for value in metadata.get("training_donors", ())))
    held_out_donors = tuple(sorted(str(value) for value in report.get("held_out_donors", ())))
    if tuple(sorted(str(value) for value in report.get("training_donors", ()))) != training_donors:
        raise ValueError("initializer evidence training donors differ from checkpoint metadata")
    feature_space_id = str(metadata.get("feature_space_id", ""))
    latent_space_id = str(metadata.get("latent_space_id", ""))
    if report.get("feature_space_id") != feature_space_id:
        raise ValueError("initializer evidence feature space differs from checkpoint metadata")
    if report.get("latent_space_id") != latent_space_id:
        raise ValueError("initializer evidence latent space differs from checkpoint metadata")
    type_hash = ordered_identity_sha256(type_names)
    if report.get("type_ontology_sha256") != type_hash:
        raise ValueError("initializer evidence ontology differs from checkpoint metadata")

    payload = {
        "schema": ValidatedInitializationReceipt.SCHEMA,
        "status": "complete",
        "pass": True,
        "checkpoint_sha256": checkpoint_sha256,
        "feature_space_id": feature_space_id,
        "latent_space_id": latent_space_id,
        "type_ontology_sha256": type_hash,
        "training_donors": list(training_donors),
        "held_out_donors": list(held_out_donors),
        "capabilities": {"broad_type": True, "image_to_latent": True},
        "evidence_report": str(report_path),
        "evidence_report_sha256": report_sha256,
    }
    candidate = ValidatedInitializationReceipt(
        checkpoint_sha256=checkpoint_sha256,
        feature_space_id=feature_space_id,
        latent_space_id=latent_space_id,
        type_ontology_sha256=type_hash,
        training_donors=training_donors,
        held_out_donors=held_out_donors,
        capabilities=("broad_type", "image_to_latent"),
        evidence_report=str(report_path),
        evidence_report_sha256=report_sha256,
    )
    candidate.validate_binding(
        checkpoint_sha256=checkpoint_sha256,
        feature_space_id=feature_space_id,
        latent_space_id=latent_space_id,
        type_names=type_names,
        target_donors=held_out_donors,
        receipt_path=output_path,
    )
    for path, expected_sha256 in (
        (checkpoint_path, checkpoint_sha256),
        (report_path, report_sha256),
        *bound_records,
    ):
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise ValueError("initialization receipt input changed during creation: %s" % path)
    reject_output_input_collisions(
        (output_path,),
        (checkpoint_path, report_path, *(path for path, _ in bound_records)),
        label="initialization receipt",
    )
    atomic_json_dump(payload, output_path)
    receipt = ValidatedInitializationReceipt.load_json(output_path)
    receipt.validate_binding(
        checkpoint_sha256=checkpoint_sha256,
        feature_space_id=feature_space_id,
        latent_space_id=latent_space_id,
        type_names=type_names,
        target_donors=held_out_donors,
        receipt_path=output_path,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
