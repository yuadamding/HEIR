#!/usr/bin/env python3
"""Evaluate image-conditioned utility while substituting only the molecular bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.evaluation import MORPHOLOGY_RIDGE_REPORT_SCHEMA, evaluate_reference_utility
from heir.utils import atomic_json_dump, reject_output_input_collisions, sha256_file


def _scalar(archive: Mapping[str, np.ndarray], name: str) -> object:
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise ValueError("reference-utility %s must be scalar" % name)
    return value.item()


def _load_prerequisites(paths: Sequence[Path]) -> list[Mapping[str, object]]:
    if len(paths) < 2:
        raise ValueError("reference utility requires primary and external morphology reports")
    result = []
    roles = []
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("morphology prerequisite report is invalid") from error
        role = str(report.get("experiment_role", "")) if isinstance(report, dict) else ""
        if (
            not isinstance(report, dict)
            or report.get("schema_version") != MORPHOLOGY_RIDGE_REPORT_SCHEMA
            or report.get("component_pass") is not True
            or report.get("oracle_type_only") is not True
            or not role
        ):
            raise ValueError("reference utility requires passing frozen morphology reports")
        roles.append(role)
        result.append({"role": role, "path": str(path), "sha256": sha256_file(path)})
    if "primary_hest_uni2h" not in roles:
        raise ValueError("reference utility requires a passing primary_hest_uni2h report")
    if not any(role.startswith("external_confirmation_") for role in roles):
        raise ValueError("reference utility requires a genuinely external confirmation report")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--prerequisite-report",
        type=Path,
        action="append",
        required=True,
        help="repeat for the primary HEST and genuine external confirmation reports",
    )
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--minimum-effect", type=float, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    source = args.input.expanduser().resolve()
    prerequisite_paths = tuple(path.expanduser().resolve() for path in args.prerequisite_report)
    output = args.report_output.expanduser().resolve()
    inputs = (source, *prerequisite_paths)
    if any(not path.is_file() for path in inputs) or len(set(inputs)) != len(inputs):
        raise ValueError("reference-utility inputs must be distinct existing files")
    reject_output_input_collisions((output,), inputs, label="matched reference utility")
    before = {str(path): sha256_file(path) for path in inputs}
    prerequisites = _load_prerequisites(prerequisite_paths)

    with np.load(source, allow_pickle=False) as archive:
        required = {
            "schema_version",
            "query_image_state_latent",
            "query_molecular_target_latent",
            "query_types",
            "query_donors",
            "query_observation_ids",
            "query_section_ids",
            "query_disease_states",
            "query_site_ids",
            "query_assay_ids",
            "query_quality_bins",
            "query_depth_bins",
            "frozen_image_model_sha256",
            "query_source_sha256",
            "bank_names",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("reference-utility input is missing: %s" % ", ".join(missing))
        if str(_scalar(archive, "schema_version")) != "heir.reference_utility_input.v1":
            raise ValueError("reference-utility input schema is unsupported")
        names = tuple(str(value) for value in archive["bank_names"])
        if not names or len(set(names)) != len(names):
            raise ValueError("reference-utility bank names must be non-empty and unique")
        banks = {}
        bank_fields = (
            "role",
            "latent",
            "type_labels",
            "donor_ids",
            "observation_ids",
            "section_ids",
            "disease_states",
            "site_ids",
            "assay_ids",
            "quality_bins",
            "depth_bins",
            "latent_model_sha256",
            "source_sha256",
        )
        for index, name in enumerate(names):
            payload = {}
            for field in bank_fields:
                key = "bank_%d_%s" % (index, field)
                if key not in archive.files:
                    raise ValueError("reference-utility bank is incomplete: %s" % name)
                value = archive[key]
                payload[field] = _scalar(archive, key) if value.ndim == 0 else np.array(value)
            banks[name] = payload
        report = evaluate_reference_utility(
            np.array(archive["query_image_state_latent"]),
            np.array(archive["query_molecular_target_latent"]),
            np.array(archive["query_types"]),
            np.array(archive["query_donors"]),
            np.array(archive["query_observation_ids"]),
            np.array(archive["query_section_ids"]),
            np.array(archive["query_disease_states"]),
            np.array(archive["query_site_ids"]),
            np.array(archive["query_assay_ids"]),
            np.array(archive["query_quality_bins"]),
            np.array(archive["query_depth_bins"]),
            banks,
            repeats=args.repeats,
            minimum_effect=args.minimum_effect,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
        )
        frozen_image_model_sha256 = str(_scalar(archive, "frozen_image_model_sha256"))
        query_source_sha256 = str(_scalar(archive, "query_source_sha256"))

    for path in inputs:
        if sha256_file(path) != before[str(path)]:
            raise RuntimeError("reference-utility input changed during execution")
    result = {
        **report,
        "input": {"path": str(source), "sha256": before[str(source)]},
        "frozen_image_model_sha256": frozen_image_model_sha256,
        "query_source_sha256": query_source_sha256,
        "morphology_prerequisites": prerequisites,
        "authorizes_uot_or_refinement": False,
        "authorizes_full_heir": False,
    }
    atomic_json_dump(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
