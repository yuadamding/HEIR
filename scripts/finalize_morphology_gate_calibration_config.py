#!/usr/bin/env python3
"""Bind pre-outcome H-MEAS selections into a new calibration configuration."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from heir.data.study_manifest import StudyManifest
from heir.evaluation.measurement_gate import load_passing_measurement_receipt
from heir.evaluation.power import (
    PENDING_DESIGN_BINDING_STATUS,
    build_confirmatory_design_binding,
    canonical_sha256,
    validate_confirmatory_design_binding,
    validate_exact_gate_settings,
)
from heir.utils import atomic_json_dump, reject_output_input_collisions, sha256_file


def _json_object(path: Path, name: str) -> Mapping[str, object]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("%s is not valid JSON" % name) from error
    if not isinstance(content, Mapping):
        raise ValueError("%s must be a JSON object" % name)
    return content


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--measurement-report", type=Path, required=True)
    parser.add_argument(
        "--stratum-topology",
        type=Path,
        required=True,
        help="outcome-free donor/section/type topology frozen before H-CELL opening",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    manifest_path = args.study_manifest.expanduser().resolve()
    measurement_path = args.measurement_report.expanduser().resolve()
    topology_path = args.stratum_topology.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise ValueError("completed calibration configuration already exists and is immutable")
    reject_output_input_collisions(
        (output_path,),
        (config_path, manifest_path, measurement_path, topology_path),
        label="morphology calibration design finalization",
    )

    config = _json_object(config_path, "calibration configuration")
    if set(config) != {"schema", "confidence_level", "exact_gate_settings", "thresholds"} or (
        config.get("schema") != "heir.morphology_gate_calibration_config.v2"
    ):
        raise ValueError("calibration configuration is incomplete or unsupported")
    settings = config.get("exact_gate_settings")
    if not isinstance(settings, Mapping):
        raise ValueError("calibration exact-gate settings are malformed")
    pending = validate_confirmatory_design_binding(
        settings.get("confirmatory_design_binding"),
        allow_pending=True,
    )
    if pending.get("status") != PENDING_DESIGN_BINDING_STATUS:
        raise ValueError("calibration configuration already has a completed design binding")

    manifest = StudyManifest.load(manifest_path)
    if "H-CELL" not in manifest.hypothesis_ids or manifest.status == "opened":
        raise ValueError("design finalization requires an unopened H-CELL study manifest")
    prerequisites = manifest.content.get("prerequisites")
    if not isinstance(prerequisites, Mapping):
        raise ValueError("H-CELL manifest lacks H-MEAS prerequisites")
    expected_receipt_sha = prerequisites.get("measurement_report_sha256")
    expected_manifest_sha = prerequisites.get("measurement_study_manifest_sha256")
    expected_source_sha = prerequisites.get("measurement_source_sha256")
    if not all(
        isinstance(value, str) and value
        for value in (
            expected_receipt_sha,
            expected_manifest_sha,
            expected_source_sha,
        )
    ):
        raise ValueError("H-CELL manifest must first bind the completed H-MEAS receipt")
    measurement = load_passing_measurement_receipt(
        measurement_path,
        expected_receipt_sha256=str(expected_receipt_sha),
        expected_study_manifest_sha256=str(expected_manifest_sha),
        expected_source_sha256=str(expected_source_sha),
    )
    audit = measurement.get("locked_test_audit")
    if not isinstance(audit, Mapping) or (
        audit.get("status") != "not_opened"
        or audit.get("source_locked_rows") != 0
        or audit.get("source_declares_outcomes_materialized") is not False
        or audit.get("used_for_authorization") is not False
    ):
        raise ValueError("H-MEAS receipt does not prove that locked outcomes remained unopened")
    selection = measurement.get("target_selection_receipt")
    if not isinstance(selection, Mapping):
        raise ValueError("H-MEAS receipt lacks its development-only target selection")
    topology = _json_object(topology_path, "stratum topology")
    if set(topology) != {
        "schema",
        "analysis_plan_sha256",
        "ordered_planned_stratum_ids",
        "planned_stratum_minimum_evaluation_cells",
        "locked_outcomes_used",
    } or (
        topology.get("schema") != "heir.confirmatory_stratum_topology.v1"
        or topology.get("analysis_plan_sha256") != manifest.content.get("analysis_plan_sha256")
        or topology.get("locked_outcomes_used") is not False
    ):
        raise ValueError("stratum topology is not an outcome-free H-CELL design artifact")
    binding = build_confirmatory_design_binding(
        manifest.content,
        measurement_receipt_sha256=sha256_file(measurement_path),
        ordered_target_gene_ids=selection["ordered_reliable_gene_ids"],
        supported_fine_type_ids=selection["supported_fine_type_ids"],
        ordered_planned_stratum_ids=topology["ordered_planned_stratum_ids"],
        planned_stratum_minimum_evaluation_cells=topology[
            "planned_stratum_minimum_evaluation_cells"
        ],
    )
    if settings.get("confirmatory_analysis_plan_sha256") != manifest.content.get(
        "analysis_plan_sha256"
    ):
        raise ValueError("calibration configuration belongs to a different analysis plan")

    completed = copy.deepcopy(config)
    completed["exact_gate_settings"]["confirmatory_design_binding"] = dict(binding)
    validate_exact_gate_settings(completed["exact_gate_settings"])
    atomic_json_dump(completed, output_path)
    print(
        json.dumps(
            {
                "schema": completed["schema"],
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "exact_gate_settings_sha256": canonical_sha256(completed["exact_gate_settings"]),
                "confirmatory_design_binding": binding,
                "stratum_topology_sha256": sha256_file(topology_path),
                "locked_outcomes_used": False,
                "authorizes_final_inference": False,
                "next_blocker": (
                    "freeze the six-condition quantitative truth-matrix DGP and per-trial "
                    "attestation protocol before any authorizing calibration"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
