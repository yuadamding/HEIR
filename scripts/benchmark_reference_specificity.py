#!/usr/bin/env python3
"""Run the second-stage matched-versus-wrong reference specificity test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from heir.evaluation import MORPHOLOGY_RIDGE_REPORT_SCHEMA, evaluate_reference_specificity
from heir.utils import atomic_json_dump, reject_output_input_collisions, sha256_file


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--primary-report", type=Path, required=True)
    parser.add_argument("--replication-report", type=Path, required=True)
    parser.add_argument("--confirmation-report", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve()
    prerequisite_paths = tuple(
        path.expanduser().resolve()
        for path in (args.primary_report, args.replication_report, args.confirmation_report)
    )
    output = args.report_output.expanduser().resolve()
    inputs = (source, *prerequisite_paths)
    if any(not path.is_file() for path in inputs) or len(set(inputs)) != len(inputs):
        raise ValueError("reference-specificity inputs must be distinct existing files")
    reject_output_input_collisions((output,), inputs, label="reference specificity")
    before = {str(path): sha256_file(path) for path in inputs}
    expected_roles = ("primary_hoptimus1", "replication_h0mini", "confirmation_xenium")
    prerequisites = []
    for path, expected_role in zip(prerequisite_paths, expected_roles):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("morphology prerequisite report is invalid") from error
        if (
            not isinstance(report, dict)
            or report.get("schema_version") != MORPHOLOGY_RIDGE_REPORT_SCHEMA
            or report.get("experiment_role") != expected_role
            or report.get("component_pass") is not True
            or report.get("oracle_type_only") is not True
        ):
            raise ValueError("reference specificity requires all three passing morphology stages")
        prerequisites.append(
            {"role": expected_role, "path": str(path), "sha256": before[str(path)]}
        )
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "query_latent",
            "query_types",
            "query_donors",
            "matched_bank_latent",
            "matched_bank_types",
            "wrong_bank_names",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("reference-specificity input is missing: %s" % ", ".join(missing))
        names = tuple(str(value) for value in archive["wrong_bank_names"])
        wrong = {}
        for index, name in enumerate(names):
            latent_name = "wrong_bank_%d_latent" % index
            types_name = "wrong_bank_%d_types" % index
            if latent_name not in archive or types_name not in archive:
                raise ValueError("wrong reference bank arrays are incomplete")
            wrong[name] = (np.array(archive[latent_name]), np.array(archive[types_name]))
        report = evaluate_reference_specificity(
            np.array(archive["query_latent"]),
            np.array(archive["query_types"]),
            np.array(archive["query_donors"]),
            np.array(archive["matched_bank_latent"]),
            np.array(archive["matched_bank_types"]),
            wrong,
            repeats=args.repeats,
            seed=args.seed,
        )
    for path in inputs:
        if sha256_file(path) != before[str(path)]:
            raise RuntimeError("reference-specificity input changed during execution")
    result = {
        **report,
        "input": {"path": str(source), "sha256": before[str(source)]},
        "morphology_prerequisites": prerequisites,
        "authorizes_uot_or_refinement": False,
    }
    atomic_json_dump(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
