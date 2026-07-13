#!/usr/bin/env python3
"""Issue a locked, source-bound H-MEAS receipt for registered observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.data.study_manifest import StudyManifest
from heir.evaluation.measurement_gate import MeasurementThresholds, evaluate_measurement_gate
from heir.utils import (
    atomic_json_dump,
    reject_output_input_collisions,
    runtime_environment,
    sha256_file,
)


def _measurement_randomization(content: Mapping[str, object]) -> Mapping[str, object]:
    value = content.get("randomization")
    if not isinstance(value, Mapping):
        raise ValueError("locked study manifest lacks randomization settings")
    nested = value.get("measurement")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("randomization.measurement must be a mapping")
        return nested
    return value


def _scalar(archive: Mapping[str, np.ndarray], name: str) -> object:
    if name not in archive:
        raise ValueError("registered source lacks %s" % name)
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise ValueError("registered source %s must be scalar" % name)
    return value.item()


def _sha256_scalar(archive: Mapping[str, np.ndarray], name: str) -> str:
    digest = str(_scalar(archive, name))
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError("registered source %s must be a lowercase SHA-256" % name)
    return digest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest_path = args.study_manifest.expanduser().resolve()
    source_path = args.source.expanduser().resolve()
    output_path = args.report_output.expanduser().resolve()
    if not manifest_path.is_file() or not source_path.is_file():
        raise ValueError("measurement benchmark inputs must be existing files")
    if output_path.exists():
        raise ValueError("measurement report output already exists and is immutable")
    reject_output_input_collisions(
        (output_path,), (manifest_path, source_path), label="measurement gate"
    )
    repository_root = Path(__file__).resolve().parents[1]
    manifest = StudyManifest.load(
        manifest_path,
        require_status="locked",
        verify_runtime=True,
        repository_root=repository_root,
    )
    if manifest.study_stage != "measurement_development":
        raise ValueError("H-MEAS requires a measurement_development study manifest")
    if "H-MEAS" not in manifest.hypothesis_ids:
        raise ValueError("locked study manifest does not authorize H-MEAS")
    observations = manifest.content["observations"]
    if not isinstance(observations, Mapping):
        raise ValueError("locked study observation definition is malformed")
    target_variants = observations.get("target_variants")
    if not isinstance(target_variants, list) or not target_variants:
        raise ValueError("locked study lacks target variants")
    randomization = _measurement_randomization(manifest.content)
    split_salt = randomization.get("transcript_split_salt")
    if not isinstance(split_salt, str) or not split_salt:
        raise ValueError("locked study lacks a transcript_split_salt")
    thresholds = MeasurementThresholds.from_study_manifest(manifest.content)

    source_stat = source_path.stat()
    source_sha256 = sha256_file(source_path)
    with np.load(source_path, allow_pickle=False) as archive:
        # Registered observations are constructed once before either scientific stage.
        # The immutable source hash is bound into this H-MEAS receipt; the later H-CELL
        # manifest binds this receipt rather than requiring the source to predict its hash.
        source_identity_receipts = {
            name: _sha256_scalar(archive, name)
            for name in (
                "source_file_manifest_sha256",
                "registration_manifest_sha256",
                "target_manifest_sha256",
                "segmentation_manifest_sha256",
                "exclusion_policy_sha256",
            )
        }
        report = evaluate_measurement_gate(
            archive,
            thresholds,
            development_donors=manifest.development_donors,
            locked_test_donors=manifest.locked_test_donors,
            target_variants=tuple(str(value) for value in target_variants),
            split_salt=split_salt,
            study_manifest_sha256=manifest.sha256,
            source_sha256=source_sha256,
        )
    final_stat = source_path.stat()
    if (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
    ) != (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mtime_ns,
    ):
        raise RuntimeError("registered source changed during the measurement benchmark")

    result = {
        **report,
        "study_manifest": {"path": str(manifest.path), "sha256": manifest.sha256},
        "source": {"path": str(source_path), "sha256": source_sha256},
        "source_identity_receipts": source_identity_receipts,
        "runtime": runtime_environment(),
    }
    atomic_json_dump(dict(result), output_path)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "pass": result["pass"],
                "report": str(output_path),
                "study_manifest_sha256": manifest.sha256,
                "source_sha256": source_sha256,
                "target_selection_receipt_sha256": result["target_selection_receipt"][
                    "receipt_content_sha256"
                ],
                "report_sha256": sha256_file(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
