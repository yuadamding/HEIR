#!/usr/bin/env python3
"""Run the oracle-type ridge go/no-go test on frozen development and locked data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

from heir.data import MorphologyRidgeDatasetArtifact
from heir.evaluation import evaluate_morphology_ridge_gate, validate_experiment_identity
from heir.utils import atomic_json_dump, reject_output_input_collisions, sha256_file


def _numbers(value: str, *, integer: bool) -> Union[Tuple[float, ...], Tuple[int, ...]]:
    try:
        parsed = tuple(int(item) if integer else float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all grid values must be positive")
    return parsed


def _load_calibration(path: Optional[Path]) -> Optional[Mapping[str, object]]:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("calibration receipt is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("calibration receipt root must be an object")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--locked-test-data", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument(
        "--experiment-role",
        choices=(
            "primary_hoptimus1",
            "replication_h0mini",
            "context_sensitivity",
            "confirmation_xenium",
            "regional_hescape_hoptimus1",
            "regional_hescape_uni2h",
        ),
        required=True,
    )
    parser.add_argument("--ranks", default="2,4,6")
    parser.add_argument("--ridge-penalties", default="0.1,1,10,100")
    parser.add_argument("--permutation-seeds", default="17,29,41")
    parser.add_argument("--permutations-per-seed", type=int, default=100)
    parser.add_argument("--total-permutations", type=int, default=None)
    parser.add_argument("--final-inference", action="store_true")
    parser.add_argument("--calibration-receipt", type=Path, default=None)
    parser.add_argument("--minimum-support", type=int, default=10)
    parser.add_argument("--minimum-null-shuffled-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-strata-coverage", type=float, default=0.80)
    parser.add_argument("--donor-bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--donor-bootstrap-seed", type=int, default=1701)
    parser.add_argument("--minimum-development-donors", type=int, default=5)
    parser.add_argument(
        "--minimum-locked-donors",
        type=int,
        default=None,
        help="defaults to four for regional UNI2-h and five for cell-level tests",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    development_path = args.development_data.expanduser().resolve()
    locked_path = args.locked_test_data.expanduser().resolve()
    report_path = args.report_output.expanduser().resolve()
    calibration_path = (
        args.calibration_receipt.expanduser().resolve()
        if args.calibration_receipt is not None
        else None
    )
    inputs = (development_path, locked_path) + (
        (calibration_path,) if calibration_path is not None else ()
    )
    if development_path == locked_path or any(not path.is_file() for path in inputs):
        raise ValueError("development and locked-test inputs must be distinct existing files")
    reject_output_input_collisions((report_path,), inputs, label="morphology ridge benchmark")
    before = {str(path): sha256_file(path) for path in inputs}
    calibration = _load_calibration(calibration_path)

    development = MorphologyRidgeDatasetArtifact.load_npz(development_path, role="development")
    locked = MorphologyRidgeDatasetArtifact.load_npz(locked_path, role="locked_test")
    validate_experiment_identity(development, args.experiment_role)
    report = evaluate_morphology_ridge_gate(
        development,
        locked,
        ranks=_numbers(args.ranks, integer=True),
        alphas=_numbers(args.ridge_penalties, integer=False),
        permutation_seeds=_numbers(args.permutation_seeds, integer=True),
        permutations_per_seed=args.permutations_per_seed,
        total_permutations=args.total_permutations,
        final_inference=args.final_inference,
        minimum_support=args.minimum_support,
        minimum_null_shuffled_fraction=args.minimum_null_shuffled_fraction,
        minimum_strata_coverage=args.minimum_strata_coverage,
        minimum_development_donors=args.minimum_development_donors,
        minimum_locked_donors=args.minimum_locked_donors,
        donor_bootstrap_iterations=args.donor_bootstrap_iterations,
        donor_bootstrap_seed=args.donor_bootstrap_seed,
        calibration_receipt=calibration,
        device=args.device,
    )
    for path in inputs:
        if not path.is_file() or sha256_file(path) != before[str(path)]:
            raise RuntimeError("morphology ridge input changed during execution: %s" % path)
    result = {
        **report,
        "experiment_role": args.experiment_role,
        "provenance": {
            "development_data": {
                "path": str(development_path),
                "sha256": before[str(development_path)],
            },
            "locked_test_data": {
                "path": str(locked_path),
                "sha256": before[str(locked_path)],
            },
            "calibration_receipt": (
                {
                    "path": str(calibration_path),
                    "sha256": before[str(calibration_path)],
                }
                if calibration_path is not None
                else None
            ),
            "feature_space_id": development.feature_space_id,
            "feature_checkpoint_sha256": development.feature_checkpoint_sha256,
            "encoder_name": development.encoder_name,
            "crop_scale": development.crop_scale,
            "cohort_id": development.cohort_id,
            "cohort_release": development.cohort_release,
            "assay": development.assay,
            "observation_level": development.observation_level,
            "target_construction": development.target_construction,
            "molecular_space_id": development.molecular_space_id,
            "gene_ids": list(development.gene_ids),
            "type_names": list(development.type_names),
            "stain_feature_names": list(development.stain_feature_names),
            "technical_covariate_names": list(development.technical_covariate_names),
            "composition_feature_names": list(development.composition_feature_names),
            "oracle_label_source_sha256": development.label_source_sha256,
            "development_reference_source_sha256": development.reference_source_sha256,
            "locked_reference_source_sha256": locked.reference_source_sha256,
            "registration_source_sha256": development.registration_source_sha256,
            "exclusion_policy_sha256": development.exclusion_policy_sha256,
        },
    }
    atomic_json_dump(result, report_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["component_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
