#!/usr/bin/env python3
"""Run the manifest-locked, measurement-qualified HEST morphology gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from heir.data import MorphologyRidgeDatasetArtifact, StudyManifest, ordered_ids_sha256
from heir.evaluation import evaluate_morphology_ridge_gate, validate_experiment_identity
from heir.evaluation.control_models import (
    HEST_CROP_CONTRACT,
    REQUIRED_HEST_CROP_IDS,
    REQUIRED_MODEL_FAMILIES,
    feature_family_registry,
)
from heir.evaluation.measurement_gate import load_passing_measurement_receipt
from heir.utils import atomic_json_dump, reject_output_input_collisions, sha256_file


def _mapping(value: object, name: str, required: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError("locked study manifest %s is incomplete" % name)
    return value


def _sha256(value: object, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("%s must be a lowercase SHA-256" % name)
    return digest


def _load_json(path: Path, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("%s is not valid JSON" % name) from error
    if not isinstance(value, Mapping):
        raise ValueError("%s root must be an object" % name)
    return value


def _measurement_receipt(
    manifest: StudyManifest, path: Path
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    prerequisites = _mapping(
        manifest.content.get("prerequisites"),
        "prerequisites",
        {
            "measurement_report_sha256",
            "measurement_study_manifest_sha256",
            "measurement_source_sha256",
        },
    )
    report = load_passing_measurement_receipt(
        path,
        expected_receipt_sha256=_sha256(
            prerequisites["measurement_report_sha256"],
            "prerequisites.measurement_report_sha256",
        ),
        expected_study_manifest_sha256=_sha256(
            prerequisites["measurement_study_manifest_sha256"],
            "prerequisites.measurement_study_manifest_sha256",
        ),
        expected_source_sha256=_sha256(
            prerequisites["measurement_source_sha256"],
            "prerequisites.measurement_source_sha256",
        ),
    )
    selection = _mapping(
        report.get("target_selection_receipt"),
        "H-MEAS target_selection_receipt",
        {
            "selection_partition",
            "primary_target_variant",
            "ordered_reliable_gene_ids",
            "ordered_reliable_gene_panel_sha256",
            "supported_fine_type_ids",
            "supported_fine_type_panel_sha256",
            "locked_test_molecular_outcomes_used",
        },
    )
    genes = tuple(str(value) for value in selection["ordered_reliable_gene_ids"])
    fine_types = tuple(str(value) for value in selection["supported_fine_type_ids"])
    if (
        selection.get("schema") != "heir.measurement_target_selection.v1"
        or selection.get("pass") is not True
        or selection["selection_partition"] != "development_only"
        or selection["primary_target_variant"] != "nucleus_overlapping_transcripts"
        or selection["locked_test_molecular_outcomes_used"] is not False
        or selection["ordered_reliable_gene_panel_sha256"] != ordered_ids_sha256(genes)
        or selection["supported_fine_type_panel_sha256"] != ordered_ids_sha256(fine_types)
    ):
        raise ValueError("H-MEAS target selection is not a frozen development-only receipt")
    return report, selection


def _calibration_receipt(
    manifest: StudyManifest, path: Optional[Path]
) -> Optional[Mapping[str, object]]:
    gate = _mapping(manifest.content.get("morphology_gate"), "morphology_gate", {"final_inference"})
    if gate["final_inference"] is not True:
        if path is not None:
            raise ValueError("exploratory manifest does not authorize a calibration input")
        return None
    expected = _sha256(
        gate.get("calibration_receipt_sha256"),
        "morphology_gate.calibration_receipt_sha256",
    )
    if path is None or not path.is_file() or sha256_file(path) != expected:
        raise ValueError("final inference requires the exact locked calibration receipt")
    return _load_json(path, "calibration receipt")


def _positive_numbers(value: object, name: str, *, integer: bool) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("locked %s must be a nonempty list" % name)
    try:
        result = tuple(int(item) if integer else float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError("locked %s contains a non-number" % name) from error
    if any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise ValueError("locked %s must contain unique positive values" % name)
    return tuple(float(value) for value in result)


def _gate_settings(manifest: StudyManifest) -> Mapping[str, object]:
    grid = _mapping(
        manifest.content["hyperparameter_grid"],
        "hyperparameter_grid",
        {"ranks", "ridge_penalties"},
    )
    randomization = _mapping(
        manifest.content["randomization"],
        "randomization",
        {"seeds", "permutations_per_seed", "unit"},
    )
    endpoint = _mapping(
        manifest.content["primary_endpoint"],
        "primary_endpoint",
        {"minimum_effect"},
    )
    coverage = _mapping(
        manifest.content["coverage_requirements"],
        "coverage_requirements",
        {
            "minimum_development_donors_per_fine_type",
            "minimum_locked_donors_per_fine_type",
            "minimum_evaluation_cells_per_donor_type",
            "minimum_positive_supported_fraction",
        },
    )
    thresholds = _mapping(
        manifest.content["decision_thresholds"],
        "decision_thresholds",
        {"minimum_shuffled_delta_r2", "maximum_empirical_p"},
    )
    gate = _mapping(
        manifest.content.get("morphology_gate"),
        "morphology_gate",
        {
            "experiment_role",
            "scientific_scope",
            "final_inference",
            "minimum_final_permutations",
            "minimum_coordinate_delta",
            "minimum_stain_delta",
            "maximum_direct_contrast_p",
            "minimum_mask_implementation_pass_fraction",
            "minimum_null_shuffled_fraction",
            "minimum_strata_coverage",
            "minimum_expression_error_reduction",
            "minimum_basis_ceiling_r2",
            "donor_bootstrap_iterations",
            "donor_bootstrap_seed",
            "prespecified_fixed_hyperparameters",
        },
    )
    seeds = _positive_numbers(randomization["seeds"], "randomization.seeds", integer=True)
    per_seed = int(randomization["permutations_per_seed"])
    if per_seed < 1:
        raise ValueError("locked permutations_per_seed must be positive")
    if randomization["unit"] != "donor_x_fine_type_x_spatial_roi":
        raise ValueError(
            "morphology randomization unit differs from the locked donor/type/ROI unit"
        )
    return {
        "experiment_role": str(gate["experiment_role"]),
        "scientific_scope": str(gate["scientific_scope"]),
        "ranks": tuple(
            int(value) for value in _positive_numbers(grid["ranks"], "ranks", integer=True)
        ),
        "alphas": _positive_numbers(grid["ridge_penalties"], "ridge penalties", integer=False),
        "permutation_seeds": tuple(int(value) for value in seeds),
        "permutations_per_seed": per_seed,
        "total_permutations": len(seeds) * per_seed,
        "final_inference": bool(gate["final_inference"]),
        "minimum_final_permutations": int(gate["minimum_final_permutations"]),
        "minimum_support": int(coverage["minimum_evaluation_cells_per_donor_type"]),
        "minimum_development_donors": int(coverage["minimum_development_donors_per_fine_type"]),
        "minimum_locked_donors": int(coverage["minimum_locked_donors_per_fine_type"]),
        "minimum_macro_r2": float(endpoint["minimum_effect"]),
        "minimum_shuffle_delta": float(thresholds["minimum_shuffled_delta_r2"]),
        "maximum_permutation_p": float(thresholds["maximum_empirical_p"]),
        "minimum_coordinate_delta": float(gate["minimum_coordinate_delta"]),
        "minimum_stain_delta": float(gate["minimum_stain_delta"]),
        "maximum_direct_contrast_p": float(gate["maximum_direct_contrast_p"]),
        "minimum_mask_implementation_pass_fraction": float(
            gate["minimum_mask_implementation_pass_fraction"]
        ),
        "minimum_null_shuffled_fraction": float(gate["minimum_null_shuffled_fraction"]),
        "minimum_strata_coverage": float(gate["minimum_strata_coverage"]),
        "minimum_positive_strata_fraction": float(coverage["minimum_positive_supported_fraction"]),
        "minimum_expression_error_reduction": float(gate["minimum_expression_error_reduction"]),
        "minimum_basis_ceiling_r2": float(gate["minimum_basis_ceiling_r2"]),
        "donor_bootstrap_iterations": int(gate["donor_bootstrap_iterations"]),
        "donor_bootstrap_seed": int(gate["donor_bootstrap_seed"]),
        "prespecified_fixed_hyperparameters": bool(gate["prespecified_fixed_hyperparameters"]),
    }


def _validate_bindings(
    manifest: StudyManifest,
    measurement_path: Path,
    selection: Mapping[str, object],
    development: MorphologyRidgeDatasetArtifact,
    locked: MorphologyRidgeDatasetArtifact,
) -> None:
    genes = tuple(str(value) for value in selection["ordered_reliable_gene_ids"])
    types = tuple(str(value) for value in selection["supported_fine_type_ids"])
    prerequisites = manifest.content["prerequisites"]
    gate = _mapping(
        manifest.content.get("morphology_gate"),
        "morphology_gate",
        {"scientific_scope"},
    )
    observations = _mapping(
        manifest.content.get("observations"),
        "observations",
        {"level", "registration_method"},
    )
    reference_splits = _mapping(
        manifest.content.get("reference_splits"),
        "reference_splits",
        {"primary_split_id", "split_ids"},
    )
    expected_split_ids = tuple(str(value) for value in reference_splits["split_ids"])
    encoder = _mapping(
        manifest.content.get("encoder"),
        "encoder",
        {"feature_space_id", "checkpoint_sha256"},
    )
    if (
        development.study_manifest_sha256 != manifest.sha256
        or locked.study_manifest_sha256 != manifest.sha256
        or development.measurement_receipt_sha256 != sha256_file(measurement_path)
        or locked.measurement_receipt_sha256 != sha256_file(measurement_path)
        or development.measurement_source_sha256 != prerequisites["measurement_source_sha256"]
        or locked.measurement_source_sha256 != prerequisites["measurement_source_sha256"]
        or development.gene_ids != genes
        or locked.gene_ids != genes
        or development.type_names != types
        or locked.type_names != types
        or development.scientific_scope != gate["scientific_scope"]
        or locked.scientific_scope != gate["scientific_scope"]
        or development.reference_split_ids != expected_split_ids
        or locked.reference_split_ids != expected_split_ids
        or development.feature_space_id != encoder["feature_space_id"]
        or locked.feature_space_id != encoder["feature_space_id"]
        or development.feature_checkpoint_sha256 != encoder["checkpoint_sha256"]
        or locked.feature_checkpoint_sha256 != encoder["checkpoint_sha256"]
        or development.observation_level != observations["level"]
        or locked.observation_level != observations["level"]
        or development.registration_method != observations["registration_method"]
        or locked.registration_method != observations["registration_method"]
        or development.target_construction != "nucleus_overlapping_xenium_transcripts"
        or locked.target_construction != "nucleus_overlapping_xenium_transcripts"
        or ordered_ids_sha256(development.type_marker_gene_ids)
        != manifest.content["type_marker_panel_sha256"]
        or ordered_ids_sha256(locked.type_marker_gene_ids)
        != manifest.content["type_marker_panel_sha256"]
        or set(development.crop_ids) != set(REQUIRED_HEST_CROP_IDS)
        or set(locked.crop_ids) != set(REQUIRED_HEST_CROP_IDS)
        or {
            crop_id: (role, family)
            for crop_id, role, family in zip(
                development.crop_ids,
                development.crop_roles,
                development.crop_comparison_families,
            )
        }
        != HEST_CROP_CONTRACT
        or {
            crop_id: (role, family)
            for crop_id, role, family in zip(
                locked.crop_ids,
                locked.crop_roles,
                locked.crop_comparison_families,
            )
        }
        != HEST_CROP_CONTRACT
        or development.technical_covariate_names != ("log1p_library_size",)
        or locked.technical_covariate_names != ("log1p_library_size",)
        or not development.spatial_control_feature_names
        or not locked.spatial_control_feature_names
        or tuple(sorted(set(development.donor_ids.tolist())))
        != tuple(sorted(manifest.development_donors))
        or tuple(sorted(set(locked.donor_ids.tolist())))
        != tuple(sorted(manifest.locked_test_donors))
    ):
        raise ValueError("prepared morphology artifacts differ from the locked H-MEAS study")
    protection = _mapping(
        manifest.content.get("lock_protection"),
        "lock_protection",
        {
            "reserved_exclusively_for",
            "reserved_donor_ids",
            "prior_outcome_access_confirmed_false",
        },
    )
    if (
        protection["reserved_exclusively_for"] != "H-CELL"
        or protection["prior_outcome_access_confirmed_false"] is not True
        or set(str(value) for value in protection["reserved_donor_ids"])
        != set(manifest.locked_test_donors)
    ):
        raise ValueError("locked HEST outcomes were not exclusively protected for H-CELL")
    if development.cohort_id != "HEST" or locked.cohort_id != "HEST":
        raise ValueError("only HEST may enter the internal locked morphology benchmark")
    for artifact in (development, locked):
        membership = artifact.coverage_audit.get(
            "reference_membership_sha256_by_split"
        )
        if (
            not isinstance(membership, Mapping)
            or set(str(value) for value in membership)
            != set(artifact.reference_split_ids)
            or len(set(str(value) for value in membership.values()))
            != len(artifact.reference_split_ids)
        ):
            raise ValueError("prepared reference-split memberships are missing or degenerate")
        for split_id, digest in membership.items():
            _sha256(digest, "reference membership %s" % split_id)
        registry = feature_family_registry(artifact)
        missing = sorted(
            family for family in REQUIRED_MODEL_FAMILIES if registry.get(family) is None
        )
        if missing:
            raise ValueError(
                "prepared morphology artifact omits locked control arms: %s" % ", ".join(missing)
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--measurement-report", type=Path, required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--locked-test-data", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    manifest_path = args.study_manifest.expanduser().resolve()
    measurement_path = args.measurement_report.expanduser().resolve()
    development_path = args.development_data.expanduser().resolve()
    locked_path = args.locked_test_data.expanduser().resolve()
    calibration_path = (
        args.calibration_receipt.expanduser().resolve()
        if args.calibration_receipt is not None
        else None
    )
    report_path = args.report_output.expanduser().resolve()
    inputs = (manifest_path, measurement_path, development_path, locked_path) + (
        (calibration_path,) if calibration_path is not None else ()
    )
    if len(set(inputs)) != len(inputs) or any(not path.is_file() for path in inputs):
        raise ValueError("all benchmark inputs must be distinct existing files")
    reject_output_input_collisions((report_path,), inputs, label="morphology ridge benchmark")
    before = {str(path): sha256_file(path) for path in inputs}

    manifest = StudyManifest.load(manifest_path, require_status="locked")
    if manifest.study_stage != "confirmatory_morphology":
        raise ValueError("morphology benchmark requires a confirmatory morphology manifest")
    measurement, selection = _measurement_receipt(manifest, measurement_path)
    calibration = _calibration_receipt(manifest, calibration_path)
    settings = _gate_settings(manifest)
    development = MorphologyRidgeDatasetArtifact.load_npz(development_path, role="development")
    locked = MorphologyRidgeDatasetArtifact.load_npz(locked_path, role="locked_test")
    _validate_bindings(manifest, measurement_path, selection, development, locked)
    validate_experiment_identity(development, str(settings["experiment_role"]))
    report = evaluate_morphology_ridge_gate(
        development,
        locked,
        ranks=settings["ranks"],
        alphas=settings["alphas"],
        permutation_seeds=settings["permutation_seeds"],
        permutations_per_seed=settings["permutations_per_seed"],
        total_permutations=settings["total_permutations"],
        final_inference=settings["final_inference"],
        minimum_final_permutations=settings["minimum_final_permutations"],
        minimum_support=settings["minimum_support"],
        minimum_development_donors=settings["minimum_development_donors"],
        minimum_locked_donors=settings["minimum_locked_donors"],
        minimum_macro_r2=settings["minimum_macro_r2"],
        minimum_shuffle_delta=settings["minimum_shuffle_delta"],
        minimum_coordinate_delta=settings["minimum_coordinate_delta"],
        minimum_stain_delta=settings["minimum_stain_delta"],
        maximum_direct_contrast_p=settings["maximum_direct_contrast_p"],
        minimum_mask_implementation_pass_fraction=settings[
            "minimum_mask_implementation_pass_fraction"
        ],
        minimum_null_shuffled_fraction=settings["minimum_null_shuffled_fraction"],
        minimum_strata_coverage=settings["minimum_strata_coverage"],
        maximum_permutation_p=settings["maximum_permutation_p"],
        minimum_positive_strata_fraction=settings["minimum_positive_strata_fraction"],
        minimum_expression_error_reduction=settings["minimum_expression_error_reduction"],
        minimum_basis_ceiling_r2=settings["minimum_basis_ceiling_r2"],
        donor_bootstrap_iterations=settings["donor_bootstrap_iterations"],
        donor_bootstrap_seed=settings["donor_bootstrap_seed"],
        prespecified_fixed_hyperparameters=settings["prespecified_fixed_hyperparameters"],
        calibration_receipt=calibration,
        device=args.device,
    )
    for path in inputs:
        if not path.is_file() or sha256_file(path) != before[str(path)]:
            raise RuntimeError("morphology ridge input changed during execution: %s" % path)
    result = {
        **report,
        "experiment_role": settings["experiment_role"],
        "scientific_settings_source": "locked_study_manifest_only",
        "measurement_gate_pass": measurement["pass"],
        "provenance": {
            "study_manifest": {
                "path": str(manifest_path),
                "sha256": before[str(manifest_path)],
            },
            "measurement_report": {
                "path": str(measurement_path),
                "sha256": before[str(measurement_path)],
            },
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
            "gene_ids": list(development.gene_ids),
            "type_names": list(development.type_names),
            "crop_ids": list(development.crop_ids),
            "crop_roles": list(development.crop_roles),
        },
    }
    atomic_json_dump(result, report_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["component_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
