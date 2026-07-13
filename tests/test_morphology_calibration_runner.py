from __future__ import annotations

import copy
import json
import os
import resource
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

import heir.evaluation.morphology_calibration as calibration_compiler
import heir.evaluation.morphology_calibration_runner as runner
from heir.evaluation.control_models import HEST_CROP_CONTRACT
from heir.evaluation.hierarchical_metrics import donor_section_type_coverage
from heir.evaluation.morphology_calibration import compile_actual_gate_calibration_receipt
from heir.evaluation.permutations import null_stratum_activity
from heir.evaluation.power import (
    ACTUAL_GATE_REPORT_SCHEMA,
    CALIBRATION_EVIDENCE_SCHEMA,
    GLOBAL_NULL_CONDITION,
    PRELIMINARY_ALTERNATIVE_CONDITION,
    REQUIRED_CALIBRATION_SCENARIOS,
    REQUIRED_COMPLETE_GATE_CHECKS,
    REQUIRED_HYPOTHESIS_DECISIONS,
    REQUIRED_LOCKED_MEASUREMENT_AUDIT_CONTRACT,
    canonical_sha256,
    required_simultaneous_confidence_level,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "morphology_gate_calibration.json"


def _settings() -> Mapping[str, object]:
    content = json.loads(CONFIG.read_text(encoding="utf-8"))
    settings = copy.deepcopy(content["exact_gate_settings"])
    settings["complete_gate_check_ids"] = list(REQUIRED_COMPLETE_GATE_CHECKS)
    settings["confirmatory_design_binding"] = (
        runner.synthetic_completed_confirmatory_design_binding()
    )
    return settings


def test_checked_in_runner_config_waits_for_completed_h_meas_binding() -> None:
    with pytest.raises(ValueError, match="pending pre-H-MEAS"):
        runner.load_calibration_run_config(CONFIG)


def test_synthetic_calibration_builder_covers_the_frozen_experiment() -> None:
    binding = runner.synthetic_completed_confirmatory_design_binding()
    for scenario in REQUIRED_CALIBRATION_SCENARIOS:
        development, locked = runner.build_synthetic_calibration_pair(
            scenario,
            PRELIMINARY_ALTERNATIVE_CONDITION,
            0,
        )
        assert development.cohort_id == "SYNTHETIC_CALIBRATION"
        assert locked.cohort_id == "SYNTHETIC_CALIBRATION"
        assert development.authorizes_nucleus_intrinsic_claim is False
        assert locked.authorizes_nucleus_intrinsic_claim is False
        assert len(set(locked.donor_ids)) == 5
        assert tuple(development.crop_ids) == tuple(HEST_CROP_CONTRACT)
        assert len(development.crop_ids) == 18
        assert development.reference_split_ids == (
            "primary",
            "reference_hash_fold_0",
            "reference_hash_fold_1",
        )
        assert tuple(development.planned_stratum_ids) + tuple(locked.planned_stratum_ids) == tuple(
            binding["ordered_planned_stratum_ids"]
        )
        assert (
            development.planned_stratum_manifest_sha256
            == binding["planned_stratum_manifest_sha256"]
        )
        assert locked.planned_stratum_manifest_sha256 == binding["planned_stratum_manifest_sha256"]
        for artifact in (development, locked):
            assert set(artifact.registration_quality_strata.tolist()) == {
                "best",
                "intermediate",
                "near_threshold",
            }
            observed_quality_support = {
                (str(donor), int(type_index), band): int(
                    np.count_nonzero(
                        (artifact.donor_ids.astype(str) == str(donor))
                        & (artifact.type_labels == int(type_index))
                        & (artifact.registration_quality_strata == band)
                    )
                )
                for donor, type_index in zip(artifact.donor_ids, artifact.type_labels)
                for band in ("best", "intermediate", "near_threshold")
            }
            assert min(observed_quality_support.values()) >= 20
            section_quality_support = {
                (str(donor), str(section), int(type_index), band): int(
                    np.count_nonzero(
                        (artifact.donor_ids.astype(str) == str(donor))
                        & (artifact.section_ids.astype(str) == str(section))
                        & (artifact.type_labels == int(type_index))
                        & (artifact.registration_quality_strata == band)
                    )
                )
                for donor, section, type_index in zip(
                    artifact.donor_ids,
                    artifact.section_ids,
                    artifact.type_labels,
                )
                for band in ("best", "intermediate", "near_threshold")
            }
            assert min(section_quality_support.values()) >= 20
            observed = {
                "%s|%s|%s" % (donor, section, artifact.type_names[int(type_index)])
                for donor, section, type_index in zip(
                    artifact.donor_ids,
                    artifact.section_ids,
                    artifact.type_labels,
                )
            }
            assert observed <= set(artifact.planned_stratum_ids)
            assert artifact.coverage_audit["retained_fraction"] == pytest.approx(
                len(observed) / len(artifact.planned_stratum_ids)
            )
            assert artifact.coverage_audit["source_rows_before_frozen_qc"] > len(
                artifact.observation_ids
            )
            assert artifact.coverage_audit["evaluation_rows_after_frozen_qc"] == len(
                artifact.observation_ids
            )
            assert artifact.coverage_audit["source_qc_filtered_rows"] == len(observed)
            assert np.max(artifact.registration_quality_scores) <= 1.0
            assert all(
                report["pass"] is True for report in artifact.reference_evaluation_balance.values()
            )
        assert development.coverage_audit["locked_measurement_audit"] is None
        locked_audit = locked.coverage_audit["locked_measurement_audit"]
        if scenario == "variable_transcript_reliability":
            assert locked_audit["pass"] is False
        else:
            assert locked_audit["pass"] is True
        assert locked_audit["planned_stratum_reliability"]["planned_count"] == len(
            locked.planned_stratum_ids
        )
        assert (
            locked_audit["summaries"]["rows_before_frozen_qc"]
            == locked.coverage_audit["source_rows_before_frozen_qc"]
        )
        assert locked_audit["summaries"]["rows_after_frozen_qc"] == len(locked.observation_ids)
        assert min(
            report["rows"]
            for report in locked_audit["donor_section_type_reliability"].values()
            if report["rows"] > 0
        ) >= int(locked_audit["thresholds"]["minimum_reliability_rows"])
        development.validate_compatible(locked)

    _, reliable_locked = runner.build_synthetic_calibration_pair(
        "variable_transcript_reliability",
        PRELIMINARY_ALTERNATIVE_CONDITION,
        1,
    )
    assert reliable_locked.coverage_audit["locked_measurement_audit"]["pass"] is True

    strict_balance, _ = runner._synthetic_reference_evaluation_balance(
        contract={
            "maximum_reference_evaluation_absolute_smd": 0.0,
            "maximum_reference_evaluation_categorical_total_variation": 0.0,
        },
        split_ids=("strict",),
        observation_ids=reliable_locked.observation_ids,
        donor_ids=reliable_locked.donor_ids,
        section_ids=reliable_locked.section_ids,
        type_labels=reliable_locked.type_labels,
        type_names=reliable_locked.type_names,
        disease_states=reliable_locked.disease_states,
        site_ids=reliable_locked.site_ids,
        batch_ids=reliable_locked.batch_ids,
        balance_groups=reliable_locked.registration_quality_strata,
        feature_matrix=np.column_stack(
            (
                reliable_locked.coordinate_features,
                reliable_locked.technical_covariates,
                reliable_locked.stain_features,
                reliable_locked.nuclear_morphometrics,
                reliable_locked.cell_morphometrics,
                reliable_locked.cellvit_context_features,
                reliable_locked.local_density_features,
                reliable_locked.boundary_features,
                reliable_locked.spatial_control_features,
            )
        ),
        feature_names=(
            "coordinate::0",
            "coordinate::1",
            *("technical::%s" % value for value in reliable_locked.technical_covariate_names),
            *("stain::%s" % value for value in reliable_locked.stain_feature_names),
            *("nuclear::%s" % value for value in reliable_locked.nuclear_morphometric_names),
            *("cell::%s" % value for value in reliable_locked.cell_morphometric_names),
            *("cellvit::%s" % value for value in reliable_locked.cellvit_context_feature_names),
            *("density::%s" % value for value in reliable_locked.local_density_feature_names),
            *("boundary::%s" % value for value in reliable_locked.boundary_feature_names),
            *("spatial::%s" % value for value in reliable_locked.spatial_control_feature_names),
        ),
    )
    assert strict_balance["strict"]["pass"] is False

    _, missing_locked = runner.build_synthetic_calibration_pair(
        "missing_fine_types",
        PRELIMINARY_ALTERNATIVE_CONDITION,
        0,
    )
    assert set(missing_locked.type_labels.tolist()) == {0, 1}
    assert not np.any((missing_locked.donor_ids == "locked_0") & (missing_locked.type_labels == 1))
    assert missing_locked.coverage_audit["retained_fraction"] == pytest.approx(0.9)

    development, _ = runner.build_synthetic_calibration_pair(
        "inactive_permutation_strata",
        PRELIMINARY_ALTERNATIVE_CONDITION,
        0,
    )
    activity = null_stratum_activity(
        development.donor_ids,
        development.type_labels,
        development.roi_ids,
    )
    assert activity["eligible_row_fraction"] == pytest.approx(0.95)

    _, locked = runner.build_synthetic_calibration_pair(
        "section_effects",
        PRELIMINARY_ALTERNATIVE_CONDITION,
        0,
    )
    section_coverage = donor_section_type_coverage(
        locked.donor_ids,
        locked.section_ids,
        locked.type_labels,
        minimum_support=20,
        num_types=len(locked.type_names),
    )
    assert section_coverage is not None
    assert section_coverage["retained_fraction"] == 1.0


def test_synthetic_locked_audit_uses_full_transcript_library_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_audit = runner.locked_measurement_audit_report
    observed_calls = 0

    def audited_call(**kwargs):
        nonlocal observed_calls
        observed_calls += 1
        half_a_counts = np.asarray(kwargs["half_a_counts"])
        half_b_counts = np.asarray(kwargs["half_b_counts"])
        assert np.all(
            np.asarray(kwargs["half_a_library_sizes"]) > half_a_counts.sum(axis=1, dtype=np.uint64)
        )
        assert np.all(
            np.asarray(kwargs["half_b_library_sizes"]) > half_b_counts.sum(axis=1, dtype=np.uint64)
        )
        return production_audit(**kwargs)

    monkeypatch.setattr(runner, "locked_measurement_audit_report", audited_call)
    runner.build_synthetic_calibration_pair(
        "spatial_autocorrelation",
        PRELIMINARY_ALTERNATIVE_CONDITION,
        1,
    )
    assert observed_calls == 1


def test_production_runner_freezes_six_quantitative_truth_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(runner.DEDICATED_PROCESS_ENV, "1")
    plan = runner.CalibrationRunPlan(
        exact_gate_settings=_settings(),
        trials_per_condition=runner.PRODUCTION_TRIALS_PER_CONDITION,
        smoke_test=False,
    )
    contract = runner._run_contract(plan, _settings())
    assert tuple(contract["conditions"]) == runner.AUTHORIZING_CALIBRATION_CONDITIONS
    assert len(contract["conditions"]) == 6
    dgp = contract["dgp_effect_spec"]
    assert dgp["authorizing_boundary_calibration"] is True
    definitions = dgp["effect_definition"]["condition_definitions"]
    assert set(definitions) == set(contract["conditions"])
    mixed = definitions[runner.BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS["G3_mixed_intrinsic_context"]]
    assert mixed["target_component_population_r2"] == {
        "nucleus_intrinsic": 0.05,
        "extrinsic_context": 0.05,
    }
    assert mixed["total_morphology_population_r2"] == 0.10
    mixed_truth = dgp["decision_truth_by_condition"][
        runner.BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS["G3_mixed_intrinsic_context"]
    ]
    assert {name for name, value in mixed_truth.items() if value} == {
        "G2_local_context",
        "G3_nucleus_intrinsic",
        "G3_context_only",
        "G3_mixed_intrinsic_context",
    }
    assert plan.production_contract_satisfied is True


def test_synthetic_locked_audit_enforces_every_frozen_measurement_criterion() -> None:
    scenario = "spatial_autocorrelation"
    condition = PRELIMINARY_ALTERNATIVE_CONDITION
    trial_index = 1
    _, locked = runner.build_synthetic_calibration_pair(scenario, condition, trial_index)
    audit = locked.coverage_audit["locked_measurement_audit"]
    assert audit["pass"] is True
    assert audit["thresholds"] == REQUIRED_LOCKED_MEASUREMENT_AUDIT_CONTRACT
    assert set(audit["distribution_checks"]) == {
        "annotation_nucleus",
        "annotation_cell",
        "cell_nucleus",
        "nucleus_diameter_relative",
        "nearest_neighbor_relative",
        "segmentation",
        "crop_padding",
        "registration_qc_matches_recomputed",
        "segmentation_qc_matches_recomputed",
        "crop_qc_matches_recomputed",
        "locked_measurement_qc_matches_recomputed_conjunction",
        "donor_type_reliability_fraction",
        "planned_donor_section_type_reliability_fraction",
    }
    assert set(audit["summaries"]["registration"]) == {
        "annotation_to_nucleus_distance_um",
        "annotation_to_cell_distance_um",
        "native_cell_to_nucleus_distance_um",
        "annotation_error_over_median_nucleus_diameter",
        "annotation_error_over_median_nearest_neighbor_distance",
    }
    assert audit["summaries"]["segmentation"]["maximum_nucleus_outside_cell_fraction"] == 0.01
    assert audit["summaries"]["segmentation"]["maximum_area_ratio_outlier_fraction"] == 0.05
    assert set(audit["summaries"]["crop_padding"]) == set(HEST_CROP_CONTRACT)
    assert set(audit["worst_section_reliability_by_donor_type"]) == set(
        audit["donor_type_reliability"]
    )

    common = {
        "scenario": scenario,
        "trial_index": trial_index,
        "seed": runner._trial_seed(1729, scenario, condition, trial_index),
        "donor_ids": locked.donor_ids,
        "section_ids": locked.section_ids,
        "type_labels": locked.type_labels,
        "type_names": locked.type_names,
        "gene_ids": locked.gene_ids,
        "planned_stratum_ids": locked.planned_stratum_ids,
        "molecular_residual": locked.molecular_targets - locked.reference_means,
        "registration_quality_scores": locked.registration_quality_scores,
        "technical_covariates": locked.technical_covariates,
        "coordinate_features": locked.coordinate_features,
    }
    failing_thresholds = {
        "maximum_annotation_nucleus_p95_um": 0.0,
        "maximum_annotation_cell_p95_um": 0.0,
        "maximum_cell_nucleus_p95_um": 0.0,
        "maximum_registration_nucleus_diameter_ratio_p95": 0.0,
        "maximum_registration_nearest_neighbor_ratio_p95": 0.0,
        "maximum_registration_outlier_fraction": -1.0,
        "maximum_nucleus_outside_cell_fraction": -1.0,
        "minimum_nucleus_cell_area_ratio": 0.31,
        "maximum_nucleus_cell_area_ratio": 0.29,
        "maximum_segmentation_outlier_fraction": -1.0,
        "maximum_crop_padding_p95": 0.0,
        "mostly_padded_cutoff": 0.0,
        "maximum_mostly_padded_fraction": -1.0,
        "minimum_within_fine_type_reliability": 1.0,
        "minimum_reliability_rows": len(locked.observation_ids) + 1,
    }
    for field, value in failing_thresholds.items():
        contract = {**REQUIRED_LOCKED_MEASUREMENT_AUDIT_CONTRACT, field: value}
        failed = runner._synthetic_locked_measurement_audit(
            contract=contract,
            **common,
        )
        assert failed["pass"] is False, field

    _, missing_locked = runner.build_synthetic_calibration_pair(
        "missing_fine_types",
        condition,
        trial_index,
    )
    strict_fraction = runner._synthetic_locked_measurement_audit(
        contract={
            **REQUIRED_LOCKED_MEASUREMENT_AUDIT_CONTRACT,
            "minimum_locked_donor_type_reliability_fraction": 1.0,
        },
        scenario="missing_fine_types",
        trial_index=trial_index,
        seed=runner._trial_seed(1729, "missing_fine_types", condition, trial_index),
        donor_ids=missing_locked.donor_ids,
        section_ids=missing_locked.section_ids,
        type_labels=missing_locked.type_labels,
        type_names=missing_locked.type_names,
        gene_ids=missing_locked.gene_ids,
        planned_stratum_ids=missing_locked.planned_stratum_ids,
        molecular_residual=missing_locked.molecular_targets - missing_locked.reference_means,
        registration_quality_scores=missing_locked.registration_quality_scores,
        technical_covariates=missing_locked.technical_covariates,
        coordinate_features=missing_locked.coordinate_features,
    )
    assert strict_fraction["pass"] is False


def test_authorizing_dgp_places_signal_in_the_prespecified_image_sources() -> None:
    def apparent_multivariate_r2(artifact, crop_id: str) -> float:
        features = artifact.image_feature_tensor[:, artifact.crop_ids.index(crop_id), :]
        design = np.column_stack((np.ones(len(features)), features))
        target = artifact.molecular_targets - artifact.reference_means
        prediction = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        denominator = np.square(target - target.mean(axis=0)).sum(axis=0)
        return float(np.mean(1.0 - np.square(target - prediction).sum(axis=0) / denominator))

    artifacts = {
        condition: runner.build_synthetic_calibration_pair(
            "crop_family_multiplicity",
            condition,
            0,
        )[0]
        for condition in runner.AUTHORIZING_CALIBRATION_CONDITIONS
    }
    null = artifacts[GLOBAL_NULL_CONDITION]
    assert apparent_multivariate_r2(null, "crop_112um") < 0.05

    g2 = artifacts[runner.BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS["G2_local_context"]]
    assert apparent_multivariate_r2(g2, "crop_112um") > (
        apparent_multivariate_r2(g2, "blank_patch") + 0.02
    )

    for decision_id, informative, negative in (
        ("G3_nucleus_intrinsic", "nucleus_mask_only", "nucleus_mask_blurred_112um"),
        ("G3_cell_intrinsic", "cell_mask_only", "cell_mask_blurred_112um"),
        (
            "G3_context_only",
            "target_cell_removed_112um",
            "target_cell_removed_blurred_112um",
        ),
    ):
        artifact = artifacts[runner.BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS[decision_id]]
        assert apparent_multivariate_r2(artifact, informative) > (
            apparent_multivariate_r2(artifact, negative) + 0.02
        )

    mixed = artifacts[runner.BOUNDARY_CONDITION_IDS_BY_HYPOTHESIS["G3_mixed_intrinsic_context"]]
    full = apparent_multivariate_r2(mixed, "crop_112um")
    assert full > apparent_multivariate_r2(mixed, "nucleus_mask_only") + 0.02
    assert full > apparent_multivariate_r2(mixed, "target_cell_removed_112um") + 0.02


def test_runner_calls_actual_entrypoint_hashes_reports_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _settings()
    original_affinity = os.sched_getaffinity(0)
    original_address_space_limit = resource.getrlimit(resource.RLIMIT_AS)
    pool_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    original_pool_environment = {name: os.environ.get(name) for name in pool_variables}
    settings_sha256 = canonical_sha256(settings)
    calls: list[Mapping[str, object]] = []

    def fake_production_gate(development, locked, **kwargs):
        assert len(os.sched_getaffinity(0)) == 1
        for variable in pool_variables:
            assert os.environ[variable] == "1"
        assert development.cohort_id == "SYNTHETIC_CALIBRATION"
        assert locked.cohort_id == "SYNTHETIC_CALIBRATION"
        assert len(set(development.donor_ids.tolist())) == 10
        assert len(set(locked.donor_ids.tolist())) == 5
        assert tuple(development.crop_ids) == tuple(HEST_CROP_CONTRACT)
        assert len(development.reference_split_ids) == 3
        assert kwargs["final_inference"] is True
        assert kwargs["synthetic_calibration_mode"] is True
        assert kwargs["total_permutations"] == settings["permutations_per_null"]
        assert kwargs["permutation_seeds"] == tuple(settings["permutation_seeds"])
        assert kwargs["permutations_per_seed"] == settings["permutations_per_seed"]
        assert kwargs["ranks"] == tuple(settings["target_rank_grid"])
        assert kwargs["alphas"] == tuple(settings["ridge_penalty_grid"])
        assert (
            kwargs["confirmatory_analysis_plan_sha256"]
            == settings["confirmatory_analysis_plan_sha256"]
        )
        for name, expected in settings["gate_parameters"].items():
            assert kwargs[name] == expected
        calls.append(kwargs)
        component_pass = len(calls) % 2 == 0
        seed_rows = [
            {
                "seed": seed,
                "required_unique_permutations": settings["permutations_per_seed"],
                "generated_unique_permutations": settings["permutations_per_seed"],
            }
            for seed in settings["permutation_seeds"]
        ]
        development_artifact_sha256 = runner.morphology_artifact_content_sha256(development)
        locked_artifact_sha256 = runner.morphology_artifact_content_sha256(locked)
        realization_sha256 = canonical_sha256(
            {
                "calibration_trial_identity": kwargs["calibration_trial_identity"],
                "calibration_run_contract_sha256": kwargs["calibration_run_contract_sha256"],
                "development_artifact_sha256": development_artifact_sha256,
                "locked_artifact_sha256": locked_artifact_sha256,
            }
        )
        report = {
            "schema_version": ACTUAL_GATE_REPORT_SCHEMA,
            "component_pass": component_pass,
            "final_inference": True,
            "synthetic_calibration_execution": True,
            "scientific_authorization_suppressed": True,
            "calibration_exact_gate_settings_sha256": settings_sha256,
            "calibration_exact_gate_settings": settings,
            "calibration_trial_identity": kwargs["calibration_trial_identity"],
            "calibration_run_contract_sha256": kwargs["calibration_run_contract_sha256"],
            "calibration_development_artifact_sha256": development_artifact_sha256,
            "calibration_locked_artifact_sha256": locked_artifact_sha256,
            "calibration_trial_realization_sha256": realization_sha256,
            "checks": {name: component_pass for name in REQUIRED_COMPLETE_GATE_CHECKS},
            "hypothesis_decisions": {
                name: {
                    "tested": True,
                    "pass": False,
                    **(
                        {
                            "registration_quality_sensitivity_pass": False,
                            "registration_quality_sensitivity": {"synthetic_test": True},
                        }
                        if name in {"G3_nucleus_intrinsic", "G3_cell_intrinsic"}
                        else (
                            {
                                (
                                    "incremental_intrinsic_registration_quality_sensitivity_pass"
                                ): False,
                                "incremental_intrinsic_registration_quality_sensitivity": {
                                    "synthetic_test": True
                                },
                            }
                            if name == "G3_mixed_intrinsic_context"
                            else {}
                        )
                    ),
                }
                for name in REQUIRED_HYPOTHESIS_DECISIONS
            },
            "morphology_source_conclusion": (
                "inconclusive"
                if len(calls) == 3
                else (
                    "no_morphology_specific_information"
                    if len(calls) % 2 == 1
                    else "mixed_intrinsic_and_contextual_information"
                )
            ),
            "permutation_control": {
                "total_permutations": settings["permutations_per_null"],
                "seeds": seed_rows,
            },
            "spatial_block_permutation_control": {
                "total_permutations": settings["permutations_per_null"],
                "seeds": seed_rows,
            },
            "authorizes_full_heir": False,
            "authorizes_population_inference": False,
            "authorizes_external_generalization": False,
            "authorizes_validated_regional_association": False,
            "authorizes_nucleus_intrinsic_claim": False,
            "authorizes_cell_intrinsic_claim": False,
            "test_report_serial": len(calls),
        }
        return report

    monkeypatch.setattr(runner, "evaluate_morphology_ridge_gate", fake_production_gate)
    checkpoint = tmp_path / "calibration.checkpoint.json"
    plan = runner.CalibrationRunPlan(
        exact_gate_settings=settings,
        trials_per_condition=1,
        smoke_test=True,
        device="cpu",
    )
    execution = runner.run_actual_gate_calibration(
        plan,
        checkpoint_path=checkpoint,
    )
    assert len(calls) == 2 * len(REQUIRED_CALIBRATION_SCENARIOS)
    assert execution["production_contract_satisfied"] is False
    assert execution["authorizes_scientific_claims"] is False
    assert execution["authorizes_final_inference"] is False
    assert execution["synthetic_data_only"] is True
    assert execution["resource_limits"]["max_cpu_threads"] == 1
    assert execution["resource_limits"]["maximum_process_rss_gib"] == 16.0
    assert execution["resource_limits"]["maximum_address_space_gib"] == 64.0
    assert execution["resource_limits"]["process_isolation"] == "in_process_smoke"
    assert (
        execution["resource_limits"]["observed_thread_pools"]["address_space"]["maximum_gib"]
        == 64.0
    )
    assert (
        execution["resource_limits"]["observed_thread_pools"]["cpu_affinity"]["logical_cpu_count"]
        == 1
    )
    assert os.sched_getaffinity(0) == original_affinity
    assert resource.getrlimit(resource.RLIMIT_AS) == original_address_space_limit
    assert {name: os.environ.get(name) for name in pool_variables} == original_pool_environment
    evidence = execution["evidence"]
    assert evidence["schema"] == CALIBRATION_EVIDENCE_SCHEMA
    assert execution["evidence_content_sha256"] == canonical_sha256(evidence)
    contract = evidence["run_contract"]
    assert evidence["run_contract_sha256"] == canonical_sha256(contract)
    manifest = evidence["trial_report_manifest"]
    assert manifest["report_reference_count"] == 2 * len(REQUIRED_CALIBRATION_SCENARIOS)
    assert manifest["manifest_content_sha256"] == canonical_sha256(
        {name: value for name, value in manifest.items() if name != "manifest_content_sha256"}
    )
    assert manifest["storage"]["kind"] == "content_addressed_directory"
    for report_hash in {
        value
        for scenario in manifest["ordered_report_sha256s_by_scenario_condition"].values()
        for hashes in scenario.values()
        for value in hashes
    }:
        assert (
            Path(manifest["storage"]["root_path"]) / report_hash[:2] / (report_hash + ".json")
        ).is_file()
    recomputed = calibration_compiler._recompute_evidence_from_trial_manifest(
        manifest,
        settings=settings,
        condition_ids=tuple(contract["conditions"]),
        expected_trials=1,
        base_seed=int(contract["base_seed"]),
        run_contract_sha256=evidence["run_contract_sha256"],
        decision_truth_by_condition=contract["dgp_effect_spec"]["decision_truth_by_condition"],
    )
    assert recomputed == evidence["scenario_results"]

    swapped = copy.deepcopy(manifest)
    first_scenario = REQUIRED_CALIBRATION_SCENARIOS[0]
    first_condition_id, second_condition_id = tuple(contract["conditions"])
    first_hash = swapped["ordered_report_sha256s_by_scenario_condition"][first_scenario][
        first_condition_id
    ][0]
    second_hash = swapped["ordered_report_sha256s_by_scenario_condition"][first_scenario][
        second_condition_id
    ][0]
    swapped["ordered_report_sha256s_by_scenario_condition"][first_scenario][first_condition_id][
        0
    ] = second_hash
    swapped["ordered_report_sha256s_by_scenario_condition"][first_scenario][second_condition_id][
        0
    ] = first_hash
    swapped["manifest_content_sha256"] = canonical_sha256(
        {name: value for name, value in swapped.items() if name != "manifest_content_sha256"}
    )
    with pytest.raises(ValueError, match="frozen trial identity"):
        calibration_compiler._recompute_evidence_from_trial_manifest(
            swapped,
            settings=settings,
            condition_ids=tuple(contract["conditions"]),
            expected_trials=1,
            base_seed=int(contract["base_seed"]),
            run_contract_sha256=evidence["run_contract_sha256"],
            decision_truth_by_condition=contract["dgp_effect_spec"]["decision_truth_by_condition"],
        )

    first_path = Path(manifest["storage"]["root_path"]) / first_hash[:2] / (first_hash + ".json")
    bypass = json.loads(first_path.read_text(encoding="utf-8"))
    bypass["hypothesis_decisions"]["G3_nucleus_intrinsic"]["pass"] = True
    with pytest.raises(ValueError, match="bypasses registration-quality sensitivity"):
        runner.actual_gate_trial_outcome(
            bypass,
            exact_gate_settings=settings,
            expected_trial_identity=bypass["calibration_trial_identity"],
            expected_run_contract_sha256=evidence["run_contract_sha256"],
            expected_decision_truth=contract["dgp_effect_spec"]["decision_truth_by_condition"][
                first_condition_id
            ],
        )

    mixed_bypass = json.loads(first_path.read_text(encoding="utf-8"))
    mixed_bypass["hypothesis_decisions"]["G3_mixed_intrinsic_context"]["pass"] = True
    with pytest.raises(
        ValueError,
        match="mixed decision bypasses incremental registration-quality sensitivity",
    ):
        runner.actual_gate_trial_outcome(
            mixed_bypass,
            exact_gate_settings=settings,
            expected_trial_identity=mixed_bypass["calibration_trial_identity"],
            expected_run_contract_sha256=evidence["run_contract_sha256"],
            expected_decision_truth=contract["dgp_effect_spec"]["decision_truth_by_condition"][
                first_condition_id
            ],
        )

    realization_bypass = json.loads(first_path.read_text(encoding="utf-8"))
    realization_bypass["calibration_trial_realization_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="deterministic trial realization"):
        runner.actual_gate_trial_outcome(
            realization_bypass,
            exact_gate_settings=settings,
            expected_trial_identity=realization_bypass["calibration_trial_identity"],
            expected_run_contract_sha256=evidence["run_contract_sha256"],
            expected_decision_truth=contract["dgp_effect_spec"]["decision_truth_by_condition"][
                first_condition_id
            ],
        )

    duplicated = copy.deepcopy(manifest)
    duplicated["ordered_report_sha256s_by_scenario_condition"][first_scenario][second_condition_id][
        0
    ] = first_hash
    all_hashes = [
        digest
        for scenario_rows in duplicated["ordered_report_sha256s_by_scenario_condition"].values()
        for condition_hashes in scenario_rows.values()
        for digest in condition_hashes
    ]
    duplicated["unique_report_count"] = len(set(all_hashes))
    duplicated["manifest_content_sha256"] = canonical_sha256(
        {name: value for name, value in duplicated.items() if name != "manifest_content_sha256"}
    )
    with pytest.raises(ValueError, match="frozen trial identity"):
        calibration_compiler._recompute_evidence_from_trial_manifest(
            duplicated,
            settings=settings,
            condition_ids=tuple(contract["conditions"]),
            expected_trials=1,
            base_seed=int(contract["base_seed"]),
            run_contract_sha256=evidence["run_contract_sha256"],
            decision_truth_by_condition=contract["dgp_effect_spec"]["decision_truth_by_condition"],
        )
    assert contract["base_seed"] == 1729
    assert contract["dgp_effect_spec"]["authorizing_boundary_calibration"] is False
    assert (
        "minimum meaningful effect"
        in contract["dgp_effect_spec"]["effect_definition"]["scientific_interpretation"]
    )
    assert "preliminary" in execution["non_authorizing_reason"]
    first_condition = evidence["scenario_results"][REQUIRED_CALIBRATION_SCENARIOS[0]][
        GLOBAL_NULL_CONDITION
    ]
    assert first_condition["trial_report_set_sha256"] == canonical_sha256(
        {
            "ordered_actual_gate_report_sha256": manifest[
                "ordered_report_sha256s_by_scenario_condition"
            ][REQUIRED_CALIBRATION_SCENARIOS[0]][GLOBAL_NULL_CONDITION]
        }
    )
    assert first_condition["actual_gate_executions"] == 1
    assert first_condition["complete_gate_passes"] == 0
    assert first_condition["hypothesis_decision_passes"] == {
        name: 0 for name in REQUIRED_HYPOTHESIS_DECISIONS
    }
    assert (
        first_condition["morphology_source_conclusion_counts"]["no_morphology_specific_information"]
        == 1
    )
    second_null = evidence["scenario_results"][REQUIRED_CALIBRATION_SCENARIOS[1]][
        GLOBAL_NULL_CONDITION
    ]
    assert second_null["morphology_source_conclusion_counts"]["inconclusive"] == 1
    assert first_condition["permutation_nulls"]["local_roi_seed_counts"] == {
        "17": 333,
        "29": 333,
        "41": 333,
    }
    assert checkpoint.is_file()
    checkpoint_content = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_content["authorizes_final_inference"] is False

    def unexpected_gate_call(*args, **kwargs):
        raise AssertionError("completed calibration trials must resume from the checkpoint")

    monkeypatch.setattr(runner, "evaluate_morphology_ridge_gate", unexpected_gate_call)
    resumed = runner.run_actual_gate_calibration(plan, checkpoint_path=checkpoint)
    assert resumed["evidence"] == evidence

    tampered = json.loads(checkpoint.read_text(encoding="utf-8"))
    first_key = "%s.%s" % (REQUIRED_CALIBRATION_SCENARIOS[0], GLOBAL_NULL_CONDITION)
    tampered["completed_trials"][first_key][0]["component_pass"] = True
    checkpoint.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash differs"):
        runner.run_actual_gate_calibration(plan, checkpoint_path=checkpoint)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="preliminary.*cannot issue"):
        compile_actual_gate_calibration_receipt(
            settings,
            config["thresholds"],
            evidence,
            confidence_level=required_simultaneous_confidence_level(),
        )


def test_reduced_trials_require_explicit_non_authorizing_smoke_mode() -> None:
    plan = runner.CalibrationRunPlan(
        exact_gate_settings=_settings(),
        trials_per_condition=1,
    )
    with pytest.raises(ValueError, match="non-authorizing"):
        plan.validate()


def test_non_smoke_run_requires_dedicated_cli_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(runner.DEDICATED_PROCESS_ENV, raising=False)
    plan = runner.CalibrationRunPlan(
        exact_gate_settings=_settings(),
        trials_per_condition=runner.PRODUCTION_TRIALS_PER_CONDITION,
    )
    with pytest.raises(RuntimeError, match="dedicated calibration CLI"):
        runner.run_actual_gate_calibration(plan)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"max_cpu_threads": 0}, "max_cpu_threads"),
        ({"maximum_process_rss_gib": 0.0}, "maximum_process_rss_gib"),
        ({"maximum_address_space_gib": 0.0}, "maximum_address_space_gib"),
    ),
)
def test_runner_rejects_invalid_resource_limits(overrides, message: str) -> None:
    parameters = {
        "exact_gate_settings": _settings(),
        "trials_per_condition": 1,
        "smoke_test": True,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        runner.CalibrationRunPlan(**parameters).validate()


def test_runner_refuses_to_start_above_rss_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "_process_rss_gib", lambda: 17.0)
    monkeypatch.setattr(runner, "_process_virtual_memory_gib", lambda: 0.5)
    plan = runner.CalibrationRunPlan(
        exact_gate_settings=_settings(),
        trials_per_condition=1,
        smoke_test=True,
        maximum_process_rss_gib=16.0,
    )
    with pytest.raises(MemoryError, match="RSS.*exceeds"):
        runner.run_actual_gate_calibration(plan, checkpoint_path=tmp_path / "checkpoint.json")


def test_runner_refuses_to_start_above_address_space_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "_process_virtual_memory_gib", lambda: 2.0)
    plan = runner.CalibrationRunPlan(
        exact_gate_settings=_settings(),
        trials_per_condition=1,
        smoke_test=True,
        maximum_process_rss_gib=1.0,
        maximum_address_space_gib=1.0,
    )
    with pytest.raises(MemoryError, match="virtual memory.*exceeds"):
        runner.run_actual_gate_calibration(plan, checkpoint_path=tmp_path / "checkpoint.json")


def test_address_space_limit_never_raises_a_stricter_existing_soft_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    limits: list[tuple[int, int]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (8 * gib, resource.RLIM_INFINITY))
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, value: limits.append(value))
    monkeypatch.setattr(runner, "_process_virtual_memory_gib", lambda: 2.0)

    with runner._address_space_limit(16.0) as observed:
        assert observed["maximum_gib"] == 8.0
        assert observed["preexisting_soft_limit_preserved"] is True

    assert limits == [
        (8 * gib, resource.RLIM_INFINITY),
        (8 * gib, resource.RLIM_INFINITY),
    ]
