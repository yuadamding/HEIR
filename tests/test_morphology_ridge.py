from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import heir.evaluation.morphology_gate as gate_module
import heir.evaluation.morphology_ridge as ridge_module
from heir.data import MorphologyRidgeDatasetArtifact
from heir.evaluation import (
    donor_type_block_permutation,
    evaluate_morphology_ridge_gate,
    fit_oracle_ridge_probe,
    predict_oracle_ridge,
    validate_experiment_identity,
)
from heir.evaluation.control_models import HEST_CROP_CONTRACT
from heir.evaluation.morphology_calibration_runner import (
    build_synthetic_calibration_pair,
    synthetic_completed_confirmatory_design_binding,
)
from heir.evaluation.power import (
    G2_MULTIPLICITY_METHOD,
    G3_MULTIPLICITY_METHOD,
    PRELIMINARY_ALTERNATIVE_CONDITION,
    REQUIRED_G3_CONTRAST_PAIRS,
)


def _artifact(
    donors: tuple[str, ...],
    *,
    role: str,
    source_offset: int,
    coordinate_signal: bool = False,
) -> MorphologyRidgeDatasetArtifact:
    features = []
    targets = []
    means = []
    coordinates = []
    labels = []
    donor_ids = []
    block_ids = []
    roi_ids = []
    observation_ids = []
    for donor_position, donor in enumerate(donors):
        for type_index in range(2):
            state = np.asarray([-1.0, 0.7, -0.2, 0.4, 0.9, -0.8, 0.1, -0.5, 0.3, -0.4, 1.0, -0.9])
            local_coordinate = np.sin(
                np.arange(12, dtype=np.float64) * 2.3 + donor_position + type_index
            )
            if coordinate_signal:
                state = local_coordinate.copy()
            sign = -1.0 if type_index == 0 else 1.0
            feature = np.column_stack((state, sign * state, np.full(12, sign)))
            reference = np.tile(
                np.asarray([5.0 * donor_position, 3.0 * type_index, -2.0 * donor_position]),
                (12, 1),
            )
            residual = np.column_stack((state, sign * 0.5 * state, np.zeros(12)))
            features.append(feature)
            targets.append(reference + residual)
            means.append(reference)
            coordinates.append(np.column_stack((local_coordinate, np.square(local_coordinate))))
            labels.extend([type_index] * 12)
            donor_ids.extend([donor] * 12)
            block_ids.extend([f"{donor}/section_{donor}/block_{index // 4}" for index in range(12)])
            roi_ids.extend(
                [
                    f"{donor}/section_{donor}/type_{type_index}/roi_{index // 4}"
                    for index in range(12)
                ]
            )
            observation_ids.extend([f"{donor}_{type_index}_{index}" for index in range(12)])
    cells = len(labels)
    section_ids = np.asarray([value.split("/")[-2] for value in block_ids])
    donor_disease = {
        donor: ("Control" if index % 2 == 0 else "Disease") for index, donor in enumerate(donors)
    }
    disease_states = np.asarray([donor_disease[value] for value in donor_ids])
    feature_values = np.concatenate(features)
    target_values = np.concatenate(targets)
    reference_values = np.concatenate(means)
    planned_strata = tuple(
        sorted(
            {
                "%s|%s|%s" % (donor, section, ("epithelial", "immune")[label])
                for donor, section, label in zip(donor_ids, section_ids, labels)
            }
        )
    )
    return MorphologyRidgeDatasetArtifact(
        observation_ids=np.asarray(observation_ids),
        donor_ids=np.asarray(donor_ids),
        block_ids=np.asarray(block_ids),
        roi_ids=np.asarray(roi_ids),
        type_labels=np.asarray(labels, dtype=np.int64),
        type_names=("epithelial", "immune"),
        frozen_features=feature_values,
        molecular_targets=target_values,
        reference_means=reference_values,
        coordinate_features=np.concatenate(coordinates),
        stain_features=np.empty((cells, 0), dtype=np.float64),
        stain_feature_names=(),
        composition_features=np.empty((cells, 0), dtype=np.float64),
        composition_feature_names=(),
        technical_covariates=np.empty((cells, 0), dtype=np.float64),
        technical_covariate_names=(),
        gene_ids=("G1", "G2", "G3"),
        type_marker_gene_ids=("MARKER",),
        feature_space_id="frozen-test-features-v1",
        feature_checkpoint_sha256="1" * 64,
        molecular_space_id="log-normalized-test-genes-v1",
        reference_source_sha256=str(source_offset) * 64,
        label_source_sha256="3" * 64,
        target_source_sha256=str(source_offset + 1) * 64,
        registration_source_sha256="6" * 64,
        exclusion_policy_sha256="7" * 64,
        registration_method="high-confidence-one-to-one",
        encoder_name="frozen-synthetic-encoder",
        crop_scale="small_cell_centered",
        cohort_id="HEST",
        cohort_release="synthetic-locked-study",
        assay="Xenium",
        observation_level="cell",
        target_construction="registered_cell_expression",
        reference_pool_independent=True,
        labels_independent_of_images=True,
        registration_is_one_to_one=True,
        role=role,
        section_ids=section_ids,
        disease_states=disease_states,
        site_ids=np.repeat("site_a", cells),
        batch_ids=np.repeat("batch_a", cells),
        image_feature_tensor=feature_values[:, None, :],
        crop_ids=("crop_112um",),
        crop_roles=("registered_cell_local_context_112um",),
        crop_comparison_families=("g2_primary",),
        primary_crop_id="crop_112um",
        nuclear_morphometrics=np.empty((cells, 0), dtype=np.float64),
        nuclear_morphometric_names=(),
        cell_morphometrics=np.empty((cells, 0), dtype=np.float64),
        cell_morphometric_names=(),
        cellvit_context_features=np.empty((cells, 0), dtype=np.float64),
        cellvit_context_feature_names=(),
        local_density_features=np.empty((cells, 0), dtype=np.float64),
        local_density_feature_names=(),
        boundary_features=np.empty((cells, 0), dtype=np.float64),
        boundary_feature_names=(),
        spatial_control_features=np.concatenate(coordinates),
        spatial_control_feature_names=("x", "x_squared"),
        planned_stratum_ids=planned_strata,
        planned_stratum_manifest_sha256="8" * 64,
        coverage_audit={
            "retained_fraction": 1.0,
            "reference_membership_sha256_by_split": {"primary": "c" * 64},
            "locked_measurement_audit": {"pass": True},
        },
        reference_evaluation_balance={"primary": {"pass": True}},
        study_manifest_sha256="9" * 64,
        opening_receipt_sha256="d" * 64,
        measurement_receipt_sha256="a" * 64,
        measurement_source_sha256="b" * 64,
        hypothesis_ids=("H-CELL",),
        scientific_scope="registered_cell_local_context_association",
        evidence_scope="internal_locked_hest",
        authorizes_nucleus_intrinsic_claim=False,
        registration_quality_scores=np.zeros(cells, dtype=np.float64),
        registration_quality_strata=np.repeat("best", cells),
        registration_quality_cutoffs={
            "best": 0.25,
            "intermediate": 0.6,
            "near_threshold": 1.0,
        },
        registration_quality_definition=(
            "max(annotation_nucleus_error/section_median_nucleus_diameter/diameter_limit,"
            "annotation_nucleus_error/section_median_nearest_neighbor_distance/neighbor_limit)"
        ),
        reference_split_ids=("primary",),
        reference_means_by_split=reference_values[:, None, :],
    )


def _with_required_nuisance_features(
    artifact: MorphologyRidgeDatasetArtifact,
) -> MorphologyRidgeDatasetArtifact:
    values = artifact.coordinate_features[:, :1].copy()
    return replace(
        artifact,
        stain_features=values,
        stain_feature_names=("stain",),
        nuclear_morphometrics=values,
        nuclear_morphometric_names=("nuclear",),
        cell_morphometrics=values,
        cell_morphometric_names=("cell",),
        cellvit_context_features=values,
        cellvit_context_feature_names=("cellvit",),
        local_density_features=values,
        local_density_feature_names=("density",),
        boundary_features=values,
        boundary_feature_names=("boundary",),
    )


def test_g3_registration_quality_requires_best_stratum_effect() -> None:
    artifact = _artifact(
        ("locked_1", "locked_2"),
        role="locked_test",
        source_offset=8,
    )
    within_type_position = np.tile(np.arange(12), 4)
    best = within_type_position < 6
    artifact = replace(
        artifact,
        registration_quality_scores=np.where(best, 0.1, 0.8),
        registration_quality_strata=np.where(best, "best", "near_threshold"),
    )
    truth = np.tile(np.linspace(-1.0, 1.0, 12), 4)[:, None]
    comparator = np.zeros_like(truth)

    def sensitivity(focal: np.ndarray, *, contrast_name: str = "nucleus_test") -> dict[str, object]:
        focal_macro, _, _ = gate_module.macro_r2(
            truth,
            focal,
            artifact.donor_ids,
            artifact.type_labels,
            4,
        )
        comparator_macro, _, _ = gate_module.macro_r2(
            truth,
            comparator,
            artifact.donor_ids,
            artifact.type_labels,
            4,
        )
        return dict(
            gate_module._registration_quality_contrast_sensitivity(
                {"focal": (focal, truth), "comparator": (comparator, truth)},
                {contrast_name: ("focal", "comparator")},
                {
                    contrast_name: {
                        "mean_delta_r2": focal_macro - comparator_macro,
                    }
                },
                artifact,
                minimum_support=4,
                minimum_delta=0.01,
            )
        )

    best_only_signal = np.where(best[:, None], truth, comparator)
    robust = sensitivity(best_only_signal)
    assert robust["contrasts"]["nucleus_test"]["pass"] is True
    assert robust["contrasts"]["nucleus_test"][
        "quality_noninferiority_margin_delta_r2"
    ] == pytest.approx(0.01)
    assert robust["contrasts"]["nucleus_test"]["best_registration_noninferior_to_all_rows"] is True

    near_threshold_only_signal = np.where(best[:, None], comparator, truth)
    suspicious = sensitivity(near_threshold_only_signal)
    assert suspicious["contrasts"]["nucleus_test"]["pass"] is False
    assert (
        suspicious["contrasts"]["nucleus_test"]["effect_not_driven_only_by_near_threshold_rows"]
        is False
    )

    incremental_name = "full_context_vs_target_removed_white"
    incremental_quality = sensitivity(
        near_threshold_only_signal,
        contrast_name=incremental_name,
    )["contrasts"][incremental_name]
    source_flags = gate_module._g3_quality_gated_source_flags(
        {
            "nucleus_test": {"pass": True},
            "context_test": {"pass": True},
            incremental_name: {"pass": True},
            "full_context_vs_nucleus_test": {"pass": True},
        },
        {
            "nucleus_test": {"pass": True},
            incremental_name: incremental_quality,
        },
    )
    assert source_flags["full_vs_context_unstratified"] is True
    assert source_flags["full_vs_context_quality_sensitivity_pass"] is False
    assert source_flags["mixed_information"] is False


def test_oracle_per_type_ridge_uses_supplied_matched_reference_means() -> None:
    development = _artifact(
        ("development_1", "development_2", "development_3"),
        role="development",
        source_offset=4,
    )
    fit = fit_oracle_ridge_probe(
        development.frozen_features,
        development.molecular_targets,
        development.reference_means,
        development.type_labels,
        development.donor_ids,
        development.technical_covariates,
        num_types=2,
        rank=1,
        alpha=1.0e-4,
        device="cpu",
    )
    coordinates, prediction = predict_oracle_ridge(
        fit,
        development.frozen_features,
        development.reference_means,
        development.type_labels,
    )
    assert coordinates.shape == (len(development.observation_ids), 1)
    np.testing.assert_allclose(prediction, development.molecular_targets, atol=2.0e-4)

    shifted_reference = development.reference_means + 100.0
    _, shifted = predict_oracle_ridge(
        fit, development.frozen_features, shifted_reference, development.type_labels
    )
    np.testing.assert_allclose(shifted - prediction, 100.0)
    assert not np.allclose(fit.coefficients[0], fit.coefficients[1])


def test_spatial_block_permutation_is_deterministic_and_crosses_blocks() -> None:
    artifact = _artifact(
        ("development_1", "development_2", "development_3"),
        role="development",
        source_offset=4,
    )
    first = donor_type_block_permutation(
        artifact.donor_ids, artifact.type_labels, artifact.block_ids, seed=17
    )
    second = donor_type_block_permutation(
        artifact.donor_ids, artifact.type_labels, artifact.block_ids, seed=17
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(artifact.donor_ids, artifact.donor_ids[first])
    np.testing.assert_array_equal(artifact.type_labels, artifact.type_labels[first])
    assert np.mean(artifact.block_ids != artifact.block_ids[first]) == 1.0


def test_multicandidate_null_reselects_pipeline_hyperparameters() -> None:
    development = _artifact(
        tuple(f"development_{index}" for index in range(5)),
        role="development",
        source_offset=4,
    )
    locked = _artifact(
        tuple(f"locked_{index}" for index in range(5)),
        role="locked_test",
        source_offset=8,
    )
    control = ridge_module._evaluate_permutation_null(
        development,
        locked,
        null_kind="local_within_roi",
        matched=1.0,
        ranks=(1, 2),
        alphas=(0.1, 1.0),
        permutation_seeds=(17,),
        total_permutations=2,
        minimum_support=8,
        minimum_shuffle_delta=0.0,
        maximum_permutation_p=1.0,
        minimum_shuffled_fraction=0.5,
        include_composition=False,
        prespecified_fixed_hyperparameters=False,
        device="cpu",
    )
    assert control["full_pipeline_hyperparameters_reselected"] is True
    assert control["hyperparameter_selection"] == "repeated_development_donor_fold_selection"
    assert sum(row["count"] for row in control["selected_hyperparameter_counts"]) == 2


def test_ridge_gate_refits_preserving_null_and_does_not_authorize_heir() -> None:
    development = _artifact(
        tuple(f"development_{index}" for index in range(5)),
        role="development",
        source_offset=4,
    )
    locked = _artifact(
        tuple(f"locked_{index}" for index in range(5)),
        role="locked_test",
        source_offset=8,
    )
    development = replace(
        development,
        reference_split_ids=("primary", "alternate"),
        reference_means_by_split=np.stack(
            (development.reference_means, development.reference_means), axis=1
        ),
        reference_evaluation_balance={
            "primary": {"pass": True},
            "alternate": {"pass": True},
        },
        coverage_audit={
            **development.coverage_audit,
            "reference_membership_sha256_by_split": {
                "primary": "c" * 64,
                "alternate": "d" * 64,
            },
        },
    )
    locked = replace(
        locked,
        reference_split_ids=("primary", "alternate"),
        reference_means_by_split=np.stack((locked.reference_means, locked.reference_means), axis=1),
        reference_evaluation_balance={
            "primary": {"pass": True},
            "alternate": {"pass": True},
        },
        coverage_audit={
            **locked.coverage_audit,
            "reference_membership_sha256_by_split": {
                "primary": "e" * 64,
                "alternate": "f" * 64,
            },
        },
    )
    report = evaluate_morphology_ridge_gate(
        development,
        locked,
        ranks=(1,),
        alphas=(1.0e-4,),
        permutation_seeds=(17, 29, 41),
        permutations_per_seed=100,
        minimum_support=8,
        prespecified_fixed_hyperparameters=True,
        device="cpu",
    )
    assert report["component_pass"] is True
    assert report["authorizes_full_heir"] is False
    assert report["oracle_type_only"] is True
    assert report["nucleus_hypothesis_tested"] is False
    assert report["morphology_source_conclusion"] == "not_tested"
    assert report["crop_source_not_inferred_from_observation_level"] is True
    assert report["hypothesis_decisions"]["G2_local_context"]["tested"] is True
    assert [row["split_id"] for row in report["reference_split_sensitivity"]] == [
        "primary",
        "alternate",
    ]
    control = report["permutation_control"]
    assert control["training_probe_refit_for_each_permutation"] is True
    assert control["hyperparameter_selection"] == "manifest_prespecified_single_candidate"
    assert control["total_permutations"] == 300
    assert control["unique_permutations"] is True
    assert control["one_combined_scientific_permutation_pool"] is True
    assert control["seeds_are_generation_streams_not_independent_tests"] is True
    assert [row["required_unique_permutations"] for row in control["seeds"]] == [100] * 3
    assert [row["generated_unique_permutations"] for row in control["seeds"]] == [100] * 3
    assert all(row["empirical_p"] < 0.01 for row in control["seeds"])
    block_control = report["spatial_block_permutation_control"]
    assert block_control["total_permutations"] == 300
    assert all(row["minimum_cross_block_fraction"] == 1.0 for row in block_control["seeds"])
    assert report["primary_metrics"]["donor_equal_type_equal_residual_coordinate_r2"] > 0.95
    assert report["checks"]["beats_coordinate_only"] is True
    assert report["checks"]["exact_donor_paired_main_effect"] is True
    assert (
        report["hypothesis_decisions"]["G2_local_context"]["multiplicity_method"]
        == G2_MULTIPLICITY_METHOD
    )
    assert report["coverage"]["locked_donor_type"]["supported_fraction"] == 1.0
    assert report["stratification"]["section"]["available"] is True
    assert len(report["leave_one_locked_donor_out"]["matched_macro_r2"]) == 5
    assert report["donor_bootstrap"]["matched_macro_r2"]["ci_95"][0] > 0.0

    repeated = evaluate_morphology_ridge_gate(
        development,
        locked,
        ranks=(1,),
        alphas=(1.0e-4,),
        permutation_seeds=(17, 29, 41),
        permutations_per_seed=100,
        minimum_support=8,
        prespecified_fixed_hyperparameters=True,
        device="cpu",
    )
    assert repeated["permutation_control"] == control
    assert repeated["spatial_block_permutation_control"] == block_control


def test_ridge_gate_rejects_too_few_permutations_and_coordinate_only_signal() -> None:
    development = _artifact(
        tuple(f"development_{index}" for index in range(5)),
        role="development",
        source_offset=4,
        coordinate_signal=True,
    )
    locked = _artifact(
        tuple(f"locked_{index}" for index in range(5)),
        role="locked_test",
        source_offset=8,
        coordinate_signal=True,
    )
    with pytest.raises(ValueError, match="at least 100 total permutations"):
        evaluate_morphology_ridge_gate(
            development,
            locked,
            ranks=(1,),
            alphas=(1.0,),
            permutation_seeds=(17, 29, 41),
            permutations_per_seed=99,
            total_permutations=99,
            minimum_support=8,
            device="cpu",
        )

    report = evaluate_morphology_ridge_gate(
        development,
        locked,
        ranks=(1,),
        alphas=(1.0e-4,),
        permutation_seeds=(17, 29, 41),
        permutations_per_seed=100,
        minimum_support=8,
        prespecified_fixed_hyperparameters=True,
        device="cpu",
    )
    assert report["checks"]["beats_coordinate_only"] is False
    assert report["component_pass"] is False


@pytest.mark.parametrize(
    "overrides",
    (
        {"permutation_seeds": (17.9,)},
        {"permutations_per_seed": 100.5},
        {"total_permutations": 100.5},
        {"ranks": (1.5,)},
    ),
)
def test_direct_gate_rejects_nonintegral_randomization_and_rank_contract(
    overrides: dict[str, object],
) -> None:
    development = _artifact(
        tuple(f"development_{index}" for index in range(5)),
        role="development",
        source_offset=4,
    )
    locked = _artifact(
        tuple(f"locked_{index}" for index in range(5)),
        role="locked_test",
        source_offset=8,
    )
    arguments: dict[str, object] = {
        "ranks": (1,),
        "alphas": (1.0,),
        "permutation_seeds": (17,),
        "permutations_per_seed": 100,
        "minimum_support": 8,
        "device": "cpu",
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match="exact positive integer"):
        evaluate_morphology_ridge_gate(development, locked, **arguments)


def test_final_inference_requires_calibration_and_999_unique_permutations(
    calibration_receipt,
) -> None:
    development, locked = build_synthetic_calibration_pair(
        "spatial_autocorrelation",
        PRELIMINARY_ALTERNATIVE_CONDITION,
        0,
    )
    development = replace(
        development,
        cohort_id="HEST",
        cohort_release="synthetic-final-contract-fixture",
        opening_receipt_sha256="d" * 64,
    )
    locked = replace(
        locked,
        cohort_id="HEST",
        cohort_release="synthetic-final-contract-fixture",
        opening_receipt_sha256="d" * 64,
    )
    binding = calibration_receipt["exact_gate_settings"]["confirmatory_design_binding"]
    with pytest.raises(ValueError, match="requires a calibration receipt"):
        evaluate_morphology_ridge_gate(
            development,
            locked,
            ranks=(1,),
            alphas=(1.0,),
            permutations_per_seed=333,
            total_permutations=999,
            minimum_support=20,
            final_inference=True,
            confirmatory_design_binding=binding,
            device="cpu",
        )
    with pytest.raises(ValueError, match="at least 999 unique permutations"):
        evaluate_morphology_ridge_gate(
            development,
            locked,
            ranks=(1,),
            alphas=(1.0,),
            permutations_per_seed=333,
            total_permutations=998,
            minimum_support=20,
            final_inference=True,
            calibration_receipt=calibration_receipt,
            confirmatory_design_binding=binding,
            device="cpu",
        )
    with pytest.raises(ValueError, match="frozen contract|rank/ridge grid"):
        evaluate_morphology_ridge_gate(
            development,
            locked,
            ranks=(1,),
            alphas=(1.0,),
            permutations_per_seed=333,
            total_permutations=999,
            minimum_support=20,
            final_inference=True,
            calibration_receipt=calibration_receipt,
            confirmatory_design_binding=binding,
            confirmatory_analysis_plan_sha256=calibration_receipt["exact_gate_settings"][
                "confirmatory_analysis_plan_sha256"
            ],
            device="cpu",
        )


def test_synthetic_calibration_rejects_rows_outside_the_bound_topology() -> None:
    development, locked = build_synthetic_calibration_pair(
        "spatial_autocorrelation",
        PRELIMINARY_ALTERNATIVE_CONDITION,
        0,
    )
    section_ids = development.section_ids.copy()
    section_ids[0] = "unplanned_section"
    development = replace(development, section_ids=section_ids)
    trial_identity = {
        "scenario": "spatial_autocorrelation",
        "condition": PRELIMINARY_ALTERNATIVE_CONDITION,
        "trial_index": 0,
        "trial_seed": 0,
    }

    with pytest.raises(ValueError, match="artifacts differ from the confirmatory design binding"):
        evaluate_morphology_ridge_gate(
            development,
            locked,
            ranks=(1,),
            alphas=(1.0,),
            permutations_per_seed=333,
            total_permutations=999,
            minimum_support=20,
            final_inference=True,
            confirmatory_design_binding=synthetic_completed_confirmatory_design_binding(),
            synthetic_calibration_mode=True,
            calibration_trial_identity=trial_identity,
            calibration_run_contract_sha256="a" * 64,
            device="cpu",
        )


def test_direct_final_gate_requires_every_frozen_nuisance_family() -> None:
    development = _with_required_nuisance_features(
        _artifact(
            tuple(f"development_{index}" for index in range(5)),
            role="development",
            source_offset=4,
        )
    )
    locked = _with_required_nuisance_features(
        _artifact(
            tuple(f"locked_{index}" for index in range(5)),
            role="locked_test",
            source_offset=8,
        )
    )
    development = replace(
        development,
        cellvit_context_features=np.empty((len(development.observation_ids), 0)),
        cellvit_context_feature_names=(),
    )
    locked = replace(
        locked,
        cellvit_context_features=np.empty((len(locked.observation_ids), 0)),
        cellvit_context_feature_names=(),
    )

    with pytest.raises(ValueError, match="cellvit_context_only"):
        evaluate_morphology_ridge_gate(
            development,
            locked,
            final_inference=True,
            device="cpu",
        )


def test_synthetic_calibration_bypass_cannot_accept_biological_artifacts() -> None:
    development = _artifact(
        tuple(f"development_{index}" for index in range(5)),
        role="development",
        source_offset=4,
    )
    locked = _artifact(
        tuple(f"locked_{index}" for index in range(5)),
        role="locked_test",
        source_offset=8,
    )
    with pytest.raises(ValueError, match="only explicitly synthetic"):
        evaluate_morphology_ridge_gate(
            development,
            locked,
            final_inference=True,
            synthetic_calibration_mode=True,
            device="cpu",
        )
    with pytest.raises(ValueError, match="only explicitly synthetic"):
        evaluate_morphology_ridge_gate(
            replace(development, cohort_id="SYNTHETIC_CALIBRATION"),
            replace(locked, cohort_id="SYNTHETIC_CALIBRATION"),
            device="cpu",
        )


def test_g3_intrinsic_flags_require_prespecified_direct_crop_contrasts() -> None:
    def with_ladder(artifact: MorphologyRidgeDatasetArtifact):
        zeros = np.zeros_like(artifact.frozen_features)
        intrinsic_ids = {
            "nucleus_mask_only",
            "nucleus_mask_mean_fill_112um",
            "cell_mask_only",
            "cell_mask_mean_fill_112um",
        }
        crop_ids = tuple(HEST_CROP_CONTRACT)
        return replace(
            artifact,
            hypothesis_ids=("H-CELL", "H-INTRINSIC"),
            image_feature_tensor=np.stack(
                tuple(
                    artifact.frozen_features
                    if crop_id == "crop_112um" or crop_id in intrinsic_ids
                    else zeros
                    for crop_id in crop_ids
                ),
                axis=1,
            ),
            crop_ids=crop_ids,
            crop_roles=tuple(HEST_CROP_CONTRACT[value][0] for value in crop_ids),
            crop_comparison_families=tuple(HEST_CROP_CONTRACT[value][1] for value in crop_ids),
        )

    development = with_ladder(
        _artifact(
            tuple(f"development_{index}" for index in range(5)),
            role="development",
            source_offset=4,
        )
    )
    locked = with_ladder(
        _artifact(
            tuple(f"locked_{index}" for index in range(5)),
            role="locked_test",
            source_offset=8,
        )
    )
    report = evaluate_morphology_ridge_gate(
        development,
        locked,
        ranks=(1,),
        alphas=(1.0e-4,),
        total_permutations=100,
        minimum_support=8,
        prespecified_fixed_hyperparameters=True,
        device="cpu",
    )
    assert report["nucleus_hypothesis_tested"] is True
    assert report["cell_intrinsic_hypothesis_tested"] is True
    nucleus = report["hypothesis_decisions"]["G3_nucleus_intrinsic"]
    assert nucleus["tested"] is True
    assert nucleus["focal_family"] == "nucleus_mask_image"
    assert nucleus["strongest_comparator_family"] != "nucleus_mask_image"
    assert nucleus["pass"] is False  # exploratory runs never authorize G3
    assert report["morphology_source_conclusion"] == "not_tested"
    frozen = report["frozen_morphology_contrast_family"]
    assert frozen["multiplicity_method"] == G3_MULTIPLICITY_METHOD
    assert set(frozen["contrasts"]) == set(REQUIRED_G3_CONTRAST_PAIRS)
    assert all("familywise_adjusted_p" in row for row in frozen["contrasts"].values())
    quality = report["registration_quality_sensitivity"]
    assert quality["tested"] is True
    assert quality["row_counts"] == {
        "best": len(locked.observation_ids),
        "intermediate": 0,
        "near_threshold": 0,
    }
    assert set(quality["contrasts"]) == set(REQUIRED_G3_CONTRAST_PAIRS)
    assert all("strata" in row for row in quality["contrasts"].values())


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"intrinsic_prespecified": False}, "not_tested"),
        ({"final_inference": False}, "not_tested"),
        ({"familywise_tested": False}, "inconclusive"),
        ({"measurement_valid": False}, "inconclusive"),
        ({"any_source_contrast_positive": True}, "inconclusive"),
        ({"component_pass": False}, "no_morphology_specific_information"),
        ({}, "no_morphology_specific_information"),
        (
            {"component_pass": False, "nucleus_specific": True},
            "inconclusive",
        ),
        ({"nucleus_specific": True}, "nucleus_dominant"),
        ({"cell_specific": True}, "cell_dominant"),
        ({"context_specific": True}, "context_dominant"),
        (
            {
                "nucleus_specific": True,
                "context_specific": True,
                "full_vs_context": True,
                "full_vs_intrinsic": True,
            },
            "mixed_intrinsic_and_contextual_information",
        ),
        (
            {"nucleus_specific": True, "cell_specific": True},
            "multiple_sources_without_incremental_combination",
        ),
    ),
)
def test_morphology_source_conclusion_fails_closed(
    overrides: dict[str, bool], expected: str
) -> None:
    inputs = {
        "intrinsic_prespecified": True,
        "final_inference": True,
        "familywise_tested": True,
        "measurement_valid": True,
        "component_pass": True,
        "any_source_contrast_positive": False,
        "nucleus_specific": False,
        "cell_specific": False,
        "context_specific": False,
        "full_vs_context": False,
        "full_vs_intrinsic": False,
    }
    inputs.update(overrides)

    conclusion = gate_module._classify_morphology_source(**inputs)

    assert conclusion == expected
    if expected in {"not_tested", "inconclusive"}:
        assert conclusion != "no_morphology_specific_information"


def test_ridge_artifact_rejects_marker_leakage_and_donor_overlap() -> None:
    development = _artifact(
        tuple(f"development_{index}" for index in range(5)),
        role="development",
        source_offset=4,
    )
    locked = _artifact(
        tuple(f"locked_{index}" for index in range(5)),
        role="locked_test",
        source_offset=8,
    )
    development.validate()
    development.validate_compatible(locked)

    with pytest.raises(ValueError, match="marker genes leak"):
        replace(development, type_marker_gene_ids=("G1",)).validate()
    with pytest.raises(ValueError, match="donors overlap"):
        development.validate_compatible(
            replace(locked, donor_ids=np.repeat("development_1", len(locked.donor_ids)))
        )


def test_ridge_artifact_binds_standalone_reference_means_to_primary_split() -> None:
    artifact = _artifact(
        tuple(f"development_{index}" for index in range(5)),
        role="development",
        source_offset=4,
    )
    drifted_by_split = artifact.reference_means_by_split.copy()
    drifted_by_split[0, 0, 0] += 1.0

    with pytest.raises(ValueError, match="primary frozen reference split"):
        replace(artifact, reference_means_by_split=drifted_by_split).validate()


def test_primary_identity_rejects_historical_cross_modal_encoder() -> None:
    artifact = _artifact(
        tuple(f"development_{index}" for index in range(5)),
        role="development",
        source_offset=4,
    )
    with pytest.raises(ValueError, match="H-optimus-1"):
        validate_experiment_identity(
            replace(artifact, encoder_name="omiclip-loki-coca-vit-l-14"),
            "primary_hoptimus1",
        )
    validate_experiment_identity(
        replace(artifact, encoder_name="bioptimus/H-optimus-1"),
        "primary_hoptimus1",
    )
    regional = replace(
        artifact,
        encoder_name="bioptimus/H-optimus-1",
        crop_scale="full_context",
        cohort_id="HESCAPE",
        opening_receipt_sha256="",
        cohort_release="human-lung-healthy-panel",
        observation_level="pseudo_spot_55um",
        target_construction="sum_pooled_xenium_transcripts",
        evidence_scope="development_pilot",
        hypothesis_ids=("H-REGIONAL",),
        scientific_scope="development_only_regional_pilot",
    )
    validate_experiment_identity(regional, "regional_hescape_hoptimus1")
    with pytest.raises(ValueError, match="cell-level targets"):
        validate_experiment_identity(
            replace(regional, crop_scale="small_cell_centered"), "primary_hoptimus1"
        )


def _regional_uni2h(artifact: MorphologyRidgeDatasetArtifact) -> MorphologyRidgeDatasetArtifact:
    rows = len(artifact.observation_ids)
    index = np.arange(rows, dtype=np.float64)
    composition = np.column_stack(
        (
            np.sin(index * 0.31),
            np.cos(index * 0.43),
            np.sin(index * 0.59 + 0.2),
            np.cos(index * 0.71 - 0.3),
        )
    )
    return replace(
        artifact,
        encoder_name="MahmoodLab/UNI2-h",
        crop_scale="full_context",
        cohort_id="HESCAPE",
        opening_receipt_sha256="",
        cohort_release="human-lung-healthy-panel",
        observation_level="pseudo_spot_55um",
        target_construction="sum_pooled_xenium_transcripts",
        evidence_scope="development_pilot",
        hypothesis_ids=("H-REGIONAL",),
        scientific_scope="development_only_regional_pilot",
        stain_features=artifact.coordinate_features.copy(),
        stain_feature_names=(
            "rgb_mean_r",
            "rgb_mean_g",
        ),
        composition_features=composition,
        composition_feature_names=(
            "composition_epithelial",
            "composition_immune",
            "composition_stromal",
            "composition_endothelial",
        ),
        technical_covariates=np.log1p(100.0 + np.mod(index, 17.0))[:, None],
        technical_covariate_names=("log1p_library_size",),
    )


def test_hescape_regional_artifact_cannot_enter_locked_morphology_gate() -> None:
    development = _regional_uni2h(
        _artifact(
            tuple(f"development_{index}" for index in range(5)),
            role="development",
            source_offset=4,
        )
    )
    locked = _regional_uni2h(
        _artifact(
            tuple(f"locked_{index}" for index in range(4)),
            role="locked_test",
            source_offset=8,
        )
    )
    validate_experiment_identity(development, "regional_hescape_uni2h")
    with pytest.raises(ValueError, match="development-pilot"):
        evaluate_morphology_ridge_gate(
            development,
            locked,
            ranks=(1,),
            alphas=(1.0e-4,),
            permutation_seeds=(17, 29, 41),
            permutations_per_seed=100,
            minimum_support=8,
            prespecified_fixed_hyperparameters=True,
            device="cpu",
        )


def test_uni2h_identity_requires_named_composition_and_stain_controls() -> None:
    regional = _regional_uni2h(
        _artifact(
            tuple(f"development_{index}" for index in range(5)),
            role="development",
            source_offset=4,
        )
    )
    with pytest.raises(ValueError, match="frozen log-library covariate"):
        validate_experiment_identity(
            replace(
                regional,
                technical_covariates=np.empty((len(regional.observation_ids), 0)),
                technical_covariate_names=(),
            ),
            "regional_hescape_uni2h",
        )
    with pytest.raises(ValueError, match="four frozen RNA-only composition scores"):
        validate_experiment_identity(
            replace(
                regional,
                composition_features=np.empty((len(regional.observation_ids), 0)),
                composition_feature_names=(),
            ),
            "regional_hescape_uni2h",
        )
    with pytest.raises(ValueError, match="stain-statistics controls"):
        validate_experiment_identity(
            replace(
                regional,
                stain_features=np.empty((len(regional.observation_ids), 0)),
                stain_feature_names=(),
            ),
            "regional_hescape_uni2h",
        )
