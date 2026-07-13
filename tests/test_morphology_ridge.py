from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from heir.data import MorphologyRidgeDatasetArtifact
from heir.evaluation import (
    evaluate_morphology_ridge_gate,
    fit_oracle_ridge_probe,
    predict_oracle_ridge,
    validate_experiment_identity,
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
            state = np.linspace(-1.0, 1.0, 12)
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
            block_ids.extend([f"{donor}_evaluation"] * 12)
            roi_ids.extend([f"{donor}_type_{type_index}"] * 12)
            observation_ids.extend([f"{donor}_{type_index}_{index}" for index in range(12)])
    cells = len(labels)
    return MorphologyRidgeDatasetArtifact(
        observation_ids=np.asarray(observation_ids),
        donor_ids=np.asarray(donor_ids),
        block_ids=np.asarray(block_ids),
        roi_ids=np.asarray(roi_ids),
        type_labels=np.asarray(labels, dtype=np.int64),
        type_names=("epithelial", "immune"),
        frozen_features=np.concatenate(features),
        molecular_targets=np.concatenate(targets),
        reference_means=np.concatenate(means),
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
        cohort_id="HESCAPE",
        cohort_release="human-lung-healthy-panel",
        assay="Xenium",
        observation_level="cell",
        target_construction="registered_cell_expression",
        reference_pool_independent=True,
        labels_independent_of_images=True,
        registration_is_one_to_one=True,
        role=role,
    )


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
    report = evaluate_morphology_ridge_gate(
        development,
        locked,
        ranks=(1,),
        alphas=(1.0e-4,),
        permutation_seeds=(17, 29, 41),
        permutations_per_seed=100,
        minimum_support=8,
        device="cpu",
    )
    assert report["component_pass"] is True
    assert report["authorizes_full_heir"] is False
    assert report["oracle_type_only"] is True
    control = report["permutation_control"]
    assert control["training_probe_refit_for_each_permutation"] is True
    assert control["total_permutations"] == 300
    assert all(row["empirical_p"] < 0.01 for row in control["seeds"])
    assert report["primary_metrics"]["donor_equal_type_equal_residual_coordinate_r2"] > 0.95
    assert report["checks"]["beats_coordinate_only"] is True

    repeated = evaluate_morphology_ridge_gate(
        development,
        locked,
        ranks=(1,),
        alphas=(1.0e-4,),
        permutation_seeds=(17, 29, 41),
        permutations_per_seed=100,
        minimum_support=8,
        device="cpu",
    )
    assert repeated["permutation_control"] == control


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
    with pytest.raises(ValueError, match="at least 100 permutations"):
        evaluate_morphology_ridge_gate(
            development,
            locked,
            ranks=(1,),
            alphas=(1.0,),
            permutation_seeds=(17, 29, 41),
            permutations_per_seed=99,
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
        device="cpu",
    )
    assert report["checks"]["beats_coordinate_only"] is False
    assert report["component_pass"] is False


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
        observation_level="pseudo_spot_55um",
        target_construction="sum_pooled_xenium_transcripts",
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
        observation_level="pseudo_spot_55um",
        target_construction="sum_pooled_xenium_transcripts",
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


def test_uni2h_regional_gate_reports_raw_and_composition_adjusted_endpoints() -> None:
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
    report = evaluate_morphology_ridge_gate(
        development,
        locked,
        ranks=(1,),
        alphas=(1.0e-4,),
        permutation_seeds=(17, 29, 41),
        permutations_per_seed=100,
        minimum_support=8,
        device="cpu",
    )
    assert report["component_pass"] is True
    assert report["nucleus_hypothesis_tested"] is False
    assert report["regional_hypothesis_tested"] is True
    assert report["scientific_scope"] == "regional_pseudospot_exploratory"
    endpoints = report["regional_endpoints"]
    assert endpoints["correction_coefficients_fit_on_development_only"] is True
    assert endpoints["composition_adjusted"] is not None
    assert (
        endpoints["composition_adjusted"][
            "donor_equal_niche_equal_residual_coordinate_r2"
        ]
        > 0.0
    )
    assert report["checks"]["composition_adjusted_positive"] is True
    assert report["checks"]["beats_coordinate_only"] is True
    assert report["checks"]["beats_stain_statistics_only"] is True


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
