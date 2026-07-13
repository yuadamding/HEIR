from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from heir.evaluation.control_models import feature_family_registry
from heir.evaluation.hierarchical_metrics import (
    donor_dominance,
    exact_paired_randomization,
)
from heir.evaluation.power import validate_calibration_receipt
from heir.evaluation.residual_targets import correct_residuals, fit_type_technical_effects
from heir.evaluation.weighted_basis import donor_weights, weighted_standardization


def test_donor_balanced_scaling_equalizes_unequal_row_counts() -> None:
    donors = np.asarray(["large"] * 99 + ["small"])
    values = np.concatenate((np.zeros((99, 1)), np.asarray([[10.0]])), axis=0)
    weights = donor_weights(donors)
    assert weights[donors == "large"].sum() == pytest.approx(
        weights[donors == "small"].sum()
    )
    mean, scale = weighted_standardization(values, weights)
    np.testing.assert_allclose(mean, [5.0])
    np.testing.assert_allclose(scale, [5.0])


def test_nuisance_correction_is_fit_separately_within_fine_type() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    donors = np.asarray(["D1", "D1", "D2", "D2"] * 2)
    covariates = np.asarray([[0.0], [1.0], [0.0], [1.0]] * 2)
    centered = covariates[:, 0] - 0.5
    residual = np.where(labels == 0, 2.0 * centered, -3.0 * centered)[:, None]
    means, coefficients = fit_type_technical_effects(
        covariates, residual, donors, labels, num_types=2
    )
    np.testing.assert_allclose(means, [[0.5], [0.5]])
    np.testing.assert_allclose(coefficients[:, 0, 0], [2.0, -3.0])
    corrected = correct_residuals(
        residual,
        np.zeros_like(residual),
        covariates,
        labels,
        means,
        coefficients,
    )
    np.testing.assert_allclose(corrected, 0.0, atol=1.0e-12)


def test_exact_donor_randomization_and_dominance_are_deterministic() -> None:
    effects = {"D1": 0.1, "D2": 0.2, "D3": 0.3}
    inference = exact_paired_randomization(effects)
    assert inference["enumerations"] == 8
    assert inference["one_sided_p"] == pytest.approx(1.0 / 8.0)
    dominance = donor_dominance(effects)
    assert dominance["largest_positive_donor"] == "D3"
    assert dominance["largest_positive_share"] == pytest.approx(0.5)


def test_calibration_receipt_fails_closed_for_final_inference() -> None:
    with pytest.raises(ValueError, match="requires a calibration receipt"):
        validate_calibration_receipt(None, required=True)
    receipt = {
        "schema": "heir.morphology_gate_calibration.v1",
        "simulation_sha256": "1" * 64,
        "thresholds_sha256": "2" * 64,
        "maximum_complete_gate_false_pass_probability": 0.05,
        "power_at_minimum_meaningful_effect": 0.80,
        "locked_outcomes_used": False,
        "complete_gate_executed": True,
        "scenario_families": [
            "no_image_effect",
            "weak_image_effect",
            "minimum_meaningful_effect",
            "larger_image_effect",
            "donor_heterogeneity",
            "section_heterogeneity",
            "spatial_autocorrelation",
            "missing_donor_type_strata",
            "measurement_reliability",
        ],
    }
    validated = validate_calibration_receipt(receipt, required=True)
    assert validated["available"] is True
    with pytest.raises(ValueError, match="error and power requirements"):
        validate_calibration_receipt(
            {**receipt, "power_at_minimum_meaningful_effect": 0.79}, required=True
        )


def test_control_registry_exposes_morphometric_context_and_combined_families() -> None:
    rows = 4
    artifact = SimpleNamespace(
        observation_ids=np.asarray(["a", "b", "c", "d"]),
        technical_covariates=np.ones((rows, 1)),
        coordinate_features=np.ones((rows, 2)),
        stain_features=np.ones((rows, 3)),
        frozen_features=np.ones((rows, 5)),
        crop_scale="full_context",
        nuclear_morphometrics=np.ones((rows, 4)),
        cell_morphometrics=np.ones((rows, 6)),
        context_features=np.ones((rows, 7)),
        nucleus_mask_features=np.ones((rows, 8)),
        cell_mask_features=np.ones((rows, 9)),
    )
    registry = feature_family_registry(artifact)
    assert registry["nuclear_morphometrics_only"].shape == (rows, 4)
    assert registry["cell_morphometrics_only"].shape == (rows, 6)
    assert registry["context_only"].shape == (rows, 7)
    assert registry["nucleus_mask_image"].shape == (rows, 8)
    assert registry["cell_mask_image"].shape == (rows, 9)
    assert registry["full_context_image"].shape == (rows, 5)
    assert registry["image_plus_morphometrics"].shape == (rows, 15)
