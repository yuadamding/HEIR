from __future__ import annotations

from typing import Mapping

import pytest

from heir.evaluation.morphology_calibration import (
    REQUIRED_SCENARIO_FAMILIES,
    calibrate_morphology_gate,
)


@pytest.fixture(scope="session")
def calibration_scenario_config() -> Mapping[str, object]:
    return {
        "seed": 1701,
        "replicates_per_condition": 10,
        "development_donors": 4,
        "evaluation_donors": 6,
        "fine_types": 2,
        "cells_per_donor_type": 8,
        "target_genes": 4,
        "crop_families": 2,
        "permutations": 19,
        "minimum_effect_loading": 1.2,
        "scenario_families": list(REQUIRED_SCENARIO_FAMILIES),
    }


@pytest.fixture(scope="session")
def calibration_thresholds() -> Mapping[str, object]:
    return {
        "maximum_complete_gate_false_pass_probability": 0.05,
        "minimum_power_at_minimum_meaningful_effect": 0.80,
        "minimum_macro_r2": 0.02,
        "minimum_image_minus_nuisance_r2": 0.01,
        "maximum_exact_signflip_p": 0.05,
        "minimum_positive_donor_fraction": 0.66,
        "maximum_largest_positive_donor_share": 0.60,
        "minimum_supported_strata_fraction": 0.70,
        "minimum_active_permutation_strata_fraction": 0.70,
        "minimum_gene_reliability": 0.50,
        "minimum_reliable_gene_fraction": 0.50,
        "minimum_reliable_donor_fraction": 0.70,
        "ridge_alpha": 1.0,
    }


@pytest.fixture(scope="session")
def calibration_receipt(
    calibration_scenario_config: Mapping[str, object],
    calibration_thresholds: Mapping[str, object],
) -> Mapping[str, object]:
    return calibrate_morphology_gate(
        calibration_scenario_config, calibration_thresholds
    )
