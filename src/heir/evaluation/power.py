"""Fail-closed calibration receipt hook for morphology-gate inference."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Optional

CALIBRATION_RECEIPT_SCHEMA = "heir.morphology_gate_calibration.v1"
CALIBRATION_ENGINE = "heir.synthetic_complete_morphology_gate.v1"
REQUIRED_CALIBRATION_SCENARIOS = (
    "spatial_autocorrelation",
    "disease_imbalance",
    "section_effects",
    "missing_fine_types",
    "variable_transcript_reliability",
    "unbalanced_donor_cell_counts",
    "inactive_permutation_strata",
    "nuisance_selection",
    "target_panel_selection",
    "crop_family_multiplicity",
)
REQUIRED_COMPLETE_GATE_CHECKS = (
    "development_target_panel_reliable",
    "supported_strata",
    "permutation_strata_active",
    "image_macro_r2",
    "beats_strongest_nuisance",
    "exact_donor_signflip",
    "local_roi_permutation_null",
    "spatial_block_permutation_null",
    "both_permutation_nulls_active",
    "positive_donor_fraction",
    "not_single_donor_driven",
    "development_only_crop_selection",
    "development_only_target_selection",
)


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_calibration_receipt(
    receipt: Optional[Mapping[str, object]], *, required: bool
) -> Mapping[str, object]:
    """Validate pre-lock type-I-error and power evidence without running simulations."""

    if receipt is None:
        if required:
            raise ValueError("final morphology inference requires a calibration receipt")
        return {"available": False, "required": False}
    required_fields = {
        "schema",
        "pass",
        "engine",
        "simulation_sha256",
        "scenario_config",
        "scenario_config_sha256",
        "thresholds",
        "thresholds_sha256",
        "scenario_results",
        "complete_gate_check_ids",
        "maximum_complete_gate_false_pass_probability",
        "power_at_minimum_meaningful_effect",
        "locked_outcomes_used",
        "synthetic_data_only",
        "complete_gate_executed",
        "scenario_families",
        "calibrated",
        "receipt_content_sha256",
    }
    if not required_fields.issubset(receipt) or receipt["schema"] != CALIBRATION_RECEIPT_SCHEMA:
        raise ValueError("morphology gate calibration receipt is incomplete")
    for name in (
        "simulation_sha256",
        "scenario_config_sha256",
        "thresholds_sha256",
        "receipt_content_sha256",
    ):
        digest = str(receipt[name])
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("calibration receipt %s must be a lowercase SHA-256" % name)
    false_pass = float(receipt["maximum_complete_gate_false_pass_probability"])
    power = float(receipt["power_at_minimum_meaningful_effect"])
    if not math.isfinite(false_pass) or not math.isfinite(power):
        raise ValueError("calibration aggregate error and power must be finite")
    if not isinstance(receipt["scenario_families"], list):
        raise ValueError("calibration receipt scenario_families must be a list")
    scenarios = [str(value) for value in receipt["scenario_families"]]
    if len(scenarios) != len(set(scenarios)) or set(scenarios) != set(
        REQUIRED_CALIBRATION_SCENARIOS
    ):
        raise ValueError("calibration receipt lacks required empirical stress families")
    check_ids = receipt["complete_gate_check_ids"]
    if (
        not isinstance(check_ids, list)
        or len(check_ids) != len(set(str(value) for value in check_ids))
        or set(str(value) for value in check_ids) != set(REQUIRED_COMPLETE_GATE_CHECKS)
    ):
        raise ValueError("calibration receipt did not execute the complete gate contract")
    scenario_config = receipt["scenario_config"]
    thresholds = receipt["thresholds"]
    results = receipt["scenario_results"]
    if not all(isinstance(value, Mapping) for value in (scenario_config, thresholds, results)):
        raise ValueError("calibration receipt configuration or results are malformed")
    if receipt["scenario_config_sha256"] != _sha256(scenario_config):
        raise ValueError("calibration scenario configuration hash differs")
    if receipt["thresholds_sha256"] != _sha256(thresholds):
        raise ValueError("calibration threshold hash differs")
    if set(str(value) for value in results) != set(scenarios):
        raise ValueError("calibration results differ from configured scenario families")
    replicates = scenario_config.get("replicates_per_condition")
    permutations = scenario_config.get("permutations")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, (int, float))
        or int(replicates) != replicates
        or int(replicates) < 10
        or isinstance(permutations, bool)
        or not isinstance(permutations, (int, float))
        or int(permutations) != permutations
        or int(permutations) < 19
    ):
        raise ValueError("calibration receipt has too few empirical replicates")
    null_rates = []
    power_rates = []
    for scenario in scenarios:
        result = results.get(scenario)
        null = result.get("null") if isinstance(result, Mapping) else None
        effect = result.get("minimum_meaningful_effect") if isinstance(result, Mapping) else None
        for condition in (null, effect):
            if not isinstance(condition, Mapping):
                raise ValueError("calibration scenario lacks an empirical condition")
            trials = condition.get("trials")
            passes = condition.get("complete_gate_passes")
            fraction = condition.get("complete_gate_pass_fraction")
            permutation_nulls = condition.get("permutation_nulls")
            if (
                isinstance(trials, bool)
                or not isinstance(trials, (int, float))
                or int(trials) != trials
                or int(trials) != int(replicates)
                or isinstance(passes, bool)
                or not isinstance(passes, (int, float))
                or int(passes) != passes
                or not 0 <= int(passes) <= int(trials)
                or not isinstance(fraction, (int, float))
                or not math.isfinite(float(fraction))
                or abs(float(fraction) - int(passes) / int(trials)) > 1.0e-12
                or not isinstance(permutation_nulls, Mapping)
                or permutation_nulls.get("local_roi_permutations") != int(permutations)
                or permutation_nulls.get("spatial_block_permutations") != int(permutations)
            ):
                raise ValueError("calibration empirical pass counts are inconsistent")
        null_rates.append(float(null["complete_gate_pass_fraction"]))
        power_rates.append(float(effect["complete_gate_pass_fraction"]))
    recomputed_false_pass = max(null_rates)
    recomputed_power = min(power_rates)
    if abs(false_pass - recomputed_false_pass) > 1.0e-12 or abs(power - recomputed_power) > 1.0e-12:
        raise ValueError("calibration aggregate error or power differs from scenarios")
    simulation_core = {
        "engine": receipt["engine"],
        "scenario_config_sha256": receipt["scenario_config_sha256"],
        "thresholds_sha256": receipt["thresholds_sha256"],
        "scenario_results": results,
    }
    if receipt["simulation_sha256"] != _sha256(simulation_core):
        raise ValueError("calibration simulation hash differs")
    receipt_core = {
        str(name): value for name, value in receipt.items() if name != "receipt_content_sha256"
    }
    if receipt["receipt_content_sha256"] != _sha256(receipt_core):
        raise ValueError("calibration receipt content hash differs")
    maximum_allowed_false_pass = min(
        0.05,
        float(thresholds.get("maximum_complete_gate_false_pass_probability", -1.0)),
    )
    minimum_required_power = max(
        0.80,
        float(thresholds.get("minimum_power_at_minimum_meaningful_effect", 2.0)),
    )
    if (
        receipt.get("pass") is not True
        or receipt.get("calibrated") is not True
        or receipt.get("engine") != CALIBRATION_ENGINE
        or false_pass > maximum_allowed_false_pass
        or power < minimum_required_power
        or receipt["locked_outcomes_used"] is not False
        or receipt["synthetic_data_only"] is not True
        or receipt["complete_gate_executed"] is not True
    ):
        raise ValueError("calibration receipt does not satisfy frozen error and power requirements")
    return {
        "available": True,
        "required": required,
        "schema": CALIBRATION_RECEIPT_SCHEMA,
        "simulation_sha256": str(receipt["simulation_sha256"]),
        "scenario_config_sha256": str(receipt["scenario_config_sha256"]),
        "thresholds_sha256": str(receipt["thresholds_sha256"]),
        "receipt_content_sha256": str(receipt["receipt_content_sha256"]),
        "maximum_complete_gate_false_pass_probability": false_pass,
        "power_at_minimum_meaningful_effect": power,
        "locked_outcomes_used": False,
        "synthetic_data_only": True,
        "complete_gate_executed": True,
        "scenario_families": sorted(scenarios),
    }


__all__ = [
    "CALIBRATION_ENGINE",
    "CALIBRATION_RECEIPT_SCHEMA",
    "REQUIRED_CALIBRATION_SCENARIOS",
    "REQUIRED_COMPLETE_GATE_CHECKS",
    "validate_calibration_receipt",
]
