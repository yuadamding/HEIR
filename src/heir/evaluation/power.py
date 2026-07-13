"""Fail-closed calibration receipt hook for morphology-gate inference."""

from __future__ import annotations

from typing import Mapping, Optional

CALIBRATION_RECEIPT_SCHEMA = "heir.morphology_gate_calibration.v1"


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
        "simulation_sha256",
        "thresholds_sha256",
        "maximum_complete_gate_false_pass_probability",
        "power_at_minimum_meaningful_effect",
        "locked_outcomes_used",
        "complete_gate_executed",
        "scenario_families",
    }
    if not required_fields.issubset(receipt) or receipt["schema"] != CALIBRATION_RECEIPT_SCHEMA:
        raise ValueError("morphology gate calibration receipt is incomplete")
    for name in ("simulation_sha256", "thresholds_sha256"):
        digest = str(receipt[name])
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("calibration receipt %s must be a lowercase SHA-256" % name)
    false_pass = float(receipt["maximum_complete_gate_false_pass_probability"])
    power = float(receipt["power_at_minimum_meaningful_effect"])
    required_scenarios = {
        "no_image_effect",
        "weak_image_effect",
        "minimum_meaningful_effect",
        "larger_image_effect",
        "donor_heterogeneity",
        "section_heterogeneity",
        "spatial_autocorrelation",
        "missing_donor_type_strata",
        "measurement_reliability",
    }
    if not isinstance(receipt["scenario_families"], list):
        raise ValueError("calibration receipt scenario_families must be a list")
    scenarios = {str(value) for value in receipt["scenario_families"]}
    if (
        false_pass > 0.05
        or power < 0.80
        or receipt["locked_outcomes_used"] is not False
        or receipt["complete_gate_executed"] is not True
        or not required_scenarios <= scenarios
    ):
        raise ValueError("calibration receipt does not satisfy frozen error and power requirements")
    return {
        "available": True,
        "required": required,
        "schema": CALIBRATION_RECEIPT_SCHEMA,
        "simulation_sha256": str(receipt["simulation_sha256"]),
        "thresholds_sha256": str(receipt["thresholds_sha256"]),
        "maximum_complete_gate_false_pass_probability": false_pass,
        "power_at_minimum_meaningful_effect": power,
        "locked_outcomes_used": False,
        "complete_gate_executed": True,
        "scenario_families": sorted(scenarios),
    }


__all__ = [
    "CALIBRATION_RECEIPT_SCHEMA",
    "validate_calibration_receipt",
]
