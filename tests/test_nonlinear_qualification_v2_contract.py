from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from heir.evaluation import nonlinear_qualification_v2_contract as contract

ROOT = Path(__file__).parents[1]
PROTOCOL = json.loads(
    (ROOT / "configs/hest_nonlinear_qualification_v2.json").read_text(encoding="utf-8")
)
MANIFEST_PATH = ROOT / "manifests/studies/hest_nonlinear_qualification_v2.retrospective.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
RESULT_PATH = ROOT / "manifests/studies/hest_nonlinear_qualification_v2.result.json"
RESULT = json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def _passing_d1() -> dict[str, float]:
    return {
        name: threshold + (0.01 if operator in {">=", ">"} else -0.001)
        for name, (operator, threshold) in contract.D1_THRESHOLDS.items()
    }


def _passing_d2() -> dict[str, float]:
    return {
        name: threshold + (0.01 if operator in {">=", ">"} else -0.001)
        for name, (operator, threshold) in contract.D2_THRESHOLDS.items()
    }


def test_registered_v2_protocol_validates_as_delta_over_v1() -> None:
    value = contract.validate_protocol(PROTOCOL, repository_root=ROOT)
    assert (
        contract.file_sha256(ROOT / "configs/hest_nonlinear_qualification_v2.json")
        == contract.V2_PROTOCOL_SHA256
    )
    assert value["registration_band_selection"]["selected_quality_band_id"] == "Q3"
    assert value["molecular_outcomes_read_during_v2_registration"] is False
    assert value["authorizes_h_cell"] is False


def test_registered_v2_manifest_binds_exact_protocol_and_outcome_blind_inputs() -> None:
    value = contract.validate_manifest(
        MANIFEST,
        PROTOCOL,
        repository_root=ROOT,
        protocol_file_sha256=contract.V2_PROTOCOL_SHA256,
    )
    assert value["molecular_outcomes_opened_under_v2"] is False
    assert value["execution"]["full_biological_execution_authorized_at_registration"] is False


def test_v2_result_preserves_registration_and_fails_closed() -> None:
    assert RESULT["protocol_binding"]["sha256"] == contract.V2_PROTOCOL_SHA256
    assert RESULT["registration_manifest_binding"]["sha256"] == contract.file_sha256(
        MANIFEST_PATH
    )
    assert RESULT["source_access"]["molecular_outcomes_opened_under_v2"] is True
    assert RESULT["failure"]["completed_observed_outer_folds"] == 0
    assert RESULT["decision"]["stop_nonlinear_expansion_on_this_representation"] is True
    assert RESULT["decision"]["model_grid_or_threshold_changed_after_outcome_open"] is False
    assert RESULT["authorizes_h_cell"] is False
    assert RESULT["authorizes_h_intrinsic"] is False
    assert RESULT["authorizes_h_ref"] is False
    assert RESULT["authorizes_full_heir"] is False


@pytest.mark.parametrize(
    "tamper",
    ["blank", "topology", "band", "support", "primary_support", "authorization"],
)
def test_protocol_identity_and_fail_closed_rules_reject_tampering(tamper: str) -> None:
    value = copy.deepcopy(PROTOCOL)
    if tamper == "blank":
        value["outcome_blind_inputs"]["blank_supplement"]["feature_sha256"] = "0" * 64
    elif tamper == "topology":
        value["outcome_blind_inputs"]["registration_topology"]["selection_status"] = "failed"
    elif tamper == "band":
        value["registration_band_selection"]["selected_quality_band_id"] = "Q2"
    elif tamper == "support":
        value["registration_sensitivity"]["minimum_donor_type_support"] = 4
    elif tamper == "primary_support":
        value["registration_sensitivity"]["primary_support_20_R2_threshold_changed"] = True
    else:
        value["authorizes_h_cell"] = True
    with pytest.raises(ValueError):
        contract.validate_protocol(value, repository_root=ROOT)


def test_manifest_rejects_premature_execution_or_artifact_drift() -> None:
    value = copy.deepcopy(MANIFEST)
    value["execution"]["full_biological_execution_authorized_at_registration"] = True
    with pytest.raises(ValueError):
        contract.validate_manifest(
            value,
            PROTOCOL,
            repository_root=ROOT,
            protocol_file_sha256=contract.V2_PROTOCOL_SHA256,
        )


def test_D1_and_D2_are_separate_and_permanently_nonauthorizing() -> None:
    d2 = {"N3": _passing_d2(), "N4": _passing_d2()}
    result = contract.evaluate_decisions(_passing_d1(), d2)
    assert result["supports_prospective_local_context_estimator_design"] is True
    assert result["supports_intrinsic_localization_followup"] is True
    assert result["authorizes_h_intrinsic"] is False

    failed_d1 = _passing_d1()
    failed_d1["A_star_minus_B2_R2"] = 0.0
    result = contract.evaluate_decisions(failed_d1, d2)
    assert result["supports_prospective_local_context_estimator_design"] is False
    assert result["supports_intrinsic_localization_followup"] is False

    unavailable = _passing_d1()
    unavailable["registration_quality_one_sided_lower_bound"] = None
    result = contract.evaluate_decisions(unavailable, d2)
    assert result["D1"]["passed"] is False


def test_exact_sign_flip_uses_one_synchronous_sign_per_donor_across_family() -> None:
    effects = {
        name: {f"D{index}": 0.1 for index in range(6)}
        for name in ("A_star_vs_B2", "A_star_vs_N0", "A_star_vs_C_blank")
    }
    result = contract.exact_sign_flip_max_test(effects)
    assert result["sign_patterns"] == 64
    assert result["familywise_adjusted_p"] == {
        "A_star_vs_B2": 1 / 64,
        "A_star_vs_N0": 1 / 64,
        "A_star_vs_C_blank": 1 / 64,
    }


def test_synthetic_calibration_uses_simultaneous_exact_bounds() -> None:
    counts = {
        condition: {decision: 190 if expected else 0 for decision, expected in decisions.items()}
        for condition, decisions in contract.CALIBRATION_EXPECTATIONS.items()
    }
    result = contract.evaluate_synthetic_calibration(counts)
    assert result["calibration_passed"] is True
    assert result["HEST_molecular_execution_authorized"] is True
    assert result["simultaneous_family_assertions"] == 20

    counts["global_null"]["D1"] = 20
    result = contract.evaluate_synthetic_calibration(counts)
    assert result["calibration_passed"] is False
    assert result["HEST_molecular_execution_authorized"] is False
