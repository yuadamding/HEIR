from __future__ import annotations

from heir.evaluation import GATE_IDS, evaluate_authorizations


def _receipt(gate: str, passed: bool = True) -> dict[str, object]:
    hypotheses = {
        "G0": ["H-MEAS"],
        "G1": ["H-REGIONAL"],
        "G2": ["H-CELL"],
        "G3": ["H-INTRINSIC"],
        "G4": ["H-CELL"],
        "G5": ["H-EXT"],
        "G6": ["H-REF"],
        "G7": ["H-END2END"],
        "G8": ["H-COMP"],
        "G9": ["H-EXT", "H-COMP"],
    }
    value = {
        "gate_id": gate,
        "pass": passed,
        "hypothesis_ids": hypotheses[gate],
        "report_sha256": "0123456789abcdef"[int(gate[1:])] * 64,
    }
    if gate == "G3":
        value["intrinsic_conclusion"] = "nucleus"
    return value


def test_authorizations_follow_dependencies_not_generic_component_pass() -> None:
    receipts = {"G0": _receipt("G0"), "G2": _receipt("G2"), "G3": _receipt("G3")}
    report = evaluate_authorizations(receipts)
    assert report["authorizations"]["morphology_association"] is True
    assert report["authorizations"]["nucleus_intrinsic_claim"] is True
    assert report["authorizations"]["external_generalization"] is False
    assert report["authorizations"]["full_heir_claim"] is False


def test_full_claim_requires_every_gate_and_intrinsic_cell_evidence() -> None:
    receipts = {gate: _receipt(gate) for gate in GATE_IDS}
    report = evaluate_authorizations(receipts)
    assert report["authorizations"]["full_heir_claim"] is True

    receipts["G5"] = _receipt("G5", passed=False)
    report = evaluate_authorizations(receipts)
    assert report["authorizations"]["full_heir_claim"] is False
    assert report["authorizations"]["personalized_reference_claim"] is False
