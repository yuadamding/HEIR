"""Dependency-derived scientific authorizations for the HEIR evidence bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence, Union

from heir.utils import sha256_file

PathLike = Union[str, Path]

GATE_IDS = tuple("G%d" % index for index in range(10))
HYPOTHESIS_IDS = {
    "H-MEAS",
    "H-REGIONAL",
    "H-CELL",
    "H-INTRINSIC",
    "H-REF",
    "H-END2END",
    "H-COMP",
    "H-EXT",
}


def _sha256(value: object, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("authorization %s must be a lowercase SHA-256" % name)
    return digest


def load_gate_receipts(paths: Sequence[PathLike]) -> Mapping[str, Mapping[str, object]]:
    """Load unique gate receipts and bind each to its exact report bytes."""

    receipts = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("authorization gate receipt is invalid JSON") from error
        if not isinstance(receipt, Mapping):
            raise ValueError("authorization gate receipt must be an object")
        gate = str(receipt.get("gate_id", ""))
        if gate not in GATE_IDS or gate in receipts:
            raise ValueError("authorization gate receipt identity is invalid or duplicated")
        declared = _sha256(receipt.get("report_sha256", ""), "report_sha256")
        report_path = Path(str(receipt.get("report_path", ""))).expanduser().resolve()
        if not report_path.is_file() or sha256_file(report_path) != declared:
            raise ValueError("authorization gate report differs from its receipt")
        receipts[gate] = dict(receipt)
    return receipts


def evaluate_authorizations(
    receipts: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    """Calculate named claims from the fixed gate dependency graph."""

    unknown = set(receipts) - set(GATE_IDS)
    if unknown:
        raise ValueError("authorization contains unknown gates: %s" % ", ".join(sorted(unknown)))
    gates = {gate: False for gate in GATE_IDS}
    evidence = {}
    intrinsic = "none"
    for gate, receipt in receipts.items():
        if not isinstance(receipt, Mapping) or receipt.get("gate_id") != gate:
            raise ValueError("authorization receipt gate identity differs")
        if not isinstance(receipt.get("pass"), bool):
            raise ValueError("authorization receipt pass must be boolean")
        hypotheses = receipt.get("hypothesis_ids")
        if (
            not isinstance(hypotheses, list)
            or not hypotheses
            or any(value not in HYPOTHESIS_IDS for value in hypotheses)
        ):
            raise ValueError("authorization receipt hypothesis IDs are invalid")
        report_sha = _sha256(receipt.get("report_sha256", ""), "report_sha256")
        gates[gate] = bool(receipt["pass"])
        evidence[gate] = {
            "pass": gates[gate],
            "hypothesis_ids": list(hypotheses),
            "report_sha256": report_sha,
        }
        if gate == "G3" and gates[gate]:
            intrinsic = str(receipt.get("intrinsic_conclusion", ""))
            if intrinsic not in {"nucleus", "cell", "context"}:
                raise ValueError("passing G3 needs a precise intrinsic conclusion")

    morphology = gates["G0"] and gates["G2"]
    external = morphology and gates["G5"]
    authorizations = {
        "run_morphology_experiments": gates["G0"],
        "regional_association": gates["G0"] and gates["G1"],
        "morphology_association": morphology,
        "nucleus_intrinsic_claim": morphology and gates["G3"] and intrinsic == "nucleus",
        "cell_intrinsic_claim": morphology
        and gates["G3"]
        and intrinsic in {"nucleus", "cell"},
        "local_context_claim": morphology and gates["G3"],
        "encoder_robust_association": morphology and gates["G4"],
        "external_generalization": external,
        "personalized_reference_claim": external and gates["G6"],
        "oracle_free_claim": external and gates["G7"],
        "heir_component_claim": external and gates["G6"] and gates["G7"] and gates["G8"],
        "full_heir_claim": all(gates.values()) and intrinsic in {"nucleus", "cell"},
    }
    return {
        "schema": "heir.authorization_report.v1",
        "gates": gates,
        "evidence": evidence,
        "intrinsic_conclusion": intrinsic,
        "authorizations": authorizations,
        "failed_or_missing_gates": [gate for gate in GATE_IDS if not gates[gate]],
    }


__all__ = ["GATE_IDS", "HYPOTHESIS_IDS", "evaluate_authorizations", "load_gate_receipts"]
