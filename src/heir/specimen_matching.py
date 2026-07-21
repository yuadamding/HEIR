"""Fail-closed validation for query-specific specimen/reference assignments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

PRIMARY_MATCH_TIERS = frozenset(
    {
        "T0_exact_same_block_independent_aliquot",
        "T0_adjacent_curl_same_parent_specimen",
    }
)
REFERENCE_MODALITIES = frozenset({"scRNA-seq", "snRNA-seq", "scFFPE-seq"})

SPECIMEN_ASSIGNMENT_V1_TSV_FIELDS = (
    "assignment_id",
    "query_id",
    "query_donor_id",
    "query_procedure_id",
    "query_region",
    "query_specimen_id",
    "query_block_id",
    "query_he_section_id",
    "query_st_section_id",
    "query_capture_id",
    "reference_sample_id",
    "reference_donor_id",
    "reference_specimen_id",
    "reference_block_id",
    "reference_aliquot_or_curl_id",
    "reference_modality",
    "claimed_match_tier",
    "verified_match_tier",
    "primary_eligible",
    "primary_selected",
    "metadata_evidence_path",
    "metadata_evidence_sha256",
    "metadata_row_key",
    "payload_identity_verified",
    "registration_verified",
    "verification_status",
    "target_access_state",
    "selection_tiebreak_sha256",
    "same_donor_wrong_block_reference_candidates",
    "exclusion_reason",
)

_TSV_BOOLEAN_FIELDS = frozenset(
    {
        "primary_eligible",
        "primary_selected",
        "payload_identity_verified",
        "registration_verified",
    }
)
_UNRESOLVED = frozenset({"", "na", "n/a", "none", "not_available", "tbd", "unknown", "unresolved"})
_UNRESOLVED_PREFIXES = ("not_resolved_", "pending_", "unresolved_")


def _is_unresolved(value: str) -> bool:
    normalized = value.strip().lower().replace(" ", "_")
    return normalized in _UNRESOLVED or normalized.startswith(_UNRESOLVED_PREFIXES)


def _text(assignment: Mapping[str, object], field: str, errors: list[str]) -> str:
    raw_value = assignment.get(field)
    if not isinstance(raw_value, str):
        errors.append(f"{field} must be a string")
        return ""
    value = raw_value.strip()
    if _is_unresolved(value):
        errors.append(f"{field} is unresolved")
    return value


def load_specimen_assignment_tsv(path: str | Path) -> list[dict[str, object]]:
    """Load the frozen v1 candidate manifest with strict header and bool parsing."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != SPECIMEN_ASSIGNMENT_V1_TSV_FIELDS:
            raise ValueError("unexpected specimen-assignment v1 TSV header")
        rows: list[dict[str, object]] = []
        for line_number, raw_row in enumerate(reader, start=2):
            row: dict[str, object] = dict(raw_row)
            for field in _TSV_BOOLEAN_FIELDS:
                value = str(row[field]).strip().lower()
                if value not in {"true", "false"}:
                    raise ValueError(f"line {line_number}: {field} must be true or false")
                row[field] = value == "true"
            rows.append(row)
    return rows


def primary_specimen_match_errors(assignment: Mapping[str, object]) -> tuple[str, ...]:
    """Return every reason a query/reference assignment cannot enter primary M3."""

    errors: list[str] = []
    _text(assignment, "assignment_id", errors)
    _text(assignment, "query_id", errors)
    tier = _text(assignment, "verified_match_tier", errors)
    query_donor = _text(assignment, "query_donor_id", errors)
    reference_donor = _text(assignment, "reference_donor_id", errors)
    query_specimen = _text(assignment, "query_specimen_id", errors)
    reference_specimen = _text(assignment, "reference_specimen_id", errors)
    query_block = _text(assignment, "query_block_id", errors)
    reference_block = _text(assignment, "reference_block_id", errors)
    query_he = _text(assignment, "query_he_section_id", errors)
    query_st = _text(assignment, "query_st_section_id", errors)
    _text(assignment, "reference_sample_id", errors)
    reference_material = _text(assignment, "reference_aliquot_or_curl_id", errors)
    reference_modality = _text(assignment, "reference_modality", errors)
    _text(assignment, "reference_assay_chemistry", errors)

    if tier not in PRIMARY_MATCH_TIERS:
        errors.append("verified_match_tier is not primary T0")
    if query_donor != reference_donor:
        errors.append("query and reference donor IDs differ")
    if query_specimen != reference_specimen:
        errors.append("query and reference canonical specimen IDs differ")
    if query_block != reference_block:
        errors.append("query and reference block IDs differ")
    if reference_material in {query_he, query_st}:
        errors.append("reference material overlaps a registered query section")
    if reference_modality not in REFERENCE_MODALITIES:
        errors.append("reference_modality is not an accepted sc/snRNA assay")

    required_true = (
        "payload_identity_verified",
        "registration_verified",
        "chain_of_custody_verified",
        "canonical_parent_specimen_id_verified",
        "reference_modality_verified",
        "reference_assay_chemistry_verified",
        "independent_reference_material",
    )
    for field in required_true:
        if assignment.get(field) is not True:
            errors.append(f"{field} must be true")

    required_false = (
        "contains_registered_query_material",
        "selection_uses_query_truth",
        "paper_level_evidence_only",
    )
    for field in required_false:
        if assignment.get(field) is not False:
            errors.append(f"{field} must be false")

    if (
        tier == "T0_adjacent_curl_same_parent_specimen"
        and assignment.get("adjacency_verified") is not True
    ):
        errors.append("adjacency_verified must be true for the adjacent-curl tier")

    return tuple(dict.fromkeys(errors))


def validate_primary_specimen_match(assignment: Mapping[str, object]) -> None:
    """Raise when an assignment is not a verified exact-specimen primary match."""

    errors = primary_specimen_match_errors(assignment)
    if errors:
        raise ValueError("invalid primary specimen match: " + "; ".join(errors))
