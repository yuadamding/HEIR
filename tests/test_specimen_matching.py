from __future__ import annotations

from pathlib import Path

import pytest

from heir.specimen_matching import (
    SPECIMEN_ASSIGNMENT_V1_TSV_FIELDS,
    load_specimen_assignment_tsv,
    primary_specimen_match_errors,
    validate_primary_specimen_match,
)


def _exact_assignment(**overrides: object) -> dict[str, object]:
    assignment: dict[str, object] = {
        "assignment_id": "A001",
        "query_id": "Q001",
        "verified_match_tier": "T0_exact_same_block_independent_aliquot",
        "query_donor_id": "D01",
        "reference_donor_id": "D01",
        "query_specimen_id": "D01_BlockA",
        "reference_specimen_id": "D01_BlockA",
        "query_block_id": "BlockA",
        "reference_block_id": "BlockA",
        "query_he_section_id": "BlockA_HE_01",
        "query_st_section_id": "BlockA_ST_02",
        "reference_sample_id": "R001",
        "reference_aliquot_or_curl_id": "BlockA_snRNA_aliquot_03",
        "reference_modality": "snRNA-seq",
        "payload_identity_verified": True,
        "registration_verified": True,
        "chain_of_custody_verified": True,
        "canonical_parent_specimen_id_verified": True,
        "reference_modality_verified": True,
        "reference_assay_chemistry": "10x_Chromium_3prime_v3",
        "reference_assay_chemistry_verified": True,
        "independent_reference_material": True,
        "contains_registered_query_material": False,
        "selection_uses_query_truth": False,
        "paper_level_evidence_only": False,
    }
    assignment.update(overrides)
    return assignment


def test_exact_same_block_independent_aliquot_is_accepted() -> None:
    assignment = _exact_assignment()

    assert primary_specimen_match_errors(assignment) == ()
    validate_primary_specimen_match(assignment)


def test_adjacent_curl_from_same_parent_specimen_is_accepted() -> None:
    assignment = _exact_assignment(
        verified_match_tier="T0_adjacent_curl_same_parent_specimen",
        reference_aliquot_or_curl_id="BlockA_adjacent_curl_03",
        adjacency_verified=True,
    )

    validate_primary_specimen_match(assignment)


def test_same_donor_different_block_is_rejected_from_primary() -> None:
    assignment = _exact_assignment(
        verified_match_tier="T2_same_donor_different_block",
        reference_specimen_id="D01_BlockB",
        reference_block_id="BlockB",
        reference_aliquot_or_curl_id="BlockB_snRNA_aliquot_01",
    )

    with pytest.raises(ValueError, match="not primary T0"):
        validate_primary_specimen_match(assignment)


def test_same_lesion_separate_core_is_rejected_from_primary() -> None:
    assignment = _exact_assignment(
        verified_match_tier="T1_same_lesion_procedure_separate_core",
        reference_specimen_id="D01_CoreB",
        reference_block_id="CoreB",
        reference_aliquot_or_curl_id="CoreB_snRNA_aliquot_01",
    )

    with pytest.raises(ValueError, match="not primary T0"):
        validate_primary_specimen_match(assignment)


@pytest.mark.parametrize(
    "tier",
    [
        "T1_same_lesion_procedure_separate_core",
        "T2_same_donor_different_block",
        "T3_wrong_donor_same_tissue_disease",
        "T4_query_excluded_generic_atlas",
        "X_unverified_same_tissue_or_integrated",
    ],
)
def test_every_nonprimary_tier_is_rejected_even_with_otherwise_exact_ids(tier: str) -> None:
    with pytest.raises(ValueError, match="not primary T0"):
        validate_primary_specimen_match(_exact_assignment(verified_match_tier=tier))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_donor_id", "D02", "donor IDs differ"),
        ("reference_specimen_id", "D01_BlockB", "canonical specimen IDs differ"),
        ("reference_block_id", "BlockB", "block IDs differ"),
    ],
)
def test_T0_identity_mismatches_are_each_rejected(field: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_primary_specimen_match(_exact_assignment(**{field: value}))


@pytest.mark.parametrize("overlap_field", ["query_he_section_id", "query_st_section_id"])
def test_reference_material_cannot_overlap_registered_query_material(overlap_field: str) -> None:
    assignment = _exact_assignment()
    assignment["reference_aliquot_or_curl_id"] = assignment[overlap_field]

    with pytest.raises(ValueError, match="overlaps a registered query section"):
        validate_primary_specimen_match(assignment)


def test_paper_only_or_unverified_evidence_fails_closed() -> None:
    assignment = _exact_assignment(
        payload_identity_verified=False,
        chain_of_custody_verified=False,
        paper_level_evidence_only=True,
    )

    errors = primary_specimen_match_errors(assignment)
    assert "payload_identity_verified must be true" in errors
    assert "chain_of_custody_verified must be true" in errors
    assert "paper_level_evidence_only must be false" in errors


@pytest.mark.parametrize(
    "field",
    [
        "contains_registered_query_material",
        "selection_uses_query_truth",
        "paper_level_evidence_only",
    ],
)
def test_missing_or_nonboolean_negative_evidence_fails_closed(field: str) -> None:
    assignment = _exact_assignment()
    assignment.pop(field)

    assert f"{field} must be false" in primary_specimen_match_errors(assignment)

    assignment[field] = "false"
    assert f"{field} must be false" in primary_specimen_match_errors(assignment)


def test_target_truth_cannot_select_an_assignment() -> None:
    with pytest.raises(ValueError, match="selection_uses_query_truth must be false"):
        validate_primary_specimen_match(_exact_assignment(selection_uses_query_truth=True))


def test_unresolved_reference_aliquot_fails_closed() -> None:
    with pytest.raises(ValueError, match="reference_aliquot_or_curl_id is unresolved"):
        validate_primary_specimen_match(
            _exact_assignment(reference_aliquot_or_curl_id="unresolved")
        )


def test_compound_manifest_placeholder_is_unresolved() -> None:
    errors = primary_specimen_match_errors(
        _exact_assignment(reference_aliquot_or_curl_id="unresolved_pending_snRNA_payload_metadata")
    )

    assert "reference_aliquot_or_curl_id is unresolved" in errors


def test_nonstring_identifiers_fail_closed_in_the_executable_validator() -> None:
    assignment = _exact_assignment(query_donor_id=101, reference_donor_id=101)

    errors = primary_specimen_match_errors(assignment)
    assert "query_donor_id must be a string" in errors
    assert "reference_donor_id must be a string" in errors


@pytest.mark.parametrize(
    "field",
    [
        "assignment_id",
        "query_id",
        "reference_sample_id",
        "reference_modality",
        "reference_assay_chemistry",
    ],
)
def test_missing_query_specific_identity_fields_fail_closed(field: str) -> None:
    assignment = _exact_assignment()
    assignment.pop(field)

    assert f"{field} must be a string" in primary_specimen_match_errors(assignment)


def test_unaccepted_reference_modality_fails_closed() -> None:
    errors = primary_specimen_match_errors(_exact_assignment(reference_modality="spatial"))

    assert "reference_modality is not an accepted sc/snRNA assay" in errors


def test_adjacent_curl_requires_explicit_adjacency_evidence() -> None:
    errors = primary_specimen_match_errors(
        _exact_assignment(verified_match_tier="T0_adjacent_curl_same_parent_specimen")
    )

    assert "adjacency_verified must be true for the adjacent-curl tier" in errors


def test_frozen_candidate_manifest_loads_with_strict_typed_booleans() -> None:
    path = Path(__file__).parents[1] / "manifests/spatialdlpfc_specimen_reference_assignments.tsv"
    rows = load_specimen_assignment_tsv(path)

    assert len(rows) == 19
    assert tuple(rows[0]) == SPECIMEN_ASSIGNMENT_V1_TSV_FIELDS
    assert all(row["primary_eligible"] is False for row in rows)
    assert all(row["payload_identity_verified"] is False for row in rows)
    assert all(primary_specimen_match_errors(row) for row in rows)
