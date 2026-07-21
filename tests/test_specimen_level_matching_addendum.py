from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]
ADDENDUM_PATH = ROOT / "configs/specimen_level_matching_addendum.json"
ADDENDUM_SHA256 = "f0a252602b8d0927a3ca259fb24385c5afae4bca7952261e96d0f47b8ddca97f"
DEFINITIVE_SHA256 = "6033a2d7db6cb5095d014f984c6f6519a4a7c073e41b627379a47525236b668a"
PILOT_SHA256 = "f483c0d40e8e29746cb7e4694ca8a3666e2d7196acc7ab2c02e3cc6c0c9b20e5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_addendum() -> dict:
    return json.loads(ADDENDUM_PATH.read_text(encoding="utf-8"))


def _load_assignments() -> list[dict[str, str]]:
    path = ROOT / "manifests/spatialdlpfc_specimen_reference_assignments.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_addendum_and_both_parent_protocols_are_byte_frozen() -> None:
    addendum = _load_addendum()
    relationship = addendum["relationship_to_frozen_protocols"]

    assert _sha256(ADDENDUM_PATH) == ADDENDUM_SHA256
    assert _sha256(ROOT / relationship["definitive_protocol"]["path"]) == DEFINITIVE_SHA256
    assert _sha256(ROOT / relationship["pilot_protocol"]["path"]) == PILOT_SHA256
    assert relationship["definitive_protocol"]["bytes_may_change"] is False
    assert relationship["pilot_protocol"]["bytes_may_change"] is False
    assert (
        relationship[
            "registered_before_final_eligible_block_assignment_and_ST_expression_target_access"
        ]
        is True
    )
    assert relationship["narrows_parent_primary_cohort_admission"] is True
    assert (
        relationship["changes_parent_model_estimand_primary_arms_gates_thresholds_or_decisions"]
        is False
    )
    assert relationship["adds_separate_nongating_T2_diagnostic"] is True


def test_addendum_binds_review_and_all_supporting_receipts() -> None:
    addendum = _load_addendum()

    assert addendum["source_review"]["sha256"] == (
        "845ada0952c7a76efb4ee84e92538e33fed144faa8982257f135de21c4dea42b"
    )
    for binding in addendum["bound_evidence"].values():
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]


def test_only_verified_exact_specimen_tiers_can_enter_primary_M3() -> None:
    addendum = _load_addendum()
    tiers = addendum["match_tiers"]
    allowed = addendum["primary_assignment_invariants"]["allowed_verified_match_tiers"]

    assert allowed == [
        "T0_exact_same_block_independent_aliquot",
        "T0_adjacent_curl_same_parent_specimen",
    ]
    assert {name for name, tier in tiers.items() if tier["primary_M3_eligible"]} == set(allowed)
    assert all(tiers[name]["primary_M3_eligible"] is False for name in tiers if name not in allowed)


def test_primary_invariants_require_same_parent_but_independent_material() -> None:
    invariants = _load_addendum()["primary_assignment_invariants"]

    assert invariants["query_and_reference_donor_ids_equal"] is True
    assert invariants["query_H_and_E_ST_and_reference_canonical_parent_specimen_ids_equal"] is True
    assert invariants["query_H_and_E_and_ST_block_ids_equal"] is True
    assert (
        invariants[
            "reference_aliquot_or_source_material_is_independent_from_query_section_material"
        ]
        is True
    )
    assert (
        invariants["registered_query_cells_sections_or_source_material_may_occur_in_reference"]
        is False
    )
    assert invariants["row_level_metadata_and_payload_evidence_required"] is True
    assert invariants["paper_level_pairing_statement_sufficient"] is False
    assert invariants["same_donor_different_block_action"] == "T2_control_never_primary"


def test_current_spatialdlpfc_rows_are_candidates_not_admitted_matches() -> None:
    rows = _load_assignments()

    assert len(rows) == 19
    assert len({row["assignment_id"] for row in rows}) == 19
    assert len({row["query_donor_id"] for row in rows}) == 10
    assert all(
        row["claimed_match_tier"] == "T0_exact_same_block_independent_aliquot" for row in rows
    )
    assert all(row["verified_match_tier"] == "unresolved" for row in rows)
    assert all(row["primary_eligible"] == "false" for row in rows)
    assert all(row["primary_selected"] == "false" for row in rows)
    assert all(row["payload_identity_verified"] == "false" for row in rows)
    assert all(row["registration_verified"] == "false" for row in rows)
    assert all(row["target_access_state"] == "opaque_ST_object_not_loaded" for row in rows)


def test_candidate_metadata_is_query_specific_and_donor_block_consistent() -> None:
    rows = _load_assignments()

    for row in rows:
        assert row["query_donor_id"] == row["reference_donor_id"]
        assert row["query_specimen_id"] == row["reference_specimen_id"]
        assert row["query_block_id"] == row["reference_block_id"]
        assert row["selection_tiebreak_sha256"] == (
            hashlib.sha256(
                (
                    "845ada0952c7a76efb4ee84e92538e33fed144faa8982257f135de21c4dea42b"
                    f"|{row['query_donor_id']}|{row['query_block_id']}|{row['query_capture_id']}"
                ).encode()
            ).hexdigest()
        )


def test_assignment_schema_covers_the_candidate_manifest_and_primary_evidence() -> None:
    schema = json.loads(
        (ROOT / "configs/schemas/specimen_reference_assignment.schema.json").read_text(
            encoding="utf-8"
        )
    )
    rows = _load_assignments()
    manifest_fields = set(rows[0])

    assert manifest_fields <= set(schema["properties"])
    assert set(schema["required"]) <= manifest_fields

    primary_rule = schema["allOf"][0]["then"]
    assert set(primary_rule["required"]) == {
        "chain_of_custody_verified",
        "canonical_parent_specimen_id_verified",
        "reference_modality_verified",
        "reference_assay_chemistry",
        "reference_assay_chemistry_verified",
        "independent_reference_material",
        "contains_registered_query_material",
        "selection_uses_query_truth",
        "paper_level_evidence_only",
    }


def test_br2720_filename_typo_is_resolved_only_at_metadata_level() -> None:
    rows = [row for row in _load_assignments() if row["query_donor_id"] == "Br2720"]
    receipt = json.loads(
        (ROOT / "manifests/spatialdlpfc_metadata_identity_receipt.json").read_text(encoding="utf-8")
    )

    assert len(rows) == 2
    assert all(
        row["verification_status"] == "metadata_filename_typo_resolved_payload_unverified"
        for row in rows
    )
    assert all(row["primary_eligible"] == "false" for row in rows)
    assert receipt["conclusion"]["metadata_identity_conflict_resolved"] is True
    assert receipt["conclusion"]["payload_level_exact_specimen_match_verified"] is False


def test_selection_counts_unique_donors_and_cannot_replace_after_target_access() -> None:
    selection = _load_addendum()["spatialDLPFC_primary_selection"]
    selected = [row for row in _load_assignments() if row["primary_selected"] == "true"]
    selected_counts = Counter(row["query_donor_id"] for row in selected)

    assert selection["minimum_unique_primary_donors"] == 8
    assert selection["one_primary_query_block_per_donor"] is True
    assert selection["multiple_blocks_or_sections_increase_independent_donor_count"] is False
    assert selection["currently_primary_eligible_donors"] == 0
    assert (
        selection[
            "ST_counts_expression_similarity_outcome_derived_annotations_predictions_or_loss_may_affect_selection"
        ]
        is False
    )
    assert selection["post_target_QC_failure_action"] == (
        "mark_selected_donor_nonevaluable_without_replacement"
    )
    assert all(count <= 1 for count in selected_counts.values())


def test_same_donor_wrong_block_is_separate_nongating_diagnostic() -> None:
    diagnostic = _load_addendum()["reference_specificity_diagnostic"]

    assert diagnostic["T2_supplemental_label"] == "R_same_donor_wrong_block"
    assert diagnostic["T2_is_a_new_primary_arm"] is False
    assert diagnostic["T2_may_enter_or_rescue_the_frozen_pilot_primary_decision"] is False
    assert diagnostic["T2_may_be_silently_reinterpreted_as_M6"] is False
    assert diagnostic["T2_control_banks"] == (
        "use_every_payload_qualified_alternate_same_donor_block_and_average_losses_within_donor"
    )
    assert (
        diagnostic["supplemental_inference"]["multiple_alternate_blocks_are_averaged_within_donor"]
        is True
    )
    assert diagnostic["permitted_spatialDLPFC_T2_claim"] == (
        "same_donor_different_region_or_block_reference_penalty"
    )
    assert diagnostic["pure_physical_block_specificity_claim"].startswith("unresolved_")


def test_donor_only_matcher_and_unsealed_target_keep_execution_blocked() -> None:
    addendum = _load_addendum()
    blocker = addendum["executable_compatibility_blocker"]
    boundary = addendum["target_seal_and_authorization"]

    for key in (
        "query_specific_validator",
        "query_specific_validator_tests",
        "current_reference_specificity_utility",
        "historical_reference_bank_schema",
    ):
        binding = blocker[key]
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]
    assert blocker["historical_donor_only_matcher_may_assign_primary_M3"] is False
    assert (
        "wire_the_query_specific_validator_into_the_future_pilot_assignment_loader"
        in blocker["required_before_execution"]
    )
    assert (
        boundary["processed_Visium_object_checksum_bound_but_role_separated_or_ACL_sealed"] is False
    )
    assert boundary["current_exFAT_permissions_enforce_read_separation"] is False
    assert boundary["ST_target_access_authorized"] is False
    assert boundary["biological_validation_authorized"] is False
    assert boundary["H_optimus_1_only"] is True
    assert boundary["UNI2_h_prohibited"] is True


def test_gse243280_is_mechanism_only_and_metadata_only() -> None:
    receipt = json.loads(
        (ROOT / "manifests/gse243280_metadata_receipt.json").read_text(encoding="utf-8")
    )

    candidate = receipt["exact_mechanism_candidate"]
    assert receipt["expression_or_spatial_target_downloaded_or_opened"] is False
    assert candidate["reference"]["GEO_sample"] == "GSM7782698"
    assert candidate["regional_truth"]["GEO_sample"] == "GSM7782699"
    assert candidate["claimed_match_tier"] == "T0_adjacent_curl_same_parent_specimen"
    assert candidate["verification_status"] == (
        "claimed_T0_metadata_candidate_payload_and_adjacency_unverified"
    )
    assert candidate["population_inference_eligible"] is False
    assert receipt["paper_evidence"]["paper_level_evidence_alone_can_admit_T0"] is False
    assert receipt["current_status"]["biological_execution_authorized"] is False
