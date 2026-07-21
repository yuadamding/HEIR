from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
PILOT_PATH = ROOT / "configs/independent_regional_pilot_protocol.json"
PILOT_PROTOCOL_SHA256 = "f483c0d40e8e29746cb7e4694ca8a3666e2d7196acc7ab2c02e3cc6c0c9b20e5"
ACQUISITION_RECEIPT_PATH = ROOT / "manifests/spatialdlpfc_pilot_acquisition_receipt.json"
ACQUISITION_RECEIPT_SHA256 = "5988dff71dab6bcaf8dd0297ad8352e10ed1108ed60d05c6a753254008f0e698"


def _load() -> dict:
    return json.loads(PILOT_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_pilot_protocol_bytes_are_frozen() -> None:
    assert _sha256(PILOT_PATH) == PILOT_PROTOCOL_SHA256


def test_post_registration_acquisition_receipt_preserves_target_seal() -> None:
    receipt = json.loads(ACQUISITION_RECEIPT_PATH.read_text(encoding="utf-8"))

    assert _sha256(ACQUISITION_RECEIPT_PATH) == ACQUISITION_RECEIPT_SHA256
    assert receipt["pilot_protocol"] == {
        "path": "configs/independent_regional_pilot_protocol.json",
        "sha256": PILOT_PROTOCOL_SHA256,
        "frozen_before_Visium_download": True,
    }
    assert receipt["eligibility"]["pilot_authorized"] is False
    assert receipt["eligibility"]["prediction_target_access_authorized"] is False
    assert receipt["eligibility"]["payload_qualified_exact_same_block_donors"] == 0
    assert receipt["payloads"]["processed_Visium_SpatialExperiment"]["access_state"] == (
        "opaque_checksum_sealed_not_loaded"
    )
    assert (
        receipt["payloads"]["processed_Visium_SpatialExperiment"][
            "satisfies_raw_or_minimally_processed_Visium_requirement"
        ]
        is False
    )


def test_pilot_is_separate_and_does_not_mutate_definitive_protocol() -> None:
    protocol = _load()
    relationship = protocol["definitive_protocol_relationship"]
    definitive = ROOT / relationship["path"]

    assert _sha256(definitive) == relationship["sha256"]
    assert relationship["sha256"] == (
        "6033a2d7db6cb5095d014f984c6f6519a4a7c073e41b627379a47525236b668a"
    )
    assert relationship["amends_definitive_protocol"] is False
    assert relationship["definitive_18_to_24_donor_standard_unchanged"] is True
    assert relationship["pilot_results_may_be_pooled_with_definitive_confirmation"] is False


def test_pilot_remains_blocked_until_eight_exact_triples_qualify() -> None:
    protocol = _load()
    execution = protocol["execution"]
    cohort = protocol["cohort_entry_contract"]

    assert execution["authorized"] is False
    assert execution["selected_cohort"] is None
    assert execution["score_target_open_authorized"] is False
    assert execution["minimum_exact_tri_modal_donors_before_authorization"] == 8
    assert cohort["independent_donors_minimum"] == 8
    assert cohort["independent_donors_maximum"] == 12
    assert cohort["exact_same_block_or_same_specimen_H_and_E_ST_sc_or_snRNA"] == "required"
    assert cohort["fewer_than_eight_exact_triples_action"] == (
        "pilot_ineligible_stop_without_target_access"
    )
    assert cohort["different_anterior_middle_or_posterior_blocks_are_interchangeable"] is False
    assert cohort["different_blocks_are_measurement_replicates"] is False


def test_frozen_model_is_hoptimus_only_one_step_and_unchanged() -> None:
    protocol = _load()
    model = protocol["frozen_model_identity"]
    identity = protocol["execution_identity"]

    assert model["source"] == "inherit_exactly_from_definitive_protocol"
    assert model["image_encoder"] == "bioptimus/H-optimus-1"
    assert model["image_encoder_revision"] == "3592cb220dec7a150c5d7813fb56e68bd57473b9"
    assert model["gene_panel"] == "unchanged_frozen_256_gene_order"
    assert model["molecular_latent_dimension"] == 20
    assert model["iterative_refinement_steps"] == 0
    assert model["fine_tuning"] == "prohibited"
    assert model["UNI2_h"] == "prohibited_not_run"
    assert model["candidate_outcomes_may_change_identity"] is False
    assert identity["kind"] == "frozen_leave_one_donor_out_training_and_scoring_procedure"
    assert identity["fully_pretrained_hash_bound_HEIR_checkpoint_available"] is False
    assert identity["heldout_donor_ST_role"] == "score_only_after_fold_specific_seal"
    assert identity["training_donor_ST_role"] == "fit_frozen_decoder_and_procedure_only"


def test_evidence_layers_cannot_erase_a_lower_tier_result() -> None:
    hierarchy = _load()["evidence_hierarchy"]

    assert hierarchy["pilot_primary_external_replication"] == [
        "C1_M3_less_than_M0",
        "C2_M3_paired_less_than_M4_shuffled_or_offset",
    ]
    assert hierarchy["full_core_regional_hypothesis"] == [
        "C3_valid_ST_floor_less_than_M3",
        "C1_M3_less_than_M0",
        "C2_spatial_attribution",
    ]
    assert hierarchy["strong_multimodal_synergy"] == "C6_M3_less_than_M1"
    assert hierarchy["continuous_state_inference"] == ("C5_M3_supported_less_than_M2_supported")
    assert hierarchy["pilot_may_authorize_full_core_regional_hypothesis"] is False
    assert hierarchy["stronger_tier_failure_may_erase_lower_tier_pass"] is False


def test_primary_execution_is_ordered_and_fail_closed() -> None:
    execution = _load()["ordered_execution"]

    assert execution["phase_0"] == (
        "qualify_and_freeze_exact_same_block_subset_before_ST_target_access"
    )
    assert execution["phase_1"] == "run_M0_and_M3_once"
    assert execution["phase_1_pass"] == {
        "minimum_relative_reduction": 0.05,
        "minimum_favorable_donor_fraction": 0.70,
        "paired_95_percent_interval_lower_bound": "greater_than_zero",
        "one_sided_exact_donor_sign_flip_p_max": 0.05,
        "minimum_prespecified_stratum_relative_reduction": -0.02,
    }
    assert execution["phase_2_requires_phase_1_decision"] is True
    assert execution["phase_2_requires_phase_1_pass"] is False
    assert execution["phase_2_pass"]["offset_um"] == [0, 55, 110, 220, 440]
    assert execution["phase_3"].endswith("otherwise_mark_unresolved")


def test_power_plan_does_not_overstate_the_sealed_donor_range() -> None:
    power = _load()["donor_level_power_planning"]

    assert power["planning_alternative"]["relative_reduction"] == 0.05
    assert power["development_source"]["outcome_exposed_development_only"] is True
    assert power["development_source"]["donors"] == 13
    assert power["development_source"]["paired_absolute_contrast_SD"] == 0.2236079978
    assert power["development_source"]["paired_absolute_contrast_SD_90_percent_upper_bound"] == (
        0.3085154
    )
    assert [row["donors"] for row in power["power_by_analyzable_donors"]] == [18, 21, 24]
    assert (
        power["power_by_analyzable_donors"][-1]["positive_95_percent_interval_conservative_SD"]
        == 0.485
    )
    assert power["pilot_positive_95_percent_interval_power_observed_SD"] == {
        "8_donors": 0.282,
        "10_donors": 0.359,
        "12_donors": 0.431,
    }
    assert any(
        "not_conservatively_80_percent_powered" in conclusion
        for conclusion in power["interpretation"]
    )


def test_floor_absence_is_unresolved_and_not_failure() -> None:
    floor = _load()["ST_floor"]

    assert floor["preference_order"] == [
        "registered_technical_replicate",
        "independently_registered_serial_spatial_section",
        "validated_negative_binomial_count_split_oracle",
    ]
    assert floor["same_target_as_M3"] is True
    assert floor["serial_section_claims_identical_spot_observations"] is False
    assert floor["target_expression_similarity_may_define_registration"] is False
    assert floor["unavailable_conclusion"] == "left_hand_inequality_unresolved_not_failed"
    assert floor["S1_is_floor"] is False


def test_high_fidelity_failure_does_not_erase_primary_pilot_evidence() -> None:
    quality = _load()["molecular_quality_reporting"]

    assert quality["mean_loss_and_state_reconstruction_reported_separately"] is True
    assert (
        quality["definitive_high_fidelity_thresholds_are_diagnostics_not_pilot_primary_gates"]
        is True
    )
    assert quality["high_fidelity_failure_may_negate_C1_or_C2_pilot_result"] is False


def test_acquisition_is_opaque_and_target_routing_is_metadata_only() -> None:
    protocol = _load()
    acquisition = protocol["spatialDLPFC_acquisition"]
    boundary = protocol["target_boundary"]

    assert acquisition["local_support_inventory"] == {
        "files": 227,
        "bytes": 386759455,
        "high_resolution_TIFFs": 0,
        "molecular_expression_objects": 0,
    }
    assert acquisition["downloaded_payloads_may_be_loaded_before_subset_freeze"] is False
    assert acquisition["opaque_downloads_must_be_checksum_bound"] is True
    evidence = acquisition["metadata_routing_evidence"]
    routing = ROOT / evidence["candidate_routing_manifest"]["path"]
    assert _sha256(routing) == evidence["candidate_routing_manifest"]["sha256"]
    assert evidence["same_block_candidate_rows"] == 19
    assert evidence["candidate_donors"] == 10
    assert evidence["Br2720_alignment_filename_typo_rows"] == 2
    snrna = acquisition["official_processed_payloads"]["snRNA_SingleCellExperiment"]
    assert snrna["status_at_registration"] == ("downloaded_opaque_integrity_verified_not_extracted")
    assert snrna["actual_bytes"] == snrna["expected_bytes"] == 4035795545
    assert snrna["sha256"] == ("15176538edd4d632fb19376229fd83b9446b90cfbc7cf0de7fbc599443a49c75")
    assert acquisition["raw_payload_access"]["local_transfer_status"] == (
        "blocked_no_Globus_authentication_and_destination_configuration"
    )
    assert boundary["stage_1_before_any_ST_count_access"] == [
        "metadata_only_donor_and_block_routing",
        "opaque_payload_checksum_sealing",
        "H_and_E_registration",
        "reference_qualification",
        "freeze_exact_donor_section_subset",
    ]
    assert boundary["routing_uses_metadata_only"] is True
    assert boundary["heldout_donor_ST_counts_are_score_only_after_fold_seal"] is True
    assert boundary["training_donor_ST_counts_may_fit_frozen_LODO_procedure"] is True
    assert boundary["heldout_ST_may_select_donors_blocks_genes_states_or_thresholds"] is False
    assert boundary["one_registered_execution_of_frozen_M0_and_M3_LODO_pipeline"] is True


def test_resource_limits_match_definitive_execution_limits() -> None:
    assert _load()["resource_limits"] == {
        "maximum_CPU_threads": 4,
        "maximum_visible_GPUs": 1,
        "maximum_GPU_memory_fraction": 0.60,
        "outer_folds_serial": True,
        "swap_permitted": False,
        "out_of_memory_action": "fail_closed_without_identity_change",
    }
