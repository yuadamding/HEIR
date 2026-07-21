from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = ROOT / "configs/independent_regional_confirmation_protocol.json"
DEFINITIVE_PROTOCOL_SHA256 = "6033a2d7db6cb5095d014f984c6f6519a4a7c073e41b627379a47525236b668a"


def _load_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_definitive_protocol_bytes_remain_sealed() -> None:
    assert _sha256(PROTOCOL_PATH) == DEFINITIVE_PROTOCOL_SHA256


def test_confirmation_stays_blocked_without_an_eligible_cohort() -> None:
    protocol = _load_protocol()
    execution = protocol["execution"]
    sequence = protocol["action_sequence_status"]

    assert execution["status"] == "blocked_no_eligible_independent_cohort"
    assert execution["authorized"] is False
    assert execution["selected_cohort"] is None
    assert execution["score_target_open_authorized"] is False
    assert len(execution["blockers"]) == 3
    assert sequence["1_lock_NatCommun_development_only"] == "complete"
    assert sequence["2_select_independent_single_tissue_cohort"] == (
        "blocked_no_eligible_downloaded_cohort"
    )
    assert sequence["4_precompute_target_free_inputs"] == (
        "not_started_requires_selected_eligible_cohort"
    )
    assert sequence["5_run_Gate_A"] == "not_started_not_authorized"
    assert sequence["13_iterative_refinement"] == "prohibited"


def test_natcommun_is_immutable_development_only_evidence() -> None:
    lock = _load_protocol()["development_lock"]

    assert lock["analysis_status"] == "exposed_development_only_non_confirmatory"
    assert lock["new_outcome_exposed_fits"] == "prohibited"
    assert lock["new_tuning_or_threshold_selection"] == "prohibited"
    assert lock["confirmatory_reuse"] == "prohibited"

    for binding in lock["bindings"].values():
        if not isinstance(binding, dict) or "path" not in binding:
            continue
        path = ROOT / binding["path"]
        assert path.is_file()
        if "bytes" in binding:
            assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]


def test_encoder_and_one_step_model_identity_are_frozen() -> None:
    protocol = _load_protocol()
    model = protocol["frozen_model_identity"]
    manifest_binding = model["encoder_manifest"]
    manifest_path = ROOT / manifest_binding["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert _sha256(manifest_path) == manifest_binding["sha256"]
    assert model["image_encoder"] == "bioptimus/H-optimus-1"
    assert model["image_encoder_revision"] == manifest["revision"]
    assert model["encoder_checkpoint_sha256"] == manifest["checkpoint_sha256"]
    assert model["UNI2_h"] == "prohibited_not_run"
    assert model["H_and_E_context_um"] == 112
    assert model["molecular_latent_dimension"] == 20
    assert model["iterative_refinement_steps"] == 0
    assert model["confirmation_outcomes_may_change_identity"] is False


def test_primary_estimand_and_all_scientific_arms_are_explicit() -> None:
    protocol = _load_protocol()
    estimand = protocol["primary_estimand"]

    assert estimand["ordering"] == ("L_ST < L(M3_H_and_E_plus_matched_sc_or_snRNA) < L(M0_H_and_E)")
    assert estimand["spatial_attribution"] == (
        "L(M3_paired_H_and_E_plus_R) < L(M4_shuffled_or_offset_H_and_E_plus_R)"
    )
    assert estimand["loss"] == ("donor_and_section_balanced_heldout_negative_binomial_deviance")
    assert estimand["primary_gene_panel_size"] == 256
    assert estimand["secondary_latent_dimension"] == 20
    assert set(protocol["arms"]) == {
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5a",
        "M5b",
        "M6",
        "M7",
        "F_ST",
        "S1",
        "S3",
        "S4",
    }
    assert protocol["gates"]["C"]["S1_is_measurement_floor"] is False


def test_cohort_and_reference_entry_contract_precedes_target_opening() -> None:
    protocol = _load_protocol()
    cohort = protocol["cohort_entry_contract"]
    boundary = protocol["target_boundary"]

    assert cohort["one_tissue_or_indication"] is True
    assert cohort["independent_donors_minimum"] == 18
    assert cohort["independent_donors_maximum"] == 24
    assert cohort["preferred_query_sections_per_donor"] == 2
    assert cohort["reference_modality_independently_verified"] is True
    assert cohort["minimum_qualified_reference_cells_per_retained_coarse_type"] == 50
    assert cohort["outcomes_unopened_before_lock"] is True
    assert cohort["unsupported_type_substitution"] == "prohibited"
    assert boundary["Gate_A_runs_first"] is True
    assert boundary["Gate_B_may_run_after_Gate_A_failure"] is False
    assert boundary["Gates_C_D_E_F_may_run_after_Gate_A_failure"] is True
    assert (
        boundary["post_A_failure_gates_are_diagnostic_and_cannot_rescue_the_core_hypothesis"]
        is True
    )
    assert boundary["confirmation_target_may_select_or_change"] == []
    assert boundary["genes_crop_fusion_floor_states_iterations_or_thresholds_may_be_tuned"] is False


def test_gate_hierarchy_and_pass_thresholds_match_the_action_plan() -> None:
    protocol = _load_protocol()
    gates = protocol["gates"]

    assert protocol["gate_order"] == ["A", "B", "C", "D", "E", "F"]
    assert gates["A"]["minimum_relative_reduction"] == 0.05
    assert gates["A"]["minimum_favorable_donor_fraction"] == 0.70
    assert gates["A"]["one_sided_exact_sign_flip_or_paired_randomization_p_max"] == 0.05
    assert gates["A"]["severe_prespecified_stratum_reversal_permitted"] is False
    assert gates["A"]["minimum_prespecified_stratum_relative_reduction"] == -0.02

    assert gates["B"]["requires_gate_pass"] == ["A"]
    for gate_id in ("C", "D", "E", "F"):
        assert gates[gate_id]["requires_gate_decision"] == ["A"]
    assert gates["B"]["offset_spot_diameters"] == [0, 1, 2, 4, 8]
    assert gates["B"]["visium_offset_um_when_applicable"] == [0, 55, 110, 220, 440]
    assert gates["B"]["M3_vs_M4_minimum_favorable_donor_fraction"] == 0.70
    assert gates["B"]["M3_vs_M4_one_sided_exact_sign_flip_p_max"] == 0.05
    assert gates["B"]["hotspot_high_residual_quantile"] == 0.90
    assert gates["C"]["minimum_favorable_donor_fraction"] == 0.70
    assert gates["C"]["one_sided_donor_test_p_max"] == 0.05
    assert gates["D"]["one_sided_exact_sign_flip_p_max"] == 0.05
    assert gates["E"]["Holm_adjusted_p_max"] == 0.05
    assert gates["E"]["minimum_favorable_donor_fraction"] == 0.70
    assert (
        gates["E"]["maximum_single_donor_or_prespecified_group_fraction_of_positive_effect"] == 0.50
    )
    assert gates["F"]["minimum_M3_to_M0_reliability_adjusted_variance_ratio"] == 0.80
    assert gates["F"]["maximum_M0_minus_M3_rare_state_recall"] == 0.05
    assert gates["F"]["maximum_M3_to_M0_program_covariance_error_ratio"] == 1.10
    assert gates["F"]["minimum_M3_to_M0_median_gene_dynamic_range_ratio"] == 0.80
    assert gates["F"]["calibration_slope_interval"] == [0.80, 1.20]
    assert gates["F"]["minimum_state_favorable_donor_fraction"] == 0.70


def test_regional_decision_precedes_any_cell_level_extension() -> None:
    protocol = _load_protocol()

    assert protocol["decision"]["core_regional_requires"] == [
        "A_pass",
        "B_pass",
        "C_pass",
        "F_no_unacceptable_molecular_collapse",
    ]
    assert protocol["decision"]["personalized_reference_claim_requires"] == "E_pass"
    assert protocol["decision"]["strong_multimodal_synergy_requires"] == "C6_pass"
    assert protocol["decision"]["iteration_reconsideration_requires"] == [
        "A_pass",
        "B_pass",
        "C5_pass",
        "F_state_preservation_pass",
    ]
    assert protocol["cell_level_extension"]["status"] == "deferred_until_core_regional_pass"
    assert protocol["cell_level_extension"]["endpoints"] == {
        "total_expression": "L_total=L(Y_i_ST,predicted_Y_i)",
        "within_fine_type_residual_target": "r_i=Y_i_ST-mu_R_d_t",
        "fine_type_source": "independently_defined_fine_type_t",
        "within_type_state_loss": "L_state=L(r_i,predicted_r_i)",
    }
    assert set(protocol["cell_level_extension"]["localization_interpretation"]) == {
        "only_context_passes",
        "cell_mask_passes_beyond_context",
        "nucleus_mask_passes_beyond_context",
        "full_image_passes_but_masks_fail",
        "shuffled_or_offset_equals_paired",
    }
    assert protocol["cell_level_extension"]["failure_claim_boundary"] == "regional_only"


def test_stopping_conclusions_are_frozen_before_confirmation() -> None:
    stopping = _load_protocol()["stopping_conclusions"]

    assert stopping == {
        "A_fail": "right_hand_inequality_not_replicated_stop",
        "A_pass_B_fail": "reference_helps_but_H_and_E_spatial_contribution_unproven",
        "paired_signal_without_A": (
            "H_and_E_contains_spatial_signal_but_does_not_improve_target_loss"
        ),
        "C_fail": "do_not_claim_left_hand_inequality",
        "mean_loss_pass_F_fail": "denoised_conditional_mean_only_not_molecular_heterogeneity",
        "E_fail": "no_sample_specific_reference_value_generic_reference_may_suffice",
        "A_and_B_pass_C5_fail": "composition_or_type_routing_not_continuous_state",
        "regional_pass_cell_fail": "regional_inference_only",
        "regional_and_cell_pass": "key_hypothesis_supported_within_frozen_scope",
    }


def test_resource_limits_remain_bounded_and_fail_closed() -> None:
    limits = _load_protocol()["resource_limits"]

    assert limits == {
        "maximum_CPU_threads": 4,
        "maximum_visible_GPUs": 1,
        "maximum_GPU_memory_fraction": 0.60,
        "outer_folds_serial": True,
        "swap_permitted": False,
        "out_of_memory_action": "fail_closed_without_identity_change",
    }
