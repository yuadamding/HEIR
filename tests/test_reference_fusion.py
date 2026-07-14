from __future__ import annotations

import numpy as np
import pytest

from heir.evaluation.reference_fusion import (
    PrototypeBank,
    adaptive_residual_fusion,
    anchored_iteration,
    build_matched_wrong_generic_banks,
    build_reference_prototypes,
    deterministic_group_derangement,
    donor_section_macro_loss,
    donor_type_normalized_loss,
    equalize_bank_strata,
    evaluate_stage_gate,
    fit_reference_calibrator,
    fit_target_basis,
    reference_support_audit,
    residual_fusion,
    soft_reference_state,
    split_unique_molecules,
    type_routed_reference_state,
    within_type_reference_residuals,
)


def test_unique_molecule_split_is_disjoint_and_lane_duplicates_stay_together():
    section = ["S1", "S1", "S1", "S1"]
    barcode = ["A", "A", "A", "B"]
    feature = ["G1", "G1", "G2", "G1"]
    umi = ["U1", "U1", "U2", "U3"]
    half_a, half_b, receipt = split_unique_molecules(
        section, barcode, feature, umi, seed=9
    )
    assert half_a[0] == half_a[1]
    assert not np.any(half_a & half_b)
    assert np.all(half_a | half_b)
    assert receipt["unique_molecules"] == 3
    assert receipt["reconstructs_all_records"] is True


def test_target_basis_never_uses_heldout_donor_values():
    values = np.array([[0.0, 1.0], [2.0, 3.0], [500.0, -500.0]])
    donors = np.array(["A", "B", "C"])
    first = fit_target_basis(values, donors, ["A", "B"], n_components=2)
    altered = values.copy()
    altered[2] = [-1.0e9, 1.0e9]
    second = fit_target_basis(altered, donors, ["A", "B"], n_components=2)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.scale, second.scale)
    assert first.fit_donors == ("A", "B")


def test_reference_prototypes_are_deterministic_and_provenance_aligned():
    values = np.arange(24, dtype=float).reshape(8, 3)
    donors = np.array(["A"] * 4 + ["B"] * 4)
    types = np.array(["T"] * 2 + ["M"] * 2 + ["T"] * 2 + ["M"] * 2)
    observations = np.array([f"o{i}" for i in range(8)])
    first = build_reference_prototypes(
        values, donors, types, observations, max_prototypes_per_type=2, seed=4
    )
    second = build_reference_prototypes(
        values, donors, types, observations, max_prototypes_per_type=2, seed=4
    )
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.prototype_ids, second.prototype_ids)
    assert len(first.states) == 8
    assert first.weights.sum() == 8


def test_reference_calibration_excludes_heldout_target():
    reference = np.array([[0.0], [2.0], [4.0]])
    target = np.array([[1.0], [5.0], [9.0]])
    donors = np.array(["A", "B", "C"])
    first = fit_reference_calibrator(reference, donors, target, donors, ["A", "B"])
    altered = target.copy()
    altered[2, 0] = -10000
    second = fit_reference_calibrator(reference, donors, altered, donors, ["A", "B"])
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.target_mean, second.target_mean)
    assert first.fit_donors == ("A", "B")


def test_soft_reference_and_zero_correction_keep_h_image_central():
    h = np.array([[0.2, 0.8], [0.8, 0.2]])
    bank = np.array([[0.0, 1.0], [1.0, 0.0]])
    reference = soft_reference_state(h, bank, [1.0, 1.0], temperature=0.1)
    assert reference[0, 1] > reference[0, 0]
    assert reference[1, 0] > reference[1, 1]
    output = residual_fusion(h, reference, 0.0)
    np.testing.assert_array_equal(output, h)
    assert output is not h


def test_type_routing_marks_absent_types_and_uses_natural_fallback():
    bank = PrototypeBank(
        states=np.array([[0.0, 2.0], [2.0, 0.0], [4.0, 0.0]]),
        weights=np.array([1.0, 1.0, 2.0]),
        donor_ids=np.array(["A", "A", "A"]),
        type_labels=np.array(["T", "M", "M"]),
        prototype_ids=np.array(["p0", "p1", "p2"]),
    )
    routed, covered = type_routed_reference_state(["M", "absent"], bank)
    np.testing.assert_allclose(routed[0], [10.0 / 3.0, 0.0])
    np.testing.assert_allclose(routed[1], [2.5, 0.5])
    np.testing.assert_array_equal(covered, [True, False])


def test_bank_equalization_retains_equal_common_strata_only():
    selected, receipt = equalize_bank_strata(
        ["matched", "matched", "wrong", "wrong", "wrong"],
        [["T", "M", "T", "T", "X"], ["high", "low", "high", "high", "high"]],
        ["m0", "m1", "w0", "w1", "w2"],
        seed=3,
    )
    assert len(selected) == 2
    assert receipt["common_strata"] == 1
    assert receipt["per_bank_rows"] == 1
    assert receipt["query_outcomes_used"] is False


def test_support_audit_and_adaptive_fusion_abstain_out_of_support():
    image = np.array([[0.0], [4.0]])
    bank = np.array([[0.0], [1.0]])
    audit = reference_support_audit(image, bank, maximum_distance=2.0)
    fused, row_alpha = adaptive_residual_fusion(
        image, np.array([[1.0], [1.0]]), audit["support_weight"], 0.5
    )
    assert audit["supported"].tolist() == [True, False]
    np.testing.assert_allclose(row_alpha, [0.5, 0.0])
    np.testing.assert_allclose(fused, [[0.5], [4.0]])


def test_bank_builder_excludes_query_from_wrong_and_generic():
    bank = PrototypeBank(
        states=np.arange(12, dtype=float).reshape(6, 2),
        weights=np.ones(6),
        donor_ids=np.array(["A", "A", "B", "B", "C", "C"]),
        type_labels=np.array(["x"] * 6),
        prototype_ids=np.array([f"p{i}" for i in range(6)]),
    )
    result = build_matched_wrong_generic_banks(
        bank, "A", "lung", {"A": "lung", "B": "lung", "C": "breast"}
    )
    assert result["wrong_donors"] == ["B"]
    assert np.all(bank.donor_ids[result["matched"]] == "A")
    assert np.all(bank.donor_ids[result["generic"]] == "B")
    assert result["query_donor_excluded_from_generic"] is True


def test_derangement_is_deterministic_grouped_and_has_no_fixed_points():
    groups = np.array(["s1"] * 4 + ["s2"] * 3)
    observations = np.array([f"o{i}" for i in range(7)])
    first = deterministic_group_derangement(groups, observations, seed=8)
    second = deterministic_group_derangement(groups, observations, seed=8)
    np.testing.assert_array_equal(first, second)
    assert not np.any(first == np.arange(7))
    np.testing.assert_array_equal(groups[first], groups)
    with pytest.raises(ValueError, match="fewer than two"):
        deterministic_group_derangement(["one"], ["o"])


def test_donor_section_loss_gives_donors_and_sections_equal_weight():
    truth = np.zeros((5, 1))
    prediction = np.array([[1.0], [1.0], [3.0], [2.0], [2.0]])
    result = donor_section_macro_loss(
        truth,
        prediction,
        ["A", "A", "A", "B", "B"],
        ["a1", "a1", "a2", "b1", "b1"],
    )
    # A: mean(section losses 1 and 9)=5; B: 4; donor macro=4.5.
    assert result["donor_mse"] == {"A": 5.0, "B": 4.0}
    assert result["donor_section_macro_mse"] == 4.5


def test_within_type_residuals_use_exact_matched_sc_mean_and_report_missing():
    residual, means, covered, receipt = within_type_reference_residuals(
        [[5.0], [7.0], [9.0]],
        ["A", "A", "B"],
        ["T", "M", "T"],
        [[1.0], [3.0], [4.0]],
        ["A", "A", "A"],
        ["T", "T", "M"],
    )
    np.testing.assert_allclose(residual[:2], [[3.0], [3.0]])
    np.testing.assert_allclose(means[:2], [[2.0], [4.0]])
    assert np.isnan(residual[2, 0])
    np.testing.assert_array_equal(covered, [True, True, False])
    assert receipt["coverage"] == pytest.approx(2 / 3)
    assert receipt["generic_fallback_used"] is False


def test_donor_type_loss_matches_prespecified_normalized_formula():
    truth = np.array([[0.0], [2.0], [0.0], [4.0], [0.0], [2.0]])
    prediction = np.array([[1.0], [1.0], [2.0], [2.0], [0.0], [2.0]])
    result = donor_type_normalized_loss(
        truth,
        prediction,
        ["A", "A", "A", "A", "B", "B"],
        ["T", "T", "M", "M", "T", "T"],
        section_ids=["a1", "a2", "a1", "a2", "b1", "b1"],
    )
    # A/T: SSE 2 / SST 2 = 1; A/M: 8 / 8 = 1; B/T: 0 / 2 = 0.
    assert result["donor_loss"] == {"A": 1.0, "B": 0.0}
    assert result["donor_type_balanced_loss"] == 0.5
    assert result["donor_section_type_balanced_loss"] == 0.5


def test_failed_stage_gate_prevents_iteration():
    gate = evaluate_stage_gate(
        {"A": 1.0, "B": 1.0},
        {"A": 0.99, "B": 1.01},
        {"A": 1.2, "B": 1.2},
        {"A": 1.2, "B": 1.2},
        floor_loss_by_donor={"A": 0.2, "B": 0.2},
        median_variance_ratio=0.8,
    )
    assert gate["passed"] is False
    assert gate["decision"] == "iteration_not_run_failed_inner_gate"
    assert gate["criteria"]["relative_gain"] is False


def test_iteration_is_always_anchored_to_original_h():
    h = np.array([[0.0]])
    bank = np.array([[1.0]])
    final, history = anchored_iteration(h, bank, [1.0], [0.5, 0.5])
    assert len(history) == 2
    # A recursive unanchored update would be 0.75; anchored stays at 0.5.
    np.testing.assert_allclose(final, [[0.5]])
