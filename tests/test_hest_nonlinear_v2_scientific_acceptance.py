from __future__ import annotations

import numpy as np
import pytest

from heir.evaluation import nonlinear_qualification_v2_acceptance as acceptance
from heir.evaluation.hierarchical_metrics import (
    donor_section_type_macro_r2,
    macro_error_reduction,
    macro_r2,
)
from heir.evaluation.nonlinear_qualification_contract import canonical_sha256
from heir.evaluation.nonlinear_qualification_v2_acceptance import (
    NULL_FAMILIES,
    ArmPrediction,
    QualificationInputs,
    RegistrationComparison,
    score_qualification,
    synthetic_trial_inputs,
)
from heir.evaluation.nonlinear_qualification_v2_contract import CALIBRATION_EXPECTATIONS


def test_indexed_scoring_is_exactly_equal_to_shared_hierarchical_metrics() -> None:
    rng = np.random.default_rng(71)
    donors = np.repeat(["D1", "D2"], 80)
    sections = np.tile(np.repeat(["S1", "S2"], 40), 2)
    labels = np.tile(np.repeat([0, 1], 20), 4)
    truth = rng.normal(size=(len(donors), 3))
    prediction = truth + rng.normal(scale=0.4, size=truth.shape)
    baseline = np.zeros_like(truth)
    donor_type = acceptance._group_indices(donors, labels)
    donor_section_type = acceptance._group_indices(donors, sections, labels)

    pairs = (
        (
            acceptance._donor_type_summary(truth, prediction, donor_type, 20),
            macro_r2(truth, prediction, donors, labels, 20),
        ),
        (
            acceptance._donor_type_summary(
                truth,
                prediction,
                donor_type,
                20,
                baseline=baseline,
            ),
            macro_error_reduction(truth, prediction, baseline, donors, labels, 20),
        ),
        (
            acceptance._donor_section_type_r2(
                truth,
                prediction,
                donor_section_type,
                20,
            ),
            donor_section_type_macro_r2(
                truth,
                prediction,
                donors,
                sections,
                labels,
                20,
            ),
        ),
    )
    assert all(canonical_sha256(indexed) == canonical_sha256(shared) for indexed, shared in pairs)


def _topology() -> tuple[np.ndarray, ...]:
    donor_names = [f"D{index:02d}" for index in range(12)]
    donors = np.repeat(donor_names, 60)
    labels = np.tile(np.repeat([0, 1], 30), len(donor_names))
    sections = np.asarray([f"{donor}-S" for donor in donors])
    within_type = np.tile(np.arange(30), len(donor_names) * 2)
    quality = within_type < 10
    best = within_type < 5
    return donors, sections, labels, quality, best, np.asarray(donor_names)


def _qualification(*, localized: str | None = None) -> QualificationInputs:
    donors, sections, labels, quality, best, included = _topology()
    local = np.tile(np.linspace(-1.0, 1.0, 30), len(included) * 2)
    truth = np.column_stack((local, 0.7 * local[::-1]))
    ranks = np.full(len(truth), 2, dtype=np.int64)
    lower = np.full_like(truth, -0.7)
    upper = np.full_like(truth, 0.7)

    def shifted(offset: float) -> np.ndarray:
        return truth + offset

    predictions = {
        "B2": shifted(0.60),
        "N0": shifted(0.65),
        "N1": shifted(0.45 if localized else 0.05),
        "N3": shifted(0.05 if localized == "N3" else 0.45 if localized else 0.05),
        "N4": shifted(0.05 if localized == "N4" else 0.45 if localized else 0.05),
        "N5": shifted(0.45 if localized else 0.05),
        "A_star": shifted(0.05),
        "C_blank": shifted(0.65),
        "C_blank_single": shifted(0.65),
    }
    arms = {
        name: ArmPrediction(truth, prediction, ranks, prediction, lower, upper)
        for name, prediction in predictions.items()
    }
    pairs = {
        "A_star_vs_N0": ("A_star", "N0"),
        "N1_vs_N5": ("N1", "N5"),
        "N3_vs_N5": ("N3", "N5"),
        "N4_vs_N5": ("N4", "N5"),
    }
    registration = {
        name: RegistrationComparison(truth, predictions[focal], predictions[comparator], ranks)
        for name, (focal, comparator) in pairs.items()
    }
    return QualificationInputs(
        donor_ids=donors,
        section_ids=sections,
        type_labels=labels,
        molecular_truth=truth,
        reference_prediction=np.zeros_like(truth),
        arms=arms,
        registration=registration,
        quality_mask=quality,
        all_rows_mask=np.ones(len(truth), dtype=bool),
        best_mask=best,
        included_donor_ids=included.tolist(),
        refitted_null_empirical_p={name: 1 / 101 for name in NULL_FAMILIES},
    )


def test_shared_scoring_separates_context_estimation_from_localization() -> None:
    context = score_qualification(_qualification(), registration_bootstrap_resamples=1_000)
    assert context["decisions"]["D1"]["passed"] is True
    assert context["decisions"]["D2"]["passed"] is False
    assert context["authorizes_h_cell"] is False

    nucleus = score_qualification(
        _qualification(localized="N3"), registration_bootstrap_resamples=1_000
    )
    assert nucleus["decisions"]["D1"]["passed"] is True
    assert nucleus["decisions"]["D2"]["passed"] is True
    assert nucleus["decisions"]["D2"]["arms"]["N3"]["passed_all_arm_rules"] is True
    assert nucleus["decisions"]["D2"]["arms"]["N4"]["passed_all_arm_rules"] is False


def test_registration_confined_signal_fails_the_shared_D1_gate() -> None:
    inputs = _qualification(localized="N3")
    quality = np.asarray(inputs.quality_mask)
    focal = np.asarray(inputs.arms["A_star"].coordinate_prediction).copy()
    focal[quality] = inputs.arms["N0"].coordinate_prediction[quality]
    arms = dict(inputs.arms)
    original = arms["A_star"]
    arms["A_star"] = ArmPrediction(
        original.coordinate_truth,
        focal,
        original.coordinate_ranks,
        focal,
        original.rare_lower,
        original.rare_upper,
    )
    registration = dict(inputs.registration)
    original_comparison = registration["A_star_vs_N0"]
    registration["A_star_vs_N0"] = RegistrationComparison(
        original_comparison.truth,
        focal,
        original_comparison.comparator_prediction,
        original_comparison.coordinate_ranks,
    )
    changed = QualificationInputs(
        **{
            **inputs.__dict__,
            "arms": arms,
            "registration": registration,
        }
    )
    result = score_qualification(changed, registration_bootstrap_resamples=1_000)
    noninferiority = result["registration_sensitivity"]["A_star_vs_N0"]["noninferiority"]
    assert noninferiority["estimate"] < -0.01
    assert noninferiority["passed"] is False
    assert result["decisions"]["D1"]["passed"] is False


@pytest.mark.parametrize("condition", CALIBRATION_EXPECTATIONS)
def test_frozen_synthetic_condition_matches_its_expected_decisions(condition: str) -> None:
    donors, sections, labels, quality, best, included = _topology()
    inputs = synthetic_trial_inputs(
        donors,
        sections,
        labels,
        quality,
        best,
        included.tolist(),
        condition=condition,
        trial_index=0,
    )
    result = score_qualification(inputs, registration_bootstrap_resamples=1_000)
    observed = {
        decision: bool(result["decisions"][decision]["passed"]) for decision in ("D1", "D2")
    }
    assert observed == CALIBRATION_EXPECTATIONS[condition]
