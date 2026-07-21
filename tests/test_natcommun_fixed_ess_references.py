from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_runner():
    path = Path(__file__).parents[1] / "scripts/validate_natcommun_fixed_ess_references.py"
    spec = importlib.util.spec_from_file_location("validate_natcommun_fixed_ess_references", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _synthetic_plan_and_latent() -> tuple[dict[str, np.ndarray], np.ndarray]:
    cells, banks, types, depth, features, mitochondrial = [], [], [], [], [], []
    latent_rows = []
    repeats_by_bank = {"A": 8, "B": 9, "C": 9}
    bank_offset = {"A": -4.0, "B": 1.0, "C": 5.0}
    for bank in ("A", "B", "C"):
        repeats = repeats_by_bank[bank]
        for type_index, type_name in enumerate(("T1", "T2")):
            for state in range(3):
                angle = 2.0 * np.pi * state / 3.0
                for high in (False, True):
                    for replicate in range(repeats):
                        cells.append(f"{bank}-{type_name}-s{state}-q{int(high)}-r{replicate:02d}")
                        banks.append(bank)
                        types.append(type_name)
                        qc = 3.0 if high else 1.0
                        depth.append(qc)
                        features.append(qc)
                        mitochondrial.append(qc)
                        row = np.zeros(runner.FROZEN_LATENT_DIM, dtype=np.float64)
                        row[0] = 8.0 * np.cos(angle)
                        row[1] = 8.0 * np.sin(angle)
                        row[2] = 0.75 * float(high)
                        fraction = (replicate + 1.0) / repeats
                        row[3] = 3.0 * fraction + 0.15 * state * fraction
                        row[4] = bank_offset[bank]
                        row[5] = 2.0 * type_index
                        row[6] = fraction**2
                        latent_rows.append(row)
    plan = runner._compute_weight_plan(
        heldout_donor="A",
        wrong_donor_ids=["B", "C"],
        cell_ids=cells,
        bank_ids=banks,
        type_ids=types,
        total_umi=depth,
        n_features=features,
        percent_mt=mitochondrial,
    )
    runner._validate_weight_plan(plan, donor="A")
    return plan, np.vstack(latent_rows)


def _state_authority(plan: dict[str, np.ndarray], latent: np.ndarray) -> dict[str, object]:
    return runner._fit_state_balance_authority(
        plan,
        latent,
        seed=runner.FROZEN_BASE_SEED,
        iterations=25,
    )


def _with_matched_only_types(
    extra_types: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    base, latent = _synthetic_plan_and_latent()
    cells = np.asarray(base["cell_ids"]).astype(str).tolist()
    banks = np.asarray(base["bank_ids"]).astype(str).tolist()
    types = np.asarray(base["type_ids"]).astype(str).tolist()
    depth = np.asarray(base["total_umi"], dtype=np.float64).tolist()
    features = np.asarray(base["n_features_rna"], dtype=np.float64).tolist()
    mitochondrial = np.asarray(base["percent_mt"], dtype=np.float64).tolist()
    additions = []
    for type_name in extra_types:
        for index in range(8):
            cells.append(f"A-{type_name}-{index:02d}")
            banks.append("A")
            types.append(type_name)
            value = 1.0 if index < 4 else 3.0
            depth.append(value)
            features.append(value)
            mitochondrial.append(value)
            additions.append(np.zeros(runner.FROZEN_LATENT_DIM, dtype=np.float64))
    plan = runner._compute_weight_plan(
        heldout_donor="A",
        wrong_donor_ids=["B", "C"],
        cell_ids=cells,
        bank_ids=banks,
        type_ids=types,
        total_umi=depth,
        n_features=features,
        percent_mt=mitochondrial,
        model_type_names=["T1", "T2"],
    )
    runner._validate_weight_plan(plan, donor="A")
    return plan, np.vstack((latent, additions))


def _adapter_fixture():
    plan, latent = _synthetic_plan_and_latent()
    core = runner._load_baseline_runner()._import_core()
    adapter = runner._WeightedReferenceAdapter(core, plan, lambda: latent)
    cells = np.asarray(plan["cell_ids"]).astype(str)
    banks = np.asarray(plan["bank_ids"]).astype(str)
    types = np.asarray(plan["type_ids"]).astype(str)
    mixtures = {}
    for bank in ("A", "B", "C"):
        keep = banks == bank
        mixtures[bank] = adapter(
            latent[keep],
            donor_ids=banks[keep],
            type_ids=types[keep],
            observation_ids=cells[keep],
            source_modality="single_cell",
            n_components=3,
            seed=runner.FROZEN_BASE_SEED,
        )
    generic = banks != "A"
    mixtures["generic"] = adapter(
        latent[generic],
        donor_ids=banks[generic],
        type_ids=types[generic],
        observation_ids=cells[generic],
        source_modality="single_cell",
        donor_equal=True,
        n_components=3,
        seed=runner.FROZEN_BASE_SEED,
    )
    return plan, latent, adapter, mixtures


def _capture_receipt(reference_hash: str) -> dict[str, object]:
    roles = (
        "aligned_train_st",
        "aligned_train_sc",
        "aligned_matched_sc",
        "unaligned_train_st",
        "unaligned_train_sc",
    )
    rows = (2, 3, 2, 2, 3)
    return {
        "contract": "frozen_encode_sequence_v1",
        "calls": [
            {"ordinal": index, "role": role, "rows": rows[index]}
            for index, role in enumerate(roles)
        ],
        "reference_latent_sha256": reference_hash,
        "score_target_opened": False,
    }


def _prediction_fixture():
    plan, latent, adapter, _ = _adapter_fixture()
    assert adapter.state_authority is not None
    state = adapter.state_authority
    authority = adapter.authority_receipt()
    expected_plan_sha = "registered-plan"
    predictions = {
        "fixed_reference_plan_semantic_sha256": np.asarray(expected_plan_sha),
        "fixed_reference_QC_common_type_names": np.asarray(plan["common_type_names"]),
        "fixed_reference_common_type_names": np.asarray(state["evaluable_type_names"]),
        "fixed_reference_single_bank_cell_weight_ESS": np.asarray(state["single_bank_target_ess"]),
        "fixed_reference_generic_pooled_cell_weight_ESS": np.asarray(
            state["generic_pooled_target_ess"]
        ),
        "fixed_reference_hard_subsampling_used": np.asarray(False),
        "fixed_reference_support_restriction_used": np.asarray(
            bool(state["support_restriction_used"])
        ),
        "fixed_reference_adapter_calls_json": np.asarray(json.dumps(adapter.calls)),
        "fixed_reference_state_capture_json": np.asarray(
            json.dumps(_capture_receipt(str(authority["reference_latent_sha256"])))
        ),
        "fixed_reference_state_authority_json": np.asarray(json.dumps(authority)),
        "query_H_composition": np.asarray([[0.5, 0.5]], dtype=np.float32),
        "gate3_supported_type_mask": np.asarray([True, True]),
        "gate3_supported_composition_mass": np.asarray([1.0], dtype=np.float64),
        "fixed_reference_gate3_mass_receipt_precision": np.asarray(
            "float64_exact_recomputation_from_frozen_H_and_E_composition"
        ),
        "fixed_reference_gate3_threshold_rounding_rows": np.asarray(0),
        **adapter.authority_arrays(),
    }
    baseline = SimpleNamespace(
        _validate_prediction_artifact=lambda *args, **kwargs: None,
        _semantic_array_hash=lambda arrays: expected_plan_sha,
        _fold_seed=lambda seed, donor: seed,
    )
    public = {
        "wrong_donor_ids": np.asarray(["B", "C"]),
        "train_st_counts": np.zeros((2, 1)),
        "train_sc_counts": np.zeros((3, 1)),
        "train_sc_type_ids": np.asarray(["T1", "T2", "T1"]),
        "matched_sc_counts": np.zeros((2, 1)),
    }
    return plan, predictions, baseline, public


def test_distance_tilt_is_exact_positive_and_relabel_invariant() -> None:
    distances = np.asarray([0.0, 0.2, 0.7, 1.0, 0.1, 0.4, 0.8, 1.0])
    categories = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    masses = {0: 0.35, 1: 0.65}
    minimum, maximum = runner._distance_tilt_interval(distances, categories, masses)
    target = (minimum + maximum) / 2.0
    weights, tilt = runner._distance_tilt_at_exact_ess(distances, categories, masses, target)
    assert 0 < tilt < runner.FROZEN_STATE_MAX_TILT
    assert np.all(weights > 0)
    assert np.isclose(runner._ess(weights), target)
    assert np.isclose(weights[categories == 0].sum(), masses[0])
    assert np.isclose(weights[categories == 1].sum(), masses[1])

    permutation = np.asarray([6, 1, 4, 3, 0, 7, 2, 5])
    permuted, _ = runner._distance_tilt_at_exact_ess(
        distances[permutation], categories[permutation], masses, target
    )
    inverse = np.argsort(permutation)
    assert np.allclose(permuted[inverse], weights)


def test_qc_plan_is_target_free_common_support_only() -> None:
    plan, _ = _synthetic_plan_and_latent()
    assert plan["schema"].item() == runner.PLAN_SCHEMA
    assert plan["common_type_names"].tolist() == ["T1", "T2"]
    assert set(plan["category_joint_strata"].tolist()) == {0, 7}
    assert "single_bank_cell_weights" not in plan
    assert "generic_cell_weights" not in plan
    assert not bool(plan["hard_subsampling_used"])
    assert plan["state_balance_method"].item() == runner.STATE_BALANCE_METHOD


def test_global_type_universe_reports_matched_only_exclusion_and_conditional_coverage() -> None:
    plan, latent = _with_matched_only_types(("T3",))
    authority = _state_authority(plan, latent)
    assert authority["all_observed_type_names"] == ("T1", "T2", "T3")
    assert np.isclose(authority["conditional_evaluable_type_fraction"], 1.0)
    assert np.isclose(authority["global_evaluable_type_fraction"], 2.0 / 3.0)
    excluded = {
        value["Level1_type"]: value for value in authority["initial_type_support_diagnostics"]
    }
    assert excluded["T3"]["status"] == "excluded_before_QC_support"
    assert excluded["T3"]["reason"] == "missing_from_one_or_more_compared_banks"
    matched_eligibility = authority["support_eligibility_by_bank"]["A"]
    assert matched_eligibility["outside_common_type_or_QC_support_cells"] == 8
    assert not np.any(np.asarray(plan["adapter_input_expected"])[-8:])
    receipt = runner._adapter_call_receipt(
        plan,
        authority,
        mode="single_bank",
        donor_ids=("A",),
    )
    assert receipt["input_cells"] == matched_eligibility["input_cells"] - 8


def test_global_type_coverage_floor_fails_closed() -> None:
    plan, latent = _with_matched_only_types(("T3", "T4", "T5"))
    with pytest.raises(RuntimeError, match="insufficient naturally supported type coverage"):
        _state_authority(plan, latent)


def test_training_only_natural_state_authority_matches_all_exact_ess_contracts() -> None:
    plan, latent = _synthetic_plan_and_latent()
    authority = _state_authority(plan, latent)
    assert authority["anchor_training_donor_ids"] == ("B", "C")
    assert "A" not in authority["anchor_training_donor_ids"]
    assert authority["evaluable_type_names"] == ("T1", "T2")
    component_types = np.asarray(authority["component_type_names"]).astype(str)
    assert all(np.count_nonzero(component_types == value) in {2, 3} for value in ("T1", "T2"))
    assert all(
        np.isclose(group["ESS"], group["cells"])
        and group["bank_component_separation_ratio"]
        >= runner.FROZEN_STATE_MIN_ANCHOR_SEPARATION_RATIO
        for contract in authority["type_contracts"]
        for group in contract["natural_support"]
    )

    banks = np.asarray(plan["bank_ids"]).astype(str)
    assignment = np.asarray(authority["assignment"], dtype=np.int16)
    single = np.asarray(authority["single_weights"], dtype=np.float64)
    generic = np.asarray(authority["generic_weights"], dtype=np.float64)
    masses = np.asarray(authority["component_masses"], dtype=np.float64)
    targets = np.asarray(authority["component_target_ess"], dtype=np.float64)
    total = float(authority["single_bank_target_ess"])
    for bank in ("A", "B", "C"):
        assert np.isclose(single[banks == bank].sum(), 1.0)
        assert np.isclose(runner._ess(single[banks == bank]), total)
        for component in range(len(masses)):
            rows = (banks == bank) & (assignment == component)
            assert np.isclose(single[rows].sum(), masses[component])
            assert np.isclose(runner._ess(single[rows]), targets[component])
    assert np.isclose(generic.sum(), 1.0)
    assert np.isclose(runner._ess(generic), total)
    for component in range(len(masses)):
        pooled = assignment == component
        assert np.isclose(generic[pooled].sum(), masses[component])
        assert np.isclose(runner._ess(generic[pooled]), targets[component])
        for bank in ("B", "C"):
            rows = (banks == bank) & pooled
            assert np.isclose(generic[rows].sum(), masses[component] / 2)
            assert np.isclose(runner._ess(generic[rows]), targets[component] / 2)


def test_anchor_geometry_and_weights_are_row_and_cell_id_invariant() -> None:
    plan, latent = _synthetic_plan_and_latent()
    first = _state_authority(plan, latent)
    rng = np.random.default_rng(91)
    permutation = rng.permutation(len(latent))
    permuted_plan = {}
    for name, value in plan.items():
        array = np.asarray(value).copy()
        if array.shape == (len(latent),):
            array = array[permutation]
        permuted_plan[name] = array
    permuted_plan["cell_ids"] = np.asarray([f"renamed-{index:04d}" for index in range(len(latent))])
    second = _state_authority(permuted_plan, latent[permutation])
    assert first["state_anchor_sha256"] == second["state_anchor_sha256"]
    assert np.allclose(first["component_anchors"], second["component_anchors"])
    assert np.allclose(first["component_masses"], second["component_masses"])
    assert np.allclose(first["component_target_ess"], second["component_target_ess"])
    inverse = np.argsort(permutation)
    assert np.allclose(
        np.asarray(second["single_weights"])[inverse],
        np.asarray(first["single_weights"]),
    )
    assert np.allclose(
        np.asarray(second["generic_weights"])[inverse],
        np.asarray(first["generic_weights"]),
    )


def test_matched_reference_does_not_change_training_anchor_geometry() -> None:
    plan, latent = _synthetic_plan_and_latent()
    first = _state_authority(plan, latent)
    banks = np.asarray(plan["bank_ids"]).astype(str)
    altered = latent.copy()
    matched_rows = np.flatnonzero(banks == "A")
    altered[matched_rows, 7] += np.linspace(-0.05, 0.05, len(matched_rows))
    second = _state_authority(plan, altered)
    assert first["state_anchor_sha256"] == second["state_anchor_sha256"]
    assert first["anchor_training_weight_sha256"] == second["anchor_training_weight_sha256"]
    assert first["reference_latent_sha256"] != second["reference_latent_sha256"]


def test_missing_matched_states_are_not_manufactured() -> None:
    plan, latent = _synthetic_plan_and_latent()
    banks = np.asarray(plan["bank_ids"]).astype(str)
    collapsed = latent.copy()
    collapsed[banks == "A", 0] = 8.0
    collapsed[banks == "A", 1] = 0.0
    with pytest.raises(RuntimeError, match="no evaluable naturally supported state type"):
        _state_authority(plan, collapsed)


def test_largest_all_bank_estimable_K_reports_matched_support_coarsening() -> None:
    plan, latent = _synthetic_plan_and_latent()
    cells = np.asarray(plan["cell_ids"]).astype(str)
    merged = latent.copy()
    for type_name in ("T1", "T2"):
        for high in (0, 1):
            for replicate in range(8):
                destination = np.flatnonzero(cells == f"A-{type_name}-s1-q{high}-r{replicate:02d}")[
                    0
                ]
                source = np.flatnonzero(cells == f"A-{type_name}-s0-q{high}-r{replicate:02d}")[0]
                merged[destination] = merged[source]
    authority = _state_authority(plan, merged)
    for contract in authority["type_contracts"]:
        assert contract["component_count"] == 2
        assert contract["component_count_candidates"] == [
            {
                "K": 3,
                "status": "ineligible",
                "reason": "natural_support_failed::A::1",
            },
            {"K": 2, "status": "all_bank_estimable", "reason": "eligible"},
        ]


def test_adapter_uses_only_natural_components_and_exact_weights() -> None:
    plan, _, adapter, mixtures = _adapter_fixture()
    assert adapter.state_authority is not None
    authority = adapter.authority_receipt()
    assert authority["schema"] == runner.AUTHORITY_SCHEMA
    assert authority["heldout_donor_excluded_from_anchor_fit"] is True
    assert authority["anchor_training_donor_ids"] == ["B", "C"]
    assert all(value["component_count"] in {2, 3} for value in authority["type_diagnostics"])
    assert len(adapter.calls) == 4
    assert {value["mode"] for value in adapter.calls} == {
        "single_bank",
        "generic_donor_equal",
    }
    assert all(
        value["minimum_state_component_contributing_ESS"] >= runner.FROZEN_STATE_SUPPORT_ESS
        for value in adapter.calls
    )
    assert len({value["component_count"] for value in adapter.calls}) == 1
    assert all(value["model_component_richness_matched"] is True for value in adapter.calls)
    generic_call = next(value for value in adapter.calls if value["mode"] == "generic_donor_equal")
    assert {value["donor"] for value in generic_call["model_state_groups"]} == {
        "generic_donor_equal"
    }
    assert generic_call["minimum_model_component_ESS"] >= runner.FROZEN_STATE_SUPPORT_ESS
    assert len(mixtures["generic"].weights) == len(mixtures["A"].weights)
    assert set(mixtures["generic"].donor_ids.astype(str)) == {"generic_donor_equal"}
    assert np.allclose(mixtures["generic"].weights, mixtures["A"].weights)
    assert set(authority["support_eligibility_by_bank"]) == {"A", "B", "C"}
    assert all(
        value["retention_fraction_of_input"] >= runner.FROZEN_STATE_MIN_GLOBAL_CELL_COVERAGE
        for value in authority["support_eligibility_by_bank"].values()
    )
    assert authority["all_within_support_weights_positive"] is True
    assert not bool(plan["hard_subsampling_used"])


def test_adapter_rejects_latent_different_from_captured_training_encoder() -> None:
    plan, latent = _synthetic_plan_and_latent()
    core = runner._load_baseline_runner()._import_core()
    adapter = runner._WeightedReferenceAdapter(core, plan, lambda: latent)
    banks = np.asarray(plan["bank_ids"]).astype(str)
    types = np.asarray(plan["type_ids"]).astype(str)
    cells = np.asarray(plan["cell_ids"]).astype(str)
    keep = banks == "A"
    altered = latent[keep].copy()
    altered[0, 0] += 1.0e-3
    with pytest.raises(ValueError, match="captured frozen encoder latent"):
        adapter(
            altered,
            donor_ids=banks[keep],
            type_ids=types[keep],
            observation_ids=cells[keep],
            source_modality="single_cell",
            n_components=3,
            seed=runner.FROZEN_BASE_SEED,
        )


def test_prediction_validator_binds_natural_state_contract() -> None:
    plan, predictions, baseline, public = _prediction_fixture()
    runner._validate_fixed_prediction(
        baseline=baseline,
        predictions=predictions,
        public=public,
        plan=plan,
        donor="A",
        epochs=80,
    )


@pytest.mark.parametrize(
    "corrupt,match",
    [
        (
            lambda prediction: prediction["fixed_reference_state_anchors"].__setitem__(
                (0, 0), prediction["fixed_reference_state_anchors"][0, 0] + 1.0
            ),
            "recomputed state array",
        ),
        (
            lambda prediction: prediction["fixed_reference_single_bank_cell_weights"].__setitem__(
                np.flatnonzero(prediction["fixed_reference_single_bank_cell_weights"] > 0)[0],
                0.0,
            ),
            "joint state support|component ESS/mass",
        ),
        (
            lambda prediction: prediction["fixed_reference_state_component_target_ESS"].__setitem__(
                0, prediction["fixed_reference_state_component_target_ESS"][0] + 1.0
            ),
            "recomputed state array",
        ),
    ],
)
def test_prediction_validator_rejects_array_corruption(corrupt, match) -> None:
    plan, original, baseline, public = _prediction_fixture()
    predictions = {name: np.asarray(value).copy() for name, value in original.items()}
    corrupt(predictions)
    with pytest.raises(ValueError, match=match):
        runner._validate_fixed_prediction(
            baseline=baseline,
            predictions=predictions,
            public=public,
            plan=plan,
            donor="A",
            epochs=80,
        )


def test_prediction_validator_rejects_target_donor_in_anchor_receipt() -> None:
    plan, predictions, baseline, public = _prediction_fixture()
    authority = json.loads(runner._scalar_text(predictions["fixed_reference_state_authority_json"]))
    authority["anchor_training_donor_ids"] = ["A", "B", "C"]
    predictions["fixed_reference_state_authority_json"] = np.asarray(json.dumps(authority))
    with pytest.raises(ValueError, match="recomputed state receipt"):
        runner._validate_fixed_prediction(
            baseline=baseline,
            predictions=predictions,
            public=public,
            plan=plan,
            donor="A",
            epochs=80,
        )


def test_prediction_validator_rejects_plausible_false_natural_support_receipt() -> None:
    plan, predictions, baseline, public = _prediction_fixture()
    authority = json.loads(runner._scalar_text(predictions["fixed_reference_state_authority_json"]))
    authority["type_diagnostics"][0]["natural_support"][0]["cells"] += 1
    predictions["fixed_reference_state_authority_json"] = np.asarray(json.dumps(authority))
    with pytest.raises(ValueError, match="recomputed state receipt"):
        runner._validate_fixed_prediction(
            baseline=baseline,
            predictions=predictions,
            public=public,
            plan=plan,
            donor="A",
            epochs=80,
        )


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "empty_weight_fits"))
def test_prediction_validator_rejects_incomplete_or_duplicate_state_receipts(
    mutation: str,
) -> None:
    plan, predictions, baseline, public = _prediction_fixture()
    authority = json.loads(runner._scalar_text(predictions["fixed_reference_state_authority_json"]))
    if mutation == "missing":
        authority["type_diagnostics"][0]["natural_support"].pop()
    elif mutation == "duplicate":
        authority["type_diagnostics"][0]["natural_support"].append(
            authority["type_diagnostics"][0]["natural_support"][0]
        )
    else:
        authority["component_diagnostics"][0]["weight_fits"] = []
    predictions["fixed_reference_state_authority_json"] = np.asarray(json.dumps(authority))
    with pytest.raises(ValueError, match="recomputed state receipt"):
        runner._validate_fixed_prediction(
            baseline=baseline,
            predictions=predictions,
            public=public,
            plan=plan,
            donor="A",
            epochs=80,
        )


def test_prediction_validator_rejects_generic_QC_swap_that_preserves_mass_and_ESS() -> None:
    plan, original, baseline, public = _prediction_fixture()
    predictions = {name: np.asarray(value).copy() for name, value in original.items()}
    banks = np.asarray(plan["bank_ids"]).astype(str)
    categories = np.asarray(plan["category_index"], dtype=np.int64)
    assignment = np.asarray(predictions["fixed_reference_state_assignment"], dtype=np.int16)
    generic = np.asarray(predictions["fixed_reference_generic_cell_weights"], dtype=np.float64)
    component = int(np.min(assignment[assignment >= 0]))
    candidate_categories = sorted(set(categories[(banks == "B") & (assignment == component)]))
    assert len(candidate_categories) >= 2
    left = np.flatnonzero(
        (banks == "B") & (assignment == component) & (categories == candidate_categories[0])
    )
    right = np.flatnonzero(
        (banks == "B") & (assignment == component) & (categories == candidate_categories[1])
    )
    pairs = [
        (first, second)
        for first in left
        for second in right
        if not np.isclose(generic[first], generic[second], rtol=0.0, atol=1.0e-15)
    ]
    assert pairs
    first, second = pairs[0]
    before_mass = generic.sum()
    before_ess = runner._ess(generic)
    generic[first], generic[second] = generic[second], generic[first]
    assert np.isclose(generic.sum(), before_mass)
    assert np.isclose(runner._ess(generic), before_ess)
    predictions["fixed_reference_generic_cell_weights"] = generic
    with pytest.raises(ValueError, match="recomputed state array"):
        runner._validate_fixed_prediction(
            baseline=baseline,
            predictions=predictions,
            public=public,
            plan=plan,
            donor="A",
            epochs=80,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "input_count",
        "retained_count",
        "component_cells",
        "component_mass",
        "component_ess",
        "duplicate_component",
        "minimum_ess",
        "model_component_mass",
    ),
)
def test_prediction_validator_rejects_any_adapter_call_receipt_corruption(
    mutation: str,
) -> None:
    plan, predictions, baseline, public = _prediction_fixture()
    calls = json.loads(runner._scalar_text(predictions["fixed_reference_adapter_calls_json"]))
    call = calls[-1]
    if mutation == "input_count":
        call["input_cells"] = 0
    elif mutation == "retained_count":
        call["retained_cells"] = 0
    elif mutation == "component_cells":
        call["state_groups"][0]["components"][0]["cells"] += 1
    elif mutation == "component_mass":
        call["state_groups"][0]["components"][0]["mass"] = -1.0
    elif mutation == "component_ess":
        call["state_groups"][0]["components"][0]["ESS"] = -1.0
    elif mutation == "duplicate_component":
        call["state_groups"][0]["components"][1]["component_index"] = call["state_groups"][0][
            "components"
        ][0]["component_index"]
    elif mutation == "minimum_ess":
        call["minimum_state_component_contributing_ESS"] = -999.0
    else:
        call["model_state_groups"][0]["components"][0]["mass"] = -1.0
    predictions["fixed_reference_adapter_calls_json"] = np.asarray(json.dumps(calls))
    with pytest.raises(ValueError, match="exact adapter call receipts"):
        runner._validate_fixed_prediction(
            baseline=baseline,
            predictions=predictions,
            public=public,
            plan=plan,
            donor="A",
            epochs=80,
        )


def test_protocol_freezes_scientific_and_target_boundary(tmp_path: Path) -> None:
    path = Path(__file__).parents[1] / "configs/natcommun_fixed_ess_reference_sensitivity.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    runner._validate_protocol(path, protocol)
    assert protocol["schema"] == "heir.natcommun_fixed_ess_reference_protocol.v3"
    assert protocol["model_freeze"]["image_encoder"] == "bioptimus/H-optimus-1"
    assert protocol["model_freeze"]["UNI2_h"] == "prohibited_not_run"
    state = protocol["reference_support"]["state_diversity"]
    assert "query_excluded" in state["anchor_fit"]
    assert "hard_nearest" in state["natural_assignment"]
    assert "does_not_force_absent_states" in state["scope_boundary"]
    assert state["natural_support"]["minimum_ESS_per_bank_type_component"] == 4.0
    assert state["proximity_tilt"]["minimum_relative_positive_weight"] == 0.0001
    assert state["hard_subsampling"] is False
    assert "largest_all_bank_estimable" in state["component_count"]
    assert state["minimum_global_cell_coverage_fraction_per_bank"] == 0.5
    assert state["minimum_global_evaluable_type_fraction_over_all_observed_types"] == 0.5
    assert "model_component_richness" in state
    assert state["all_within_support_weights_positive"] is True
    assert "Sinkhorn" not in json.dumps(protocol)
    assert protocol["target_boundary"]["global_prediction_validation_before_any_target"]
    changed = json.loads(json.dumps(protocol))
    changed["reference_support"]["state_diversity"]["maximum_components_per_type"] = 4
    changed_path = tmp_path / "changed_protocol.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="fully frozen"):
        runner._validate_protocol(changed_path, changed)


def test_prepared_manifest_rejects_score_target_authority(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    manifest = {
        "schema": runner.PREPARED_SCHEMA,
        "runner_sha256": runner._sha256(Path(runner.__file__).resolve()),
        "image_encoder": runner.HOPTIMUS_REPOSITORY,
        "uni2_h_run": False,
        "base_seed": runner.FROZEN_BASE_SEED,
        "protocol": str(protocol.resolve()),
        "protocol_sha256": runner._sha256(protocol),
        "folds": {"A": {"score_target_path": "/forbidden/target.npz"}},
    }
    (output / "prepared_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="contains score-target authority"):
        runner._read_prepared(output, protocol)


def test_gate3_precision_promotion_preserves_frozen_decision() -> None:
    composition = np.asarray([[0.6, 0.2999999836705683, 0.1000000163294317]])
    predictions = {
        "query_H_composition": composition,
        "gate3_supported_type_mask": np.asarray([True, True, False]),
        "matched_reference_component_count_by_type": np.asarray([3, 3, 0]),
        "gate3_supported_composition_mass": np.asarray(
            [composition[0, :2].sum()], dtype=np.float32
        ),
        "gate3_supported_score_eligible": np.asarray([False]),
    }
    runner._promote_gate3_support_receipt_precision(predictions)
    assert predictions["gate3_supported_composition_mass"].dtype == np.float64
    assert not bool(predictions["gate3_supported_score_eligible"][0])
    assert int(predictions["fixed_reference_gate3_threshold_rounding_rows"]) == 1


def test_aligned_latent_capture_seals_training_and_matched_roles() -> None:
    public = {
        "train_st_counts": np.zeros((2, 3)),
        "train_sc_counts": np.zeros((3, 3)),
        "matched_sc_counts": np.zeros((2, 3)),
        "train_sc_cell_ids": np.asarray(["wrong-0", "unused", "wrong-2"]),
        "matched_sc_cell_ids": np.asarray(["matched-0", "matched-1"]),
        "wrong_train_sc_index": np.asarray([0, 2]),
    }
    plan = {"cell_ids": np.asarray(["matched-0", "matched-1", "wrong-0", "wrong-2"])}
    capture = runner._AlignedLatentCapture(public, plan)
    ordinal = 0

    def original(module, counts, *, modality, device):
        nonlocal ordinal
        ordinal += 1
        return np.repeat(
            np.asarray([[float(ordinal)] * runner.FROZEN_LATENT_DIM]),
            len(counts),
            axis=0,
        )

    wrapped = capture.wrap(original)
    for counts, modality in (
        (public["train_st_counts"], "st"),
        (public["train_sc_counts"], "scrna"),
        (public["matched_sc_counts"], "scrna"),
        (public["train_st_counts"], "st"),
        (public["train_sc_counts"], "scrna"),
    ):
        wrapped(object(), counts, modality=modality, device="cpu")
    observed = capture.plan_latent()
    assert np.all(observed[:2] == 3.0)
    assert np.all(observed[2:] == 2.0)
    assert capture.receipt()["score_target_opened"] is False
    with pytest.raises(RuntimeError, match="extra calls"):
        wrapped(object(), public["train_sc_counts"], modality="scrna", device="cpu")
