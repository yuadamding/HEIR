from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _script():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_hest_scientific_reanalysis.py"
    spec = importlib.util.spec_from_file_location("benchmark_hest_scientific_reanalysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _script()


def _visible_controls(
    *,
    failed_morphology: tuple[str, ...] = (),
    failed_classification: tuple[str, ...] = (),
):
    morphology_names = (
        "nucleus_area_um2",
        "nucleus_perimeter_um",
        "nucleus_circularity",
        "nucleus_solidity",
        "nucleus_gray_mean",
        "nucleus_hematoxylin_od_mean",
        "nucleus_glcm_contrast",
    )
    morphology_targets = {
        name: {
            "donor_type_macro_reference_error_reduction": (
                -0.01 if name in failed_morphology else 0.1
            )
        }
        for name in morphology_names
    }
    controls = {
        target: {
            "model": {
                "donor_balanced": {
                    "donor_macro_balanced_accuracy": (
                        baseline if target in failed_classification else model
                    )
                }
            },
            "training_majority_baseline": {
                "donor_balanced": {"donor_macro_balanced_accuracy": baseline}
            },
        }
        for target, model, baseline in (
            ("broad_lineage", 0.4, 0.25),
            ("fine_type", 0.1, 0.03),
        )
    }
    controls["nuclear_morphology"] = {
        "full_context": {"scores": {"targets": morphology_targets}},
        "nucleus_mask_only": {"scores": {"targets": {}}},
    }
    return controls


def _visible_control_gate(
    *,
    failed_morphology: tuple[str, ...] = (),
    failed_classification: tuple[str, ...] = (),
):
    return RUNNER.positive_control_gate(
        _visible_controls(
            failed_morphology=failed_morphology,
            failed_classification=failed_classification,
        )
    )


def test_alpha_selection_is_independent_by_target() -> None:
    scores = np.asarray(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.7, 0.2], [0.1, 0.9]],
        ]
    )
    indices, selected = RUNNER._selected_alpha_indices(scores)
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(selected, [0.8, 0.85])


def test_best_control_is_selected_inside_outer_training_fold_per_target() -> None:
    rng = np.random.default_rng(17)
    donors = np.repeat(["D0", "D1", "D2", "D3", "D4"], 40)
    sections = np.repeat(["S0", "S1", "S2", "S3", "S4"], 40)
    fine_types = np.repeat("T", len(donors))
    broad_types = np.repeat("L", len(donors))
    first = rng.normal(size=len(donors))
    second = rng.normal(size=len(donors))
    targets = np.column_stack(
        (
            first + rng.normal(scale=0.02, size=len(first)),
            second + rng.normal(scale=0.02, size=len(second)),
        )
    )
    controls = {"first": first[:, None], "second": second[:, None]}

    plan = RUNNER._fit_control_plan(
        targets,
        controls,
        donors,
        sections,
        fine_types,
        broad_types,
        ("target_a", "target_b"),
        architecture="pooled",
        weighting="donor_type",
        inner_folds=2,
        seed=4,
        device="cpu",
    )

    for heldout, fold in plan.outer_folds.items():
        assert heldout not in fold["training_donors"]
        assert fold["selected_control_family_by_target"] == {
            "target_a": "first",
            "target_b": "second",
        }
        assert fold["heldout_outcomes_used_for_selection"] is False
    assert np.isfinite(plan.predictions["selected_best_control"]).all()


def test_intrinsic_claim_fails_closed_without_matched_artifact_crops() -> None:
    source = SimpleNamespace(crop_ids=RUNNER.REQUIRED_BASE_CROPS)
    boundary = RUNNER._artifact_boundary(source)
    assert boundary["current_source_sufficient_for_h_intrinsic"] is False
    assert boundary["direct_biological_contrast_status"] == ("blocked_requires_new_crop_embeddings")
    assert set(boundary["missing"]) == set(RUNNER.ARTIFACT_CONTROL_CROPS)


def test_heldout_outcome_cannot_change_its_nested_fit() -> None:
    rng = np.random.default_rng(91)
    donors = np.repeat(["D0", "D1", "D2", "D3", "D4"], 40)
    sections = donors.copy()
    fine_types = np.repeat("T", len(donors))
    broad_types = np.repeat("L", len(donors))
    signal = rng.normal(size=len(donors))
    targets = signal[:, None] + rng.normal(scale=0.05, size=(len(donors), 1))
    controls = {
        "signal": signal[:, None],
        "noise": rng.normal(size=(len(donors), 1)),
    }

    first = RUNNER._fit_control_plan(
        targets,
        controls,
        donors,
        sections,
        fine_types,
        broad_types,
        ("target",),
        architecture="pooled",
        weighting="donor_type",
        inner_folds=2,
        seed=12,
        device="cpu",
    )
    perturbed = targets.copy()
    perturbed[donors == "D0"] += 1_000_000.0
    second = RUNNER._fit_control_plan(
        perturbed,
        controls,
        donors,
        sections,
        fine_types,
        broad_types,
        ("target",),
        architecture="pooled",
        weighting="donor_type",
        inner_folds=2,
        seed=12,
        device="cpu",
    )

    assert first.outer_folds["D0"] == second.outer_folds["D0"]
    np.testing.assert_allclose(
        first.predictions["selected_best_control"][donors == "D0"],
        second.predictions["selected_best_control"][donors == "D0"],
    )


def test_broad_lineage_heads_route_rows_to_distinct_relationships() -> None:
    x = np.asarray([-2.0, -1.0, 1.0, 2.0, -2.0, -1.0, 1.0, 2.0, -1.5, 1.5, -1.5, 1.5])[:, None]
    broad = np.asarray(["A"] * 4 + ["B"] * 4 + ["A"] * 2 + ["B"] * 2)
    donors = np.asarray(["A0", "A0", "A1", "A1", "B0", "B0", "B1", "B1"] + ["H0", "H0", "H0", "H0"])
    sections = donors.copy()
    fine_types = broad.copy()
    train = np.arange(8)
    test = np.arange(8, 12)
    y = np.where(broad[:8] == "A", x[:8, 0], -x[:8, 0])[:, None]

    prediction = RUNNER._architecture_predict_grid(
        x[train],
        y,
        x[test],
        train,
        test,
        donors,
        sections,
        fine_types,
        broad,
        architecture="broad_lineage_heads",
        weighting="donor_type",
        alphas=(0.01,),
        device="cpu",
    )[0, :, 0]

    expected = np.asarray([-1.5, 1.5, 1.5, -1.5])
    assert np.mean(np.square(prediction - expected)) < 0.05


def test_score_models_emits_only_the_fitted_support_threshold() -> None:
    truth = np.asarray([-1.0, 1.0, -2.0, 2.0])[:, None]
    identities = np.asarray(["D0", "D0", "D1", "D1"])
    scores = RUNNER._score_models(
        truth,
        {"model": truth.copy()},
        identities,
        identities,
        np.repeat("T", 4),
        ("target",),
        minimum_support=2,
    )
    assert list(scores["model"]) == ["2"]
    assert scores["model"]["2"]["rows"] == 4


def test_crop_arms_share_control_plan_and_inner_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SimpleNamespace(
        nucleus_targets=np.zeros((4, 1)),
        roles=np.repeat("evaluation", 4),
        images=np.stack(
            [np.full((4, 1), value, dtype=np.float32) for value in range(4)],
            axis=1,
        ),
        crop_ids=RUNNER.REQUIRED_BASE_CROPS,
        program_names=("program",),
    )
    donors = np.asarray(["D0", "D0", "D1", "D1"])
    sections = donors.copy()
    fine_types = np.repeat("T", 4)
    broad_types = np.repeat("L", 4)
    shared_control_plan = object()
    image_calls = []

    monkeypatch.setattr(
        RUNNER,
        "_program_outcome",
        lambda *_args, **_kwargs: (np.zeros((4, 1)), np.ones(4, dtype=bool)),
    )
    monkeypatch.setattr(
        RUNNER,
        "_subset_eval",
        lambda *_args: (
            donors,
            sections,
            fine_types,
            broad_types,
            {"control": np.zeros((4, 1))},
        ),
    )
    monkeypatch.setattr(
        RUNNER,
        "_fit_control_plan",
        lambda *_args, **_kwargs: shared_control_plan,
    )

    def fake_image_fit(raw_image, *_args, **kwargs):
        control_plan = _args[3]
        marker = float(raw_image[0, 0])
        image_calls.append((marker, control_plan, kwargs["seed"]))
        per_donor = {donor: {"donor_type_r2": marker} for donor in sorted(set(donors.tolist()))}
        score = {"targets": {"program": {"per_donor": per_donor}}}
        return {"image_models": {"selected_best_control_plus_foundation_image": {"20": score}}}

    monkeypatch.setattr(RUNNER, "_fit_image_plan", fake_image_fit)
    report = RUNNER.crop_sensitivity_analysis(
        source,
        device="cpu",
        inner_folds=2,
        seed=31,
    )

    assert sorted(marker for marker, _plan, _seed in image_calls) == [0.0, 1.0, 2.0, 3.0]
    assert all(plan is shared_control_plan for _marker, plan, _seed in image_calls)
    assert len({seed for _marker, _plan, seed in image_calls}) == 1
    assert report["matched_tuning_contract"]["primary_observed_arm_reused"] is False


def test_base_report_is_strict_json_and_permanently_non_authorizing() -> None:
    source = SimpleNamespace(
        path=Path("/tmp/source.npz"),
        sha256="source-hash",
        encoder_name=RUNNER.UNI2H_REPOSITORY,
        encoder_revision="d517a8dd47902dd7c308b3c36f63bce47e7b9a43",
        encoder_manifest_sha256="manifest-hash",
        encoder_parity_receipt_sha256="",
        encoder_parity_receipt_path="",
        encoder_comparison_source_sha256="",
        encoder_comparison_non_encoder_identity_sha256="",
        encoder_comparison_receipt=None,
        donors=np.asarray(["D0", "D1"]),
        sections=np.asarray(["S0", "S1"]),
        crop_ids=RUNNER.REQUIRED_BASE_CROPS,
        crop_roles=("context", "cell", "nucleus", "removed"),
        crop_mask_modes=("none", "cell", "nucleus", "removed"),
        crop_fill_modes=("none", "white", "white", "white"),
    )
    report = RUNNER._base_report(
        source,
        seed=1,
        device="cpu",
        inner_folds=3,
        phase="full",
        allow_gate_failed_uni2_baseline_only=False,
    )

    import json

    json.dumps(report, allow_nan=False)
    assert report["schema"] == "heir.hest_scientific_reanalysis.v2"
    assert report["authorizes_h_cell"] is False
    assert report["authorizes_h_intrinsic"] is False
    assert report["authorizes_reference_refinement"] is False
    assert report["authorizes_full_heir"] is False
    assert report["requested_phase"] == "full"
    assert report["allow_gate_failed_uni2_baseline_only"] is False
    assert report["inner_folds"] == 3
    assert report["implementation_receipt"]["file_sha256"]


def test_cuda_determinism_requires_workspace_config_and_disables_nondeterminism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        RUNNER._configure_cuda_determinism()

    calls = []
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        RUNNER.torch,
        "use_deterministic_algorithms",
        lambda enabled: calls.append(enabled),
    )
    monkeypatch.setattr(RUNNER.torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(RUNNER.torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(RUNNER.torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(RUNNER.torch.backends.cudnn, "allow_tf32", True)
    RUNNER._configure_cuda_determinism()
    assert calls == [True]
    assert RUNNER.torch.backends.cudnn.deterministic is True
    assert RUNNER.torch.backends.cudnn.benchmark is False
    assert RUNNER.torch.backends.cuda.matmul.allow_tf32 is False
    assert RUNNER.torch.backends.cudnn.allow_tf32 is False


def test_cuda_numeric_receipt_binds_hardware_and_deterministic_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        RUNNER.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(name="Test GPU", total_memory=24 * 1024**3),
    )
    monkeypatch.setattr(
        RUNNER.torch.cuda, "get_device_capability", lambda _index: (9, 0)
    )
    monkeypatch.setattr(
        RUNNER.torch, "are_deterministic_algorithms_enabled", lambda: True
    )
    monkeypatch.setattr(RUNNER.torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(RUNNER.torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(RUNNER.torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(RUNNER.torch.backends.cudnn, "allow_tf32", False)
    receipt = RUNNER._numeric_backend_receipt("cuda")
    assert receipt["gpu_name"] == "Test GPU"
    assert receipt["gpu_capability"] == [9, 0]
    assert receipt["gpu_total_memory_bytes"] == 24 * 1024**3
    assert receipt["deterministic_algorithms_enabled"] is True
    assert receipt["cublas_workspace_config"] == ":4096:8"
    assert receipt["cudnn_deterministic"] is True
    assert receipt["cudnn_benchmark"] is False
    assert receipt["cuda_matmul_allow_tf32"] is False
    assert receipt["cudnn_allow_tf32"] is False


def test_paired_encoder_report_aligns_exact_donors_and_registered_comparator(
    tmp_path: Path,
) -> None:
    primary_donors = {donor: 0.10 for donor in RUNNER.EXPECTED_DONORS}
    comparator_donors = {donor: 0.05 for donor in RUNNER.EXPECTED_DONORS}
    primary_donors[RUNNER.EXPECTED_DONORS[0]] = 0.20
    comparator_donors[RUNNER.EXPECTED_DONORS[1]] = 0.15
    primary_effects = {
        program: {"per_donor_effect": primary_donors}
        for program in RUNNER.FROZEN_PROGRAM_NAMES
    }
    comparator_effects = {
        program: {"per_donor_effect": comparator_donors}
        for program in RUNNER.FROZEN_PROGRAM_NAMES
    }
    primary_representations = {
        representation: {"central_increment": primary_effects}
        for representation in RUNNER.REQUIRED_PAIRED_ENCODER_REPRESENTATIONS
    }
    comparator_representations = {
        representation: {"central_increment": comparator_effects}
        for representation in RUNNER.REQUIRED_PAIRED_ENCODER_REPRESENTATIONS
    }
    implementation_receipt = {
        "file_sha256": {"runner.py": "same"},
        "python": "3.12.test",
        "platform": "Linux-test",
        "numpy": "2.0.test",
        "torch": "2.0.test",
        "torch_cuda_runtime": "12.test",
    }
    primary = {
        "encoder": RUNNER.HOPTIMUS1_REPOSITORY,
        "allow_gate_failed_uni2_baseline_only": False,
        "encoder_comparison_receipt": {"only_encoder_changed": True},
        "positive_controls": _visible_controls(),
        "positive_control_gate": _visible_control_gate(),
        "implementation_receipt": json.loads(json.dumps(implementation_receipt)),
        "numeric_backend": {
            "requested_device": "cuda",
            "torch_threads": 2,
            "cuda_available": True,
            "ridge_cuda_dtype": "float32",
            "cpu_thread_environment": {"OMP_NUM_THREADS": "2"},
            "gpu_name": "test-gpu",
            "gpu_capability": [9, 0],
            "gpu_total_memory_bytes": 1024,
            "deterministic_algorithms_enabled": True,
            "cublas_workspace_config": ":4096:8",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
        "measurement": {"shared_outcomes": True},
        "observed_nucleus_programs": {
            "target_scope": "nucleus_overlap",
            "endpoint": "residual_program_score",
            "evaluation_rows": 15,
            "evaluation_mask_sha256": "1" * 64,
            "program_outcome_sha256": "2" * 64,
            "minimum_reference_and_scoring_support": RUNNER.PRIMARY_SUPPORT,
            "primary_crop": "crop_112um",
            "program_names": list(RUNNER.FROZEN_PROGRAM_NAMES),
            "donors": list(RUNNER.EXPECTED_DONORS),
            "representations": primary_representations,
        },
    }
    comparator = {
        "encoder": RUNNER.UNI2H_REPOSITORY,
        "source_sha256": RUNNER.REGISTERED_SOURCE_SHA256,
        "execution_status": "scientific_reanalysis_complete",
        "positive_controls": _visible_controls(),
        "positive_control_gate": _visible_control_gate(),
        "implementation_receipt": json.loads(json.dumps(implementation_receipt)),
        "numeric_backend": {
            "requested_device": "cuda",
            "torch_threads": 2,
            "cuda_available": True,
            "ridge_cuda_dtype": "float32",
            "cpu_thread_environment": {"OMP_NUM_THREADS": "2"},
            "gpu_name": "test-gpu",
            "gpu_capability": [9, 0],
            "gpu_total_memory_bytes": 1024,
            "deterministic_algorithms_enabled": True,
            "cublas_workspace_config": ":4096:8",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
        "measurement": {"shared_outcomes": True},
        "observed_nucleus_programs": {
            "target_scope": "nucleus_overlap",
            "endpoint": "residual_program_score",
            "evaluation_rows": 15,
            "evaluation_mask_sha256": "1" * 64,
            "program_outcome_sha256": "2" * 64,
            "minimum_reference_and_scoring_support": RUNNER.PRIMARY_SUPPORT,
            "primary_crop": "crop_112um",
            "program_names": list(RUNNER.FROZEN_PROGRAM_NAMES),
            "donors": list(RUNNER.EXPECTED_DONORS),
            "representations": comparator_representations,
        },
    }
    primary["implementation_receipt"]["command"] = [
        "scripts/benchmark_hest_scientific_reanalysis.py",
        "--phase",
        "full",
        "--expected-encoder",
        RUNNER.HOPTIMUS1_REPOSITORY,
    ]
    comparator["implementation_receipt"]["command"] = [
        "scripts/benchmark_hest_scientific_reanalysis.py",
        "--phase",
        "full",
        "--expected-encoder",
        RUNNER.UNI2H_REPOSITORY,
    ]
    shared_contract = {
        "schema": RUNNER.SCHEMA,
        "analysis_status": "retrospective_exposed_non_authorizing",
        "study_stage": "retrospective_exposed",
        "requested_phase": "full",
        "donors": list(RUNNER.EXPECTED_DONORS),
        "sections": list(RUNNER.EXPECTED_SECTIONS),
        "encoder_feature_width": 1536,
        "crop_contract": {"crop_ids": ["crop_112um"]},
        "folding": "leave_one_donor_out",
        "inner_folding": "grouped_training_donors",
        "inner_folds": 3,
        "alpha_grid": list(RUNNER.ALPHAS),
        "target_standardization": "training_fold_only",
        "pca_fit": "training_fold_only",
        "training_weighting": "equal_donor_then_type_then_cell",
        "primary_support": RUNNER.PRIMARY_SUPPORT,
        "support_sensitivities": list(RUNNER.SUPPORT_THRESHOLDS),
        "seed": 20260713,
    }
    primary.update(shared_contract)
    comparator.update(shared_contract)
    comparator_path = tmp_path / "uni2h-report.json"
    comparator_path.write_text(json.dumps(comparator), encoding="utf-8")
    preflight = RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)
    primary["same_runner_uni2_comparator_preflight"] = preflight
    report = RUNNER.paired_encoder_comparison_report(primary, comparator_path)
    assert tuple(report["representations"]) == (
        RUNNER.REQUIRED_PAIRED_ENCODER_REPRESENTATIONS
    )
    assert all(
        tuple(arm["programs"]) == RUNNER.FROZEN_PROGRAM_NAMES
        for arm in report["representations"].values()
    )
    program = report["representations"]["full_1536_broad_lineage_heads"]["programs"][
        RUNNER.FROZEN_PROGRAM_NAMES[0]
    ]
    assert program["per_donor_delta"][RUNNER.EXPECTED_DONORS[0]] == pytest.approx(0.15)
    assert program["per_donor_delta"][RUNNER.EXPECTED_DONORS[1]] == pytest.approx(-0.05)
    assert program["mean_delta"] == pytest.approx(0.05)
    assert program["Hoptimus1_better_fraction"] == pytest.approx(14 / 15)
    assert preflight["passed"] is True
    assert preflight["comparison_report_sha256"] == RUNNER._sha256(comparator_path)
    assert preflight["required_representations"] == list(
        RUNNER.REQUIRED_PAIRED_ENCODER_REPRESENTATIONS
    )

    changed_after_preflight = json.loads(json.dumps(comparator))
    changed_after_preflight["elapsed_seconds"] = 1.0
    comparator_path.write_text(json.dumps(changed_after_preflight), encoding="utf-8")
    with pytest.raises(ValueError, match="bytes changed after preflight"):
        RUNNER.paired_encoder_comparison_report(primary, comparator_path)
    comparator_path.write_text(json.dumps(comparator), encoding="utf-8")

    nucleus_primary = json.loads(json.dumps(primary))
    nucleus_primary["requested_phase"] = "nucleus"
    nucleus_primary["implementation_receipt"]["command"][2] = "nucleus"
    nucleus_comparator = json.loads(json.dumps(comparator))
    nucleus_comparator["requested_phase"] = "nucleus"
    nucleus_comparator["implementation_receipt"]["command"][2] = "nucleus"
    nucleus_comparator["execution_status"] = "observed_nucleus_program_models_complete"
    comparator_path.write_text(json.dumps(nucleus_comparator), encoding="utf-8")
    assert RUNNER.same_runner_uni2_comparator_preflight(
        nucleus_primary, comparator_path
    )["passed"] is True

    baseline_comparator = json.loads(json.dumps(comparator))
    baseline_gate = _visible_control_gate(
        failed_morphology=RUNNER.UNI2_BASELINE_GEOMETRY_TARGETS
    )
    baseline_comparator["positive_controls"] = _visible_controls(
        failed_morphology=RUNNER.UNI2_BASELINE_GEOMETRY_TARGETS
    )
    baseline_comparator["positive_control_gate"] = baseline_gate
    baseline_comparator["allow_gate_failed_uni2_baseline_only"] = True
    baseline_comparator["implementation_receipt"]["command"].append(
        "--allow-gate-failed-uni2-baseline-only"
    )
    baseline_comparator["uni2_baseline_only_eligibility"] = (
        RUNNER.uni2_baseline_only_eligibility(RUNNER.UNI2H_REPOSITORY, baseline_gate)
    )
    baseline_comparator["analysis_status"] = (
        "retrospective_exposed_non_authorizing_baseline"
    )
    baseline_comparator["molecular_analysis_role"] = (
        "retrospective_exposed_non_authorizing_baseline"
    )
    baseline_comparator["comparison_inference_allowed"] = False
    baseline_comparator["descriptive_only"] = True
    comparator_path.write_text(json.dumps(baseline_comparator), encoding="utf-8")
    baseline_preflight = RUNNER.same_runner_uni2_comparator_preflight(
        primary, comparator_path
    )
    assert baseline_preflight["comparator_gate_passed"] is False
    assert baseline_preflight["comparator_baseline_only"] is True
    assert baseline_preflight["comparison_inference_allowed"] is False
    assert baseline_preflight["descriptive_only"] is True
    assert (
        baseline_preflight["baseline_only_amendment_timing"]
        == RUNNER.UNI2_BASELINE_AMENDMENT_TIMING
    )
    baseline_primary = json.loads(json.dumps(primary))
    baseline_primary["same_runner_uni2_comparator_preflight"] = baseline_preflight
    baseline_pair = RUNNER.paired_encoder_comparison_report(
        baseline_primary, comparator_path
    )
    assert baseline_pair["comparator_gate_passed"] is False
    assert baseline_pair["comparator_baseline_only"] is True
    assert baseline_pair["comparison_inference_allowed"] is False

    relabeled_baseline_gate = json.loads(json.dumps(baseline_comparator))
    relabeled_baseline_gate["positive_control_gate"] = _visible_control_gate()
    comparator_path.write_text(json.dumps(relabeled_baseline_gate), encoding="utf-8")
    with pytest.raises(ValueError, match="stored positive-control gate was relabeled"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    missing_opt_in = json.loads(json.dumps(baseline_comparator))
    missing_opt_in["allow_gate_failed_uni2_baseline_only"] = False
    comparator_path.write_text(json.dumps(missing_opt_in), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks the explicit opt-in"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    duplicate_opt_in = json.loads(json.dumps(baseline_comparator))
    duplicate_opt_in["implementation_receipt"]["command"].append(
        "--allow-gate-failed-uni2-baseline-only"
    )
    comparator_path.write_text(json.dumps(duplicate_opt_in), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid UNI2-h baseline-only opt-in count"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    hoptimus_command_opt_in = json.loads(json.dumps(primary))
    hoptimus_command_opt_in["implementation_receipt"]["command"].append(
        "--allow-gate-failed-uni2-baseline-only"
    )
    comparator_path.write_text(json.dumps(comparator), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid UNI2-h baseline-only opt-in count"):
        RUNNER.same_runner_uni2_comparator_preflight(
            hoptimus_command_opt_in, comparator_path
        )

    wrong_encoder_command = json.loads(json.dumps(comparator))
    wrong_encoder_command["implementation_receipt"]["command"][-1] = (
        RUNNER.HOPTIMUS1_REPOSITORY
    )
    comparator_path.write_text(json.dumps(wrong_encoder_command), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent --expected-encoder"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    string_command = json.loads(json.dumps(comparator))
    string_command["implementation_receipt"]["command"] = "not-a-sequence-of-args"
    comparator_path.write_text(json.dumps(string_command), encoding="utf-8")
    with pytest.raises(ValueError, match="command is missing or malformed"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    wrong_phase_command = json.loads(json.dumps(comparator))
    wrong_phase_command["implementation_receipt"]["command"][2] = "nucleus"
    comparator_path.write_text(json.dumps(wrong_phase_command), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent --phase"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    missing_positive_controls = json.loads(json.dumps(comparator))
    missing_positive_controls.pop("positive_controls")
    comparator_path.write_text(json.dumps(missing_positive_controls), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks positive controls"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    incomplete = json.loads(json.dumps(comparator))
    incomplete["observed_nucleus_programs"]["representations"].pop(
        RUNNER.REQUIRED_PAIRED_ENCODER_REPRESENTATIONS[1]
    )
    comparator_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="required full/PCA"):
        RUNNER.paired_encoder_comparison_report(primary, comparator_path)
    with pytest.raises(ValueError, match="required full/PCA"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    incomplete = json.loads(json.dumps(comparator))
    first_representation = RUNNER.REQUIRED_PAIRED_ENCODER_REPRESENTATIONS[0]
    first_program = RUNNER.FROZEN_PROGRAM_NAMES[0]
    incomplete["observed_nucleus_programs"]["representations"][first_representation][
        "central_increment"
    ] = {
        first_program: comparator_effects[first_program],
    }
    comparator_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="complete frozen six-program"):
        RUNNER.paired_encoder_comparison_report(primary, comparator_path)

    malformed = json.loads(json.dumps(comparator))
    malformed["observed_nucleus_programs"]["representations"][first_representation] = None
    comparator_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="representation is malformed"):
        RUNNER.paired_encoder_comparison_report(primary, comparator_path)

    malformed = json.loads(json.dumps(comparator))
    malformed["observed_nucleus_programs"]["representations"][first_representation][
        "central_increment"
    ][first_program] = None
    comparator_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="effect is malformed"):
        RUNNER.paired_encoder_comparison_report(primary, comparator_path)

    incomplete = json.loads(json.dumps(comparator))
    incomplete["observed_nucleus_programs"]["representations"][first_representation][
        "central_increment"
    ][first_program]["per_donor_effect"].pop(RUNNER.EXPECTED_DONORS[-1])
    comparator_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="all 15 frozen donors"):
        RUNNER.paired_encoder_comparison_report(primary, comparator_path)

    mismatched_inner_folds = json.loads(json.dumps(comparator))
    mismatched_inner_folds["inner_folds"] = 2
    comparator_path.write_text(json.dumps(mismatched_inner_folds), encoding="utf-8")
    with pytest.raises(ValueError, match="inner_folds"):
        RUNNER.paired_encoder_comparison_report(primary, comparator_path)

    mismatched_phase = json.loads(json.dumps(comparator))
    mismatched_phase["requested_phase"] = "nucleus"
    mismatched_phase["execution_status"] = "observed_nucleus_program_models_complete"
    comparator_path.write_text(json.dumps(mismatched_phase), encoding="utf-8")
    with pytest.raises(ValueError, match="requested_phase"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    partial = json.loads(json.dumps(comparator))
    partial["execution_status"] = "reference_support_sensitivity_in_progress"
    comparator_path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete for requested phase full"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    blocked_gate = json.loads(json.dumps(comparator))
    blocked_gate["positive_control_gate"]["passed"] = False
    blocked_gate["positive_control_gate"]["molecular_interpretation_allowed"] = False
    blocked_gate["allow_gate_failed_uni2_baseline_only"] = True
    comparator_path.write_text(json.dumps(blocked_gate), encoding="utf-8")
    with pytest.raises(ValueError, match="stored positive-control gate was relabeled"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    tampered_controls = json.loads(json.dumps(comparator))
    tampered_controls["positive_controls"]["fine_type"]["model"]["donor_balanced"][
        "donor_macro_balanced_accuracy"
    ] = 0.0
    comparator_path.write_text(json.dumps(tampered_controls), encoding="utf-8")
    with pytest.raises(ValueError, match="stored positive-control gate was relabeled"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    blocked_primary = json.loads(json.dumps(primary))
    blocked_primary["positive_control_gate"]["passed"] = False
    comparator_path.write_text(json.dumps(comparator), encoding="utf-8")
    with pytest.raises(ValueError, match="H-optimus-1 primary stored.*relabeled"):
        RUNNER.same_runner_uni2_comparator_preflight(blocked_primary, comparator_path)

    different_runtime = json.loads(json.dumps(comparator))
    different_runtime["implementation_receipt"]["torch"] = "different"
    comparator_path.write_text(json.dumps(different_runtime), encoding="utf-8")
    with pytest.raises(ValueError, match="implementation runtimes: torch"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    different_threads = json.loads(json.dumps(comparator))
    different_threads["numeric_backend"]["cpu_thread_environment"][
        "OMP_NUM_THREADS"
    ] = "4"
    comparator_path.write_text(json.dumps(different_threads), encoding="utf-8")
    with pytest.raises(ValueError, match="different numeric backends"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    different_gpu = json.loads(json.dumps(comparator))
    different_gpu["numeric_backend"]["gpu_name"] = "different-gpu"
    comparator_path.write_text(json.dumps(different_gpu), encoding="utf-8")
    with pytest.raises(ValueError, match="different numeric backends"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    nondeterministic = json.loads(json.dumps(comparator))
    nondeterministic["numeric_backend"]["cudnn_benchmark"] = True
    comparator_path.write_text(json.dumps(nondeterministic), encoding="utf-8")
    with pytest.raises(ValueError, match="required deterministic CUDA contract"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    nondeterministic_primary = json.loads(json.dumps(primary))
    nondeterministic_primary["numeric_backend"]["cudnn_benchmark"] = True
    nondeterministic_pair = json.loads(json.dumps(comparator))
    nondeterministic_pair["numeric_backend"]["cudnn_benchmark"] = True
    comparator_path.write_text(json.dumps(nondeterministic_pair), encoding="utf-8")
    with pytest.raises(ValueError, match="required deterministic CUDA contract"):
        RUNNER.same_runner_uni2_comparator_preflight(
            nondeterministic_primary, comparator_path
        )

    invalid_hash = json.loads(json.dumps(comparator))
    invalid_hash["observed_nucleus_programs"]["program_outcome_sha256"] = "not-a-sha"
    comparator_path.write_text(json.dumps(invalid_hash), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid program_outcome_sha256"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    invalid_support = json.loads(json.dumps(comparator))
    invalid_support["observed_nucleus_programs"][
        "minimum_reference_and_scoring_support"
    ] = 0
    comparator_path.write_text(json.dumps(invalid_support), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid minimum_reference"):
        RUNNER.same_runner_uni2_comparator_preflight(primary, comparator_path)

    mismatched_population = json.loads(json.dumps(comparator))
    mismatched_population["observed_nucleus_programs"]["evaluation_mask_sha256"] = "3" * 64
    comparator_path.write_text(json.dumps(mismatched_population), encoding="utf-8")
    mismatched_preflight = RUNNER.same_runner_uni2_comparator_preflight(
        primary, comparator_path
    )
    mismatched_primary = json.loads(json.dumps(primary))
    mismatched_primary["same_runner_uni2_comparator_preflight"] = mismatched_preflight
    with pytest.raises(ValueError, match="observed-program analysis populations"):
        RUNNER.paired_encoder_comparison_report(mismatched_primary, comparator_path)

    comparator_path.write_text(json.dumps(comparator), encoding="utf-8")
    comparator["implementation_receipt"]["file_sha256"] = {"runner.py": "different"}
    comparator_path.write_text(json.dumps(comparator), encoding="utf-8")
    with pytest.raises(ValueError, match="same frozen implementation"):
        RUNNER.paired_encoder_comparison_report(primary, comparator_path)
    comparator["implementation_receipt"]["file_sha256"] = {"runner.py": "same"}

    comparator["source_sha256"] = "0" * 64
    comparator_path.write_text(json.dumps(comparator), encoding="utf-8")
    with pytest.raises(ValueError, match="registered UNI2-h"):
        RUNNER.paired_encoder_comparison_report(primary, comparator_path)


def test_positive_control_gate_is_frozen_to_natural_unmasked_visible_signal() -> None:
    morphology_targets = {
        name: {"donor_type_macro_reference_error_reduction": 0.1}
        for name in (
            "nucleus_area_um2",
            "nucleus_perimeter_um",
            "nucleus_circularity",
            "nucleus_solidity",
            "nucleus_gray_mean",
            "nucleus_hematoxylin_od_mean",
            "nucleus_glcm_contrast",
        )
    }
    controls = {
        target: {
            "model": {"donor_balanced": {"donor_macro_balanced_accuracy": model}},
            "training_majority_baseline": {
                "donor_balanced": {"donor_macro_balanced_accuracy": baseline}
            },
        }
        for target, model, baseline in (
            ("broad_lineage", 0.4, 0.25),
            ("fine_type", 0.1, 0.03),
        )
    }
    controls["nuclear_morphology"] = {
        "full_context": {"scores": {"targets": morphology_targets}},
        "nucleus_mask_only": {"scores": {"targets": {}}},
    }
    gate = RUNNER.positive_control_gate(controls)
    assert gate["passed"] is True
    assert gate["evaluated_before_molecular_models"] is True
    assert gate["nucleus_mask_only_role"] == "secondary_attribution_not_used_for_gate"

    morphology_targets["nucleus_glcm_contrast"][
        "donor_type_macro_reference_error_reduction"
    ] = -0.01
    failed = RUNNER.positive_control_gate(controls)
    assert failed["passed"] is False
    assert failed["molecular_interpretation_allowed"] is False


def test_uni2_baseline_only_eligibility_is_exact_and_preserves_failed_gate() -> None:
    gate = _visible_control_gate(
        failed_morphology=RUNNER.UNI2_BASELINE_GEOMETRY_TARGETS
    )
    frozen_gate = json.loads(json.dumps(gate))
    receipt = RUNNER.uni2_baseline_only_eligibility(RUNNER.UNI2H_REPOSITORY, gate)
    assert receipt["eligible"] is True
    assert receipt["role"] == "retrospective_exposed_non_authorizing_baseline"
    assert receipt["amendment_timing"] == RUNNER.UNI2_BASELINE_AMENDMENT_TIMING
    assert set(receipt["failed_geometry_targets"]) == set(
        RUNNER.UNI2_BASELINE_GEOMETRY_TARGETS
    )
    assert receipt["comparison_inference_allowed"] is False
    assert receipt["descriptive_only"] is True
    assert gate == frozen_gate
    assert gate["passed"] is False
    assert gate["molecular_interpretation_allowed"] is False


@pytest.mark.parametrize(
    ("failed_morphology", "failed_classification", "technical_mutation"),
    (
        (("nucleus_area_um2",), (), None),
        (RUNNER.UNI2_BASELINE_GEOMETRY_TARGETS, ("fine_type",), None),
        (
            RUNNER.UNI2_BASELINE_GEOMETRY_TARGETS
            + ("nucleus_hematoxylin_od_mean",),
            (),
            None,
        ),
        (
            RUNNER.UNI2_BASELINE_GEOMETRY_TARGETS + ("nucleus_glcm_contrast",),
            (),
            None,
        ),
        (RUNNER.UNI2_BASELINE_GEOMETRY_TARGETS, (), "technical"),
        (RUNNER.UNI2_BASELINE_GEOMETRY_TARGETS, (), "extra"),
    ),
)
def test_uni2_baseline_only_rejects_type_stain_texture_technical_or_extra_failures(
    failed_morphology: tuple[str, ...],
    failed_classification: tuple[str, ...],
    technical_mutation: str | None,
) -> None:
    gate = _visible_control_gate(
        failed_morphology=failed_morphology,
        failed_classification=failed_classification,
    )
    if technical_mutation == "technical":
        gate["thresholds_frozen_without_hoptimus_molecular_outcomes"] = False
    elif technical_mutation == "extra":
        gate["full_context_morphology"]["unexpected_target"] = {
            "metric": "donor_type_macro_reference_error_reduction",
            "minimum_required": 0.0,
            "observed": -0.1,
            "passed": False,
        }
    receipt = RUNNER.uni2_baseline_only_eligibility(RUNNER.UNI2H_REPOSITORY, gate)
    assert receipt["eligible"] is False
    assert receipt["comparison_inference_allowed"] is False


def test_markdown_discloses_failed_gate_baseline_only_role(tmp_path: Path) -> None:
    output = tmp_path / "report.md"
    RUNNER._write_markdown(
        output,
        {
            "source_sha256": "source",
            "implementation_receipt": {
                "file_sha256": {
                    "scripts/benchmark_hest_scientific_reanalysis.py": "runner"
                },
                "git_head": "head",
                "git_worktree_dirty_at_start": False,
            },
            "donors": ["D0"],
            "sections": ["S0"],
            "encoder": RUNNER.UNI2H_REPOSITORY,
            "uni2_baseline_only_eligibility": {"eligible": True},
            "descriptive_baseline_summary": {
                "representation": "pca_512_broad_lineage_heads",
                "program_effects": {
                    "fibrotic_mesenchymal": {
                        "mean_delta_r2": 0.1,
                        "positive_donor_fraction": 0.6,
                    }
                },
            },
            "artifact_control_boundary": {
                "h_intrinsic_cell_status": "blocked",
                "h_intrinsic_nucleus_status": "blocked",
            },
        },
    )
    text = output.read_text(encoding="utf-8")
    assert "original UNI2-h positive-control gate remains **FAIL**" in text
    assert RUNNER.UNI2_BASELINE_AMENDMENT_TIMING in text
    assert "Comparator inference is prohibited" in text
    assert "Revised primary program probe" not in text
    assert "Decision:" not in text
    assert "sign-flip" not in text.lower()
    assert "holm" not in text.lower()
    assert "p-value" not in text.lower()


def test_uni2_descriptive_summary_contains_no_decision_semantics() -> None:
    effect = {
        "donor_count": len(RUNNER.EXPECTED_DONORS),
        "mean_effect": 0.1,
        "positive_fraction": 0.6,
        "exact_sign_flip_p": 0.2,
        "holm_adjusted_exact_sign_flip_p": 0.4,
    }
    observed = {
        "representations": {
            "pca_512_broad_lineage_heads": {
                "central_increment": {
                    program: dict(effect) for program in RUNNER.FROZEN_PROGRAM_NAMES
                }
            }
        }
    }
    summary = RUNNER._descriptive_uni2_baseline_summary(observed)
    assert summary["descriptive_only"] is True
    assert set(summary["program_effects"]) == set(RUNNER.FROZEN_PROGRAM_NAMES)
    banned = (
        "inference",
        "support",
        "negative_evidence",
        "decision",
        "_p",
        "p_value",
        "sign_flip",
        "holm",
    )

    def assert_sanitized(value) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                assert not any(term in str(key).lower() for term in banned)
                assert_sanitized(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_sanitized(nested)
        elif isinstance(value, str):
            assert not any(term in value.lower() for term in banned)

    assert_sanitized(summary)


def test_benchmark_stops_before_molecular_models_when_positive_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SimpleNamespace(
        encoder_name=RUNNER.HOPTIMUS1_REPOSITORY,
        nucleus_targets=np.zeros((1, 1)),
        roles=np.asarray(["evaluation"]),
    )
    statuses = []
    monkeypatch.setattr(RUNNER, "load_source", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(RUNNER, "_base_report", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(RUNNER, "measurement_analysis", lambda *_args: {})
    monkeypatch.setattr(RUNNER, "positive_control_analysis", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        RUNNER,
        "positive_control_gate",
        lambda *_args: {"passed": False, "molecular_interpretation_allowed": False},
    )
    monkeypatch.setattr(
        RUNNER,
        "observed_program_analysis",
        lambda *_args, **_kwargs: pytest.fail("molecular model ran before positive gate passed"),
    )
    monkeypatch.setattr(
        RUNNER,
        "_write_json",
        lambda _path, report: statuses.append(report["execution_status"]),
    )
    RUNNER.benchmark(
        tmp_path / "source.npz",
        tmp_path / "report.json",
        None,
        phase="full",
        device="cpu",
        inner_folds=2,
        seed=7,
        expected_encoder=RUNNER.HOPTIMUS1_REPOSITORY,
        comparison_report_path=tmp_path / "same-runner-uni2h.json",
    )
    assert statuses[-1] == "blocked_positive_control_gate_failed"


def test_uni2_gate_failed_baseline_does_not_continue_without_explicit_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SimpleNamespace(
        encoder_name=RUNNER.UNI2H_REPOSITORY,
        nucleus_targets=np.zeros((1, 1)),
        roles=np.asarray(["evaluation"]),
    )
    statuses = []
    failed_gate = _visible_control_gate(
        failed_morphology=RUNNER.UNI2_BASELINE_GEOMETRY_TARGETS
    )
    monkeypatch.setattr(RUNNER, "load_source", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(
        RUNNER,
        "_base_report",
        lambda *_args, **_kwargs: {
            "analysis_status": "retrospective_exposed_non_authorizing"
        },
    )
    monkeypatch.setattr(RUNNER, "measurement_analysis", lambda *_args: {})
    monkeypatch.setattr(RUNNER, "positive_control_analysis", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        RUNNER,
        "positive_control_gate",
        lambda *_args: failed_gate,
    )
    monkeypatch.setattr(
        RUNNER,
        "observed_program_analysis",
        lambda *_args, **_kwargs: pytest.fail("UNI2 molecular baseline ran without opt-in"),
    )
    monkeypatch.setattr(
        RUNNER,
        "_write_json",
        lambda _path, report: statuses.append(report["execution_status"]),
    )
    RUNNER.benchmark(
        tmp_path / "source.npz",
        tmp_path / "report.json",
        None,
        phase="nucleus",
        device="cpu",
        inner_folds=2,
        seed=7,
        expected_encoder=RUNNER.UNI2H_REPOSITORY,
    )
    assert statuses[-1] == "blocked_positive_control_gate_failed"


def test_exact_uni2_gate_failed_baseline_continues_descriptively_with_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SimpleNamespace(
        encoder_name=RUNNER.UNI2H_REPOSITORY,
        nucleus_targets=np.zeros((1, 1)),
        roles=np.asarray(["evaluation"]),
    )
    reports = []
    observed_calls = []
    failed_gate = _visible_control_gate(
        failed_morphology=RUNNER.UNI2_BASELINE_GEOMETRY_TARGETS
    )
    monkeypatch.setattr(RUNNER, "load_source", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(
        RUNNER,
        "_base_report",
        lambda *_args, **_kwargs: {
            "analysis_status": "retrospective_exposed_non_authorizing"
        },
    )
    monkeypatch.setattr(RUNNER, "measurement_analysis", lambda *_args: {})
    monkeypatch.setattr(RUNNER, "positive_control_analysis", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        RUNNER,
        "positive_control_gate",
        lambda *_args: failed_gate,
    )

    def fake_observed(*_args, **_kwargs):
        observed_calls.append(True)
        return {"representations": {}}

    monkeypatch.setattr(RUNNER, "observed_program_analysis", fake_observed)
    monkeypatch.setattr(
        RUNNER,
        "_descriptive_uni2_baseline_summary",
        lambda *_args, **_kwargs: {
            "schema": "heir.hest_uni2_descriptive_baseline_summary.v1",
            "role": "retrospective_exposed_non_authorizing_baseline",
            "descriptive_only": True,
            "representation": "pca_512_broad_lineage_heads",
            "program_effects": {},
        },
    )
    monkeypatch.setattr(
        RUNNER, "reference_support_sensitivity_analysis", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        RUNNER,
        "_write_json",
        lambda _path, report: reports.append(dict(report)),
    )
    RUNNER.benchmark(
        tmp_path / "source.npz",
        tmp_path / "report.json",
        None,
        phase="nucleus",
        device="cpu",
        inner_folds=2,
        seed=7,
        expected_encoder=RUNNER.UNI2H_REPOSITORY,
        allow_gate_failed_uni2_baseline_only=True,
    )
    assert observed_calls == [True]
    final = reports[-1]
    assert final["execution_status"] == "observed_nucleus_program_models_complete"
    assert final["positive_control_gate"]["passed"] is False
    assert final["positive_control_gate"]["molecular_interpretation_allowed"] is False
    assert final["uni2_baseline_only_eligibility"]["eligible"] is True
    assert final["analysis_status"] == "retrospective_exposed_non_authorizing_baseline"
    assert final["comparison_inference_allowed"] is False
    assert final["descriptive_only"] is True
    assert "scientific_summary" not in final
    assert final["descriptive_baseline_summary"]["descriptive_only"] is True


def test_uni2_baseline_only_opt_in_rejects_hoptimus_and_nonmolecular_phase(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="invalid for H-optimus-1"):
        RUNNER.benchmark(
            tmp_path / "source.npz",
            tmp_path / "report.json",
            None,
            phase="nucleus",
            device="cpu",
            inner_folds=2,
            seed=7,
            expected_encoder=RUNNER.HOPTIMUS1_REPOSITORY,
            comparison_report_path=tmp_path / "comparator.json",
            allow_gate_failed_uni2_baseline_only=True,
        )
    with pytest.raises(ValueError, match="requires a nucleus or full phase"):
        RUNNER.benchmark(
            tmp_path / "source.npz",
            tmp_path / "report.json",
            None,
            phase="positive",
            device="cpu",
            inner_folds=2,
            seed=7,
            expected_encoder=RUNNER.UNI2H_REPOSITORY,
            allow_gate_failed_uni2_baseline_only=True,
        )


def test_hoptimus_nucleus_or_full_requires_same_runner_comparator(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a same-runner UNI2-h report"):
        RUNNER.benchmark(
            tmp_path / "source.npz",
            tmp_path / "report.json",
            None,
            phase="nucleus",
            device="cpu",
            inner_folds=2,
            seed=7,
            expected_encoder=RUNNER.HOPTIMUS1_REPOSITORY,
        )


def test_hoptimus_comparator_preflight_fails_before_molecular_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SimpleNamespace(
        encoder_name=RUNNER.HOPTIMUS1_REPOSITORY,
        nucleus_targets=np.zeros((1, 1)),
        roles=np.asarray(["evaluation"]),
    )
    statuses = []
    monkeypatch.setattr(RUNNER, "load_source", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(RUNNER, "_base_report", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(RUNNER, "measurement_analysis", lambda *_args: {})
    monkeypatch.setattr(RUNNER, "positive_control_analysis", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        RUNNER,
        "positive_control_gate",
        lambda *_args: {"passed": True, "molecular_interpretation_allowed": True},
    )

    def fail_preflight(*_args, **_kwargs):
        raise ValueError("same-runner comparator contract mismatch")

    monkeypatch.setattr(RUNNER, "same_runner_uni2_comparator_preflight", fail_preflight)
    monkeypatch.setattr(
        RUNNER,
        "observed_program_analysis",
        lambda *_args, **_kwargs: pytest.fail("molecular fit ran after preflight failure"),
    )
    monkeypatch.setattr(
        RUNNER,
        "_write_json",
        lambda _path, report: statuses.append(report["execution_status"]),
    )
    with pytest.raises(ValueError, match="same-runner comparator contract mismatch"):
        RUNNER.benchmark(
            tmp_path / "source.npz",
            tmp_path / "report.json",
            None,
            phase="nucleus",
            device="cpu",
            inner_folds=2,
            seed=7,
            expected_encoder=RUNNER.HOPTIMUS1_REPOSITORY,
            comparison_report_path=tmp_path / "same-runner-uni2h.json",
        )
    assert statuses[-1] == "blocked_same_runner_uni2_comparator_preflight_failed"


def test_hoptimus_benchmark_uses_full_1536_primary_after_positive_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SimpleNamespace(
        encoder_name=RUNNER.HOPTIMUS1_REPOSITORY,
        nucleus_targets=np.zeros((1, 1)),
        roles=np.asarray(["evaluation"]),
    )
    observed_calls = []
    primary_calls = []
    execution_order = []
    monkeypatch.setattr(RUNNER, "load_source", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(RUNNER, "_base_report", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(RUNNER, "measurement_analysis", lambda *_args: {})
    monkeypatch.setattr(RUNNER, "positive_control_analysis", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        RUNNER,
        "positive_control_gate",
        lambda *_args: {"passed": True, "molecular_interpretation_allowed": True},
    )

    def fake_observed(*_args, **kwargs):
        execution_order.append("observed")
        observed_calls.append(kwargs["full_representation_sensitivity"])
        return {"representations": {}}

    def fake_summary(*_args, **kwargs):
        primary_calls.append(kwargs["primary_representation"])
        return {}

    monkeypatch.setattr(RUNNER, "observed_program_analysis", fake_observed)
    monkeypatch.setattr(RUNNER, "_primary_probe_summary", fake_summary)
    monkeypatch.setattr(
        RUNNER,
        "same_runner_uni2_comparator_preflight",
        lambda *_args, **_kwargs: execution_order.append("preflight") or {"passed": True},
    )
    monkeypatch.setattr(
        RUNNER, "paired_encoder_comparison_report", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        RUNNER, "reference_support_sensitivity_analysis", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(RUNNER, "_write_json", lambda *_args, **_kwargs: None)
    RUNNER.benchmark(
        tmp_path / "source.npz",
        tmp_path / "report.json",
        None,
        phase="nucleus",
        device="cpu",
        inner_folds=2,
        seed=7,
        representation_profile="primary",
        expected_encoder=RUNNER.HOPTIMUS1_REPOSITORY,
        comparison_report_path=tmp_path / "same-runner-uni2h.json",
    )
    assert observed_calls == [True]
    assert primary_calls == ["full_1536_broad_lineage_heads"]
    assert execution_order[:2] == ["preflight", "observed"]
