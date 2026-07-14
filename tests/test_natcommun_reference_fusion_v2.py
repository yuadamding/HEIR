from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from heir.evaluation.reference_fusion_v2 import ReferenceCalibrator


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_natcommun_reference_fusion_v2.py"
    spec = importlib.util.spec_from_file_location("benchmark_natcommun_reference_fusion_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _visible_rows() -> dict[str, dict[str, str]]:
    return {
        donor: {
            "truth": runner.DONOR_INDICATION[donor],
            "prediction": runner.DONOR_INDICATION[donor],
            "baseline_prediction": "dlbcl",
        }
        for donor in runner.PRIMARY_DONORS
    }


def _passing_components() -> dict[str, object]:
    primary_sections = set(runner.EXPECTED_SECTIONS) - {"B2_2"}
    crop_arms = {}
    for crop_id in runner.CROP_IDS:
        crop_arms[crop_id] = {
            "features": {
                "passed": True,
                "encoder_id": runner.HOPTIMUS_ENCODER_ID,
                "crop_id": crop_id,
                "finite": True,
                "centered_frobenius_norm": 2.0,
                "varying_dimensions": 2,
                "per_section": {
                    section: {
                        "rows": 2,
                        "centered_frobenius_norm": 1.0,
                        "varying_dimensions": 1,
                    }
                    for section in primary_sections
                },
            },
            "visible_control": runner._recompute_visible_control(_visible_rows()),
        }
    return {
        "common": {
            "registration": {
                "passed": True,
                "blinded": True,
                "sections": {
                    section: {
                        "manual_status": "passed",
                        "maximum_padding_fraction": 0.0,
                    }
                    for section in runner.EXPECTED_SECTIONS
                },
            },
            "controls": {
                "passed": True,
                "blank": {},
                "shuffle": {donor: {"fixed_points": 0} for donor in runner.PRIMARY_DONORS},
                "banks": {donor: {} for donor in runner.PRIMARY_DONORS},
                "matched_wrong_and_same_indication_generic_available": True,
            },
            "reliability": {
                "passed": True,
                "outer_training_only": True,
                "heldout_ST_used": False,
                "folds": {donor: {"status": "feasible"} for donor in runner.PRIMARY_DONORS},
            },
        },
        "crop_arms": crop_arms,
    }


def test_protocol_is_isolated_from_non_gating_hest_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = json.loads(runner.PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["hest_architecture_diagnostic"].update(
        {
            "report_path": str(tmp_path / "intentionally-absent-hest.json"),
            "regional_benchmark_blocking": False,
        }
    )
    isolated = tmp_path / "protocol.json"
    isolated.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(runner, "PROTOCOL_PATH", isolated)

    loaded = runner._load_protocol(_sha256(isolated))

    assert loaded["hest_architecture_diagnostic"]["regional_benchmark_blocking"] is False
    assert not (tmp_path / "intentionally-absent-hest.json").exists()

    protocol["hest_architecture_diagnostic"]["regional_benchmark_blocking"] = True
    isolated.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected scientific scope"):
        runner._load_protocol(_sha256(isolated))


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    (
        (("hest_architecture_diagnostic", "report_sha256"), "0" * 64),
        (("hest_architecture_diagnostic", "source_sha256"), "0" * 64),
        (("hest_architecture_diagnostic", "frozen_result"), "tampered_result"),
        (("familywise_alpha",), 0.10),
        (
            ("regional_preflight", "visible_control", "metric"),
            "ordinary_donor_macro_accuracy",
        ),
        (
            ("immutable_computation_dependencies", "source_builder", "path"),
            "scripts/not_the_registered_builder.py",
        ),
        (("encoders", "secondary", "may_rescue_primary_failure"), True),
        (
            (
                "pre_result_technical_amendment",
                "amended_completion_rule",
                "maximum_iterations",
            ),
            100,
        ),
        (
            (
                "pre_result_technical_amendment",
                "completion_scan",
                "maximum_exact_label_stability_iteration",
            ),
            181,
        ),
        (
            ("pre_result_technical_amendment", "encoder_execution_scope", "uni2_h"),
            "secondary_run",
        ),
        (
            ("reference_representation", "latent_input"),
            "raw_snRNA_latent",
        ),
        (
            ("reference_representation", "primary", "maximum_iterations"),
            100,
        ),
        (
            ("reference_representation", "primary", "convergence_rule"),
            "center_shift_tolerance",
        ),
        (("resource_limits", "maximum_cpu_threads"), 9),
        (("decision", "cell_level_HEIR_authorized"), True),
        (("decision", "regional_research_software_authorized_if_supported"), False),
    ),
)
def test_protocol_rejects_mutated_scientific_contract_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[str, ...],
    replacement: object,
) -> None:
    protocol = json.loads(runner.PROTOCOL_PATH.read_text(encoding="utf-8"))
    parent = protocol
    for field in field_path[:-1]:
        parent = parent[field]
    parent[field_path[-1]] = replacement
    isolated = tmp_path / "mutated-protocol.json"
    isolated.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(runner, "PROTOCOL_PATH", isolated)

    with pytest.raises(ValueError):
        runner._load_protocol(_sha256(isolated))


def test_visible_control_is_recomputed_and_forged_headlines_fail_closed() -> None:
    result = runner._recompute_visible_control(_visible_rows())
    assert result["passed"] is True
    assert result["model"] == 1.0
    assert result["outer_training_majority_baseline"] == pytest.approx(1.0 / 3.0)
    assert runner._component_gate(_passing_components())["passed"] is True

    for field, forged in (("model", 0.0), ("increment", 999.0), ("passed", False)):
        components = _passing_components()
        components["crop_arms"]["target_55um"]["visible_control"][field] = forged
        with pytest.raises(ValueError, match=f"visible-control {field} is inconsistent"):
            runner._component_gate(components)


def test_visible_control_nested_receipts_and_outcome_free_claim_fail_closed() -> None:
    components = _passing_components()
    components["crop_arms"]["target_55um"]["visible_control"]["per_indication"]["breast"][
        "model_accuracy"
    ] = 0.0
    with pytest.raises(ValueError, match="visible-control .* inconsistent"):
        runner._component_gate(components)

    components = _passing_components()
    components["crop_arms"]["context_112um"]["visible_control"]["uses_ST_or_reference_outcomes"] = (
        True
    )
    with pytest.raises(ValueError, match="visible-control .* inconsistent"):
        runner._component_gate(components)


def test_component_gate_is_exact_and_fail_closed() -> None:
    components = _passing_components()
    components["common"]["registration"]["sections"]["B1_2"]["manual_status"] = "failed"
    components["common"]["registration"]["passed"] = False
    gate = runner._component_gate(components)
    assert gate["passed"] is False
    assert gate["common_component_passes"]["registration"] is False
    assert gate["hest_geometry_gate_required"] is False

    incomplete = _passing_components()
    incomplete["common"].pop("controls")
    with pytest.raises(ValueError, match="common or crop-arm components are incomplete"):
        runner._component_gate(incomplete)

    extra = _passing_components()
    extra["hest"] = {"passed": False}
    with pytest.raises(ValueError, match="components are incomplete"):
        runner._component_gate(extra)


@pytest.mark.parametrize("crop_id", runner.CROP_IDS)
def test_each_crop_arm_is_independently_required_to_pass(crop_id: str) -> None:
    components = _passing_components()
    features = components["crop_arms"][crop_id]["features"]
    features["centered_frobenius_norm"] = 0.0
    features["passed"] = False

    gate = runner._component_gate(components)

    other_crop = next(value for value in runner.CROP_IDS if value != crop_id)
    assert gate["passed"] is False
    assert gate["all_crop_arms_required"] is True
    assert gate["crop_arm_passes"][crop_id] is False
    assert gate["crop_arm_component_passes"][crop_id]["features"] is False
    assert gate["crop_arm_passes"][other_crop] is True


@pytest.mark.parametrize(
    ("role", "repository", "forged_authority", "forged_rescue"),
    (
        ("secondary_non_authorizing", "MahmoodLab/UNI2-h", True, False),
        ("secondary_non_authorizing", "MahmoodLab/UNI2-h", False, True),
        ("primary", "bioptimus/H-optimus-1", False, False),
    ),
)
def test_preflight_verification_rejects_forged_encoder_authority(
    tmp_path: Path,
    role: str,
    repository: str,
    forged_authority: bool,
    forged_rescue: bool,
) -> None:
    supplement_path = (tmp_path / "supplement.npz").resolve()
    inputs = runner._EncoderInputs(
        encoder_id=(runner.HOPTIMUS_ENCODER_ID if role == "primary" else runner.UNI2_ENCODER_ID),
        role=role,
        repository=repository,
        crop_sources={crop_id: object() for crop_id in runner.CROP_IDS},
        supplement_path=supplement_path,
        supplement_sha256="7" * 64,
        supplement_receipt={"bound": True},
        implementation_files={},
    )
    report = {
        "schema": runner.PREFLIGHT_SCHEMA,
        "source_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "registration_review_sha256": "3" * 64,
        "encoder": {
            "id": inputs.encoder_id,
            "role": role,
            "repository": repository,
            "primary_decision_authority": forged_authority,
            "may_rescue_primary": forged_rescue,
        },
        "supplement": {
            "path": str(supplement_path),
            "sha256": inputs.supplement_sha256,
            "receipt": inputs.supplement_receipt,
        },
    }
    path = tmp_path / "forged-preflight.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    args = argparse.Namespace(
        preflight_report=path,
        expected_preflight_report_sha256=_sha256(path),
        expected_source_sha256="1" * 64,
        expected_protocol_sha256="2" * 64,
        expected_registration_review_sha256="3" * 64,
    )

    with pytest.raises(ValueError, match="preflight report identities are inconsistent"):
        runner._verify_preflight_report(
            args,
            source=object(),
            inputs=inputs,
            legacy=object(),
            protocol={},
            current_implementation={},
        )


def test_markdown_states_secondary_encoder_cannot_authorize_or_rescue() -> None:
    markdown = runner._markdown(
        {
            "encoder": {"id": runner.UNI2_ENCODER_ID, "role": "secondary_non_authorizing"},
            "encoder_decision": {"decision": "supported"},
            "crop_decisions": {},
            "experiments": {},
        }
    )

    assert "cannot authorize cell-level HEIR claims" in markdown
    assert "secondary and cannot authorize, rescue, or override" in markdown
    assert "results are not pooled across encoders" in markdown


def _identity_cli_arguments(sha: str) -> list[str]:
    return [
        "--source",
        "source.npz",
        "--expected-source-sha256",
        sha,
        "--registration-review",
        "registration.json",
        "--expected-registration-review-sha256",
        sha,
        "--expected-protocol-sha256",
        sha,
        "--expected-runner-sha256",
        sha,
        "--expected-reference-v2-sha256",
        sha,
        "--device",
        "cuda",
    ]


def _hoptimus_cli_arguments(sha: str) -> list[str]:
    return [
        "--crop-55-supplement",
        "crop.npz",
        "--expected-crop-55-supplement-sha256",
        sha,
        "--expected-crop-builder-sha256",
        sha,
    ]


def _uni2_cli_arguments(sha: str) -> list[str]:
    return [
        "--uni2-supplement",
        "uni2.npz",
        "--expected-uni2-supplement-sha256",
        sha,
        "--expected-uni2-builder-sha256",
        sha,
        "--expected-uni2-adapter-sha256",
        sha,
        "--expected-encoder-base-sha256",
        sha,
        "--expected-encoder-factory-sha256",
        sha,
    ]


def test_benchmark_cli_preserves_registered_alpha_one_endpoint() -> None:
    sha = "a" * 64
    args = runner.parse_args(
        [
            "benchmark-hoptimus",
            *_identity_cli_arguments(sha),
            "--preflight-report",
            "preflight.json",
            "--expected-preflight-report-sha256",
            sha,
            *_hoptimus_cli_arguments(sha),
            "--output-dir",
            "output",
            "--fusion-alphas",
            "0,0.1,0.25,0.5,0.75,1",
        ]
    )
    assert args.command_name == "benchmark-hoptimus"
    assert args.handler is runner._run_hoptimus_benchmark
    assert args.fusion_alphas == runner.DEFAULT_FUSION_ALPHAS
    assert args.fusion_alphas[-1] == 1.0
    assert not any("hest" in str(value).lower() for value in vars(args))

    with pytest.raises(argparse.ArgumentTypeError, match="cannot exceed 1"):
        runner._float_grid("0,1.01", positive=False, upper=1.0)


def test_hoptimus_preflight_cli_is_encoder_specific() -> None:
    sha = "b" * 64
    args = runner.parse_args(
        [
            "preflight-hoptimus",
            *_identity_cli_arguments(sha),
            *_hoptimus_cli_arguments(sha),
            "--output",
            "hoptimus-preflight.json",
        ]
    )

    assert args.command_name == "preflight-hoptimus"
    assert args.handler is runner._run_hoptimus_preflight
    assert args.output == Path("hoptimus-preflight.json")
    assert not hasattr(args, "uni2_supplement")


@pytest.mark.parametrize(
    ("command", "terminal_arguments", "handler_name"),
    (
        (
            "preflight-uni2",
            ["--output", "uni2-preflight.json"],
            "_run_uni2_preflight",
        ),
        (
            "benchmark-uni2",
            [
                "--preflight-report",
                "uni2-preflight.json",
                "--expected-preflight-report-sha256",
                "c" * 64,
                "--output-dir",
                "uni2-output",
            ],
            "_run_uni2_benchmark",
        ),
    ),
)
def test_uni2_cli_has_separate_preflight_and_benchmark_commands(
    command: str, terminal_arguments: list[str], handler_name: str
) -> None:
    sha = "c" * 64
    args = runner.parse_args(
        [
            command,
            *_identity_cli_arguments(sha),
            *_uni2_cli_arguments(sha),
            *terminal_arguments,
        ]
    )

    assert args.command_name == command
    assert args.handler is getattr(runner, handler_name)
    assert args.uni2_supplement == Path("uni2.npz")
    assert not hasattr(args, "crop_55_supplement")


def test_registered_experiment_specs_are_exactly_eight_primary_and_two_centroid() -> None:
    specs = runner._registered_experiment_specs()
    experiment_ids = {
        (
            f"{crop_id}::"
            f"{'state_kmeans_8' if prototypes == 8 else 'type_centroid_1'}::"
            f"{endpoint}::{bank}"
        )
        for crop_id, endpoint, bank, prototypes, _role in specs
    }
    expected_primary = {
        name for crop_id in runner.CROP_IDS for name in runner._primary_experiment_names(crop_id)
    }
    expected_centroid = {
        f"{crop_id}::type_centroid_1::program_total::natural" for crop_id in runner.CROP_IDS
    }

    assert len(specs) == 10
    assert len(experiment_ids) == 10
    assert experiment_ids == expected_primary | expected_centroid
    assert sum(prototypes == 8 and role == "primary" for *_, prototypes, role in specs) == 8
    assert (
        sum(prototypes == 1 and role == "secondary_diagnostic" for *_, prototypes, role in specs)
        == 2
    )


def test_experiment_checkpoint_reuses_exact_identity_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    key = "target_55um::state_kmeans_8::program_total::natural"
    path = runner._experiment_checkpoint_path(tmp_path, key)
    identity = {
        "schema": "heir.natcommun_experiment_identity.v1",
        "encoder_id": runner.HOPTIMUS_ENCODER_ID,
        "supplement_sha256": "d" * 64,
        "preflight_report_sha256": "1" * 64,
        "registration_review_sha256": "2" * 64,
        "numeric_backend_sha256": "3" * 64,
        "cpu_threads": 4,
        "experiment": key,
    }
    result = {
        "crop_id": "target_55um",
        "reference_representation": "state_kmeans_8",
        "headline": {"M3_loss": 0.5},
    }

    assert runner._load_experiment_checkpoint(path, identity) is None
    runner._write_experiment_checkpoint(path, identity, result)
    assert runner._load_experiment_checkpoint(path, identity) == result

    for field, replacement in (
        ("supplement_sha256", "e" * 64),
        ("preflight_report_sha256", "4" * 64),
        ("registration_review_sha256", "5" * 64),
        ("numeric_backend_sha256", "6" * 64),
        ("cpu_threads", 8),
    ):
        stale_identity = {**identity, field: replacement}
        with pytest.raises(ValueError, match="checkpoint is stale or corrupted"):
            runner._load_experiment_checkpoint(path, stale_identity)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result"]["headline"]["M3_loss"] = 99.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint is stale or corrupted"):
        runner._load_experiment_checkpoint(path, identity)


def test_adaptive_fusion_allows_alpha_one_and_abstains_when_unsupported() -> None:
    image = np.asarray([[0.0], [4.0]])
    reference = np.asarray([[2.0], [2.0]])
    diagnostics = {
        "support_distance": np.asarray([0.0, np.inf]),
        "type_coverage": np.asarray([1.0, 1.0]),
        "reference_uncertainty": np.asarray([0.0, 0.0]),
    }

    prediction, receipt = runner._adaptive_fusion_v2(image, reference, diagnostics, 1.0, 1.0)

    np.testing.assert_allclose(prediction[:, 0], [2.0, 4.0])
    np.testing.assert_allclose(receipt["adaptive_alpha"], [1.0, 0.0])
    np.testing.assert_array_equal(receipt["abstained_fallback_to_H"], [False, True])


def test_indication_equal_ridge_and_fusion_selection_differs_from_donor_equal() -> None:
    donors = np.asarray(["B1", "D1", "D2", "D3", "D4", "D5", "D6"])
    sections = np.asarray([f"{donor}_section" for donor in donors])
    features = np.arange(len(donors), dtype=np.float64)[:, None]
    truth = np.zeros((len(donors), 1), dtype=np.float64)
    minority_error = np.asarray([[4.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    majority_error = np.asarray([[0.0], [2.0], [2.0], [2.0], [2.0], [2.0], [2.0]])
    candidates = (minority_error, majority_error)

    class FakeRidgeFit:
        fit_device = "cuda:0"

        def predict(self, values):
            indices = np.asarray(values)[:, 0].astype(int)
            return np.stack([candidate[indices] for candidate in candidates])

    class FakeLegacy:
        @staticmethod
        def donor_section_macro_loss(target, prediction, donor_ids, _section_ids):
            donor_mse = {
                donor: float(
                    np.mean(np.square(prediction[donor_ids == donor] - target[donor_ids == donor]))
                )
                for donor in sorted(set(donor_ids.tolist()))
            }
            return {
                "donor_mse": donor_mse,
                "donor_section_macro_mse": float(np.mean(tuple(donor_mse.values()))),
            }

        @staticmethod
        def grouped_donor_folds(donor_ids, *, n_splits, seed):
            del n_splits, seed
            indices = np.arange(len(donor_ids))
            return ((indices, indices),)

        @staticmethod
        def fit_weighted_ridge_grid(*_args, **_kwargs):
            return FakeRidgeFit()

        @staticmethod
        def _donor_section_weights(donor_ids, _section_ids):
            return np.ones(len(donor_ids))

        @staticmethod
        def _retrieve(image, _type_probabilities, _bank, _type_names, _temperature):
            rows = len(image)
            return np.zeros((rows, 1)), {
                "support_distance": np.ones(rows),
                "type_coverage": np.ones(rows),
                "reference_uncertainty": np.zeros(rows),
            }

        @staticmethod
        def _adaptive_fusion(image, _reference, _diagnostics, alpha, _threshold):
            indices = np.asarray(image)[:, 0].astype(int)
            selected = candidates[0 if float(alpha) == 0.0 else 1]
            return selected[indices], {}

    legacy = FakeLegacy()
    donor_equal_losses = [
        legacy.donor_section_macro_loss(truth, candidate, donors, sections)[
            "donor_section_macro_mse"
        ]
        for candidate in candidates
    ]
    assert donor_equal_losses[0] < donor_equal_losses[1]

    ridge_alpha, ridge_losses = runner._select_ridge_alpha_indication_equal(
        legacy,
        features,
        truth,
        donors,
        sections,
        (0.1, 10.0),
        seed=17,
        device="cuda",
    )
    fusion_alpha, temperature, _threshold, fusion_losses = (
        runner._select_fusion_parameters_indication_equal(
            legacy,
            features,
            np.ones((len(donors), 1)),
            truth,
            donors,
            sections,
            {donor: object() for donor in donors},
            np.asarray(["type"]),
            (1.0,),
            (0.0, 1.0),
        )
    )

    assert ridge_alpha == 10.0
    assert ridge_losses["10"] < ridge_losses["0.1"]
    assert fusion_alpha == 1.0
    assert temperature == 1.0
    assert fusion_losses["temperature=1|alpha=1"] < fusion_losses["temperature=1|alpha=0"]


def test_visible_control_receipt_names_indication_equal_selection_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indication_names = np.asarray(sorted(runner.PRIMARY_DONORS_BY_INDICATION))
    donor_ids = np.asarray(runner.PRIMARY_DONORS)
    indication_ids = np.asarray([runner.DONOR_INDICATION[donor] for donor in donor_ids])
    image_features = np.eye(len(indication_names))[
        np.asarray([int(np.flatnonzero(indication_names == value)[0]) for value in indication_ids])
    ]
    source = argparse.Namespace(
        spot_ids=np.asarray([f"spot-{donor}" for donor in donor_ids]),
        section_ids=np.asarray([f"section-{donor}" for donor in donor_ids]),
        donor_ids=donor_ids,
        indication_ids=indication_ids,
        image_features=image_features,
    )
    monkeypatch.setattr(
        runner,
        "_select_ridge_alpha_indication_equal",
        lambda *_args, **_kwargs: (1.0, {"1": 0.0}),
    )
    legacy = argparse.Namespace(
        _fit_predict_ridge=lambda _train_x, _train_y, test_x, *_args: test_x,
    )

    receipt = runner._visible_control(
        source,
        legacy,
        ridge_alphas=runner.DEFAULT_RIDGE_ALPHAS,
        maximum_per_section=128,
        seed=17,
        device="cuda",
    )

    assert receipt["metric"] == "indication_balanced_donor_macro_accuracy"
    assert receipt["passed"] is True
    assert {fold["selection_weighting"] for fold in receipt["fold_receipts"].values()} == {
        "indication_equal_then_donor_equal_within_indication"
    }


def test_program_quality_hooks_capture_fold_retained_m0_and_m3_without_matrices() -> None:
    def fake_score(*_args):
        return {"delegated": True}

    def fake_rare(*_args):
        return {"delegated": True}

    def fake_gate(_half_a, _half_b, _donors, _training_donors, names):
        return {
            "status": "feasible",
            "retained_programs": np.asarray(names).astype(str).tolist(),
        }

    binding = runner._V2ProgramQualityBindings(fake_score, fake_rare, fake_gate)
    donors = np.repeat(np.asarray(runner.PRIMARY_DONORS), 3)
    sections = np.asarray([f"{donor}::section" for donor in donors])
    local = np.tile(np.asarray([0.0, 1.0, 2.0]), len(runner.PRIMARY_DONORS))
    truth = np.column_stack((local, 2.0 * local, 3.0 * local))
    names = np.asarray(["program_a", "program_b", "program_c"])
    thresholds = np.full_like(truth, 0.5)
    types = np.repeat("type", len(truth))

    binding.start("program_total")
    for heldout in runner.PRIMARY_DONORS:
        training = [donor for donor in runner.PRIMARY_DONORS if donor != heldout]
        binding.program_reliability_gate(truth, truth, donors, training, names)
    for model in runner.PROGRAM_SCORE_CALL_MODELS:
        prediction = truth if model == "M3" else np.zeros_like(truth)
        assert binding.score_model(truth, prediction, donors, sections, types)["delegated"]
    for model in runner.PROGRAM_RARE_STATE_CALL_MODELS:
        prediction = truth if model == "M3" else np.zeros_like(truth)
        assert binding.rare_state_metrics(truth, prediction, thresholds, names)["delegated"]
    receipt = binding.finish("program_total")

    assert receipt is not None
    assert receipt["schema"] == runner.PROGRAM_QUALITY_SCHEMA
    assert receipt["target_names"] == names.tolist()
    assert set(receipt["folds"]) == set(runner.PRIMARY_DONORS)
    assert receipt["variance_preservation"]["M0"][
        "median_within_section_variance_ratio"
    ] == pytest.approx(0.0)
    assert receipt["variance_preservation"]["M3"][
        "median_within_section_variance_ratio"
    ] == pytest.approx(1.0)
    assert receipt["rare_state_recall_coverage"]["M0"]["program_a"]["recall"] == 0.0
    assert receipt["rare_state_recall_coverage"]["M3"]["program_a"]["recall"] == 1.0
    assert receipt["instrumentation"]["large_prediction_matrices_retained_after_call"] is False


def test_program_quality_hooks_record_infeasible_reliability_without_crashing() -> None:
    def fake_gate(_half_a, _half_b, _donors, training_donors, names):
        blocked = "B1" not in training_donors
        return {
            "status": "blocked_fewer_than_three_reliable_programs" if blocked else "feasible",
            "retained_programs": [] if blocked else np.asarray(names).astype(str).tolist(),
        }

    binding = runner._V2ProgramQualityBindings(
        lambda *_args: {},
        lambda *_args: {},
        fake_gate,
    )
    donors = np.asarray(runner.PRIMARY_DONORS)
    truth = np.arange(len(donors) * 3, dtype=np.float64).reshape(len(donors), 3)
    names = np.asarray(["program_a", "program_b", "program_c"])
    sections = np.asarray([f"{donor}::section" for donor in donors])

    binding.start("program_total")
    for heldout in runner.PRIMARY_DONORS:
        training = [donor for donor in runner.PRIMARY_DONORS if donor != heldout]
        binding.program_reliability_gate(truth, truth, donors, training, names)
    for _model in runner.PROGRAM_SCORE_CALL_MODELS:
        binding.score_model(truth, truth, donors, sections, np.repeat("type", len(donors)))
    for _model in runner.PROGRAM_RARE_STATE_CALL_MODELS:
        binding.rare_state_metrics(truth, truth, np.zeros_like(truth), names)
    receipt = binding.finish("program_total")

    assert receipt is not None
    assert receipt["status"] == "blocked_program_reliability_infeasible"
    assert receipt["folds"]["B1"]["retained_programs"] == []
    assert receipt["variance_preservation"] == {}
    assert receipt["rare_state_recall_coverage"] == {}


def _experiment_with_indication_m3(m3_by_indication: dict[str, float]) -> dict[str, object]:
    return {
        "headline": {
            "per_donor": {
                donor: {
                    "M0_loss": 10.0,
                    "M3_loss": m3_by_indication[runner.DONOR_INDICATION[donor]],
                }
                for donor in runner.PRIMARY_DONORS
            }
        }
    }


def test_indication_balance_requires_two_of_three_and_no_severe_reversal() -> None:
    two_positive = runner._indication_summary(
        _experiment_with_indication_m3({"breast": 9.0, "lung": 9.0, "dlbcl": 10.2}),
        _effect_thresholds(),
    )
    assert two_positive["positive_indication_count"] == 2
    assert two_positive["no_severe_reversal"] is True
    assert two_positive["heterogeneity_passed"] is True

    severe = runner._indication_summary(
        _experiment_with_indication_m3({"breast": 9.0, "lung": 9.0, "dlbcl": 10.6}),
        _effect_thresholds(),
    )
    assert severe["positive_indication_count"] == 2
    assert severe["per_indication"]["dlbcl"]["severe_reversal"] is True
    assert severe["heterogeneity_passed"] is False

    one_positive = runner._indication_summary(
        _experiment_with_indication_m3({"breast": 9.0, "lung": 10.0, "dlbcl": 10.0}),
        _effect_thresholds(),
    )
    assert one_positive["positive_indication_count"] == 1
    assert one_positive["no_severe_reversal"] is True
    assert one_positive["heterogeneity_passed"] is False


def _passing_crop_inputs(
    crop_id: str,
) -> tuple[dict[str, dict[str, object]], dict[str, float]]:
    experiments: dict[str, dict[str, object]] = {}
    adjusted: dict[str, float] = {}
    for name in runner._primary_experiment_names(crop_id):
        endpoint = name.split("::")[2]
        rare_state = {
            "marker_a": {"truth_positive": 10, "recall": 0.80},
            "marker_b": {"truth_positive": 8, "recall": 0.60},
            "marker_c": {"truth_positive": 0, "recall": None},
        }
        rare_state_m3 = {
            "marker_a": {"truth_positive": 10, "recall": 0.75},
            "marker_b": {"truth_positive": 8, "recall": 0.55},
            "marker_c": {"truth_positive": 0, "recall": None},
        }
        qualified_program_quality = {
            "schema": runner.PROGRAM_QUALITY_SCHEMA,
            "status": "complete",
            "endpoint": "program_total",
            "selection_scope": ("outer_training_reliability_qualified_programs_per_heldout_donor"),
            "heldout_ST_used_for_program_selection": False,
            "target_names": ["marker_a", "marker_b", "marker_c"],
            "folds": {
                donor: {
                    "status": "feasible",
                    "fit_donors": sorted(
                        value for value in runner.PRIMARY_DONORS if value != donor
                    ),
                    "retained_programs": ["marker_a", "marker_b", "marker_c"],
                    "retained_program_count": 3,
                }
                for donor in runner.PRIMARY_DONORS
            },
            "variance_preservation": {
                "M0": {"median_within_section_variance_ratio": 0.70},
                "M3": {"median_within_section_variance_ratio": 0.75},
            },
            "rare_state_recall_coverage": {
                "M0": rare_state,
                "M3": rare_state_m3,
            },
        }
        experiments[name] = {
            "endpoint": endpoint,
            "target_names": ["marker_a", "marker_b", "marker_c"],
            "headline": {
                "status": "evaluable",
                "relative_MSE_gain_M3_vs_M0": 0.10,
                "positive_donor_fraction_M3_vs_M0": 0.80,
            },
            "scores": {
                "M0": {
                    "rare_state_recall_coverage": {
                        target: dict(row) for target, row in rare_state.items()
                    }
                },
                "M3": {
                    "variance_preservation": {"median_within_section_variance_ratio": 0.75},
                    "rare_state_recall_coverage": {
                        target: dict(row) for target, row in rare_state_m3.items()
                    },
                },
            },
            "coverage_uncertainty_abstention": {
                "M3": {
                    "median_type_coverage": 0.80,
                    "abstention_fraction": 0.10,
                }
            },
            "paired_inference": {
                comparison: {"mean_effect": 0.1, "exact_sign_flip_p": 0.001}
                for comparison in runner.ALL_REGISTERED_COMPARISONS
            },
            "indication_balance": {"heterogeneity_passed": True},
        }
        if endpoint == "program_total":
            experiments[name]["v2_reliability_qualified_program_quality"] = (
                qualified_program_quality
            )
        adjusted.update(
            {f"{name}::{comparison}": 0.01 for comparison in runner.ALL_REGISTERED_COMPARISONS}
        )
    return experiments, adjusted


def _effect_thresholds() -> dict[str, object]:
    protocol = json.loads(runner.PROTOCOL_PATH.read_text(encoding="utf-8"))
    return protocol["effect_thresholds"]


def _all_primary_experiments() -> dict[str, dict[str, object]]:
    experiments: dict[str, dict[str, object]] = {}
    for crop_id in runner.CROP_IDS:
        crop_experiments, _adjusted = _passing_crop_inputs(crop_id)
        experiments.update(crop_experiments)
    return experiments


def test_registered_p_value_family_is_exactly_64_keys_and_fills_absent_tests() -> None:
    experiments = _all_primary_experiments()
    absent_comparison = runner.ALL_REGISTERED_COMPARISONS[-1]
    partial_experiment = runner._primary_experiment_names("target_55um")[0]
    missing_experiment = runner._primary_experiment_names("context_112um")[0]
    del experiments[partial_experiment]["paired_inference"][absent_comparison]
    del experiments[missing_experiment]

    raw, missing = runner._registered_p_value_family(experiments)

    expected_keys = {
        f"{experiment_name}::{comparison}"
        for crop_id in runner.CROP_IDS
        for experiment_name in runner._primary_experiment_names(crop_id)
        for comparison in runner.ALL_REGISTERED_COMPARISONS
    }
    expected_missing = {
        f"{partial_experiment}::{absent_comparison}",
        *{
            f"{missing_experiment}::{comparison}"
            for comparison in runner.ALL_REGISTERED_COMPARISONS
        },
    }
    assert len(raw) == 64
    assert set(raw) == expected_keys
    assert set(missing) == expected_missing
    assert all(raw[key] == 1.0 for key in missing)
    assert all(raw[key] == 0.001 for key in expected_keys - expected_missing)


def test_registered_p_value_family_rejects_unregistered_comparisons() -> None:
    experiments = _all_primary_experiments()
    first = runner._primary_experiment_names("target_55um")[0]
    experiments[first]["paired_inference"]["unregistered_post_hoc_comparison"] = {
        "exact_sign_flip_p": 0.001
    }

    with pytest.raises(ValueError, match="unregistered paired comparisons"):
        runner._registered_p_value_family(experiments)


@pytest.mark.parametrize("invalid_p", (np.nan, -0.001, 1.001))
def test_registered_p_value_family_rejects_invalid_probabilities(invalid_p: float) -> None:
    experiments = _all_primary_experiments()
    first = runner._primary_experiment_names("target_55um")[0]
    comparison = runner.ALL_REGISTERED_COMPARISONS[0]
    experiments[first]["paired_inference"][comparison]["exact_sign_flip_p"] = invalid_p

    with pytest.raises(ValueError, match="paired comparison p-value is invalid"):
        runner._registered_p_value_family(experiments)


def test_crop_decision_does_not_require_m8_but_missing_primary_control_blocks() -> None:
    crop_id = "target_55um"
    experiments, adjusted = _passing_crop_inputs(crop_id)

    passed = runner._crop_decision(crop_id, experiments, adjusted, _effect_thresholds(), 0.05)

    assert passed["supported"] is True
    assert passed["M8_used_to_block"] is False
    assert all("M8" not in key for key in passed["registered_control_comparisons"])
    assert all(row["passed"] for row in passed["scientific_quality_guardrails"].values())

    first = runner._primary_experiment_names(crop_id)[0]
    missing = runner.ALL_REGISTERED_COMPARISONS[-1]
    del experiments[first]["paired_inference"][missing]
    del adjusted[f"{first}::{missing}"]
    blocked = runner._crop_decision(crop_id, experiments, adjusted, _effect_thresholds(), 0.05)
    assert blocked["status"] == "blocked_fail_closed"
    assert blocked["decision"] == "blocked_indeterminate"
    assert f"{first}::{missing}" in blocked["blocked_or_missing"]


@pytest.mark.parametrize(
    ("operator", "observed", "expected"),
    (
        (">=", 0.5, True),
        (">=", 0.5 - 0.5e-12, True),
        (">=", 0.5 - 2.0e-12, False),
        ("<=", 0.5, True),
        ("<=", 0.5 + 0.5e-12, True),
        ("<=", 0.5 + 2.0e-12, False),
    ),
)
def test_quality_threshold_boundaries_are_inclusive_with_declared_tolerance(
    operator: str, observed: float, expected: bool
) -> None:
    result = runner._threshold_check(
        name="boundary",
        observed=observed,
        threshold=0.5,
        operator=operator,
    )

    assert result["evaluable"] is True
    assert result["passed"] is expected


def test_crop_decision_passes_exact_registered_boundaries_but_not_p_above_alpha() -> None:
    crop_id = "target_55um"
    experiments, adjusted = _passing_crop_inputs(crop_id)
    for experiment in experiments.values():
        experiment["headline"]["relative_MSE_gain_M3_vs_M0"] = 0.05
        experiment["headline"]["positive_donor_fraction_M3_vs_M0"] = 0.70
        experiment["coverage_uncertainty_abstention"]["M3"].update(
            {"median_type_coverage": 0.5, "abstention_fraction": 0.5}
        )
        if experiment["endpoint"] == "program_total":
            qualified = experiment["v2_reliability_qualified_program_quality"]
            qualified["variance_preservation"]["M3"]["median_within_section_variance_ratio"] = 0.5
            qualified["rare_state_recall_coverage"]["M3"]["marker_a"]["recall"] = 0.5
            qualified["rare_state_recall_coverage"]["M3"]["marker_b"]["recall"] = 0.5
        else:
            experiment["scores"]["M3"]["variance_preservation"][
                "median_within_section_variance_ratio"
            ] = 0.5
    adjusted = {key: 0.05 for key in adjusted}

    boundary = runner._crop_decision(
        crop_id,
        experiments,
        adjusted,
        _effect_thresholds(),
        0.05,
    )

    assert boundary["supported"] is True
    assert all(
        row["passed"] for row in boundary["effect_size_donor_and_indication_checks"].values()
    )
    assert all(row["passed"] for row in boundary["scientific_quality_guardrails"].values())
    first_key = next(iter(adjusted))
    adjusted[first_key] = np.nextafter(0.05, np.inf)
    above_alpha = runner._crop_decision(
        crop_id,
        experiments,
        adjusted,
        _effect_thresholds(),
        0.05,
    )
    assert above_alpha["status"] == "evaluable"
    assert above_alpha["supported"] is False
    assert above_alpha["registered_control_comparisons"][first_key]["passed"] is False


def test_severe_indication_reversal_boundary_is_inclusive() -> None:
    summary = runner._indication_summary(
        _experiment_with_indication_m3({"breast": 9.0, "lung": 9.0, "dlbcl": 10.5}),
        _effect_thresholds(),
    )

    assert summary["per_indication"]["dlbcl"]["relative_MSE_gain_M3_vs_M0"] == pytest.approx(-0.05)
    assert summary["per_indication"]["dlbcl"]["severe_reversal"] is True
    assert summary["no_severe_reversal"] is False
    assert summary["heterogeneity_passed"] is False


@pytest.mark.parametrize(
    ("metric", "value", "reason"),
    (
        (
            "median_within_section_variance_ratio",
            0.49,
            "M3_median_within_section_variance_ratio:below_minimum",
        ),
        ("median_type_coverage", 0.49, "M3_median_type_coverage:below_minimum"),
        ("abstention_fraction", 0.51, "M3_abstention_fraction:above_maximum"),
    ),
)
def test_scientific_quality_threshold_failures_prevent_crop_support(
    metric: str, value: float, reason: str
) -> None:
    crop_id = "context_112um"
    experiments, adjusted = _passing_crop_inputs(crop_id)
    first = runner._primary_experiment_names(crop_id)[0]
    if metric == "median_within_section_variance_ratio":
        experiments[first]["v2_reliability_qualified_program_quality"]["variance_preservation"][
            "M3"
        ][metric] = value
    else:
        experiments[first]["coverage_uncertainty_abstention"]["M3"][metric] = value

    decision = runner._crop_decision(crop_id, experiments, adjusted, _effect_thresholds(), 0.05)

    guardrail = decision["scientific_quality_guardrails"][first]
    assert decision["status"] == "evaluable"
    assert decision["decision"] == "not_supported"
    assert decision["supported"] is False
    assert guardrail["status"] == "failed_threshold"
    assert reason in guardrail["failure_reasons"]


def test_program_guardrails_use_only_fold_qualified_metrics_and_fail_closed() -> None:
    crop_id = "context_112um"
    experiments, _ = _passing_crop_inputs(crop_id)
    first = runner._primary_experiment_names(crop_id)[0]
    experiment = experiments[first]
    qualified = experiment["v2_reliability_qualified_program_quality"]

    experiment["scores"]["M3"]["variance_preservation"]["median_within_section_variance_ratio"] = (
        99.0
    )
    qualified["variance_preservation"]["M3"]["median_within_section_variance_ratio"] = 0.49
    failed = runner._scientific_quality_guardrails(experiment, _effect_thresholds())
    assert failed["passed"] is False
    assert failed["variance_metric_source"].startswith("outer_training_reliability")

    qualified["variance_preservation"]["M3"]["median_within_section_variance_ratio"] = 0.75
    experiment["scores"]["M3"]["variance_preservation"]["median_within_section_variance_ratio"] = (
        0.0
    )
    for row in experiment["scores"]["M3"]["rare_state_recall_coverage"].values():
        if row["truth_positive"]:
            row["recall"] = 0.0
    passed = runner._scientific_quality_guardrails(experiment, _effect_thresholds())
    assert passed["passed"] is True
    assert passed["program_quality_payload_valid"] is True

    experiment.pop("v2_reliability_qualified_program_quality")
    blocked = runner._scientific_quality_guardrails(experiment, _effect_thresholds())
    assert blocked["evaluable"] is False
    assert blocked["program_quality_payload_valid"] is False
    assert "qualified_M0_or_M3_metrics_missing" in blocked["rare_state_collapse"]["reason"]


def test_rare_state_median_and_single_target_collapse_are_explicit() -> None:
    crop_id = "target_55um"
    experiments, _ = _passing_crop_inputs(crop_id)
    first = runner._primary_experiment_names(crop_id)[0]
    experiment = experiments[first]
    for target, recall in {"marker_a": 0.50, "marker_b": 0.30}.items():
        experiment["v2_reliability_qualified_program_quality"]["rare_state_recall_coverage"]["M3"][
            target
        ]["recall"] = recall

    median_failed = runner._scientific_quality_guardrails(experiment, _effect_thresholds())[
        "rare_state_collapse"
    ]

    assert median_failed["median_M0_minus_M3_recall_drop"] == pytest.approx(0.30)
    assert median_failed["median_drop_passed"] is False
    assert median_failed["single_target_drop_passed"] is True
    assert median_failed["violating_target_drops"] == {}
    assert median_failed["passed"] is False

    qualified_rare = experiment["v2_reliability_qualified_program_quality"][
        "rare_state_recall_coverage"
    ]["M3"]
    qualified_rare["marker_a"]["recall"] = 0.49
    qualified_rare["marker_b"]["recall"] = 0.60
    single_failed = runner._scientific_quality_guardrails(experiment, _effect_thresholds())[
        "rare_state_collapse"
    ]

    assert single_failed["median_M0_minus_M3_recall_drop"] == pytest.approx(0.155)
    assert single_failed["median_drop_passed"] is True
    assert single_failed["single_target_drop_passed"] is False
    assert single_failed["violating_target_drops"] == pytest.approx({"marker_a": 0.31})


def test_missing_quality_metrics_block_crop_and_m8_never_substitutes() -> None:
    crop_id = "context_112um"
    experiments, adjusted = _passing_crop_inputs(crop_id)
    first = runner._primary_experiment_names(crop_id)[0]
    experiments[first].pop("v2_reliability_qualified_program_quality")
    experiments[first]["scores"].pop("M3")
    experiments[first]["scores"]["M8"] = {
        "variance_preservation": {"median_within_section_variance_ratio": 99.0},
        "rare_state_recall_coverage": {},
    }
    experiments[first]["coverage_uncertainty_abstention"]["M8"] = {
        "median_type_coverage": 1.0,
        "abstention_fraction": 0.0,
    }

    decision = runner._crop_decision(crop_id, experiments, adjusted, _effect_thresholds(), 0.05)

    guardrail = decision["scientific_quality_guardrails"][first]
    assert decision["status"] == "blocked_fail_closed"
    assert decision["decision"] == "blocked_indeterminate"
    assert f"{first}::scientific_quality_guardrails" in decision["blocked_or_missing"]
    assert guardrail["evaluable"] is False
    assert guardrail["M8_used_to_block"] is False
    assert decision["M8_used_to_block"] is False


def test_calibration_adapter_uses_global_fallback_for_sparse_indications() -> None:
    calibrator = ReferenceCalibrator(
        coefficients=np.asarray([[2.0]]),
        source_mean=np.asarray([0.0]),
        target_mean=np.asarray([1.0]),
        fit_donors=("B1", "B3", "L1", "D1"),
        ridge_alpha=1.0,
        mode="indication_diagonal",
        pairing_unit="donor",
        indication_labels=("breast", "lung", "dlbcl"),
        indication_slopes=np.asarray([[3.0], [99.0], [99.0]]),
        indication_source_means=np.zeros((3, 1)),
        indication_target_means=np.asarray([[5.0], [99.0], [99.0]]),
        donor_indications=(),
        paired_summary_rows=4,
    )
    adapter = runner._CalibrationAdapter(
        calibrator=calibrator,
        reference_donor_ids=np.asarray(["B1", "L1", "D1"]),
        indication_fallback_to_global=frozenset(("lung", "dlbcl")),
        selection_receipt={},
    )

    transformed = adapter.transform([[1.0], [2.0], [3.0]])

    np.testing.assert_allclose(transformed[:, 0], [8.0, 5.0, 7.0])
    with pytest.raises(ValueError, match="aligned full reference matrix"):
        adapter.transform([[1.0]])


def test_v2_calibrator_excludes_non_fit_donor_outcomes_from_fit_and_cache() -> None:
    donor_ids = np.asarray(["B1", "B1", "B3", "D1", "D2", "L1", "L2", "B4"])
    fit_donors = ("B1", "B3", "D1", "D2", "L1", "L2")
    reference = np.asarray(
        [
            [0.0, 1.0],
            [0.5, 1.5],
            [2.0, 0.5],
            [3.0, 4.0],
            [4.0, 5.0],
            [6.0, 2.0],
            [7.0, 3.0],
            [99.0, 99.0],
        ]
    )
    target = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [5.0, 2.0],
            [7.0, 9.0],
            [9.0, 11.0],
            [13.0, 5.0],
            [15.0, 7.0],
            [10.0, 10.0],
        ]
    )
    altered_target = target.copy()
    altered_target[donor_ids == "B4"] = np.asarray([[-1.0e12, 1.0e12]])

    cached_bindings = runner._V2MethodBindings(object())
    first = cached_bindings.fit_calibrator(
        reference,
        donor_ids,
        target,
        donor_ids,
        fit_donors,
    )
    second = cached_bindings.fit_calibrator(
        reference,
        donor_ids,
        altered_target,
        donor_ids,
        fit_donors,
    )
    independent_bindings = runner._V2MethodBindings(object())
    independently_refit = independent_bindings.fit_calibrator(
        reference,
        donor_ids,
        altered_target,
        donor_ids,
        fit_donors,
    )

    first_receipt, second_receipt = cached_bindings.calibration_receipts
    independent_receipt = independent_bindings.calibration_receipts[0]
    expected_summaries = sorted(fit_donors)
    for receipt in (first_receipt, second_receipt, independent_receipt):
        assert receipt["paired_fit_donor_mean_summaries"] == expected_summaries
        assert "B4" not in receipt["paired_fit_donor_mean_summaries"]
        assert receipt["non_fit_donor_outcomes_used"] is False
        assert receipt["heldout_donor_outcomes_used"] is False

    assert first_receipt["calibration_cache_hit"] is False
    assert second_receipt["calibration_cache_hit"] is True
    assert independent_receipt["calibration_cache_hit"] is False
    assert len(cached_bindings._calibration_cache) == 1
    assert (
        first_receipt["calibration_cache_key"]
        == second_receipt["calibration_cache_key"]
        == independent_receipt["calibration_cache_key"]
    )
    assert (
        first_receipt["selected_ridge_alpha"]
        == second_receipt["selected_ridge_alpha"]
        == independent_receipt["selected_ridge_alpha"]
    )
    assert first.selection_receipt == second.selection_receipt
    assert first.selection_receipt == independently_refit.selection_receipt
    np.testing.assert_allclose(
        first.calibrator.coefficients,
        independently_refit.calibrator.coefficients,
    )
    np.testing.assert_allclose(
        first.calibrator.indication_slopes,
        independently_refit.calibrator.indication_slopes,
    )
    np.testing.assert_allclose(first.transform(reference), second.transform(reference))
    np.testing.assert_allclose(first.transform(reference), independently_refit.transform(reference))


@dataclasses.dataclass(frozen=True)
class _CropSource:
    path: Path
    spot_ids: np.ndarray
    image_features: np.ndarray
    section_ids: np.ndarray
    source_receipt: dict[str, object]
    blank_image_feature: np.ndarray
    blank_receipt: dict[str, object]


def _source_fixture(tmp_path: Path) -> tuple[_CropSource, dict[str, np.ndarray]]:
    sections = np.repeat(np.asarray(runner.EXPECTED_SECTIONS), 2)
    barcodes = np.asarray([f"bc{index:02d}" for index in range(len(sections))])
    spot_ids = np.char.add(np.char.add(sections, ":"), barcodes)
    pixel_xy = np.arange(len(sections) * 2, dtype=np.float64).reshape(-1, 2)
    native = np.ones((len(sections), 1536), dtype=np.float16)
    native[:, 0] = np.arange(1, len(sections) + 1, dtype=np.float16)
    parity = {
        "status": "passed",
        "schema": "heir.hoptimus1_official_local_parity.v1",
        "receipt_sha256": "1" * 64,
        "encoder_manifest_sha256": _sha256(runner.HOPTIMUS_MANIFEST_PATH),
        "implementation_sha256": "2" * 64,
        "runtime_sha256": "3" * 64,
    }
    source_sections = []
    for section in runner.EXPECTED_SECTIONS:
        rows = np.flatnonzero(sections == section)
        source_sections.append(
            {
                "section": section,
                "embedding": {
                    "crop": {
                        "physical_width_um": 112.0,
                        "crop_width_fullres_pixels": 224,
                    },
                    "registration_qc": {"maximum_padding_fraction": 0.0},
                    "barcodes_sha256": runner._array_sha256(np.asarray(barcodes[rows], dtype="S")),
                    "pixel_xy_sha256": runner._array_sha256(pixel_xy[rows]),
                },
            }
        )
    source_receipt = {
        "schema": "heir.natcommun_regional_source_receipt.v2",
        "builder_implementation_sha256": runner.FROZEN_V1_BUILDER_SHA256,
        "encoder": {
            "manifest_sha256": _sha256(runner.HOPTIMUS_MANIFEST_PATH),
            "repository": "bioptimus/H-optimus-1",
            "revision": "3592cb220dec7a150c5d7813fb56e68bd57473b9",
            "device": "cuda",
            "stored_feature_dtype": "float16",
            "official_local_parity": parity,
        },
        "sections": source_sections,
    }
    source_path = tmp_path / "source.npz"
    np.savez_compressed(
        source_path,
        schema_version=np.asarray("heir.natcommun_regional_source.v2"),
        spot_ids=spot_ids,
        barcode_ids=barcodes,
        section_ids=sections,
        pixel_xy=pixel_xy,
        image_features=native,
    )
    primary = sections != "B2_2"
    source = _CropSource(
        path=source_path,
        spot_ids=spot_ids[primary],
        image_features=native[primary].astype(np.float64),
        section_ids=sections[primary],
        source_receipt=source_receipt,
        blank_image_feature=np.ones(1536),
        blank_receipt={"encoder": "hoptimus1"},
    )
    return source, {
        "spot_ids": spot_ids,
        "barcodes": barcodes,
        "sections": sections,
        "pixel_xy": pixel_xy,
        "native": native,
    }


def _hoptimus_encoder(source: _CropSource) -> dict[str, object]:
    manifest = json.loads(runner.HOPTIMUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "manifest_path": str(runner.HOPTIMUS_MANIFEST_PATH),
        "manifest_sha256": _sha256(runner.HOPTIMUS_MANIFEST_PATH),
        "architecture": manifest["architecture"],
        "checkpoint_filename": manifest["checkpoint_filename"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "config_filename": manifest["config_filename"],
        "config_sha256": manifest["config_sha256"],
        "feature_width": 1536,
        "input_pixels": 224,
        "model_mpp": manifest["model_mpp"],
        "device": "cuda",
        "fine_tuning": "none_frozen_eval_inference",
        "official_local_parity": source.source_receipt["encoder"]["official_local_parity"],
    }


def _write_crop_supplement(
    path: Path,
    source: _CropSource,
    raw: dict[str, np.ndarray],
    *,
    receipt_mutator=None,
    feature_mutator=None,
) -> str:
    source_sha = _sha256(source.path)
    builder_sha = _sha256(runner.CROP_BUILDER_PATH)
    features = np.ones((len(raw["spot_ids"]), 1536), dtype=np.float16)
    features[:, 0] = np.arange(2, len(features) + 2, dtype=np.float16)
    encoder = _hoptimus_encoder(source)
    source_contract = {
        "path": str(source.path.resolve()),
        "sha256": source_sha,
        "schema": "heir.natcommun_regional_source.v2",
        "builder_implementation_sha256": runner.FROZEN_V1_BUILDER_SHA256,
        "spot_count": len(raw["spot_ids"]),
        "spot_ids_sha256": runner._array_sha256(np.asarray(raw["spot_ids"], dtype="S")),
        "native_112um_feature_stats": {
            "global": runner._strict_feature_stats(raw["native"], "native"),
            "per_section": {
                section: runner._strict_feature_stats(
                    raw["native"][raw["sections"] == section], section
                )
                for section in runner.EXPECTED_SECTIONS
            },
        },
    }
    section_receipts = []
    for section, source_section in zip(runner.EXPECTED_SECTIONS, source.source_receipt["sections"]):
        rows = np.flatnonzero(raw["sections"] == section).astype(np.int64)
        section_receipts.append(
            {
                "schema": runner.CROP_SECTION_CACHE_SCHEMA,
                "builder_implementation_sha256": builder_sha,
                "frozen_v1_builder": {"sha256": runner.FROZEN_V1_BUILDER_SHA256},
                "source_sha256": source_sha,
                "section": section,
                "row_indices_sha256": runner._array_sha256(rows),
                "spot_ids_sha256": runner._array_sha256(
                    np.asarray(raw["spot_ids"][rows], dtype="S")
                ),
                "barcodes_sha256": runner._array_sha256(
                    np.asarray(raw["barcodes"][rows], dtype="S")
                ),
                "pixel_xy_sha256": runner._array_sha256(raw["pixel_xy"][rows]),
                "source_crop": source_section["embedding"]["crop"],
                "target_crop": {
                    "construction": "white_outside_registered_center_square_on_native_112um_canvas",
                    "source_canvas_physical_width_um": 112.0,
                    "retained_center_physical_width_um": 55.0,
                    "centering_rule": (
                        "independent_floor(center_minus_width_over_two)_registered_bounds"
                    ),
                    "outside_value": "white_RGB_uint8_255",
                    "separate_55um_resize": False,
                },
                "registration_qc": source_section["embedding"]["registration_qc"],
                "encoder": encoder,
                "resampling": {
                    "target_canvas_pixels": [224, 224],
                    "implementation": "frozen_v1_Pillow.Image.Resampling.BICUBIC",
                    "qualified_against_official_loader": True,
                },
                "stored_feature_dtype": "float16",
                "feature_stats": runner._strict_feature_stats(features[rows], section),
            }
        )
    receipt = {
        "schema": runner.CROP_SUPPLEMENT_RECEIPT_SCHEMA,
        "builder_implementation_sha256": builder_sha,
        "source": source_contract,
        "crop_construction": {
            "source_canvas_physical_width_um": 112.0,
            "retained_center_physical_width_um": 55.0,
            "operation": (
                "extract_the_registered_112um_canvas_then_whiten_everything_outside_the_"
                "independently_registered_centered_55um_square"
            ),
            "white_value": "RGB_uint8_255",
            "resize_after_masking": (
                "same_single_frozen_v1_Pillow_bicubic_112um_canvas_to_224_pixels"
            ),
            "separate_55um_crop_resize_prohibited": True,
            "model_magnification_unchanged": True,
        },
        "encoder": encoder,
        "sections": section_receipts,
        "feature_stats": runner._strict_feature_stats(features, "global"),
        "row_alignment": {
            "output_spot_ids_exactly_equal_source": True,
            "all_source_rows_written_exactly_once": True,
        },
    }
    if receipt_mutator is not None:
        receipt_mutator(receipt)
    stored_features = features if feature_mutator is None else feature_mutator(features.copy())
    np.savez_compressed(
        path,
        schema_version=np.asarray(runner.CROP_SUPPLEMENT_SCHEMA),
        spot_ids=raw["spot_ids"],
        image_features_55um=stored_features,
        source_sha256=np.asarray(source_sha),
        receipt_json=np.asarray(json.dumps(receipt)),
    )
    return _sha256(path)


def test_crop_supplement_requires_complete_hash_bound_contract(tmp_path: Path) -> None:
    source, raw = _source_fixture(tmp_path)
    supplement = tmp_path / "crop.npz"
    supplement_sha = _write_crop_supplement(supplement, source, raw)

    loaded, receipt = runner._load_crop_supplement(
        supplement, supplement_sha, source, _sha256(runner.CROP_BUILDER_PATH)
    )

    assert loaded.shape == source.image_features.shape
    assert loaded.dtype == np.float64
    assert receipt["source"]["sha256"] == _sha256(source.path)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda receipt: receipt.update(builder_implementation_sha256="0" * 64),
        lambda receipt: receipt["encoder"].update(revision="tampered"),
        lambda receipt: receipt["encoder"].update(official_local_parity={}),
        lambda receipt: receipt["sections"][0]["feature_stats"].update(array_sha256="0" * 64),
        lambda receipt: receipt["sections"].pop(),
    ],
)
def test_crop_supplement_receipt_tampering_fails_closed(tmp_path: Path, mutator) -> None:
    source, raw = _source_fixture(tmp_path)
    path = tmp_path / "crop.npz"
    sha = _write_crop_supplement(path, source, raw, receipt_mutator=mutator)
    with pytest.raises(ValueError):
        runner._load_crop_supplement(path, sha, source, _sha256(runner.CROP_BUILDER_PATH))


def test_crop_supplement_feature_dtype_and_content_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    source, raw = _source_fixture(tmp_path)
    for name, mutation in (
        ("content", lambda value: np.where(np.indices(value.shape)[0] == 0, value + 1, value)),
        ("dtype", lambda value: value.astype(np.float32)),
    ):
        path = tmp_path / f"crop_{name}.npz"
        sha = _write_crop_supplement(path, source, raw, feature_mutator=mutation)
        with pytest.raises(ValueError):
            runner._load_crop_supplement(path, sha, source, _sha256(runner.CROP_BUILDER_PATH))


def _uni2_implementation() -> dict[str, str]:
    return {
        "builder_sha256": _sha256(runner.UNI2_BUILDER_PATH),
        "uni2h_adapter_sha256": _sha256(runner.UNI2_ADAPTER_PATH),
        "encoder_base_sha256": _sha256(runner.ENCODER_BASE_PATH),
        "encoder_factory_sha256": _sha256(runner.ENCODER_FACTORY_PATH),
    }


def _uni2_encoder(source: _CropSource) -> dict[str, object]:
    manifest = json.loads(runner.UNI2_MANIFEST_PATH.read_text(encoding="utf-8"))
    model_dir = source.path.parent / "uni2_model"
    model_dir.mkdir(exist_ok=True)
    checkpoint = model_dir / manifest["checkpoint_filename"]
    config = model_dir / manifest["config_filename"]
    checkpoint.touch()
    config.touch()
    return {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "architecture": manifest["architecture"],
        "manifest_path": str(runner.UNI2_MANIFEST_PATH),
        "manifest_sha256": _sha256(runner.UNI2_MANIFEST_PATH),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "config_path": str(config),
        "config_sha256": manifest["config_sha256"],
        "feature_width": 1536,
        "input_pixels": 224,
        "model_mpp": manifest["model_mpp"],
        "normalization": manifest["normalization"],
        "interpolation": manifest["interpolation"],
        "pooling_rule": manifest["pooling_rule"],
        "license": manifest["license"],
        "known_training_datasets": manifest["known_training_datasets"],
        "evaluation_overlap": manifest["evaluation_overlap"],
        "device": "cuda",
        "fine_tuning": "none_frozen_eval_inference",
        "official_local_parity_claim": "none_not_assessed",
        "qualification_role": "manifest_hash_bound_secondary_sensitivity",
    }


def _write_uni2_supplement(
    path: Path,
    source: _CropSource,
    raw: dict[str, np.ndarray],
    *,
    receipt_mutator=None,
    payload_mutator=None,
) -> str:
    source_sha = _sha256(source.path)
    implementation = _uni2_implementation()
    encoder = _uni2_encoder(source)
    natural = np.ones((len(raw["spot_ids"]), 1536), dtype=np.float16)
    centered = np.ones_like(natural)
    natural[:, 0] = np.arange(3, len(natural) + 3, dtype=np.float16)
    centered[:, 1] = np.arange(4, len(centered) + 4, dtype=np.float16)
    blank = np.ones(1536, dtype=np.float16)
    blank_receipt = {
        "construction": "all_white_RGB_uint8_255_at_manifest_input_pixels",
        "applies_to": ["natural_112um", "centered_55um_whitened"],
        "semantic_reason": "an_all_white_canvas_is_identical_under_both_crop_constructions",
        "input_shape": [1, 224, 224, 3],
        "stored_dtype": "float16",
        "finite": True,
        "squared_norm": float(np.square(blank.astype(np.float64)).sum()),
        "array_sha256": runner._array_sha256(blank),
    }
    sections = []
    for section, source_section in zip(runner.EXPECTED_SECTIONS, source.source_receipt["sections"]):
        rows = np.flatnonzero(raw["sections"] == section).astype(np.int64)
        sections.append(
            {
                "schema": runner.UNI2_SECTION_CACHE_SCHEMA,
                "implementation": implementation,
                "frozen_v1_builder": {"sha256": runner.FROZEN_V1_BUILDER_SHA256},
                "source_sha256": source_sha,
                "section": section,
                "row_indices_sha256": runner._array_sha256(rows),
                "spot_ids_sha256": runner._array_sha256(
                    np.asarray(raw["spot_ids"][rows], dtype="S")
                ),
                "barcodes_sha256": runner._array_sha256(
                    np.asarray(raw["barcodes"][rows], dtype="S")
                ),
                "pixel_xy_sha256": runner._array_sha256(raw["pixel_xy"][rows]),
                "source_crop": source_section["embedding"]["crop"],
                "target_crop": {
                    "construction": "white_outside_registered_center_square_on_native_112um_canvas",
                    "source_canvas_physical_width_um": 112.0,
                    "retained_center_physical_width_um": 55.0,
                    "centering_rule": "independent_floor_registered_bounds",
                    "outside_value": "white_RGB_uint8_255",
                },
                "registration_qc": source_section["embedding"]["registration_qc"],
                "encoder": encoder,
                "preprocessing": {
                    "input_to_encoder": "native_registered_uint8_RGB_canvas",
                    "explicit_pre_encoder_resize": False,
                    "encoder_internal_resize": "torch_bilinear_align_corners_false_antialias_true",
                    "natural_112um": "unmodified_registered_canvas",
                    "centered_55um": "same_canvas_white_outside_registered_55um_square",
                },
                "stored_feature_dtype": "float16",
                "feature_stats": {
                    "natural_112um": runner._strict_feature_stats(natural[rows], section),
                    "centered_55um": runner._strict_feature_stats(centered[rows], section),
                },
            }
        )
    receipt = {
        "schema": runner.UNI2_SUPPLEMENT_RECEIPT_SCHEMA,
        "implementation": implementation,
        "source": {
            "path": str(source.path.resolve()),
            "sha256": source_sha,
            "schema": "heir.natcommun_regional_source.v2",
            "builder_implementation_sha256": runner.FROZEN_V1_BUILDER_SHA256,
            "spot_count": len(raw["spot_ids"]),
            "spot_ids_sha256": runner._array_sha256(np.asarray(raw["spot_ids"], dtype="S")),
        },
        "encoder": encoder,
        "preprocessing": {
            "natural_112um": "registered_native_canvas_passed_directly_to_encoder.encode",
            "centered_55um": (
                "same_native_canvas_white_outside_registered_center_55um_then_encoder.encode"
            ),
            "explicit_pre_encoder_resize": False,
            "only_resize": "UNI2HEncoder_manifest_bound_bilinear_interpolation",
            "official_local_parity_claim": "none_not_assessed",
        },
        "blank_image_control": blank_receipt,
        "sections": sections,
        "feature_stats": {
            "natural_112um": runner._strict_feature_stats(natural, "natural"),
            "centered_55um": runner._strict_feature_stats(centered, "centered"),
        },
        "row_alignment": {
            "output_spot_ids_exactly_equal_source": True,
            "all_source_rows_written_exactly_once": True,
        },
        "scientific_role": {
            "encoder": "secondary_sensitivity_scored_separately_from_H_optimus_1",
            "observation_level": "regional_Visium_v2_spot_not_cellular",
            "not_authorized": [
                "pooling_UNI2_h_with_H_optimus_1_primary_results",
                "official_local_UNI2_h_parity_claim",
            ],
        },
    }
    if receipt_mutator is not None:
        receipt_mutator(receipt)
    payload = {"natural": natural, "centered": centered, "blank": blank}
    if payload_mutator is not None:
        payload_mutator(payload)
    np.savez_compressed(
        path,
        schema_version=np.asarray(runner.UNI2_SUPPLEMENT_SCHEMA),
        spot_ids=raw["spot_ids"],
        image_features_112um=payload["natural"],
        image_features_55um=payload["centered"],
        blank_image_feature_vector=payload["blank"],
        source_sha256=np.asarray(source_sha),
        receipt_json=np.asarray(json.dumps(receipt)),
    )
    return _sha256(path)


def _load_uni2(path: Path, sha: str, source: _CropSource):
    implementation = _uni2_implementation()
    return runner._load_uni2_supplement(
        path,
        sha,
        source,
        expected_builder_sha256=implementation["builder_sha256"],
        expected_adapter_sha256=implementation["uni2h_adapter_sha256"],
        expected_encoder_base_sha256=implementation["encoder_base_sha256"],
        expected_encoder_factory_sha256=implementation["encoder_factory_sha256"],
    )


def _patch_uni2_model_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = json.loads(runner.UNI2_MANIFEST_PATH.read_text(encoding="utf-8"))
    original = runner._sha256

    def checked(path: Path) -> str:
        candidate = Path(path)
        if candidate.parent.name == "uni2_model":
            if candidate.name == manifest["checkpoint_filename"]:
                return str(manifest["checkpoint_sha256"])
            if candidate.name == manifest["config_filename"]:
                return str(manifest["config_sha256"])
        return original(candidate)

    monkeypatch.setattr(runner, "_sha256", checked)


def test_uni2_supplement_returns_separate_aligned_sources_and_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_uni2_model_hashes(monkeypatch)
    source, raw = _source_fixture(tmp_path)
    path = tmp_path / "uni2.npz"
    sha = _write_uni2_supplement(path, source, raw)

    crops, receipt = _load_uni2(path, sha, source)

    assert set(crops) == {"context_112um", "target_55um"}
    assert crops["context_112um"].image_features.shape == source.image_features.shape
    assert crops["target_55um"].blank_image_feature.shape == (1536,)
    assert crops["target_55um"].blank_receipt == receipt["blank_image_control"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda receipt: receipt["implementation"].update(uni2h_adapter_sha256="0" * 64),
        lambda receipt: receipt["encoder"].update(checkpoint_sha256="0" * 64),
        lambda receipt: receipt["preprocessing"].update(official_local_parity_claim="passed"),
        lambda receipt: receipt["blank_image_control"].update(array_sha256="0" * 64),
        lambda receipt: receipt["sections"][0]["feature_stats"]["natural_112um"].update(
            array_sha256="0" * 64
        ),
    ],
)
def test_uni2_supplement_receipt_tampering_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutator
) -> None:
    _patch_uni2_model_hashes(monkeypatch)
    source, raw = _source_fixture(tmp_path)
    path = tmp_path / "uni2.npz"
    sha = _write_uni2_supplement(path, source, raw, receipt_mutator=mutator)
    with pytest.raises(ValueError):
        _load_uni2(path, sha, source)


def test_uni2_supplement_matrix_and_blank_tampering_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_uni2_model_hashes(monkeypatch)
    source, raw = _source_fixture(tmp_path)
    mutations = (
        lambda payload: payload["natural"].__setitem__((0, 0), 99),
        lambda payload: payload.update(centered=payload["centered"].astype(np.float32)),
        lambda payload: payload["blank"].__setitem__(0, 99),
    )
    for index, mutation in enumerate(mutations):
        path = tmp_path / f"uni2_tampered_{index}.npz"
        sha = _write_uni2_supplement(path, source, raw, payload_mutator=mutation)
        with pytest.raises(ValueError):
            _load_uni2(path, sha, source)
