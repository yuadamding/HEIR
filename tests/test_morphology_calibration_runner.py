from __future__ import annotations

import copy
import json
import os
import resource
from pathlib import Path
from typing import Mapping

import pytest

import heir.evaluation.morphology_calibration_runner as runner
from heir.evaluation.control_models import HEST_CROP_CONTRACT
from heir.evaluation.hierarchical_metrics import donor_section_type_coverage
from heir.evaluation.morphology_calibration import compile_actual_gate_calibration_receipt
from heir.evaluation.permutations import null_stratum_activity
from heir.evaluation.power import (
    ACTUAL_GATE_REPORT_SCHEMA,
    CALIBRATION_EVIDENCE_SCHEMA,
    GLOBAL_NULL_CONDITION,
    PRELIMINARY_ALTERNATIVE_CONDITION,
    REQUIRED_CALIBRATION_SCENARIOS,
    REQUIRED_COMPLETE_GATE_CHECKS,
    REQUIRED_HYPOTHESIS_DECISIONS,
    canonical_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "morphology_gate_calibration.json"


def _settings() -> Mapping[str, object]:
    content = json.loads(CONFIG.read_text(encoding="utf-8"))
    settings = copy.deepcopy(content["exact_gate_settings"])
    settings["confirmatory_design_binding"] = (
        runner.synthetic_completed_confirmatory_design_binding()
    )
    return settings


def test_checked_in_runner_config_waits_for_completed_h_meas_binding() -> None:
    with pytest.raises(ValueError, match="pending pre-H-MEAS"):
        runner.load_calibration_run_config(CONFIG)


def test_synthetic_calibration_builder_covers_the_frozen_experiment() -> None:
    binding = runner.synthetic_completed_confirmatory_design_binding()
    for scenario in REQUIRED_CALIBRATION_SCENARIOS:
        development, locked = runner.build_synthetic_calibration_pair(
            scenario,
            PRELIMINARY_ALTERNATIVE_CONDITION,
            0,
        )
        assert development.cohort_id == "SYNTHETIC_CALIBRATION"
        assert locked.cohort_id == "SYNTHETIC_CALIBRATION"
        assert development.authorizes_nucleus_intrinsic_claim is False
        assert locked.authorizes_nucleus_intrinsic_claim is False
        assert len(set(locked.donor_ids)) == 5
        assert tuple(development.crop_ids) == tuple(HEST_CROP_CONTRACT)
        assert len(development.crop_ids) == 18
        assert development.reference_split_ids == (
            "primary",
            "reference_hash_fold_0",
            "reference_hash_fold_1",
        )
        assert tuple(development.planned_stratum_ids) + tuple(locked.planned_stratum_ids) == tuple(
            binding["ordered_planned_stratum_ids"]
        )
        assert (
            development.planned_stratum_manifest_sha256
            == binding["planned_stratum_manifest_sha256"]
        )
        assert locked.planned_stratum_manifest_sha256 == binding["planned_stratum_manifest_sha256"]
        for artifact in (development, locked):
            observed = {
                "%s|%s|%s" % (donor, section, artifact.type_names[int(type_index)])
                for donor, section, type_index in zip(
                    artifact.donor_ids,
                    artifact.section_ids,
                    artifact.type_labels,
                )
            }
            assert observed <= set(artifact.planned_stratum_ids)
            assert artifact.coverage_audit["retained_fraction"] == pytest.approx(
                len(observed) / len(artifact.planned_stratum_ids)
            )
        development.validate_compatible(locked)

    _, missing_locked = runner.build_synthetic_calibration_pair(
        "missing_fine_types",
        PRELIMINARY_ALTERNATIVE_CONDITION,
        0,
    )
    assert set(missing_locked.type_labels.tolist()) == {0}

    development, _ = runner.build_synthetic_calibration_pair(
        "inactive_permutation_strata",
        PRELIMINARY_ALTERNATIVE_CONDITION,
        0,
    )
    activity = null_stratum_activity(
        development.donor_ids,
        development.type_labels,
        development.roi_ids,
    )
    assert activity["eligible_row_fraction"] == pytest.approx(0.95)

    _, locked = runner.build_synthetic_calibration_pair(
        "section_effects",
        PRELIMINARY_ALTERNATIVE_CONDITION,
        0,
    )
    section_coverage = donor_section_type_coverage(
        locked.donor_ids,
        locked.section_ids,
        locked.type_labels,
        minimum_support=20,
        num_types=len(locked.type_names),
    )
    assert section_coverage is not None
    assert section_coverage["retained_fraction"] == 1.0


def test_runner_calls_actual_entrypoint_hashes_reports_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _settings()
    original_affinity = os.sched_getaffinity(0)
    original_address_space_limit = resource.getrlimit(resource.RLIMIT_AS)
    pool_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    original_pool_environment = {name: os.environ.get(name) for name in pool_variables}
    settings_sha256 = canonical_sha256(settings)
    calls: list[Mapping[str, object]] = []
    reports: list[Mapping[str, object]] = []

    def fake_production_gate(development, locked, **kwargs):
        assert len(os.sched_getaffinity(0)) == 1
        for variable in pool_variables:
            assert os.environ[variable] == "1"
        assert development.cohort_id == "SYNTHETIC_CALIBRATION"
        assert locked.cohort_id == "SYNTHETIC_CALIBRATION"
        assert len(set(development.donor_ids.tolist())) == 10
        assert len(set(locked.donor_ids.tolist())) == 5
        assert tuple(development.crop_ids) == tuple(HEST_CROP_CONTRACT)
        assert len(development.reference_split_ids) == 3
        assert kwargs["final_inference"] is True
        assert kwargs["synthetic_calibration_mode"] is True
        assert kwargs["total_permutations"] == settings["permutations_per_null"]
        assert kwargs["permutation_seeds"] == tuple(settings["permutation_seeds"])
        assert kwargs["permutations_per_seed"] == settings["permutations_per_seed"]
        assert kwargs["ranks"] == tuple(settings["target_rank_grid"])
        assert kwargs["alphas"] == tuple(settings["ridge_penalty_grid"])
        assert (
            kwargs["confirmatory_analysis_plan_sha256"]
            == settings["confirmatory_analysis_plan_sha256"]
        )
        for name, expected in settings["gate_parameters"].items():
            assert kwargs[name] == expected
        calls.append(kwargs)
        seed_rows = [
            {
                "seed": seed,
                "required_unique_permutations": settings["permutations_per_seed"],
                "generated_unique_permutations": settings["permutations_per_seed"],
            }
            for seed in settings["permutation_seeds"]
        ]
        report = {
            "schema_version": ACTUAL_GATE_REPORT_SCHEMA,
            "component_pass": len(calls) % 2 == 0,
            "final_inference": True,
            "synthetic_calibration_execution": True,
            "scientific_authorization_suppressed": True,
            "calibration_exact_gate_settings_sha256": settings_sha256,
            "checks": {name: True for name in REQUIRED_COMPLETE_GATE_CHECKS},
            "hypothesis_decisions": {
                name: {"tested": True, "pass": False} for name in REQUIRED_HYPOTHESIS_DECISIONS
            },
            "morphology_source_conclusion": (
                "inconclusive"
                if len(calls) == 3
                else (
                    "no_morphology_specific_information"
                    if len(calls) % 2 == 1
                    else "mixed_intrinsic_and_contextual_information"
                )
            ),
            "permutation_control": {
                "total_permutations": settings["permutations_per_null"],
                "seeds": seed_rows,
            },
            "spatial_block_permutation_control": {
                "total_permutations": settings["permutations_per_null"],
                "seeds": seed_rows,
            },
            "authorizes_full_heir": False,
            "authorizes_population_inference": False,
            "authorizes_external_generalization": False,
            "authorizes_validated_regional_association": False,
            "authorizes_nucleus_intrinsic_claim": False,
            "authorizes_cell_intrinsic_claim": False,
            "test_report_serial": len(calls),
        }
        reports.append(report)
        return report

    monkeypatch.setattr(runner, "evaluate_morphology_ridge_gate", fake_production_gate)
    checkpoint = tmp_path / "calibration.checkpoint.json"
    plan = runner.CalibrationRunPlan(
        exact_gate_settings=settings,
        trials_per_condition=1,
        smoke_test=True,
        device="cpu",
    )
    execution = runner.run_actual_gate_calibration(
        plan,
        checkpoint_path=checkpoint,
    )
    assert len(calls) == 2 * len(REQUIRED_CALIBRATION_SCENARIOS)
    assert execution["production_contract_satisfied"] is False
    assert execution["authorizes_scientific_claims"] is False
    assert execution["authorizes_final_inference"] is False
    assert execution["synthetic_data_only"] is True
    assert execution["resource_limits"]["max_cpu_threads"] == 1
    assert execution["resource_limits"]["maximum_process_rss_gib"] == 16.0
    assert execution["resource_limits"]["maximum_address_space_gib"] == 64.0
    assert execution["resource_limits"]["process_isolation"] == "in_process_smoke"
    assert (
        execution["resource_limits"]["observed_thread_pools"]["address_space"]["maximum_gib"]
        == 64.0
    )
    assert (
        execution["resource_limits"]["observed_thread_pools"]["cpu_affinity"]["logical_cpu_count"]
        == 1
    )
    assert os.sched_getaffinity(0) == original_affinity
    assert resource.getrlimit(resource.RLIMIT_AS) == original_address_space_limit
    assert {name: os.environ.get(name) for name in pool_variables} == original_pool_environment
    evidence = execution["evidence"]
    assert evidence["schema"] == CALIBRATION_EVIDENCE_SCHEMA
    assert execution["evidence_content_sha256"] == canonical_sha256(evidence)
    contract = evidence["run_contract"]
    assert evidence["run_contract_sha256"] == canonical_sha256(contract)
    assert contract["base_seed"] == 1729
    assert contract["dgp_effect_spec"]["authorizing_boundary_calibration"] is False
    assert (
        "minimum meaningful effect"
        in contract["dgp_effect_spec"]["effect_definition"]["scientific_interpretation"]
    )
    assert "preliminary" in execution["non_authorizing_reason"]
    first_condition = evidence["scenario_results"][REQUIRED_CALIBRATION_SCENARIOS[0]][
        GLOBAL_NULL_CONDITION
    ]
    assert first_condition["trial_report_set_sha256"] == canonical_sha256(
        {"ordered_actual_gate_report_sha256": [canonical_sha256(reports[0])]}
    )
    assert first_condition["actual_gate_executions"] == 1
    assert first_condition["complete_gate_passes"] == 0
    assert first_condition["hypothesis_decision_passes"] == {
        name: 0 for name in REQUIRED_HYPOTHESIS_DECISIONS
    }
    assert (
        first_condition["morphology_source_conclusion_counts"]["no_morphology_specific_information"]
        == 1
    )
    second_null = evidence["scenario_results"][REQUIRED_CALIBRATION_SCENARIOS[1]][
        GLOBAL_NULL_CONDITION
    ]
    assert second_null["morphology_source_conclusion_counts"]["inconclusive"] == 1
    assert first_condition["permutation_nulls"]["local_roi_seed_counts"] == {
        "17": 333,
        "29": 333,
        "41": 333,
    }
    assert checkpoint.is_file()
    checkpoint_content = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_content["authorizes_final_inference"] is False

    def unexpected_gate_call(*args, **kwargs):
        raise AssertionError("completed calibration trials must resume from the checkpoint")

    monkeypatch.setattr(runner, "evaluate_morphology_ridge_gate", unexpected_gate_call)
    resumed = runner.run_actual_gate_calibration(plan, checkpoint_path=checkpoint)
    assert resumed["evidence"] == evidence

    tampered = json.loads(checkpoint.read_text(encoding="utf-8"))
    first_key = "%s.%s" % (REQUIRED_CALIBRATION_SCENARIOS[0], GLOBAL_NULL_CONDITION)
    tampered["completed_trials"][first_key][0]["component_pass"] = True
    checkpoint.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash differs"):
        runner.run_actual_gate_calibration(plan, checkpoint_path=checkpoint)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="preliminary.*cannot issue"):
        compile_actual_gate_calibration_receipt(
            settings,
            config["thresholds"],
            evidence,
            confidence_level=config["confidence_level"],
        )


def test_reduced_trials_require_explicit_non_authorizing_smoke_mode() -> None:
    plan = runner.CalibrationRunPlan(
        exact_gate_settings=_settings(),
        trials_per_condition=1,
    )
    with pytest.raises(ValueError, match="non-authorizing"):
        plan.validate()


def test_non_smoke_run_requires_dedicated_cli_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(runner.DEDICATED_PROCESS_ENV, raising=False)
    plan = runner.CalibrationRunPlan(
        exact_gate_settings=_settings(),
        trials_per_condition=runner.PRODUCTION_TRIALS_PER_CONDITION,
    )
    with pytest.raises(RuntimeError, match="dedicated calibration CLI"):
        runner.run_actual_gate_calibration(plan)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"max_cpu_threads": 0}, "max_cpu_threads"),
        ({"maximum_process_rss_gib": 0.0}, "maximum_process_rss_gib"),
        ({"maximum_address_space_gib": 0.0}, "maximum_address_space_gib"),
    ),
)
def test_runner_rejects_invalid_resource_limits(overrides, message: str) -> None:
    parameters = {
        "exact_gate_settings": _settings(),
        "trials_per_condition": 1,
        "smoke_test": True,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        runner.CalibrationRunPlan(**parameters).validate()


def test_runner_refuses_to_start_above_rss_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_process_rss_gib", lambda: 17.0)
    monkeypatch.setattr(runner, "_process_virtual_memory_gib", lambda: 0.5)
    plan = runner.CalibrationRunPlan(
        exact_gate_settings=_settings(),
        trials_per_condition=1,
        smoke_test=True,
        maximum_process_rss_gib=16.0,
    )
    with pytest.raises(MemoryError, match="RSS.*exceeds"):
        runner.run_actual_gate_calibration(plan)


def test_runner_refuses_to_start_above_address_space_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_process_virtual_memory_gib", lambda: 2.0)
    plan = runner.CalibrationRunPlan(
        exact_gate_settings=_settings(),
        trials_per_condition=1,
        smoke_test=True,
        maximum_process_rss_gib=1.0,
        maximum_address_space_gib=1.0,
    )
    with pytest.raises(MemoryError, match="virtual memory.*exceeds"):
        runner.run_actual_gate_calibration(plan)


def test_address_space_limit_never_raises_a_stricter_existing_soft_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    limits: list[tuple[int, int]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (8 * gib, resource.RLIM_INFINITY))
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, value: limits.append(value))
    monkeypatch.setattr(runner, "_process_virtual_memory_gib", lambda: 2.0)

    with runner._address_space_limit(16.0) as observed:
        assert observed["maximum_gib"] == 8.0
        assert observed["preexisting_soft_limit_preserved"] is True

    assert limits == [
        (8 * gib, resource.RLIM_INFINITY),
        (8 * gib, resource.RLIM_INFINITY),
    ]
