"""Focused tests for compact, hash-bound snPATHO public summaries."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from heir.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_snpatho_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("summarize_snpatho_benchmarks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)


def _metric_summary(value: float) -> dict:
    return {
        "median_gene_spearman": value,
        "median_gene_mse": 1.0 - value,
        "prediction_constant_scored_zero_count": 0,
    }


def _deepbench_fixture(
    tmp_path: Path,
    *,
    strict_ordering_pass: bool = True,
    followup_evidence_complete: bool = True,
) -> tuple:
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        """schema_version: heir.snpatho_deepbench_plan.v1
benchmark:
  name: snpatho_deepbench_v1
randomness:
  primary_seeds: [17, 41, 89, 131, 197]
execution:
  optional_artifacts:
    refined_predictions:
      sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    five_seed_predictions:
      sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    refinement_matrix_summary:
      sha256: dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
    native_scanvi_checkpoint:
      sha256: cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
""",
        encoding="utf-8",
    )
    methods = {}
    values = {
        SUMMARY.HEIR_ROUND0: 0.10,
        SUMMARY.HISTORICAL_HARD: 0.08,
        SUMMARY.HISTORICAL_SOFT: 0.07,
        SUMMARY.HISTORICAL_PSEUDOBULK: 0.00,
        SUMMARY.R1_HARD: 0.06,
        SUMMARY.R1_SOFT: 0.05,
        SUMMARY.REFINED: 0.11,
        SUMMARY.FINAL_SHUFFLE: 0.01,
    }
    for method, value in values.items():
        methods[method] = {
            "summary": _metric_summary(value),
            "per_gene": {"gene_names": ["g%d" % index for index in range(1000)]},
        }
    case = {
        "section_id": "4066",
        "qc": {
            "reference_nuclei": 20,
            "segmented_nuclei": 40,
            "spots_total": 12,
            "spots_at_least_3_nuclei": 10,
        },
        "reference_type_support": {
            "prediction_cell_types": ["A", "B"],
            "reference_supported_prediction_cell_types": ["A", "B"],
            "missing_prediction_cell_types": [],
            "hard_assignment_global_fallback_cells": 0,
        },
        "r1_reference_type_support": {"reference_supported_prediction_cell_types": ["A", "B"]},
        "provenance": {
            "r1_reference": {
                "selected_observations": 15,
                "manifest_sha256": "d" * 64,
            }
        },
        "methods": methods,
    }
    method_macro = {
        method: {
            "metrics": {
                metric: {"macro_mean": metric_value}
                for metric, metric_value in _metric_summary(value).items()
            }
        }
        for method, value in values.items()
    }
    repeated = {
        "null_permutations": 100,
        "null_median": 0.01,
        "null_empirical_percentile_interval_95": {"lower": -0.01, "upper": 0.02},
        "observed_heir_empirical_percentile_in_null": 0.9,
        "observed_heir_above_null_95_upper": True,
    }
    report = {
        "schema_version": SUMMARY.DEEPBENCH_SCHEMA,
        "benchmark": {
            "name": "snpatho_deepbench_v1",
            "plan_sha256": sha256_file(plan),
        },
        "reporting": {
            "full_plan_complete": False,
            "seeds_available": [17],
            "seeds_requested": [17, 41, 89, 131, 197],
        },
        "cases": [case],
        "method_macro": method_macro,
        "primary": {
            "bootstrap": {
                "ci_lower": -0.01,
                "ci_upper": 0.03,
                "bootstrap_fraction_delta_positive": 0.75,
            },
            "diagnostic_contrast": "round0_minus_hard",
            "diagnostic_status": "passes_available_criteria",
            "diagnostic_statistic": {
                "specimen_formula": "median_g(delta)",
                "macro_formula": "mean_d(median_g(delta))",
            },
            "macro_delta": 0.02,
            "requested_primary_contrast": "refined_minus_joint_matched_r1",
            "requested_primary_status": "developmental_joint_contrast_only_not_primary",
            "requested_primary_blockers": ["independent clean annotation is absent"],
            "requested_primary_contrast_requirement": {
                "status": "passes_developmental_joint_contrast",
                "endpoint": "paired_median_gene_spearman_delta",
                "evidence_scope": "developmental_one_seed_integrated_label_sensitivity",
                "full_primary_claim": False,
                "requires_both_contrasts": True,
                "refined_beats_both_matched_ffpe_r1_baselines": True,
                "contrasts": {
                    "matched_ffpe_r1_hard": {
                        "baseline_method": "hard",
                        "macro_delta": 0.03,
                        "macro_delta_positive": True,
                        "specimens": [
                            {
                                "section_id": "4066",
                                "median_paired_per_gene_spearman_delta": 0.03,
                            }
                        ],
                        "per_gene": [1.0] * 1000,
                    },
                    "matched_ffpe_r1_soft": {
                        "baseline_method": "soft",
                        "macro_delta": 0.02,
                        "macro_delta_positive": True,
                        "specimens": [
                            {
                                "section_id": "4066",
                                "median_paired_per_gene_spearman_delta": 0.02,
                            }
                        ],
                    },
                },
            },
            "full_primary_evidence": {
                "eligible_for_full_primary_claim": False,
                "gates": {
                    "prespecified_five_seed_matrix": True,
                    "scored_refinement_matrix_complete": True,
                    "refinement_matrix_strict_ordering_pass": strict_ordering_pass,
                    "required_negative_controls": True,
                    "required_followup_evidence_complete": followup_evidence_complete,
                    "execution_provenance_verified": True,
                },
                "refinement_matrix": {
                    "registered": True,
                    "matrix_status": "complete",
                    "strict_ordering_status": ("pass" if strict_ordering_pass else "fail"),
                    "summary_sha256": "d" * 64,
                },
            },
            "specimens": [
                {
                    "section_id": "4066",
                    (
                        "median_paired_per_gene_spearman_delta_vs_"
                        "historical_integrated_hard_type_mean"
                    ): 0.02,
                    "repeated_final_record_shuffle_null_comparison": repeated,
                }
            ],
        },
        "readiness": [
            {"component": "locked_round0_predictions", "status": "ready", "reason": "ready"},
            {
                "component": "five_seed_predictions",
                "status": "ready_provenance_validated_five_seed_matrix",
                "reason": "all planned artifacts were hash validated",
            },
            {
                "component": "refinement_matrix_summary",
                "status": (
                    "ready_provenance_validated_matrix_strict_ordering_passed"
                    if strict_ordering_pass
                    else "consumed_provenance_validated_matrix_strict_ordering_failed"
                ),
                "reason": "the complete scored matrix passed"
                if strict_ordering_pass
                else "the complete scored matrix failed strict ordering",
            },
            {
                "component": "primary_spot_qc",
                "status": "partial",
                "reason": "one requested QC covariate is absent",
            },
        ],
        "historical_lock": {
            "revalidation_plan_sha256": "e" * 64,
            "report_sha256": "f" * 64,
        },
        "reference_policy": {},
        "shuffle_policy": {"does_not_replace": ["image shuffle"]},
        "final_cell_record_shuffle_null": {
            "equal_weight_specimen_macro": {
                "permutations": 100,
                "median": 0.01,
                "empirical_percentile_interval_95": {"lower": -0.01, "upper": 0.02},
            }
        },
    }
    full_json = tmp_path / "report.json"
    full_json.write_text(json.dumps(report), encoding="utf-8")
    full_tsv = tmp_path / "report.tsv"
    full_tsv.write_text("row\n", encoding="utf-8")
    full_markdown = tmp_path / "report.md"
    full_markdown.write_text("# Report\n", encoding="utf-8")
    return plan, full_json, full_tsv, full_markdown, report


def _contains_key(value: object, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(_contains_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_deepbench_summary_preserves_contract_native_evidence_and_hashes(
    tmp_path: Path,
) -> None:
    plan, full_json, full_tsv, full_markdown, report = _deepbench_fixture(tmp_path)

    summary = SUMMARY.build_public_summary(
        full_json=full_json,
        plan_path=plan,
        full_tsv=full_tsv,
        full_markdown=full_markdown,
    )

    assert summary["schema"] == "heir.snpatho_deepbench_public_summary.v2"
    assert summary["report_schema"] == "heir.snpatho_deepbench.v2"
    assert summary["cases"]["4066"]["native_refined_developmental_median_gene_spearman"] == 0.11
    assert (
        summary["macro"][
            "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean"
        ]
        == 0.02
    )
    assert summary["macro"]["bootstrap_fraction_delta_positive"] == 0.75
    assert "paired_bootstrap_probability_positive" not in summary["macro"]
    assert (
        summary["native_refined_developmental_contrast"]["status"]
        == "passes_developmental_joint_contrast"
    )
    assert summary["five_seed_evidence"]["prespecified_gate_passed"] is True
    assert summary["five_seed_evidence"]["prediction_manifest_complete"] is True
    assert summary["five_seed_evidence"]["scored_matrix_complete"] is True
    assert summary["five_seed_evidence"]["strict_ordering_status"] == "pass"
    assert summary["five_seed_evidence"]["strict_ordering_passed"] is True
    assert summary["five_seed_evidence"]["required_negative_controls_complete"] is True
    assert summary["five_seed_evidence"]["required_followup_evidence_complete"] is True
    assert summary["five_seed_evidence"]["execution_provenance_verified"] is True
    assert (
        summary["five_seed_evidence"]["refinement_matrix_readiness"]["summary_sha256"] == "d" * 64
    )
    assert summary["five_seed_evidence"]["planned_seeds"] == [17, 41, 89, 131, 197]
    assert summary["provenance"]["deepbench_plan_sha256"] == sha256_file(plan)
    assert summary["provenance"]["full_local_json_sha256"] == sha256_file(full_json)
    assert summary["provenance"]["full_local_tsv_sha256"] == sha256_file(full_tsv)
    assert summary["provenance"]["full_local_markdown_sha256"] == sha256_file(full_markdown)
    assert summary["provenance"]["five_seed_prediction_manifest_sha256"] == "b" * 64
    assert summary["provenance"]["refinement_matrix_summary_sha256"] == "d" * 64
    assert not _contains_key(summary, "per_gene")
    assert len(json.dumps(summary)) < len(json.dumps(report)) // 5


def test_deepbench_summary_complete_matrix_failure_keeps_prespecified_gate_closed(
    tmp_path: Path,
) -> None:
    plan, full_json, _, _, _ = _deepbench_fixture(
        tmp_path,
        strict_ordering_pass=False,
    )

    summary = SUMMARY.build_public_summary(full_json=full_json, plan_path=plan)
    evidence = summary["five_seed_evidence"]

    assert evidence["prediction_manifest_complete"] is True
    assert evidence["scored_matrix_complete"] is True
    assert evidence["strict_ordering_status"] == "fail"
    assert evidence["strict_ordering_passed"] is False
    assert evidence["required_negative_controls_complete"] is True
    assert evidence["required_followup_evidence_complete"] is True
    assert evidence["prespecified_gate_passed"] is False
    assert evidence["refinement_matrix_readiness"]["status"] == (
        "consumed_provenance_validated_matrix_strict_ordering_failed"
    )


def test_deepbench_summary_missing_followup_evidence_keeps_prespecified_gate_closed(
    tmp_path: Path,
) -> None:
    plan, full_json, _, _, _ = _deepbench_fixture(
        tmp_path,
        followup_evidence_complete=False,
    )

    summary = SUMMARY.build_public_summary(full_json=full_json, plan_path=plan)
    evidence = summary["five_seed_evidence"]

    assert evidence["prediction_manifest_complete"] is True
    assert evidence["scored_matrix_complete"] is True
    assert evidence["strict_ordering_passed"] is True
    assert evidence["required_negative_controls_complete"] is True
    assert evidence["execution_provenance_verified"] is True
    assert evidence["required_followup_evidence_complete"] is False
    assert evidence["prespecified_gate_passed"] is False


def test_deepbench_cli_replaces_output_atomically_and_rejects_wrong_plan(
    tmp_path: Path,
) -> None:
    plan, full_json, full_tsv, full_markdown, report = _deepbench_fixture(tmp_path)
    output = tmp_path / "public" / "summary.json"
    output.parent.mkdir()
    output.write_text("old", encoding="utf-8")

    exit_code = SUMMARY.main(
        [
            "--full-json",
            str(full_json),
            "--plan",
            str(plan),
            "--full-tsv",
            str(full_tsv),
            "--full-markdown",
            str(full_markdown),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert (
        json.loads(output.read_text(encoding="utf-8"))["schema"] == SUMMARY.DEEPBENCH_PUBLIC_SCHEMA
    )
    assert not list(output.parent.glob("*.tmp"))

    report["benchmark"]["plan_sha256"] = "0" * 64
    full_json.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="plan SHA-256"):
        SUMMARY.build_public_summary(full_json=full_json, plan_path=plan)


def _matrix_fixture(tmp_path: Path) -> tuple:
    report = {
        "schema": SUMMARY.REFINEMENT_MATRIX_SCHEMA,
        "status": "blocked_evidence",
        "matrix_status": "complete",
        "primary_evidence_status": "blocked",
        "execution_provenance_verified": False,
        "execution_transform_hash_verified": False,
        "strict_ordering_status": "blocked",
        "analysis_role": "native_scanvi_published_integrated_annotation_sensitivity",
        "annotation_provenance": "published integrated labels",
        "request": {"samples": ["4066"], "seeds": [17, 41]},
        "requested_artifact_count": 4,
        "scored_artifact_count": 3,
        "manifests": {
            "frozen_truth": {"path": "/private/truth.json", "sha256": "a" * 64},
            "native_r1": {"path": "/private/native.json", "sha256": "b" * 64},
            "additional_evidence": None,
        },
        "inputs": {
            "4066": {
                "truth": {
                    "path": "/private/truth.npz",
                    "sha256": "c" * 64,
                    "hash_validation": "matched",
                },
                "native_r1_reference": {
                    "path": "/private/reference.npz",
                    "sha256": "d" * 64,
                    "hash_validation": "matched",
                },
            }
        },
        "evidence_ready": {},
        "macro_summaries": {
            "variants": {"refined": {"huge": [1.0] * 1000}},
            "contrasts": {
                "refined_minus_round0": {
                    "case_count": 2,
                    "evaluable_case_count": 2,
                    "median_of_case_median_deltas": 0.02,
                    "mean_of_case_median_deltas": 0.01,
                    "pooled_gene_case_median_delta": 0.015,
                    "pooled_gene_case_positive_fraction": 0.6,
                }
            },
        },
        "strict_ordering_checks": [
            {"name": "refined_gt_round0", "status": "pass"},
            {"name": "refined_gt_round0", "status": "fail"},
            {"name": "refined_gt_image_shuffle", "status": "blocked"},
        ],
        "strict_ordering_summary": {
            "status": "blocked",
            "pass_count": 1,
            "fail_count": 1,
            "blocked_count": 1,
            "required_policy": "refined must beat every comparator",
        },
        "matrix_blockers": [
            {
                "code": "missing_requested_artifact",
                "message": "prediction absent",
                "sample": "4066",
                "seed": 41,
                "variant": "image_shuffle",
                "path": "/private/prediction.npz",
            },
            {
                "code": "missing_requested_artifact",
                "message": "prediction absent",
                "sample": "4066",
                "seed": 17,
                "variant": "image_shuffle",
                "path": "/private/other.npz",
            },
        ],
        "evidence_blockers": [
            {
                "code": "missing_evidence_clean_independent_reannotation",
                "requirement": "clean_independent_reannotation",
                "message": "clean annotation absent",
                "path": None,
            }
        ],
        "cases": [{"methods": {"heir": {"per_gene": [1.0] * 1000}}}],
        "paired_gene_spearman_contrasts": [{"per_gene": [1.0] * 1000}],
    }
    report["blockers"] = report["matrix_blockers"] + report["evidence_blockers"]
    full_json = tmp_path / "matrix.json"
    full_json.write_text(json.dumps(report), encoding="utf-8")
    full_tsv = tmp_path / "matrix.tsv"
    full_tsv.write_text("row\n", encoding="utf-8")
    full_markdown = tmp_path / "matrix.md"
    full_markdown.write_text("# Matrix\n", encoding="utf-8")
    return full_json, full_tsv, full_markdown, report


def test_refinement_matrix_summary_is_macro_only_and_groups_check_blockers(
    tmp_path: Path,
) -> None:
    full_json, full_tsv, full_markdown, report = _matrix_fixture(tmp_path)

    summary = SUMMARY.build_public_summary(
        full_json=full_json,
        full_tsv=full_tsv,
        full_markdown=full_markdown,
    )

    assert summary["schema"] == "heir.snpatho_refinement_matrix_public_summary.v1"
    assert summary["strict_ordering_status"] == "blocked"
    assert summary["execution_provenance_verified"] is False
    assert summary["execution_transform_hash_verified"] is False
    assert (
        summary["macro_contrasts"]["refined_minus_round0"]["median_of_case_median_deltas"] == 0.02
    )
    assert summary["strict_ordering"]["check_counts"] == {
        "total": 3,
        "pass": 1,
        "fail": 1,
        "blocked": 1,
    }
    assert summary["strict_ordering"]["by_check"]["refined_gt_round0"]["total"] == 2
    assert summary["blockers"]["by_code"]["missing_requested_artifact"] == 2
    grouped = next(
        row for row in summary["blockers"]["groups"] if row["code"] == "missing_requested_artifact"
    )
    assert grouped["count"] == 2
    assert grouped["seeds"] == [17, 41]
    assert "/private/" not in json.dumps(summary)
    assert not _contains_key(summary, "per_gene")
    assert len(json.dumps(summary)) < len(json.dumps(report)) // 5
    assert summary["provenance"]["full_local_json_sha256"] == sha256_file(full_json)


def test_refinement_matrix_summary_rejects_inconsistent_check_counts(tmp_path: Path) -> None:
    full_json, _, _, report = _matrix_fixture(tmp_path)
    report["strict_ordering_summary"]["blocked_count"] = 0
    full_json.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="check counts are inconsistent"):
        SUMMARY.build_public_summary(full_json=full_json)


def test_refinement_matrix_summary_marks_legacy_wrong_donor_coverage_blocked(
    tmp_path: Path,
) -> None:
    full_json, _, _, report = _matrix_fixture(tmp_path)
    report["request"].update(
        {
            "samples": ["4066", "4399", "4411"],
            "control_seeds": [17, 41, 89],
            "wrong_donor_target": "4399",
            "wrong_donor_source": "4411",
        }
    )
    report["strict_ordering_checks"].extend(
        {"name": "refined_gt_wrong_donor", "status": "pass"} for _ in range(3)
    )
    report["strict_ordering_summary"]["pass_count"] += 3
    full_json.write_text(json.dumps(report), encoding="utf-8")

    summary = SUMMARY.build_public_summary(full_json=full_json)

    assert summary["effective_matrix_status"] == "blocked"
    assert summary["effective_strict_ordering_status"] == "blocked"
    assert summary["wrong_donor_coverage"] == {
        "status": "blocked_incomplete",
        "complete": False,
        "expected_pairing_count_per_control_seed": 6,
        "observed_pairing_count_per_control_seed": 1,
        "expected_case_count": 18,
        "observed_check_count": 3,
        "missing_case_count": 15,
        "missing_pairings": [
            {"target": "4066", "source": "4399"},
            {"target": "4066", "source": "4411"},
            {"target": "4399", "source": "4066"},
            {"target": "4411", "source": "4066"},
            {"target": "4411", "source": "4399"},
        ],
    }
