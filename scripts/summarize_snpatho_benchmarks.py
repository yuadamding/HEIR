#!/usr/bin/env python3
"""Build compact, hash-bound public summaries from local snPATHO reports.

The full DeepBench and refinement-matrix reports contain large per-gene arrays
that are useful locally but unsuitable for version control.  This script keeps
the public status, macro results, blockers, and provenance while binding the
summary to the full local outputs by SHA-256.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from heir.utils import sha256_file

DEEPBENCH_SCHEMA = "heir.snpatho_deepbench.v2"
DEEPBENCH_PLAN_SCHEMA = "heir.snpatho_deepbench_plan.v1"
DEEPBENCH_PUBLIC_SCHEMA = "heir.snpatho_deepbench_public_summary.v2"
REFINEMENT_MATRIX_SCHEMA = "heir.snpatho_refinement_matrix.v1"
REFINEMENT_MATRIX_PUBLIC_SCHEMA = "heir.snpatho_refinement_matrix_public_summary.v1"

HEIR_ROUND0 = "heir_round0_historical_integrated_reference_library_size_weighted"
HISTORICAL_HARD = "historical_integrated_hard_type_mean"
HISTORICAL_SOFT = "historical_integrated_soft_type_mean"
HISTORICAL_PSEUDOBULK = "historical_integrated_snrna_pseudobulk"
R1_HARD = "r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean"
R1_SOFT = "r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean"
REFINED = "refined_heir_matched_ffpe_r1_reference_library_size_weighted"
FINAL_SHUFFLE = (
    "heir_final_cell_record_shuffle_historical_integrated_reference_library_size_weighted"
)


def _json_object(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError("%s is absent: %s" % (name, path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("%s is not valid JSON: %s" % (name, path)) from error
    if not isinstance(payload, Mapping):
        raise ValueError("%s must contain a JSON object" % name)
    return payload


def _yaml_object(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError("%s is absent: %s" % (name, path))
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("%s is not valid YAML: %s" % (name, path)) from error
    if not isinstance(payload, Mapping):
        raise ValueError("%s must contain a YAML object" % name)
    return payload


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % name)
    return value


def _rows(value: object, name: str) -> list:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError("%s must be a list of objects" % name)
    return value


def _method_summary(
    case: Mapping[str, Any],
    method: str,
    *,
    required: bool = True,
) -> Optional[Mapping[str, Any]]:
    methods = _mapping(case.get("methods"), "DeepBench case methods")
    candidate = methods.get(method)
    if candidate is None and not required:
        return None
    row = _mapping(candidate, "DeepBench method %s" % method)
    return _mapping(row.get("summary"), "DeepBench method %s summary" % method)


def _metric(summary: Mapping[str, Any], name: str) -> Any:
    if name not in summary:
        raise ValueError("DeepBench method summary is missing %s" % name)
    return summary[name]


def _macro_metric(report: Mapping[str, Any], method: str, metric: str) -> Any:
    method_macro = _mapping(report.get("method_macro"), "DeepBench method_macro")
    row = _mapping(method_macro.get(method), "DeepBench macro method %s" % method)
    metrics = _mapping(row.get("metrics"), "DeepBench macro metrics for %s" % method)
    value = _mapping(metrics.get(metric), "DeepBench macro metric %s/%s" % (method, metric))
    if "macro_mean" not in value:
        raise ValueError("DeepBench macro metric %s/%s lacks macro_mean" % (method, metric))
    return value["macro_mean"]


def _optional_artifact_digest(plan: Mapping[str, Any], name: str) -> Optional[str]:
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        return None
    artifacts = execution.get("optional_artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    row = artifacts.get(name)
    if not isinstance(row, Mapping):
        return None
    digest = row.get("sha256")
    if digest is None:
        return None
    digest = str(digest)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("optional artifact %s has an invalid SHA-256" % name)
    return digest


def _referenced_file(path_value: object, anchors: Sequence[Path]) -> Optional[Path]:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve() if path.is_file() else None
    for anchor in anchors:
        for root in (anchor.parent, *anchor.parents):
            candidate = (root / path).resolve()
            if candidate.is_file():
                return candidate
    return None


def _native_refined_contrast(primary: Mapping[str, Any]) -> Dict[str, Any]:
    candidate = primary.get("requested_primary_contrast_requirement")
    if not isinstance(candidate, Mapping):
        return {
            "status": "unavailable",
            "evidence_scope": "not_reported",
            "full_primary_claim": False,
            "contrasts": {},
        }
    result = {
        key: candidate.get(key)
        for key in (
            "status",
            "endpoint",
            "evidence_scope",
            "full_primary_claim",
            "requires_both_contrasts",
            "refined_beats_both_matched_ffpe_r1_baselines",
            "refined_method",
            "required_baseline_methods",
            "required_contrasts",
            "success_formula",
            "missing",
        )
        if key in candidate
    }
    contrasts = candidate.get("contrasts")
    compact_contrasts: Dict[str, Any] = {}
    if isinstance(contrasts, Mapping):
        for name, value in sorted(contrasts.items()):
            if not isinstance(value, Mapping):
                raise ValueError("native refined contrast %s must be an object" % name)
            specimens = value.get("specimens", [])
            if not isinstance(specimens, list):
                raise ValueError("native refined contrast specimen rows must be a list")
            compact_contrasts[str(name)] = {
                key: value.get(key)
                for key in ("baseline_method", "macro_delta", "macro_delta_positive")
                if key in value
            }
            compact_contrasts[str(name)]["specimens"] = [
                {
                    key: row.get(key)
                    for key in ("section_id", "median_paired_per_gene_spearman_delta")
                    if key in row
                }
                for row in specimens
                if isinstance(row, Mapping)
            ]
    result["contrasts"] = compact_contrasts
    return result


def _readiness_by_component(report: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    rows = _rows(report.get("readiness"), "DeepBench readiness")
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        component = str(row.get("component", "")).strip()
        if not component or component in result:
            raise ValueError("DeepBench readiness components must be non-empty and unique")
        result[component] = row
    return result


def _five_seed_evidence(
    report: Mapping[str, Any],
    primary: Mapping[str, Any],
    readiness: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> Dict[str, Any]:
    prediction_row = readiness.get("five_seed_predictions")
    matrix_row = readiness.get("refinement_matrix_summary")
    reporting = _mapping(report.get("reporting"), "DeepBench reporting")
    full_evidence = primary.get("full_primary_evidence")
    gates = full_evidence.get("gates") if isinstance(full_evidence, Mapping) else None
    matrix = full_evidence.get("refinement_matrix") if isinstance(full_evidence, Mapping) else None
    plan_randomness = plan.get("randomness")
    planned_seeds = (
        plan_randomness.get("primary_seeds") if isinstance(plan_randomness, Mapping) else None
    )
    gate_values = {
        name: bool(isinstance(gates, Mapping) and gates.get(name) is True)
        for name in (
            "prespecified_five_seed_matrix",
            "scored_refinement_matrix_complete",
            "refinement_matrix_strict_ordering_pass",
            "required_negative_controls",
            "required_followup_evidence_complete",
            "execution_provenance_verified",
        )
    }
    planned_matrix_digest = _optional_artifact_digest(plan, "refinement_matrix_summary")
    reported_matrix_digest = matrix.get("summary_sha256") if isinstance(matrix, Mapping) else None
    if reported_matrix_digest is not None:
        reported_matrix_digest = str(reported_matrix_digest)
        if len(reported_matrix_digest) != 64 or any(
            character not in "0123456789abcdef" for character in reported_matrix_digest
        ):
            raise ValueError("refinement matrix summary has an invalid SHA-256")
    if (
        planned_matrix_digest is not None
        and reported_matrix_digest is not None
        and planned_matrix_digest != reported_matrix_digest
    ):
        raise ValueError("refinement matrix summary SHA-256 differs from the DeepBench plan")
    return {
        "status": "unavailable" if prediction_row is None else prediction_row.get("status"),
        "reason": (
            "No five-seed artifact was reported"
            if prediction_row is None
            else prediction_row.get("reason")
        ),
        "prediction_manifest_complete": gate_values["prespecified_five_seed_matrix"],
        "scored_matrix_complete": gate_values["scored_refinement_matrix_complete"],
        "strict_ordering_status": (
            matrix.get("strict_ordering_status") if isinstance(matrix, Mapping) else None
        ),
        "strict_ordering_passed": gate_values["refinement_matrix_strict_ordering_pass"],
        "required_negative_controls_complete": gate_values["required_negative_controls"],
        "required_followup_evidence_complete": gate_values["required_followup_evidence_complete"],
        "execution_provenance_verified": gate_values["execution_provenance_verified"],
        "prespecified_gate_passed": all(gate_values.values()),
        "refinement_matrix_readiness": {
            "status": "unavailable" if matrix_row is None else matrix_row.get("status"),
            "reason": (
                "No scored refinement-matrix summary was reported"
                if matrix_row is None
                else matrix_row.get("reason")
            ),
            "summary_sha256": reported_matrix_digest or planned_matrix_digest,
        },
        "planned_seeds": planned_seeds,
        "historical_reporting_seeds_available": reporting.get("seeds_available"),
        "historical_reporting_seeds_requested": reporting.get("seeds_requested"),
    }


def _deepbench_limitations(
    report: Mapping[str, Any],
    primary: Mapping[str, Any],
    readiness: Mapping[str, Mapping[str, Any]],
) -> list:
    values = []
    blockers = primary.get("requested_primary_blockers", [])
    if isinstance(blockers, list):
        values.extend(str(value) for value in blockers)
    for component in (
        "primary_ffpe_snpatho_reference",
        "primary_spot_qc",
        "hierarchical_spatial_bootstrap",
        "externally_frozen_ood_rule",
        "complete_negative_control_matrix",
        "untouched_external_cohort",
    ):
        row = readiness.get(component)
        if row is not None and row.get("reason"):
            values.append(str(row["reason"]))
    shuffle = report.get("shuffle_policy")
    if isinstance(shuffle, Mapping):
        does_not_replace = shuffle.get("does_not_replace")
        if isinstance(does_not_replace, list) and does_not_replace:
            values.append(
                "The final-cell-record shuffle does not replace: %s"
                % ", ".join(str(value) for value in does_not_replace)
            )
    return list(dict.fromkeys(values))


def _deepbench_provenance(
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    full_json: Path,
    plan_path: Path,
    full_tsv: Optional[Path],
    full_markdown: Optional[Path],
) -> Dict[str, Any]:
    historical = _mapping(report.get("historical_lock"), "DeepBench historical_lock")
    provenance: Dict[str, Any] = {
        "deepbench_plan_sha256": sha256_file(plan_path),
        "locked_plan_sha256": historical.get("revalidation_plan_sha256"),
        "locked_report_sha256": historical.get("report_sha256"),
        "full_local_json_sha256": sha256_file(full_json),
    }
    cases = _rows(report.get("cases"), "DeepBench cases")
    r1_digests = set()
    for case in cases:
        case_provenance = case.get("provenance")
        r1 = case_provenance.get("r1_reference") if isinstance(case_provenance, Mapping) else None
        if isinstance(r1, Mapping) and r1.get("manifest_sha256"):
            r1_digests.add(str(r1["manifest_sha256"]))
    if len(r1_digests) > 1:
        raise ValueError("DeepBench cases disagree on the R1 reference manifest SHA-256")
    if r1_digests:
        provenance["r1_reference_manifest_sha256"] = next(iter(r1_digests))
    reference_policy = report.get("reference_policy")
    workflow_path = (
        reference_policy.get("machine_readable_workflow_audit")
        if isinstance(reference_policy, Mapping)
        else None
    )
    resolved_workflow = _referenced_file(workflow_path, (plan_path, full_json))
    if resolved_workflow is not None:
        provenance["workflow_audit_sha256"] = sha256_file(resolved_workflow)
    for artifact, field in (
        ("refined_predictions", "refined_prediction_manifest_sha256"),
        ("five_seed_predictions", "five_seed_prediction_manifest_sha256"),
        ("refinement_matrix_summary", "refinement_matrix_summary_sha256"),
        ("native_scanvi_checkpoint", "native_scanvi_manifest_sha256"),
    ):
        digest = _optional_artifact_digest(plan, artifact)
        if digest is not None:
            provenance[field] = digest
    if full_tsv is not None:
        provenance["full_local_tsv_sha256"] = sha256_file(full_tsv)
    if full_markdown is not None:
        provenance["full_local_markdown_sha256"] = sha256_file(full_markdown)
    return provenance


def build_deepbench_public_summary(
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    full_json: Path,
    plan_path: Path,
    full_tsv: Optional[Path] = None,
    full_markdown: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reduce a full DeepBench v2 report to its public summary contract."""

    if report.get("schema_version") != DEEPBENCH_SCHEMA:
        raise ValueError("DeepBench report schema must be %s" % DEEPBENCH_SCHEMA)
    if plan.get("schema_version") != DEEPBENCH_PLAN_SCHEMA:
        raise ValueError("DeepBench plan schema must be %s" % DEEPBENCH_PLAN_SCHEMA)
    plan_digest = sha256_file(plan_path)
    benchmark = _mapping(report.get("benchmark"), "DeepBench benchmark")
    if benchmark.get("plan_sha256") != plan_digest:
        raise ValueError("DeepBench report plan SHA-256 does not match the supplied plan")
    plan_benchmark = _mapping(plan.get("benchmark"), "DeepBench plan benchmark")
    if plan_benchmark.get("name") != benchmark.get("name"):
        raise ValueError("DeepBench report and plan benchmark names differ")

    cases = _rows(report.get("cases"), "DeepBench cases")
    primary = _mapping(report.get("primary"), "DeepBench primary")
    primary_rows = {
        str(row.get("section_id")): row
        for row in _rows(primary.get("specimens"), "DeepBench primary specimens")
    }
    compact_cases: Dict[str, Any] = {}
    for case in cases:
        section = str(case.get("section_id", "")).strip()
        if not section or section in compact_cases:
            raise ValueError("DeepBench case section IDs must be non-empty and unique")
        qc = _mapping(case.get("qc"), "DeepBench case QC")
        support = _mapping(case.get("reference_type_support"), "DeepBench reference support")
        r1_support = _mapping(
            case.get("r1_reference_type_support"), "DeepBench R1 reference support"
        )
        provenance = _mapping(case.get("provenance"), "DeepBench case provenance")
        r1_reference = _mapping(provenance.get("r1_reference"), "DeepBench R1 provenance")
        primary_row = _mapping(primary_rows.get(section), "DeepBench primary specimen %s" % section)
        repeated = _mapping(
            primary_row.get("repeated_final_record_shuffle_null_comparison"),
            "DeepBench repeated shuffle comparison",
        )
        interval = _mapping(
            repeated.get("null_empirical_percentile_interval_95"),
            "DeepBench repeated shuffle interval",
        )
        heir = _method_summary(case, HEIR_ROUND0)
        hard = _method_summary(case, HISTORICAL_HARD)
        soft = _method_summary(case, HISTORICAL_SOFT)
        r1_hard = _method_summary(case, R1_HARD)
        r1_soft = _method_summary(case, R1_SOFT)
        pseudobulk = _method_summary(case, HISTORICAL_PSEUDOBULK)
        shuffle = _method_summary(case, FINAL_SHUFFLE)
        row = {
            "reference_nuclei": qc.get("reference_nuclei"),
            "ffpe_r1_reference_nuclei": r1_reference.get("selected_observations"),
            "segmented_nuclei": qc.get("segmented_nuclei"),
            "author_qc_spots": qc.get("spots_total"),
            "spots_at_least_3_nuclei": qc.get("spots_at_least_3_nuclei"),
            "prediction_types": len(support.get("prediction_cell_types", [])),
            "historical_reference_supported_types": len(
                support.get("reference_supported_prediction_cell_types", [])
            ),
            "ffpe_r1_reference_supported_types": len(
                r1_support.get("reference_supported_prediction_cell_types", [])
            ),
            "missing_prediction_types": support.get("missing_prediction_cell_types", []),
            "global_fallback_cells": support.get("hard_assignment_global_fallback_cells"),
            "heir_median_gene_spearman": _metric(heir, "median_gene_spearman"),
            "heir_median_gene_mse": _metric(heir, "median_gene_mse"),
            "historical_hard_type_mean_median_gene_spearman": _metric(hard, "median_gene_spearman"),
            "historical_hard_type_mean_median_gene_mse": _metric(hard, "median_gene_mse"),
            "historical_soft_type_mean_median_gene_spearman": _metric(soft, "median_gene_spearman"),
            "historical_soft_type_mean_median_gene_mse": _metric(soft, "median_gene_mse"),
            "ffpe_r1_hard_type_mean_median_gene_spearman": _metric(r1_hard, "median_gene_spearman"),
            "ffpe_r1_hard_type_mean_median_gene_mse": _metric(r1_hard, "median_gene_mse"),
            "ffpe_r1_soft_type_mean_median_gene_spearman": _metric(r1_soft, "median_gene_spearman"),
            "ffpe_r1_soft_type_mean_median_gene_mse": _metric(r1_soft, "median_gene_mse"),
            "final_cell_record_shuffle_draw_0_median_gene_spearman": _metric(
                shuffle, "median_gene_spearman"
            ),
            "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean": (
                primary_row.get(
                    "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean"
                )
            ),
            "repeated_final_cell_record_shuffle_null": {
                "permutations": repeated.get("null_permutations"),
                "median": repeated.get("null_median"),
                "empirical_ci_95": [interval.get("lower"), interval.get("upper")],
                "heir_empirical_percentile": repeated.get(
                    "observed_heir_empirical_percentile_in_null"
                ),
                "heir_above_null_95_upper": repeated.get("observed_heir_above_null_95_upper"),
            },
            "prediction_constant_scored_zero_counts": {
                "heir": _metric(heir, "prediction_constant_scored_zero_count"),
                "historical_hard_type_mean": _metric(hard, "prediction_constant_scored_zero_count"),
                "historical_soft_type_mean": _metric(soft, "prediction_constant_scored_zero_count"),
                "ffpe_r1_hard_type_mean": _metric(r1_hard, "prediction_constant_scored_zero_count"),
                "ffpe_r1_soft_type_mean": _metric(r1_soft, "prediction_constant_scored_zero_count"),
                "historical_pseudobulk": _metric(
                    pseudobulk, "prediction_constant_scored_zero_count"
                ),
            },
        }
        refined = _method_summary(case, REFINED, required=False)
        if refined is not None:
            row["native_refined_developmental_median_gene_spearman"] = _metric(
                refined, "median_gene_spearman"
            )
            row["native_refined_developmental_median_gene_mse"] = _metric(
                refined, "median_gene_mse"
            )
        compact_cases[section] = row

    bootstrap = _mapping(primary.get("bootstrap"), "DeepBench bootstrap")
    final_null = _mapping(
        report.get("final_cell_record_shuffle_null"), "DeepBench final shuffle null"
    )
    null_macro = _mapping(final_null.get("equal_weight_specimen_macro"), "DeepBench shuffle macro")
    null_interval = _mapping(
        null_macro.get("empirical_percentile_interval_95"), "DeepBench shuffle macro interval"
    )
    above_null = sum(
        bool(
            _mapping(
                row.get("repeated_final_record_shuffle_null_comparison"),
                "DeepBench repeated shuffle comparison",
            ).get("observed_heir_above_null_95_upper")
        )
        for row in primary_rows.values()
    )
    macro = {
        "heir_median_gene_spearman": _macro_metric(report, HEIR_ROUND0, "median_gene_spearman"),
        "heir_median_gene_mse": _macro_metric(report, HEIR_ROUND0, "median_gene_mse"),
        "historical_hard_type_mean_median_gene_spearman": _macro_metric(
            report, HISTORICAL_HARD, "median_gene_spearman"
        ),
        "historical_hard_type_mean_median_gene_mse": _macro_metric(
            report, HISTORICAL_HARD, "median_gene_mse"
        ),
        "historical_soft_type_mean_median_gene_spearman": _macro_metric(
            report, HISTORICAL_SOFT, "median_gene_spearman"
        ),
        "historical_soft_type_mean_median_gene_mse": _macro_metric(
            report, HISTORICAL_SOFT, "median_gene_mse"
        ),
        "ffpe_r1_hard_type_mean_median_gene_spearman": _macro_metric(
            report, R1_HARD, "median_gene_spearman"
        ),
        "ffpe_r1_hard_type_mean_median_gene_mse": _macro_metric(report, R1_HARD, "median_gene_mse"),
        "ffpe_r1_soft_type_mean_median_gene_spearman": _macro_metric(
            report, R1_SOFT, "median_gene_spearman"
        ),
        "ffpe_r1_soft_type_mean_median_gene_mse": _macro_metric(report, R1_SOFT, "median_gene_mse"),
        "final_cell_record_shuffle_draw_0_median_gene_spearman": _macro_metric(
            report, FINAL_SHUFFLE, "median_gene_spearman"
        ),
        "final_cell_record_shuffle_draw_0_median_gene_mse": _macro_metric(
            report, FINAL_SHUFFLE, "median_gene_mse"
        ),
        "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean": (
            primary.get("macro_delta")
        ),
        "paired_bootstrap_ci_95": [bootstrap.get("ci_lower"), bootstrap.get("ci_upper")],
        "bootstrap_fraction_delta_positive": bootstrap.get("bootstrap_fraction_delta_positive"),
        "repeated_final_cell_record_shuffle_null": {
            "permutations": null_macro.get("permutations"),
            "median": null_macro.get("median"),
            "empirical_ci_95": [null_interval.get("lower"), null_interval.get("upper")],
        },
        "heir_above_repeated_null_95_upper_in_specimens": above_null,
        "required_specimens_above_repeated_null_95_upper": len(compact_cases) // 2 + 1,
    }
    if REFINED in _mapping(report.get("method_macro"), "DeepBench method_macro"):
        macro["native_refined_developmental_median_gene_spearman"] = _macro_metric(
            report, REFINED, "median_gene_spearman"
        )
        macro["native_refined_developmental_median_gene_mse"] = _macro_metric(
            report, REFINED, "median_gene_mse"
        )

    readiness = _readiness_by_component(report)
    readiness_counts = Counter(str(row.get("status")) for row in readiness.values())
    diagnostic = _mapping(primary.get("diagnostic_statistic"), "DeepBench diagnostic statistic")
    reporting = _mapping(report.get("reporting"), "DeepBench reporting")
    full_evidence = primary.get("full_primary_evidence")
    eligible = (
        bool(full_evidence.get("eligible_for_full_primary_claim"))
        if isinstance(full_evidence, Mapping)
        else False
    )
    runtime = report.get("runtime")
    runtime_summary = (
        {
            key: runtime.get(key)
            for key in ("wall_seconds", "peak_rss_gib", "note")
            if key in runtime
        }
        if isinstance(runtime, Mapping)
        else {
            "wall_seconds": None,
            "peak_rss_gib": None,
            "note": "Runtime was not recorded in the full report",
        }
    )
    return {
        "schema": DEEPBENCH_PUBLIC_SCHEMA,
        "report_schema": DEEPBENCH_SCHEMA,
        "benchmark_scope": "retrospective_capture_aware_historical_round0_diagnostic",
        "full_plan_complete": bool(reporting.get("full_plan_complete")),
        "requested_primary_endpoint_testable": eligible,
        "requested_primary_endpoint": primary.get("requested_primary_contrast"),
        "diagnostic_endpoint": primary.get("diagnostic_contrast"),
        "diagnostic_specimen_formula": diagnostic.get("specimen_formula"),
        "diagnostic_macro_formula": diagnostic.get("macro_formula"),
        "diagnostic_endpoint_succeeded": str(primary.get("diagnostic_status", "")).startswith(
            "pass"
        ),
        "cases": compact_cases,
        "macro": macro,
        "native_refined_developmental_contrast": _native_refined_contrast(primary),
        "five_seed_evidence": _five_seed_evidence(report, primary, readiness, plan),
        "readiness": dict(sorted(readiness_counts.items())),
        "limitations": _deepbench_limitations(report, primary, readiness),
        "runtime": runtime_summary,
        "provenance": _deepbench_provenance(
            report,
            plan,
            full_json=full_json,
            plan_path=plan_path,
            full_tsv=full_tsv,
            full_markdown=full_markdown,
        ),
    }


def _compact_macro_contrasts(report: Mapping[str, Any]) -> Dict[str, Any]:
    macro = _mapping(report.get("macro_summaries"), "refinement matrix macro_summaries")
    contrasts = _mapping(macro.get("contrasts"), "refinement matrix macro contrasts")
    result: Dict[str, Any] = {}
    keys = (
        "case_count",
        "evaluable_case_count",
        "median_of_case_median_deltas",
        "mean_of_case_median_deltas",
        "pooled_gene_case_median_delta",
        "pooled_gene_case_positive_fraction",
    )
    for name, candidate in sorted(contrasts.items()):
        row = _mapping(candidate, "refinement matrix macro contrast %s" % name)
        result[str(name)] = {key: row.get(key) for key in keys if key in row}
    return result


def _compact_blockers(report: Mapping[str, Any]) -> Dict[str, Any]:
    blockers = _rows(report.get("blockers"), "refinement matrix blockers")
    matrix_blockers = _rows(report.get("matrix_blockers"), "refinement matrix matrix_blockers")
    evidence_blockers = _rows(
        report.get("evidence_blockers"), "refinement matrix evidence_blockers"
    )
    execution_blockers = _rows(
        report.get("execution_provenance_blockers", []),
        "refinement matrix execution_provenance_blockers",
    )
    by_code = Counter(str(row.get("code", "unspecified")) for row in blockers)
    by_requirement = Counter(
        str(row["requirement"]) for row in blockers if row.get("requirement") is not None
    )
    grouped: Dict[tuple, Dict[str, Any]] = {}
    samples: Dict[tuple, set] = defaultdict(set)
    seeds: Dict[tuple, set] = defaultdict(set)
    variants: Dict[tuple, set] = defaultdict(set)
    for row in blockers:
        key = (
            str(row.get("code", "unspecified")),
            None if row.get("requirement") is None else str(row.get("requirement")),
            str(row.get("message", "")),
        )
        if key not in grouped:
            grouped[key] = {
                "code": key[0],
                "requirement": key[1],
                "message": key[2],
                "count": 0,
            }
        grouped[key]["count"] += 1
        if row.get("sample") is not None:
            samples[key].add(str(row["sample"]))
        if row.get("seed") is not None:
            seeds[key].add(int(row["seed"]))
        if row.get("variant") is not None:
            variants[key].add(str(row["variant"]))
    details = []
    for key, value in sorted(grouped.items(), key=lambda item: item[0]):
        value["samples"] = sorted(samples[key])
        value["seeds"] = sorted(seeds[key])
        value["variants"] = sorted(variants[key])
        details.append(value)
    return {
        "total_count": len(blockers),
        "matrix_count": len(matrix_blockers),
        "evidence_count": len(evidence_blockers),
        "execution_provenance_count": len(execution_blockers),
        "by_code": dict(sorted(by_code.items())),
        "by_requirement": dict(sorted(by_requirement.items())),
        "groups": details,
    }


def _matrix_provenance(
    report: Mapping[str, Any],
    *,
    full_json: Path,
    full_tsv: Optional[Path],
    full_markdown: Optional[Path],
) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {"full_local_json_sha256": sha256_file(full_json)}
    if full_tsv is not None:
        provenance["full_local_tsv_sha256"] = sha256_file(full_tsv)
    if full_markdown is not None:
        provenance["full_local_markdown_sha256"] = sha256_file(full_markdown)
    manifests = _mapping(report.get("manifests"), "refinement matrix manifests")
    provenance["manifests"] = {
        str(name): {
            key: row.get(key)
            for key in (
                "sha256",
                "schema",
                "manifest_role",
                "execution_mode",
                "stage_count",
                "original_execution_source_verified",
                "execution_transform_hash_verified",
                "execution_provenance_verified",
            )
            if isinstance(row, Mapping) and row.get(key) is not None
        }
        if isinstance(row, Mapping)
        else None
        for name, row in sorted(manifests.items())
    }
    inputs = _mapping(report.get("inputs"), "refinement matrix inputs")
    compact_inputs: Dict[str, Any] = {}
    for sample, candidate in sorted(inputs.items()):
        row = _mapping(candidate, "refinement matrix input %s" % sample)
        compact_inputs[str(sample)] = {}
        for name in ("truth", "native_r1_reference"):
            artifact = row.get(name)
            if isinstance(artifact, Mapping):
                compact_inputs[str(sample)][name] = {
                    key: artifact.get(key)
                    for key in ("sha256", "hash_validation")
                    if artifact.get(key) is not None
                }
    provenance["inputs"] = compact_inputs
    evidence_ready = report.get("evidence_ready")
    if isinstance(evidence_ready, Mapping):
        provenance["evidence_ready"] = {
            str(name): {
                key: row.get(key)
                for key in ("status", "sha256")
                if isinstance(row, Mapping) and row.get(key) is not None
            }
            for name, row in sorted(evidence_ready.items())
            if isinstance(row, Mapping)
        }
    return provenance


def build_refinement_matrix_public_summary(
    report: Mapping[str, Any],
    *,
    full_json: Path,
    full_tsv: Optional[Path] = None,
    full_markdown: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reduce a full refinement matrix to macro/check/blocker evidence."""

    if report.get("schema") != REFINEMENT_MATRIX_SCHEMA:
        raise ValueError("refinement matrix report schema must be %s" % REFINEMENT_MATRIX_SCHEMA)
    checks = _rows(report.get("strict_ordering_checks"), "refinement matrix strict ordering checks")
    status_counts = Counter(str(row.get("status", "unspecified")) for row in checks)
    check_names: Dict[str, Counter] = defaultdict(Counter)
    for row in checks:
        check_names[str(row.get("name", "unspecified"))][str(row.get("status", "unspecified"))] += 1
    stated = _mapping(
        report.get("strict_ordering_summary"), "refinement matrix strict ordering summary"
    )
    for status in ("pass", "fail", "blocked"):
        expected = stated.get(status + "_count")
        if expected is not None and int(expected) != status_counts[status]:
            raise ValueError("refinement matrix strict-ordering check counts are inconsistent")
    if stated.get("status") != report.get("strict_ordering_status"):
        raise ValueError("refinement matrix strict-ordering statuses are inconsistent")
    per_check = {}
    for name, counts in sorted(check_names.items()):
        per_check[name] = {
            "total": sum(counts.values()),
            "pass": counts["pass"],
            "fail": counts["fail"],
            "blocked": counts["blocked"],
        }
    request = _mapping(report.get("request"), "refinement matrix request")
    samples = request.get("samples", [])
    control_seeds = request.get("control_seeds", [])
    if not isinstance(samples, list) or any(not isinstance(value, str) for value in samples):
        raise ValueError("refinement matrix request samples must be a string list")
    if not isinstance(control_seeds, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in control_seeds
    ):
        raise ValueError("refinement matrix control seeds must be an integer list")
    expected_pairings = {
        (target, source) for target in samples for source in samples if source != target
    }
    raw_pairings = request.get("wrong_donor_pairings")
    if raw_pairings is None:
        raw_pairings = [
            {
                "target": request.get("wrong_donor_target"),
                "source": request.get("wrong_donor_source"),
            }
        ]
    if not isinstance(raw_pairings, list) or any(
        not isinstance(row, Mapping) for row in raw_pairings
    ):
        raise ValueError("refinement matrix wrong-donor pairings are malformed")
    observed_pairings = {(str(row.get("target")), str(row.get("source"))) for row in raw_pairings}
    wrong_check_count = sum(
        sum(counts.values())
        for name, counts in check_names.items()
        if name == "refined_gt_wrong_donor"
    )
    expected_wrong_cases = len(expected_pairings) * len(control_seeds)
    missing_pairings = sorted(expected_pairings - observed_pairings)
    missing_wrong_cases = max(0, expected_wrong_cases - wrong_check_count)
    wrong_donor_coverage = {
        "status": (
            "complete"
            if observed_pairings == expected_pairings and missing_wrong_cases == 0
            else "blocked_incomplete"
        ),
        "complete": observed_pairings == expected_pairings and missing_wrong_cases == 0,
        "expected_pairing_count_per_control_seed": len(expected_pairings),
        "observed_pairing_count_per_control_seed": len(observed_pairings),
        "expected_case_count": expected_wrong_cases,
        "observed_check_count": wrong_check_count,
        "missing_case_count": missing_wrong_cases,
        "missing_pairings": [
            {"target": target, "source": source} for target, source in missing_pairings
        ],
    }
    return {
        "schema": REFINEMENT_MATRIX_PUBLIC_SCHEMA,
        "report_schema": REFINEMENT_MATRIX_SCHEMA,
        "status": report.get("status"),
        "matrix_status": report.get("matrix_status"),
        "effective_matrix_status": (
            report.get("matrix_status") if wrong_donor_coverage["complete"] else "blocked"
        ),
        "primary_evidence_status": report.get("primary_evidence_status"),
        "execution_provenance_verified": report.get("execution_provenance_verified") is True,
        "execution_transform_hash_verified": (
            report.get("execution_transform_hash_verified") is True
        ),
        "strict_ordering_status": report.get("strict_ordering_status"),
        "effective_strict_ordering_status": (
            report.get("strict_ordering_status") if wrong_donor_coverage["complete"] else "blocked"
        ),
        "analysis_role": report.get("analysis_role"),
        "annotation_provenance": report.get("annotation_provenance"),
        "request": report.get("request"),
        "wrong_donor_coverage": wrong_donor_coverage,
        "artifacts": {
            "requested": report.get("requested_artifact_count"),
            "scored": report.get("scored_artifact_count"),
        },
        "macro_contrasts": _compact_macro_contrasts(report),
        "strict_ordering": {
            "status": report.get("strict_ordering_status"),
            "required_policy": stated.get("required_policy"),
            "check_counts": {
                "total": len(checks),
                "pass": status_counts["pass"],
                "fail": status_counts["fail"],
                "blocked": status_counts["blocked"],
            },
            "by_check": per_check,
        },
        "blockers": _compact_blockers(report),
        "provenance": _matrix_provenance(
            report,
            full_json=full_json,
            full_tsv=full_tsv,
            full_markdown=full_markdown,
        ),
    }


def build_public_summary(
    *,
    full_json: Path,
    plan_path: Optional[Path] = None,
    full_tsv: Optional[Path] = None,
    full_markdown: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load, identify, validate, and compact one supported full report."""

    full_json = full_json.expanduser().resolve()
    full_tsv = None if full_tsv is None else full_tsv.expanduser().resolve()
    full_markdown = None if full_markdown is None else full_markdown.expanduser().resolve()
    for name, path in (("full TSV", full_tsv), ("full Markdown", full_markdown)):
        if path is not None and not path.is_file():
            raise ValueError("%s is absent: %s" % (name, path))
    report = _json_object(full_json, "full local report")
    if report.get("schema_version") == DEEPBENCH_SCHEMA:
        if plan_path is None:
            raise ValueError("--plan is required for a DeepBench report")
        plan_path = plan_path.expanduser().resolve()
        plan = _yaml_object(plan_path, "DeepBench plan")
        return build_deepbench_public_summary(
            report,
            plan,
            full_json=full_json,
            plan_path=plan_path,
            full_tsv=full_tsv,
            full_markdown=full_markdown,
        )
    if report.get("schema") == REFINEMENT_MATRIX_SCHEMA:
        if plan_path is not None:
            raise ValueError("--plan is only valid for a DeepBench report")
        return build_refinement_matrix_public_summary(
            report,
            full_json=full_json,
            full_tsv=full_tsv,
            full_markdown=full_markdown,
        )
    raise ValueError("unsupported full-report schema")


def _atomic_write_json(destination: Path, payload: Mapping[str, Any]) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-json", "--report", dest="full_json", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--full-tsv", "--tsv", dest="full_tsv", type=Path)
    parser.add_argument("--full-markdown", "--markdown", dest="full_markdown", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _arguments(argv)
    input_paths = {
        args.full_json.expanduser().resolve(),
        *(
            path.expanduser().resolve()
            for path in (args.plan, args.full_tsv, args.full_markdown)
            if path is not None
        ),
    }
    output = args.output.expanduser().resolve()
    if output in input_paths:
        raise ValueError("public summary output must differ from every input")
    summary = build_public_summary(
        full_json=args.full_json,
        plan_path=args.plan,
        full_tsv=args.full_tsv,
        full_markdown=args.full_markdown,
    )
    _atomic_write_json(output, summary)
    print(
        json.dumps(
            {
                "output": str(output),
                "schema": summary["schema"],
                "sha256": sha256_file(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
