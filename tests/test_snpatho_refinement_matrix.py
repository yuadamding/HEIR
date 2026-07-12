"""Focused tests for the standalone native snPATHO refinement-matrix evaluator."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from scipy import sparse

from heir.data import PrototypeSet, RNAReference, SpatialTruthArtifact
from heir.inference import PredictionBundle
from heir.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_snpatho_refinement_matrix.py"
SPEC = importlib.util.spec_from_file_location("benchmark_snpatho_refinement_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)


def _prediction(
    path: Path,
    *,
    sample: str,
    seed: int,
    round_id: int,
    spot_linear_expression: np.ndarray,
    control: Optional[str] = None,
    wrong_donor_source: str = "donor_b",
    prototype_sha256: str = "b" * 64,
) -> None:
    nucleus_ids = np.asarray(["n%d" % index for index in range(6)])
    expression = np.repeat(np.asarray(spot_linear_expression, dtype=np.float32), 2, axis=0)
    probabilities = np.full((6, 2), 0.5, dtype=np.float32)
    bundle = PredictionBundle(
        nucleus_ids=nucleus_ids,
        coordinates_um=np.column_stack((np.arange(6), np.zeros(6))).astype(np.float32),
        type_probabilities=probabilities,
        type_names=np.asarray(["A", "B"]),
        labels=np.zeros(6, dtype=np.int64),
        prototype_probabilities=probabilities,
        prototype_ids=np.asarray(["pA", "pB"]),
        latent_mean=np.zeros((6, 2), dtype=np.float32),
        latent_variance=np.ones((6, 2), dtype=np.float32),
        expression_mean=np.log1p(expression),
        expression_lower=np.log1p(expression),
        expression_upper=np.log1p(expression),
        gene_names=np.asarray(["g1", "g2", "g3"]),
        unknown_probability=np.zeros(6, dtype=np.float32),
        abstain_score=np.zeros(6, dtype=np.float32),
        abstain=np.zeros(6, dtype=bool),
        ood_score=np.zeros(6, dtype=np.float32),
        refinement_round=round_id,
        expression_interval_semantics=PredictionBundle.CONDITIONAL_KNOWN_STATE,
        expression_mean_available=np.ones(6, dtype=bool),
        expression_interval_available=np.ones(6, dtype=bool),
        sample_id=sample,
        donor_id=sample,
        slide_id=sample,
        checkpoint_sha256="a" * 64,
        prototype_sha256=prototype_sha256,
        histology_sha256="c" * 64,
        latent_space_id="latent-native-test",
        model_version="test-v1",
        inference_seed=seed,
        latent_samples=2,
        probability_threshold=0.35,
        artifact_threshold=0.5,
        expression_space_id="log1p-cpm-10000-v1",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle.to_npz(path)
    flags = {
        "graph_node_shuffle": control == "graph_shuffle",
        "image_feature_shuffle": control == "image_shuffle",
        "no_graph": control == "no_graph",
        "prototype_only": control == "prototype_only",
        "wrong_donor": control == "wrong_donor",
        "prototype_donor_id": wrong_donor_source if control == "wrong_donor" else sample,
        "seed": seed,
    }
    telemetry = path.with_name("prediction.telemetry.json")
    telemetry.write_text(
        json.dumps(
            {
                "schema": "heir.inference_telemetry.v1",
                "prediction_sha256": sha256_file(path),
                "nuclei": 6,
                "negative_control": flags,
            }
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict:
    repository = tmp_path / "repository"
    root = repository / "artifacts" / "snpatho" / "r1_scanvi"
    sample = "sample_a"
    seed = 17
    genes = np.asarray(["g1", "g2", "g3"])
    nucleus_ids = np.asarray(["n%d" % index for index in range(6)])
    truth_linear = np.asarray([[10, 0, 1], [5, 5, 10], [0, 10, 3]], dtype=np.float32)
    round0_linear = np.asarray([[10, 5, 1], [0, 0, 3], [5, 10, 10]], dtype=np.float32)
    control_linear = np.full((3, 3), 2.0, dtype=np.float32)

    truth = SpatialTruthArtifact(
        observed_expression=np.log1p(truth_linear),
        gene_names=genes,
        spot_ids=np.asarray(["s0", "s1", "s2"]),
        nucleus_ids=nucleus_ids,
        nucleus_spot_index=np.asarray([0, 0, 1, 1, 2, 2]),
        spot_library_sizes=np.full(3, 100.0),
        spot_coordinates_px=np.asarray([[0, 0], [10, 0], [20, 0]], dtype=np.float64),
        nucleus_spot_distance_px=np.zeros(6),
        analysis_role="locked_validation",
        cohort_id="snpatho",
        donor_id=sample,
        specimen_id=sample,
        block_id=sample + "_FFPE",
        section_id=sample,
        outer_fold="outer",
        inner_fold="inner",
        barcode_suffix_policy="exact",
        spot_radius_px=5.0,
        source_artifacts=np.asarray(["counts", "manifest"]),
        source_sha256=np.asarray(["d" * 64, "f" * 64]),
        source_roles=np.asarray(["locked_spatial_counts", "shared_manifest"]),
    )
    truth_path = repository / "truth.npz"
    truth.save_npz(truth_path)
    reference = RNAReference(
        sample_id=sample,
        cell_ids=np.asarray(["a1", "a2", "b1", "b2"]),
        gene_ids=genes,
        counts=sparse.csr_matrix([[10, 0, 1], [8, 0, 1], [0, 10, 1], [0, 8, 1]]),
        library_sizes=np.asarray([20, 18, 20, 18], dtype=np.float64),
        latent=np.zeros((4, 2), dtype=np.float32),
        cell_type_labels=np.asarray(["A", "A", "B", "B"]),
        donor_ids=np.asarray([sample] * 4),
        sample_ids=np.asarray([sample] * 4),
        latent_space_id="latent-native-test",
        block_id=sample + "_FFPE",
        source_count_sha256="e" * 64,
    )
    reference_path = root / sample / "reference500_scanvi.npz"
    reference_path.parent.mkdir(parents=True)
    reference.save_npz(reference_path)

    truth_manifest = repository / "truth_manifest.json"
    truth_manifest.write_text(
        json.dumps(
            {
                "schema_version": "heir.snpatho_benchmark_plan.v1",
                "cases": [
                    {
                        "section_id": sample,
                        "truth": str(truth_path.relative_to(repository)),
                        "truth_sha256": sha256_file(truth_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    native_manifest = repository / "native_manifest.json"
    native_manifest.write_text(
        json.dumps(
            {
                "schema": "heir.snpatho_scanvi_r1_manifest.v1",
                "annotation_provenance": "published integrated test annotation",
                "latent_space_id": "latent-native-test",
                "expression_space_id": "log1p-cpm-10000-v1",
                "specimens": {
                    sample: {
                        "latent_reference": str(reference_path.relative_to(repository)),
                        "latent_reference_sha256": sha256_file(reference_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    round0 = root / sample / "model_refinement_r1_v1_seed17_round0" / "predictions.npz"
    refined_root = root / sample / "model_refinement_r1_v1_seed17_refined"
    _prediction(
        round0,
        sample=sample,
        seed=seed,
        round_id=0,
        spot_linear_expression=round0_linear,
    )
    _prediction(
        refined_root / "predictions.npz",
        sample=sample,
        seed=seed,
        round_id=4,
        spot_linear_expression=truth_linear,
    )
    for round_id in (1, 2, 3):
        _prediction(
            refined_root / ("round_%d" % round_id) / "predictions.npz",
            sample=sample,
            seed=seed,
            round_id=round_id,
            spot_linear_expression=round0_linear,
        )
    for control in ("prototype_only", "image_shuffle", "graph_shuffle", "no_graph"):
        _prediction(
            refined_root / ("control_" + control) / "predictions.npz",
            sample=sample,
            seed=seed,
            round_id=4,
            spot_linear_expression=control_linear,
            control=control,
        )
    wrong_donor_prototypes = (
        root
        / "donor_b"
        / "model_refinement_r1_v1_seed17_refined"
        / "prototypes"
        / "donor_b__donor_b.npz"
    )
    wrong_donor_prototypes.parent.mkdir(parents=True, exist_ok=True)
    PrototypeSet(
        prototype_ids=np.asarray(["pA", "pB"]),
        sample_ids=np.asarray(["donor_b", "donor_b"]),
        cell_type_labels=np.asarray(["A", "B"]),
        means=np.zeros((2, 2), dtype=np.float32),
        variances=np.ones((2, 2), dtype=np.float32),
        weights=np.asarray([0.5, 0.5]),
        donor_id="donor_b",
        block_id="donor_b_FFPE",
    ).save_npz(wrong_donor_prototypes)
    _prediction(
        refined_root / "control_wrong_donor_donor_b" / "predictions.npz",
        sample=sample,
        seed=seed,
        round_id=4,
        spot_linear_expression=control_linear,
        control="wrong_donor",
        prototype_sha256=sha256_file(wrong_donor_prototypes),
    )
    return {
        "repository": repository,
        "artifact_root": root,
        "truth_manifest_path": truth_manifest,
        "native_manifest_path": native_manifest,
        "samples": (sample,),
        "seeds": (seed,),
        "trajectory_seed": seed,
        "controls": MATRIX.DEFAULT_CONTROLS,
        "control_seeds": (seed,),
        "wrong_donor_target": sample,
        "wrong_donor_source": "donor_b",
        "minimum_nuclei": 2,
    }


def test_complete_matrix_scores_full_metrics_deltas_and_strict_ordering(tmp_path: Path) -> None:
    report = MATRIX.evaluate_matrix(**_fixture(tmp_path))

    assert report["matrix_status"] == "complete"
    assert report["strict_ordering_status"] == "pass"
    assert report["status"] == "blocked_evidence"
    assert report["primary_evidence_status"] == "blocked"
    assert report["execution_provenance_verified"] is False
    assert report["execution_transform_hash_verified"] is False
    assert [row["code"] for row in report["execution_provenance_blockers"]] == [
        "missing_refinement_run_manifest"
    ]
    assert len(report["evidence_blockers"]) == len(MATRIX.EVIDENCE_REQUIREMENTS)
    assert report["scored_artifact_count"] == report["requested_artifact_count"] == 10
    assert report["request"]["requested_wrong_donor_pairings"] == [
        {"target": "sample_a", "source": "donor_b"}
    ]
    assert report["request"]["expected_in_cohort_wrong_donor_pairings"] == []
    refined = next(case for case in report["cases"] if case["variant"] == "refined")
    assert set(refined["methods"]) == {
        MATRIX.METHOD,
        MATRIX.HARD_BASELINE,
        MATRIX.SOFT_BASELINE,
    }
    assert len(refined["methods"][MATRIX.METHOD]["per_gene"]["gene_names"]) == 3
    assert (
        refined["paired_gene_spearman_deltas"]["heir_minus_hard_baseline"]["summary"][
            "median_delta"
        ]
        > 0
    )
    assert report["macro_summaries"]["variants"]["refined"][MATRIX.METHOD]["case_count"] == 1
    assert (
        report["macro_summaries"]["contrasts"]["trajectory_round4_minus_round3"]["case_count"] == 1
    )
    assert [row["status"] for row in report["trajectory"]["sample_a"]] == [
        "scored",
        "scored",
        "scored",
        "scored",
    ]
    assert all(row["status"] == "pass" for row in report["strict_ordering_checks"])


def test_wrong_donor_requests_cover_all_six_directed_pairings(tmp_path: Path) -> None:
    samples = ("4066", "4399", "4411")
    requests = MATRIX.build_requests(
        artifact_root=tmp_path,
        samples=samples,
        seeds=(17,),
        trajectory_seed=17,
        controls=("wrong_donor",),
        control_seeds=(17,),
    )
    wrong = [request for request in requests if request.control == "wrong_donor"]
    assert len(wrong) == 6
    assert {(request.sample, request.prototype_donor_id) for request in wrong} == {
        (target, source) for target in samples for source in samples if source != target
    }
    assert len({request.case_id for request in wrong}) == 6
    assert all(request.prototype_source is not None for request in wrong)

    external = MATRIX.build_requests(
        artifact_root=tmp_path,
        samples=("sample_a",),
        seeds=(17,),
        trajectory_seed=17,
        controls=("wrong_donor",),
        control_seeds=(17,),
        wrong_donor_target="sample_a",
        wrong_donor_source="external_donor",
    )
    external_wrong = next(request for request in external if request.control == "wrong_donor")
    assert external_wrong.prototype_source == (
        tmp_path
        / "external_donor"
        / "model_refinement_r1_v1_seed17_refined"
        / "prototypes"
        / "external_donor__external_donor.npz"
    )


def test_external_wrong_donor_without_resolvable_source_bank_blocks_matrix(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    source = (
        arguments["artifact_root"]
        / "donor_b"
        / "model_refinement_r1_v1_seed17_refined"
        / "prototypes"
        / "donor_b__donor_b.npz"
    )
    source.unlink()

    report = MATRIX.evaluate_matrix(**arguments)

    assert report["matrix_status"] == "blocked"
    assert any(
        row["code"] == "missing_requested_artifact"
        and "missing requested wrong-donor source prototype" in row["message"]
        for row in report["matrix_blockers"]
    )


def test_wrong_donor_summary_reports_mean_worst_and_site_matched() -> None:
    samples = ("4066", "4399", "4411")
    pairings = tuple(
        (target, source) for target in samples for source in samples if source != target
    )
    cases = []
    checks = []
    deltas = {}
    for index, (target, source) in enumerate(pairings, start=1):
        case_id = "%s/seed17/wrong_donor_%s" % (target, source)
        delta = index / 100.0
        deltas[(target, source)] = delta
        cases.append(
            {
                "case_id": case_id,
                "sample": target,
                "seed": 17,
                "control": "wrong_donor",
                "prototype_donor_id": source,
            }
        )
        checks.append(
            {
                "name": "refined_gt_wrong_donor",
                "sample": target,
                "seed": 17,
                "right_case_id": case_id,
                "status": "pass",
                "paired_median_per_gene_spearman_delta": delta,
            }
        )

    summary = MATRIX._wrong_donor_summary(
        cases,
        checks,
        samples=samples,
        control_seeds=(17,),
        sample_sites=MATRIX.DEFAULT_SAMPLE_SITES,
        wrong_donor_pairings=pairings,
    )

    assert summary["coverage_complete"] is True
    assert summary["all_directed"]["mean_paired_median_delta"] == pytest.approx(
        np.mean(list(deltas.values()))
    )
    assert summary["all_directed"]["worst_paired_median_delta"] == pytest.approx(0.01)
    assert summary["site_matched"]["expected_case_count"] == 2
    assert summary["site_matched"]["mean_paired_median_delta"] == pytest.approx(
        np.mean([deltas[("4399", "4411")], deltas[("4411", "4399")]])
    )


def test_prediction_binding_rejects_run_manifest_input_mismatch(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    request = MATRIX.build_requests(
        artifact_root=arguments["artifact_root"],
        samples=arguments["samples"],
        seeds=arguments["seeds"],
        trajectory_seed=arguments["trajectory_seed"],
        controls=(),
        control_seeds=(),
        wrong_donor_target=arguments["wrong_donor_target"],
        wrong_donor_source=arguments["wrong_donor_source"],
    )[0]
    sample_inputs = MATRIX.load_sample_inputs(
        sample="sample_a",
        truth_manifest_path=arguments["truth_manifest_path"],
        truth_manifest=MATRIX._json_object(arguments["truth_manifest_path"], "truth"),
        native_manifest_path=arguments["native_manifest_path"],
        native_manifest=MATRIX._json_object(arguments["native_manifest_path"], "native"),
        repository=arguments["repository"],
    )
    prediction = PredictionBundle.from_npz(request.prediction)
    stage = {
        "stage_id": "sample_a/seed17/predict_round0",
        "outputs": {
            "prediction": {"sha256": sha256_file(request.prediction)},
            "telemetry": {"sha256": sha256_file(request.telemetry)},
        },
        "inputs": {
            "checkpoint": {"sha256": prediction.checkpoint_sha256},
            "prototype": {"sha256": prediction.prototype_sha256},
            "histology": {"sha256": prediction.histology_sha256},
            "ood": {"sha256": prediction.ood_sha256},
        },
    }
    MATRIX.load_prediction(
        request,
        sample_inputs,
        wrong_donor_source=arguments["wrong_donor_source"],
        run_stage=stage,
    )
    stage["inputs"]["checkpoint"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="run-manifest checkpoint input"):
        MATRIX.load_prediction(
            request,
            sample_inputs,
            wrong_donor_source=arguments["wrong_donor_source"],
            run_stage=stage,
        )


def test_missing_control_and_telemetry_hash_mismatch_are_explicit_blockers(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    refined = arguments["artifact_root"] / "sample_a" / "model_refinement_r1_v1_seed17_refined"
    (refined / "control_image_shuffle" / "predictions.npz").unlink()
    prototype_telemetry = refined / "control_prototype_only" / "prediction.telemetry.json"
    payload = json.loads(prototype_telemetry.read_text(encoding="utf-8"))
    payload["prediction_sha256"] = "0" * 64
    prototype_telemetry.write_text(json.dumps(payload), encoding="utf-8")

    report = MATRIX.evaluate_matrix(**arguments)

    assert report["matrix_status"] == "blocked"
    assert report["strict_ordering_status"] == "blocked"
    codes = [row["code"] for row in report["matrix_blockers"]]
    assert "missing_requested_artifact" in codes
    assert "invalid_requested_artifact" in codes
    messages = "\n".join(row["message"] for row in report["matrix_blockers"])
    assert "prediction SHA-256 does not match inference telemetry" in messages
    checks = {row["name"]: row["status"] for row in report["strict_ordering_checks"]}
    assert checks["refined_gt_image_shuffle"] == "blocked"
    assert checks["round0_gt_prototype_only"] == "blocked"


def test_cli_writes_atomic_json_tsv_and_markdown(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    json_output = tmp_path / "out" / "matrix.json"
    tsv_output = tmp_path / "out" / "matrix.tsv"
    markdown_output = tmp_path / "out" / "matrix.md"
    exit_code = MATRIX.main(
        [
            "--repository",
            str(arguments["repository"]),
            "--artifact-root",
            str(arguments["artifact_root"]),
            "--truth-manifest",
            str(arguments["truth_manifest_path"]),
            "--native-manifest",
            str(arguments["native_manifest_path"]),
            "--sample",
            "sample_a",
            "--seed",
            "17",
            "--trajectory-seed",
            "17",
            "--control-seed",
            "17",
            "--wrong-donor-target",
            "sample_a",
            "--wrong-donor-source",
            "donor_b",
            "--minimum-nuclei",
            "2",
            "--json-output",
            str(json_output),
            "--tsv-output",
            str(tsv_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["strict_ordering_status"] == "pass"
    assert "method_summary\tsample\tseed" not in tsv_output.read_text(encoding="utf-8")
    assert tsv_output.read_text(encoding="utf-8").startswith("row_type\tsample\tseed")
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "full primary evidence: **blocked**" in markdown
    assert "Paired delta" in markdown
    assert "paired median across per-gene Spearman differences" in markdown
    tsv_rows = list(
        csv.DictReader(io.StringIO(tsv_output.read_text(encoding="utf-8")), delimiter="\t")
    )
    strict_row = next(row for row in tsv_rows if row["row_type"] == "strict_ordering")
    strict_check = payload["strict_ordering_checks"][0]
    assert strict_row["metric"] == "paired_median_per_gene_spearman_delta"
    assert float(strict_row["value"]) == pytest.approx(
        strict_check["paired_median_per_gene_spearman_delta"]
    )
    assert not list(json_output.parent.glob("*.tmp"))


def _evidence_manifest(
    tmp_path: Path,
    requirement: str,
    artifact_payload: object,
) -> tuple[Path, Path]:
    artifact = tmp_path / (requirement + ".json")
    artifact.write_text(json.dumps(artifact_payload), encoding="utf-8")
    manifest = tmp_path / "evidence.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MATRIX.EVIDENCE_MANIFEST_SCHEMA,
                "artifacts": {
                    requirement: {
                        "status": "complete",
                        "path": artifact.name,
                        "sha256": sha256_file(artifact),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, artifact


def test_hash_valid_dummy_json_cannot_clear_an_evidence_blocker(tmp_path: Path) -> None:
    manifest, artifact = _evidence_manifest(tmp_path, "unknown_mass_sweep", {})

    blockers, ready, provenance = MATRIX._evidence_status(manifest, tmp_path)

    unknown = next(row for row in blockers if row["requirement"] == "unknown_mass_sweep")
    assert unknown["code"] == "invalid_evidence_unknown_mass_sweep"
    assert unknown["path"] == str(artifact.resolve())
    assert "schema must be" in unknown["message"]
    assert "unknown_mass_sweep" not in ready
    assert provenance == {
        "path": str(manifest.resolve()),
        "sha256": sha256_file(manifest),
    }


def test_blocked_evidence_preserves_hash_bound_reason(tmp_path: Path) -> None:
    artifact = tmp_path / "unknown.json"
    artifact.write_text('{"status":"blocked"}\n', encoding="utf-8")
    manifest = tmp_path / "evidence.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MATRIX.EVIDENCE_MANIFEST_SCHEMA,
                "artifacts": {
                    "unknown_mass_sweep": {
                        "status": "blocked",
                        "path": artifact.name,
                        "sha256": sha256_file(artifact),
                        "message": "checkpoint mass provenance is incomplete",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    blockers, ready, _ = MATRIX._evidence_status(manifest, tmp_path)

    unknown = next(row for row in blockers if row["requirement"] == "unknown_mass_sweep")
    assert unknown["message"] == "checkpoint mass provenance is incomplete"
    assert unknown["path"] == str(artifact.resolve())
    assert "unknown_mass_sweep" not in ready


def test_valid_complete_unknown_mass_report_clears_only_its_blocker(tmp_path: Path) -> None:
    samples = list(MATRIX.UNKNOWN_MASS_EVIDENCE_SAMPLES)
    masses = list(MATRIX.UNKNOWN_MASS_EVIDENCE_VALUES)
    cases = []
    for sample in samples:
        for mass in masses:
            label = ("%.2f" % mass).replace(".", "p")
            cases.append(
                {
                    "case_id": f"{sample}/seed17/unknown_mass_{label}_refined",
                    "sample": sample,
                    "seed": 17,
                    "unknown_mass": mass,
                    "unknown_mass_label": label,
                    "endpoints": {
                        "round0": {
                            "case_id": f"{sample}/seed17/unknown_mass_{label}_round0",
                            "refinement_round": 0,
                            "prediction": {
                                "sha256": "a" * 64,
                                "unknown_mass": mass,
                                "unknown_mass_metadata_binding": {
                                    "round0": "checkpoint_and_manifest_bound",
                                    "refined": "checkpoint_and_manifest_bound",
                                },
                            },
                            "metrics": {"median_gene_spearman": 0.0},
                        },
                        "refined": {
                            "case_id": f"{sample}/seed17/unknown_mass_{label}_refined",
                            "refinement_round": 4,
                            "prediction": {
                                "sha256": "b" * 64,
                                "unknown_mass": mass,
                                "unknown_mass_metadata_binding": {
                                    "round0": "checkpoint_and_manifest_bound",
                                    "refined": "checkpoint_and_manifest_bound",
                                },
                            },
                            "metrics": {"median_gene_spearman": 0.1},
                        },
                    },
                    "paired_gene_spearman_deltas": {
                        "refined_minus_round0": {"median_delta": 0.1},
                        "heir_minus_hard_baseline": {"median_delta": 0.1},
                        "heir_minus_soft_baseline": {"median_delta": 0.1},
                    },
                }
            )
    contract = {
        "requirement": "unknown_mass_sweep",
        "samples": samples,
        "seed": 17,
        "unknown_masses": masses,
        "expected_case_count": 15,
        "expected_prediction_count": 30,
    }
    report = {
        "schema": MATRIX.EVIDENCE_ARTIFACT_SCHEMAS["unknown_mass_sweep"],
        "requirement": "unknown_mass_sweep",
        "contract": contract,
        "status": "complete",
        "request": {
            **{key: value for key, value in contract.items() if key != "requirement"},
            "minimum_nuclei": 3,
        },
        "scored_case_count": 15,
        "scored_prediction_count": 30,
        "blockers": [],
        "cases": cases,
        "stability": {
            "status": "stable",
            "direction_stable_across_masses": True,
        },
    }
    manifest, artifact = _evidence_manifest(tmp_path, "unknown_mass_sweep", report)

    blockers, ready, _ = MATRIX._evidence_status(manifest, tmp_path)

    assert not any(row["requirement"] == "unknown_mass_sweep" for row in blockers)
    assert len(blockers) == len(MATRIX.EVIDENCE_REQUIREMENTS) - 1
    assert ready["unknown_mass_sweep"] == {
        "status": "contract_validated_complete",
        "path": str(artifact.resolve()),
        "sha256": sha256_file(artifact),
        "schema": "heir.snpatho_unknown_mass_sensitivity.v1",
        "requirement": "unknown_mass_sweep",
        "paired_case_count": 15,
        "validated_endpoint_count": 30,
        "stability_status": "stable",
    }

    report["cases"][0]["endpoints"]["round0"]["prediction"]["unknown_mass_metadata_binding"][
        "round0"
    ] = "legacy_checkpoint_manifest_bound"
    manifest, artifact = _evidence_manifest(tmp_path, "unknown_mass_sweep", report)

    blockers, ready, _ = MATRIX._evidence_status(manifest, tmp_path)

    unknown = next(row for row in blockers if row["requirement"] == "unknown_mass_sweep")
    assert unknown["code"] == "invalid_evidence_unknown_mass_sweep"
    assert "not checkpoint-and-manifest bound" in unknown["message"]
    assert "unknown_mass_sweep" not in ready


def _support_artifact(tmp_path: Path, name: str, payload: dict) -> dict:
    path = tmp_path / "support" / (name + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {"path": str(path.relative_to(tmp_path)), "sha256": sha256_file(path)}


def _wrap_evidence(requirement: str, contract: dict, cases: list[dict], **extra: object) -> dict:
    return {
        "schema": MATRIX.EVIDENCE_ARTIFACT_SCHEMAS[requirement],
        "requirement": requirement,
        "status": "complete",
        "blockers": [],
        "contract": contract,
        "scored_case_count": len(cases),
        "cases": cases,
        **extra,
    }


def _future_control_reports(tmp_path: Path) -> dict[str, dict]:
    samples = list(MATRIX.UNKNOWN_MASS_EVIDENCE_SAMPLES)
    seeds = list(MATRIX.FOLLOWUP_EVIDENCE_SEEDS)
    reference_id = "pan_tissue_atlas_v1"
    reference = _support_artifact(
        tmp_path,
        "generic_reference",
        {
            "schema": "heir.evidence.generic_atlas_reference.v1",
            "reference_id": reference_id,
            "donor_ids": ["atlas_donor_a", "atlas_donor_b"],
            "cell_count": 20,
        },
    )
    ontology = _support_artifact(
        tmp_path,
        "generic_ontology",
        {
            "schema": "heir.evidence.generic_atlas_ontology.v1",
            "reference_id": reference_id,
            "type_names": ["A", "B"],
        },
    )
    prototype = _support_artifact(
        tmp_path,
        "generic_prototype",
        {
            "schema": "heir.evidence.generic_atlas_prototype.v1",
            "reference_id": reference_id,
            "donor_ids": ["atlas_donor_a", "atlas_donor_b"],
            "type_names": ["A", "B"],
            "prototype_count": 2,
        },
    )
    generic_cases = []
    for sample in samples:
        for seed in seeds:
            identity = {"sample": sample, "seed": seed, "reference_id": reference_id}
            prediction = _support_artifact(
                tmp_path,
                f"generic_prediction_{sample}_{seed}",
                {
                    "schema": "heir.evidence.generic_atlas_prediction.v1",
                    **identity,
                    "status": "complete",
                    "reference_sha256": reference["sha256"],
                    "ontology_sha256": ontology["sha256"],
                    "prototype_sha256": prototype["sha256"],
                },
            )
            score = _support_artifact(
                tmp_path,
                f"generic_score_{sample}_{seed}",
                {
                    "schema": "heir.evidence.generic_atlas_score.v1",
                    **identity,
                    "status": "complete",
                    "prediction_sha256": prediction["sha256"],
                    "metric": "median_gene_spearman",
                    "statistic": 0.1,
                },
            )
            generic_cases.append(
                {**identity, "prediction_artifact": prediction, "score_artifact": score}
            )
    generic = _wrap_evidence(
        "generic_atlas",
        {
            "requirement": "generic_atlas",
            "samples": samples,
            "seeds": seeds,
            "references": [
                {
                    "reference_id": reference_id,
                    "reference_artifact": reference,
                    "ontology_artifact": ontology,
                    "prototype_artifact": prototype,
                }
            ],
            "expected_case_count": len(generic_cases),
        },
        generic_cases,
    )

    label_cases = []
    metric_name = "paired_median_gene_spearman_delta"
    observed_values = []
    null_values = []
    for sample in samples:
        for seed in seeds:
            identity = {"sample": sample, "seed": seed}
            observed_value = 0.25
            observed_values.append(observed_value)
            observed = _support_artifact(
                tmp_path,
                f"label_observed_{sample}_{seed}",
                {
                    "schema": "heir.evidence.label_permutation_observed.v1",
                    **identity,
                    "metric": metric_name,
                    "statistic": observed_value,
                },
            )
            draws = []
            for draw_index in range(100):
                material = f"{sample}|{seed}|{draw_index}|label_permutation_v1"
                draw_seed = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)
                map_payload = {
                    "source_labels": ["A", "B"],
                    "permuted_labels": ["B", "A"],
                }
                statistic = (draw_index - 50) / 1000.0
                null_values.append(statistic)
                draws.append(
                    {
                        "draw_index": draw_index,
                        "draw_seed": draw_seed,
                        **map_payload,
                        "map_sha256": MATRIX._canonical_evidence_sha256(map_payload),
                        "statistic": statistic,
                    }
                )
            draw_manifest = _support_artifact(
                tmp_path,
                f"label_draws_{sample}_{seed}",
                {
                    "schema": "heir.evidence.label_permutation_draws.v1",
                    **identity,
                    "metric": metric_name,
                    "draws": draws,
                },
            )
            label_cases.append(
                {
                    **identity,
                    "observed_score_artifact": observed,
                    "draw_manifest_artifact": draw_manifest,
                }
            )
    null_array = np.asarray(null_values)
    observed_mean = float(np.mean(observed_values))
    label = _wrap_evidence(
        "label_permutation",
        {
            "requirement": "label_permutation",
            "samples": samples,
            "seeds": seeds,
            "permutation_count": 100,
            "draw_seed_scheme": "sha256_label_permutation_v1",
            "expected_case_count": len(label_cases),
        },
        label_cases,
        null_result_summary={
            "permutation_count": 100,
            "case_count": len(label_cases),
            "draw_count": len(null_values),
            "metric": metric_name,
            "observed_statistic": observed_mean,
            "null_mean": float(np.mean(null_array)),
            "null_standard_deviation": float(np.std(null_array)),
            "empirical_p_value": float(
                (1 + np.sum(np.abs(null_array) >= abs(observed_mean))) / (len(null_array) + 1)
            ),
        },
    )

    states = ["A"]
    state_cases = []
    for sample in samples:
        for seed in seeds:
            identity = {"sample": sample, "seed": seed, "omitted_state": "A"}
            omitted_reference = _support_artifact(
                tmp_path,
                f"state_reference_{sample}_{seed}",
                {
                    "schema": "heir.evidence.state_omission_reference.v1",
                    **identity,
                    "source_states": ["A", "B"],
                    "retained_states": ["B"],
                },
            )
            omitted_prototype = _support_artifact(
                tmp_path,
                f"state_prototype_{sample}_{seed}",
                {
                    "schema": "heir.evidence.state_omission_prototype.v1",
                    **identity,
                    "reference_sha256": omitted_reference["sha256"],
                    "type_names": ["B"],
                },
            )
            omitted_prediction = _support_artifact(
                tmp_path,
                f"state_prediction_{sample}_{seed}",
                {
                    "schema": "heir.evidence.state_omission_prediction.v1",
                    **identity,
                    "status": "complete",
                    "reference_sha256": omitted_reference["sha256"],
                    "prototype_sha256": omitted_prototype["sha256"],
                    "type_names": ["B"],
                },
            )
            risk = _support_artifact(
                tmp_path,
                f"state_risk_{sample}_{seed}",
                {
                    "schema": "heir.evidence.state_omission_risk_coverage.v1",
                    **identity,
                    "prediction_sha256": omitted_prediction["sha256"],
                    "coverage": [0.5, 1.0],
                    "risk": [0.1, 0.2],
                },
            )
            state_cases.append(
                {
                    **identity,
                    "reference_artifact": omitted_reference,
                    "prototype_artifact": omitted_prototype,
                    "prediction_artifact": omitted_prediction,
                    "risk_coverage_artifact": risk,
                }
            )
    state = _wrap_evidence(
        "state_omission",
        {
            "requirement": "state_omission",
            "samples": samples,
            "seeds": seeds,
            "omitted_states": states,
            "expected_case_count": len(state_cases),
        },
        state_cases,
    )

    sizes = list(MATRIX.REFERENCE_DOWNSAMPLING_SIZES)
    downsampling_cases = []
    source_count = 6000
    for sample in samples:
        all_cell_ids = [f"{sample}_cell_{index}" for index in range(source_count)]
        for seed in seeds:
            for size in sizes:
                identity = {"sample": sample, "seed": seed, "reference_size": size}
                selected = all_cell_ids if size == "all" else all_cell_ids[:size]
                material = f"{sample}|{seed}|{size}|reference_downsampling_v1"
                draw_seed = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)
                cell_ids_sha = MATRIX._canonical_evidence_sha256(selected)
                draw = _support_artifact(
                    tmp_path,
                    f"downsample_draw_{sample}_{seed}_{size}",
                    {
                        "schema": "heir.evidence.reference_downsampling_draw.v1",
                        **identity,
                        "draw_seed": draw_seed,
                        "source_cell_count": source_count,
                        "is_full_reference": size == "all",
                        "cell_ids": selected,
                        "cell_ids_sha256": cell_ids_sha,
                    },
                )
                prediction = _support_artifact(
                    tmp_path,
                    f"downsample_prediction_{sample}_{seed}_{size}",
                    {
                        "schema": "heir.evidence.reference_downsampling_prediction.v1",
                        **identity,
                        "status": "complete",
                        "draw_manifest_sha256": draw["sha256"],
                        "cell_ids_sha256": cell_ids_sha,
                    },
                )
                metric = _support_artifact(
                    tmp_path,
                    f"downsample_metric_{sample}_{seed}_{size}",
                    {
                        "schema": "heir.evidence.reference_downsampling_metric.v1",
                        **identity,
                        "prediction_sha256": prediction["sha256"],
                        "metric": "median_gene_spearman",
                        "statistic": 0.1,
                    },
                )
                downsampling_cases.append(
                    {
                        **identity,
                        "cell_id_draw_artifact": draw,
                        "prediction_artifact": prediction,
                        "metric_artifact": metric,
                    }
                )
    downsampling = _wrap_evidence(
        "reference_downsampling",
        {
            "requirement": "reference_downsampling",
            "samples": samples,
            "seeds": seeds,
            "reference_sizes": sizes,
            "draw_seed_scheme": "sha256_reference_downsampling_v1",
            "expected_case_count": len(downsampling_cases),
        },
        downsampling_cases,
    )
    return {
        "generic_atlas": generic,
        "label_permutation": label,
        "state_omission": state,
        "reference_downsampling": downsampling,
    }


def _clean_and_untouched_reports(tmp_path: Path) -> dict[str, dict]:
    cell_ids = ["cell1", "cell2", "cell3"]
    cell_manifest = _support_artifact(
        tmp_path,
        "clean_cell_ids",
        {"schema": "heir.evidence.clean_reannotation_cell_ids.v1", "cell_ids": cell_ids},
    )
    annotation = _support_artifact(
        tmp_path,
        "clean_annotation",
        {
            "schema": "heir.evidence.clean_reannotation_annotation_table.v1",
            "independent": True,
            "published_integrated_labels_used": False,
            "cell_ids": cell_ids,
            "labels": ["A", "A", "B"],
        },
    )
    ontology = _support_artifact(
        tmp_path,
        "clean_ontology",
        {"schema": "heir.evidence.clean_reannotation_ontology.v1", "type_names": ["A", "B"]},
    )
    markers = _support_artifact(
        tmp_path,
        "clean_markers",
        {
            "schema": "heir.evidence.clean_reannotation_markers.v1",
            "status": "complete",
            "supported_types": ["A", "B"],
        },
    )
    qc = _support_artifact(
        tmp_path,
        "clean_qc",
        {
            "schema": "heir.evidence.clean_reannotation_qc.v1",
            "status": "pass",
            "cell_count": 3,
            "aligned_cell_count": 3,
            "annotation_sha256": annotation["sha256"],
        },
    )
    adjudication = _support_artifact(
        tmp_path,
        "clean_adjudication",
        {
            "schema": "heir.evidence.clean_reannotation_adjudication.v1",
            "status": "complete",
            "cell_count": 3,
            "unresolved_count": 0,
            "annotation_sha256": annotation["sha256"],
            "ontology_sha256": ontology["sha256"],
        },
    )
    clean = {
        "schema": MATRIX.EVIDENCE_ARTIFACT_SCHEMAS["clean_independent_reannotation"],
        "requirement": "clean_independent_reannotation",
        "status": "complete",
        "blockers": [],
        "contract": {
            "requirement": "clean_independent_reannotation",
            "annotation_cell_count": 3,
            "annotation_provenance": {
                "independent": True,
                "published_integrated_labels_used": False,
                "method": "marker_qc_adjudication",
            },
            "artifacts": {
                "cell_ids": cell_manifest,
                "annotation_table": annotation,
                "ontology": ontology,
                "marker_evidence": markers,
                "qc": qc,
                "adjudication": adjudication,
            },
        },
    }

    cohort_id = "external_locked_v1"
    development = list(MATRIX.UNKNOWN_MASS_EVIDENCE_SAMPLES)
    target = ["external_donor_1"]
    freeze = _support_artifact(
        tmp_path,
        "untouched_freeze",
        {
            "schema": "heir.evidence.untouched_cohort_freeze.v1",
            "cohort_id": cohort_id,
            "analysis_role": "untouched_locked_confirmatory_validation",
            "freeze_before_truth_access": True,
            "development_donor_ids": development,
            "target_donor_ids": target,
            "frozen_at": "2026-01-01T00:00:00Z",
        },
    )
    prediction = _support_artifact(
        tmp_path,
        "untouched_prediction",
        {
            "schema": "heir.evidence.untouched_cohort_prediction.v1",
            "cohort_id": cohort_id,
            "donor_ids": target,
            "status": "complete",
            "frozen": True,
            "created_before_truth_access": True,
            "freeze_manifest_sha256": freeze["sha256"],
        },
    )
    truth = _support_artifact(
        tmp_path,
        "untouched_truth",
        {
            "schema": "heir.evidence.untouched_cohort_truth.v1",
            "cohort_id": cohort_id,
            "donor_ids": target,
            "locked": True,
            "opened_after_prediction_freeze": True,
            "freeze_manifest_sha256": freeze["sha256"],
        },
    )
    evaluation = _support_artifact(
        tmp_path,
        "untouched_evaluation",
        {
            "schema": "heir.evidence.untouched_cohort_evaluation.v1",
            "cohort_id": cohort_id,
            "donor_ids": target,
            "status": "complete",
            "analysis_role": "untouched_locked_confirmatory_validation",
            "prediction_sha256": prediction["sha256"],
            "truth_sha256": truth["sha256"],
            "metrics": {"median_gene_spearman": 0.1},
        },
    )
    untouched = {
        "schema": MATRIX.EVIDENCE_ARTIFACT_SCHEMAS["untouched_external_cohort"],
        "requirement": "untouched_external_cohort",
        "status": "complete",
        "blockers": [],
        "contract": {
            "requirement": "untouched_external_cohort",
            "cohort_id": cohort_id,
            "analysis_role": "untouched_locked_confirmatory_validation",
            "untouched": True,
            "locked": True,
            "freeze_before_truth_access": True,
            "development_donor_ids": development,
            "target_donor_ids": target,
            "artifacts": {
                "freeze_manifest": freeze,
                "prediction": prediction,
                "truth": truth,
                "evaluation": evaluation,
            },
        },
    }
    return {
        "clean_independent_reannotation": clean,
        "untouched_external_cohort": untouched,
    }


def _validate_test_evidence(tmp_path: Path, requirement: str, report: dict) -> dict:
    return MATRIX._validate_evidence_artifact(
        requirement,
        report,
        report_path=tmp_path / (requirement + ".json"),
        repository=tmp_path,
    )


def test_future_contracts_accept_recursive_hash_bound_artifacts(tmp_path: Path) -> None:
    reports = {
        **_future_control_reports(tmp_path),
        **_clean_and_untouched_reports(tmp_path),
    }

    validated = {
        requirement: _validate_test_evidence(tmp_path, requirement, report)
        for requirement, report in reports.items()
    }

    assert validated["generic_atlas"]["validated_supporting_artifact_count"] == 33
    assert validated["label_permutation"]["validated_draw_count"] == 1500
    assert validated["state_omission"]["validated_supporting_artifact_count"] == 60
    assert validated["reference_downsampling"]["validated_supporting_artifact_count"] == 180
    assert validated["clean_independent_reannotation"]["validated_annotation_cell_count"] == 3
    assert validated["untouched_external_cohort"]["validated_target_donor_count"] == 1


@pytest.mark.parametrize(
    ("requirement", "mutation", "message"),
    (
        (
            "generic_atlas",
            lambda report: report["contract"]["references"][0]["reference_artifact"].update(
                {"sha256": "bad"}
            ),
            "SHA-256",
        ),
        (
            "label_permutation",
            lambda report: report["contract"].update({"permutation_count": 99}),
            "at least 100",
        ),
        (
            "state_omission",
            lambda report: report["contract"].update({"omitted_states": []}),
            "non-empty",
        ),
        (
            "reference_downsampling",
            lambda report: report["contract"].update({"reference_sizes": [1000, 2500, "all"]}),
            "must be exactly",
        ),
    ),
)
def test_future_control_contracts_reject_requirement_specific_gaps(
    tmp_path: Path,
    requirement: str,
    mutation: object,
    message: str,
) -> None:
    report = copy.deepcopy(_future_control_reports(tmp_path)[requirement])
    mutation(report)

    with pytest.raises(ValueError, match=message):
        _validate_test_evidence(tmp_path, requirement, report)


@pytest.mark.parametrize(
    "requirement",
    (
        "generic_atlas",
        "label_permutation",
        "state_omission",
        "reference_downsampling",
        "clean_independent_reannotation",
        "untouched_external_cohort",
    ),
)
def test_hash_valid_dummy_supporting_artifact_is_rejected(
    tmp_path: Path,
    requirement: str,
) -> None:
    reports = {
        **_future_control_reports(tmp_path),
        **_clean_and_untouched_reports(tmp_path),
    }
    report = reports[requirement]
    dummy = _support_artifact(tmp_path, "dummy_" + requirement, {})
    if requirement == "generic_atlas":
        report["contract"]["references"][0]["reference_artifact"] = dummy
    elif requirement == "label_permutation":
        report["cases"][0]["draw_manifest_artifact"] = dummy
    elif requirement == "state_omission":
        report["cases"][0]["reference_artifact"] = dummy
    elif requirement == "reference_downsampling":
        report["cases"][0]["cell_id_draw_artifact"] = dummy
    elif requirement == "clean_independent_reannotation":
        report["contract"]["artifacts"]["annotation_table"] = dummy
    else:
        report["contract"]["artifacts"]["freeze_manifest"] = dummy

    with pytest.raises(ValueError, match="schema"):
        _validate_test_evidence(tmp_path, requirement, report)
