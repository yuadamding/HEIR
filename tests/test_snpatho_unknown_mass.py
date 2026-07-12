"""Focused synthetic tests for the standalone unknown-mass evaluator."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import sparse

from heir.data import PrototypeSet, RNAReference, SpatialTruthArtifact
from heir.inference import PredictionBundle
from heir.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_snpatho_unknown_mass.py"
SPEC = importlib.util.spec_from_file_location("benchmark_snpatho_unknown_mass", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SENSITIVITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SENSITIVITY
SPEC.loader.exec_module(SENSITIVITY)


def _prediction(
    path: Path,
    *,
    sample: str,
    selected_round: int,
    expression_linear: np.ndarray,
    checkpoint: Path,
    prototype: Path,
    histology: Path,
    ood: Path,
) -> None:
    nucleus_ids = np.asarray(["n%d" % index for index in range(6)])
    expression = np.repeat(np.asarray(expression_linear, dtype=np.float32), 2, axis=0)
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
        refinement_round=selected_round,
        expression_interval_semantics=PredictionBundle.CONDITIONAL_KNOWN_STATE,
        expression_mean_available=np.ones(6, dtype=bool),
        expression_interval_available=np.ones(6, dtype=bool),
        sample_id=sample,
        donor_id=sample,
        slide_id=sample,
        checkpoint_sha256=sha256_file(checkpoint),
        prototype_sha256=sha256_file(prototype),
        histology_sha256=sha256_file(histology),
        ood_sha256=sha256_file(ood),
        latent_space_id="latent-native-test",
        model_version="test-v1",
        ood_training_donors=np.asarray(["synthetic_reference"]),
        inference_seed=17,
        latent_samples=2,
        probability_threshold=0.35,
        artifact_threshold=0.5,
        expression_space_id="log1p-cpm-10000-v1",
    )
    bundle.to_npz(path)
    path.with_name("prediction.telemetry.json").write_text(
        json.dumps(
            {
                "schema": "heir.inference_telemetry.v1",
                "prediction_path": str(path.resolve()),
                "prediction_sha256": sha256_file(path),
                "nuclei": 6,
                "genes": 3,
                "negative_control": {
                    "prototype_only": False,
                    "image_feature_shuffle": False,
                    "graph_node_shuffle": False,
                    "no_graph": False,
                    "wrong_donor": False,
                    "prototype_donor_id": sample,
                    "seed": 17,
                },
            }
        ),
        encoding="utf-8",
    )


def _sensitivity_artifact(
    repository: Path,
    artifact_root: Path,
    *,
    sample: str,
    mass: float,
    selected_round: int,
    expression_linear: np.ndarray,
    histology: Path,
    ood: Path,
    reference_sha256: str,
) -> None:
    round0, refined = SENSITIVITY._directories(artifact_root, sample, mass)
    round0.mkdir(parents=True, exist_ok=True)
    refined.mkdir(parents=True, exist_ok=True)
    parent = round0 / "heir.pt"
    torch.save(
        {
            "metadata": {
                "schema": "heir.trained_model.v1",
                "uot_unknown_mass": mass,
                "uot_unknown_mass_mode": "fixed",
            }
        },
        parent,
    )
    (round0 / "history.json").write_text("{}\n", encoding="utf-8")
    view = round0 / "refinement_views.npz"
    view.write_bytes(b"synthetic refinement views")

    native_prototype = (
        repository / "artifacts" / "snpatho" / "r1_scanvi" / sample / "prototypes_rare_complete.npz"
    )
    prototype = refined / "prototypes" / ("%s__%s.npz" % (sample, sample))
    prototype.parent.mkdir(parents=True)
    prototype_value = PrototypeSet(
        prototype_ids=np.asarray(["pA", "pB"]),
        sample_ids=np.asarray([sample, sample]),
        cell_type_labels=np.asarray(["A", "B"]),
        means=np.zeros((2, 2), dtype=np.float32),
        variances=np.ones((2, 2), dtype=np.float32),
        weights=np.asarray([0.5, 0.5]),
        n_cells=np.asarray([2, 2]),
        latent_space_id="latent-native-test",
        donor_id=sample,
        block_id=sample + "_FFPE",
        source_reference_sha256=reference_sha256,
    )
    prototype_value.save_npz(prototype)
    if not native_prototype.is_file():
        prototype_value.save_npz(native_prototype)

    round_count = max(selected_round, 1)
    rounds = [
        {
            "round_id": round_id,
            "committed": selected_round > 0 and round_id <= selected_round,
        }
        for round_id in range(1, round_count + 1)
    ]
    stopped_reason = "validation_degraded_rollback" if selected_round == 0 else "maximum_rounds"
    checkpoint = refined / "heir_refined.pt"
    metadata = {
        "schema": "heir.refined_model.v1",
        "seed": 17,
        "refinement_round": selected_round,
        "refinement_rounds_executed": len(rounds),
        "refinement_rounds": rounds,
        "refinement_stopped_reason": stopped_reason,
        "refinement_round_zero_validation_loss": 1.25,
        "parent_checkpoint": str(parent.resolve()),
        "parent_checkpoint_sha256": sha256_file(parent),
        "refinement_view_artifacts": [
            {
                "key": "%s::%s::%s_train" % (sample, sample, sample),
                "path": str(view.resolve()),
                "sha256": sha256_file(view),
            }
        ],
        "gene_names": ["g1", "g2", "g3"],
        "latent_space_id": "latent-native-test",
        "expression_space_id": "log1p-cpm-10000-v1",
        "uot_unknown_mass": mass,
        "uot_unknown_mass_mode": "fixed",
    }
    torch.save({"metadata": metadata}, checkpoint)
    (refined / "refinement.json").write_text(
        json.dumps(
            {
                "selected_round": selected_round,
                "rounds": rounds,
                "stopped_reason": stopped_reason,
                "round_zero_validation_loss": 1.25,
                "prototype_artifacts": {"%s::%s" % (sample, sample): str(prototype.resolve())},
                "round_checkpoints": {},
            }
        ),
        encoding="utf-8",
    )
    _prediction(
        refined / "predictions.npz",
        sample=sample,
        selected_round=selected_round,
        expression_linear=expression_linear,
        checkpoint=checkpoint,
        prototype=prototype,
        histology=histology,
        ood=ood,
    )
    _prediction(
        round0 / "predictions.npz",
        sample=sample,
        selected_round=0,
        expression_linear=np.asarray(expression_linear)[::-1],
        checkpoint=parent,
        prototype=native_prototype,
        histology=histology,
        ood=ood,
    )


def _fixture(tmp_path: Path, *, custom_output_root: bool = False) -> dict:
    repository = tmp_path / "repository"
    molecular_root = repository / "artifacts" / "snpatho" / "r1_scanvi"
    artifact_root = tmp_path / "clean-unknown-mass-output" if custom_output_root else molecular_root
    sample = "sample_a"
    for relative in SENSITIVITY._RUNNER.UNKNOWN_MASS_SOURCE_FILES:
        source = ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    genes = np.asarray(["g1", "g2", "g3"])
    nucleus_ids = np.asarray(["n%d" % index for index in range(6)])
    truth_linear = np.asarray([[10, 0, 1], [5, 5, 10], [0, 10, 3]], dtype=np.float32)

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
    reference_path = molecular_root / sample / "reference500_scanvi.npz"
    reference_path.parent.mkdir(parents=True)
    reference.save_npz(reference_path)

    histology = repository / "artifacts" / "snpatho" / sample / "histology_full.npz"
    ood = repository / "artifacts" / "snpatho" / sample / "ood_target_calibrated.npz"
    histology.parent.mkdir(parents=True, exist_ok=True)
    histology.write_bytes(b"synthetic histology")
    ood.write_bytes(b"synthetic OOD")
    for selected_round, mass in enumerate(SENSITIVITY.UNKNOWN_MASSES):
        _sensitivity_artifact(
            repository,
            artifact_root,
            sample=sample,
            mass=mass,
            selected_round=selected_round,
            expression_linear=truth_linear,
            histology=histology,
            ood=ood,
            reference_sha256=sha256_file(reference_path),
        )

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
                "annotation_provenance": "published integrated synthetic annotation",
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
    plan = SENSITIVITY._RUNNER.build_plan(
        repository,
        samples=(sample,),
        seeds=(17,),
        unknown_mass_sensitivity=True,
        artifact_root=artifact_root,
    )
    for stage in plan:
        for _, stage_input in stage.inputs:
            if not stage_input.exists():
                stage_input.parent.mkdir(parents=True, exist_ok=True)
                stage_input.write_bytes(b"synthetic frozen input")
    records = [
        {
            "sample": stage.sample,
            "seed": stage.seed,
            "stage": stage.name,
            "unknown_mass": stage.unknown_mass,
            "status": "skipped_valid",
        }
        for stage in plan
    ]
    run_manifest = (
        repository / "artifacts" / "snpatho" / "unknown_mass_sensitivity_v1" / "run_manifest.json"
    )
    run_manifest.parent.mkdir(parents=True)
    run_manifest.write_text(
        json.dumps(
            SENSITIVITY._RUNNER.build_unknown_mass_manifest(
                repository,
                plan,
                records,
                samples=(sample,),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "repository": repository,
        "artifact_root": artifact_root,
        "run_manifest_path": run_manifest,
        "truth_manifest_path": truth_manifest,
        "native_manifest_path": native_manifest,
        "samples": (sample,),
        "minimum_nuclei": 2,
    }


def test_complete_grid_accepts_safety_selected_rounds_and_reports_stability(
    tmp_path: Path,
) -> None:
    report = SENSITIVITY.evaluate_unknown_mass(**_fixture(tmp_path))

    assert report["status"] == "complete"
    assert report["requirement"] == "unknown_mass_sweep"
    assert report["contract"]["expected_case_count"] == 5
    assert report["contract"]["expected_prediction_count"] == 10
    assert report["scored_case_count"] == 5
    assert report["scored_prediction_count"] == 10
    source_identity = report["scorer_source_identity"]
    assert source_identity["schema"] == "heir.source_identity.v1"
    source_files = [row["relative_path"] for row in source_identity["files"]]
    assert source_files[:3] == [
        "scripts/benchmark_snpatho_unknown_mass.py",
        "scripts/benchmark_snpatho_refinement_matrix.py",
        "scripts/run_snpatho_refinement_benchmark.py",
    ]
    assert "src/heir/inference.py" in source_files
    assert "src/heir/evaluation/deepbench.py" in source_files
    assert "pyproject.toml" in source_files
    assert len(source_identity["aggregate_sha256"]) == 64
    assert report["blockers"] == []
    assert report["manifests"]["unknown_mass_run"]["execution_mode"] == "all_skipped_valid"
    assert [case["refinement_round"] for case in report["cases"]] == [0, 1, 2, 3, 4]
    assert report["stability"]["status"] == "stable"
    assert report["stability"]["practical_status_stable_across_masses"] is True
    assert report["stability"]["direction_stable_across_masses"] is True
    assert report["stability"]["refined_beats_round0_at_every_mass"] is True
    assert report["stability"]["heir_beats_both_baselines_at_every_mass"] is True
    assert set(report["cases"][0]["metrics"]) == {
        SENSITIVITY.METHOD,
        SENSITIVITY.HARD_BASELINE,
        SENSITIVITY.SOFT_BASELINE,
    }
    assert "per_gene" not in json.dumps(report)
    for case in report["cases"]:
        assert set(case["endpoints"]) == {"round0", "refined"}
        assert set(case["paired_gene_spearman_deltas"]) == set(SENSITIVITY.CONTRASTS)
        assert all(
            delta["practical_status"] == "pass" and delta["raw_sign_status"] == "positive"
            for delta in case["paired_gene_spearman_deltas"].values()
        )
        assert case["endpoints"]["round0"]["refinement_round"] == 0
        for endpoint in case["endpoints"].values():
            assert set(endpoint["diagnostics"]) == {
                "abstention_fraction",
                "public_expression_coverage",
                "mean_unknown_probability",
            }
        assert case["prediction"]["telemetry_prediction_sha256_match"] is True
        assert case["prediction"]["run_manifest_stage_bound"] is True
        assert all(
            endpoint["prediction"]["run_manifest_stage_bound"] is True
            for endpoint in case["endpoints"].values()
        )
        assert len(case["prediction"]["refinement_audit_sha256"]) == 64
        assert len(case["prediction"]["parent_checkpoint_sha256"]) == 64

    changed_cases = json.loads(json.dumps(report["cases"]))
    changed_cases[-1]["paired_gene_spearman_deltas"]["heir_minus_soft_baseline"][
        "median_delta"
    ] = -0.25
    unstable = SENSITIVITY._stability(changed_cases, ("sample_a",))
    assert unstable["status"] == "unstable"
    assert unstable["direction_stable_across_masses"] is False
    assert unstable["heir_beats_both_baselines_at_every_mass"] is False


def test_unknown_mass_scorer_accepts_hash_bound_clean_custom_artifact_root(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path, custom_output_root=True)

    report = SENSITIVITY.evaluate_unknown_mass(**arguments)

    assert report["status"] == "complete"
    assert Path(report["request"]["artifact_root"]) == arguments["artifact_root"].resolve()
    assert report["scored_prediction_count"] == 10

    output = tmp_path / "outputs"
    SENSITIVITY.write_report(
        report,
        json_output=output / "sensitivity.json",
        tsv_output=output / "sensitivity.tsv",
        markdown_output=output / "sensitivity.md",
    )
    assert json.loads((output / "sensitivity.json").read_text())["scored_prediction_count"] == 10
    assert (output / "sensitivity.tsv").read_text().startswith("sample\tseed\tunknown_mass")
    assert "Refined-round0" in (output / "sensitivity.md").read_text()
    assert not list(output.glob("*.tmp"))


def test_missing_manifested_output_fails_before_scoring(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    _, refined = SENSITIVITY._directories(
        arguments["artifact_root"], "sample_a", SENSITIVITY.UNKNOWN_MASSES[-1]
    )
    (refined / "prediction.telemetry.json").unlink()

    with pytest.raises(FileNotFoundError, match="manifested stage output"):
        SENSITIVITY.evaluate_unknown_mass(**arguments)


def test_checkpoint_or_audit_tampering_fails_closed(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    _, refined = SENSITIVITY._directories(arguments["artifact_root"], "sample_a", 0.05)
    audit_path = refined / "refinement.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["selected_round"] = 4
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="output SHA-256 is stale"):
        SENSITIVITY.evaluate_unknown_mass(**arguments)


def test_unknown_mass_checkpoint_metadata_is_strict_and_required() -> None:
    with pytest.raises(ValueError, match="lacks serialized unknown-mass metadata"):
        SENSITIVITY._unknown_mass_metadata_binding({}, 0.05, label="legacy")
    assert (
        SENSITIVITY._unknown_mass_metadata_binding(
            {"uot_unknown_mass": 0.05, "uot_unknown_mass_mode": "fixed"},
            0.05,
            label="new",
        )
        == "checkpoint_and_manifest_bound"
    )
    with pytest.raises(ValueError, match="partial unknown-mass metadata"):
        SENSITIVITY._unknown_mass_metadata_binding(
            {"uot_unknown_mass": 0.05}, 0.05, label="partial"
        )
    with pytest.raises(ValueError, match="differs from its run manifest"):
        SENSITIVITY._unknown_mass_metadata_binding(
            {"uot_unknown_mass": 0.20, "uot_unknown_mass_mode": "fixed"},
            0.05,
            label="wrong",
        )
