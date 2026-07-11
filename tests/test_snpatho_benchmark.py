import hashlib
import json

import numpy as np
import pytest
from scipy import sparse

from heir.data import RNAReference, SpatialTruthArtifact
from heir.evaluation import (
    BENCHMARK_METHODS,
    load_snpatho_plan,
    run_snpatho_benchmark,
    write_snpatho_benchmark,
)
from heir.inference import PredictionBundle


def _write_case(tmp_path, *, locked_count_hash="1" * 64):
    genes = np.asarray(["g1", "g2"])
    panel = tmp_path / "panel.tsv"
    panel.write_text("g1\ng2\n")
    panel_hash = hashlib.sha256(panel.read_bytes()).hexdigest()
    checkpoint_hash = "a" * 64
    nucleus_ids = np.asarray(["sample::n%d" % index for index in range(6)])
    cell_expression = np.log1p(
        np.asarray(
            [[100, 0], [100, 0], [0, 100], [0, 100], [50, 50], [50, 50]],
            dtype=np.float32,
        )
    )
    probabilities = np.asarray(
        [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]],
        dtype=np.float32,
    )
    prediction = PredictionBundle(
        nucleus_ids=nucleus_ids,
        coordinates_um=np.column_stack((np.arange(6), np.zeros(6))).astype(np.float32),
        type_probabilities=probabilities,
        type_names=np.asarray(["A", "B"]),
        labels=probabilities.argmax(axis=1),
        prototype_probabilities=probabilities,
        prototype_ids=np.asarray(["pA", "pB"]),
        latent_mean=np.zeros((6, 2), dtype=np.float32),
        latent_variance=np.ones((6, 2), dtype=np.float32),
        expression_mean=cell_expression,
        expression_lower=np.maximum(cell_expression - 0.1, 0),
        expression_upper=cell_expression + 0.1,
        gene_names=genes,
        unknown_probability=np.zeros(6, dtype=np.float32),
        abstain_score=np.asarray([0, 0, 0, 0, 0, 1], dtype=np.float32),
        abstain=np.asarray([False, False, False, False, False, True]),
        ood_score=np.zeros(6, dtype=np.float32),
        refinement_round=0,
        sample_id="sample",
        donor_id="donor",
        slide_id="section",
        checkpoint_sha256=checkpoint_hash,
        prototype_sha256="b" * 64,
        histology_sha256="c" * 64,
        latent_space_id="latent-v1",
        model_version="frozen-v1",
        expression_space_id="log1p-cpm-10000-v1",
        inference_seed=17,
        latent_samples=20,
        probability_threshold=0.6,
        artifact_threshold=0.5,
    )
    prediction_path = tmp_path / "prediction.npz"
    prediction.to_npz(prediction_path)

    reference = RNAReference(
        sample_id="sample",
        cell_ids=np.asarray(["r0", "r1", "r2", "r3"]),
        gene_ids=genes,
        counts=sparse.csr_matrix([[10, 0], [8, 0], [0, 10], [0, 8]], dtype=np.float32),
        library_sizes=np.asarray([20, 16, 20, 16], dtype=np.float64),
        cell_type_labels=np.asarray(["A", "A", "B", "B"]),
        donor_ids=np.asarray(["donor"] * 4),
        sample_ids=np.asarray(["sample"] * 4),
        source_count_sha256="e" * 64,
    )
    reference_path = tmp_path / "reference.npz"
    reference.save_npz(reference_path)

    observed = np.log1p(np.asarray([[100, 0], [0, 100], [50, 50]], dtype=np.float32))
    truth = SpatialTruthArtifact(
        observed_expression=observed,
        gene_names=genes,
        spot_ids=np.asarray(["s0", "s1", "s2"]),
        nucleus_ids=nucleus_ids,
        nucleus_spot_index=np.asarray([0, 0, 1, 1, 2, 2]),
        spot_library_sizes=np.asarray([100, 100, 100], dtype=np.float64),
        spot_coordinates_px=np.asarray([[0, 0], [20, 0], [40, 0]], dtype=np.float64),
        nucleus_spot_distance_px=np.zeros(6, dtype=np.float64),
        analysis_role="locked_validation",
        cohort_id="snpatho_seq",
        donor_id="donor",
        specimen_id="sample",
        block_id="block",
        section_id="section",
        outer_fold="fold_0",
        inner_fold="inner_0",
        barcode_suffix_policy="exact",
        spot_radius_px=5.0,
        source_artifacts=np.asarray(
            ["counts", "positions", "scales", "nuclei", "panel", "manifest"]
        ),
        source_sha256=np.asarray(
            [locked_count_hash, "2" * 64, "3" * 64, "4" * 64, panel_hash, "6" * 64]
        ),
        source_roles=np.asarray(
            [
                "locked_spatial_counts",
                "locked_spatial_coordinates",
                "locked_spatial_scalefactors",
                "sample_segmentation",
                "canonical_gene_panel",
                "shared_manifest",
            ]
        ),
    )
    truth_path = tmp_path / "truth.npz"
    truth.save_npz(truth_path)

    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "schema": "heir.inference_telemetry.v1",
                "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                "wall_seconds": 2.0,
                "peak_cuda_memory_bytes": 2 * 1024**3,
                "device_type": "cuda",
                "device_name": "test GPU",
                "mixed_precision": True,
                "nuclei": 6,
            }
        )
    )
    plan_path = tmp_path / "plan.json"
    case = {
        "section_id": "section",
        "checkpoint_sha256": checkpoint_hash,
        "predictions": prediction_path.name,
        "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "truth": truth_path.name,
        "truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        "matched_reference": reference_path.name,
        "matched_reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "telemetry": telemetry_path.name,
        "telemetry_sha256": hashlib.sha256(telemetry_path.read_bytes()).hexdigest(),
    }
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "heir.snpatho_benchmark_plan.v1",
                "checkpoint_sha256": checkpoint_hash,
                "gene_panel": panel.name,
                "gene_panel_sha256": panel_hash,
                "frozen_model_version": "frozen-v1",
                "cases": [case],
            }
        )
    )
    return plan_path


def test_snpatho_benchmark_scores_baselines_coverage_telemetry_and_bootstrap(tmp_path):
    plan = load_snpatho_plan(_write_case(tmp_path))
    first = run_snpatho_benchmark(
        plan,
        require_complete=False,
        iterations=100,
        minimum_donors=2,
        seed=17,
    )
    second = run_snpatho_benchmark(
        plan,
        require_complete=False,
        iterations=100,
        minimum_donors=2,
        seed=17,
    )
    assert first.to_dict() == second.to_dict()
    case = first.cases[0]
    assert tuple(case.methods) == BENCHMARK_METHODS
    assert case.methods["heir"]["median_gene_mse"] == pytest.approx(0.0)
    assert case.methods["matched_snrna_pseudobulk"]["median_gene_pearson"] is None
    assert case.coverage["cell_coverage"] == pytest.approx(5 / 6)
    assert case.telemetry["peak_cuda_memory_gib"] == pytest.approx(2.0)
    assert case.telemetry["nuclei_per_second"] == pytest.approx(3.0)
    assert len(first.benchmark.comparisons) == 15
    assert {summary.status.value for summary in first.benchmark.summaries} == {
        "data_limited",
        "missing",
    }

    json_path = tmp_path / "report.json"
    tsv_path = tmp_path / "report.tsv"
    write_snpatho_benchmark(first, json_path=json_path, tsv_path=tsv_path)
    payload = json.loads(json_path.read_text())
    isolation = payload["isolation"]
    assert isolation["target_spatial_expression_used_for_training"] is False
    assert isolation["target_histology_used_for_training"] is True
    assert isolation["target_spatial_metadata_used_for_capture_filtering"] is True
    assert payload["schema_version"] == "heir.snpatho_benchmark.v1"
    assert tsv_path.read_text().startswith("record_type\t")


def test_snpatho_benchmark_requires_all_three_sections_by_default(tmp_path):
    plan = load_snpatho_plan(_write_case(tmp_path))
    with pytest.raises(ValueError, match="requires sections 4066, 4399, and 4411"):
        run_snpatho_benchmark(plan, iterations=10)


def test_snpatho_benchmark_rejects_target_hash_overlap(tmp_path):
    plan = load_snpatho_plan(_write_case(tmp_path, locked_count_hash="a" * 64))
    with pytest.raises(ValueError, match="overlaps prediction inputs"):
        run_snpatho_benchmark(
            plan,
            require_complete=False,
            iterations=10,
        )


@pytest.mark.parametrize(
    "field",
    [
        "predictions_sha256",
        "truth_sha256",
        "matched_reference_sha256",
        "telemetry_sha256",
    ],
)
def test_snpatho_plan_loader_rejects_each_stale_case_hash(tmp_path, field):
    plan_path = _write_case(tmp_path)
    payload = json.loads(plan_path.read_text())
    payload["cases"][0][field] = "0" * 64
    plan_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="SHA-256 differs from the frozen snPATHO plan"):
        load_snpatho_plan(plan_path)


def test_snpatho_plan_requires_explicit_case_checkpoint_and_telemetry_hash(tmp_path):
    plan_path = _write_case(tmp_path)
    payload = json.loads(plan_path.read_text())
    del payload["cases"][0]["checkpoint_sha256"]
    del payload["cases"][0]["telemetry_sha256"]
    plan_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="case 0 is incomplete"):
        load_snpatho_plan(plan_path)


def test_snpatho_benchmark_rechecks_frozen_artifacts_after_plan_load(tmp_path):
    plan = load_snpatho_plan(_write_case(tmp_path))
    plan.cases[0].telemetry.write_text("{}")

    with pytest.raises(ValueError, match="telemetry SHA-256 differs"):
        run_snpatho_benchmark(plan, require_complete=False, iterations=10)
