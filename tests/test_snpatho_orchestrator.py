import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_snpatho_pipeline.py"
SPEC = importlib.util.spec_from_file_location("snpatho_orchestrator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ORCHESTRATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORCHESTRATOR
SPEC.loader.exec_module(ORCHESTRATOR)


def _runner(tmp_path, *, execute=False):
    return ORCHESTRATOR.PipelineRunner(
        repository=ROOT,
        execute=execute,
        status_path=tmp_path / "status.json",
        events_path=tmp_path / "events.jsonl",
        logs_directory=tmp_path / "logs",
    )


def _stage(tmp_path, *, outputs=None, locked=False):
    return ORCHESTRATOR.Stage(
        name="synthetic",
        sample="4066",
        outputs=tuple(outputs or (tmp_path / "output.npz",)),
        requires=(),
        command=lambda: ("definitely-not-executed", "--argument"),
        validate=lambda: {"valid": True},
        locked_target=locked,
    )


def test_dry_run_records_command_without_launching_it(tmp_path):
    runner = _runner(tmp_path)
    assert runner.run(_stage(tmp_path)) == "planned"
    assert runner.records[-1]["status"] == "planned"
    assert "definitely-not-executed" in runner.records[-1]["command"]
    assert (tmp_path / "status.json").is_file()
    assert (tmp_path / "events.jsonl").read_text().count("\n") == 1


def test_partial_stage_is_blocked_instead_of_overwritten(tmp_path):
    first = tmp_path / "first.npz"
    first.write_bytes(b"partial")
    runner = _runner(tmp_path)
    status = runner.run(_stage(tmp_path, outputs=(first, tmp_path / "second.json")))
    assert status == "blocked_partial"
    assert first.read_bytes() == b"partial"
    assert runner.records[-1]["status"] == "blocked_partial"


def test_locked_target_execution_requires_explicit_prediction_gate(tmp_path):
    runner = _runner(tmp_path, execute=True)
    with pytest.raises(ORCHESTRATOR.PipelineError, match="before predictions were frozen"):
        runner.run(_stage(tmp_path, locked=True))


def test_v02_config_binds_final_all_visium_panel():
    panel = ROOT / "manifests" / "gene_panel_snpatho_500.tsv"
    config_path = ROOT / "configs" / "experiments" / "snpatho_v0_2.yaml"
    payload = yaml.safe_load(config_path.read_text())
    genes = [
        line.split("\t", 1)[0]
        for line in panel.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    digest = hashlib.sha256(panel.read_bytes()).hexdigest()
    assert len(genes) == len(set(genes)) == 500
    assert digest == "22ddb91188b3b124d5cf3ec0f7ae81017399d141e39647b0dce80675119fe927"
    assert payload["molecular_prior"]["gene_panel_sha256"] == digest
    assert payload["segmentation"]["method"] == "spaceranger-segment"
    assert payload["leakage_policy"]["model_freeze_precedes_truth_materialization"] is True
    assert payload["uncertainty"]["target_histology_calibration_quantile"] == 0.95
    assert ORCHESTRATOR.PREDICTION_PHASE.index("calibrate_ood") == (
        ORCHESTRATOR.PREDICTION_PHASE.index("prepare_histology") + 1
    )


def test_v02_config_uses_evaluator_metric_and_baseline_identifiers():
    config_path = ROOT / "configs" / "experiments" / "snpatho_v0_2.yaml"
    payload = yaml.safe_load(config_path.read_text())

    assert payload["evaluation"]["metrics"] == [
        "median_gene_spearman",
        "median_gene_pearson",
        "median_gene_mse",
        "mean_location_cosine",
        "fraction_genes_defined",
        "cell_coverage",
        "assigned_cell_coverage",
        "spot_coverage",
        "abstention_rate",
        "inference_wall_seconds",
        "peak_cuda_memory_gib",
        "nuclei_per_second",
    ]
    assert payload["evaluation"]["baselines"] == [
        "matched_snrna_pseudobulk",
        "heir_spatial_shuffle",
        "matched_type_mean",
    ]
    assert payload["optimization"]["bag_size"] >= payload["optimization"]["maximum_train_cells"]
    assert (
        payload["optimization"]["maximum_sample_cells"]
        >= payload["optimization"]["maximum_train_cells"]
    )
    assert payload["model"]["graph_hidden_dim"] == 256
    assert payload["model"]["graph_output_dim"] == 256
    assert payload["model"]["graph_layers"] == 3
    assert payload["model"]["trunk_hidden_dims"] == [512, 256]


def test_source_binding_rejects_any_path_hash_or_role_drift():
    expected = (("/artifact.npz", "a" * 64, "sample_assay"),)
    ORCHESTRATOR._validate_source_binding(
        label="test",
        artifacts=("/artifact.npz",),
        hashes=("a" * 64,),
        roles=("sample_assay",),
        expected=expected,
    )
    with pytest.raises(ValueError, match="provenance differs"):
        ORCHESTRATOR._validate_source_binding(
            label="test",
            artifacts=("/artifact.npz",),
            hashes=("b" * 64,),
            roles=("sample_assay",),
            expected=expected,
        )


def test_benchmark_tsv_must_exactly_match_json_aggregate(tmp_path):
    donor = {
        "cohort_id": "snpatho_seq",
        "donor_id": "4066",
        "method": "heir",
        "metric": "median_gene_mse",
        "value": 1.25,
        "status": "ok",
        "reason": "",
        "n_observations": 10,
    }
    report = {
        "aggregate": {
            "donor_metrics": [donor],
            "summaries": [],
            "comparisons": [],
        }
    }
    path = tmp_path / "benchmark.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ORCHESTRATOR._BENCHMARK_TSV_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                field: ORCHESTRATOR._tsv_value(
                    "donor_metric" if field == "record_type" else donor.get(field)
                )
                for field in ORCHESTRATOR._BENCHMARK_TSV_FIELDS
            }
        )
    assert ORCHESTRATOR._validate_benchmark_tsv(report, path) == 1

    rows = path.read_text(encoding="utf-8").splitlines()
    fields = rows[0].split("\t")
    values = rows[1].split("\t")
    values[fields.index("value")] = "9.0"
    path.write_text("\n".join((rows[0], "\t".join(values))) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON aggregate and TSV records disagree"):
        ORCHESTRATOR._validate_benchmark_tsv(report, path)
