from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from heir.evaluation.broad_type import (
    BROAD_TYPE_INSPECTION_SCHEMA,
    BROAD_TYPE_REPORT_SCHEMA,
    BroadTypeGateBlocked,
    inspect_broad_type_gate,
    load_broad_type_plan,
    run_broad_type_gate,
)
from heir.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "benchmark_broad_types", ROOT / "scripts/benchmark_broad_types.py"
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = SCRIPT
SCRIPT_SPEC.loader.exec_module(SCRIPT)

LABEL_COLUMNS = (
    "section_id",
    "nucleus_id",
    "donor_id",
    "roi_id",
    "compartment",
    "broad_type",
    "reviewer_confidence",
    "reviewer_count",
    "adjudication_status",
    "annotation_source",
    "independent_of_heir_predictions",
)


def _artifact(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _write_ontology(path: Path) -> None:
    path.write_text(
        "broad_type\tdescription\nA\ttype A\nB\ttype B\nC\ttype C\n",
        encoding="utf-8",
    )


def _write_dataset(root: Path, donor: str, *, missing_compartment: bool = False) -> dict:
    rng = np.random.default_rng(int(donor[-1]) + 91)
    nucleus_ids = []
    features = []
    rows = []
    for roi_index in range(4):
        for class_index, label in enumerate(("A", "B", "C")):
            for cell_index in range(4):
                nucleus_id = "%s-r%d-%s-%d" % (donor, roi_index, label, cell_index)
                feature = np.zeros(6, dtype=np.float32)
                feature[class_index] = 3.0
                feature[3 + (roi_index % 3)] = 0.2
                feature += rng.normal(0.0, 0.15, size=6).astype(np.float32)
                nucleus_ids.append(nucleus_id)
                features.append(feature)
                rows.append(
                    {
                        "section_id": donor,
                        "nucleus_id": nucleus_id,
                        "donor_id": donor,
                        "roi_id": "roi-%d" % roi_index,
                        "compartment": (
                            ""
                            if missing_compartment and len(rows) == 0
                            else "compartment-%d" % roi_index
                        ),
                        "broad_type": label,
                        "reviewer_confidence": "0.95",
                        "reviewer_count": "2",
                        "adjudication_status": "adjudicated",
                        "annotation_source": "independent_pathology_review",
                        "independent_of_heir_predictions": "true",
                    }
                )
    features_path = root / (donor + "-features.npz")
    np.savez_compressed(
        features_path,
        nucleus_ids=np.asarray(nucleus_ids, dtype=str),
        features=np.asarray(features, dtype=np.float32),
    )
    labels_path = root / (donor + "-labels.tsv")
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "section_id": donor,
        "donor_id": donor,
        "features": _artifact(features_path),
        "labels": _artifact(labels_path),
    }


def _write_plan(
    tmp_path: Path,
    *,
    status: str = "ready",
    seeds: Optional[List[int]] = None,
    graph: bool = False,
    residual: bool = False,
    missing_compartment: bool = False,
) -> Path:
    ontology = tmp_path / "ontology.tsv"
    _write_ontology(ontology)
    datasets = [
        _write_dataset(
            tmp_path, "donor%d" % index, missing_compartment=missing_compartment and index == 0
        )
        for index in range(3)
    ]
    plan = {
        "schema_version": "heir.broad_type_supervised_plan.v1",
        "status": status,
        "name": "synthetic_broad_gate",
        "device": "cpu",
        "seeds": seeds or [17, 41, 89],
        "graph": {"enabled": graph},
        "molecular_residual": {"enabled": residual},
        "label_policy": {
            "generated_by_pipeline": False,
            "independent_of_heir_predictions_required": True,
            "roi_compartment_required": True,
            "reviewer_confidence_required": True,
        },
        "defaults": {
            "minimum_reviewer_confidence": 0.7,
            "minimum_reviewers": 2,
            "minimum_class_count": 3,
            "accepted_adjudication_statuses": ["adjudicated"],
        },
        "splits": {"calibration_fraction": 0.25, "test_fraction": 0.25},
        "models": {
            "logistic_maximum_iterations": 100,
            "mlp_hidden_dim": 8,
            "mlp_epochs": 8,
            "mlp_learning_rate": 0.02,
            "mlp_weight_decay": 0.0,
        },
        "gate_thresholds": {
            "minimum_macro_f1": 0.0,
            "minimum_image_shuffle_macro_f1_delta": -1.0,
            "maximum_ece": 1.0,
            "minimum_predicted_class_occupancy_fraction": 0.0,
            "minimum_seed_donor_run_pass_fraction": 1.0,
        },
        "tasks": [
            {
                "task_id": "synthetic_lodo",
                "analysis_scope": "test_only",
                "split_policy": "leave_one_donor_out",
                "ontology": _artifact(ontology),
                "datasets": datasets,
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_broad_type_plan_requires_three_seeds_no_graph_and_no_residual(tmp_path: Path) -> None:
    with pytest.raises(BroadTypeGateBlocked, match="three distinct"):
        load_broad_type_plan(_write_plan(tmp_path, seeds=[17, 41]))

    plan = json.loads(_write_plan(tmp_path).read_text(encoding="utf-8"))
    plan["graph"]["enabled"] = True
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(BroadTypeGateBlocked, match="graph.enabled=false"):
        load_broad_type_plan(tmp_path / "plan.json")

    plan["graph"]["enabled"] = False
    plan["molecular_residual"]["enabled"] = True
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(BroadTypeGateBlocked, match="molecular_residual.enabled=false"):
        load_broad_type_plan(tmp_path / "plan.json")


def test_broad_type_labels_fail_closed_without_compartment(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, missing_compartment=True)
    with pytest.raises(BroadTypeGateBlocked, match="missing compartment"):
        run_broad_type_gate(plan, device_name="cpu")


def test_inspection_reports_pending_labels_without_training(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path, status="labels_pending")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["tasks"][0]["datasets"][0]["labels"] = {
        "path": str(tmp_path / "not-reviewed.tsv"),
        "sha256": None,
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = inspect_broad_type_gate(plan_path)

    assert report["schema_version"] == BROAD_TYPE_INSPECTION_SCHEMA
    assert report["status"] == "blocked_evidence"
    assert not report["ready_to_run"]
    assert not report["biological_success_claimed"]
    assert report["execution_contract"] == {
        "graph_used": False,
        "molecular_residual_used": False,
        "labels_generated_by_pipeline": False,
        "minimum_seed_count": 3,
    }
    assert any("labels_pending" in blocker for blocker in report["blockers"])
    assert any("sha256" in blocker for blocker in report["blockers"])

    inspection_output = tmp_path / "inspection.json"
    assert (
        SCRIPT.main(
            [
                "--plan",
                str(plan_path),
                "--output",
                str(inspection_output),
                "--inspect",
            ]
        )
        == 0
    )
    assert json.loads(inspection_output.read_text(encoding="utf-8"))["status"] == (
        "blocked_evidence"
    )

    blocked_output = tmp_path / "blocked.json"
    assert (
        SCRIPT.main(["--plan", str(plan_path), "--output", str(blocked_output), "--device", "cpu"])
        == 2
    )
    blocked = json.loads(blocked_output.read_text(encoding="utf-8"))
    assert blocked["schema_version"] == BROAD_TYPE_REPORT_SCHEMA
    assert blocked["status"] == "blocked_evidence"
    assert not blocked["overall_gate"]["pass"]


def test_broad_type_gate_runs_models_null_calibration_and_grouped_splits(tmp_path: Path) -> None:
    report = run_broad_type_gate(_write_plan(tmp_path), device_name="cpu")

    assert report["schema_version"] == BROAD_TYPE_REPORT_SCHEMA
    assert report["status"] == "complete"
    assert not report["biological_success_claimed"]
    assert report["execution"]["graph_used"] is False
    assert report["execution"]["molecular_residual_used"] is False
    task = report["tasks"][0]
    assert len(task["splits"]) == 9
    assert len(task["runs"]) == 18
    assert {run["model"] for run in task["runs"]} == {
        "balanced_logistic_probe",
        "balanced_frozen_feature_mlp",
    }
    assert all(split["donor_independent_test"] for split in task["splits"])
    assert all(split["roi_disjoint"] for split in task["splits"])
    assert len({split["assignment_sha256"] for split in task["splits"]}) == 6
    assert {
        donor: sum(split["held_out_test_donor"] == donor for split in task["splits"])
        for donor in ("donor0", "donor1", "donor2")
    } == {"donor0": 3, "donor1": 3, "donor2": 3}
    for model in task["summary"]["models"].values():
        assert model["run_count"] == 9
        assert model["seed_count"] == 3
        assert model["held_out_donor_count"] == 3
    for run in task["runs"]:
        assert len(run["image_shuffle"]["permutation_sha256"]) == 64
        for arm in (run["real"], run["image_shuffle_null"]):
            assert "macro_f1" in arm["metrics"]
            assert "ece" in arm["metrics"]
            assert "predicted_class_occupancy_fraction" in arm["occupancy"]
            assert "aurc" in arm["risk_coverage"]
            assert set(arm["risk_coverage"]["risk_at_coverage"]) == {
                "0.50",
                "0.70",
                "0.80",
                "0.90",
                "1.00",
            }
    assert task["gate"]["pass"]
    assert report["overall_gate"]["pass"]


def test_committed_broad_type_schemas_and_ontologies_are_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (
        root / "configs/schemas/broad_type_supervised_plan.schema.json",
        root / "configs/schemas/broad_type_supervised_report.schema.json",
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["$schema"].endswith("2020-12/schema")
    breast = (root / "configs/ontologies/snpatho_broad_4066.tsv").read_text(encoding="utf-8")
    liver = (root / "configs/ontologies/snpatho_broad_liver_metastases.tsv").read_text(
        encoding="utf-8"
    )
    assert "malignant_epithelial" in breast
    assert "metastatic_malignant_epithelial" in liver
    assert "ductal_cholangiocyte" in liver

    plan = root / "configs/broad_type_supervised_gate.example.json"
    readiness = json.loads(
        (root / "reports/snpatho_broad_type_gate_readiness.json").read_text(encoding="utf-8")
    )
    assert readiness["plan"]["sha256"] == sha256_file(plan)
    assert readiness["status"] == "blocked_evidence"
    assert not readiness["ready_to_run"]
    assert not readiness["biological_success_claimed"]
    assert [task["task_id"] for task in readiness["tasks"]] == [
        "snpatho_4066_broad",
        "snpatho_liver_metastases_broad",
    ]
