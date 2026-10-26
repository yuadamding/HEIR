"""Two frozen backbones share geometry and molecular inputs, never encoder identity."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_joint_training import sample_fixture
from test_joint_workflow import _report

from heir.personalized import joint_workflow as workflow
from heir.personalized.protein_features import save_joint_inputs
from heir.personalized.study import _sha
from heir.personalized.train import make_patient_folds


def _json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def backbone_files(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    inputs, original_model = sample_fixture()[0].inputs, sample_fixture()[1]
    model = replace(original_model, feature_dim=1536)
    settings = {"epochs": 100, "warmup_epochs": 30, "learning_rate": 0.001}
    folds = make_patient_folds([f"p{i}" for i in range(13)])
    declarations, studies, configs = {}, {}, {}
    for name in ("hoptimus1", "uni2h"):
        manifest = repo / "manifests/encoders" / f"{name}.json"
        pin = _sha(manifest)
        current_model = replace(model, image_encoder_manifest_sha256=pin)
        prep = {
            "region_size": 100,
            "max_states": 4,
            "feature_cache_dir": str(tmp_path / (name + "-cache")),
            "feature_encoder_manifest_sha256": pin,
        }
        config = {
            "schema": "heir.joint_region_workflow.v1",
            "preparation": prep,
            "model": current_model.to_dict(),
            "training": settings,
            "image_backbone": {
                "backbone_id": name,
                "mode": "frozen",
                "manifest": str(manifest),
                "manifest_sha256": pin,
                "feature_dim": 1536,
                "feature_cache_dir": prep["feature_cache_dir"],
                "crop_mode": "historical_registered_region",
            },
            "pilot_arms": ["H", "HR", "HRP"],
        }
        config_path = tmp_path / f"{name}.json"
        _json(config_path, config)
        prepared = tmp_path / (name + "-prepared")
        prepared.mkdir()
        original_config = json.loads(json.dumps(config))
        if name == "hoptimus1":
            original_config.pop("image_backbone")
            original_config.pop("pilot_arms")
            original_config["model"].pop("image_encoder_manifest_sha256")
        _json(prepared / "config.json", original_config)
        with (prepared / "folds.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(folds[0]))
            writer.writeheader()
            writer.writerows(folds)
        records = {}
        for i in range(13):
            biopsy = f"b{i}"
            sample = replace(
                inputs,
                patient_id=f"p{i}",
                specimen_id=f"p{i}-s",
                image_features=np.tile(inputs.image_features, (1, 512)),
                image_encoder_manifest_sha256=pin,
            )
            provenance = {"sources": {"he": "same", "rna": "same", "protein": "same"}}
            save_joint_inputs(sample, prepared / "inputs" / biopsy, provenance)
            records[biopsy] = {
                "patient_id": f"p{i}",
                "specimen_id": f"p{i}-s",
                "provenance": provenance,
            }
        metadata = {
            "schema": "heir.joint_prepared.v1",
            "root": "unused_fixture",
            "config_sha256": _sha(prepared / "config.json"),
            "folds_sha256": _sha(prepared / "folds.csv"),
            "biopsies": records,
            "spatial_targets_loaded": False,
        }
        _json(prepared / "prepared.json", metadata)
        arms = list(workflow.PRINCIPAL_ARMS)
        if name == "uni2h":
            arms.remove("RP")
        declarations[name] = {
            "experiment_config": config_path.name,
            "experiment_config_sha256": _sha(config_path),
            "prepared_directory": str(prepared),
            "principal_arms": arms,
            "control_arms": list(workflow.CONTROL_ARMS),
            "planned_fits": 40 if name == "hoptimus1" else 35,
        }
        configs[name] = config_path
        studies[name] = tmp_path / (name + "-study")
    matrix = {
        "schema": "heir.joint_backbone_matrix.v1",
        "backbones": declarations,
        "seeds": [0],
        "outer_folds": list(range(5)),
        "principal_fits": 55,
        "mechanistic_fits": 20,
        "total_fits": 75,
        "shared_RP_source": "hoptimus1",
        "further_seeds_in_this_revision": [],
        "panel_or_context_extensions_in_this_revision": False,
        "decision_frozen_before_first_heldout_result": True,
        "additional_seed_17_42_fits_authorized": False,
        "expanded_or_broader_panels_scored": False,
        "paired_backbone_arms": [a for a in workflow.PRINCIPAL_ARMS if a != "RP"],
    }
    matrix_path = tmp_path / "matrix.json"
    _json(matrix_path, matrix)
    protocols = {}
    for name, declaration in declarations.items():
        protocol = {
            "schema": "heir.joint_patient_study.v1",
            "extension_label": name,
            "backbone_id": name,
            "matrix_protocol": matrix_path.name,
            "matrix_protocol_sha256": _sha(matrix_path),
            "model_config_sha256": declaration["experiment_config_sha256"],
            "principal_arms": declaration["principal_arms"],
            "control_arms": declaration["control_arms"],
            "outer_folds": list(range(5)),
            "seeds": [0],
            "control_seeds": [0],
            "test_prediction_before_target_loading": True,
            "no_train_validation_refit": True,
            "no_further_seed_panel_or_context_runs_in_this_revision": True,
        }
        path = tmp_path / f"{name}-protocol.json"
        _json(path, protocol)
        protocols[name] = path
    return matrix_path, matrix, configs, protocols, studies


def test_backbone_matrix_exact75_and_bound_legacy_preparation_reuse(backbone_files):
    matrix_path, matrix, _, protocols, _ = backbone_files
    checked = workflow._backbone_matrix(matrix_path)
    assert checked["principal_fits"] + checked["mechanistic_fits"] == 75
    plans = []
    for name in ("hoptimus1", "uni2h"):
        prepared = matrix["backbones"][name]["prepared_directory"]
        _, _, model, protocol, _ = workflow._study_configuration(prepared, protocols[name])
        plan = workflow.fit_schedule(protocol, [0])
        assert len(plan) == (40 if name == "hoptimus1" else 35)
        assert workflow._arm_config(model, "RP").image_encoder_manifest_sha256 is None
        plans.extend(plan)
    assert len(plans) == 75
    assert sum(r["arm"] == "RP" for r in plans) == 5
    assert {r["seed"] for r in plans} == {0}


def test_override_rejects_unlisted_nonimage_rule_change(backbone_files):
    matrix_path, matrix, configs, protocols, _ = backbone_files
    for name, config_path in configs.items():
        config = json.loads(config_path.read_text())
        config["unknown_training_rule"] = "changed"
        _json(config_path, config)
        matrix["backbones"][name]["experiment_config_sha256"] = _sha(config_path)
    _json(matrix_path, matrix)
    for name, path in protocols.items():
        protocol = json.loads(path.read_text())
        protocol["matrix_protocol_sha256"] = _sha(matrix_path)
        protocol["model_config_sha256"] = _sha(configs[name])
        _json(path, protocol)
    with pytest.raises(ValueError, match="override differs"):
        workflow._study_configuration(
            matrix["backbones"]["hoptimus1"]["prepared_directory"], protocols["hoptimus1"]
        )


def _mock_completed(backbone_files):
    matrix_path, matrix, configs, protocols, studies = backbone_files
    for name, directory in studies.items():
        directory.mkdir()
        prepared = Path(matrix["backbones"][name]["prepared_directory"])
        metadata, _, _ = workflow._prepared(prepared)
        for output_name, source in (
            ("protocol.json", protocols[name]),
            ("config.json", configs[name]),
            ("folds.csv", prepared / "folds.csv"),
        ):
            (directory / output_name).write_bytes(source.read_bytes())
        protocol = json.loads(protocols[name].read_text())
        identity = {
            "prepared_sha256": _sha(prepared / "prepared.json"),
            "input_metadata_sha256": workflow._input_inventory(prepared, metadata),
            "protocol_sha256": _sha(protocols[name]),
            "experiment_config_sha256": _sha(configs[name]),
            "matrix_protocol_sha256": _sha(matrix_path),
            "folds_sha256": _sha(directory / "folds.csv"),
            "source_file_sha256": {"fixture.py": "same_source"},
        }
        _json(directory / "identity.json", identity)
        folds = workflow.validate_folds(directory / "folds.csv", {f"p{i}" for i in range(13)})
        records, exports, receipts = [], [], {fold: {} for fold in folds}
        plan = workflow.fit_schedule(protocol)
        for fit in plan:
            fold_dir = directory / f"fold-{fit['outer_fold']}"
            fold_dir.mkdir(exist_ok=True)
            (fold_dir / f"{fit['arm']}.bundle").mkdir()
            components = ["full"]
            if "R" in fit["arm"].split("_")[0] and fit["arm"] != "HRP_transfer_only":
                components.append("transfer")
            if fit["arm"] == "HRP":
                components.extend(workflow.INTERVENTIONS)
            for biopsy, source in metadata["biopsies"].items():
                if folds[fit["outer_fold"]][source["patient_id"]] != "test":
                    continue
                for component in components:
                    path = fold_dir / f"{biopsy}-{fit['arm']}-{component}.npz"
                    path.write_bytes(b"fixture frozen prediction bytes")
                    report = {
                        **_report(source["patient_id"], fit["arm"], 0, 0.1),
                        "biopsy_id": biopsy,
                        "outer_fold": fit["outer_fold"],
                        "component": component,
                        "prediction_sha256": _sha(path),
                    }
                    _json(path.with_suffix(".scores.json"), report)
                    exports.append({"path": str(path), **report})
                    records.append(report)
                    receipts[fit["outer_fold"]][path.name] = _sha(path)
        for fold, receipt in receipts.items():
            _json(
                directory / f"fold-{fold}/test_predictions_complete.json",
                {"test_targets_loaded": False, "prediction_sha256": receipt},
            )
        _json(directory / "exports.json", exports)
        _json(directory / "summary.json", {"fit_count": len(plan), "postfit_transfer": {}})
        for table in (
            "patient_scores",
            "error_decomposition",
            "modality_interventions",
            "assignment_diagnostics",
            "marker_coverage",
        ):
            workflow._tsv(directory / f"{table}.tsv", records)
    return studies


def test_combined_report_shares_RP_and_does_not_count_extra_fits(backbone_files, tmp_path):
    studies = _mock_completed(backbone_files)
    result = workflow.compare_backbones(
        studies["hoptimus1"], studies["uni2h"], backbone_files[0], tmp_path / "combined"
    )
    assert result["total_fit_count"] == 75
    assert result["principal_fit_count"] == 55
    assert not result["shared_RP_refitted_for_UNI2h"]
    for name in ("hoptimus1", "uni2h"):
        assert result["backbones"][name]["arms"]["RP"]["n_patients"] == 13
        assert "HRP-minus-RP" in result["backbones"][name]["paired_summary"]


def test_omitting_same_patient_from_both_studies_is_rejected(backbone_files, tmp_path):
    studies = _mock_completed(backbone_files)
    for directory in studies.values():
        path = directory / "exports.json"
        _json(path, [r for r in json.loads(path.read_text()) if r["patient_id"] != "p0"])
    with pytest.raises(ValueError, match="export inventory"):
        workflow.compare_backbones(
            studies["hoptimus1"], studies["uni2h"], backbone_files[0], tmp_path / "invalid"
        )


def test_different_molecular_payloads_rejected_even_with_matching_geometry(backbone_files):
    _, matrix, _, _, _ = backbone_files
    identities = {}
    for name, declaration in matrix["backbones"].items():
        prepared = Path(declaration["prepared_directory"])
        if name == "uni2h":
            path = prepared / "inputs/b0/metadata.json"
            data = json.loads(path.read_text())
            data["payloads"]["reference"]["sha256"] = "0" * 64
            _json(path, data)
        metadata, _, _ = workflow._prepared(prepared)
        identities[name] = {
            "prepared_sha256": _sha(prepared / "prepared.json"),
            "input_metadata_sha256": workflow._input_inventory(prepared, metadata),
        }
    with pytest.raises(ValueError, match="exact geometry, RNA and protein"):
        workflow._matching_backbone_inputs(matrix, identities)
