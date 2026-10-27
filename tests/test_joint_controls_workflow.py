"""Matched controls, frozen coverage eligibility and legacy study preservation."""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_joint_training import sample_fixture
from test_joint_workflow import _prepared_fixture, _report

from heir.personalized import joint_workflow as workflow
from heir.personalized.joint_model import JointModel
from heir.personalized.observation import ObservationOperator
from heir.personalized.serialization import save_prediction


def _target(sample):
    return sample.targets, sample.observation, sample.observation_groups, {}


def _coverage(inputs, selected):
    protein = inputs.protein_features
    observed = np.zeros_like(protein.observed)
    observed[selected, 0] = True
    return replace(inputs, protein_features=replace(protein, observed=observed))


def test_actual_control_matrix_has_thirty_matched_fits_and_preserves_old_seventyfive():
    directory = Path(__file__).resolve().parents[1] / "configs/personalized"
    matrix = workflow._backbone_matrix(directory / "joint_controls_study.json")
    assert matrix["total_fits"] == 30
    for backbone in matrix["backbones"]:
        protocol = json.loads((directory / f"joint_controls_{backbone}_study.json").read_text())
        schedule = workflow.fit_schedule(protocol, [0])
        assert len(schedule) == 15
        assert {r["arm"] for r in schedule} == {"HRP", "HRP_mlp", "HRP_coverage_only"}
        assert {r["seed"] for r in schedule} == {0}
        with pytest.raises(ValueError, match="seed inventory"):
            workflow.fit_schedule(protocol, [17])
        settings, model = workflow._configuration(directory / f"joint_controls_{backbone}.json")
        before = torch.random.get_rng_state().clone()
        counts = workflow._parameter_comparison(model, settings)
        assert torch.equal(before, torch.random.get_rng_state())
        assert counts["HRP"]["assignment_trainable_parameters"] == 2048
        assert counts["HRP_mlp"]["assignment_trainable_parameters"] == 2080
        assert counts["HRP_mlp"]["relative_difference_assignment_trainable_parameters"] == 0.015625
        assert workflow._arm_config(model, "HRP_mlp").assignment_network == "pairwise_mlp"
        assert workflow._arm_config(model, "HRP_coverage_only").protein_signal == "coverage_only"
        assert all(
            workflow._arm_config(model, arm).correction_initialization == "zero"
            for arm in ("HRP", *workflow.EXTENSION_CONTROL_ARMS)
        )
    assert workflow._backbone_matrix(directory / "joint_backbones_study.json")["total_fits"] == 75
    assert workflow.CONTROL_ARMS == ("HRP_constant_reference", "HRP_transfer_only")
    assert "protein_intensity_permuted" not in workflow.INTERVENTIONS


def test_committed_matrices_resolve_exact_manifests_without_workstation_access(
    tmp_path, monkeypatch
):
    source = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "portable-checkout"
    shutil.copytree(source / "configs", checkout / "configs")
    shutil.copytree(source / "manifests", checkout / "manifests")
    paths = [
        p for directory in ("configs", "manifests") for p in (checkout / directory).rglob("*.json")
    ]
    original_bytes = {p: p.read_bytes() for p in paths}
    real_open, real_is_file, real_is_dir = Path.open, Path.is_file, Path.is_dir

    def inaccessible(path):
        return str(path).startswith("/storage/HE_GPT")

    def guarded_open(path, *args, **kwargs):
        if inaccessible(path):
            raise FileNotFoundError("simulated absent workstation storage")
        if path.suffix in {".pt", ".bin", ".safetensors"}:
            pytest.fail("configuration validation must not read a model checkpoint")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(
        Path, "is_file", lambda path: False if inaccessible(path) else real_is_file(path)
    )
    monkeypatch.setattr(
        Path, "is_dir", lambda path: False if inaccessible(path) else real_is_dir(path)
    )
    folder = checkout / "configs" / "personalized"
    assert workflow._backbone_matrix(folder / "joint_controls_study.json")["total_fits"] == 30
    assert workflow._backbone_matrix(folder / "joint_backbones_study.json")["total_fits"] == 75
    assert all(path.read_bytes() == contents for path, contents in original_bytes.items())
    # Valid JSON with changed bytes must fail even at a valid fallback location.
    manifest = checkout / "manifests" / "encoders" / "hoptimus1.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        workflow._backbone_matrix(folder / "joint_controls_study.json")


def test_manifest_user_asset_root_requires_exact_bytes_and_no_checkpoint(tmp_path):
    from heir.features.manifest_resolution import resolve_encoder_manifest

    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    manifest = asset_root / "encoder.json"
    manifest.write_bytes(b'{"identity": "fixture manifest, no checkpoint"}\n')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    configured = tmp_path / "retired-workstation" / "encoder.json"
    assert (
        resolve_encoder_manifest(
            configured,
            digest,
            repository_root=tmp_path / "empty-checkout",
            asset_roots=(asset_root,),
        )
        == manifest
    )
    with pytest.raises(FileNotFoundError, match="supplied asset roots"):
        resolve_encoder_manifest(configured, digest, repository_root=tmp_path / "empty-checkout")
    configured.parent.mkdir()
    configured.write_bytes(b'{"wrong": "identity"}\n')
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        resolve_encoder_manifest(configured, digest, asset_roots=(asset_root,))


def test_common_coverage_requires_every_positive_contributor_and_ignores_target_values():
    sample, _ = sample_fixture()
    inputs = _coverage(sample.inputs, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    operator = ObservationOperator(
        sample.inputs.cell_ids,
        sample.targets.observation_ids,
        np.concatenate((np.arange(12), [0, 2])),
        np.concatenate((np.arange(12), [1, 1])),
        np.concatenate((np.ones(12), [0.5, 0.0])),
    )
    target = sample.targets, operator, sample.observation_groups, {}
    restricted, support = workflow._coverage_target(target, inputs)
    assert restricted[0].observation_ids == tuple(f"o{i}" for i in range(2, 12))
    assert support["excluded_observation_ids"] == ["o0", "o1"]
    assert support["target_expression_used_for_coverage"] is False
    changed = replace(sample.targets, expression=sample.targets.expression[::-1].copy())
    _, altered_support = workflow._coverage_target((changed, *target[1:]), inputs)
    assert support == altered_support
    assert restricted[1].weights[-1] == 0
    # Even if a caller mutates the arrays after operator validation, an empty
    # contributing row must not pass the all-contributors coverage criterion.
    operator.weights[operator.rows == 2] = 0
    _, empty_contributor = workflow._coverage_target(target, inputs)
    assert "o2" in empty_contributor["excluded_observation_ids"]


def _exports(tmp_path, sample, config, arms=("HRP",)):
    exports = []
    prediction = JointModel(config)(sample.inputs)
    for arm in arms:
        for component in ("full", "protein_unavailable", "protein_intensity_permuted"):
            path = tmp_path / f"{arm}-{component}.npz"
            save_prediction(prediction, path, {})
            exports.append(
                {
                    "path": str(path),
                    "patient_id": sample.inputs.patient_id,
                    "specimen_id": sample.inputs.specimen_id,
                    "biopsy_id": "fixture-0",
                    "arm": arm,
                    "seed": 0,
                    "outer_fold": 0,
                    "component": component,
                }
            )
    return exports


def test_all_interventions_and_arms_use_the_original_common_rows(tmp_path, monkeypatch):
    sample, config = sample_fixture()
    covered = _coverage(sample.inputs, [0, 1, 2, 6, 7, 8])
    monkeypatch.setattr(workflow, "_inputs", lambda *args: (covered, {}))
    exports = _exports(tmp_path, sample, config, ("HRP", "HRP_mlp", "HRP_coverage_only"))
    reports, rows, errors = workflow._evaluate_coverage_exports(
        "unused", exports, {"fixture-0": _target(sample)}, config
    )
    expected = [f"o{i}" for i in [0, 1, 2, 6, 7, 8]]
    assert all(r["evaluation_observation_ids"] == expected for r in reports)
    assert all(r["normalization_gene_ids"] == list(config.normalization_gene_ids) for r in reports)
    assert all(r["coverage_support"]["covered_prediction_regions"] == 6 for r in reports)
    assert rows and errors
    summary = workflow.summarize_joint(reports)
    assert len(summary["paired_summary"]) == 2
    assert all(
        r["mean_paired_delta_normalized_log_mse"] == 0 for r in summary["paired_summary"].values()
    )


@pytest.mark.parametrize(
    "selected,reason",
    [
        ([], "no_common_original_protein_coverage"),
        ([0, 1, 2], "no_common_protein_coverage_in_some_replicates"),
    ],
)
def test_missing_coverage_is_explicit_undefined_not_dropped_replicate(
    tmp_path, monkeypatch, selected, reason
):
    sample, config = sample_fixture()
    covered = _coverage(sample.inputs, selected)
    monkeypatch.setattr(workflow, "_inputs", lambda *args: (covered, {}))
    exports = _exports(tmp_path, sample, config)
    reports, _, _ = workflow._evaluate_coverage_exports(
        "unused", exports, {"fixture-0": _target(sample)}, config
    )
    for report in reports:
        assert report["headline_status"] == reason
        assert report[workflow._SCORE] is None
        assert report["normalized_log_mse"] is None
        assert "r2" in report["coverage_support"]["replicates_without_covered_observations"]
        assert {r["assay_replicate"] for r in report["replicate_scores"]} == {"r1", "r2"}
    assert workflow.summarize_joint(reports)["arms"]["HRP"][workflow._SCORE] is None


def test_new_control_pairing_rejects_changed_target_support():
    rows = [_report("patient", arm, 0, 0.1) for arm in ("HRP", *workflow.EXTENSION_CONTROL_ARMS)]
    assert set(workflow.summarize_joint(rows)["paired_summary"]) == {
        "HRP-minus-HRP_mlp",
        "HRP-minus-HRP_coverage_only",
    }
    rows[-1]["evaluation_observation_ids"] = ["another"]
    with pytest.raises(ValueError, match="identical target rows"):
        workflow.summarize_joint(rows)


def test_actual_three_control_fit_paths_export_before_accessing_test_targets(tmp_path, monkeypatch):
    prepared, _, base, config_path, _ = _prepared_fixture(tmp_path, n_patients=3)
    settings = json.loads(config_path.read_text())
    base = replace(base, correction_initialization="zero")
    settings["additional_interventions"] = ["protein_intensity_permuted"]
    targets = {f"fixture-{i}": _target(sample_fixture(f"p{i}")[0]) for i in (0, 1)}
    directory = tmp_path / "fold"
    directory.mkdir()
    monkeypatch.setattr(workflow, "qualified_targets", lambda *a, **kw: pytest.fail("target read"))
    exports = []
    for arm in ("HRP", *workflow.EXTENSION_CONTROL_ARMS):
        current, _, _ = workflow._fit_and_export(
            prepared,
            settings,
            base,
            arm,
            0,
            ["fixture-0"],
            ["fixture-1"],
            ["fixture-2"],
            targets,
            None,
            None,
            directory,
            0,
            {"p0": "train", "p1": "validation", "p2": "test"},
        )
        exports.extend(current)
        receipt = json.loads((directory / f"{arm}-seed-0.acceptance.json").read_text())
        assert receipt["save_reload_expression_exact"] and receipt["save_reload_assignment_exact"]
        assert not receipt["test_targets_loaded"]
    assert len(exports) == 11
    assert {r["component"] for r in exports if r["arm"] == "HRP"} == {
        "full",
        "transfer",
        *workflow.INTERVENTIONS,
        "protein_intensity_permuted",
    }
    assert all(Path(e["path"]).is_file() for e in exports)


def test_common_intervention_tables_include_explicit_same_support(tmp_path, monkeypatch):
    sample, config = sample_fixture()
    covered = _coverage(sample.inputs, [0, 1, 2, 6, 7, 8])
    monkeypatch.setattr(workflow, "_inputs", lambda *args: (covered, {}))
    exports = _exports(tmp_path, sample, config)
    targets = {"fixture-0": _target(sample)}
    reports, scores, errors = workflow._evaluate_exports(exports, targets, config)
    common = workflow._evaluate_coverage_exports("unused", exports, targets, config)
    output = tmp_path / "results"
    output.mkdir()
    result = workflow._complete_run(
        output,
        reports,
        scores,
        errors,
        [],
        [],
        exports,
        {"fit_count": 1},
        coverage_results=common,
    )
    assert result[workflow._COVERAGE]["arms"]["HRP"]["n_patients"] == 1
    rows = workflow._read_table(output / "modality_interventions.tsv")
    assert len(rows) == 6
    original = result[workflow._COVERAGE]["coverage_by_biopsy"]["fixture-0"]
    for row in rows:
        if row["evaluation_support"] == workflow._COVERAGE:
            assert (
                json.loads(row["evaluation_observation_ids"])
                == original["evaluation_observation_ids"]
            )
            assert row["change_summary_prediction_regions"] == "6"
        else:
            assert len(json.loads(row["evaluation_observation_ids"])) == 12
            assert row["change_summary_prediction_regions"] == "12"


def test_control_study_rejects_old_preparation_without_explicit_correspondence(tmp_path):
    prepared, _, base, config_path, folds = _prepared_fixture(tmp_path, n_patients=13)
    config = json.loads(config_path.read_text())
    config["model"] = replace(base, correction_initialization="zero").to_dict()
    config.update(
        controls_revision="v010",
        common_protein_coverage=True,
        additional_interventions=["protein_intensity_permuted"],
        assignment_parameter_relative_tolerance=0.02,
        full_model_parameter_relative_tolerance=0.01,
    )
    config_path.write_text(json.dumps(config))
    metadata_path = prepared / "prepared.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["config_sha256"] = workflow._sha(config_path)
    metadata_path.write_text(json.dumps(metadata))
    protocol = {
        "schema": "heir.joint_patient_study.v1",
        "extension_label": "fixture_controls",
        "model_config_sha256": workflow._sha(config_path),
        "principal_arms": ["HRP"],
        "control_arms": list(workflow.EXTENSION_CONTROL_ARMS),
        "seeds": [0],
        "control_seeds": [0],
        "outer_folds": list(range(5)),
    }
    protocol_path = tmp_path / "controls.json"
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="explicit protein correspondence"):
        workflow.study_joint(prepared, protocol_path, folds, [0], tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_new_matrix_combines_only_thirty_fits_and_checks_common_report_identity(
    tmp_path,
    request,
):
    # Reuse the old matrix receipt fixture, then declare the separate new matrix.
    from test_joint_backbones import _mock_completed, backbone_files

    files = backbone_files.__wrapped__(tmp_path)
    matrix_path, matrix, configs, protocols, studies = files
    matrix.update(
        schema="heir.joint_controls_matrix.v1",
        principal_fits=10,
        total_fits=30,
        shared_RP_source=None,
        paired_backbone_arms=["HRP", *workflow.EXTENSION_CONTROL_ARMS],
        correction_initialization="zero",
    )
    for name, config_path in configs.items():
        config = json.loads(config_path.read_text())
        config["model"]["correction_initialization"] = "zero"
        config.update(
            controls_revision="v010",
            common_protein_coverage=True,
            additional_interventions=["protein_intensity_permuted"],
            assignment_parameter_relative_tolerance=0.02,
            full_model_parameter_relative_tolerance=0.01,
        )
        config_path.write_text(json.dumps(config))
        declaration = matrix["backbones"][name]
        declaration.update(
            experiment_config_sha256=workflow._sha(config_path),
            principal_arms=["HRP"],
            control_arms=list(workflow.EXTENSION_CONTROL_ARMS),
            planned_fits=15,
        )
        prepared = Path(declaration["prepared_directory"])
        (prepared / "config.json").write_bytes(config_path.read_bytes())
        metadata_path = prepared / "prepared.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["config_sha256"] = workflow._sha(config_path)
        metadata_path.write_text(json.dumps(metadata))
        # This metadata-only comparison fixture verifies payload identity. Actual
        # correspondence decoding and membership are covered by the adapter tests.
        for path in (prepared / "inputs").glob("*/metadata.json"):
            payload = path.parent / "protein_correspondence.npz"
            payload.write_bytes(b"identical bound correspondence fixture")
            record = json.loads(path.read_text())
            record["payloads"]["protein_correspondence"] = {
                "file": payload.name,
                "sha256": workflow._sha(payload),
            }
            path.write_text(json.dumps(record))
    matrix_path.write_text(json.dumps(matrix))
    for name, path in protocols.items():
        protocol = json.loads(path.read_text())
        protocol.update(
            matrix_protocol_sha256=workflow._sha(matrix_path),
            model_config_sha256=workflow._sha(configs[name]),
            principal_arms=["HRP"],
            control_arms=list(workflow.EXTENSION_CONTROL_ARMS),
        )
        path.write_text(json.dumps(protocol))
    _mock_completed(files)
    for name, directory in studies.items():
        exports_path = directory / "exports.json"
        exports = json.loads(exports_path.read_text())
        for row in list(exports):
            if row["arm"] != "HRP" or row["component"] != "protein_permuted":
                continue
            path = Path(row["path"])
            changed_path = path.with_name(path.stem + "-intensity.npz")
            changed_path.write_bytes(path.read_bytes())
            changed = {**row, "path": str(changed_path), "component": "protein_intensity_permuted"}
            changed_path.with_suffix(".scores.json").write_text(json.dumps(changed))
            receipt_path = path.parent / "test_predictions_complete.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["prediction_sha256"][changed_path.name] = workflow._sha(changed_path)
            receipt_path.write_text(json.dumps(receipt))
            exports.append(changed)
        exports_path.write_text(json.dumps(exports))
        for row in exports:
            Path(row["path"]).with_suffix(".common-coverage.scores.json").write_text(
                json.dumps(row)
            )
        for table in (
            "common_coverage_patient_scores",
            "common_coverage_error_decomposition",
            "common_coverage_paired_contrasts",
        ):
            workflow._tsv(directory / f"{table}.tsv", exports)
        config, model = workflow._configuration(configs[name])
        (directory / "parameter_comparison.json").write_text(
            json.dumps(workflow._parameter_comparison(model, config))
        )
    result = workflow.compare_backbones(
        studies["hoptimus1"], studies["uni2h"], matrix_path, tmp_path / "combined-controls"
    )
    assert result["total_fit_count"] == 30
    assert result["shared_RP_fit_count"] == 0
    assert set(result["backbones"]["hoptimus1"]["arms"]) == {
        "HRP",
        *workflow.EXTENSION_CONTROL_ARMS,
    }
    assert result[workflow._COVERAGE]["uni2h"]["arms"]["HRP"]["n_patients"] == 13
    first = json.loads((studies["uni2h"] / "exports.json").read_text())[0]
    report_path = Path(first["path"]).with_suffix(".common-coverage.scores.json")
    report = json.loads(report_path.read_text())
    report["patient_id"] = "incorrect-patient"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="common-coverage prediction integrity"):
        workflow.compare_backbones(
            studies["hoptimus1"], studies["uni2h"], matrix_path, tmp_path / "invalid-controls"
        )
