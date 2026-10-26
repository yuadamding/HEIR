"""The optional broader panel is frozen from training patients alone."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_joint_training import sample_fixture
from test_joint_workflow import _prepared_fixture, _protocol

from heir.personalized import cohort
from heir.personalized import joint_workflow as workflow
from heir.personalized.protein_features import save_joint_inputs
from heir.personalized.study import _sha

SELECTION = {
    "schema": "heir.training_outcome_selection.v1",
    "max_genes": 4,
    "minimum_detection": 0.01,
    "minimum_patients": 2,
    "missing_heldout_panel_policy": "explicit_unevaluable_no_reselection",
}


def test_freeze_panel_opens_only_training_sources_and_joins_detection_by_gene(
    tmp_path, monkeypatch
):
    prepared = tmp_path / "prepared"
    references, samples, sources = {}, {}, []
    for patient in ("train1", "train2"):
        sample, base = sample_fixture(patient)
        reference = sample.inputs.reference
        expression = np.column_stack((reference.expression, [0.0, 2.0, 20.0, 50.0]))
        reference = replace(
            reference,
            gene_ids=(*reference.gene_ids, "variable"),
            expression=expression,
            variance=np.zeros((4, 4)),
        )
        inputs = replace(sample.inputs, reference=reference)
        save_joint_inputs(inputs, prepared / "inputs" / patient, {"fixture": patient})
        references[patient] = reference
        samples[patient] = sample

    def spatial_axis(root, biopsy):
        assert biopsy in samples
        sources.append(biopsy)
        return ("variable", "b", "a", "untrained"), {"sha256": biopsy}

    def load_targets(root, biopsy, inputs, provenance, *, gene_ids):
        assert biopsy in samples
        source = samples[biopsy]
        assert gene_ids == ("variable", "b", "a", "untrained")
        values = np.column_stack(
            (
                np.ones(12),
                source.targets.expression[:, 1],
                source.targets.expression[:, 0],
                np.ones(12),
            )
        )
        targets = replace(
            source.targets, gene_ids=gene_ids, expression=values, observed=np.ones((12, 4), bool)
        )
        return targets, source.observation, {"observation_replicates": source.observation_groups}

    monkeypatch.setattr(workflow, "_spatial_axis", spatial_axis)
    monkeypatch.setattr(cohort, "load_cohort_targets", load_targets)
    output = tmp_path / "fold"
    output.mkdir()
    model = workflow.freeze_fold_outcomes(
        prepared, {"root": "not-opened"}, base, ["train1", "train2"], SELECTION, output
    )
    assert model.gene_ids == (*base.gene_ids, "variable")
    assert model.normalization_gene_ids == model.supervised_gene_ids == model.gene_ids
    record = json.loads((output / "outcome_panel.json").read_text())
    assert record["training_patient_ids"] == ["train1", "train2"]
    assert record["frozen_before_validation_or_test_access"]
    assert not record["selection_uses_validation_or_test"]
    assert sources == ["train1", "train2"]


def test_postfreeze_coverage_never_drops_genes_or_opens_test_ST_early(tmp_path, monkeypatch):
    prepared, sample, base, _, _ = _prepared_fixture(tmp_path)
    metadata, _, _ = workflow._prepared(prepared)
    calls = []

    def spatial(root, biopsy):
        calls.append(biopsy)
        return sample.targets.gene_ids, {"sha256": "target"}

    monkeypatch.setattr(workflow, "_spatial_axis", spatial)
    panel = (*base.gene_ids, "absent")
    report = workflow.panel_coverage(prepared, metadata, ["fixture-0"], panel, spatial=False)
    assert calls == []
    assert report["fixture-0"]["missing_RNA_gene_ids"] == ["absent"]
    assert report["fixture-0"]["missing_spatial_gene_ids"] == []
    report = workflow.panel_coverage(prepared, metadata, ["fixture-0"], panel, spatial=True)
    assert calls == ["fixture-0"]
    assert report["fixture-0"]["missing_spatial_gene_ids"] == ["absent"]
    assert not report["fixture-0"]["panel_changed"]
    assert panel == (*base.gene_ids, "absent")


def test_missing_panel_patient_is_explicit_in_paired_and_cohort_headlines(tmp_path):
    prepared, _, model, _, _ = _prepared_fixture(tmp_path)
    metadata, _, _ = workflow._prepared(prepared)
    plan = [
        {"arm": arm, "seed": seed, "outer_fold": 0} for arm in ("HRP", "HP") for seed in (0, 17, 42)
    ]
    rows = workflow.unevaluable_reports(
        metadata, ["fixture-0"], plan, model, {"reason": "missing_gene", "missing_gene_ids": ["a"]}
    )
    summary = workflow.summarize_joint(rows)
    assert summary["arms"]["HRP"]["n_patients"] == 1
    assert summary["arms"]["HRP"][workflow._SCORE] is None
    assert len(summary["paired_contrasts"]) == 1
    assert summary["paired_contrasts"][0]["delta_" + workflow._SCORE] is None
    assert not any(row["prediction_available"] for row in rows)


def extension_fixture(tmp_path):
    prepared, _, base, config_path, folds = _prepared_fixture(tmp_path, n_patients=13)
    config = json.loads(config_path.read_text())
    config.update(extension_label="broader_training_only", outcome_selection=SELECTION)
    config_path.write_text(json.dumps(config))
    metadata_path = prepared / "prepared.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(config_sha256=_sha(config_path), rna_reference_scope="full_measured_axis")
    metadata_path.write_text(json.dumps(metadata))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(_protocol(_sha(config_path))))
    return prepared, base, folds, protocol


def test_missing_validation_panel_skips_fits_but_preserves_all13_patients(tmp_path, monkeypatch):
    prepared, base, folds, protocol = extension_fixture(tmp_path)

    def freeze(prepared, metadata, base, training, settings, directory):
        workflow._write(directory / "outcome_panel.json", {"train": training})
        return base

    def coverage(prepared, metadata, biopsies, genes, *, spatial):
        return {biopsies[0]: {"missing_spatial_gene_ids": ["unavailable"]}}

    monkeypatch.setattr(workflow, "freeze_fold_outcomes", freeze)
    monkeypatch.setattr(workflow, "panel_coverage", coverage)
    monkeypatch.setattr(workflow, "_fit_and_export", lambda *a, **k: pytest.fail("must not fit"))
    result = workflow.study_joint(prepared, protocol, folds, [0, 17, 42], tmp_path / "output")
    assert result["fit_count"] == 0
    assert result["planned_fit_count"] == 100
    assert len(result["explicit_panel_exclusions"]) == 5
    assert result["arms"]["HRP"]["n_patients"] == 13
    assert result["arms"]["HRP"][workflow._SCORE] is None


def test_config_requires_explicit_bounded_extension_and_pilot_rejects_selection(tmp_path):
    prepared, _, _, protocol = extension_fixture(tmp_path)
    config_path = prepared / "config.json"
    config = json.loads(config_path.read_text())
    workflow._configuration(config_path)
    config["outcome_selection"]["max_genes"] = 257
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="bounded training-only"):
        workflow._configuration(config_path)


def test_missing_test_ST_panel_keeps_predictions_after_export_barrier(tmp_path, monkeypatch):
    from heir.personalized.joint_model import JointModel
    from heir.personalized.serialization import load_prediction, save_prediction

    prepared, base, folds_path, protocol = extension_fixture(tmp_path)
    output = tmp_path / "output"
    current_fold, fit_calls = [None], []
    folds = workflow.validate_folds(folds_path, {f"p{i}" for i in range(13)})

    def freeze(prepared, metadata, base, training, settings, directory):
        current_fold[0] = int(directory.name.split("-")[-1])
        workflow._write(directory / "outcome_panel.json", {"train": training})
        return base

    def coverage(prepared, metadata, biopsies, genes, *, spatial):
        fold = current_fold[0]
        assert (output / f"fold-{fold}/outcome_panel.json").exists()
        if not spatial or all(
            folds[fold][metadata["biopsies"][b]["patient_id"]] == "validation" for b in biopsies
        ):
            return {}
        assert sum(f == fold for f in fit_calls) == 20
        assert (output / f"fold-{fold}/test_predictions_complete.json").exists()
        return {b: {"missing_spatial_gene_ids": ["unavailable"]} for b in biopsies}

    def fit_export(*args, **kwargs):
        arm, seed, test, directory, fold = args[3], args[4], args[7], args[11], args[12]
        fit_calls.append(fold)
        exports = []
        for biopsy in test:
            patient = f"p{biopsy.split('-')[-1]}"
            sample, config = sample_fixture(patient)
            prediction = JointModel(config)(sample.inputs)
            path = directory / f"{biopsy}-{arm}-{seed}.npz"
            save_prediction(prediction, path)
            exports.append(
                {
                    "path": str(path),
                    "biopsy_id": biopsy,
                    "arm": arm,
                    "seed": seed,
                    "outer_fold": fold,
                    "component": "full",
                }
            )
        return exports, [], []

    monkeypatch.setattr(workflow, "freeze_fold_outcomes", freeze)
    monkeypatch.setattr(workflow, "panel_coverage", coverage)
    monkeypatch.setattr(workflow, "_fit_and_export", fit_export)
    monkeypatch.setattr(workflow, "_target", lambda *args: (None, None, None, {}))
    result = workflow.study_joint(prepared, protocol, folds_path, [0, 17, 42], output)
    assert result["fit_count"] == 100
    assert result["arms"]["HRP"]["n_patients"] == 13
    assert result["arms"]["HRP"][workflow._SCORE] is None
    exports = json.loads((output / "exports.json").read_text())
    assert exports
    for item in exports:
        prediction, _ = load_prediction(item["path"])
        assert len(prediction.cell_ids) == 12
        report = json.loads(workflow.Path(item["path"]).with_suffix(".scores.json").read_text())
        assert report["prediction_available"]
        assert report["headline_status"] == "unevaluable_frozen_normalization_panel"
        assert report["normalized_log_mse"] is None


def test_broader_protocol_has30fits_and_hash_bound_separate_configuration():
    folder = workflow.Path(__file__).resolve().parents[1] / "configs/personalized"
    protocol = json.loads((folder / "joint_broader_study.json").read_text())
    config, model = workflow._configuration(folder / protocol["model_config"])
    assert _sha(folder / protocol["model_config"]) == protocol["model_config_sha256"]
    assert len(workflow.fit_schedule(protocol)) == 30
    assert len(model.gene_ids) == 9
    assert len(model.marker_ids) == 50
    assert config["outcome_selection"]["max_genes"] == 256
    assert protocol["control_arms"] == []
