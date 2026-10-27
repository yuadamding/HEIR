"""Representation diagnostics inspect saved predictors without spatial targets."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_joint_model import joint_fixture

from heir.personalized.joint_model import JointModel
from heir.personalized.protein_features import ProteinFeatureScaler, save_joint_inputs
from heir.personalized.representations import (
    _similarity,
    _variation,
    analyze_saved_representations,
    representation_diagnostics,
)
from heir.personalized.serialization import save_prediction
from heir.personalized.train import FittedJointModel


def _fitted():
    inputs, config = joint_fixture()
    return inputs, FittedJointModel(
        JointModel(config),
        np.zeros(3),
        np.ones(3),
        np.zeros(2),
        np.ones(2),
        ProteinFeatureScaler.fit([inputs]),
        (inputs.patient_id,),
        (),
        [],
        {},
        {"prediction_unit": inputs.prediction_unit, "coordinate_units": inputs.coordinate_units},
        {"epoch": 1, "stage": "joint", "correction_enabled": True},
    )


def test_variation_and_reference_similarity_have_explicit_population_axes():
    report = _variation(np.array([[0, 2], [2, 2], [4, 2]]))
    assert report["population_std_across_rows"]["maximum"] == pytest.approx(np.sqrt(8 / 3))
    assert report["constant_dimensions"] == 1
    similarity = _similarity(np.array([[1, 0], [1, 0], [0, 1], [0, 0]]), ("a", "a", "b", "b"))
    assert similarity["within_broad_type_pairs"]["mean"] == 1
    assert similarity["between_broad_type_pairs"]["mean"] == 0
    assert similarity["undefined_zero_norm_pairs"] == 3


def test_diagnostics_capture_real_saturation_logits_and_preserve_model_state():
    inputs, fitted = _fitted()
    with torch.no_grad():
        fitted.model.image_encoder[0].weight.zero_()
        fitted.model.image_encoder[0].bias.fill_(4)
    expected = fitted.predict(inputs)
    fitted.model.train()
    report = representation_diagnostics(fitted, inputs, expected_prediction=expected)
    assert fitted.model.training
    assert not any(module._forward_hooks for module in fitted.model.modules())
    assert report["saved_prediction_exact"]
    assert not report["targets_used"] and not report["fitting_performed"]
    assert report["stages"]["raw_frozen_image_features"]["mean_variance_across_rows"] > 0
    assert report["stages"]["image_projection_h"]["constant_dimensions"] == 32
    assert (
        report["tanh_saturation"]["image_projection_h"][
            "fraction_absolute_output_at_least_threshold"
        ]
        == 1
    )
    assert report["stages"]["type_logits"]["feature_width"] == 2
    assert report["stages"]["state_logits"]["feature_width"] == 4
    assert report["stages"]["pooled_type_descriptors"]["n_rows"] == 2
    assert report["stages"]["linear_reference_transfer_qB"]["feature_width"] == 3
    assert set(report["assignment_logit_spread"]["within_type_state_logit_std_per_region"]) == {
        "immune",
        "tumor",
    }
    json.dumps(report, allow_nan=False)


def test_saved_analysis_never_opens_target_files_and_writes_only_new_report(tmp_path, monkeypatch):
    inputs, fitted = _fitted()
    prepared, study = tmp_path / "prepared", tmp_path / "study"
    prepared.mkdir()
    (prepared / "prepared.json").write_text("{}")
    directory = prepared / "inputs" / "fixture"
    save_joint_inputs(inputs, directory, {})
    bundle = study / "fold-0" / "HRP-seed-0.bundle"
    bundle.parent.mkdir(parents=True)
    fitted.save(bundle)
    prediction = bundle.parent / "test-fixture.npz"
    save_prediction(fitted.predict(inputs), prediction)
    exports = [
        {
            "path": str(prediction),
            "biopsy_id": "fixture",
            "arm": "HRP",
            "outer_fold": 0,
            "seed": 0,
            "component": "full",
        }
    ]
    (study / "exports.json").write_text(json.dumps(exports))

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (study / "identity.json").write_text(
        json.dumps(
            {
                "prepared_sha256": sha(prepared / "prepared.json"),
                "input_metadata_sha256": {"fixture": sha(directory / "metadata.json")},
                "source_file_sha256": {"fixture_model_source": "fixture"},
            }
        )
    )
    before = {path: sha(path) for path in tmp_path.rglob("*") if path.is_file()}
    original_open = Path.open

    def no_targets(path, *args, **kwargs):
        if "targets" in path.parts:
            pytest.fail("representation analysis must not open any target table")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_targets)
    output = tmp_path / "new-analysis.json"
    result = analyze_saved_representations(prepared, study, output)
    assert result["arms"]["HRP"]["n_predictions"] == 1
    assert result["records"][0]["saved_prediction_exact"]
    assert all(sha(path) == digest for path, digest in before.items())
    with pytest.raises(FileExistsError):
        analyze_saved_representations(prepared, study, output)
