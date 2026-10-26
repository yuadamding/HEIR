"""A frozen image backbone is identified by its manifest, never by feature width."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_joint_model import FIXTURE_BACKBONE, arm, joint_fixture
from test_joint_training import sample_fixture

from heir.personalized.joint_model import JointModel
from heir.personalized.protein_features import load_joint_inputs, save_joint_inputs
from heir.personalized.train import FittedJointModel, fit_joint, load_fitted_model

OTHER_BACKBONE = "b" * 64


def _rehash_bundle(directory):
    hashes = json.loads((directory / "sha256.json").read_text())
    hashes["bundle.json"] = hashlib.sha256((directory / "bundle.json").read_bytes()).hexdigest()
    (directory / "sha256.json").write_text(json.dumps(hashes))


def test_same1536_width_other_backbone_is_rejected_before_forward():
    inputs, config = joint_fixture()
    inputs = replace(inputs, image_features=np.ones((len(inputs.cell_ids), 1536)))
    config = replace(config, feature_dim=1536)
    model = JointModel(config)
    expected = model(inputs)
    other = replace(inputs, image_encoder_manifest_sha256=OTHER_BACKBONE)
    with pytest.raises(ValueError, match="encoder manifest identity differs"):
        model(other)
    # Regions, feature values, and dimensionality are identical; only the encoder differs.
    assert np.array_equal(inputs.image_features, other.image_features)
    assert expected.expression.shape == (12, 3)


def test_unpinned_or_malformed_image_identity_is_not_a_synthetic_escape():
    inputs, config = joint_fixture()
    for pin in (None, "H-optimus-1", "z" * 64, "a" * 63):
        with pytest.raises(ValueError, match="pinned lowercase SHA-256"):
            replace(inputs, image_encoder_manifest_sha256=pin)
        with pytest.raises(ValueError, match="pinned lowercase SHA-256"):
            replace(config, image_encoder_manifest_sha256=pin)
    rp_inputs, rp_config = arm(inputs, config, "RP")
    assert rp_inputs.image_encoder_manifest_sha256 is None
    assert rp_config.image_encoder_manifest_sha256 is None
    with pytest.raises(ValueError, match="absent image modality"):
        replace(rp_inputs, image_encoder_manifest_sha256=FIXTURE_BACKBONE)
    with pytest.raises(ValueError, match="absent image modality"):
        replace(rp_config, image_encoder_manifest_sha256=FIXTURE_BACKBONE)


def test_fit_and_reloaded_bundle_enforce_exact_backbone(tmp_path):
    sample, config = sample_fixture()
    other = replace(
        sample, inputs=replace(sample.inputs, image_encoder_manifest_sha256=OTHER_BACKBONE)
    )
    with pytest.raises(ValueError, match="encoder manifest identity differs"):
        fit_joint([other], [], config, epochs=1)
    fitted = fit_joint([sample], [], config, epochs=1, warmup_epochs=0)
    path = tmp_path / "bundle"
    fitted.save(path)
    metadata = json.loads((path / "bundle.json").read_text())
    assert metadata["schema"] == "heir.joint.bundle.v2"
    assert metadata["model_config"]["image_encoder_manifest_sha256"] == FIXTURE_BACKBONE
    restored = load_fitted_model(path)
    torch.testing.assert_close(
        restored.predict(sample.inputs).expression,
        fitted.predict(sample.inputs).expression,
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="encoder manifest identity differs"):
        restored.predict(other.inputs)


def test_old_prepared_identity_recovers_from_bound_cache_provenance_but_RP_does_not_read_it(
    tmp_path,
):
    inputs, _ = joint_fixture()
    provenance = {
        "image_features": {
            "encoder_manifest_sha256": FIXTURE_BACKBONE,
            "metadata": {"encoder_manifest_sha256": FIXTURE_BACKBONE},
        }
    }
    path = tmp_path / "inputs"
    save_joint_inputs(inputs, path, provenance)
    record = json.loads((path / "metadata.json").read_text())
    del record["image_encoder_manifest_sha256"]
    (path / "metadata.json").write_text(json.dumps(record))
    restored, _ = load_joint_inputs(path)
    assert restored.image_encoder_manifest_sha256 == FIXTURE_BACKBONE
    record["provenance"]["image_features"]["metadata"]["encoder_manifest_sha256"] = OTHER_BACKBONE
    (path / "metadata.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="embedded cache identity"):
        load_joint_inputs(path)
    (path / "image.npz").unlink()
    rp, _ = load_joint_inputs(path, modalities=("R", "P"))
    assert rp.image_features is None and rp.image_encoder_manifest_sha256 is None


def test_new_prepared_explicit_identity_cannot_disagree_with_provenance(tmp_path):
    inputs, _ = joint_fixture()
    path = tmp_path / "inputs"
    save_joint_inputs(inputs, path, {"image_features": {"encoder_manifest_sha256": OTHER_BACKBONE}})
    with pytest.raises(ValueError, match="cache provenance"):
        load_joint_inputs(path)


def test_historical_joint_bundle_needs_hash_bound_identity_and_preserves_predictions(tmp_path):
    sample, config = sample_fixture()
    preparation = {
        "joint_workflow": {"preparation": {"feature_encoder_manifest_sha256": FIXTURE_BACKBONE}}
    }
    fitted = fit_joint([sample], [], config, epochs=1, preparation=preparation)
    directory = tmp_path / "historical"
    fitted.save(directory)
    metadata = json.loads((directory / "bundle.json").read_text())
    metadata["schema"] = "heir.joint.bundle.v1"
    del metadata["model_config"]["image_encoder_manifest_sha256"]
    (directory / "bundle.json").write_text(json.dumps(metadata))
    _rehash_bundle(directory)
    restored = FittedJointModel.load(directory)
    assert restored.model.config.image_encoder_manifest_sha256 == FIXTURE_BACKBONE
    torch.testing.assert_close(
        restored.predict(sample.inputs).expression,
        fitted.predict(sample.inputs).expression,
        rtol=0,
        atol=0,
    )
    del metadata["preparation"]["joint_workflow"]["preparation"]["feature_encoder_manifest_sha256"]
    (directory / "bundle.json").write_text(json.dumps(metadata))
    _rehash_bundle(directory)
    with pytest.raises(ValueError, match="lacks a bound image encoder identity"):
        FittedJointModel.load(directory)
