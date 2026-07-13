import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from heir.data import HistologyBag
from heir.uncertainty import MahalanobisOOD

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "calibrate_target_ood.py"
SPEC = importlib.util.spec_from_file_location("calibrate_target_ood", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CALIBRATION
SPEC.loader.exec_module(CALIBRATION)


def _inputs(tmp_path):
    rng = np.random.default_rng(31)
    development = rng.normal(size=(40, 3)).astype(np.float32)
    base = MahalanobisOOD().fit(
        development,
        analysis_role="development",
        quantile=0.9,
        training_donors=("B1",),
        feature_space_id="omiclip-test-v1",
    )
    base.source_sha256 = ("a" * 64,)
    base_path = tmp_path / "base.npz"
    base.to_npz(base_path)
    target = rng.normal(loc=0.5, size=(23, 3)).astype(np.float32)
    bag = HistologyBag(
        slide_id="4066",
        nucleus_ids=np.asarray(["n%d" % index for index in range(len(target))]),
        features=target,
        coordinates_um=np.column_stack((np.arange(len(target)), np.zeros(len(target)))),
        sample_id="4066",
        donor_id="4066",
        block_id="4066_FFPE",
        feature_space_id="omiclip-test-v1",
    )
    histology_path = tmp_path / "histology.npz"
    bag.save_npz(histology_path)
    return base_path, histology_path, target


def test_target_histology_telemetry_preserves_development_threshold_and_is_byte_stable(
    tmp_path,
):
    base_path, histology_path, target = _inputs(tmp_path)
    outputs = []
    for suffix in ("first", "second"):
        output = tmp_path / (suffix + ".npz")
        provenance = tmp_path / (suffix + ".json")
        CALIBRATION.calibrate(
            base_ood_path=base_path,
            histology_path=histology_path,
            sample_id="4066",
            quantile=0.95,
            output_path=output,
            provenance_path=provenance,
            score_batch_size=7,
        )
        outputs.append(output)

        base = MahalanobisOOD.from_npz(base_path)
        calibrated = MahalanobisOOD.from_npz(output)
        np.testing.assert_array_equal(calibrated.mean, base.mean)
        np.testing.assert_array_equal(calibrated.precision, base.precision)
        assert calibrated.training_donors == base.training_donors == ("B1",)
        assert calibrated.source_sha256 == base.source_sha256 == ("a" * 64,)
        target_quantile = float(np.quantile(base.score(target).astype(np.float64), 0.95))
        assert calibrated.threshold == base.threshold
        assert calibrated.quantile == base.quantile == 0.9

        payload = json.loads(provenance.read_text())
        assert payload["sample_id"] == "4066"
        assert payload["target_expression_accessed"] is False
        assert payload["threshold_source"] == "development_detector"
        assert payload["descriptive_target_quantile"] == 0.95
        assert payload["descriptive_target_quantile_value"] == target_quantile
        assert payload["score_stats"]["count"] == len(target)
        assert (
            payload["inputs"]["base_ood"]["sha256"]
            == hashlib.sha256(base_path.read_bytes()).hexdigest()
        )
        assert (
            payload["inputs"]["histology"]["sha256"]
            == hashlib.sha256(histology_path.read_bytes()).hexdigest()
        )
        with np.load(output, allow_pickle=False) as archive:
            assert not bool(np.asarray(archive["target_expression_accessed"]).item())
            assert str(np.asarray(archive["sample_id"]).item()) == "4066"
            assert str(np.asarray(archive["threshold_source"]).item()) == ("development_detector")
            assert float(np.asarray(archive["target_score_quantile_value"]).item()) == (
                target_quantile
            )

    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_target_histology_calibration_rejects_identity_drift_and_existing_outputs(tmp_path):
    base_path, histology_path, _ = _inputs(tmp_path)
    with pytest.raises(ValueError, match="sample_id"):
        CALIBRATION.calibrate(
            base_ood_path=base_path,
            histology_path=histology_path,
            sample_id="wrong",
            quantile=0.95,
            output_path=tmp_path / "wrong.npz",
            provenance_path=tmp_path / "wrong.json",
            score_batch_size=8,
        )

    output = tmp_path / "existing.npz"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="already exist"):
        CALIBRATION.calibrate(
            base_ood_path=base_path,
            histology_path=histology_path,
            sample_id="4066",
            quantile=0.95,
            output_path=output,
            provenance_path=tmp_path / "existing.json",
            score_batch_size=8,
        )
    assert output.read_bytes() == b"keep"


def test_target_score_distribution_cannot_change_development_threshold(tmp_path):
    base_path, histology_path, _ = _inputs(tmp_path)
    original = HistologyBag.load_npz(histology_path)
    shifted_path = tmp_path / "shifted_histology.npz"
    shifted = HistologyBag(
        slide_id=original.slide_id,
        nucleus_ids=original.nucleus_ids,
        features=original.features + 100.0,
        coordinates_um=original.coordinates_um,
        sample_id=original.sample_id,
        donor_id=original.donor_id,
        block_id=original.block_id,
        feature_space_id=original.feature_space_id,
    )
    shifted.save_npz(shifted_path)
    thresholds = []
    descriptive_values = []
    for label, source in (("original", histology_path), ("shifted", shifted_path)):
        output = tmp_path / (label + ".npz")
        provenance = tmp_path / (label + ".json")
        payload = CALIBRATION.calibrate(
            base_ood_path=base_path,
            histology_path=source,
            sample_id="4066",
            quantile=0.95,
            output_path=output,
            provenance_path=provenance,
            score_batch_size=8,
        )
        thresholds.append(MahalanobisOOD.from_npz(output).threshold)
        descriptive_values.append(payload["descriptive_target_quantile_value"])

    assert thresholds[0] == thresholds[1] == MahalanobisOOD.from_npz(base_path).threshold
    assert descriptive_values[0] != descriptive_values[1]
