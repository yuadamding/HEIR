import numpy as np
import pytest

from heir.uncertainty import MahalanobisOOD


def test_ood_roundtrip_binds_donors_sources_and_feature_space(tmp_path):
    rng = np.random.default_rng(11)
    features = rng.normal(size=(30, 4)).astype(np.float32)
    detector = MahalanobisOOD().fit(
        features,
        analysis_role="development",
        training_donors=("d2", "d1"),
        feature_space_id="encoder-checkpoint-sha256",
    )
    detector.source_sha256 = ("a" * 64,)
    output = tmp_path / "ood.npz"
    detector.to_npz(output)

    loaded = MahalanobisOOD.from_npz(output)
    assert loaded.training_donors == ("d1", "d2")
    assert loaded.feature_space_id == "encoder-checkpoint-sha256"
    np.testing.assert_allclose(loaded.score(features), detector.score(features))

    with np.load(output, allow_pickle=False) as archive:
        malformed = {name: np.array(archive[name], copy=True) for name in archive.files}
    malformed["precision"] = np.eye(3)
    malformed_path = tmp_path / "malformed.npz"
    np.savez_compressed(malformed_path, **malformed)
    with pytest.raises(ValueError, match="precision"):
        MahalanobisOOD.from_npz(malformed_path)


def test_ood_fit_rejects_locked_data_and_unidentified_feature_space():
    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    with pytest.raises(ValueError, match="locked-test"):
        MahalanobisOOD().fit(
            features,
            analysis_role="locked_validation",
            feature_space_id="encoder-v1",
        )
    with pytest.raises(ValueError, match="feature_space_id"):
        MahalanobisOOD().fit(features, analysis_role="development")


def test_ood_score_uses_optimized_einsum_without_changing_quadratic_form(monkeypatch):
    rng = np.random.default_rng(23)
    width = 16
    factor = rng.normal(size=(width, width))
    precision = factor.T @ factor + np.eye(width) * 0.1
    mean = rng.normal(size=width)
    features = rng.normal(size=(41, width))
    detector = MahalanobisOOD(
        mean=mean,
        precision=precision,
        threshold=1.0,
        feature_space_id="encoder-v1",
    )

    original_einsum = np.einsum
    optimized = []

    def recording_einsum(*args, **kwargs):
        optimized.append(kwargs.get("optimize"))
        return original_einsum(*args, **kwargs)

    monkeypatch.setattr(np, "einsum", recording_einsum)
    observed = detector.score(features)
    delta = features - mean
    expected = np.sqrt(np.maximum(np.sum((delta @ precision) * delta, axis=1), 0.0))

    assert optimized == [True]
    np.testing.assert_allclose(observed, expected.astype(np.float32), rtol=2.0e-6, atol=2.0e-6)
