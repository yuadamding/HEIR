from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from heir.evaluation.blank_patch_receipt import FEATURE_WIDTH, array_sha256


def _builder():
    path = Path(__file__).parents[1] / "scripts" / "build_hest_blank_patch_supplement.py"
    spec = importlib.util.spec_from_file_location("build_hest_blank_patch_supplement", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _builder()


class _Encoder:
    feature_width = FEATURE_WIDTH

    def encode(self, patches: np.ndarray) -> np.ndarray:
        assert patches.dtype == np.uint8
        assert np.all(patches == 255)
        return np.tile(np.linspace(0.1, 1.0, FEATURE_WIDTH, dtype=np.float32), (len(patches), 1))


def test_encoder_is_called_with_exact_batch_sizes_one_and_32() -> None:
    calls: list[tuple[int, ...]] = []

    class TrackingEncoder(_Encoder):
        def encode(self, patches: np.ndarray) -> np.ndarray:
            calls.append(patches.shape)
            return super().encode(patches)

    single, repeated = BUILDER._encode_blank_batches(TrackingEncoder())
    assert calls == [(1, 224, 224, 3), (32, 224, 224, 3)]
    assert single.shape == (1, FEATURE_WIDTH)
    assert repeated.shape == (32, FEATURE_WIDTH)


def test_atomic_npz_stores_one_vector_and_explicit_broadcast(tmp_path: Path) -> None:
    output = tmp_path / "supplement.npz"
    feature = np.linspace(0.1, 1.0, FEATURE_WIDTH, dtype=np.float32)[None, :]
    arrays = {
        "blank_patch_feature": feature,
        "broadcast_rows": np.asarray(36121, dtype=np.int64),
        "broadcast_rule": np.asarray("repeat_identical_feature_for_each_observation"),
    }
    BUILDER._atomic_npz(output, arrays)
    with np.load(output, allow_pickle=False) as archive:
        assert archive["blank_patch_feature"].shape == (1, FEATURE_WIDTH)
        assert int(archive["broadcast_rows"].item()) == 36121
        assert "36121" not in str(archive["blank_patch_feature"].shape)


def test_cli_exposes_no_supplied_feature_vector() -> None:
    parser = BUILDER._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--feature", "manual.npy"])


def test_batch_output_hash_is_bound_to_shape() -> None:
    one = np.ones((1, FEATURE_WIDTH), dtype=np.float32)
    repeated = np.ones((32, FEATURE_WIDTH), dtype=np.float32)
    assert array_sha256(one) != array_sha256(repeated)
