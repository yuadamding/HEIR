from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from heir.evaluation import blank_patch_receipt as receipt
from heir.features import load_encoder_manifest
from heir.features.base import sha256_file

ROOT = Path(__file__).parents[1]


def _builder():
    path = ROOT / "scripts/build_hest_blank_patch_supplement.py"
    spec = importlib.util.spec_from_file_location("build_hest_blank_patch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _builder()
MANIFEST = load_encoder_manifest(ROOT / "manifests/encoders/hoptimus1.json")


def _feature(rows: int) -> np.ndarray:
    values = np.arange(1, receipt.FEATURE_WIDTH + 1, dtype=np.float32)
    return np.repeat(values[None, :], rows, axis=0)


def _source(tmp_path: Path) -> tuple[Path, receipt.SourceIdentity]:
    path = tmp_path / "source.npz"
    np.savez(
        path,
        observation_ids=np.asarray(["cell-2", "cell-1", "cell-3"]),
        nucleus_molecular_targets=np.asarray([[object()]], dtype=object),
        image_features=np.asarray([[object()]], dtype=object),
    )
    identity = receipt.inspect_registered_source(
        path,
        expected_sha256=sha256_file(path),
        expected_rows=3,
    )
    return path, identity


def _supplement(
    tmp_path: Path, source: receipt.SourceIdentity, feature: np.ndarray
) -> receipt.SupplementIdentity:
    path = tmp_path / "blank.npz"
    with path.open("wb") as handle:
        np.savez(handle, **receipt.supplement_payload(source, MANIFEST, feature))
    return receipt.load_supplement(path)


def test_blank_pixels_are_exact_white_nhwc_uint8() -> None:
    value = receipt.blank_rgb()
    assert value.shape == (1, 224, 224, 3)
    assert value.dtype == np.uint8
    assert np.all(value == 255)


def test_source_inspection_opens_only_ordered_observation_ids(tmp_path: Path) -> None:
    path, identity = _source(tmp_path)
    assert identity.members_opened == ("observation_ids",)
    assert identity.rows == 3
    assert identity.observation_ids_sha256 == receipt.array_sha256(
        np.asarray(["cell-2", "cell-1", "cell-3"], dtype="S")
    )
    with pytest.raises(ValueError, match="not allowed"):
        receipt.read_source_member(path, "nucleus_molecular_targets")
    with pytest.raises(ValueError, match="not allowed"):
        receipt.read_source_member(path, "image_features")


def test_ordered_observation_hash_is_string_storage_invariant() -> None:
    unicode_ids = np.asarray(["cell-2", "cell-1", "cell-3"], dtype="U")
    byte_ids = np.asarray(["cell-2", "cell-1", "cell-3"], dtype="S")

    assert receipt.ordered_observation_ids_sha256(
        unicode_ids
    ) == receipt.ordered_observation_ids_sha256(byte_ids)


def test_compact_supplement_and_receipt_require_pixel_recomputation(tmp_path: Path) -> None:
    _, source = _source(tmp_path)
    one = _feature(1)
    many = _feature(32)
    supplement = _supplement(tmp_path, source, one)
    implementation = {
        "builder_sha256": "1" * 64,
        "receipt_validator_sha256": "2" * 64,
        "encoder_adapter_sha256": "3" * 64,
        "encoder_base_sha256": "4" * 64,
    }
    runtime = {
        "git_commit": "5" * 40,
        "git_worktree_dirty_at_start": False,
        "device": "cuda",
    }
    parity = receipt.batch_parity_receipt(one, many)
    value = receipt.build_receipt(
        source=source,
        supplement=supplement,
        manifest=MANIFEST,
        protocol_sha256="6" * 64,
        parity=parity,
        implementation=implementation,
        runtime=runtime,
    )
    receipt.validate_receipt(
        value,
        source=source,
        supplement=supplement,
        manifest=MANIFEST,
        protocol_sha256="6" * 64,
        expected_implementation=implementation,
        recomputed_batch_one=one,
        recomputed_batch_32=many,
    )
    assert supplement.feature.shape == (1, 1536)
    assert value["broadcast_rows"] == 3
    assert value["molecular_outcomes_read"] is False
    assert value["predictions_read"] is False

    altered = one.copy()
    altered[0, 0] += 1.0
    with pytest.raises(ValueError, match="did not reproduce"):
        receipt.validate_receipt(
            value,
            source=source,
            supplement=supplement,
            manifest=MANIFEST,
            protocol_sha256="6" * 64,
            expected_implementation=implementation,
            recomputed_batch_one=altered,
            recomputed_batch_32=many,
        )


def test_zero_and_non_float32_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        receipt.validate_feature_matrix(np.zeros((1, 1536), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="float32"):
        receipt.validate_feature_matrix(np.ones((1, 1536), dtype=np.float64), 1)


def test_batch_one_and_32_must_pass_frozen_parity() -> None:
    one = _feature(1)
    many = _feature(32)
    result = receipt.batch_parity_receipt(one, many)
    assert result["passed"] is True
    assert result["minimum_required_cosine"] == 0.9999
    opposite = -many
    with pytest.raises(ValueError, match="parity tolerance"):
        receipt.batch_parity_receipt(one, opposite)


def test_builder_constructs_both_batches_internally_and_has_no_feature_argument() -> None:
    class Encoder:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def encode(self, pixels: np.ndarray) -> np.ndarray:
            assert pixels.dtype == np.uint8
            assert np.all(pixels == 255)
            self.batch_sizes.append(len(pixels))
            return _feature(len(pixels))

    encoder = Encoder()
    one, many = BUILDER._encode_blank(encoder)
    assert encoder.batch_sizes == [1, 32]
    assert one.shape == (1, 1536)
    assert many.shape == (32, 1536)
    with pytest.raises(SystemExit):
        BUILDER._parser().parse_args(["--feature", "externally-supplied.npy"])


def test_default_and_indexed_cuda_devices_resolve_to_integer_indices() -> None:
    assert BUILDER._cuda_device_index("cuda", 3) == 3
    assert BUILDER._cuda_device_index("cuda:0", 3) == 0
    with pytest.raises(ValueError, match="indexed CUDA"):
        BUILDER._cuda_device_index("cpu", 0)


def test_frozen_protocol_and_encoder_identity_are_accepted() -> None:
    frozen = BUILDER._protocol_contract(ROOT / "configs/hest_nonlinear_qualification_v1.json")
    assert frozen["source_rows"] == 36121
    assert len(str(frozen["registered_source_sha256"])) == 64
    receipt.validate_encoder_contract(MANIFEST)


def test_receipt_schema_distinguishes_git_commit_from_sha256() -> None:
    path = ROOT / "configs/schemas/hest_blank_patch_supplement.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["properties"]["runtime"]["properties"]["git_commit"] == {
        "$ref": "#/$defs/gitCommit"
    }
    assert schema["$defs"]["gitCommit"]["pattern"] == "^[0-9a-f]{40}$"
