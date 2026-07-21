from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from heir.evaluation import blank_patch_receipt as receipt
from heir.features import load_encoder_manifest

ROOT = Path(__file__).parents[1]
MANIFEST = load_encoder_manifest(ROOT / "manifests/encoders/hoptimus1.json")
IMPLEMENTATION = {
    "builder_sha256": "b" * 64,
    "receipt_validator_sha256": "c" * 64,
    "encoder_adapter_sha256": "d" * 64,
    "encoder_base_sha256": "e" * 64,
}
RUNTIME = {"git_commit": "f" * 40, "git_worktree_dirty_at_start": False}


def _source(tmp_path: Path, rows: int = 3) -> tuple[Path, receipt.SourceIdentity]:
    path = tmp_path / "source.npz"
    np.savez(
        path,
        observation_ids=np.asarray([f"cell-{index}" for index in range(rows)]),
        nucleus_molecular_targets=np.arange(rows, dtype=np.float32),
    )
    identity = receipt.inspect_registered_source(
        path,
        expected_sha256=receipt.sha256_file(path),
        expected_rows=rows,
    )
    return path, identity


def _feature(rows: int) -> np.ndarray:
    base = np.linspace(0.1, 1.0, receipt.FEATURE_WIDTH, dtype=np.float32)
    return np.tile(base, (rows, 1))


def _write_supplement(
    tmp_path: Path, source: receipt.SourceIdentity, feature: np.ndarray
) -> receipt.SupplementIdentity:
    path = tmp_path / "blank.npz"
    np.savez(path, **receipt.supplement_payload(source, MANIFEST, feature))
    return receipt.load_supplement(path)


def _receipt(
    source: receipt.SourceIdentity, supplement: receipt.SupplementIdentity
) -> dict[str, object]:
    return dict(
        receipt.build_receipt(
            source=source,
            supplement=supplement,
            manifest=MANIFEST,
            protocol_sha256="a" * 64,
            parity=receipt.batch_parity_receipt(_feature(1), _feature(32)),
            implementation=IMPLEMENTATION,
            runtime=RUNTIME,
        )
    )


def _validate(
    value: dict[str, object],
    source: receipt.SourceIdentity,
    supplement: receipt.SupplementIdentity,
) -> None:
    receipt.validate_receipt(
        value,
        source=source,
        supplement=supplement,
        manifest=MANIFEST,
        protocol_sha256="a" * 64,
        expected_implementation=IMPLEMENTATION,
        recomputed_batch_one=_feature(1),
        recomputed_batch_32=_feature(32),
    )


def test_blank_pixels_are_exact_registered_white_nhwc_uint8() -> None:
    blank = receipt.blank_rgb()
    assert blank.shape == (1, 224, 224, 3)
    assert blank.dtype == np.uint8
    assert np.array_equal(blank, np.full((1, 224, 224, 3), 255, dtype=np.uint8))


def test_source_identity_reads_only_observation_ids(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "source.npz"
    np.savez(
        path,
        observation_ids=np.asarray(["a", "b"]),
        nucleus_molecular_targets=np.asarray([99.0, 100.0]),
    )
    opened: list[str] = []
    original = receipt.np.lib.format.read_array

    def tracked(handle, *args, **kwargs):
        opened.append(str(getattr(handle, "name", "")))
        return original(handle, *args, **kwargs)

    monkeypatch.setattr(receipt.np.lib.format, "read_array", tracked)
    value = receipt.inspect_registered_source(
        path,
        expected_sha256=receipt.sha256_file(path),
        expected_rows=2,
    )
    assert value.members_opened == ("observation_ids",)
    assert opened == ["observation_ids.npy"]


def test_direct_source_reader_rejects_non_allowlisted_member(tmp_path: Path) -> None:
    path, _ = _source(tmp_path)
    with pytest.raises(ValueError, match="not allowed"):
        receipt.read_source_member(path, "nucleus_molecular_targets")


def test_receipt_recomputes_registered_feature_hash(tmp_path: Path) -> None:
    _, source = _source(tmp_path)
    supplement = _write_supplement(tmp_path, source, _feature(1))
    _validate(_receipt(source, supplement), source, supplement)


@pytest.mark.parametrize("tamper", ["zero", "source", "feature", "dirty", "external"])
def test_receipt_rejects_forgery(tmp_path: Path, tamper: str) -> None:
    _, source = _source(tmp_path)
    feature = (
        np.zeros((1, receipt.FEATURE_WIDTH), dtype=np.float32) if tamper == "zero" else _feature(1)
    )
    if tamper == "zero":
        with pytest.raises(ValueError, match="zero vector"):
            _write_supplement(tmp_path, source, feature)
        return
    supplement = _write_supplement(tmp_path, source, feature)
    value = _receipt(source, supplement)
    if tamper == "source":
        value["registered_source_sha256"] = "0" * 64
    elif tamper == "feature":
        value["feature_sha256"] = "0" * 64
    elif tamper == "dirty":
        value["runtime"]["git_worktree_dirty_at_start"] = True
    else:
        value["externally_supplied_feature"] = True
    with pytest.raises(ValueError):
        _validate(value, source, supplement)


def test_receipt_schema_is_strict_and_documents_outcome_blinding() -> None:
    schema = json.loads(
        (ROOT / "configs/schemas/hest_blank_patch_supplement.schema.json").read_text()
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["molecular_outcomes_read"] == {"const": False}
    assert schema["properties"]["predictions_read"] == {"const": False}
    assert schema["properties"]["source_rows"] == {"const": 36121}
    assert schema["properties"]["runtime"]["properties"]["git_commit"]["pattern"] == (
        "^[0-9a-f]{40}$"
    )
