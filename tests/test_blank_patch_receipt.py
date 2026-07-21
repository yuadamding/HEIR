from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from heir.evaluation import blank_patch_receipt as receipt
from heir.features import load_encoder_manifest

ROOT = Path(__file__).parents[1]
MANIFEST = load_encoder_manifest(ROOT / "manifests/encoders/hoptimus1.json")


def _source(tmp_path: Path, rows: int = 3) -> tuple[Path, dict[str, object]]:
    path = tmp_path / "source.npz"
    np.savez(
        path,
        observation_ids=np.asarray([f"cell-{index}" for index in range(rows)]),
        nucleus_molecular_targets=np.arange(rows, dtype=np.float32),
    )
    identity = receipt.source_identity(
        path,
        expected_sha256=receipt.sha256_file(path),
        expected_rows=rows,
    )
    return path, dict(identity)


def _feature(rows: int) -> np.ndarray:
    return np.tile(np.linspace(0.1, 1.0, receipt.FEATURE_WIDTH, dtype=np.float32), (rows, 1))


def _write_supplement(
    tmp_path: Path, source: dict[str, object], feature: np.ndarray
) -> dict[str, object]:
    path = tmp_path / "blank.npz"
    np.savez(path, **receipt.supplement_arrays(feature, source=source, manifest=MANIFEST))
    return dict(receipt.load_supplement(path))


def _receipt(source: dict[str, object], supplement: dict[str, object]) -> dict[str, object]:
    parity = receipt.batch_parity_receipt(_feature(1), _feature(32))
    return dict(
        receipt.make_receipt(
            source=source,
            supplement=supplement,
            manifest=MANIFEST,
            protocol_sha256="a" * 64,
            parity=parity,
            implementation={
                "builder_sha256": "b" * 64,
                "receipt_validator_sha256": "c" * 64,
                "encoder_adapter_sha256": "d" * 64,
                "encoder_base_sha256": "e" * 64,
            },
            runtime={
                "git_commit": "f" * 40,
                "git_worktree_dirty_at_start": False,
            },
        )
    )


def test_blank_pixels_are_exact_registered_white_nhwc_uint8() -> None:
    blank = receipt.blank_pixels()
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
    value = receipt.source_identity(
        path,
        expected_sha256=receipt.sha256_file(path),
        expected_rows=2,
    )
    assert value["members_read"] == ["observation_ids"]
    assert value["molecular_outcomes_read"] is False
    assert len(opened) == 1


def test_direct_source_reader_rejects_non_allowlisted_member(tmp_path: Path) -> None:
    path, _ = _source(tmp_path)
    with pytest.raises(ValueError, match="outside the outcome-blind allowlist"):
        receipt._direct_npz_member(
            path,
            "nucleus_molecular_targets",
            allowlist=receipt.SOURCE_MEMBER_ALLOWLIST,
        )


def test_receipt_recomputes_registered_feature_hash(tmp_path: Path) -> None:
    _, source = _source(tmp_path)
    supplement = _write_supplement(tmp_path, source, _feature(1))
    value = _receipt(source, supplement)
    validation = receipt.validate_receipt(
        value,
        source=source,
        supplement=supplement,
        manifest=MANIFEST,
        protocol_sha256="a" * 64,
        recomputed_single=_feature(1),
        recomputed_repeated=_feature(32),
    )
    assert validation["blank_patch_embedding_available"] is True
    assert validation["feature_recomputed_from_registered_pixels"] is True
    assert validation["molecular_outcomes_read"] is False


@pytest.mark.parametrize("tamper", ["zero", "source", "feature", "dirty"])
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
    else:
        value["runtime"]["git_worktree_dirty_at_start"] = True
    with pytest.raises(ValueError):
        receipt.validate_receipt(
            value,
            source=source,
            supplement=supplement,
            manifest=MANIFEST,
            protocol_sha256="a" * 64,
        )


def test_receipt_schema_is_strict_and_documents_all_required_fields() -> None:
    schema = json.loads(
        (ROOT / "configs/schemas/hest_blank_patch_supplement.schema.json").read_text()
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["molecular_outcomes_read"] == {"const": False}
    assert schema["properties"]["predictions_read"] == {"const": False}
    assert schema["properties"]["source_rows"] == {"const": 36121}
