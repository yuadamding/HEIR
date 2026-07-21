from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from heir.evaluation import nonlinear_qualification_v2_data as data_module
from heir.evaluation.nonlinear_qualification_v2_contract import (
    BLANK_SUPPLEMENT_SHA256,
    SOURCE_SHA256,
    V2_PROTOCOL_SHA256,
)
from heir.evaluation.nonlinear_qualification_v2_execution import (
    BIOLOGICAL_AUTHORIZATION_SCHEMA,
)


def _source(path: Path) -> Path:
    donors: list[str] = []
    sections: list[str] = []
    labels: list[int] = []
    roles: list[str] = []
    for donor in ("D1", "D2"):
        for label in (0, 1):
            for role in ("reference", "evaluation"):
                for _ in range(2):
                    donors.append(donor)
                    sections.append(f"{donor}-S")
                    labels.append(label)
                    roles.append(role)
    rows = len(donors)
    targets = np.column_stack(
        (
            np.arange(rows, dtype=np.float64),
            np.asarray(labels, dtype=np.float64),
            np.asarray([role == "evaluation" for role in roles], dtype=np.float64),
        )
    )
    np.savez(
        path,
        observation_ids=np.asarray([f"cell-{index}" for index in range(rows)]),
        donor_ids=np.asarray(donors),
        section_ids=np.asarray(sections),
        type_labels=np.asarray(labels, dtype=np.int64),
        type_names=np.asarray(["T0", "T1"]),
        pool_roles=np.asarray(roles),
        block_ids=np.asarray([f"B{index % 2}" for index in range(rows)]),
        registration_quality_strata=np.full(rows, "best", dtype="<U16"),
        crop_ids=np.asarray(data_module.CROP_IDS),
        image_features_by_crop_and_encoder=np.arange(
            rows * len(data_module.CROP_IDS) * 4, dtype=np.float32
        ).reshape(rows, len(data_module.CROP_IDS), 4),
        technical_covariates=np.arange(rows, dtype=np.float32)[:, None],
        technical_covariate_names=np.asarray(["technical"]),
        coordinate_features=np.arange(rows, dtype=np.float32)[:, None],
        coordinate_feature_names=np.asarray(["x"]),
        stain_features=np.ones((rows, 1), dtype=np.float32),
        stain_feature_names=np.asarray(["stain"]),
        nuclear_morphometric_features=np.ones((rows, 1), dtype=np.float32) * 2,
        nuclear_morphometric_feature_names=np.asarray(["nucleus"]),
        cell_morphometric_features=np.ones((rows, 1), dtype=np.float32) * 3,
        cell_morphometric_feature_names=np.asarray(["cell"]),
        local_density_features=np.ones((rows, 1), dtype=np.float32) * 4,
        local_density_feature_names=np.asarray(["density"]),
        nucleus_molecular_targets=targets,
    )
    return path


def _authorization() -> dict[str, object]:
    return {
        "schema": BIOLOGICAL_AUTHORIZATION_SCHEMA,
        "registered_source_sha256": SOURCE_SHA256,
        "protocol_sha256": V2_PROTOCOL_SHA256,
        "device": "cpu",
        "implementation": {},
        "git_commit": "a" * 40,
        "biological_execution_authorized": True,
        "molecular_outcomes_read": False,
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_h_ref": False,
        "authorizes_full_heir": False,
    }


def _small_contract(monkeypatch) -> None:
    feature = np.ones((1, 4), dtype=np.float32)
    feature_sha256 = data_module.array_sha256(feature)
    observation_sha256 = data_module.ordered_observation_ids_sha256(
        np.asarray([f"cell-{index}" for index in range(16)])
    )
    monkeypatch.setattr(data_module, "SOURCE_ROWS", 16)
    monkeypatch.setattr(data_module, "REFERENCE_ROWS", 8)
    monkeypatch.setattr(data_module, "EVALUATION_ROWS", 8)
    monkeypatch.setattr(data_module, "TARGET_WIDTH", 3)
    monkeypatch.setattr(data_module, "IMAGE_WIDTH", 4)
    monkeypatch.setattr(data_module, "NUISANCE_WIDTH", 6)
    monkeypatch.setattr(data_module, "TYPE_COUNT", 2)
    monkeypatch.setattr(data_module, "PRIMARY_SUPPORT", 2)
    monkeypatch.setattr(data_module, "BLANK_FEATURE_SHA256", feature_sha256)
    monkeypatch.setattr(data_module, "file_sha256", lambda path: SOURCE_SHA256)
    monkeypatch.setattr(
        data_module,
        "load_supplement",
        lambda path: SimpleNamespace(
            sha256=BLANK_SUPPLEMENT_SHA256,
            registered_source_sha256=SOURCE_SHA256,
            ordered_observation_ids_sha256=observation_sha256,
            source_rows=16,
            broadcast_rows=16,
            feature=feature,
            feature_sha256=feature_sha256,
        ),
    )


def test_biological_loader_opens_the_molecular_member_exactly_once(
    tmp_path: Path, monkeypatch
) -> None:
    _small_contract(monkeypatch)
    source = _source(tmp_path / "source.npz")
    opened: list[str] = []
    original = np.lib.format.read_array

    def tracked(handle, *args, **kwargs):
        opened.append(str(getattr(handle, "name", "")))
        return original(handle, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "read_array", tracked)
    data = data_module.load_hest_biological_data(
        source,
        tmp_path / "blank.npz",
        authorization=_authorization(),
    )

    assert opened == [f"{name}.npy" for name in data_module.BIOLOGICAL_SOURCE_MEMBERS]
    assert opened.count("nucleus_molecular_targets.npy") == 1
    assert data.source_access["molecular_member_open_count"] == 1
    assert data.molecular_targets.shape == (8, 3)
    assert data.image_features.shape == (16, 4, 4)
    assert data.control_families["combined_nonimage"].shape == (8, 6)
    assert data.type_names == ("T0", "T1")


def test_biological_loader_fails_before_source_access_without_authorization(
    tmp_path: Path, monkeypatch
) -> None:
    _small_contract(monkeypatch)
    source = _source(tmp_path / "source.npz")
    opened = False

    def prohibited(self, name):
        nonlocal opened
        opened = True
        raise AssertionError(name)

    monkeypatch.setattr(data_module._BiologicalReader, "read", prohibited)
    with pytest.raises(ValueError, match="production authorization"):
        data_module.load_hest_biological_data(
            source,
            tmp_path / "blank.npz",
            authorization={},
        )
    assert opened is False


def test_biological_loader_checks_source_bytes_before_member_access(
    tmp_path: Path, monkeypatch
) -> None:
    _small_contract(monkeypatch)
    source = _source(tmp_path / "source.npz")
    monkeypatch.setattr(data_module, "file_sha256", lambda path: "0" * 64)
    opened = False

    def prohibited(self, name):
        nonlocal opened
        opened = True
        raise AssertionError(name)

    monkeypatch.setattr(data_module._BiologicalReader, "read", prohibited)
    with pytest.raises(ValueError, match="source bytes"):
        data_module.load_hest_biological_data(
            source,
            tmp_path / "blank.npz",
            authorization=_authorization(),
        )
    assert opened is False
