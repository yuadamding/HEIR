from __future__ import annotations

from pathlib import Path

import numpy as np

from heir.evaluation import nonlinear_qualification_v2_execution as execution
from heir.evaluation.registration_topology import SOURCE_MEMBER_ALLOWLIST


def _source(path: Path) -> tuple[Path, tuple[str, ...], int]:
    donors = tuple(f"D{index:02d}" for index in range(12))
    donor_ids = np.repeat(donors, 40)
    rows = len(donor_ids)
    within_donor = np.tile(np.arange(40), len(donors))
    quality = within_donor < 20
    strata = np.where(quality, "best", "failed")
    scores = np.where(quality, 0.2, 1.2)
    np.savez(
        path,
        observation_ids=np.asarray([f"cell-{index}" for index in range(rows)]),
        donor_ids=donor_ids,
        section_ids=np.asarray([f"{donor}-S" for donor in donor_ids]),
        type_labels=np.tile(np.repeat([0, 1], 20), len(donors)),
        pool_roles=np.full(rows, "evaluation", dtype="<U16"),
        registration_quality_scores=scores,
        registration_quality_strata=strata,
        nucleus_molecular_targets=np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
        image_features=np.arange(rows, dtype=np.float32),
    )
    return path, donors, int(quality.sum())


def test_calibration_topology_opens_only_the_seven_outcome_blind_members(
    tmp_path: Path, monkeypatch
) -> None:
    source, donors, formal_rows = _source(tmp_path / "source.npz")
    monkeypatch.setattr(execution, "SOURCE_ROWS", 480)
    monkeypatch.setattr(execution, "EVALUATION_ROWS", 480)
    monkeypatch.setattr(execution, "FORMAL_Q3_ROWS", formal_rows)
    monkeypatch.setattr(execution, "SELECTED_DONORS", donors)
    opened: list[str] = []
    original = np.lib.format.read_array

    def tracked(handle, *args, **kwargs):
        opened.append(str(getattr(handle, "name", "")))
        return original(handle, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "read_array", tracked)
    topology, receipt = execution.load_calibration_topology(source)

    assert topology.rows == 480
    assert opened == [f"{name}.npy" for name in SOURCE_MEMBER_ALLOWLIST]
    assert "nucleus_molecular_targets.npy" not in opened
    assert receipt["source_members_opened"] == list(SOURCE_MEMBER_ALLOWLIST)
    assert receipt["molecular_outcomes_read"] is False
    assert receipt["image_features_read"] is False


def test_source_registry_lists_the_molecular_member_without_opening_it(
    tmp_path: Path, monkeypatch
) -> None:
    source, _, _ = _source(tmp_path / "source.npz")

    def prohibited(*args, **kwargs):
        raise AssertionError("preflight must not deserialize any source payload")

    monkeypatch.setattr(np.lib.format, "read_array", prohibited)
    members = execution._source_member_registry(source)

    assert members.count(execution.MOLECULAR_MEMBER) == 1
    assert "image_features" in members
