from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from heir.evaluation import registration_topology as topology
from heir.features.base import sha256_file


def _source(
    path: Path,
    *,
    donors: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    roles: Optional[np.ndarray] = None,
) -> Path:
    rows = len(donors)
    if roles is None:
        roles = np.full(rows, "evaluation", dtype="<U16")
    scores = np.choose(
        np.asarray([topology.SOURCE_STRATA.index(value) for value in strata]),
        [0.2, 0.6, 0.9, 1.2],
    )
    np.savez(
        path,
        observation_ids=np.asarray([f"cell-{index}" for index in range(rows)]),
        donor_ids=donors,
        section_ids=np.asarray([f"{donor}-S" for donor in donors]),
        type_labels=labels.astype(np.int64),
        pool_roles=roles,
        registration_quality_scores=scores,
        registration_quality_strata=strata,
        nucleus_molecular_targets=np.arange(rows, dtype=np.float32),
        image_features=np.arange(rows, dtype=np.float32),
    )
    return path


def _balanced_arrays(
    *,
    best_per_donor: int,
    intermediate_per_donor: int = 0,
    near_per_donor: int = 0,
    failed_per_donor: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    donor_values: list[str] = []
    label_values: list[int] = []
    stratum_values: list[str] = []
    for donor_index in range(12):
        donor = f"D{donor_index + 1:02d}"
        for stratum, count in (
            ("best", best_per_donor),
            ("intermediate", intermediate_per_donor),
            ("near_threshold", near_per_donor),
            ("failed", failed_per_donor),
        ):
            donor_values.extend([donor] * count)
            label_values.extend(index % 2 for index in range(count))
            stratum_values.extend([stratum] * count)
    return (
        np.asarray(donor_values),
        np.asarray(label_values),
        np.asarray(stratum_values),
    )


def test_reader_opens_only_frozen_allowlist(tmp_path: Path, monkeypatch) -> None:
    donors, labels, strata = _balanced_arrays(best_per_donor=42)
    path = _source(tmp_path / "source.npz", donors=donors, labels=labels, strata=strata)
    opened: list[str] = []
    original = topology.np.lib.format.read_array

    def tracked(handle, *args, **kwargs):
        opened.append(str(getattr(handle, "name", "")))
        return original(handle, *args, **kwargs)

    monkeypatch.setattr(topology.np.lib.format, "read_array", tracked)
    audit = topology.audit_topology(
        path,
        expected_sha256=sha256_file(path),
        expected_rows=len(donors),
    )
    assert opened == [f"{name}.npy" for name in topology.SOURCE_MEMBER_ALLOWLIST]
    assert audit["source_members_opened"] == list(topology.SOURCE_MEMBER_ALLOWLIST)
    assert audit["molecular_outcomes_read"] is False
    assert audit["image_features_read"] is False


def test_reader_rejects_molecular_and_image_members(tmp_path: Path) -> None:
    donors, labels, strata = _balanced_arrays(best_per_donor=42)
    path = _source(tmp_path / "source.npz", donors=donors, labels=labels, strata=strata)
    reader = topology.DirectNpzReader(path)
    for name in ("nucleus_molecular_targets", "image_features", "reference_expression"):
        with pytest.raises(ValueError, match="outside the topology allowlist"):
            reader.read(name)


def test_q1_is_selected_when_best_is_narrowest_eligible_band(tmp_path: Path) -> None:
    donors, labels, strata = _balanced_arrays(best_per_donor=42)
    path = _source(tmp_path / "source.npz", donors=donors, labels=labels, strata=strata)
    audit = topology.audit_topology(
        path,
        expected_sha256=sha256_file(path),
        expected_rows=len(donors),
    )
    assert audit["bands"]["best"]["rows"] == 504
    assert audit["bands"]["best"]["selection_eligible"] is True
    assert audit["selected_band"] == {
        "quality_band_id": "Q1",
        "band_id": "best",
        "source_strata": ["best"],
        "included_donor_ids": [f"D{index:02d}" for index in range(1, 13)],
        "selection_rows": 504,
    }


def test_q2_is_selected_without_lowering_any_numeric_cutoff(tmp_path: Path) -> None:
    donors, labels, strata = _balanced_arrays(best_per_donor=20, intermediate_per_donor=22)
    path = _source(tmp_path / "source.npz", donors=donors, labels=labels, strata=strata)
    audit = topology.audit_topology(
        path,
        expected_sha256=sha256_file(path),
        expected_rows=len(donors),
    )
    assert audit["bands"]["best"]["selection_eligible"] is False
    assert audit["bands"]["best_or_intermediate"]["rows"] == 504
    assert audit["selected_band"]["quality_band_id"] == "Q2"
    assert audit["selected_band"]["source_strata"] == ["best", "intermediate"]


def test_no_band_passes_when_one_donor_exceeds_twenty_percent(tmp_path: Path) -> None:
    donors, labels, strata = _balanced_arrays(best_per_donor=20, intermediate_per_donor=22)
    extra = 400
    donors = np.concatenate((donors, np.full(extra, "D01")))
    labels = np.concatenate((labels, np.arange(extra) % 2))
    strata = np.concatenate((strata, np.full(extra, "intermediate")))
    path = _source(tmp_path / "source.npz", donors=donors, labels=labels, strata=strata)
    audit = topology.audit_topology(
        path,
        expected_sha256=sha256_file(path),
        expected_rows=len(donors),
    )
    assert (
        audit["bands"]["best_or_intermediate_or_near_threshold"]["selection_criteria"][
            "maximum_donor_fraction"
        ]["passed"]
        is False
    )
    assert audit["selected_band"] is None
    assert audit["selection_status"] == "no_frozen_cumulative_band_eligible"


def test_low_support_donor_is_excluded_before_twelve_donor_rule(tmp_path: Path) -> None:
    donors, labels, strata = _balanced_arrays(best_per_donor=42)
    donors = np.concatenate((donors, np.asarray(["D13"])))
    labels = np.concatenate((labels, np.asarray([0])))
    strata = np.concatenate((strata, np.asarray(["best"])))
    path = _source(tmp_path / "source.npz", donors=donors, labels=labels, strata=strata)
    audit = topology.audit_topology(
        path,
        expected_sha256=sha256_file(path),
        expected_rows=len(donors),
    )
    best = audit["bands"]["best"]
    assert best["donors"] == 13
    assert best["included_donor_ids"] == [f"D{index:02d}" for index in range(1, 13)]
    assert best["excluded_donors"]["D13"]["below_minimum_rows"] is True
    assert best["selection_eligible"] is True


def test_support_matrices_report_five_ten_and_twenty(tmp_path: Path) -> None:
    donors, labels, strata = _balanced_arrays(best_per_donor=42)
    path = _source(tmp_path / "source.npz", donors=donors, labels=labels, strata=strata)
    audit = topology.audit_topology(
        path,
        expected_sha256=sha256_file(path),
        expected_rows=len(donors),
    )
    best = audit["bands"]["best"]
    assert set(best["donor_type_support"]) == {"5", "10", "20"}
    assert set(best["donor_section_type_support"]) == {"5", "10", "20"}
    assert best["donor_type_support"]["20"]["supported_strata"] == 24
    assert best["donor_section_type_support"]["20"]["supported_strata"] == 24


def test_score_and_stratum_inconsistency_fails_closed(tmp_path: Path) -> None:
    donors, labels, strata = _balanced_arrays(best_per_donor=42)
    path = _source(tmp_path / "source.npz", donors=donors, labels=labels, strata=strata)
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    values["registration_quality_scores"][0] = 1.1
    np.savez(path, **values)
    with pytest.raises(ValueError, match="scores and frozen strata"):
        topology.audit_topology(
            path,
            expected_sha256=sha256_file(path),
            expected_rows=len(donors),
        )


def test_receipt_preserves_outcome_blind_stopping_state(tmp_path: Path) -> None:
    donors, labels, strata = _balanced_arrays(best_per_donor=1)
    path = _source(tmp_path / "source.npz", donors=donors, labels=labels, strata=strata)
    audit = topology.audit_topology(
        path,
        expected_sha256=sha256_file(path),
        expected_rows=len(donors),
    )
    value = topology.build_receipt(
        audit,
        implementation={"audit_script_sha256": "a" * 64},
        runtime={"git_commit": "b" * 40, "git_worktree_dirty_at_start": False},
    )
    assert value["selected_band"] is None
    assert value["outcome_blind_band_selection"] is True
    assert value["biological_experiment_run"] is False
    assert value["engineering_decision_available"] is False
