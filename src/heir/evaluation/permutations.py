"""Registration-preserving permutation families and null-activity audits."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def donor_type_roi_permutation(
    donor_ids: np.ndarray,
    type_labels: np.ndarray,
    roi_ids: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Derange rows only within donor/type/ROI strata."""

    donors = np.asarray(donor_ids).astype(str)
    labels = np.asarray(type_labels)
    rois = np.asarray(roi_ids).astype(str)
    if donors.ndim != 1 or labels.shape != donors.shape or rois.shape != donors.shape:
        raise ValueError("permutation identities must be aligned vectors")
    rng = np.random.default_rng(seed)
    result = np.arange(len(donors), dtype=np.int64)
    keys = np.column_stack((donors, labels.astype(str), rois))
    for key in sorted(set(map(tuple, keys.tolist()))):
        group = np.flatnonzero(np.all(keys == np.asarray(key), axis=1))
        if len(group) < 2:
            continue
        ordered = group[rng.permutation(len(group))]
        result[ordered] = np.roll(ordered, 1)
    if not (
        np.array_equal(donors, donors[result])
        and np.array_equal(labels, labels[result])
        and np.array_equal(rois, rois[result])
    ):
        raise RuntimeError("preserving permutation crossed a frozen stratum")
    return result


def donor_type_block_permutation(
    donor_ids: np.ndarray,
    type_labels: np.ndarray,
    block_ids: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Reassign rows across spatial blocks while preserving donor and fine type."""

    donors = np.asarray(donor_ids).astype(str)
    labels = np.asarray(type_labels)
    blocks = np.asarray(block_ids).astype(str)
    if donors.ndim != 1 or labels.shape != donors.shape or blocks.shape != donors.shape:
        raise ValueError("block-permutation identities must be aligned vectors")
    rng = np.random.default_rng(seed)
    result = np.arange(len(donors), dtype=np.int64)
    keys = np.column_stack((donors, labels.astype(str)))
    for key in sorted(set(map(tuple, keys.tolist()))):
        group = np.flatnonzero(np.all(keys == np.asarray(key), axis=1))
        if len(group) < 2:
            continue
        best = group.copy()
        best_score = (-1, -1)
        order = np.argsort(blocks[group], kind="stable")
        ordered = group[order]
        _, block_counts = np.unique(blocks[ordered], return_counts=True)
        for shift in np.cumsum(block_counts)[:-1].tolist():
            candidate = np.empty_like(group)
            candidate[order] = np.roll(ordered, -int(shift))
            score = (
                int(np.count_nonzero(blocks[group] != blocks[candidate])),
                int(np.count_nonzero(group != candidate)),
            )
            if score > best_score or (score == best_score and rng.random() < 0.5):
                best = candidate
                best_score = score
        for _ in range(128):
            candidate = group[rng.permutation(len(group))]
            score = (
                int(np.count_nonzero(blocks[group] != blocks[candidate])),
                int(np.count_nonzero(group != candidate)),
            )
            if score == (len(group), len(group)):
                best = candidate
                best_score = score
                break
            if score > best_score or (score == best_score and rng.random() < 0.5):
                best = candidate
                best_score = score
        result[group] = best
    if not (
        np.array_equal(donors, donors[result]) and np.array_equal(labels, labels[result])
    ):
        raise RuntimeError("block permutation crossed a donor/type stratum")
    return result


def null_stratum_activity(
    donor_ids: np.ndarray, type_labels: np.ndarray, stratum_ids: np.ndarray
) -> Mapping[str, object]:
    donors = np.asarray(donor_ids).astype(str)
    labels = np.asarray(type_labels).astype(str)
    strata = np.asarray(stratum_ids).astype(str)
    keys = np.column_stack((donors, labels, strata))
    sizes = np.asarray(
        [
            np.count_nonzero(np.all(keys == np.asarray(key), axis=1))
            for key in sorted(set(map(tuple, keys.tolist())))
        ],
        dtype=np.int64,
    )
    eligible_rows = int(sizes[sizes >= 2].sum())
    return {
        "total_strata": int(len(sizes)),
        "active_strata": int(np.count_nonzero(sizes >= 2)),
        "singleton_strata": int(np.count_nonzero(sizes == 1)),
        "eligible_rows": eligible_rows,
        "eligible_row_fraction": float(eligible_rows / max(len(donors), 1)),
        "roi_size_distribution": {
            "minimum": int(sizes.min()) if len(sizes) else 0,
            "median": float(np.median(sizes)) if len(sizes) else 0.0,
            "maximum": int(sizes.max()) if len(sizes) else 0,
        },
    }


def block_null_activity(
    donor_ids: np.ndarray, type_labels: np.ndarray, block_ids: np.ndarray
) -> Mapping[str, object]:
    donors = np.asarray(donor_ids).astype(str)
    labels = np.asarray(type_labels).astype(str)
    blocks = np.asarray(block_ids).astype(str)
    keys = np.column_stack((donors, labels))
    groups = sorted(set(map(tuple, keys.tolist())))
    active_rows = 0
    active_groups = 0
    for key in groups:
        selected = np.all(keys == np.asarray(key), axis=1)
        if len(set(blocks[selected].tolist())) >= 2:
            active_groups += 1
            active_rows += int(selected.sum())
    return {
        "total_donor_type_groups": len(groups),
        "active_donor_type_groups": active_groups,
        "inactive_single_block_groups": len(groups) - active_groups,
        "eligible_rows": active_rows,
        "eligible_row_fraction": float(active_rows / max(len(donors), 1)),
    }


__all__ = [
    "block_null_activity",
    "donor_type_block_permutation",
    "donor_type_roi_permutation",
    "null_stratum_activity",
]
