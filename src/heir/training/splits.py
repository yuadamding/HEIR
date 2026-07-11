"""Group-first split assignment for donors, blocks, and serial sections."""

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ..data import HistologyBag


def grouped_fold_assignment(
    donor_ids: Sequence[object],
    num_folds: int,
    strata: Optional[Sequence[object]] = None,
    seed: int = 17,
) -> Dict[str, int]:
    """Assign every donor once, approximately balancing donor-level strata."""

    donors = np.asarray([str(value) for value in donor_ids], dtype=np.str_)
    if donors.size == 0 or num_folds < 2:
        raise ValueError("at least one donor and two folds are required")
    if num_folds > len(np.unique(donors)):
        raise ValueError("num_folds cannot exceed the number of donors")
    if strata is None:
        donor_stratum = {donor: "all" for donor in np.unique(donors)}
    else:
        stratum_values = np.asarray([str(value) for value in strata], dtype=np.str_)
        if stratum_values.shape != donors.shape:
            raise ValueError("strata must align to donor_ids")
        donor_stratum: Dict[str, str] = {}
        for donor in np.unique(donors):
            observed = np.unique(stratum_values[donors == donor])
            if len(observed) != 1:
                raise ValueError("each donor must belong to one stratum")
            donor_stratum[donor] = str(observed[0])
    rng = np.random.default_rng(seed)
    by_stratum: MutableMapping[str, List[str]] = defaultdict(list)
    for donor, stratum in donor_stratum.items():
        by_stratum[stratum].append(donor)
    folds: List[List[str]] = [[] for _ in range(num_folds)]
    for stratum in sorted(by_stratum):
        values = np.asarray(sorted(by_stratum[stratum]), dtype=object)
        rng.shuffle(values)
        offset = min(range(num_folds), key=lambda index: len(folds[index]))
        for index, donor in enumerate(values.tolist()):
            folds[(offset + index) % num_folds].append(str(donor))
    return {donor: fold for fold, values in enumerate(folds) for donor in values}


def validate_grouped_splits(records: Iterable[Mapping[str, object]]) -> None:
    """Fail on any donor, block, or specimen crossing an outer analysis role."""

    entries = list(records)
    if not entries:
        raise ValueError("split records cannot be empty")
    for required in ("donor_id", "block_id", "specimen_id", "analysis_role"):
        if any(required not in record for record in entries):
            raise ValueError("split records require %s" % required)
    for group_key in ("donor_id", "block_id", "specimen_id"):
        roles: MutableMapping[str, set] = defaultdict(set)
        for record in entries:
            value = str(record[group_key]).strip()
            if value:
                roles[value].add(str(record["analysis_role"]).strip().lower())
        for value, observed in roles.items():
            collapsed = {_role_family(role) for role in observed}
            if len(collapsed) > 1:
                raise ValueError(
                    "%s %s crosses analysis splits: %s" % (group_key, value, sorted(observed))
                )


def spatial_block_split_masks(
    coordinates_um: np.ndarray,
    *,
    validation_fraction: float = 0.2,
    block_size_um: float = 512.0,
    seed: int = 17,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create disjoint train/validation masks by whole spatial blocks.

    Splitting entire blocks prevents neighboring nuclei from being divided by
    a random cell split.  Blocks are selected deterministically until the
    requested validation cell fraction is reached.
    """

    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("coordinates_um must contain at least two x/y rows")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates_um must be finite")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    if not np.isfinite(block_size_um) or block_size_um <= 0:
        raise ValueError("block_size_um must be finite and positive")
    origin = coordinates.min(axis=0, keepdims=True)
    block_xy = np.floor((coordinates - origin) / block_size_um).astype(np.int64)
    unique_blocks, inverse, counts = np.unique(
        block_xy,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if len(unique_blocks) < 2:
        raise ValueError("spatial split needs at least two occupied blocks")
    order = np.arange(len(unique_blocks), dtype=np.int64)
    np.random.default_rng(seed).shuffle(order)
    target = max(1, int(round(validation_fraction * len(coordinates))))
    selected_blocks = []
    selected_cells = 0
    for block_index in order[:-1]:
        selected_blocks.append(int(block_index))
        selected_cells += int(counts[block_index])
        if selected_cells >= target:
            break
    validation = np.isin(inverse, np.asarray(selected_blocks, dtype=np.int64))
    training = ~validation
    if not training.any() or not validation.any():
        raise RuntimeError("spatial block split produced an empty partition")
    return training, validation


def subset_histology_bag(bag: HistologyBag, mask: np.ndarray) -> HistologyBag:
    """Subset a histology graph while retaining only and reindexing internal edges."""

    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (bag.n_nuclei,) or not selected.any():
        raise ValueError("mask must select at least one HistologyBag nucleus")
    indices = np.flatnonzero(selected)
    remap = np.full(bag.n_nuclei, -1, dtype=np.int64)
    remap[indices] = np.arange(len(indices), dtype=np.int64)
    edges = np.asarray(bag.edge_index, dtype=np.int64)
    retained_edges = selected[edges[0]] & selected[edges[1]]
    subset_edges = remap[edges[:, retained_edges]]
    return HistologyBag(
        slide_id=bag.slide_id,
        nucleus_ids=bag.nucleus_ids[indices],
        features=bag.features[indices],
        coordinates_um=bag.coordinates_um[indices],
        morphology=bag.morphology[indices],
        segmentation_confidence=bag.segmentation_confidence[indices],
        artifact_probability=bag.artifact_probability[indices],
        edge_index=subset_edges,
        edge_weight=bag.edge_weight[retained_edges],
        sample_id=bag.sample_id,
        donor_id=bag.donor_id,
        block_id=bag.block_id,
        feature_space_id=bag.feature_space_id,
        histology_source_sha256=bag.histology_source_sha256,
        nuclei_source_sha256=bag.nuclei_source_sha256,
        feature_source_sha256=bag.feature_source_sha256,
    )


def _role_family(role: str) -> str:
    if role in {"train", "inner_train"}:
        return "train"
    if role in {"development", "calibration", "inner_validation", "validation"}:
        return "development"
    if role in {"locked_test", "test", "external_test"}:
        return "test"
    return role
