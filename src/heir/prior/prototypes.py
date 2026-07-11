"""Build interpretable sample/type RNA-state prototypes.

HEIR aligns image nuclei to these compact prototypes rather than constructing
an intractable image-cell by RNA-cell transport matrix. Covariances are stored
diagonally by default, which is stable for small cell states and cheap enough
for whole-slide minibatches.
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.cluster import KMeans

from ..data.arrays import PrototypeSet

ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_text(values: Sequence[object]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=np.str_)


def _atlas_type_means(
    atlas_latents: Optional[np.ndarray],
    atlas_labels: Optional[Sequence[object]],
    type_names: Sequence[str],
    fallback_latents: np.ndarray,
    fallback_labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    if atlas_latents is not None or atlas_labels is not None:
        if atlas_latents is None or atlas_labels is None:
            raise ValueError("atlas_latents and atlas_labels must be supplied together")
        atlas = np.asarray(atlas_latents, dtype=np.float64)
        labels = _as_text(atlas_labels)
        if atlas.ndim != 2 or atlas.shape[0] != labels.shape[0]:
            raise ValueError("atlas latents and labels are misaligned")
    else:
        atlas = fallback_latents
        labels = fallback_labels
    for name in type_names:
        mask = labels == name
        if np.any(mask):
            result[name] = atlas[mask].mean(axis=0)
    return result


def _merge_small_clusters(assignments: np.ndarray, centers: np.ndarray, minimum: int) -> np.ndarray:
    """Merge clusters below ``minimum`` into the nearest stable cluster."""

    labels = assignments.copy()
    while True:
        unique, counts = np.unique(labels, return_counts=True)
        small = unique[counts < minimum]
        stable = unique[counts >= minimum]
        if small.size == 0 or stable.size == 0:
            break
        changed = False
        for cluster in small:
            distances = np.square(centers[stable] - centers[cluster]).sum(axis=1)
            labels[labels == cluster] = stable[int(np.argmin(distances))]
            changed = True
        if not changed:
            break
    remap = {int(value): index for index, value in enumerate(np.unique(labels))}
    return np.asarray([remap[int(value)] for value in labels], dtype=np.int64)


def build_sample_prototypes(
    latents: np.ndarray,
    labels: Sequence[object],
    sample_id: str,
    max_prototypes_per_type: int = 10,
    minimum_cells: int = 50,
    shrinkage_kappa: float = 50.0,
    atlas_latents: Optional[np.ndarray] = None,
    atlas_labels: Optional[Sequence[object]] = None,
    variance_floor: float = 1.0e-3,
    seed: int = 17,
    include_rare_types: bool = False,
    latent_space_id: str = "",
    donor_id: str = "",
    block_id: str = "",
    source_reference_sha256: str = "",
    latent_training_donors: Sequence[str] = (),
    latent_transform_sha256: str = "",
) -> PrototypeSet:
    """Cluster each cell type and shrink unstable means toward an atlas.

    The number of states grows only when enough cells are available: one state
    per ``minimum_cells`` observations, capped by ``max_prototypes_per_type``.
    Types below ``minimum_cells`` are excluded from the sample-supported bank
    by default, so the image model must abstain rather than hallucinate them.
    Set ``include_rare_types`` only for an explicit sensitivity analysis.
    """

    latent = np.asarray(latents, dtype=np.float64)
    label_array = _as_text(labels)
    if latent.ndim != 2 or latent.shape[0] != label_array.shape[0] or latent.shape[0] == 0:
        raise ValueError("latents and labels must contain aligned, non-empty rows")
    if not np.isfinite(latent).all():
        raise ValueError("latents must be finite")
    if max_prototypes_per_type <= 0 or minimum_cells <= 0:
        raise ValueError("prototype limits must be positive")
    if shrinkage_kappa < 0 or variance_floor <= 0:
        raise ValueError("shrinkage_kappa must be non-negative and variance_floor positive")
    if not sample_id:
        raise ValueError("sample_id cannot be empty")

    type_names = sorted(np.unique(label_array).tolist())
    atlas_means = _atlas_type_means(
        atlas_latents,
        atlas_labels,
        type_names,
        latent,
        label_array,
    )
    means: List[np.ndarray] = []
    variances: List[np.ndarray] = []
    counts: List[int] = []
    type_indices: List[int] = []
    identifiers: List[str] = []

    for type_index, type_name in enumerate(type_names):
        values = latent[label_array == type_name]
        if values.shape[0] < minimum_cells and not include_rare_types:
            continue
        cluster_count = min(max_prototypes_per_type, max(1, values.shape[0] // minimum_cells))
        if cluster_count == 1:
            assignments = np.zeros(values.shape[0], dtype=np.int64)
        else:
            fitted = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10).fit(values)
            assignments = _merge_small_clusters(
                fitted.labels_.astype(np.int64),
                fitted.cluster_centers_,
                minimum_cells,
            )
        for state_index, cluster in enumerate(np.unique(assignments)):
            selected = values[assignments == cluster]
            count = int(selected.shape[0])
            alpha = count / (count + shrinkage_kappa) if shrinkage_kappa else 1.0
            raw_mean = selected.mean(axis=0)
            atlas_mean = atlas_means.get(type_name, raw_mean)
            shrunken_mean = alpha * raw_mean + (1.0 - alpha) * atlas_mean
            variance = selected.var(axis=0, ddof=1 if count > 1 else 0)
            means.append(shrunken_mean)
            variances.append(np.maximum(variance, variance_floor))
            counts.append(count)
            type_indices.append(type_index)
            safe_type = type_name.replace(" ", "_").replace("/", "_")
            identifiers.append("%s:%s:%d" % (sample_id, safe_type, state_index))

    if not means:
        raise ValueError("no cell type meets minimum_cells; merge labels or allow rare types")
    count_array = np.asarray(counts, dtype=np.int64)
    type_index_array = np.asarray(type_indices, dtype=np.int64)
    return PrototypeSet(
        prototype_ids=np.asarray(identifiers, dtype=np.str_),
        sample_ids=np.full(
            len(identifiers), sample_id, dtype=np.dtype("U%d" % max(1, len(str(sample_id))))
        ),
        cell_type_labels=np.asarray(type_names, dtype=np.str_)[type_index_array],
        means=np.stack(means).astype(np.float32),
        variances=np.stack(variances).astype(np.float32),
        weights=(count_array / count_array.sum()).astype(np.float64),
        n_cells=count_array,
        latent_space_id=latent_space_id,
        donor_id=donor_id,
        block_id=block_id,
        source_reference_sha256=source_reference_sha256,
        latent_training_donors=tuple(str(value) for value in latent_training_donors),
        latent_transform_sha256=latent_transform_sha256,
    )
