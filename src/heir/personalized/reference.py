"""Specimen-specific prototypes on a declared nonnegative linear RNA scale.

The initial representation is an explicit, ordered gene panel, not a PCA fitted
on held-out RNA. Any learned shared representation must be fitted on training
patients and supplied separately in a later adapter.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .schema import RNAReference


def _dense(value):
    return value.toarray() if hasattr(value, "toarray") else np.asarray(value)


def _validate_counts(counts):
    if len(counts.shape) != 2 or min(counts.shape) == 0:
        raise ValueError("raw counts must be a nonempty cells-by-genes matrix")
    values = counts.data if hasattr(counts, "tocsr") else np.asarray(counts)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("raw counts must be finite and nonnegative")
    if not np.allclose(values, np.round(values), atol=1e-6, rtol=0):
        raise ValueError("declared raw counts contain noninteger values")


def _identifiers(values, label):
    values = tuple(str(x) for x in values)
    if any(not x.strip() or x.lower() in {"nan", "none"} for x in values):
        raise ValueError(f"{label} contains missing identifiers")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    return values


def _gene_axis(metadata, gene_id_column=None):
    """Recover counts_var independently of the processed .var axis."""
    if gene_id_column is not None:
        try:
            values = metadata[gene_id_column]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"counts gene metadata lacks {gene_id_column!r}") from exc
    elif hasattr(metadata, "index") and not isinstance(metadata, (list, tuple)):
        values = metadata.index
    elif isinstance(metadata, dict):
        for key in ("gene_id", "gene_ids", "_index"):
            if key in metadata:
                values = metadata[key]
                break
        else:
            raise ValueError("counts_var requires an explicit gene_id_column")
    else:
        values = metadata
    if np.asarray(values).ndim != 1:
        raise ValueError("counts gene metadata must identify one gene per column")
    return _identifiers(values, "gene IDs")


def extract_raw_counts(adata, *, source="obsm", gene_id_column=None, raw_declared=False):
    """Return counts and their actual gene IDs; never infer raw counts from X.

    ``source='obsm'`` selects HTAPP's ``obsm['counts']`` and
    ``uns['counts_var']`` together. ``layer:counts`` and ``X`` additionally
    require ``raw_declared=True`` in the source manifest.
    """
    if source == "obsm":
        if "counts" not in adata.obsm or "counts_var" not in adata.uns:
            raise ValueError("HTAPP counts require both obsm['counts'] and uns['counts_var']")
        counts = adata.obsm["counts"]
        genes = _gene_axis(adata.uns["counts_var"], gene_id_column)
    elif source == "X" or source.startswith("layer:"):
        if not raw_declared:
            raise ValueError("X/layer count stores require an explicit raw-count declaration")
        if source == "X":
            counts = adata.X
        else:
            layer = source.split(":", 1)[1]
            if layer not in adata.layers:
                raise ValueError(f"missing raw count layer {layer!r}")
            counts = adata.layers[layer]
        genes = _gene_axis(adata.var, gene_id_column)
    else:
        raise ValueError("counts_source must be obsm, X, or layer:<name>")
    _validate_counts(counts)
    if counts.shape != (adata.n_obs, len(genes)):
        raise ValueError("raw count shape does not match RNA cells and recovered gene axis")
    return counts, genes


def _cluster(features, n_clusters):
    """Small deterministic farthest-point/Lloyd prototype initializer."""
    if n_clusters == 1:
        return np.zeros(len(features), dtype=np.int64)
    selected = [int(np.argmax(((features - features.mean(0)) ** 2).sum(1)))]
    nearest = ((features - features[selected[0]]) ** 2).sum(1)
    for _ in range(1, n_clusters):
        index = int(np.argmax(nearest))
        if nearest[index] <= 0:
            break
        selected.append(index)
        nearest = np.minimum(nearest, ((features - features[index]) ** 2).sum(1))
    centers = features[selected].copy()
    labels = np.zeros(len(features), dtype=np.int64)
    for _ in range(20):
        # Avoid an N-by-K-by-feature temporary for large RNA references.
        distance = (
            (features * features).sum(1)[:, None]
            + (centers * centers).sum(1)[None, :]
            - 2 * features @ centers.T
        )
        new_labels = np.argmin(distance, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for index in range(len(centers)):
            if (labels == index).any():
                centers[index] = features[labels == index].mean(0)
    return labels


def build_reference(
    counts,
    gene_ids: Sequence[str],
    broad_types: Sequence[str],
    *,
    embedding_gene_ids: Sequence[str],
    state_labels: Sequence[str] | None = None,
    max_states: int = 64,
    library_size: float = 10000.0,
) -> RNAReference:
    """Build bounded within-type prototypes without sacrificing rare types.

    Zero-library RNA cells are excluded. Means and variances use per-cell
    library-normalized *linear* expression. The fixed embedding is log1p of
    each state mean on the manifest's ordered gene panel.
    """
    _validate_counts(counts)
    genes = _identifiers(gene_ids, "gene IDs")
    panel = _identifiers(embedding_gene_ids, "embedding gene IDs")
    if not panel or not set(panel).issubset(genes):
        raise ValueError("embedding genes must be a nonempty subset of the actual count axis")
    if counts.shape[1] != len(genes) or len(broad_types) != counts.shape[0]:
        raise ValueError("RNA annotations/count axes differ")
    broad = np.asarray(broad_types, dtype=str)
    if any(not x.strip() or x.lower() in {"nan", "none"} for x in broad):
        raise ValueError("broad RNA type annotations cannot be missing")
    if not isinstance(max_states, int) or max_states < 1 or library_size <= 0:
        raise ValueError("max_states and library_size must be positive")
    totals = np.asarray(counts.sum(axis=1)).ravel()
    keep = totals > 0
    if not keep.any():
        raise ValueError("RNA reference has no positive-library cells")
    counts = counts[keep]
    broad = broad[keep]
    scale = library_size / totals[keep]
    if hasattr(counts, "tocsr"):
        normalized = counts.multiply(scale[:, None]).tocsr()
    else:
        normalized = np.asarray(counts, dtype=np.float64) * scale[:, None]
    index = {gene: i for i, gene in enumerate(genes)}
    panel_indices = [index[gene] for gene in panel]
    types = tuple(sorted(set(broad)))
    if len(types) > max_states:
        raise ValueError("max_states is smaller than the number of broad types")
    state_keys = []
    if state_labels is not None:
        if len(state_labels) != len(keep):
            raise ValueError("RNA state annotation axis differs from counts")
        labels = np.asarray(state_labels, dtype=str)[keep]
        if any(not x.strip() or x.lower() in {"nan", "none"} for x in labels):
            raise ValueError("RNA state annotations cannot be missing")
        for broad_type in types:
            for label in sorted(set(labels[broad == broad_type])):
                state_keys.append((broad_type, label, (broad == broad_type) & (labels == label)))
        if len(state_keys) > max_states:
            raise ValueError("supplied RNA states exceed max_states; qualify a smaller reference")
    else:
        allocation = {kind: 1 for kind in types}
        for _ in range(min(max_states, len(broad)) - len(types)):
            available = [kind for kind in types if allocation[kind] < int((broad == kind).sum())]
            kind = max(available, key=lambda k: np.sqrt((broad == k).sum()) / allocation[k])
            allocation[kind] += 1
        for kind in types:
            row_ids = np.flatnonzero(broad == kind)
            features = np.log1p(_dense(normalized[row_ids][:, panel_indices]))
            labels = _cluster(features, allocation[kind])
            for label in sorted(set(labels)):
                selected = np.zeros(len(broad), dtype=bool)
                selected[row_ids[labels == label]] = True
                state_keys.append((kind, str(label), selected))
    means, variances, sizes = [], [], []
    for _, _, selected in state_keys:
        values = normalized[selected]
        mean = np.asarray(values.mean(axis=0)).ravel()
        squared = values.multiply(values) if hasattr(values, "tocsr") else values * values
        means.append(mean)
        variances.append(np.maximum(np.asarray(squared.mean(axis=0)).ravel() - mean * mean, 0))
        sizes.append(int(selected.sum()))
    means = np.asarray(means, dtype=np.float32)
    sizes = np.asarray(sizes, dtype=np.int64)
    return RNAReference(
        state_ids=tuple(f"{kind}::{label}" for kind, label, _ in state_keys),
        gene_ids=genes,
        expression=means,
        embedding=np.log1p(means[:, panel_indices]).astype(np.float32),
        embedding_gene_ids=panel,
        proportions=(sizes / sizes.sum()).astype(np.float32),
        state_types=tuple(kind for kind, _, _ in state_keys),
        variance=np.asarray(variances, dtype=np.float32),
        cell_counts=sizes,
    )


def select_training_outcomes(
    references,
    spatial_detection,
    training_patient_ids,
    *,
    required_gene_ids,
    max_genes=256,
    minimum_detection=0.01,
    minimum_patients=2,
):
    """Freeze a broader panel using equal-patient RNA dispersion and training ST only.

    ``spatial_detection`` maps each training patient to (gene IDs, detection fractions).
    Fractions should average assay replicates equally. Within each RNA reference, the
    law of total variance combines state means and within-state variances. The ranking
    is the equal-patient mean of log1p(variance / (1 + mean)); ties use gene IDs.
    Required historical outcomes are retained regardless of the detection threshold.
    """
    patients = _identifiers(training_patient_ids, "training patient IDs")
    required = _identifiers(required_gene_ids, "required outcome genes")
    if not patients or set(references) != set(patients) or set(spatial_detection) != set(patients):
        raise ValueError("panel selection requires exactly the declared training patients")
    if (
        isinstance(max_genes, bool)
        or not isinstance(max_genes, int)
        or not len(required) <= max_genes <= 256
        or not 0 <= minimum_detection <= 1
        or not isinstance(minimum_patients, int)
        or not 1 <= minimum_patients <= len(patients)
    ):
        raise ValueError("invalid bounded training-panel selection rule")
    common, detection_maps = None, {}
    for patient in patients:
        genes, fraction = spatial_detection[patient]
        genes = _identifiers(genes, "spatial detection genes")
        fraction = np.asarray(fraction, dtype=float)
        if fraction.shape != (len(genes),) or not np.isfinite(fraction).all():
            raise ValueError("spatial detection must be finite on its named gene axis")
        if np.any(fraction < 0) or np.any(fraction > 1):
            raise ValueError("spatial detection fractions must lie in [0, 1]")
        detection_maps[patient] = dict(zip(genes, fraction))
        available = set(genes) & set(references[patient].gene_ids)
        common = available if common is None else common & available
    if not set(required) <= common:
        raise ValueError("required historical outcomes are absent from training inputs")
    genes = tuple(sorted(common))
    dispersions = []
    for patient in patients:
        reference = references[patient]
        indices = [reference.gene_ids.index(gene) for gene in genes]
        means = np.asarray(reference.expression[:, indices], dtype=np.float64)
        variances = np.asarray(reference.variance[:, indices], dtype=np.float64)
        weights = np.array(reference.proportions, dtype=np.float64, copy=True)
        weights /= weights.sum()
        mean = weights @ means
        variance = np.maximum(weights @ (variances + means**2) - mean**2, 0)
        dispersions.append(np.log1p(variance / (1 + mean)))
    scores = np.mean(dispersions, axis=0)
    rows = []
    for gene, score in zip(genes, scores):
        detected = int(sum(detection_maps[p][gene] >= minimum_detection for p in patients))
        rows.append(
            {
                "gene_id": gene,
                "donor_balanced_rna_dispersion": float(score),
                "training_patients_detected": detected,
                "required_historical_outcome": gene in required,
                "eligible": detected >= minimum_patients or gene in required,
            }
        )
    ranked = sorted(rows, key=lambda row: (-row["donor_balanced_rna_dispersion"], row["gene_id"]))
    additional = [
        row["gene_id"] for row in ranked if row["eligible"] and row["gene_id"] not in required
    ]
    selected = required + tuple(additional[: max_genes - len(required)])
    for row in ranked:
        row["selected"] = row["gene_id"] in selected
    return selected, {
        "schema": "heir.training_outcome_panel.v1",
        "training_patient_ids": sorted(patients),
        "max_genes": max_genes,
        "minimum_training_st_detection": minimum_detection,
        "minimum_training_patients": minimum_patients,
        "gene_ids": list(selected),
        "normalization_gene_ids": list(selected),
        "selection_uses_validation_or_test": False,
        "ranking": "equal_patient_mean_log1p_total_RNA_variance_over_one_plus_mean",
        "candidate_scores": ranked,
    }
