"""Join completed predictions to targets using fixed, explicit evaluation axes."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .observation import ObservationOperator, normalized_log_expression
from .schema import Prediction, SpatialTargets, identifiers


def aligned_observations(
    prediction: Prediction,
    targets: SpatialTargets,
    operator: ObservationOperator,
    gene_ids: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    genes = tuple(gene_ids)
    identifiers(genes, "evaluation genes")
    if prediction.patient_id != targets.patient_id or prediction.specimen_id != targets.specimen_id:
        raise ValueError("prediction/target specimen or patient identity mismatch")
    if set(operator.observation_ids) != set(targets.observation_ids):
        raise ValueError("observation geometry and targets must have identical observation IDs")
    pgenes = {key: i for i, key in enumerate(prediction.gene_ids)}
    tgenes = {key: i for i, key in enumerate(targets.gene_ids)}
    missing = set(genes) - (set(pgenes) & set(tgenes))
    if missing:
        raise ValueError(f"prespecified genes absent from prediction/targets: {sorted(missing)}")
    aggregate = operator.aggregate(prediction)
    predicted = aggregate[:, [pgenes[g] for g in genes]]
    lookup = {key: i for i, key in enumerate(targets.observation_ids)}
    rows = [lookup[key] for key in operator.observation_ids]
    columns = [tgenes[g] for g in genes]
    truth = torch.as_tensor(
        np.asarray(targets.expression)[np.ix_(rows, columns)],
        dtype=predicted.dtype,
        device=predicted.device,
    )
    mask = torch.as_tensor(
        np.asarray(targets.observed)[np.ix_(rows, columns)],
        dtype=torch.bool,
        device=predicted.device,
    )
    return predicted, truth, mask


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, including ties, without an optional scipy dependency."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2)[inverse]


def score_patient(
    samples: Sequence[tuple[Prediction, SpatialTargets, ObservationOperator]],
    gene_ids: Sequence[str],
    *,
    independent: bool = False,
    min_regions: int = 3,
) -> dict:
    """Pool a patient's regions, then report the median gene-wise spatial Spearman.

    Every requested gene is reported, including undefined correlations. Independent
    evaluation must be established by caller split provenance, never by this metric.
    """
    if not samples or min_regions < 3:
        raise ValueError("at least one specimen and min_regions >= 3 are required")
    if len({target.patient_id for _, target, _ in samples}) != 1:
        raise ValueError("score_patient requires exactly one biological patient")
    if len({target.specimen_id for _, target, _ in samples}) != len(samples):
        raise ValueError("duplicate specimens would count spatial observations twice")
    predicted_parts, truth_parts, masks = [], [], []
    for prediction, targets, operator in samples:
        predicted, truth, mask = aligned_observations(prediction, targets, operator, gene_ids)
        predicted_parts.append(normalized_log_expression(predicted, mask).detach().cpu().numpy())
        truth_parts.append(normalized_log_expression(truth, mask).detach().cpu().numpy())
        masks.append(mask.cpu().numpy())
    predicted, truth, mask = map(np.concatenate, (predicted_parts, truth_parts, masks))
    if not mask.any():
        raise ValueError("no measured values on the prespecified evaluation panel")
    rows = []
    for j, gene in enumerate(gene_ids):
        keep = mask[:, j]
        x, y = predicted[keep, j], truth[keep, j]
        reason = None
        if len(x) < min_regions:
            reason = "insufficient_regions"
        elif np.ptp(y) == 0:
            reason = "constant_target"
        elif np.ptp(x) == 0:
            reason = "constant_prediction"
        correlation = None if reason else float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])
        rows.append(
            {
                "gene_id": gene,
                "n_regions": int(keep.sum()),
                "spearman": correlation,
                "undefined_reason": reason,
                "independently_evaluated": independent,
            }
        )
    defined = [row["spearman"] for row in rows if row["spearman"] is not None]
    incomplete = any(row["undefined_reason"] == "constant_prediction" for row in rows)
    subset_median = float(np.median(defined)) if defined else None
    return {
        "patient_id": samples[0][1].patient_id,
        "expression_scale": "relative",
        "normalization": "aggregate_linear_then_assay_panel_library_1e4_then_log1p",
        "median_gene_spearman": None if incomplete else subset_median,
        "defined_subset_median_diagnostic": subset_median,
        "headline_status": (
            "undefined_prediction_on_evaluable_panel"
            if incomplete
            else "evaluable"
            if defined
            else "no_evaluable_target_genes"
        ),
        "normalized_log_mse": float(np.mean((predicted[mask] - truth[mask]) ** 2)),
        "defined_genes": len(defined),
        "requested_genes": len(rows),
        "undefined_genes": len(rows) - len(defined),
        "gene_scores": rows,
        "median_policy": "target-defined eligibility; undefined predictions invalidate headline",
        "biological_claim": "developmental prediction score; no communication claim",
    }


def score(prediction, targets, operator, gene_ids, **kwargs):
    """Score one specimen; use score_patient for patients with several specimens."""
    return score_patient([(prediction, targets, operator)], gene_ids, **kwargs)
