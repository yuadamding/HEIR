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


def normalized_observations(
    prediction: Prediction,
    targets: SpatialTargets,
    operator: ObservationOperator,
    gene_ids: Sequence[str],
    *,
    normalization_gene_ids: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate and normalize on a fixed panel before selecting outcomes.

    Both normalization and outcome IDs are joined to their actual matrix axes.
    Outcome availability is returned for the loss/metric; it does not redefine
    the denominator. Incomplete normalization coverage is an explicit error.
    """
    outcomes, panel = tuple(gene_ids), tuple(normalization_gene_ids)
    identifiers(outcomes, "outcome genes")
    identifiers(panel, "normalization genes")
    if len(panel) < 2:
        raise ValueError("normalization requires at least two distinct fixed-panel genes")
    genes = panel + tuple(gene for gene in outcomes if gene not in panel)
    predicted, truth, mask = aligned_observations(prediction, targets, operator, genes)
    indices = tuple(range(len(panel)))
    normalized_prediction = normalized_log_expression(
        predicted,
        mask,
        normalization_indices=indices,
    )
    normalized_truth = normalized_log_expression(truth, mask, normalization_indices=indices)
    selected = [genes.index(gene) for gene in outcomes]
    return normalized_prediction[:, selected], normalized_truth[:, selected], mask[:, selected]


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, including ties, without an optional scipy dependency."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2)[inverse]


def _score_specimen(
    prediction: Prediction,
    targets: SpatialTargets,
    operator: ObservationOperator,
    gene_ids: Sequence[str],
    *,
    normalization_gene_ids: Sequence[str],
    independent: bool,
    min_regions: int,
) -> dict:
    predicted, truth, mask = normalized_observations(
        prediction,
        targets,
        operator,
        gene_ids,
        normalization_gene_ids=normalization_gene_ids,
    )
    predicted, truth = (array.detach().cpu().numpy() for array in (predicted, truth))
    mask = mask.cpu().numpy()
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
        "patient_id": targets.patient_id,
        "specimen_id": targets.specimen_id,
        "expression_scale": "relative",
        "normalization": "aggregate_linear_then_fixed_complete_panel_library_1e4_then_log1p",
        "normalization_gene_ids": list(normalization_gene_ids),
        "median_gene_spearman": None if incomplete else subset_median,
        "defined_subset_median_diagnostic": subset_median,
        "headline_status": (
            "undefined_prediction_on_evaluable_panel"
            if incomplete
            else "evaluable"
            if defined
            else "no_evaluable_target_genes"
        ),
        "normalized_log_mse": (
            float(np.mean((predicted[mask] - truth[mask]) ** 2)) if mask.any() else None
        ),
        "defined_genes": len(defined),
        "requested_genes": len(rows),
        "undefined_genes": len(rows) - len(defined),
        "gene_scores": rows,
        "median_policy": "target-defined eligibility; undefined predictions invalidate headline",
        "biological_claim": "developmental prediction score; no communication claim",
    }


def score_patient(
    samples: Sequence[tuple[Prediction, SpatialTargets, ObservationOperator]],
    gene_ids: Sequence[str],
    *,
    normalization_gene_ids: Sequence[str],
    independent: bool = False,
    min_regions: int = 3,
) -> dict:
    """Score spatial variation within specimens, then average specimen medians.

    Regions from separate specimens are never pooled into a spatial correlation.
    Each target-evaluable specimen contributes equally regardless of its region
    count. Undefined predictions invalidate the headline instead of silently
    excluding unfavorable specimens. Target-only exclusions remain explicit.
    """
    if not samples or min_regions < 3:
        raise ValueError("at least one specimen and min_regions >= 3 are required")
    if len({target.patient_id for _, target, _ in samples}) != 1:
        raise ValueError("score_patient requires exactly one biological patient")
    if len({target.specimen_id for _, target, _ in samples}) != len(samples):
        raise ValueError("duplicate specimens would count spatial observations twice")
    specimens = [
        _score_specimen(
            prediction,
            target,
            operator,
            gene_ids,
            normalization_gene_ids=normalization_gene_ids,
            independent=independent,
            min_regions=min_regions,
        )
        for prediction, target, operator in samples
    ]
    incomplete = any(
        item["headline_status"] == "undefined_prediction_on_evaluable_panel" for item in specimens
    )
    defined = [
        item["median_gene_spearman"]
        for item in specimens
        if item["median_gene_spearman"] is not None
    ]
    mean = float(np.mean(defined)) if defined else None
    errors = [
        item["normalized_log_mse"] for item in specimens if item["normalized_log_mse"] is not None
    ]
    return {
        "patient_id": samples[0][1].patient_id,
        "expression_scale": "relative",
        "normalization": "aggregate_linear_then_fixed_complete_panel_library_1e4_then_log1p",
        "normalization_gene_ids": list(normalization_gene_ids),
        "evaluation_gene_ids": list(gene_ids),
        "mean_specimen_median_gene_spearman": None if incomplete else mean,
        "defined_specimen_mean_diagnostic": mean,
        "headline_status": (
            "undefined_prediction_on_evaluable_panel"
            if incomplete
            else "evaluable"
            if defined
            else "no_evaluable_target_genes"
        ),
        "normalized_log_mse": float(np.mean(errors)) if errors else None,
        "requested_genes": len(gene_ids),
        "requested_specimens": len(specimens),
        "evaluable_specimens": len(defined),
        "undefined_specimens": len(specimens) - len(defined),
        "aggregation": "equal_specimen_mean_of_within_specimen_median_gene_spearman",
        "specimen_scores": specimens,
        "biological_claim": "developmental prediction score; no communication claim",
    }


def score(
    prediction: Prediction,
    targets: SpatialTargets,
    operator: ObservationOperator,
    gene_ids: Sequence[str],
    *,
    normalization_gene_ids: Sequence[str],
    independent: bool = False,
    min_regions: int = 3,
) -> dict:
    """Score one specimen; use score_patient for an equal-specimen patient endpoint."""
    if min_regions < 3:
        raise ValueError("min_regions must be at least three")
    return _score_specimen(
        prediction,
        targets,
        operator,
        gene_ids,
        normalization_gene_ids=normalization_gene_ids,
        independent=independent,
        min_regions=min_regions,
    )
