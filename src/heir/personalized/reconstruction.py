"""Target-informed, fixed-budget RNA-reference representability diagnostics.

These mixtures use measured spatial outcomes and are not deployable predictions.
Softmax-constrained Adam supplies a local fit, not a certified reconstruction bound.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .observation import ObservationOperator, normalized_log_expression
from .schema import RNAReference, SpatialTargets, identifiers


@dataclass(frozen=True)
class ReferenceReconstruction:
    patient_id: str
    specimen_id: str
    observation_ids: tuple[str, ...]
    state_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    objective_gene_ids: tuple[str, ...]
    normalization_gene_ids: tuple[str, ...]
    q: np.ndarray
    reference_expression: np.ndarray
    expression: np.ndarray
    normalized_expression: np.ndarray
    initial_loss: np.ndarray
    final_loss: np.ndarray
    selected_start: np.ndarray
    selected_step: np.ndarray
    start_initial_loss: np.ndarray
    start_best_loss: np.ndarray
    start_final_loss: np.ndarray
    start_best_step: np.ndarray
    metadata: dict


def fit_reference_mixtures(
    reference: RNAReference,
    targets: SpatialTargets,
    *,
    gene_ids: Sequence[str],
    normalization_gene_ids: Sequence[str],
    steps: int = 1000,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> ReferenceReconstruction:
    """Fit an independent simplex mixture to every source observation.

    The two deterministic starts are uniform and RNA proportions (floored at
    1e-12 before taking logarithms). Every iterate, including initialization,
    participates in per-row best selection. The optimized scalar is the SUM of
    per-row observed-gene mean losses, so another region cannot rescale a row's
    Adam gradient. ``seed`` is recorded; neither start uses random numbers.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("reconstruction steps must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("reconstruction learning_rate must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("reconstruction seed must be a nonnegative integer")
    outcomes, panel = tuple(gene_ids), tuple(normalization_gene_ids)
    identifiers(outcomes, "reconstruction objective gene IDs")
    identifiers(panel, "reconstruction normalization gene IDs")
    if len(panel) < 2:
        raise ValueError("reconstruction requires at least two fixed normalization genes")
    genes = outcomes + tuple(g for g in panel if g not in outcomes)
    for source, axis in (("reference", reference.gene_ids), ("targets", targets.gene_ids)):
        absent = sorted(set(genes) - set(axis))
        if absent:
            raise ValueError(f"reconstruction {source} lacks genes: {absent}")
    reference_order = [reference.gene_ids.index(g) for g in genes]
    target_order = [targets.gene_ids.index(g) for g in genes]
    normalizer = tuple(genes.index(g) for g in panel)
    b = torch.tensor(reference.expression[:, reference_order], dtype=torch.float64)
    values = torch.tensor(targets.expression[:, target_order], dtype=torch.float64)
    observed = torch.tensor(targets.observed[:, target_order], dtype=torch.bool)
    truth = normalized_log_expression(values, observed, normalization_indices=normalizer)
    outcome_mask = observed[:, : len(outcomes)]
    if not outcome_mask.any(dim=1).all():
        raise ValueError("every reconstruction row needs at least one observed objective gene")
    if not torch.any(b[:, normalizer].sum(dim=1) > 0):
        raise ValueError("reference has zero expression across the entire normalization panel")

    n, k = len(targets.observation_ids), len(reference.state_ids)
    prior_floor = 1e-12
    prior = torch.tensor(reference.proportions, dtype=torch.float64).clamp_min(prior_floor)
    initial = torch.stack((torch.zeros(k, dtype=torch.float64), torch.log(prior)))
    logits = torch.nn.Parameter(initial[:, None, :].expand(2, n, k).clone())
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    count = outcome_mask.sum(dim=1)

    def evaluate():
        q = torch.softmax(logits, dim=-1)
        expression = q @ b
        flat = expression.reshape(2 * n, len(genes))
        normalized = normalized_log_expression(
            flat, torch.ones_like(flat, dtype=torch.bool), normalization_indices=normalizer
        ).reshape(2, n, len(genes))
        squared = (normalized[:, :, : len(outcomes)] - truth[None, :, : len(outcomes)]) ** 2
        loss = (squared * outcome_mask[None]).sum(dim=-1) / count[None]
        if not torch.isfinite(loss).all():
            raise ValueError("nonfinite reconstruction objective; fixed-budget fit failed")
        return q, loss

    q, loss = evaluate()
    initial_loss = loss.detach().clone()
    best_loss = loss.detach().clone()
    best_q = q.detach().clone()
    best_step = torch.zeros((2, n), dtype=torch.long)
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        loss.sum().backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise ValueError("nonfinite reconstruction gradient; fixed-budget fit failed")
        optimizer.step()
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite reconstruction logits; fixed-budget fit failed")
        q, loss = evaluate()
        with torch.no_grad():
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            best_q = torch.where(improved[:, :, None], q, best_q)
            best_step = torch.where(improved, step, best_step)

    rows = torch.arange(n)
    selected_start = best_loss.argmin(dim=0)
    selected_q = best_q[selected_start, rows]
    expression = selected_q @ b
    normalized = normalized_log_expression(
        expression,
        torch.ones_like(expression, dtype=torch.bool),
        normalization_indices=normalizer,
    )
    final_loss = best_loss[selected_start, rows]
    baseline_loss = initial_loss.min(dim=0).values
    simplex_error = float(torch.max(torch.abs(selected_q.sum(dim=1) - 1)))
    metadata = {
        "schema": "heir.reference_reconstruction.v1",
        "target_informed": True,
        "deployable_prediction": False,
        "certified_bound": False,
        "fit_kind": "local_fit",
        "he_mapper": False,
        "expression_correction": False,
        "patient_id": targets.patient_id,
        "specimen_id": targets.specimen_id,
        "objective": "observed_gene_mean_squared_fixed_panel_log1p_cp10000",
        "optimizer_reduction": "sum_of_independent_row_mean_losses",
        "constraint": "nonnegative_simplex_via_softmax",
        "optimizer": "Adam",
        "optimizer_status": "fixed_budget_completed",
        "convergence_assessed": False,
        "steps_requested": steps,
        "steps_completed_per_start": steps,
        "initializations": ["uniform", "reference_proportions"],
        "reference_proportions_floor": prior_floor,
        "learning_rate": float(learning_rate),
        "seed": seed,
        "random_initialization": False,
        "dtype": "float64",
        "device": "cpu",
        "selection": "lowest_row_loss_across_starts_and_all_iterates_including_initial",
        "tie_break": "earlier_iterate_within_start_then_uniform_start",
        "max_simplex_error": simplex_error,
        "minimum_q": float(selected_q.min()),
        "mean_initial_loss": float(baseline_loss.mean()),
        "mean_selected_loss": float(final_loss.mean()),
        "zero_reference_gene_ids": [g for i, g in enumerate(genes) if not torch.any(b[:, i] > 0)],
        "zero_panel_reference_state_ids": [
            s for i, s in enumerate(reference.state_ids) if not torch.any(b[i, normalizer] > 0)
        ],
    }

    def array(value):
        return value.detach().numpy().copy()

    return ReferenceReconstruction(
        patient_id=targets.patient_id,
        specimen_id=targets.specimen_id,
        observation_ids=targets.observation_ids,
        state_ids=reference.state_ids,
        gene_ids=genes,
        objective_gene_ids=outcomes,
        normalization_gene_ids=panel,
        q=array(selected_q),
        reference_expression=array(b),
        expression=array(expression),
        normalized_expression=array(normalized),
        initial_loss=array(baseline_loss),
        final_loss=array(final_loss),
        selected_start=array(selected_start),
        selected_step=array(best_step[selected_start, rows]),
        start_initial_loss=array(initial_loss),
        start_best_loss=array(best_loss),
        start_final_loss=array(loss),
        start_best_step=array(best_step),
        metadata=metadata,
    )


def _region_axis(operator: ObservationOperator) -> tuple[str, ...]:
    """Require one H&E region per assay row; never infer an inverse aggregation."""
    n = len(operator.observation_ids)
    if (
        len(operator.rows) != n
        or not np.array_equal(np.sort(operator.rows), np.arange(n))
        or not np.array_equal(operator.weights, np.ones(n))
    ):
        raise ValueError("reconstruction requires one unit-weight region per observation")
    columns = np.empty(n, dtype=np.int64)
    columns[operator.rows] = operator.columns
    return tuple(operator.cell_ids[c] for c in columns)


def predict_reconstruction(
    fit: ReferenceReconstruction,
    source_operator: ObservationOperator,
    evaluation_operator: ObservationOperator,
    *,
    patient_id: str,
    specimen_id: str,
) -> np.ndarray:
    """Reuse fitted qB on named shared regions without accepting evaluation counts.

    Operators contain geometry only, so the caller must explicitly supply the
    evaluation patient/specimen identity. Their stable region IDs, rather than
    numeric column positions, establish cross-replicate correspondence.
    """
    if patient_id != fit.patient_id or specimen_id != fit.specimen_id:
        raise ValueError("cross-replicate reconstruction patient/specimen identity mismatch")
    if set(source_operator.observation_ids) != set(fit.observation_ids):
        raise ValueError("source operator observation axis differs from fitted observations")
    source_regions = _region_axis(source_operator)
    if len(set(source_regions)) != len(source_regions):
        raise ValueError("source reconstruction region axis contains duplicate regions")
    evaluation_regions = _region_axis(evaluation_operator)
    absent = sorted(set(evaluation_regions) - set(source_regions))
    if absent:
        raise ValueError(f"evaluation regions absent from source reconstruction: {absent}")
    fit_rows = {key: i for i, key in enumerate(fit.observation_ids)}
    source_rows = {
        region: fit_rows[observation]
        for region, observation in zip(source_regions, source_operator.observation_ids)
    }
    order = [source_rows[region] for region in evaluation_regions]
    return fit.expression[order].copy()


def save_reconstruction(fit: ReferenceReconstruction, path: str | Path) -> None:
    """Write a new, pickle-free diagnostic artifact, distinct from model predictions."""
    payload = {
        field.name: np.asarray(getattr(fit, field.name))
        for field in fields(fit)
        if field.name != "metadata"
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(fit.metadata, sort_keys=True, allow_nan=False, separators=(",", ":"))
    )
    if any(value.dtype.kind == "O" for value in payload.values()):
        raise ValueError("reconstruction artifacts cannot contain pickled object arrays")
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **payload)
