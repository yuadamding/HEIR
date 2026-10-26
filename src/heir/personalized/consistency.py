"""Measurement consistency and fixed-model protein checks on prepared cohort objects.

These analyses reuse saved prediction/evaluation coverage. Replicate overlap is
selected by geometry and the already fixed exposure rule, never by a favorable
outcome or model score. Neither public helper fits a model or reads source data.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np

from .cohort import _restore_grid
from .cohort_cli import _subset, replicate_scores
from .diagnostics import (
    _array,
    _change,
    _normalized,
    _region_rows,
    _summary,
    permute_protein_values,
    replicate_agreement,
)


def _protein_coverage(inputs):
    n = len(inputs.cell_ids)
    coverage = {
        key: np.zeros(n, dtype=bool)
        for key in ("any_usable", "complete_observed", "complete_usable")
    }
    protein = inputs.proteins
    if protein is None:
        return coverage
    quality = np.ones_like(protein.values) if protein.quality is None else protein.quality
    usable = protein.observed & (quality > 0)
    flags = {
        "any_usable": usable.any(1),
        "complete_observed": protein.observed.all(1),
        "complete_usable": usable.all(1),
    }
    for key, values in flags.items():
        np.logical_or.at(coverage[key], _region_rows(protein.operator), values)
    return coverage


def _target_details(inputs, targets, operator, groups, metadata, normal, provenance):
    grid = _restore_grid(inputs, provenance)
    regions = _region_rows(operator)
    if operator.cell_ids != inputs.cell_ids or operator.observation_ids != targets.observation_ids:
        raise ValueError("consistency requires aligned named target and H&E observation axes")
    if len(groups) != len(targets.observation_ids):
        raise ValueError("replicate labels must align with the target rows")
    ids = tuple(metadata["all_observation_ids"])
    counts = np.asarray(metadata["region_bead_counts"])
    replicates = tuple(metadata["observation_replicates"])
    if (
        len(set(ids)) != len(ids)
        or counts.shape != (len(ids),)
        or len(replicates) != len(ids)
        or counts.dtype.kind not in "iu"
        or np.any(counts <= 0)
        or not set(targets.observation_ids) <= set(ids)
    ):
        raise ValueError("target bead-count metadata must have a complete unique original row axis")
    beads, assay = dict(zip(ids, counts)), dict(zip(ids, replicates))
    norm = [targets.gene_ids.index(gene) for gene in normal]
    exposure = targets.expression[:, norm].sum(1)
    if not targets.observed[:, norm].all() or np.any(exposure <= 0):
        raise ValueError("consistency requires previously qualified positive fixed-panel exposure")
    coverage = _protein_coverage(inputs)
    rows = []
    for i, region in enumerate(regions):
        name = targets.observation_ids[i]
        if str(assay[name]) != str(groups[i]):
            raise ValueError("replicate metadata differs from qualified observation groups")
        rows.append(
            {
                "observation_id": name,
                "region_id": inputs.cell_ids[region],
                "assay_replicate": str(groups[i]),
                "normalization_exposure": float(exposure[i]),
                "target_bead_count": int(beads[name]),
                "tissue_fraction": float(grid.tissue_fraction[region]),
                **{f"codex_{key}": bool(value[region]) for key, value in coverage.items()},
            }
        )
    return rows


def _coverage_summary(rows):
    return {
        "n_region_replicates": len(rows),
        **{
            key: _summary([r[key] for r in rows])
            for key in ("normalization_exposure", "target_bead_count", "tissue_fraction")
        },
        **{
            key: sum(r[key] for r in rows)
            for key in ("codex_any_usable", "codex_complete_observed", "codex_complete_usable")
        },
    }


def _model_scores(prediction, targets, operator, groups, genes, normal):
    report = replicate_scores(prediction, targets, operator, groups, genes, normal)
    predicted, truth, observed = _normalized(prediction, targets, operator, genes, normal)
    for replicate in report["replicate_scores"]:
        selected = np.asarray(groups) == replicate["assay_replicate"]
        for j, row in enumerate(replicate["gene_scores"]):
            keep = selected & observed[:, j]
            row["normalized_log_mse"] = (
                float(np.mean((predicted[keep, j] - truth[keep, j]) ** 2)) if keep.any() else None
            )
    return report


def measurement_consistency(
    inputs,
    predictions,
    targets,
    operator,
    groups,
    *,
    gene_ids,
    normalization_gene_ids,
    target_metadata,
    provenance,
    patient_metadata=None,
):
    """Link target replicate agreement to every supplied model on identical regions.

    ``predictions`` maps saved arm/export names to Predictions. ``target_metadata``
    includes the original ``all_observation_ids``, ``region_bead_counts`` and
    ``observation_replicates`` axes, before exposure filtering. The caller supplies
    the already qualified targets and groups; no additional eligibility is fitted.
    """
    if not predictions:
        raise ValueError("at least one saved prediction is required")
    genes, normal = tuple(gene_ids), tuple(normalization_gene_ids)
    groups = tuple(str(group) for group in groups)
    for prediction in predictions.values():
        if (
            prediction.cell_ids != inputs.cell_ids
            or prediction.patient_id != inputs.patient_id
            or prediction.specimen_id != inputs.specimen_id
            or prediction.prediction_unit != inputs.prediction_unit
            or prediction.coordinate_units != inputs.coordinate_units
            or not np.array_equal(_array(prediction.coordinates), inputs.coordinates)
        ):
            raise ValueError(
                "saved predictions must share the prepared patient and region geometry"
            )
    reference = next(iter(predictions.values()))
    details = _target_details(
        inputs, targets, operator, groups, target_metadata, normal, provenance
    )
    agreement = replicate_agreement(reference, targets, operator, groups, genes, normal)
    eligible_labels = set(groups)
    source_labels = sorted(
        map(
            str,
            target_metadata.get(
                "source_replicates", set(target_metadata["observation_replicates"])
            ),
        )
    )
    if not eligible_labels <= set(source_labels) or not set(
        map(str, target_metadata["observation_replicates"])
    ) <= set(source_labels):
        raise ValueError("observed replicate labels are absent from source metadata")
    agreement_pairs = {tuple(pair["replicates"]): pair for pair in agreement["pairs"]}
    pair_reports = []
    for labels in combinations(source_labels, 2):
        pair = agreement_pairs.get(
            labels,
            {
                "replicates": list(labels),
                "shared_eligible_region_ids": [],
                "n_shared_eligible_regions": 0,
                "gene_scores": [
                    {
                        "gene_id": gene,
                        "n_shared_eligible_regions": 0,
                        "spearman": None,
                        "normalized_log_mse": None,
                        "undefined_reason": "replicate_without_eligible_regions",
                    }
                    for gene in genes
                ],
            },
        )
        common = set(pair["shared_eligible_region_ids"])
        labels = set(pair["replicates"])
        keep = np.array(
            [r["region_id"] in common and r["assay_replicate"] in labels for r in details]
        )
        selected = [r for r, included in zip(details, keep) if included]
        models = {}
        if keep.any():
            target, mapping = _subset(targets, operator, keep)
            selected_groups = tuple(np.asarray(groups)[keep])
            models = {
                name: _model_scores(prediction, target, mapping, selected_groups, genes, normal)
                for name, prediction in predictions.items()
            }
        pair_reports.append(
            {
                **pair,
                "status": "reported"
                if common
                else "replicate_without_eligible_regions"
                if not set(labels) <= eligible_labels
                else "no_shared_eligible_regions",
                "same_region_model_scores": models,
                "coverage_by_replicate": {
                    label: _coverage_summary([r for r in selected if r["assay_replicate"] == label])
                    for label in pair["replicates"]
                },
                "exact_observations": selected,
            }
        )
    return {
        "schema": "heir.cohort_measurement_consistency.v1",
        "patient_id": inputs.patient_id,
        "specimen_id": inputs.specimen_id,
        "prediction_unit": inputs.prediction_unit,
        "coordinate_units": inputs.coordinate_units,
        "evaluation_gene_ids": list(genes),
        "normalization_gene_ids": list(normal),
        "status": "single_replicate"
        if len(source_labels) == 1
        else "reported"
        if any(p["n_shared_eligible_regions"] for p in pair_reports)
        else "replicate_without_eligible_regions"
        if len(eligible_labels) < len(source_labels)
        else "no_shared_eligible_regions",
        "replicate_labels": source_labels,
        "eligible_replicate_labels": sorted(eligible_labels),
        "pairs": pair_reports,
        "all_eligible_coverage_by_replicate": {
            label: _coverage_summary([r for r in details if r["assay_replicate"] == label])
            for label in sorted(set(groups))
        },
        "all_eligible_model_scores": {
            name: _model_scores(prediction, targets, operator, groups, genes, normal)
            for name, prediction in predictions.items()
        },
        "exact_eligible_observations": details,
        "descriptive_patient_metadata": dict(patient_metadata or {}),
        "metadata_interpretation": (
            "recorded reference modality/site; descriptive only, no subgroup inference"
        ),
        "coverage_policy": (
            "shared previously eligible H&E regions; no gene or model score selection"
        ),
        "interpretation": "measurement consistency reference, not an attainable prediction ceiling",
        "new_fitting_performed": False,
    }


def _magnitude(values):
    return {
        "mean_absolute": float(np.abs(values).mean()),
        "root_mean_square": float(np.sqrt(np.mean(values**2))),
        "max_absolute": float(np.abs(values).max()),
    }


def _logit_comparison(base, adjustment, indices):
    output = {}
    for label, a, b in (
        ("raw", base[indices], adjustment[indices]),
        (
            "centered_within_region",
            base[indices] - base[indices].mean(1, keepdims=True),
            adjustment[indices] - adjustment[indices].mean(1, keepdims=True),
        ),
    ):
        first, second = _magnitude(a), _magnitude(b)
        denominator = first["root_mean_square"]
        output[label] = {
            "base_compatibility": first,
            "actual_protein_adjustment": second,
            "adjustment_to_base_rms_ratio": second["root_mean_square"] / denominator
            if denominator > 0
            else None,
            "ratio_status": "defined" if denominator > 0 else "undefined_zero_base_rms",
        }
    return {"n_regions": int(len(indices)), **output}


def protein_pathway_diagnostics(
    fitted,
    inputs,
    *,
    scored_operator=None,
    expected_prediction=None,
    permutation_seed=0,
):
    """Observe the actual fitted protein update and fixed-weight interventions.

    The existing ``_protein_logits`` call is temporarily wrapped to record its
    input/output, then restored in ``finally`` even on failure. Its formula is
    not reimplemented. Run on an inference model not shared with another thread.
    Centered logit ratios remove state-common offsets irrelevant to softmax.
    """
    protein, model = inputs.proteins, fitted.model
    if protein is None:
        return {"status": "no_protein_evidence", "new_fitting_performed": False}
    if model.config.variant != "he_rna_protein" or model.config.assignment_mode != "learned":
        raise ValueError("protein diagnostics require the fitted learned protein baseline")
    if scored_operator is not None and scored_operator.cell_ids != inputs.cell_ids:
        raise ValueError("scored operator differs from the prepared H&E region axis")
    captured = []
    original_method = model._protein_logits
    had_local_method = "_protein_logits" in model.__dict__

    def record_update(logits, actual_inputs):
        result = original_method(logits, actual_inputs)
        captured.append(
            (_array(logits[:, 1:]).copy(), _array(result[:, 1:] - logits[:, 1:]).copy())
        )
        return result

    try:
        model._protein_logits = record_update
        original = fitted.predict(inputs)
    finally:
        if had_local_method:
            model._protein_logits = original_method
        else:
            del model._protein_logits
    if len(captured) != 1:
        raise ValueError("expected one actual protein update in baseline prediction")
    if expected_prediction is not None:
        if (
            expected_prediction.cell_ids != original.cell_ids
            or expected_prediction.gene_ids != original.gene_ids
            or expected_prediction.state_ids != original.state_ids
            or expected_prediction.patient_id != original.patient_id
            or expected_prediction.specimen_id != original.specimen_id
            or expected_prediction.prediction_unit != original.prediction_unit
            or expected_prediction.coordinate_units != original.coordinate_units
            or not np.array_equal(
                _array(expected_prediction.coordinates), _array(original.coordinates)
            )
        ):
            raise ValueError("expected saved prediction has different region/gene/state identities")
        for name in ("conditional_assignment", "transfer", "expression", "log_correction"):
            if not np.allclose(
                _array(getattr(original, name)),
                _array(getattr(expected_prediction, name)),
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError("protein replay differs from the saved prediction")
    base, adjustment = captured[0]
    scopes = {"all_he_regions": np.arange(len(inputs.cell_ids))}
    if scored_operator is not None:
        scopes["unique_scored_regions"] = np.unique(_region_rows(scored_operator))
    templates = np.asarray(protein.state_values)
    proportions = np.asarray(inputs.reference.proportions, dtype=np.float64)
    proportions = proportions / proportions.sum()
    means = proportions @ templates
    marker_rows = [
        {
            "marker_id": marker,
            "n_reference_states": len(proportions),
            "min": float(templates[:, j].min()),
            "max": float(templates[:, j].max()),
            "range": float(np.ptp(templates[:, j])),
            "std_across_states": float(templates[:, j].std()),
            "reference_proportion_weighted_mean": float(means[j]),
            "reference_proportion_weighted_std": float(
                np.sqrt(proportions @ (templates[:, j] - means[j]) ** 2)
            ),
        }
        for j, marker in enumerate(protein.marker_ids)
    ]
    shuffled, permutation = permute_protein_values(protein, seed=permutation_seed)
    conditions = {
        "zero_confidence": replace(protein, quality=np.zeros_like(protein.values)),
        "permuted_values": shuffled,
    }
    scored = scopes.get("unique_scored_regions", scopes["all_he_regions"])
    changes = {
        name: _change(original, fitted.predict(replace(inputs, proteins=evidence)), scored)
        for name, evidence in conditions.items()
    }
    if scored_operator is None:
        for change in changes.values():
            change.pop("unique_scored_regions")
    coverage = _protein_coverage(inputs)
    return {
        "schema": "heir.cohort_protein_pathway.v1",
        "status": "reported",
        "patient_id": inputs.patient_id,
        "specimen_id": inputs.specimen_id,
        "state_marker_templates": marker_rows,
        "template_weighting": "RNA reference proportions; not asserted tissue cell abundance",
        "logit_magnitude": {
            name: _logit_comparison(base, adjustment, indices) for name, indices in scopes.items()
        },
        "protein_coverage": {
            name: {
                "n_regions": int(len(indices)),
                **{key: int(values[indices].sum()) for key, values in coverage.items()},
            }
            for name, indices in scopes.items()
        },
        "permutation": permutation,
        "prediction_interventions": changes,
        "expected_saved_prediction_verified": expected_prediction is not None,
        "interpretation": (
            "fixed-weight prediction interventions, no refitting or expression-benefit claim"
        ),
        "new_fitting_performed": False,
    }
