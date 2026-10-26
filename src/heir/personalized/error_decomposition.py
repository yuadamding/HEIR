"""Decompose saved prediction errors on the evaluator's fixed normalized scale.

The bias term is the weighted mean of squared replicate/gene mean errors,
not the square of one mean pooled across genes or assay replicates.
"""

from collections import Counter

import numpy as np

from .evaluate import normalized_observations

_SCHEMA = "heir.normalized_error_decomposition.v1"
_COMPONENTS = ("normalized_log_mse", "squared_bias", "centered_mse")
_IDENTITY = "normalized_log_mse = squared_bias + centered_mse"
_WEIGHTING = (
    "observed_entry_mean_within_replicate_then_equal_defined_replicates_within_specimen"
    "_then_equal_specimens_within_patient_then_equal_patients"
)


def _gene_errors(predicted, observed):
    n = len(observed)
    if not n:
        return {
            "n": 0,
            "mean_bias": None,
            "squared_bias": None,
            "centered_mse": None,
            "mse": None,
            "predicted_std": None,
            "observed_std": None,
            "std_ratio": None,
            "std_ratio_undefined_reason": "no_observed_values",
        }
    error = predicted - observed
    mean_bias = float(error.mean())
    predicted_std, observed_std = float(predicted.std()), float(observed.std())
    return {
        "n": n,
        "mean_bias": mean_bias,
        "squared_bias": mean_bias**2,
        "centered_mse": float(np.mean((error - mean_bias) ** 2)),
        "mse": float(np.mean(error**2)),
        "predicted_std": predicted_std,
        "observed_std": observed_std,
        "std_ratio": predicted_std / observed_std if observed_std else None,
        "std_ratio_undefined_reason": None if observed_std else "constant_observed_values",
    }


def error_decomposition(prediction, targets, operator, groups, gene_ids, normalization_gene_ids):
    """Report one saved specimen's normalized error by assay replicate and gene.

    Group labels follow ``operator.observation_ids``. Outcomes and targets are
    joined by ID before normalization on the fixed panel. Computation uses
    float64 moments of the evaluator's normalized values; tiny differences from
    its float32 final reduction are rounding, not a different weighting rule.
    Standard deviations and centered errors use population denominators (ddof=0).
    """
    if (
        isinstance(groups, str)
        or len(groups) != len(operator.observation_ids)
        or any(not isinstance(group, str) or not group.strip() for group in groups)
    ):
        raise ValueError("replicate labels must name every observation row with nonempty strings")
    genes, panel = tuple(gene_ids), tuple(normalization_gene_ids)
    predicted, observed, available = normalized_observations(
        prediction, targets, operator, genes, normalization_gene_ids=panel
    )
    predicted = predicted.detach().cpu().numpy().astype(np.float64)
    observed = observed.detach().cpu().numpy().astype(np.float64)
    available = available.detach().cpu().numpy()
    labels = np.asarray(groups)
    gene_rows, replicates = [], []
    for group in sorted(set(groups)):
        selected = labels == group
        n_observed = int(available[selected].sum())
        rows = []
        for column, gene in enumerate(genes):
            keep = selected & available[:, column]
            row = {
                "patient_id": prediction.patient_id,
                "specimen_id": prediction.specimen_id,
                "assay_replicate": group,
                "gene_id": gene,
                **_gene_errors(predicted[keep, column], observed[keep, column]),
            }
            row["weight_in_replicate"] = row["n"] / n_observed if n_observed else 0.0
            rows.append(row)
        components = {
            key: sum(
                row["weight_in_replicate"] * row["mse" if key == "normalized_log_mse" else key]
                for row in rows
                if row["n"]
            )
            if n_observed
            else None
            for key in _COMPONENTS
        }
        replicates.append(
            {
                "assay_replicate": group,
                "n_rows": int(selected.sum()),
                "n_observed": n_observed,
                "undefined_reason": None if n_observed else "no_observed_outcomes",
                **components,
            }
        )
        gene_rows.extend(rows)
    defined = [row for row in replicates if row["n_observed"]]
    replicate_weights = {
        row["assay_replicate"]: 1 / len(defined) if row["n_observed"] else 0.0 for row in replicates
    }
    for row in replicates:
        row["weight_in_specimen"] = replicate_weights[row["assay_replicate"]]
    for row in gene_rows:
        row["weight_in_specimen"] = (
            row["weight_in_replicate"] * replicate_weights[row["assay_replicate"]]
        )
    return {
        "schema": _SCHEMA,
        "patient_id": prediction.patient_id,
        "specimen_id": prediction.specimen_id,
        "prediction_unit": prediction.prediction_unit,
        "coordinate_units": prediction.coordinate_units,
        "evaluation_gene_ids": list(genes),
        "normalization_gene_ids": list(panel),
        **{
            key: float(np.mean([row[key] for row in defined])) if defined else None
            for key in _COMPONENTS
        },
        "n_replicates": len(replicates),
        "n_defined_replicates": len(defined),
        "replicate_summaries": replicates,
        "per_gene": gene_rows,
        "aggregation": _WEIGHTING,
        "decomposition_identity": _IDENTITY,
        "moment_denominator": "population_n",
        "analysis_precision": "float64_moments_of_evaluator_normalized_values",
        "interpretation": "saved prediction diagnostic; no fitting or performance bound",
    }


def aggregate_error_decompositions(reports):
    """Equal-patient means matching study MSE; missing specimen scores stay missing.

    Multiply each report's per-gene ``weight_in_specimen`` by the corresponding
    ``specimen_weights`` entry to recover its contribution to the cohort mean.
    We never pool observations across patients to estimate the bias component.
    """
    reports = list(reports)
    if not reports or len({(r["patient_id"], r["specimen_id"]) for r in reports}) != len(reports):
        raise ValueError("decomposition aggregation requires nonempty, unique specimen reports")
    first = reports[0]
    axes = ("evaluation_gene_ids", "normalization_gene_ids", "prediction_unit", "coordinate_units")
    if any(
        r.get("schema") != _SCHEMA or any(r[key] != first[key] for key in axes) for r in reports
    ):
        raise ValueError("decomposition reports must share schema, ordered gene panels, and units")
    for report in reports:
        values = [report[key] for key in _COMPONENTS]
        if all(value is None for value in values):
            continue
        if any(value is None or not np.isfinite(value) or value < 0 for value in values) or (
            not np.isclose(values[0], values[1] + values[2], rtol=1e-10, atol=1e-12)
        ):
            raise ValueError("invalid or inconsistent error decomposition components")
    counts = Counter(report["patient_id"] for report in reports)
    patients = []
    for patient in sorted(counts):
        selected = [r for r in reports if r["patient_id"] == patient]
        defined = all(r["normalized_log_mse"] is not None for r in selected)
        patients.append(
            {
                "patient_id": patient,
                "n_specimens": len(selected),
                "weight_in_cohort": 1 / len(counts),
                "undefined_specimens": [
                    r["specimen_id"] for r in selected if r["normalized_log_mse"] is None
                ],
                **{
                    key: float(np.mean([r[key] for r in selected])) if defined else None
                    for key in _COMPONENTS
                },
            }
        )
    complete = all(row["normalized_log_mse"] is not None for row in patients)
    return {
        "schema": "heir.normalized_error_decomposition_summary.v1",
        **{key: first[key] for key in axes},
        **{
            key: float(np.mean([row[key] for row in patients])) if complete else None
            for key in _COMPONENTS
        },
        "n_patients": len(patients),
        "n_defined_patients": sum(row["normalized_log_mse"] is not None for row in patients),
        "patients": patients,
        "specimen_weights": [
            {
                "patient_id": r["patient_id"],
                "specimen_id": r["specimen_id"],
                "weight_in_cohort": 1 / len(counts) / counts[r["patient_id"]],
            }
            for r in reports
        ],
        "aggregation": _WEIGHTING,
        "decomposition_identity": _IDENTITY,
    }
