"""Post-fit regional diagnostics with fixed evaluation rows and normalization.

Protein interventions reuse one fitted model. They are prediction interventions,
not separately trained ablations or evidence of held-out generalization.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .cohort import _restore_grid, _verified_source_path, prepare_cohort_inputs
from .cohort_cli import (
    _config,
    _subset,
    qualified_targets,
    replicate_scores,
)
from .evaluate import _ranks, normalized_observations
from .inspection import _coordinates
from .serialization import load_prediction
from .train import FittedModel

_VARIANTS = ("he", "he_rna", "he_rna.transfer", "he_rna_protein", "he_rna_protein.transfer")


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _summary(values):
    values = np.asarray(values)
    if not values.size:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _region_rows(operator):
    """Require the cohort's single-region observation geometry, in row order."""
    n = len(operator.observation_ids)
    order = np.argsort(operator.rows)
    if (
        len(operator.rows) != n
        or not np.array_equal(operator.rows[order], np.arange(n))
        or not np.allclose(operator.weights, 1, rtol=0, atol=1e-7)
    ):
        raise ValueError("regional diagnostics require one unit-weight region per observation")
    return operator.columns[order]


def _normalized(prediction, targets, operator, genes, normal):
    return tuple(
        _array(x)
        for x in normalized_observations(
            prediction, targets, operator, genes, normalization_gene_ids=normal
        )
    )


def _effects(difference, mask, groups, genes):
    rows = []
    groups = np.asarray(groups)
    for group in sorted(set(groups)):
        for j, gene in enumerate(genes):
            values = difference[(groups == group) & mask[:, j], j]
            rows.append(
                {
                    "assay_replicate": str(group),
                    "gene_id": gene,
                    "n_region_replicates": int(len(values)),
                    "mean_signed": float(values.mean()) if len(values) else None,
                    "mean_absolute": float(np.abs(values).mean()) if len(values) else None,
                    "root_mean_square": float(np.sqrt(np.mean(values**2))) if len(values) else None,
                    "max_absolute": float(np.abs(values).max()) if len(values) else None,
                }
            )
    return rows


def correction_diagnostics(
    prediction, targets, operator, groups, gene_ids, normalization_gene_ids, *, residual_bound
):
    """Summarize effective normalized correction and raw clamp usage, without fitting.

    Raw correction scopes distinguish all H&E regions, unique scored regions,
    and each scored replicate. Normalized effects retain region-replicate rows.
    """
    if not np.isfinite(residual_bound) or residual_bound < 0:
        raise ValueError("residual_bound must be finite and nonnegative")
    if len(groups) != len(operator.observation_ids):
        raise ValueError("replicate labels must match observation rows")
    full, _, mask = _normalized(prediction, targets, operator, gene_ids, normalization_gene_ids)
    transfer, _, _ = _normalized(
        replace(prediction, expression=prediction.transfer),
        targets,
        operator,
        gene_ids,
        normalization_gene_ids,
    )
    region_rows = _region_rows(operator)
    scopes = {
        "all_he_regions": np.arange(len(prediction.cell_ids)),
        "unique_scored_regions": np.unique(region_rows),
        **{
            f"scored_replicate:{group}": region_rows[np.asarray(groups) == group]
            for group in sorted(set(groups))
        },
    }
    raw = _array(prediction.log_correction)
    bounds = []
    for scope, indices in scopes.items():
        for gene in gene_ids:
            values = raw[indices, prediction.gene_ids.index(gene)]
            bounds.append(
                {
                    "scope": scope,
                    "gene_id": gene,
                    "n_regions": int(len(values)),
                    "mean_absolute_raw_log_correction": float(np.abs(values).mean()),
                    "positive_bound_fraction": float(
                        np.isclose(values, residual_bound, atol=1e-7, rtol=0).mean()
                    )
                    if residual_bound
                    else None,
                    "negative_bound_fraction": float(
                        np.isclose(values, -residual_bound, atol=1e-7, rtol=0).mean()
                    )
                    if residual_bound
                    else None,
                    "near_either_bound_fraction": float(
                        (np.abs(values) >= 0.95 * residual_bound).mean()
                    )
                    if residual_bound
                    else None,
                }
            )
    return {
        "normalization_gene_ids": list(normalization_gene_ids),
        "effective_correction": "fixed_panel_normalized_full_minus_normalized_transfer",
        "normalized_effects": _effects(full - transfer, mask, groups, gene_ids),
        "raw_correction": bounds,
        "residual_bound": residual_bound,
        "exact_bound_absolute_tolerance": 1e-7,
        "near_bound_fraction_of_limit": 0.95,
        "zero_bound_policy": "saturation_undefined_when_bound_is_zero",
        "scored_scope": (
            "retained_target_region_replicates; raw_unique_scope_deduplicates_replicates"
        ),
    }


def replicate_agreement(prediction, targets, operator, groups, gene_ids, normalization_gene_ids):
    """Compare target replicates on their shared eligible regions and fixed panel."""
    _, truth, mask = _normalized(prediction, targets, operator, gene_ids, normalization_gene_ids)
    columns, groups = _region_rows(operator), np.asarray(groups)
    rows = []
    for first, second in itertools.combinations(sorted(set(groups)), 2):
        left = {int(columns[i]): i for i in np.flatnonzero(groups == first)}
        right = {int(columns[i]): i for i in np.flatnonzero(groups == second)}
        if len(left) != (groups == first).sum() or len(right) != (groups == second).sum():
            raise ValueError("duplicate region within one assay replicate")
        common = sorted(left.keys() & right.keys())
        a, b = [left[i] for i in common], [right[i] for i in common]
        genes = []
        for j, gene in enumerate(gene_ids):
            keep = mask[a, j] & mask[b, j]
            x, y = truth[a, j][keep], truth[b, j][keep]
            reason = (
                "insufficient_shared_regions"
                if len(x) < 3
                else "constant_replicate_target"
                if np.ptp(x) == 0 or np.ptp(y) == 0
                else None
            )
            genes.append(
                {
                    "gene_id": gene,
                    "n_shared_eligible_regions": int(len(x)),
                    "spearman": None if reason else float(np.corrcoef(_ranks(x), _ranks(y))[0, 1]),
                    "normalized_log_mse": float(np.mean((x - y) ** 2)) if len(x) else None,
                    "undefined_reason": reason,
                }
            )
        rows.append(
            {
                "replicates": [str(first), str(second)],
                "shared_eligible_region_ids": [prediction.cell_ids[i] for i in common],
                "n_shared_eligible_regions": len(common),
                "gene_scores": genes,
            }
        )
    return {
        "normalization_gene_ids": list(normalization_gene_ids),
        "pairs": rows,
        "interpretation": "target consistency reference; not a universal performance ceiling",
    }


def permute_protein_values(protein, *, seed=0):
    """Permute whole marker vectors within replicate and availability-pattern groups.

    Marker identity, missingness, geometry, and quality remain at their original
    locations. Singleton groups remain fixed. Donor identities make the exact
    permutation auditable, including random fixed points in movable groups.
    """
    rng = np.random.default_rng(seed)
    groups = {}
    for i, observation in enumerate(protein.operator.observation_ids):
        parts = observation.split(":", 2)
        if len(parts) != 3 or parts[0] != "codex":
            raise ValueError("protein intervention requires cohort codex replicate identities")
        groups.setdefault((parts[1], tuple(protein.observed[i])), []).append(i)
    donors = np.arange(len(protein.values))
    movable = 0
    for indices in groups.values():
        if len(indices) > 1:
            movable += len(indices)
            donors[indices] = rng.permutation(indices)
    return replace(protein, values=protein.values[donors].copy()), {
        "seed": seed,
        "policy": "whole_marker_rows_within_codex_replicate_and_observed_marker_pattern",
        "quality_policy": "unchanged_at_original_location",
        "n_movable_rows": movable,
        "n_moved_rows": int((donors != np.arange(len(donors))).sum()),
        "donor_observation_ids": [protein.operator.observation_ids[i] for i in donors],
        "destination_observation_ids": list(protein.operator.observation_ids),
    }


def _coverage(inputs, provenance, root, targets, operator, groups, target_record, threshold):
    grid = _restore_grid(inputs, provenance)
    source = provenance["sources"]["protein"]
    _, replicates, mapped, _ = _coordinates(_verified_source_path(root, source), source, grid)
    counts = {
        str(rep): np.bincount(
            mapped[(replicates == rep) & (mapped >= 0)], minlength=len(grid.region_ids)
        )
        for rep in sorted(set(replicates))
    }
    protein = inputs.proteins
    complete = {rep: np.zeros(len(grid.region_ids), dtype=bool) for rep in counts}
    usable = {rep: np.zeros(len(grid.region_ids), dtype=bool) for rep in counts}
    for row, column in enumerate(_region_rows(protein.operator)):
        rep = protein.operator.observation_ids[row].split(":", 2)[1]
        complete[rep][column] = protein.observed[row].all()
        quality = (
            np.ones(len(protein.marker_ids)) if protein.quality is None else protein.quality[row]
        )
        usable[rep][column] = (protein.observed[row] & (quality > 0)).all()
    # The metadata bead count axis precedes exposure filtering; join by exact IDs.
    raw_ids = target_record.get("all_observation_ids")
    if raw_ids is None:
        # The caller explicitly provides the unfiltered axis before qualification.
        raise ValueError("diagnostic coverage requires the original target observation axis")
    if (
        len(raw_ids) != len(target_record["region_bead_counts"])
        or [name.split(":", 2)[1] for name in raw_ids] != target_record["observation_replicates"]
    ):
        raise ValueError("target count metadata is not aligned to observation identities")
    bead_counts = dict(zip(raw_ids, target_record["region_bead_counts"]))
    rows = []
    for i, column in enumerate(_region_rows(operator)):
        per_rep = {rep: int(value[column]) for rep, value in counts.items()}
        fraction = float(grid.tissue_fraction[column])
        rows.append(
            {
                "observation_id": targets.observation_ids[i],
                "region_id": grid.region_ids[column],
                "assay_replicate": str(groups[i]),
                "codex_covered": any(value > 0 for value in per_rep.values()),
                "all_markers_observed": any(value[column] for value in complete.values()),
                "all_markers_usable": any(value[column] for value in usable.values()),
                "protein_source_cell_count": sum(per_rep.values()),
                "protein_cells_by_replicate": per_rep,
                "target_bead_count": int(bead_counts[targets.observation_ids[i]]),
                "tissue_fraction": fraction,
                "tissue_subset": "interior" if fraction >= threshold else "partial_boundary",
            }
        )
    summaries = []
    for rep in sorted(set(groups)):
        subset = [row for row in rows if row["assay_replicate"] == rep]
        summaries.append(
            {
                "assay_replicate": str(rep),
                "n_scored_regions": len(subset),
                **{
                    key: sum(bool(row[key]) for row in subset)
                    for key in ("codex_covered", "all_markers_observed", "all_markers_usable")
                },
                **{
                    key: _summary([row[key] for row in subset])
                    for key in ("protein_source_cell_count", "target_bead_count", "tissue_fraction")
                },
            }
        )
    return rows, summaries


def _change(original, changed, region_rows):
    result = {}
    for scope, rows in {
        "all_he_regions": np.arange(len(original.cell_ids)),
        "unique_scored_regions": np.unique(region_rows),
    }.items():
        current = {}
        for name in ("conditional_assignment", "transfer", "expression", "log_correction"):
            difference = (
                _array(getattr(changed, name))[rows] - _array(getattr(original, name))[rows]
            )
            current[name] = {
                "mean_absolute_entry_change": float(np.abs(difference).mean()),
                "max_absolute_entry_change": float(np.abs(difference).max()),
                "mean_row_l1_change": float(np.abs(difference).sum(1).mean()),
                "fraction_rows_changed_at_1e-8": float((np.abs(difference).max(1) > 1e-8).mean()),
            }
        result[scope] = current
    return result


def _scores(predictions, targets, operator, groups, genes, normal, keep=None):
    if keep is not None:
        if not np.any(keep):
            return {"status": "no_regions", "scores": {}}
        targets, operator = _subset(targets, operator, keep)
        groups = tuple(np.asarray(groups)[keep])
    return {
        "status": "reported",
        "n_region_replicates": len(targets.observation_ids),
        "scores": {
            name: replicate_scores(prediction, targets, operator, groups, genes, normal)
            for name, prediction in predictions.items()
        },
    }


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_pilot_diagnostics(
    dataset_root,
    run_directory,
    output_directory,
    *,
    permutation_seed=0,
    interior_tissue_fraction=0.9,
):
    """Analyze a saved five-arm cohort demo without fitting or changing sources."""
    root, run, output = (Path(p).resolve() for p in (dataset_root, run_directory, output_directory))
    if output.is_relative_to(root) or output.is_relative_to(run):
        raise ValueError("diagnostic output must be outside immutable dataset and input run")
    if output.exists():
        raise FileExistsError(output)
    if not 0 < interior_tissue_fraction <= 1:
        raise ValueError("interior tissue fraction must be in (0, 1]")
    config, model = _config(run / "config.json")
    run_record = json.loads((run / "run.json").read_text())
    biopsy = run_record["biopsy_id"]
    inputs, provenance = prepare_cohort_inputs(
        root, biopsy, gene_ids=model.gene_ids, **config["preparation"]
    )
    original_preparation = json.loads((run / "input_preparation.json").read_text())
    for key in ("sources", "grid", "image_features", "rna", "protein"):
        if provenance[key] != original_preparation[key]:
            raise ValueError(f"prepared {key} differs from the original fitted run")
    predictions, prediction_provenance = {}, {}
    for name in _VARIANTS:
        prediction, metadata = load_prediction(run / f"{name}.npz")
        if (
            prediction.cell_ids != inputs.cell_ids
            or prediction.gene_ids != model.gene_ids
            or prediction.patient_id != inputs.patient_id
            or prediction.specimen_id != inputs.specimen_id
            or not np.array_equal(_array(prediction.coordinates), inputs.coordinates)
            or metadata["cohort_inputs"] != original_preparation
        ):
            raise ValueError("prediction axes or preparation differ across diagnostic inputs")
        predictions[name], prediction_provenance[name] = prediction, metadata
    reference = predictions["he_rna_protein"]
    targets, operator, groups, target_record = qualified_targets(
        root, biopsy, reference, provenance, model
    )
    # The loader preserves its pre-filter row order as replicate then region index.
    retained = set(targets.observation_ids)
    all_ids = retained | set(target_record["excluded_zero_or_incomplete_exposure"])
    target_record["all_observation_ids"] = sorted(
        all_ids,
        key=lambda name: (name.split(":", 2)[1], inputs.cell_ids.index(name.split(":", 2)[2])),
    )
    previous_targets = json.loads((run / "target_coverage.json").read_text())
    for key in ("source", "retained_observation_ids", "excluded_zero_or_incomplete_exposure"):
        if target_record[key] != previous_targets[key]:
            raise ValueError("target identity or evaluation eligibility differs from original run")
    coverage_rows, coverage_summary = _coverage(
        inputs, provenance, root, targets, operator, groups, target_record, interior_tissue_fraction
    )
    genes, normal = model.gene_ids, model.normalization_gene_ids
    masks = {
        "common_complete_codex": np.array([row["all_markers_usable"] for row in coverage_rows]),
        "interior": np.array([row["tissue_subset"] == "interior" for row in coverage_rows]),
        "partial_boundary": np.array(
            [row["tissue_subset"] == "partial_boundary" for row in coverage_rows]
        ),
    }
    corrections = {
        name: correction_diagnostics(
            predictions[name],
            targets,
            operator,
            groups,
            genes,
            normal,
            residual_bound=prediction_provenance[name]["model_config"]["residual_bound"],
        )
        for name in ("he_rna", "he_rna_protein")
    }
    fitted = FittedModel.load(run / "he_rna_protein.bundle")
    if fitted.model.config.variant != "he_rna_protein" or fitted.model.config.transfer_only:
        raise ValueError("protein interventions require a fitted full protein baseline")
    # Hash the persisted config, so newly introduced default-only fields do not
    # change the identity of a historical bundle. The bundle loader validates it.
    stored_config = json.loads((run / "he_rna_protein.bundle/bundle.json").read_text())[
        "model_config"
    ]
    if stored_config != prediction_provenance["he_rna_protein"]["model_config"]:
        raise ValueError("protein bundle config differs from the original prediction")
    identity = hashlib.sha256(json.dumps(stored_config, sort_keys=True).encode())
    for key, value in sorted(fitted.model.state_dict().items()):
        identity.update(key.encode())
        identity.update(_array(value).tobytes())
    identity.update(fitted.feature_mean.tobytes())
    identity.update(fitted.feature_scale.tobytes())
    if identity.hexdigest() != prediction_provenance["he_rna_protein"]["fitted_model_sha256"]:
        raise ValueError("protein bundle identity differs from the original prediction")
    original = fitted.predict(inputs)
    for name in ("conditional_assignment", "transfer", "expression", "log_correction"):
        if not np.allclose(
            _array(getattr(original, name)), _array(getattr(reference, name)), rtol=1e-6, atol=1e-6
        ):
            raise ValueError("fitted protein replay does not reproduce the stored prediction")
    shuffled, permutation = permute_protein_values(inputs.proteins, seed=permutation_seed)
    interventions = {
        "original": inputs.proteins,
        "zero_confidence": replace(inputs.proteins, quality=np.zeros_like(inputs.proteins.values)),
        "permuted_values": shuffled,
    }
    baseline_full, _, mask = _normalized(original, targets, operator, genes, normal)
    baseline_transfer, _, _ = _normalized(
        replace(original, expression=original.transfer), targets, operator, genes, normal
    )
    intervention_reports = {}
    for name, evidence in interventions.items():
        prediction = (
            original if name == "original" else fitted.predict(replace(inputs, proteins=evidence))
        )
        transfer = replace(prediction, expression=prediction.transfer)
        full_norm, _, _ = _normalized(prediction, targets, operator, genes, normal)
        transfer_norm, _, _ = _normalized(transfer, targets, operator, genes, normal)
        intervention_reports[name] = {
            "prediction_changes": _change(original, prediction, _region_rows(operator)),
            "full_scores": replicate_scores(prediction, targets, operator, groups, genes, normal),
            "transfer_scores": replicate_scores(transfer, targets, operator, groups, genes, normal),
            "normalized_full_change": _effects(full_norm - baseline_full, mask, groups, genes),
            "normalized_transfer_change": _effects(
                transfer_norm - baseline_transfer, mask, groups, genes
            ),
        }
    report = {
        "schema": "heir.pilot_diagnostics.v1",
        "biopsy_id": biopsy,
        "evaluation_role": "development_in_sample",
        "scientific_qualification": False,
        "new_fitting_performed": False,
        "prediction_unit": reference.prediction_unit,
        "coordinate_units": reference.coordinate_units,
        "evaluation_gene_ids": list(genes),
        "normalization_gene_ids": list(normal),
        "main_result": _scores(predictions, targets, operator, groups, genes, normal),
        "coverage": coverage_summary,
        "diagnostic_subsets": {
            name: _scores(predictions, targets, operator, groups, genes, normal, keep)
            for name, keep in masks.items()
        },
        "subset_policy": {
            "interior_min_tissue_fraction": interior_tissue_fraction,
            "partial_boundary": (
                "selected H&E regions below the declared interior fraction; "
                "occupancy proxy, not anatomical boundary certification"
            ),
            "main_result_unchanged": True,
        },
        "replicate_agreement": replicate_agreement(
            reference, targets, operator, groups, genes, normal
        ),
        "correction": corrections,
        "protein_interventions": {
            "interpretation": "fixed fitted-model prediction interventions; no refitting",
            "permutation": permutation,
            "conditions": intervention_reports,
        },
        "target_coverage": target_record,
        "input_identity": {
            "run_directory": str(run),
            "artifacts_sha256": {
                name: _digest(run / name)
                for name in (
                    "run.json",
                    "config.json",
                    "input_preparation.json",
                    "target_coverage.json",
                    *[f"{v}.npz" for v in _VARIANTS],
                    "he_rna_protein.bundle/sha256.json",
                )
            },
            "sources": provenance["sources"],
            "target_source": target_record["source"],
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "scored_regions.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "protein_cells_by_replicate": json.dumps(
                    row["protein_cells_by_replicate"], sort_keys=True
                ),
            }
            for row in coverage_rows
        )
    (output / "report.md").write_text(_markdown(report))
    return report


def _markdown(report):
    def table(title, headers, rows):
        lines = [
            "",
            f"## {title}",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            cells = [
                "undefined"
                if value is None
                else f"{value:.6g}"
                if isinstance(value, float)
                else str(value)
                for value in row
            ]
            lines.append("| " + " | ".join(cells) + " |")
        return lines

    score_keys = ("mean_replicate_median_gene_spearman", "normalized_log_mse")
    lines = [
        f"# {report['biopsy_id']} fixed-fit diagnostics",
        "",
        "Development fit with the original full scored coverage retained. No new fitting.",
    ]
    lines += table(
        "Main result",
        ["Prediction", "Spearman", "Normalized log MSE"],
        [
            [name, *[score[k] for k in score_keys]]
            for name, score in report["main_result"]["scores"].items()
        ],
    )
    lines += table(
        "Scored coverage",
        ["Replicate", "Scored", "CODEX", "Complete", "Usable"],
        [
            [
                r[k]
                for k in (
                    "assay_replicate",
                    "n_scored_regions",
                    "codex_covered",
                    "all_markers_observed",
                    "all_markers_usable",
                )
            ]
            for r in report["coverage"]
        ],
    )
    lines += table(
        "Source counts and tissue",
        ["Replicate", "Quantity", "Mean", "Median", "Min", "Max"],
        [
            [r["assay_replicate"], key, *[r[key][k] for k in ("mean", "median", "min", "max")]]
            for r in report["coverage"]
            for key in ("protein_source_cell_count", "target_bead_count", "tissue_fraction")
        ],
    )
    lines += [
        "",
        f"Interior tissue fraction ≥ {report['subset_policy']['interior_min_tissue_fraction']}; "
        "partial boundary is lower selected occupancy. This is not anatomical certification.",
    ]
    lines += table(
        "Diagnostic subsets",
        ["Subset", "Prediction", "Rows", "Spearman", "MSE"],
        [
            [subset, name, content["n_region_replicates"], *[score[k] for k in score_keys]]
            for subset, content in report["diagnostic_subsets"].items()
            for name, score in content["scores"].items()
            if name in ("he_rna", "he_rna_protein")
        ],
    )
    lines += table(
        "Target replicate agreement",
        ["Replicates", "Gene", "Shared", "Spearman", "MSE"],
        [
            [
                " / ".join(pair["replicates"]),
                *[
                    r[k]
                    for k in (
                        "gene_id",
                        "n_shared_eligible_regions",
                        "spearman",
                        "normalized_log_mse",
                    )
                ],
            ]
            for pair in report["replicate_agreement"]["pairs"]
            for r in pair["gene_scores"]
        ],
    )
    lines += [
        "",
        "Replicate agreement is a measurement reference, not a universal ceiling. "
        "Correction effects below use the fixed-panel normalized full minus transfer.",
    ]
    lines += table(
        "Normalized correction effects",
        ["Model", "Replicate", "Gene", "Signed", "Absolute", "RMS"],
        [
            [
                name,
                *[
                    r[k]
                    for k in (
                        "assay_replicate",
                        "gene_id",
                        "mean_signed",
                        "mean_absolute",
                        "root_mean_square",
                    )
                ],
            ]
            for name, correction in report["correction"].items()
            for r in correction["normalized_effects"]
        ],
    )
    lines += [
        "",
        "Raw clamp equality: absolute tolerance 1e-7. Near bound: ≥95% of limit. "
        "Unique scored regions deduplicate assay replicates; all H&E includes unscored rows.",
    ]
    lines += table(
        "Raw correction saturation",
        ["Model", "Scope", "Gene", "N", "+Bound", "-Bound", "Near"],
        [
            [
                name,
                *[
                    r[k]
                    for k in (
                        "scope",
                        "gene_id",
                        "n_regions",
                        "positive_bound_fraction",
                        "negative_bound_fraction",
                        "near_either_bound_fraction",
                    )
                ],
            ]
            for name, correction in report["correction"].items()
            for r in correction["raw_correction"]
            if not r["scope"].startswith("scored_replicate:")
        ],
    )
    lines += [
        "",
        "Protein interventions reuse fitted weights. Whole marker vectors move only within "
        "replicate/availability groups; quality and missingness remain at their locations.",
    ]
    lines += table(
        "Protein interventions",
        [
            "Condition",
            "Scored mixture L1",
            "Full Spearman",
            "Full MSE",
            "Transfer Spearman",
            "Transfer MSE",
        ],
        [
            [
                name,
                r["prediction_changes"]["unique_scored_regions"]["conditional_assignment"][
                    "mean_row_l1_change"
                ],
                *[r[part][k] for part in ("full_scores", "transfer_scores") for k in score_keys],
            ]
            for name, r in report["protein_interventions"]["conditions"].items()
        ],
    )
    lines += [
        "",
        "Exact rows: `scored_regions.tsv`. Per-gene/replicate scores, permutation donors, "
        "prediction changes and source identities: `report.json`.",
    ]
    return "\n".join(lines) + "\n"
