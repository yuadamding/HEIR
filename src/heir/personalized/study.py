"""Fixed patient-held-out internal comparison of reference and correction pathways.

Validation selects only the prespecified RNA correction rank. Test predictions
are completed before test targets are loaded in each fold. Earlier cohort
exposure is retained; this workflow is not independent confirmation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from .cohort import discover_cohort, prepare_cohort_inputs
from .cohort_cli import (
    _config,
    _prediction_provenance,
    _workflow,
    comparison_scores,
    qualified_targets,
)
from .diagnostics import correction_diagnostics
from .reference import _cluster
from .schema import RNAReference
from .serialization import load_prediction, save_prediction
from .train import TrainingConfig, TrainingSample, fit, read_fold_roles

_ARMS = {
    "he": {"variant": "he"},
    "he_rna": {"variant": "he_rna"},
    "he_rna_protein": {"variant": "he_rna_protein"},
    "constant_reference": {"variant": "he_rna", "assignment_mode": "constant_reference"},
    "generic_reference": {"variant": "he_rna", "reference_source": "training_pool"},
    "rna_transfer_trained": {"variant": "he_rna", "transfer_only": True},
}
_SELECTION = "highest_equal_patient_validation_spearman_then_lower_mse_then_lower_rank"
_SCORE = "mean_replicate_median_gene_spearman"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def validate_folds(path, patient_ids, outer_folds=(0, 1, 2, 3, 4)):
    """Require disjoint roles, exact patient coverage, and one outer test per patient."""
    patients = set(patient_ids)
    if not patients or any(not isinstance(p, str) or not p.strip() for p in patients):
        raise ValueError("study requires explicit biological patient identifiers")
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {int(row["outer_fold"]) for row in rows} != set(outer_folds):
        raise ValueError("fold file must contain exactly the protocol outer folds")
    folds = {fold: read_fold_roles(path, fold) for fold in outer_folds}
    if any(set(roles) != patients for roles in folds.values()):
        raise ValueError("every outer fold must cover exactly the cohort patient IDs")
    tests = Counter(p for roles in folds.values() for p, role in roles.items() if role == "test")
    if tests != Counter({p: 1 for p in patients}):
        raise ValueError("each patient must occur in the test role exactly once")
    return folds


def _reference_hash(reference):
    digest = hashlib.sha256()
    for field in fields(RNAReference):
        value = getattr(reference, field.name)
        digest.update(field.name.encode())
        if isinstance(value, np.ndarray):
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(value.shape).encode())
            digest.update(value.tobytes(order="C"))
        else:
            digest.update(json.dumps(value, ensure_ascii=False).encode())
    return digest.hexdigest()


def pool_reference(inputs, training_patient_ids, *, gene_ids, embedding_gene_ids, max_states=32):
    """Pool training RNA states with equal patient mass, independent of all targets.

    Clustering is deterministic and unweighted over source-state log1p means;
    cluster means, total variances, and proportions use equal-patient weights.
    Multiple biopsies, if present, divide their patient's mass equally.
    """
    patients = tuple(sorted(training_patient_ids))
    if not patients or len(set(patients)) != len(patients):
        raise ValueError("generic reference requires unique training patient IDs")
    if {item.patient_id for item in inputs} != set(patients):
        raise ValueError("generic reference must use exactly the training patients")
    if len({(i.patient_id, i.specimen_id) for i in inputs}) != len(inputs):
        raise ValueError("generic reference contains duplicate training specimens")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1:
        raise ValueError("generic max_states must be a positive integer")
    panel = tuple(embedding_gene_ids)
    genes = tuple(dict.fromkeys((*gene_ids, *panel)))
    if not panel or len(set(panel)) != len(panel) or not genes:
        raise ValueError("generic reference requires fixed output and embedding gene axes")
    counts = Counter(item.patient_id for item in inputs)
    means, variances, masses, cell_counts, sources = [], [], [], [], []
    for item in sorted(inputs, key=lambda i: (i.patient_id, i.specimen_id)):
        reference = item.reference
        if not set(genes) <= set(reference.gene_ids):
            raise ValueError("generic reference genes missing from a training RNA source")
        columns = [reference.gene_ids.index(gene) for gene in genes]
        means.append(np.asarray(reference.expression[:, columns], dtype=np.float64))
        variances.append(np.asarray(reference.variance[:, columns], dtype=np.float64))
        weight = np.asarray(reference.proportions, dtype=np.float64)
        masses.append(weight / weight.sum() / len(patients) / counts[item.patient_id])
        cell_counts.append(np.asarray(reference.cell_counts, dtype=np.int64))
        sources.append(
            {
                "patient_id": item.patient_id,
                "specimen_id": item.specimen_id,
                "reference_sha256": _reference_hash(reference),
                "source_states": len(reference.state_ids),
                "source_cells": int(reference.cell_counts.sum()),
            }
        )
    means, variances = np.concatenate(means), np.concatenate(variances)
    masses, cell_counts = np.concatenate(masses), np.concatenate(cell_counts)
    panel_columns = [genes.index(gene) for gene in panel]
    labels = _cluster(np.log1p(means[:, panel_columns]), min(max_states, len(means)))
    pooled_means, pooled_variances, pooled_masses, pooled_counts = [], [], [], []
    for label in sorted(set(labels)):
        keep = labels == label
        mass = masses[keep].sum()
        mean = np.average(means[keep], weights=masses[keep], axis=0)
        variance = (
            np.average(variances[keep] + means[keep] ** 2, weights=masses[keep], axis=0) - mean**2
        )
        pooled_means.append(mean)
        pooled_variances.append(np.maximum(variance, 0))
        pooled_masses.append(mass)
        pooled_counts.append(int(cell_counts[keep].sum()))
    expression = np.asarray(pooled_means, dtype=np.float32)
    n_states = len(expression)
    pooled = RNAReference(
        state_ids=tuple(f"training_pool::{i}" for i in range(n_states)),
        gene_ids=genes,
        expression=expression,
        embedding=np.log1p(expression[:, panel_columns]),
        proportions=np.asarray(pooled_masses, dtype=np.float32),
        state_types=("training_pool",) * n_states,
        variance=np.asarray(pooled_variances, dtype=np.float32),
        cell_counts=np.asarray(pooled_counts, dtype=np.int64),
        embedding_gene_ids=panel,
    )
    metadata = {
        "schema": "heir.training_reference_pool.v1",
        "filename": "pooled_reference.npz",
        "training_patient_ids": list(patients),
        "source_references": sources,
        "weighting": "equal_patient_then_equal_biopsy_then_source_reference_proportion",
        "clustering": "deterministic_unweighted_source_state_log1p_mean_embedding",
        "variance": "weighted_total_variance_within_plus_between_source_states",
        "cell_counts": "summed_source_cells_diagnostic_not_pool_weights",
        "gene_ids": list(genes),
        "embedding_gene_ids": list(panel),
        "max_states": max_states,
        "n_source_states": len(means),
        "n_pooled_states": n_states,
        "target_expression_used": False,
        "reference_sha256": _reference_hash(pooled),
    }
    return pooled, metadata


def _save_pool(bundle, reference, metadata):
    path = Path(bundle) / metadata["filename"]
    payload = {
        f.name: np.asarray(getattr(reference, f.name), dtype=str)
        if isinstance(getattr(reference, f.name), tuple)
        else getattr(reference, f.name)
        for f in fields(RNAReference)
    }
    with path.open("xb") as handle:
        np.savez(handle, **payload, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    inventory = Path(bundle) / "sha256.json"
    hashes = json.loads(inventory.read_text())
    hashes[path.name] = _sha(path)
    inventory.write_text(json.dumps(hashes, indent=2) + "\n")


def load_pooled_reference(bundle_dir) -> RNAReference:
    """Recover the training-only numeric reference bound into a generic model bundle."""
    bundle = Path(bundle_dir)
    hashes = json.loads((bundle / "sha256.json").read_text())
    for name in ("bundle.json", "pooled_reference.npz"):
        if name not in hashes or _sha(bundle / name) != hashes[name]:
            raise ValueError("generic reference bundle integrity mismatch")
    record = json.loads((bundle / "bundle.json").read_text())
    expected = record["preparation"].get("reference_pool")
    if not expected or sorted(record["training_patient_ids"]) != expected["training_patient_ids"]:
        raise ValueError("generic reference is not bound to the bundle training patients")
    with np.load(bundle / "pooled_reference.npz", allow_pickle=False) as archive:
        names = {f.name for f in fields(RNAReference)}
        if set(archive.files) != names | {"metadata_json"}:
            raise ValueError("generic reference archive has unexpected arrays")
        metadata = json.loads(archive["metadata_json"].item())
        arguments = {
            name: tuple(archive[name].tolist())
            if archive[name].dtype.kind == "U"
            else archive[name].copy()
            for name in names
        }
        reference = RNAReference(**arguments)
    if metadata != expected or _reference_hash(reference) != expected["reference_sha256"]:
        raise ValueError("generic reference contents differ from the fitted reference identity")
    return reference


def patient_summary(reports):
    """Equal specimens within patient, then equal patients; missing scores stay explicit."""
    if not reports or len({(r["patient_id"], r["specimen_id"]) for r in reports}) != len(reports):
        raise ValueError("patient summaries require nonempty, unique specimen reports")
    rows = []
    for patient in sorted({r["patient_id"] for r in reports}):
        selected = [r for r in reports if r["patient_id"] == patient]
        values, errors = [r[_SCORE] for r in selected], [r["normalized_log_mse"] for r in selected]
        valid = all(v is not None and np.isfinite(v) for v in values)
        valid_error = all(v is not None and np.isfinite(v) for v in errors)
        rows.append(
            {
                "patient_id": patient,
                "n_specimens": len(selected),
                "spearman": float(np.mean(values)) if valid else None,
                "normalized_log_mse": float(np.mean(errors)) if valid_error else None,
                "undefined_specimens": [r["specimen_id"] for r in selected if r[_SCORE] is None],
            }
        )
    values, errors = [r["spearman"] for r in rows], [r["normalized_log_mse"] for r in rows]
    defined = [v for v in values if v is not None]
    return {
        "mean_patient_spearman": float(np.mean(values)) if len(defined) == len(rows) else None,
        "defined_patient_mean_diagnostic": float(np.mean(defined)) if defined else None,
        "mean_patient_normalized_log_mse": float(np.mean(errors))
        if all(v is not None for v in errors)
        else None,
        "n_patients": len(rows),
        "n_defined_patients": len(defined),
        "patients": rows,
        "aggregation": "equal_patient_mean_of_equal_specimen_equal_replicate_spatial_summaries",
    }


def select_rank(validation_scores, validation_patient_ids):
    """Select on complete equal-patient validation scores; never consume test reports."""
    expected = set(validation_patient_ids)
    candidates = []
    for rank, reports in sorted(validation_scores.items()):
        if {r["patient_id"] for r in reports} != expected:
            raise ValueError("rank selection may use exactly the validation patients")
        summary = patient_summary(reports)
        score, error = summary["mean_patient_spearman"], summary["mean_patient_normalized_log_mse"]
        candidates.append(
            {"rank": rank, "eligible": score is not None and error is not None, **summary}
        )
    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        raise ValueError("no rank has fully defined patient-level validation scores")
    selected = min(
        eligible,
        key=lambda c: (
            -c["mean_patient_spearman"],
            c["mean_patient_normalized_log_mse"],
            c["rank"],
        ),
    )
    return selected["rank"], {
        "selection": _SELECTION,
        "selected_rank": selected["rank"],
        "candidates": candidates,
    }


def paired_patient_contrasts(arm_reports, contrasts):
    """Report paired patient differences without pooling regions or hiding undefined arms."""
    summaries = {
        arm: {r["patient_id"]: r for r in patient_summary(reports)["patients"]}
        for arm, reports in arm_reports.items()
    }
    result = {}
    for contrast in contrasts:
        left, right = contrast.split("-minus-")
        if (
            left not in summaries
            or right not in summaries
            or set(summaries[left]) != set(summaries[right])
        ):
            raise ValueError("paired contrasts require the same patients in both declared arms")
        rows = []
        for patient in sorted(summaries[left]):
            a, b = summaries[left][patient], summaries[right][patient]
            row = {"patient_id": patient}
            for metric in ("spearman", "normalized_log_mse"):
                row[f"delta_{metric}"] = (
                    a[metric] - b[metric]
                    if a[metric] is not None and b[metric] is not None
                    else None
                )
            rows.append(row)
        valid = [r["delta_spearman"] for r in rows if r["delta_spearman"] is not None]
        errors = [
            r["delta_normalized_log_mse"] for r in rows if r["delta_normalized_log_mse"] is not None
        ]
        result[contrast] = {
            "patients": rows,
            "n_patients": len(rows),
            "n_defined_spearman_pairs": len(valid),
            "mean_paired_spearman_difference": float(np.mean(valid))
            if len(valid) == len(rows)
            else None,
            "defined_pairs_mean_diagnostic": float(np.mean(valid)) if valid else None,
            "mean_paired_normalized_log_mse_difference": float(np.mean(errors))
            if len(errors) == len(rows)
            else None,
            "n_positive_defined_spearman_pairs": sum(v > 0 for v in valid),
            "undefined_patient_ids": [r["patient_id"] for r in rows if r["delta_spearman"] is None],
            "interpretation": "positive Spearman and negative MSE differences favor the first arm",
        }
    return result


def _read_protocol(config_path, folds_path, protocol_path):
    protocol = json.loads(Path(protocol_path).read_text())
    if (
        protocol.get("schema") != "heir.internal_patient_study.v1"
        or protocol.get("outer_folds") != [0, 1, 2, 3, 4]
        or protocol.get("ranks") != [2, 16]
        or protocol.get("arms") != _ARMS
        or protocol.get("selection") != _SELECTION
        or protocol.get("prior_cohort_exposure") is not True
        or protocol.get("independent_confirmation") is not False
        or protocol.get("test_prediction_before_target_loading") is not True
        or protocol.get("no_train_validation_refit") is not True
    ):
        raise ValueError("unsupported fixed internal patient-study protocol")
    for key, path in (("config_sha256", config_path), ("folds_sha256", folds_path)):
        if protocol.get(key) != _sha(path):
            raise ValueError(f"frozen study {key} binding differs")
    if "metadata_review_sha256" in protocol:
        path = Path(protocol_path).parent / "metadata-review.json"
        if _sha(path) != protocol["metadata_review_sha256"]:
            raise ValueError("frozen metadata review binding differs")
    config, model = _config(config_path)
    if (
        tuple(protocol["outcome_gene_ids"]) != model.gene_ids
        or tuple(protocol["normalization_gene_ids"]) != model.normalization_gene_ids
    ):
        raise ValueError("study outcome or normalization panel differs from configuration")
    settings = TrainingConfig(**protocol["training"])
    if (
        settings.epochs != 100
        or settings.learning_rate != 0.001
        or settings.seed != 0
        or settings.abundance_weight != 0
    ):
        raise ValueError("study requires the frozen 100-epoch, seed-0 training settings")
    return protocol, config, model, settings


def run_study(root, config_path, folds_path, protocol_path, output):
    """Execute the fixed five-fold study, preserving all validation and test artifacts."""
    protocol, config, base, settings = _read_protocol(config_path, folds_path, protocol_path)
    cohort = discover_cohort(root)
    folds = validate_folds(
        folds_path, {r["patient_id"] for r in cohort.values()}, protocol["outer_folds"]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, source in (
        ("config.json", config_path),
        ("folds.csv", folds_path),
        ("protocol.json", protocol_path),
    ):
        (output / name).write_bytes(Path(source).read_bytes())
    if "metadata_review_sha256" in protocol:
        (output / "metadata-review.json").write_bytes(
            (Path(protocol_path).parent / "metadata-review.json").read_bytes()
        )
    package = Path(__file__).resolve().parents[1]
    _write(
        output / "identity.json",
        {
            "protocol_sha256": _sha(protocol_path),
            "config_sha256": _sha(config_path),
            "folds_sha256": _sha(folds_path),
            "runner_sha256": _sha(__file__),
            "source_file_sha256": {
                str(path.relative_to(package.parent)): _sha(path)
                for path in sorted(package.rglob("*.py"))
            },
            "prior_cohort_exposure": True,
            "independent_confirmation": False,
        },
    )
    prepared = {}
    for biopsy in sorted(cohort):
        print(f"study: preparing target-free inputs for {biopsy}", flush=True)
        prepared[biopsy] = prepare_cohort_inputs(
            root, biopsy, gene_ids=base.gene_ids, **config["preparation"]
        )
    all_reports = {arm: [] for arm in _ARMS}
    transfer_reports = {
        arm: [] for arm in ("he_rna", "he_rna_protein", "constant_reference", "generic_reference")
    }
    selections = []
    for fold, roles in folds.items():
        directory = output / f"fold-{fold}"
        directory.mkdir()
        role_biopsies = {
            role: [b for b in sorted(cohort) if roles[cohort[b]["patient_id"]] == role]
            for role in ("train", "validation", "test")
        }
        targets = {}
        for role in ("train", "validation"):
            for biopsy in role_biopsies[role]:
                inputs, provenance = prepared[biopsy]
                targets[biopsy] = qualified_targets(root, biopsy, inputs, provenance, base)
        samples = [
            TrainingSample(prepared[b][0], *targets[b][:2], targets[b][2])
            for b in role_biopsies["train"]
        ]
        preparation = {
            "cohort_workflow": _workflow(config),
            "fold_roles": roles,
            "outer_fold": fold,
            "protocol_sha256": _sha(protocol_path),
            "evaluation_role": "internal_patient_held_out",
            "training_sources": [
                {"inputs": prepared[b][1], "targets": targets[b][3]} for b in role_biopsies["train"]
            ],
        }
        candidates, validation = {}, {}
        for rank in protocol["ranks"]:
            print(f"study: fold {fold} validation candidate RNA rank {rank}", flush=True)
            current = replace(
                base,
                variant="he_rna",
                assignment_mode="learned",
                transfer_only=False,
                residual_rank=rank,
            )
            fitted = fit(samples, current, settings, preparation=preparation)
            fitted.save(directory / f"candidate-rank-{rank}.bundle")
            candidates[rank] = fitted
            validation[rank] = []
            for biopsy in role_biopsies["validation"]:
                inputs, provenance = prepared[biopsy]
                prediction = fitted.predict(inputs)
                save_prediction(
                    prediction,
                    directory / f"validation-rank-{rank}-{biopsy}.npz",
                    _prediction_provenance(fitted, provenance),
                )
                target, operator, groups, _ = targets[biopsy]
                report = comparison_scores(
                    prediction, target, operator, groups, base.gene_ids, base.normalization_gene_ids
                )
                validation[rank].append(report)
                _write(directory / f"validation-rank-{rank}-{biopsy}.scores.json", report)
        rank, selection = select_rank(
            validation, [p for p, role in roles.items() if role == "validation"]
        )
        selection.update(
            outer_fold=fold,
            fold_roles=roles,
            test_targets_loaded=False,
            refit_with_validation=False,
        )
        _write(directory / "selection.json", selection)
        selections.append(selection)
        pooled, pool_metadata = pool_reference(
            [sample.inputs for sample in samples],
            [p for p, role in roles.items() if role == "train"],
            gene_ids=base.gene_ids,
            embedding_gene_ids=base.state_gene_ids,
            max_states=protocol["max_reference_states"],
        )
        fitted_arms = {}
        for arm, overrides in protocol["arms"].items():
            print(f"study: fold {fold} arm {arm}, selected rank {rank}", flush=True)
            changes = {k: v for k, v in overrides.items() if k != "reference_source"}
            current = replace(
                base,
                **{
                    "assignment_mode": "learned",
                    "transfer_only": False,
                    "residual_rank": rank,
                    **changes,
                },
            )
            generic = arm == "generic_reference"
            arm_preparation = {**preparation, "study_arm": arm, "selected_rank": rank}
            if generic:
                arm_preparation["reference_pool"] = pool_metadata
            training = (
                [
                    replace(s, inputs=replace(s.inputs, reference=pooled, proteins=None))
                    for s in samples
                ]
                if generic
                else samples
            )
            if arm == "he_rna":
                fitted = candidates[rank]
                fitted.preparation = arm_preparation
            else:
                fitted = fit(training, current, settings, preparation=arm_preparation)
            bundle = directory / f"{arm}.bundle"
            fitted.save(bundle)
            if generic:
                _save_pool(bundle, pooled, pool_metadata)
            fitted_arms[arm] = fitted
        # Complete every arm/biopsy prediction before the first test target read.
        for biopsy in role_biopsies["test"]:
            inputs, provenance = prepared[biopsy]
            for arm, fitted in fitted_arms.items():
                current_inputs = (
                    replace(inputs, reference=pooled, proteins=None)
                    if arm == "generic_reference"
                    else inputs
                )
                source = (
                    {**provenance, "rna_reference_control": pool_metadata}
                    if arm == "generic_reference"
                    else provenance
                )
                prediction = fitted.predict(current_inputs)
                provenance_record = {
                    **_prediction_provenance(fitted, source),
                    "study_arm": arm,
                    "test_targets_loaded": False,
                }
                save_prediction(
                    prediction, directory / f"test-{biopsy}-{arm}.npz", provenance_record
                )
                if arm in transfer_reports:
                    transfer = replace(
                        prediction,
                        expression=prediction.transfer,
                        log_correction=torch.zeros_like(prediction.log_correction),
                    )
                    save_prediction(
                        transfer,
                        directory / f"test-{biopsy}-{arm}.transfer.npz",
                        {**provenance_record, "postfit_transfer_ablation": True},
                    )
        _write(
            directory / "test_predictions_complete.json",
            {
                "test_biopsies": role_biopsies["test"],
                "arms": list(fitted_arms),
                "test_targets_loaded": False,
                "prediction_sha256": {p.name: _sha(p) for p in directory.glob("test-*.npz")},
            },
        )
        fold_reports = {arm: [] for arm in _ARMS}
        for biopsy in role_biopsies["test"]:
            print(f"study: fold {fold} evaluating frozen test predictions for {biopsy}", flush=True)
            inputs, provenance = prepared[biopsy]
            target, operator, groups, coverage = qualified_targets(
                root, biopsy, inputs, provenance, base
            )
            _write(directory / f"test-{biopsy}-coverage.json", coverage)
            for arm, fitted in fitted_arms.items():
                path = directory / f"test-{biopsy}-{arm}.npz"
                prediction, _ = load_prediction(path)
                report = comparison_scores(
                    prediction, target, operator, groups, base.gene_ids, base.normalization_gene_ids
                )
                report.update(
                    outer_fold=fold,
                    study_arm=arm,
                    evaluation_role="internal_patient_held_out",
                    held_out_from_fit=True,
                )
                report["active_trainable_parameters"] = (
                    fitted.model.active_trainable_parameter_count()
                )
                report["reference_state_count"] = len(prediction.state_ids) - 1
                if arm != "he":
                    report["correction_diagnostics"] = correction_diagnostics(
                        prediction,
                        target,
                        operator,
                        groups,
                        base.gene_ids,
                        base.normalization_gene_ids,
                        residual_bound=fitted.model.config.residual_bound,
                    )
                _write(path.with_suffix(".scores.json"), report)
                fold_reports[arm].append(report)
                all_reports[arm].append(report)
                if arm in transfer_reports:
                    transfer, _ = load_prediction(directory / f"test-{biopsy}-{arm}.transfer.npz")
                    diagnostic = comparison_scores(
                        transfer,
                        target,
                        operator,
                        groups,
                        base.gene_ids,
                        base.normalization_gene_ids,
                    )
                    diagnostic.update(
                        outer_fold=fold, study_arm=arm, postfit_transfer_ablation=True
                    )
                    _write(directory / f"test-{biopsy}-{arm}.transfer.scores.json", diagnostic)
                    transfer_reports[arm].append(diagnostic)
        _write(
            directory / "summary.json",
            {
                "arms": {a: patient_summary(r) for a, r in fold_reports.items()},
                "selected_rank": rank,
            },
        )
    summary = {
        "schema": "heir.internal_patient_study_result.v1",
        "protocol_sha256": _sha(protocol_path),
        "prior_cohort_exposure": True,
        "independent_confirmation": False,
        "scientific_qualification": False,
        "selections": selections,
        "arms": {arm: patient_summary(reports) for arm, reports in all_reports.items()},
        "postfit_transfer": {
            arm: patient_summary(reports) for arm, reports in transfer_reports.items()
        },
        "paired_patient_contrasts": paired_patient_contrasts(
            all_reports, protocol["paired_contrasts"]
        ),
        "normalization_gene_ids": list(base.normalization_gene_ids),
        "outcome_gene_ids": list(base.gene_ids),
        "interpretation": (
            "internal patient-held-out assessment on a previously exposed cohort; "
            "no independent confirmation"
        ),
    }
    _write(output / "summary.json", summary)
    print("study: completed all five patient folds", flush=True)
    return summary
