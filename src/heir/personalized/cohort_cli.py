"""HTAPP tissue-region workflows using the published registered bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .cohort import (
    DEFAULT_EMBEDDING_GENES,
    DEFAULT_PROTEIN_MARKERS,
    _verified_source_path,
    discover_cohort,
    load_cohort_targets,
    prepare_cohort_inputs,
)
from .evaluate import score
from .model import ModelConfig
from .observation import ObservationOperator
from .regions import build_region_grid
from .schema import SpatialTargets
from .serialization import load_prediction, save_prediction
from .train import (
    FittedModel,
    TrainingConfig,
    TrainingSample,
    fit,
    make_patient_folds,
    read_fold_roles,
)

DEFAULT_OUTCOME_GENES = ("EPCAM", "KRT8", "KRT18", "KRT19", "VIM", "COL1A2", "DCN", "CD8A", "MKI67")


def _write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _peak_rss():
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )


def default_config():
    """Explicit small development panel, independent of spatial expression values."""
    embedding = DEFAULT_EMBEDDING_GENES
    genes = DEFAULT_OUTCOME_GENES
    return {
        "schema": "heir.htapp_region_workflow.v1",
        "preparation": {
            "region_size": 100,
            "min_tissue_fraction": 0.1,
            "embedding_gene_ids": list(embedding),
            "max_states": 32,
            "protein_cofactor": 100.0,
            "rna_cofactor": 10.0,
            "graph_radius": 150.0,
            "graph_max_neighbors": 8,
        },
        "model": ModelConfig(
            feature_dim=12,
            state_dim=len(embedding),
            gene_ids=genes,
            supervised_gene_ids=genes,
            normalization_gene_ids=genes,
            state_gene_ids=embedding,
            variant="he_rna_protein",
        ).to_dict(),
        "interpretation": "regional development baseline with handcrafted RGB control features",
    }


def _config(path):
    value = default_config() if path is None else json.loads(Path(path).read_text())
    if value.get("schema") != "heir.htapp_region_workflow.v1":
        raise ValueError("unsupported cohort workflow configuration")
    model = ModelConfig.from_dict(value["model"])
    if tuple(value["preparation"]["embedding_gene_ids"]) != model.state_gene_ids:
        raise ValueError("cohort reference embedding and model state gene axes differ")
    return value, model


def inspect_command(args):
    cohort = discover_cohort(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "cohort.json", cohort)
    _write(output / "config.json", default_config())
    folds = make_patient_folds([item["patient_id"] for item in cohort.values()])
    with (output / "folds.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["outer_fold", "patient_id", "role"])
        writer.writeheader()
        writer.writerows(folds)
    _write(
        output / "status.json",
        {
            "biopsies": len(cohort),
            "prediction_unit": "tissue_region",
            "coordinate_units": "registered_pixel",
            "fold_status": "deterministic_candidate_patient_folds_not_balanced_or_qualified",
            "target_expression_read": False,
            "scientific_qualification": False,
        },
    )


def features_command(args):
    """Cache one retained frozen encoder on H&E-only regions, without molecular payloads."""
    from heir.features import create_frozen_encoder, load_encoder_manifest
    from heir.features.base import verified_model_file

    from .region_features import write_feature_cache

    config, model = _config(args.config)
    manifest = load_encoder_manifest(args.encoder_manifest)
    if manifest.config_filename:
        verified_model_file(args.model_dir, manifest.config_filename, manifest.config_sha256)
    cohort = discover_cohort(args.root)
    if len(set(args.biopsies)) != len(args.biopsies) or not set(args.biopsies) <= set(cohort):
        raise ValueError("feature extraction requires unique known biopsy IDs")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    encoder = create_frozen_encoder(Path(args.model_dir), manifest, device=args.device)
    reports = {}
    for biopsy in args.biopsies:
        source = cohort[biopsy]["role_sources"]["he"]
        image = _verified_source_path(args.root, source)
        grid = build_region_grid(
            image,
            bin_size_pixels=config["preparation"]["region_size"],
            min_tissue_fraction=config["preparation"]["min_tissue_fraction"],
        )
        reports[biopsy] = write_feature_cache(
            image,
            grid,
            output / f"{biopsy}.npz",
            encoder,
            manifest,
            expected_image_sha256=source["sha256"],
            batch_size=args.batch_size,
        )
    config["preparation"].update(
        feature_cache_dir=str(output), feature_encoder_manifest_sha256=manifest.sha256
    )
    config["model"] = replace(model, feature_dim=manifest.feature_width).to_dict()
    config["interpretation"] = "regional development with retained frozen histology features"
    _write(output / "config.json", config)
    _write(output / "features.json", reports)


def inspect_pilot_command(args):
    """Build a visual evaluation-coverage report after target-independent region selection."""
    from .inspection import write_pilot_inspection

    config, model = _config(args.config)
    write_pilot_inspection(
        args.root,
        args.biopsy,
        args.output,
        preparation=config["preparation"],
        gene_ids=model.gene_ids,
        normalization_gene_ids=model.normalization_gene_ids,
    )


def _subset(targets, operator, keep):
    keep = np.asarray(keep, dtype=bool)
    old_rows = np.flatnonzero(keep)
    row_map = np.full(len(keep), -1, dtype=np.int64)
    row_map[old_rows] = np.arange(len(old_rows))
    selected = keep[operator.rows]
    ids = tuple(targets.observation_ids[i] for i in old_rows)
    return (
        SpatialTargets(
            observation_ids=ids,
            gene_ids=targets.gene_ids,
            expression=targets.expression[keep],
            observed=targets.observed[keep],
            patient_id=targets.patient_id,
            specimen_id=targets.specimen_id,
        ),
        ObservationOperator(
            cell_ids=operator.cell_ids,
            observation_ids=ids,
            rows=row_map[operator.rows[selected]],
            columns=operator.columns[selected],
            weights=operator.weights[selected],
        ),
    )


def qualified_targets(root, biopsy, prediction, provenance, model):
    """Apply one target-only exposure eligibility rule shared by every comparison."""
    targets, operator, metadata = load_cohort_targets(
        root, biopsy, prediction, provenance, gene_ids=model.gene_ids
    )
    if targets.observation_ids != operator.observation_ids:
        raise ValueError("cohort target rows differ from their observation operator")
    columns = [targets.gene_ids.index(g) for g in model.normalization_gene_ids]
    keep = targets.observed[:, columns].all(1) & (targets.expression[:, columns].sum(1) > 0)
    metadata["normalization_gene_ids"] = list(model.normalization_gene_ids)
    metadata["excluded_zero_or_incomplete_exposure"] = [
        name for name, included in zip(targets.observation_ids, keep) if not included
    ]
    if not keep.any():
        raise ValueError("no measured regions have complete positive fixed-panel exposure")
    groups = tuple(np.asarray(metadata["observation_replicates"])[keep].tolist())
    if not keep.all():
        targets, operator = _subset(targets, operator, keep)
    metadata["retained_observation_ids"] = list(targets.observation_ids)
    metadata["retained_observation_replicates"] = list(groups)
    return targets, operator, groups, metadata


def replicate_scores(prediction, targets, operator, groups, genes, normalization_genes):
    """Measure spatial variation within each assay replicate, then average summaries."""
    reports = []
    group_axis = np.asarray(groups)
    for group in sorted(set(groups)):
        target, mapping = _subset(targets, operator, group_axis == group)
        report = score(
            prediction,
            target,
            mapping,
            genes,
            normalization_gene_ids=normalization_genes,
            independent=False,
        )
        report["assay_replicate"] = group
        reports.append(report)
    invalid = any(
        r["headline_status"] == "undefined_prediction_on_evaluable_panel" for r in reports
    )
    defined = [r["median_gene_spearman"] for r in reports if r["median_gene_spearman"] is not None]
    errors = [r["normalized_log_mse"] for r in reports if r["normalized_log_mse"] is not None]
    return {
        "patient_id": prediction.patient_id,
        "specimen_id": prediction.specimen_id,
        "prediction_unit": prediction.prediction_unit,
        "coordinate_units": prediction.coordinate_units,
        "normalization_gene_ids": list(normalization_genes),
        "evaluation_gene_ids": list(genes),
        "mean_replicate_median_gene_spearman": None
        if invalid or not defined
        else float(np.mean(defined)),
        "headline_status": "undefined_prediction_on_evaluable_panel"
        if invalid
        else "evaluable"
        if defined
        else "no_evaluable_target_genes",
        "normalized_log_mse": float(np.mean(errors)) if errors else None,
        "aggregation": "equal_assay_replicate_mean_of_within_replicate_spatial_summaries",
        "replicate_scores": reports,
        "scientific_qualification": False,
        "interpretation": "regional prediction; no nucleus placement or communication claim",
    }


def comparison_scores(prediction, targets, operator, groups, genes, normalization_genes):
    """Partition outcomes by direct protein-marker overlap without changing exposure."""
    result = replicate_scores(prediction, targets, operator, groups, genes, normalization_genes)
    direct = {gene for _, gene in DEFAULT_PROTEIN_MARKERS}
    panels = {
        "direct_protein_marker_genes": [gene for gene in genes if gene in direct],
        "other_outcome_genes": [gene for gene in genes if gene not in direct],
    }
    result["outcome_groups"] = {
        name: {
            "gene_ids": panel,
            "status": "reported" if panel else "no_requested_outcomes_in_group",
            "scores": replicate_scores(
                prediction, targets, operator, groups, panel, normalization_genes
            )
            if panel
            else None,
        }
        for name, panel in panels.items()
    }
    result["outcome_group_interpretation"] = (
        "direct marker overlap versus other outcomes; biological independence is not implied"
    )
    return result


def _sample(root, biopsy, config, model):
    inputs, provenance = prepare_cohort_inputs(
        root, biopsy, gene_ids=model.gene_ids, **config["preparation"]
    )
    targets, operator, groups, target_record = qualified_targets(
        root, biopsy, inputs, provenance, model
    )
    return TrainingSample(inputs, targets, operator, groups), provenance, target_record


def _settings(args):
    # Equal-area tissue-region mixtures are not cell abundance estimates.
    return TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        abundance_weight=0.0,
        seed=args.seed,
        device=args.device,
    )


def _workflow(config):
    # Context variants must share preparation with their protein baseline.
    # Model configuration belongs in the fitted model, not this equality contract.
    return {"schema": config["schema"], "preparation": config["preparation"]}


def train_command(args):
    config, model = _config(args.config)
    roles = read_fold_roles(args.folds, args.outer_fold)
    cohort = discover_cohort(args.root)
    if set(roles) != {entry["patient_id"] for entry in cohort.values()}:
        raise ValueError("cohort folds must cover exactly the discovered biological patients")
    samples, sources = [], []
    for biopsy, entry in cohort.items():
        if roles[entry["patient_id"]] != "train":
            continue
        sample, provenance, target_record = _sample(args.root, biopsy, config, model)
        samples.append(sample)
        sources.append({"inputs": provenance, "targets": target_record})
    preparation = {
        "cohort_workflow": _workflow(config),
        "fold_roles": roles,
        "outer_fold": args.outer_fold,
        "training_sources": sources,
    }
    fitted = fit(
        samples,
        model,
        _settings(args),
        preparation=preparation,
        baseline=FittedModel.load(args.baseline) if args.baseline else None,
    )
    fitted.save(args.output)


def _prediction_provenance(fitted, sources):
    identity = hashlib.sha256()
    identity.update(json.dumps(fitted.model.config.to_dict(), sort_keys=True).encode())
    for key, value in sorted(fitted.model.state_dict().items()):
        identity.update(key.encode())
        identity.update(value.detach().cpu().numpy().tobytes())
    identity.update(fitted.feature_mean.tobytes())
    identity.update(fitted.feature_scale.tobytes())
    return {
        "cohort_inputs": sources,
        "biopsy_id": sources["biopsy_id"],
        "model_config": fitted.model.config.to_dict(),
        "fitted_model_sha256": identity.hexdigest(),
        "training_patient_ids": list(fitted.training_patient_ids),
        "fold_roles": fitted.preparation.get("fold_roles", {}),
        "target_payload_read": False,
        "prediction_unit": "tissue_region",
        "coordinate_units": "registered_pixel",
        "assignment_mode": fitted.model.config.assignment_mode,
        "reference_control": fitted.preparation.get("reference_pool"),
    }


def predict_command(args):
    fitted = FittedModel.load(args.bundle, device=args.device)
    config = fitted.preparation.get("cohort_workflow")
    if not config:
        raise ValueError("bundle was not fitted with the cohort region workflow")
    if args.transfer_only:
        if fitted.model.config.variant == "he":
            raise ValueError("H&E-only has no RNA transfer diagnostic")
        fitted.model.config = replace(fitted.model.config, transfer_only=True)
    inputs, sources = prepare_cohort_inputs(
        args.root, args.biopsy, gene_ids=fitted.model.config.gene_ids, **config["preparation"]
    )
    if fitted.preparation.get("reference_pool"):
        from .study import load_pooled_reference

        inputs = replace(inputs, reference=load_pooled_reference(args.bundle), proteins=None)
        sources["rna_reference_control"] = fitted.preparation["reference_pool"]
    prediction = fitted.predict(inputs)
    save_prediction(prediction, args.output, _prediction_provenance(fitted, sources))


def evaluate_command(args):
    prediction, provenance = load_prediction(args.prediction)
    model = ModelConfig.from_dict(provenance["model_config"])
    if prediction.gene_ids != model.gene_ids:
        raise ValueError("prediction gene axis differs from saved cohort model")
    sources = provenance["cohort_inputs"]
    targets, operator, groups, target_record = qualified_targets(
        args.root, sources["biopsy_id"], prediction, sources, model
    )
    genes = model.gene_ids if args.genes is None else json.loads(Path(args.genes).read_text())
    result = comparison_scores(
        prediction, targets, operator, groups, genes, model.normalization_gene_ids
    )
    role = provenance.get("fold_roles", {}).get(prediction.patient_id, "development")
    training = prediction.patient_id in provenance.get("training_patient_ids", [])
    if role in {"validation", "test"} and training:
        raise ValueError("held-out patient appears in the model training set")
    result.update(
        evaluation_role=role,
        held_out_from_fit=role in {"validation", "test"} and not training,
        annotation_policy="published_cohort_curated_rna_only_not_fold_fitted",
        target_coverage=target_record,
    )
    _write(args.output, result)


def demo_command(args):
    """One-biopsy, explicitly in-sample software acceptance; never a held-out result."""
    started = time.perf_counter()
    config, model = _config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sample, sources, target_record = _sample(args.root, args.biopsy, config, model)
    _write(output / "config.json", config)
    _write(output / "input_preparation.json", sources)
    _write(output / "target_coverage.json", target_record)
    summaries = {}
    for variant in ("he", "he_rna", "he_rna_protein"):
        current = replace(
            model,
            variant=variant,
            transfer_only=False,
            supervised_gene_ids=model.gene_ids if variant == "he" else model.supervised_gene_ids,
        )
        fitted = fit(
            [sample],
            current,
            _settings(args),
            preparation={
                "cohort_workflow": _workflow(config),
                "evaluation_role": "development_in_sample",
            },
        )
        fitted.save(output / f"{variant}.bundle")
        prediction = fitted.predict(sample.inputs)
        save_prediction(
            prediction, output / f"{variant}.npz", _prediction_provenance(fitted, sources)
        )
        result = comparison_scores(
            prediction,
            sample.targets,
            sample.observation,
            sample.observation_groups,
            current.gene_ids,
            current.normalization_gene_ids,
        )
        result["evaluation_role"] = "development_in_sample"
        _write(output / f"{variant}.scores.json", result)
        summaries[variant] = {
            "initial_training_loss": fitted.history[0]["loss"],
            "final_training_loss": fitted.history[-1]["loss"],
            "mean_replicate_median_gene_spearman": result["mean_replicate_median_gene_spearman"],
            "normalized_log_mse": result["normalized_log_mse"],
            "mean_absolute_log_correction": float(
                prediction.log_correction.detach().abs().mean().cpu()
            ),
        }
        if variant != "he":
            fitted.model.config = replace(current, transfer_only=True)
            transfer = fitted.predict(sample.inputs)
            save_prediction(
                transfer,
                output / f"{variant}.transfer.npz",
                _prediction_provenance(fitted, sources),
            )
            transfer_result = comparison_scores(
                transfer,
                sample.targets,
                sample.observation,
                sample.observation_groups,
                current.gene_ids,
                current.normalization_gene_ids,
            )
            transfer_result["evaluation_role"] = "development_in_sample"
            _write(output / f"{variant}.transfer.scores.json", transfer_result)
            summaries[f"{variant}_transfer"] = {
                "mean_replicate_median_gene_spearman": transfer_result[
                    "mean_replicate_median_gene_spearman"
                ],
                "normalized_log_mse": transfer_result["normalized_log_mse"],
                "mean_absolute_log_correction": 0.0,
            }
    _write(
        output / "run.json",
        {
            "biopsy_id": args.biopsy,
            "evaluation_role": "development_in_sample",
            "scientific_qualification": False,
            "prediction_unit": "tissue_region",
            "coordinate_units": "registered_pixel",
            "image_features": sources["image_features"],
            "n_regions": len(sample.inputs.cell_ids),
            "n_states": len(sample.inputs.reference.state_ids),
            "n_target_region_replicates": len(sample.targets.observation_ids),
            "n_genes": len(model.gene_ids),
            "epochs": args.epochs,
            "comparisons": summaries,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": _peak_rss(),
            "memory_scope": "whole_process_including_source_import_not_isolated_model_memory",
        },
    )


def controls_command(args):
    """Fit three predefined controls against an existing same-biopsy full RNA pilot."""
    config, model = _config(args.config)
    model = replace(model, variant="he_rna", transfer_only=False, assignment_mode="learned")
    baseline_path = Path(args.baseline_run)
    baseline = FittedModel.load(baseline_path / "he_rna.bundle", device=args.device)
    settings = _settings(args)
    if baseline.model.config != model or any(
        baseline.training_config.get(key) != value for key, value in asdict(settings).items()
    ):
        raise ValueError("control settings must match the existing full RNA pilot")
    if baseline.preparation.get("evaluation_role") != "development_in_sample":
        raise ValueError("controls require an explicitly in-sample pilot baseline")
    if baseline.preparation.get("cohort_workflow") != _workflow(config):
        raise ValueError("control preparation must match the existing pilot")
    sample, sources, target_record = _sample(args.root, args.biopsy, config, model)
    if list(baseline.training_patient_ids) != [sample.inputs.patient_id]:
        raise ValueError("control biopsy differs from the fitted pilot patient")
    if target_record != json.loads((baseline_path / "target_coverage.json").read_text()):
        raise ValueError("control target coverage differs from the existing pilot")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "config.json", config)
    _write(output / "input_preparation.json", sources)
    _write(output / "target_coverage.json", target_record)
    experiments = {
        "rna_transfer_trained": replace(model, transfer_only=True),
        "constant_reference": replace(model, assignment_mode="constant_reference"),
        "rna_rank2": replace(model, residual_rank=2),
    }
    results = {}
    for name, current in experiments.items():
        print(f"Fitting pilot control {name}", flush=True)
        fitted = fit(
            [sample],
            current,
            settings,
            preparation={
                "cohort_workflow": _workflow(config),
                "evaluation_role": "development_in_sample",
                "control": name,
                "constant_mixture_policy": "RNA_reference_proportions_not_true_tissue_abundance"
                if current.assignment_mode == "constant_reference"
                else None,
            },
        )
        fitted.save(output / f"{name}.bundle")
        for transfer in [False, True] if not current.transfer_only else [False]:
            fitted.model.config = replace(current, transfer_only=True) if transfer else current
            prediction = fitted.predict(sample.inputs)
            label = f"{name}.transfer" if transfer else name
            save_prediction(
                prediction, output / f"{label}.npz", _prediction_provenance(fitted, sources)
            )
            report = comparison_scores(
                prediction,
                sample.targets,
                sample.observation,
                sample.observation_groups,
                current.gene_ids,
                current.normalization_gene_ids,
            )
            report.update(
                evaluation_role="development_in_sample",
                control=name,
                postfit_transfer_ablation=transfer,
            )
            _write(output / f"{label}.scores.json", report)
            results[label] = {
                "spearman": report["mean_replicate_median_gene_spearman"],
                "normalized_log_mse": report["normalized_log_mse"],
                "headline_status": report["headline_status"],
                "active_trainable_parameters": fitted.model.active_trainable_parameter_count(),
                "mean_absolute_log_correction": float(prediction.log_correction.abs().mean()),
                "initial_training_loss": fitted.history[0]["loss"],
                "final_training_loss": fitted.history[-1]["loss"],
            }
    _write(
        output / "run.json",
        {
            "schema": "heir.pilot_controls.v1",
            "biopsy_id": args.biopsy,
            "evaluation_role": "development_in_sample",
            "scientific_qualification": False,
            "baseline_run": str(baseline_path.resolve()),
            "training": asdict(settings),
            "comparisons": results,
            "interpretation": "predefined mechanisms; no held-out or true-abundance claim",
        },
    )


def diagnostics_command(args):
    """Diagnose existing pilot predictions and fixed-model protein interventions."""
    from .diagnostics import write_pilot_diagnostics

    write_pilot_diagnostics(args.root, args.run_directory, args.output)


def study_command(args):
    """Execute a frozen internal patient-held-out protocol with validation-only selection."""
    from .study import run_study

    run_study(args.root, args.config, args.folds, args.protocol, args.output)


def capability_command(args):
    """Separate calibration, measurement consistency, and target-informed reference fitting."""
    from .capability import run_capability

    run_capability(args.root, args.study, args.protocol, args.output)


def measurement_command(args):
    """Qualify regional measurements without fitting models or loading image embeddings."""
    from .measurement_workflow import run_measurement

    run_measurement(args.root, args.study, args.capability, args.config, args.output)


def reconciliation_command(args):
    from .source_reconciliation import run_reconciliation

    run_reconciliation(args.root, args.pipeline, args.measurement, args.config, args.output)


def joint_command(args):
    """Prepare, fit, or diagnose the explicitly separate joint model family."""
    from .joint_workflow import (
        compare_backbones,
        diagnose_joint,
        pilot_joint,
        prepare_joint,
        study_joint,
    )

    if args.joint_action == "prepare":
        result = prepare_joint(args.root, args.config, args.folds, args.output)
    elif args.joint_action == "pilot":
        result = pilot_joint(args.prepared, args.biopsies, args.config, args.output)
    elif args.joint_action == "study":
        result = study_joint(args.prepared, args.config, args.folds, args.seeds, args.output)
    elif args.joint_action == "compare-backbones":
        result = compare_backbones(args.hoptimus_study, args.uni2_study, args.config, args.output)
    else:
        result = diagnose_joint(args.prepared, args.study, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


def probe_command(args):
    """Fit validation-selected direct readouts of the frozen full-width image features."""
    from .probe_workflow import run_feature_probes

    result = run_feature_probes(args.config, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


def residual_command(args):
    """Fit the fixed UNI2-h cross-fitted baseline plus molecular residual study."""
    from .residual_workflow import run_residual_study

    print(json.dumps(run_residual_study(args.config, args.output), indent=2, allow_nan=False))


def residual_diagnose_command(args):
    """Diagnose saved residual energy and baseline-error alignment without fitting."""
    from .residual_diagnostics import diagnose_residual_study

    print(json.dumps(diagnose_residual_study(args.study, args.output), indent=2, allow_nan=False))


def endpoint_command(args):
    """Freeze orthogonal regional RNA measurements without model comparison."""
    from .regional_endpoint import freeze_regional_endpoint

    print(
        json.dumps(
            freeze_regional_endpoint(args.root, args.config, args.output),
            indent=2,
            allow_nan=False,
        )
    )


def add_commands(commands):
    endpoint = commands.add_parser("cohort-endpoint", help=endpoint_command.__doc__)
    for name in ("root", "config", "output"):
        endpoint.add_argument(f"--{name}", required=True)
    endpoint.set_defaults(run=endpoint_command)
    diagnose = commands.add_parser(
        "cohort-residual-diagnose", help=residual_diagnose_command.__doc__
    )
    diagnose.add_argument("--study", required=True)
    diagnose.add_argument("--output", required=True)
    diagnose.set_defaults(run=residual_diagnose_command)
    residual = commands.add_parser("cohort-residual", help=residual_command.__doc__)
    residual.add_argument("--config", required=True)
    residual.add_argument("--output", required=True)
    residual.set_defaults(run=residual_command)
    probe = commands.add_parser("cohort-probe", help=probe_command.__doc__)
    probe.add_argument("--config", required=True)
    probe.add_argument("--output", required=True)
    probe.set_defaults(run=probe_command)
    joint = commands.add_parser("cohort-joint", help="Joint H&E–RNA–protein experiments")
    joint_commands = joint.add_subparsers(dest="joint_action", required=True)
    for action in ("prepare", "pilot", "study", "diagnose", "compare-backbones"):
        parser = joint_commands.add_parser(action)
        parser.add_argument("--output", required=True)
        if action == "prepare":
            parser.add_argument("--root", required=True)
        elif action != "compare-backbones":
            parser.add_argument("--prepared", required=True)
        if action in ("prepare", "pilot", "study", "compare-backbones"):
            parser.add_argument("--config", required=True)
        if action in ("prepare", "study"):
            parser.add_argument("--folds", required=True)
        if action == "pilot":
            parser.add_argument("--biopsies", nargs="+", default=["313-932", "514-6760"])
        if action == "study":
            parser.add_argument("--seeds", nargs="+", type=int, default=[0, 17, 42])
        if action == "diagnose":
            parser.add_argument("--study", required=True)
        if action == "compare-backbones":
            parser.add_argument("--hoptimus-study", required=True)
            parser.add_argument("--uni2-study", required=True)
        parser.set_defaults(run=joint_command)
    reconcile = commands.add_parser(
        "cohort-reconcile", help="Reconcile source identities and replay author coordinate chains"
    )
    for name in ("root", "pipeline", "measurement", "config", "output"):
        reconcile.add_argument(f"--{name}", required=True)
    reconcile.set_defaults(run=reconciliation_command)
    measurement = commands.add_parser("cohort-measurement", help=measurement_command.__doc__)
    for name in ("root", "study", "capability", "config", "output"):
        measurement.add_argument(f"--{name}", required=True)
    measurement.set_defaults(run=measurement_command)
    capability = commands.add_parser("cohort-capability", help=capability_command.__doc__)
    for name in ("root", "study", "protocol", "output"):
        capability.add_argument(f"--{name}", required=True)
    capability.set_defaults(run=capability_command)
    diagnostics = commands.add_parser("cohort-diagnose", help=diagnostics_command.__doc__)
    diagnostics.add_argument("--root", required=True)
    diagnostics.add_argument("--run", dest="run_directory", required=True)
    diagnostics.add_argument("--output", required=True)
    diagnostics.set_defaults(run=diagnostics_command)
    study = commands.add_parser("cohort-study", help=study_command.__doc__)
    for name in ("root", "config", "folds", "protocol", "output"):
        study.add_argument(f"--{name}", required=True)
    study.set_defaults(run=study_command)
    controls = commands.add_parser("cohort-controls", help=controls_command.__doc__)
    for name in ("root", "config", "baseline-run", "output"):
        controls.add_argument(f"--{name}", required=True)
    controls.add_argument("--biopsy", default="313-932")
    controls.add_argument("--epochs", type=int, default=100)
    controls.add_argument("--learning-rate", type=float, default=1e-3)
    controls.add_argument("--seed", type=int, default=0)
    controls.add_argument("--device", default="cpu")
    controls.set_defaults(run=controls_command)
    inspect = commands.add_parser("cohort-inspect", help="inventory the 13-biopsy regional cohort")
    inspect.add_argument("--root", required=True)
    inspect.add_argument("--output", required=True)
    inspect.set_defaults(run=inspect_command)
    features = commands.add_parser("cohort-features", help=features_command.__doc__)
    features.add_argument("--root", required=True)
    features.add_argument("--output", required=True)
    features.add_argument("--biopsies", nargs="+", default=["313-932"])
    features.add_argument("--config")
    features.add_argument("--encoder-manifest", required=True)
    features.add_argument("--model-dir", required=True)
    features.add_argument("--device", default="cuda")
    features.add_argument("--batch-size", type=int, default=1)
    features.set_defaults(run=features_command)
    pilot = commands.add_parser("cohort-inspect-pilot", help=inspect_pilot_command.__doc__)
    pilot.add_argument("--root", required=True)
    pilot.add_argument("--biopsy", default="313-932")
    pilot.add_argument("--config")
    pilot.add_argument("--output", required=True)
    pilot.set_defaults(run=inspect_pilot_command)
    for name, runner in (("cohort-train", train_command), ("cohort-demo", demo_command)):
        parser = commands.add_parser(name, help=runner.__doc__)
        parser.add_argument("--root", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--config")
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--learning-rate", type=float, default=1e-3)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--device", default="cpu")
        if name == "cohort-train":
            parser.add_argument("--folds", required=True)
            parser.add_argument("--outer-fold", type=int, required=True)
            parser.add_argument("--baseline")
        else:
            parser.add_argument("--biopsy", default="313-932")
        parser.set_defaults(run=runner)
    predict = commands.add_parser(
        "cohort-predict", help="predict tissue regions without spatial RNA"
    )
    predict.add_argument("--root", required=True)
    predict.add_argument("--biopsy", required=True)
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--transfer-only", action="store_true")
    predict.add_argument("--device", default="cpu")
    predict.set_defaults(run=predict_command)
    evaluate = commands.add_parser("cohort-evaluate", help="score within Slide-seq replicates")
    evaluate.add_argument("--root", required=True)
    evaluate.add_argument("--prediction", required=True)
    evaluate.add_argument("--genes", "--evaluation-genes", dest="genes")
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(run=evaluate_command)
