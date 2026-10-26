"""Bounded joint-modality experiments on the historical regional cohort.

Inputs are prepared without spatial RNA. Each fold exports every test prediction
before opening its test target tables. Seeds are repeated fits, not new patients.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .cohort import DEFAULT_PROTEIN_MARKERS, discover_cohort, prepare_joint_inputs
from .cohort_cli import _subset, comparison_scores, qualified_targets, replicate_scores
from .error_decomposition import error_decomposition
from .joint_model import JointModelConfig
from .protein_features import load_joint_inputs, save_joint_inputs
from .reference import _cluster, select_training_outcomes
from .schema import RNAReference
from .serialization import load_prediction, save_prediction
from .study import _reference_hash, _save_pool, _sha, validate_folds
from .train import FittedJointModel, TrainingSample, fit_joint

PRINCIPAL_ARMS = ("H", "HR", "HP", "RP", "HRP", "HRP_generic")
CONTROL_ARMS = ("HRP_constant_reference", "HRP_transfer_only")
EXTENSION_CONTROL_ARMS = ("HRP_mlp", "HRP_coverage_only")
INTERVENTIONS = ("protein_unavailable", "protein_permuted", "image_permuted", "correction_removed")
_COVERAGE = "common_original_protein_coverage"
_SCORE = "mean_replicate_median_gene_spearman"


def _write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _tsv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value) if isinstance(value, (list, dict, tuple)) else value
                for key, value in row.items()
            }
            for row in rows
        )


def _configuration(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema") != "heir.joint_region_workflow.v1":
        raise ValueError("unsupported joint workflow configuration")
    model_values = dict(config["model"])
    # Historical prepared configurations already bind their encoder through the
    # preparation hash; hydrate the explicit new model field without rewriting them.
    if "H" in model_values.get("modalities", ("H", "R", "P")):
        model_values.setdefault(
            "image_encoder_manifest_sha256",
            config["preparation"].get("feature_encoder_manifest_sha256"),
        )
    model = JointModelConfig.from_dict(model_values)
    backbone = config.get("image_backbone")
    if backbone is not None:
        manifest_path = Path(backbone["manifest"])
        manifest = json.loads(manifest_path.read_text())
        if (
            backbone.get("mode") != "frozen"
            or backbone.get("crop_mode") != "historical_registered_region"
            or backbone.get("feature_dim") != model.feature_dim
            or manifest.get("fine_tuning") != "prohibited"
            or manifest.get("feature_width") != model.feature_dim
            or _sha(manifest_path) != model.image_encoder_manifest_sha256
            or backbone.get("manifest_sha256") != model.image_encoder_manifest_sha256
            or config["preparation"].get("feature_encoder_manifest_sha256")
            != model.image_encoder_manifest_sha256
            or backbone.get("feature_cache_dir") != config["preparation"].get("feature_cache_dir")
        ):
            raise ValueError("joint frozen backbone/configuration identity mismatch")
    if config["preparation"]["region_size"] != 100 and not config.get("extension_label"):
        raise ValueError("nonhistorical joint geometry requires an explicit extension label")
    if config.get("controls_revision") is not None:
        if (
            config["controls_revision"] != "v010"
            or config.get("common_protein_coverage") is not True
            or config.get("additional_interventions") != ["protein_intensity_permuted"]
            or config.get("assignment_parameter_relative_tolerance") != 0.02
            or config.get("full_model_parameter_relative_tolerance") != 0.01
            or model.assignment_network != "dot_product"
            or model.protein_signal != "measured"
            or model.correction_initialization != "zero"
            or config.get("outcome_selection") is not None
        ):
            raise ValueError("unsupported matched joint control experiment")
    elif config.get("additional_interventions") or config.get("common_protein_coverage"):
        raise ValueError("new coverage/intervention reporting requires a declared control revision")
    selection = config.get("outcome_selection")
    if selection is not None:
        if (
            not config.get("extension_label")
            or selection.get("schema") != "heir.training_outcome_selection.v1"
            or selection.get("minimum_detection") != 0.01
            or selection.get("minimum_patients") != 2
            or selection.get("missing_heldout_panel_policy")
            != "explicit_unevaluable_no_reselection"
            or isinstance(selection.get("max_genes"), bool)
            or not isinstance(selection.get("max_genes"), int)
            or not len(model.gene_ids) <= selection["max_genes"] <= 256
        ):
            raise ValueError(
                "broader outcomes require a labeled, bounded training-only selection rule"
            )
    return config, model


def fit_schedule(protocol, seeds=None):
    """Enumerate the declared fits; controls have their own fixed seed inventory."""
    principal = tuple(protocol["principal_arms"])
    controls = tuple(protocol["control_arms"])
    seed_axis = tuple(protocol["seeds"] if seeds is None else seeds)
    if seed_axis != tuple(protocol["seeds"]) or len(set(seed_axis)) != len(seed_axis):
        raise ValueError("requested seeds differ from the declared study seed inventory")
    if any(a not in PRINCIPAL_ARMS for a in principal) or any(
        a not in CONTROL_ARMS + EXTENSION_CONTROL_ARMS for a in controls
    ):
        raise ValueError("unknown joint study arm")
    if len(set(principal + controls)) != len(principal + controls):
        raise ValueError("duplicate joint study arms")
    if any(a in EXTENSION_CONTROL_ARMS for a in controls) and not protocol.get("extension_label"):
        raise ValueError("additional joint controls require an explicit extension protocol")
    if not protocol.get("extension_label") and (
        principal != PRINCIPAL_ARMS
        or controls != CONTROL_ARMS
        or seed_axis != (0, 17, 42)
        or tuple(protocol["outer_folds"]) != (0, 1, 2, 3, 4)
        or tuple(protocol["control_seeds"]) != (0,)
    ):
        raise ValueError("initial joint study requires the prespecified 100-fit design")
    return [
        {"outer_fold": fold, "arm": arm, "seed": seed}
        for fold in protocol["outer_folds"]
        for arm in principal + controls
        for seed in (seed_axis if arm in principal else tuple(protocol["control_seeds"]))
    ]


def _arm_config(base, arm):
    if arm not in PRINCIPAL_ARMS + CONTROL_ARMS + EXTENSION_CONTROL_ARMS:
        raise ValueError("unknown joint arm")
    modalities = tuple(c for c in "HRP" if c in arm.split("_")[0])
    return replace(
        base,
        modalities=modalities,
        feature_dim=base.feature_dim if "H" in modalities else 0,
        image_encoder_manifest_sha256=base.image_encoder_manifest_sha256
        if "H" in modalities
        else None,
        state_dim=base.state_dim if "R" in modalities else 0,
        state_gene_ids=base.state_gene_ids if "R" in modalities else (),
        marker_ids=base.marker_ids if "P" in modalities else (),
        assignment_mode="constant_reference" if arm == "HRP_constant_reference" else "learned",
        transfer_only=arm == "HRP_transfer_only",
        assignment_network="pairwise_mlp" if arm == "HRP_mlp" else base.assignment_network,
        protein_signal="coverage_only" if arm == "HRP_coverage_only" else base.protein_signal,
    )


def _interventions(config):
    return (*INTERVENTIONS, *config.get("additional_interventions", []))


def _parameter_comparison(base, config):
    """Check declared active capacities before fitting, without changing the RNG stream."""
    from .joint_model import JointModel

    with torch.random.fork_rng():
        counts = {}
        for arm in ("HRP", *EXTENSION_CONTROL_ARMS):
            model = JointModel(_arm_config(base, arm))
            counts[arm] = {
                "active_trainable_parameters": model.active_trainable_parameter_count(),
                "assignment_trainable_parameters": model.assignment_trainable_parameter_count(),
            }
    reference = counts["HRP"]
    for arm in EXTENSION_CONTROL_ARMS:
        for field, setting in (
            ("active_trainable_parameters", "full_model_parameter_relative_tolerance"),
            ("assignment_trainable_parameters", "assignment_parameter_relative_tolerance"),
        ):
            fraction = abs(counts[arm][field] - reference[field]) / reference[field]
            if fraction > config[setting]:
                raise ValueError("joint control exceeds its declared parameter tolerance")
            counts[arm]["relative_difference_" + field] = fraction
    return counts


def typed_training_reference(inputs, *, gene_ids, embedding_gene_ids, max_states=32):
    """Pool only training RNA, preserving broad types and equal-patient source mass."""
    patients = sorted({i.patient_id for i in inputs})
    if not inputs or len({(i.patient_id, i.specimen_id) for i in inputs}) != len(inputs):
        raise ValueError("generic pooling requires unique training specimens")
    genes = tuple(dict.fromkeys((*gene_ids, *embedding_gene_ids)))
    specimen_counts = Counter(i.patient_id for i in inputs)
    means, variances, masses, counts, types, sources = [], [], [], [], [], []
    for item in sorted(inputs, key=lambda x: (x.patient_id, x.specimen_id)):
        ref = item.reference
        columns = [ref.gene_ids.index(g) for g in genes]
        means.append(ref.expression[:, columns])
        variances.append(ref.variance[:, columns])
        mass = np.asarray(ref.proportions, dtype=np.float64)
        masses.append(mass / mass.sum() / len(patients) / specimen_counts[item.patient_id])
        counts.append(ref.cell_counts)
        types.extend(ref.state_types)
        sources.append(
            {
                "patient_id": item.patient_id,
                "specimen_id": item.specimen_id,
                "reference_sha256": _reference_hash(ref),
            }
        )
    means, variances = np.concatenate(means), np.concatenate(variances)
    masses, counts, types = np.concatenate(masses), np.concatenate(counts), np.asarray(types)
    axis = sorted(set(types))
    if max_states < len(axis):
        raise ValueError("generic state budget cannot preserve every broad RNA type")
    sizes = {t: int((types == t).sum()) for t in axis}
    allocation = {t: 1 for t in axis}
    for _ in range(min(max_states, len(means)) - len(axis)):
        eligible = [t for t in axis if allocation[t] < sizes[t]]
        chosen = max(eligible, key=lambda t: (sizes[t] / (allocation[t] + 1), t))
        allocation[chosen] += 1
    panel = [genes.index(g) for g in embedding_gene_ids]
    pooled_mean, pooled_var, pooled_mass, pooled_counts, pooled_types = [], [], [], [], []
    for cell_type in axis:
        source_rows = np.flatnonzero(types == cell_type)
        labels = _cluster(np.log1p(means[source_rows][:, panel]), allocation[cell_type])
        for label in sorted(set(labels)):
            selected = source_rows[labels == label]
            mean = np.average(means[selected], weights=masses[selected], axis=0)
            variance = (
                np.average(
                    variances[selected] + means[selected] ** 2, weights=masses[selected], axis=0
                )
                - mean**2
            )
            pooled_mean.append(mean)
            pooled_var.append(np.maximum(variance, 0))
            pooled_mass.append(masses[selected].sum())
            pooled_counts.append(counts[selected].sum())
            pooled_types.append(cell_type)
    expression = np.asarray(pooled_mean, dtype=np.float32)
    reference = RNAReference(
        state_ids=tuple(f"training_pool::{i}" for i in range(len(expression))),
        gene_ids=genes,
        expression=expression,
        embedding=np.log1p(expression[:, panel]),
        proportions=np.asarray(pooled_mass, dtype=np.float32),
        state_types=tuple(pooled_types),
        variance=np.asarray(pooled_var, dtype=np.float32),
        cell_counts=np.asarray(pooled_counts, dtype=np.int64),
        embedding_gene_ids=tuple(embedding_gene_ids),
    )
    return reference, {
        "schema": "heir.training_reference_pool.v1",
        "filename": "pooled_reference.npz",
        "training_patient_ids": patients,
        "source_references": sources,
        "weighting": "equal_patient_then_equal_specimen_then_source_state_proportion",
        "clustering": "within_broad_type_deterministic_log1p_RNA_states",
        "type_state_budget": allocation,
        "target_expression_used": False,
        "reference_sha256": _reference_hash(reference),
    }


def prepare_joint(root, config_path, folds_path, output):
    config, model = _configuration(config_path)
    cohort = discover_cohort(root)
    folds = validate_folds(folds_path, {e["patient_id"] for e in cohort.values()})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "config.json").write_bytes(Path(config_path).read_bytes())
    (output / "folds.csv").write_bytes(Path(folds_path).read_bytes())
    records, cache = {}, {}
    for biopsy in sorted(cohort):
        print(f"joint prepare: {biopsy}, target-free inputs", flush=True)
        inputs, provenance = prepare_joint_inputs(
            root,
            biopsy,
            modalities=("H", "R", "P"),
            gene_ids=None if config.get("outcome_selection") else model.gene_ids,
            protein_marker_ids=model.marker_ids,
            verification_cache=cache,
            **config["preparation"],
        )
        directory = output / "inputs" / biopsy
        save_joint_inputs(inputs, directory, provenance)
        records[biopsy] = {
            "patient_id": inputs.patient_id,
            "specimen_id": inputs.specimen_id,
            "input_directory": f"inputs/{biopsy}",
            "provenance": provenance,
        }
    manifest = {
        "schema": "heir.joint_prepared.v1",
        "root": str(Path(root).resolve()),
        "config_sha256": _sha(config_path),
        "folds_sha256": _sha(folds_path),
        "biopsies": records,
        "outer_folds": list(folds),
        "spatial_targets_loaded": False,
        "shared_transforms": "fit_only_on_each_fit_training_patients_inside_fit_joint",
        "rna_reference_scope": "full_measured_axis"
        if config.get("outcome_selection")
        else "fixed_model_and_descriptor_genes",
        "target_tables": "separate_fold_targets_written_only_when_role_allows_loading",
        "scientific_qualification": False,
    }
    _write(output / "prepared.json", manifest)
    return manifest


def _prepared(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "prepared.json").read_text())
    if metadata.get("schema") != "heir.joint_prepared.v1" or metadata["spatial_targets_loaded"]:
        raise ValueError("invalid target-free joint preparation")
    if _sha(directory / "config.json") != metadata["config_sha256"]:
        raise ValueError("prepared joint model configuration changed")
    if _sha(directory / "folds.csv") != metadata["folds_sha256"]:
        raise ValueError("prepared patient folds changed")
    config, model = _configuration(directory / "config.json")
    return metadata, config, model


def _input_inventory(prepared, metadata):
    return {
        biopsy: _sha(Path(prepared) / "inputs" / biopsy / "metadata.json")
        for biopsy in sorted(metadata["biopsies"])
    }


def _check_input_inventory(prepared, metadata, expected):
    if _input_inventory(prepared, metadata) != expected:
        raise ValueError("prepared input metadata changed during the joint run")


def _inputs(prepared, biopsy, arm, pooled=None):
    modalities = tuple(c for c in "HRP" if c in arm.split("_")[0])
    if arm == "HRP_generic":
        modalities = ("H", "P")  # Do not load the query RNA payload.
    inputs, provenance = load_joint_inputs(
        Path(prepared) / "inputs" / biopsy, modalities=modalities
    )
    if arm == "HRP_generic":
        if pooled is None:
            raise ValueError("generic arm requires its training-only reference")
        inputs = replace(inputs, reference=pooled)
    return inputs, provenance


def _target(prepared, metadata, biopsy, model, directory):
    inputs, provenance = _inputs(prepared, biopsy, "H")
    targets, operator, groups, coverage = qualified_targets(
        metadata["root"], biopsy, inputs, provenance, model
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{biopsy}.npz"
    if not path.exists():
        with path.open("xb") as handle:
            np.savez(
                handle,
                observation_ids=np.asarray(targets.observation_ids),
                gene_ids=np.asarray(targets.gene_ids),
                expression=targets.expression,
                observed=targets.observed,
                groups=np.asarray(groups),
                cell_ids=np.asarray(operator.cell_ids),
                rows=operator.rows,
                columns=operator.columns,
                weights=operator.weights,
            )
        _write(directory / f"{biopsy}.json", coverage)
    return targets, operator, groups, coverage


def _transfer(prediction):
    return replace(
        prediction,
        expression=prediction.transfer,
        log_correction=torch.zeros_like(prediction.log_correction),
    )


def _assignment_rows(prediction, inputs, identifiers):
    q = prediction.conditional_assignment.detach().cpu().numpy()
    if not q.shape[1]:
        return [
            {
                **identifiers,
                "state_id": None,
                "broad_type": None,
                "status": "direct_decoder_no_reference_assignment",
            }
        ]
    entropy = -np.sum(q * np.log(np.maximum(q, 1e-30)), axis=1)
    return [
        {
            **identifiers,
            "state_id": state,
            "broad_type": cell_type,
            "mean_region_weight": float(q[:, j].mean()),
            "mean_assignment_entropy": float(entropy.mean()),
            "mean_maximum_state_weight": float(q.max(axis=1).mean()),
            "interpretation": "regional_reference_weights_not_calibrated_cell_fractions",
        }
        for j, (state, cell_type) in enumerate(
            zip(inputs.reference.state_ids, inputs.reference.state_types)
        )
    ]


def _score_export(path, target, model, identifiers, overlap=None, *, suffix="", support=None):
    prediction, _ = load_prediction(path)
    targets, operator, groups, _ = target
    report = comparison_scores(
        prediction, targets, operator, groups, model.gene_ids, model.normalization_gene_ids
    )
    if overlap is not None:
        direct = set(overlap["confirmed_gene_ids"]) | set(overlap["possible_gene_ids"])
        report["outcome_groups"] = {}
        for name, panel in (
            ("direct_protein_marker_genes", [g for g in model.gene_ids if g in direct]),
            ("other_outcome_genes", [g for g in model.gene_ids if g not in direct]),
        ):
            report["outcome_groups"][name] = {
                "gene_ids": panel,
                "status": "reported" if panel else "no_requested_outcomes_in_group",
                "scores": replicate_scores(
                    prediction, targets, operator, groups, panel, model.normalization_gene_ids
                )
                if panel
                else None,
            }
        report["protein_outcome_overlap"] = overlap
    decomposition = error_decomposition(
        prediction, targets, operator, groups, model.gene_ids, model.normalization_gene_ids
    )
    report.update(identifiers)
    report["squared_bias"] = decomposition["squared_bias"]
    report["centered_mse"] = decomposition["centered_mse"]
    report["evaluation_observation_ids"] = list(targets.observation_ids)
    report["target_gene_eligibility"] = [
        {
            "assay_replicate": r["assay_replicate"],
            "genes": [
                {
                    "gene_id": g["gene_id"],
                    "n_regions": g["n_regions"],
                    "target_exclusion": g["undefined_reason"]
                    if g["undefined_reason"] in ("insufficient_regions", "constant_target")
                    else None,
                }
                for g in r["gene_scores"]
            ],
        }
        for r in report["replicate_scores"]
    ]
    report["prediction_sha256"] = _sha(path)
    if support is not None:
        report["coverage_support"] = support
        if support["replicates_without_covered_observations"]:
            panels = [report] + [
                group["scores"]
                for group in report["outcome_groups"].values()
                if group["scores"] is not None
            ]
            for panel in panels:
                for key in (_SCORE, "normalized_log_mse", "squared_bias", "centered_mse"):
                    if key in panel:
                        panel[key + "_covered_replicate_diagnostic"] = panel[key]
                        panel[key] = None
                panel["headline_status"] = "no_common_protein_coverage_in_some_replicates"
                panel["replicate_scores"].extend(
                    _empty_coverage_replicate(group, panel["evaluation_gene_ids"])
                    for group in support["replicates_without_covered_observations"]
                )
                panel["replicate_scores"].sort(key=lambda row: row["assay_replicate"])
            report["target_gene_eligibility"].extend(
                {
                    "assay_replicate": group,
                    "genes": [
                        {"gene_id": g, "n_regions": 0, "target_exclusion": "insufficient_regions"}
                        for g in model.gene_ids
                    ],
                }
                for group in support["replicates_without_covered_observations"]
            )
    _write(path.with_suffix(suffix + ".scores.json"), report)
    _write(path.with_suffix(suffix + ".decomposition.json"), decomposition)
    rows = []
    for panel, panel_report in [("all", report)] + [
        (name, value["scores"])
        for name, value in report["outcome_groups"].items()
        if value["scores"] is not None
    ]:
        for replicate in panel_report["replicate_scores"]:
            rows.append(
                {
                    **identifiers,
                    "patient_id": prediction.patient_id,
                    "specimen_id": prediction.specimen_id,
                    "assay_replicate": replicate["assay_replicate"],
                    "outcome_panel": panel,
                    "spearman": replicate["median_gene_spearman"],
                    "normalized_log_mse": replicate["normalized_log_mse"],
                    "headline_status": replicate["headline_status"],
                    "evaluation_gene_ids": panel_report["evaluation_gene_ids"],
                }
            )
    return report, rows, [{**identifiers, **row} for row in decomposition["per_gene"]]


def _coverage_target(target, inputs):
    """Condition evaluation on original input coverage, never on measured RNA values."""
    targets, operator, groups, metadata = target
    protein = inputs.protein_features
    covered = (
        np.asarray(protein.observed).any(axis=1)
        if protein is not None
        else np.zeros(len(inputs.cell_ids), dtype=bool)
    )
    lookup = dict(zip(inputs.cell_ids, covered))
    if set(lookup) != set(operator.cell_ids):
        raise ValueError("protein coverage and target operator region axes differ")
    positive = np.asarray(operator.weights) > 0
    contributing = np.asarray([lookup[key] for key in operator.cell_ids])[
        operator.columns[positive]
    ]
    selected = np.bincount(operator.rows[positive], minlength=len(targets.observation_ids)) > 0
    np.logical_and.at(selected, operator.rows[positive], contributing)
    retained_groups = tuple(group for group, retain in zip(groups, selected) if retain)
    support = {
        "criterion": "at_least_one_qualified_marker_in_every_positive_operator_contributor",
        "coverage_source": "original_unintervened_protein_availability",
        "target_expression_used_for_coverage": False,
        "covered_prediction_regions": int(covered.sum()),
        "covered_prediction_region_ids": [
            key for key, retain in zip(inputs.cell_ids, covered) if retain
        ],
        "total_prediction_regions": len(covered),
        "evaluation_observation_ids": [
            key for key, retain in zip(targets.observation_ids, selected) if retain
        ],
        "excluded_observation_ids": [
            key for key, retain in zip(targets.observation_ids, selected) if not retain
        ],
        "replicates_without_covered_observations": sorted(set(groups) - set(retained_groups)),
        "original_replicate_ids": sorted(set(groups)),
    }
    if not selected.any():
        return None, support
    subset, mapping = _subset(targets, operator, selected)
    return (subset, mapping, retained_groups, metadata), support


def _empty_coverage_replicate(group, genes):
    return {
        "assay_replicate": group,
        "median_gene_spearman": None,
        "normalized_log_mse": None,
        "headline_status": "no_common_original_protein_coverage",
        "gene_scores": [
            {
                "gene_id": gene,
                "n_regions": 0,
                "spearman": None,
                "undefined_reason": "insufficient_regions",
            }
            for gene in genes
        ],
    }


def _empty_coverage_report(export, model, support):
    return {
        **{k: v for k, v in export.items() if k != "path"},
        "prediction_sha256": _sha(export["path"]),
        "evaluation_gene_ids": list(model.gene_ids),
        "normalization_gene_ids": list(model.normalization_gene_ids),
        "evaluation_observation_ids": [],
        "target_gene_eligibility": [],
        "replicate_scores": [
            _empty_coverage_replicate(group, model.gene_ids)
            for group in support["original_replicate_ids"]
        ],
        "outcome_groups": {},
        "coverage_support": support,
        "headline_status": "no_common_original_protein_coverage",
        _SCORE: None,
        "normalized_log_mse": None,
        "squared_bias": None,
        "centered_mse": None,
    }


def summarize_joint(reports):
    """Average seeds within biological patient before any across-patient contrast."""
    support = {}
    keys = set()
    for report in reports:
        key = tuple(report[k] for k in ("patient_id", "specimen_id", "arm", "seed", "component"))
        if key in keys:
            raise ValueError("duplicate patient/arm/seed prediction report")
        keys.add(key)
        identity = (report["patient_id"], report["specimen_id"])
        contract = {
            k: report[k]
            for k in (
                "evaluation_observation_ids",
                "evaluation_gene_ids",
                "normalization_gene_ids",
                "target_gene_eligibility",
            )
        }
        if "coverage_support" in report:
            contract["coverage_support"] = report["coverage_support"]
        if identity in support and contract != support[identity]:
            raise ValueError("joint comparisons require identical target rows and gene eligibility")
        support[identity] = contract
    arms = {}
    for arm in sorted({r["arm"] for r in reports if r["component"] == "full"}):
        records = [r for r in reports if r["arm"] == arm and r["component"] == "full"]
        patients = []
        for patient in sorted({r["patient_id"] for r in records}):
            selected = [r for r in records if r["patient_id"] == patient]
            specimens = {r["specimen_id"] for r in selected}
            seeds = {r["seed"] for r in selected}
            if len(selected) != len(specimens) * len(seeds):
                raise ValueError("joint patient summaries require every specimen at every seed")
            row = {"patient_id": patient, "seeds": sorted({r["seed"] for r in selected})}
            for key in (_SCORE, "normalized_log_mse", "squared_bias", "centered_mse"):
                values = [r[key] for r in selected]
                row[key] = float(np.mean(values)) if all(v is not None for v in values) else None
            patients.append(row)
        aggregate = {}
        for key in (_SCORE, "normalized_log_mse", "squared_bias", "centered_mse"):
            values = [r[key] for r in patients]
            aggregate[key] = float(np.mean(values)) if all(v is not None for v in values) else None
            defined = [v for v in values if v is not None]
            aggregate[key + "_defined_patient_mean_diagnostic"] = (
                float(np.mean(defined)) if defined else None
            )
        arms[arm] = {"patients": patients, "n_patients": len(patients), **aggregate}
    contrasts = []
    if "HRP" in arms:
        left = {r["patient_id"]: r for r in arms["HRP"]["patients"]}
        for comparator in ("HR", "HP", "RP", "HRP_generic", *EXTENSION_CONTROL_ARMS):
            if comparator not in arms:
                continue
            right = {r["patient_id"]: r for r in arms[comparator]["patients"]}
            if set(left) != set(right):
                raise ValueError("joint paired contrasts require identical patient sets")
            for patient in sorted(left):
                if left[patient]["seeds"] != right[patient]["seeds"]:
                    raise ValueError("joint paired arms require identical seed inventories")
                row = {
                    "contrast": f"HRP-minus-{comparator}",
                    "patient_id": patient,
                    "seed_aggregation": "within_patient_before_patient_contrast",
                }
                for key in (_SCORE, "normalized_log_mse", "squared_bias", "centered_mse"):
                    a, b = left[patient][key], right[patient][key]
                    row["delta_" + key] = a - b if a is not None and b is not None else None
                contrasts.append(row)
    paired_summary = {}
    for name in sorted({row["contrast"] for row in contrasts}):
        selected = [r for r in contrasts if r["contrast"] == name]
        result = {"n_patients": len(selected)}
        for key in (_SCORE, "normalized_log_mse", "squared_bias", "centered_mse"):
            values = [r["delta_" + key] for r in selected]
            defined = [v for v in values if v is not None]
            result["mean_paired_delta_" + key] = (
                float(np.mean(values)) if len(defined) == len(values) else None
            )
            result["n_defined_" + key] = len(defined)
        paired_summary[name] = result
    return {
        "arms": arms,
        "paired_contrasts": contrasts,
        "paired_summary": paired_summary,
        "aggregation": "equal_seed_within_patient_then_equal_patient",
        "scientific_qualification": False,
        "independent_confirmation": False,
    }


def _coverage_rows(inputs, identifiers, outcome_genes):
    feature = inputs.protein_features
    if feature is None:
        return [{**identifiers, "status": "protein_modality_unavailable"}]
    rows = []
    metadata = feature.metadata.get("panel", {}).get("qualified_features", [])
    records = {r["marker_id"]: r for r in metadata}
    for index, marker in enumerate(feature.marker_ids):
        linked = dict(DEFAULT_PROTEIN_MARKERS).get(marker)
        record = records.get(marker, {"gene_ids": [linked] if linked else []})
        confirmed, possible = record.get("gene_ids", []), record.get("possible_direct_gene_ids", [])
        rows.append(
            {
                **identifiers,
                "marker_id": marker,
                "observed_regions": int(feature.observed[:, index].sum()),
                "total_regions": len(inputs.cell_ids),
                "valid_source_cells": int(feature.valid_counts[:, index].sum()),
                "replicate_labels": list(feature.replicate_labels),
                "direct_outcome_overlap": [g for g in confirmed if g in outcome_genes],
                "possible_direct_outcome_overlap": [g for g in possible if g in outcome_genes],
                "linked_RNA_genes": confirmed,
                "gene_mapping_status": record.get("gene_mapping_status", "initial_five_panel"),
                "mean_replicate_disagreement": float(
                    feature.replicate_disagreement[:, index].mean()
                )
                if feature.replicate_disagreement is not None
                else None,
                "registration_confidence_increases_with_replicates": False,
            }
        )
    return rows


def _export_fitted(fitted, inputs, provenance, path, identifiers):
    prediction = fitted.predict(inputs)
    save_prediction(
        prediction,
        path,
        {
            "model_family": "joint",
            "model_config": fitted.model.config.to_dict(),
            "training_patient_ids": list(fitted.training_patient_ids),
            "reference_pool": fitted.preparation.get("reference_pool"),
            "sources": provenance,
            **identifiers,
            "selection": fitted.selection,
            "test_targets_loaded": identifiers["evaluation_role"] == "same_biopsy_pilot",
            "held_out_test": identifiers["evaluation_role"] != "same_biopsy_pilot",
            "scientific_qualification": False,
        },
    )
    diagnostics = fitted.predict_diagnostics(inputs)
    with Path(path).with_suffix(".diagnostics.npz").open("xb") as handle:
        np.savez(handle, **diagnostics)
    return prediction


def _intervened_inputs(inputs, seed, additional=()):
    from .protein_features import permute_protein_features

    shuffled, permutation = permute_protein_features(inputs.protein_features, seed=seed)
    image_order = np.random.default_rng(seed + 1).permutation(len(inputs.cell_ids))
    interventions = {
        "protein_unavailable": (
            replace(inputs, protein_features=None),
            {"all_protein_paths": True},
        ),
        "protein_permuted": (replace(inputs, protein_features=shuffled), permutation),
        "image_permuted": (
            replace(inputs, image_features=inputs.image_features[image_order]),
            {"image_source_row": image_order.tolist(), "coordinates_preserved": True},
        ),
    }
    if "protein_intensity_permuted" in additional:
        from .protein_features import permute_protein_intensities

        changed, details = permute_protein_intensities(inputs.protein_features, seed=seed)
        interventions["protein_intensity_permuted"] = (
            replace(inputs, protein_features=changed),
            details,
        )
    return interventions


def _fit_and_export(
    prepared,
    config,
    base,
    arm,
    seed,
    training_biopsies,
    validation_biopsies,
    test_biopsies,
    targets,
    pooled,
    pool_metadata,
    output,
    fold,
    roles,
    *,
    pilot=False,
):
    """Fit one arm and export all its test variants without receiving test targets."""
    model = _arm_config(base, arm)
    samples = []
    validation = []
    for biopsies, destination in ((training_biopsies, samples), (validation_biopsies, validation)):
        for biopsy in biopsies:
            inputs, _ = _inputs(prepared, biopsy, arm, pooled)
            target, operator, groups, _ = targets[biopsy]
            destination.append(TrainingSample(inputs, target, operator, groups))
    preparation = {
        "joint_workflow": config,
        "outer_fold": fold,
        "fold_roles": roles,
        "study_arm": arm,
        "seed": seed,
        "test_targets_loaded": pilot,
        "refit_with_validation": False,
        "held_out_test": not pilot,
        "evaluation_role": "same_biopsy_pilot" if pilot else "internal_patient_held_out",
        "source_prepared_sha256": _sha(Path(prepared) / "prepared.json"),
    }
    if (output / "outcome_panel.json").is_file():
        preparation["frozen_outcome_panel_sha256"] = _sha(output / "outcome_panel.json")
    if arm == "HRP_generic":
        preparation["reference_pool"] = pool_metadata
        preparation["generic_reference_interpretation"] = (
            "single_biopsy_training_pool_engineering_control_not_personalization_evidence"
            if pilot
            else "pooled_training_patients_only_personalization_control"
        )
    settings = {**config["training"], "seed": seed}
    print(f"joint fit: fold={fold} arm={arm} seed={seed}", flush=True)
    fitted = fit_joint(samples, validation, model, preparation=preparation, **settings)
    bundle = output / f"{arm}-seed-{seed}.bundle"
    fitted.save(bundle)
    if arm == "HRP_generic":
        _save_pool(bundle, pooled, pool_metadata)
    reloaded = FittedJointModel.load(bundle, device=settings.get("device", "cpu"))
    probe = samples[0].inputs
    original, restored = fitted.predict(probe), reloaded.predict(probe)
    if not torch.equal(original.expression, restored.expression):
        raise ValueError("joint bundle save/reload changed predictions")
    if not torch.equal(original.conditional_assignment, restored.conditional_assignment):
        raise ValueError("joint bundle save/reload changed assignments")
    _write(
        output / f"{arm}-seed-{seed}.acceptance.json",
        {
            "save_reload_expression_exact": True,
            "save_reload_assignment_exact": True,
            "training_patient_ids": list(fitted.training_patient_ids),
            "validation_patient_ids": list(fitted.validation_patient_ids),
            "selected_checkpoint": fitted.selection,
            "test_targets_loaded": pilot,
            "held_out_test": not pilot,
            "active_trainable_parameters": fitted.model.active_trainable_parameter_count(),
        },
    )
    exports, assignments, coverage = [], [], []
    for biopsy in test_biopsies:
        inputs, provenance = _inputs(prepared, biopsy, arm, pooled)
        ids = {
            "biopsy_id": biopsy,
            "arm": arm,
            "seed": seed,
            "outer_fold": fold,
            "patient_id": inputs.patient_id,
            "specimen_id": inputs.specimen_id,
            "evaluation_role": preparation["evaluation_role"],
            "component": "full",
        }
        path = output / f"test-{biopsy}-{arm}-seed-{seed}.npz"
        prediction = _export_fitted(fitted, inputs, provenance, path, ids)
        exports.append({"path": str(path), **ids})
        assignments.extend(_assignment_rows(prediction, inputs, ids))
        if "P" in model.modalities:
            coverage.extend(_coverage_rows(inputs, ids, base.gene_ids))
        if "R" in model.modalities and not model.transfer_only:
            transfer_path = path.with_name(path.stem + ".transfer.npz")
            save_prediction(
                _transfer(prediction),
                transfer_path,
                {
                    **ids,
                    "component": "transfer",
                    "postfit_transfer_ablation": True,
                    "test_targets_loaded": pilot,
                    "held_out_test": not pilot,
                },
            )
            exports.append({"path": str(transfer_path), **ids, "component": "transfer"})
        if arm == "HRP":
            for name, (altered, intervention) in _intervened_inputs(
                inputs, seed + 7100, config.get("additional_interventions", ())
            ).items():
                intervention_path = path.with_name(path.stem + f".{name}.npz")
                current_ids = {**ids, "component": name}
                _export_fitted(fitted, altered, provenance, intervention_path, current_ids)
                _write(intervention_path.with_suffix(".intervention.json"), intervention)
                exports.append({"path": str(intervention_path), **current_ids})
            correction_path = path.with_name(path.stem + ".correction_removed.npz")
            save_prediction(
                _transfer(prediction),
                correction_path,
                {
                    **ids,
                    "component": "correction_removed",
                    "test_targets_loaded": pilot,
                    "held_out_test": not pilot,
                },
            )
            exports.append({"path": str(correction_path), **ids, "component": "correction_removed"})
    return exports, assignments, coverage


def _outcome_overlap(metadata):
    confirmed, possible = set(), set()
    for biopsy in metadata["biopsies"].values():
        records = (
            biopsy["provenance"].get("protein", {}).get("panel", {}).get("qualified_features", [])
        )
        for record in records:
            confirmed.update(record.get("gene_ids", []))
            possible.update(record.get("possible_direct_gene_ids", []))
    return {
        "confirmed_gene_ids": sorted(confirmed),
        "possible_gene_ids": sorted(possible),
        "grouping": "conservative_union_confirmed_and_possible_overlap_from_panel_metadata",
    }


def _evaluate_exports(exports, targets, model, overlap=None):
    reports, rows, errors = [], [], []
    for export in exports:
        identifiers = {k: v for k, v in export.items() if k != "path"}
        report, score_rows, error_rows = _score_export(
            Path(export["path"]), targets[export["biopsy_id"]], model, identifiers, overlap
        )
        reports.append(report)
        rows.extend(score_rows)
        errors.extend(error_rows)
    return reports, rows, errors


def _evaluate_coverage_exports(prepared, exports, targets, model, overlap=None):
    """Score every arm and intervention on the same original protein-covered rows."""
    restricted = {}
    for biopsy, target in targets.items():
        inputs, _ = _inputs(prepared, biopsy, "HRP")
        restricted[biopsy] = _coverage_target(target, inputs)
    reports, rows, errors = [], [], []
    for export in exports:
        target, support = restricted[export["biopsy_id"]]
        path = Path(export["path"])
        if target is None:
            report = _empty_coverage_report(export, model, support)
            _write(path.with_suffix(".common-coverage.scores.json"), report)
            score_rows = [
                {
                    **{k: v for k, v in export.items() if k != "path"},
                    "assay_replicate": group,
                    "outcome_panel": "all",
                    "spearman": None,
                    "normalized_log_mse": None,
                    "headline_status": report["headline_status"],
                    "evaluation_gene_ids": list(model.gene_ids),
                }
                for group in support["original_replicate_ids"]
            ]
            error_rows = []
        else:
            report, score_rows, error_rows = _score_export(
                path,
                target,
                model,
                {k: v for k, v in export.items() if k != "path"},
                overlap,
                suffix=".common-coverage",
                support=support,
            )
        reports.append(report)
        rows.extend({"evaluation_support": _COVERAGE, **r} for r in score_rows)
        errors.extend({"evaluation_support": _COVERAGE, **r} for r in error_rows)
    return reports, rows, errors


def _complete_run(
    output,
    reports,
    score_rows,
    error_rows,
    assignments,
    coverage,
    exports,
    metadata,
    *,
    coverage_results=None,
):
    summary = {"schema": "heir.joint_study_result.v1", **metadata, **summarize_joint(reports)}
    transferred = [{**r, "component": "full"} for r in reports if r["component"] == "transfer"]
    summary["postfit_transfer"] = summarize_joint(transferred)["arms"] if transferred else {}
    if coverage_results is not None:
        covered_reports, covered_rows, covered_errors = coverage_results
        covered_summary = summarize_joint(covered_reports)
        covered_transfer = [
            {**r, "component": "full"} for r in covered_reports if r["component"] == "transfer"
        ]
        covered_summary["postfit_transfer"] = (
            summarize_joint(covered_transfer)["arms"] if covered_transfer else {}
        )
        covered_summary["evaluation_support"] = _COVERAGE
        covered_summary["coverage_by_biopsy"] = {
            r["biopsy_id"]: r["coverage_support"] for r in covered_reports
        }
        summary[_COVERAGE] = covered_summary
        _tsv(output / "common_coverage_patient_scores.tsv", covered_rows)
        _tsv(output / "common_coverage_error_decomposition.tsv", covered_errors)
        _tsv(output / "common_coverage_paired_contrasts.tsv", covered_summary["paired_contrasts"])
    _write(output / "summary.json", summary)
    _write(output / "exports.json", exports)
    _tsv(output / "patient_scores.tsv", score_rows)
    _tsv(output / "paired_contrasts.tsv", summary["paired_contrasts"])
    _tsv(output / "error_decomposition.tsv", error_rows)
    _tsv(output / "assignment_diagnostics.tsv", assignments)
    _tsv(output / "marker_coverage.tsv", coverage)
    intervention_rows = []
    export_paths = {
        (e["biopsy_id"], e["seed"], e["component"]): Path(e["path"])
        for e in exports
        if e["arm"] == "HRP"
    }
    intervention_reports = [("all_target_eligible", r) for r in reports]
    if coverage_results is not None:
        intervention_reports.extend((_COVERAGE, r) for r in coverage_results[0])
    original_by_support = {
        (support, r["biopsy_id"], r["seed"]): r
        for support, r in intervention_reports
        if r["arm"] == "HRP" and r["component"] == "full"
    }
    for support, report in intervention_reports:
        if (
            not report.get("prediction_available", True)
            or report["arm"] != "HRP"
            or report["component"] not in ("full", *INTERVENTIONS, "protein_intensity_permuted")
        ):
            continue
        original = original_by_support[(support, report["biopsy_id"], report["seed"])]
        row = {
            k: report[k]
            for k in ("patient_id", "specimen_id", "biopsy_id", "seed", "outer_fold", "component")
        }
        row["evaluation_support"] = support
        row["evaluation_observation_ids"] = report["evaluation_observation_ids"]
        if support == _COVERAGE:
            row["coverage_criterion"] = report["coverage_support"]["criterion"]
        for key in (_SCORE, "normalized_log_mse", "squared_bias", "centered_mse"):
            a, b = report[key], original[key]
            row[key] = a
            row["delta_from_original_" + key] = a - b if a is not None and b is not None else None
        base_prediction, _ = load_prediction(
            export_paths[(report["biopsy_id"], report["seed"], "full")]
        )
        altered, _ = load_prediction(
            export_paths[(report["biopsy_id"], report["seed"], report["component"])]
        )
        selected = torch.ones(len(base_prediction.cell_ids), dtype=torch.bool)
        if support == _COVERAGE:
            ids = set(report["coverage_support"]["covered_prediction_region_ids"])
            selected = torch.tensor([key in ids for key in base_prediction.cell_ids])
        row["change_summary_prediction_regions"] = int(selected.sum())
        row["mean_absolute_log1p_linear_expression_change"] = (
            float(
                (
                    torch.log1p(altered.expression[selected])
                    - torch.log1p(base_prediction.expression[selected])
                )
                .abs()
                .mean()
            )
            if selected.any()
            else None
        )
        row["mean_absolute_reference_mixture_change"] = (
            float(
                (
                    altered.conditional_assignment[selected]
                    - base_prediction.conditional_assignment[selected]
                )
                .abs()
                .mean()
            )
            if selected.any()
            else None
        )
        row["intervention_scope"] = "fixed_model_all_modality_dependent_paths"
        intervention_rows.append(row)
    _tsv(output / "modality_interventions.tsv", intervention_rows)
    lines = [
        "Joint H&E–RNA–protein exploratory comparison",
        "",
        "Historical registered-pixel regions; section correspondence and physical calibration "
        "remain unqualified.",
        "Seeds are averaged within patients. These results do not establish cell placement "
        "or independent confirmation.",
        "",
        "Arm | Patients | Mean spatial Spearman | Normalized MSE",
        "--- | --- | --- | ---",
    ]
    if metadata.get("evaluation_role") == "same_biopsy_pilot":
        lines.extend(
            [
                "",
                "The pilot generic pool contains only the same training biopsy. Its agreement "
                "with matched RNA is an engineering check, not a personalization comparison.",
                "",
            ]
        )
    for arm, values in summary["arms"].items():
        lines.append(
            f"{arm} | {values['n_patients']} | {values[_SCORE]} | {values['normalized_log_mse']}"
        )
    lines.extend(
        [
            "",
            "Positive paired Spearman and negative paired error differences favor HRP.",
            "Transfer exports and fixed-model interventions are reported separately "
            "from retrained controls.",
            "Any architectural advance must consider held-out contrasts, centered error, "
            "replicate consistency, and outcomes beyond direct protein markers.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def pilot_joint(prepared, biopsies, config_path, output):
    metadata, config, base = _prepared(prepared)
    if config.get("outcome_selection"):
        raise ValueError(
            "training-only broader outcome selection requires the patient-held-out study"
        )
    if _sha(config_path) != metadata["config_sha256"]:
        raise ValueError("pilot config differs from target-free prepared inputs")
    pilot_arms = tuple(config.get("pilot_arms", (*PRINCIPAL_ARMS, "HRP_transfer_only")))
    if (
        not pilot_arms
        or len(set(pilot_arms)) != len(pilot_arms)
        or any(a not in PRINCIPAL_ARMS + CONTROL_ARMS + EXTENSION_CONTROL_ARMS for a in pilot_arms)
    ):
        raise ValueError("joint pilot requires an explicit unique valid arm inventory")
    biopsies = tuple(biopsies)
    if (
        not biopsies
        or len(set(biopsies)) != len(biopsies)
        or not set(biopsies) <= set(metadata["biopsies"])
    ):
        raise ValueError("pilot requires unique prepared biopsies")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    input_inventory = _input_inventory(prepared, metadata)
    _write(
        output / "identity.json",
        {
            "prepared_sha256": _sha(Path(prepared) / "prepared.json"),
            "input_metadata_sha256": input_inventory,
            "config_sha256": metadata["config_sha256"],
            "source_file_sha256": {
                str(p): _sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        },
    )
    all_reports, all_scores, all_errors, assignments, coverage, exports = [], [], [], [], [], []
    for biopsy in biopsies:
        directory = output / biopsy
        directory.mkdir()
        target = _target(prepared, metadata, biopsy, base, directory / "targets")
        pooled, pool_metadata = typed_training_reference(
            [_inputs(prepared, biopsy, "HR")[0]],
            gene_ids=base.gene_ids,
            embedding_gene_ids=base.state_gene_ids,
            max_states=config["preparation"]["max_states"],
        )
        current_exports = []
        for arm in pilot_arms:
            exported, assigned, covered = _fit_and_export(
                prepared,
                config,
                base,
                arm,
                0,
                [biopsy],
                [],
                [biopsy],
                {biopsy: target},
                pooled,
                pool_metadata,
                directory,
                "pilot",
                {},
                pilot=True,
            )
            current_exports.extend(exported)
            assignments.extend(assigned)
            coverage.extend(covered)
        reports, scores, errors = _evaluate_exports(
            current_exports, {biopsy: target}, base, _outcome_overlap(metadata)
        )
        all_reports.extend(reports)
        all_scores.extend(scores)
        all_errors.extend(errors)
        exports.extend(current_exports)
    _check_input_inventory(prepared, metadata, input_inventory)
    return _complete_run(
        output,
        all_reports,
        all_scores,
        all_errors,
        assignments,
        coverage,
        exports,
        {
            "fit_count": len(biopsies) * len(pilot_arms),
            "pilot_arms": list(pilot_arms),
            "evaluation_role": "same_biopsy_pilot",
            "held_out_generalization": False,
            "generic_reference_interpretation": (
                "single_biopsy_training_pool; expected_equivalence_to_matched_reference; "
                "not_evidence_of_personalization"
            ),
        },
    )


def _spatial_axis(root, biopsy):
    from .cohort import _retained_gene_axis, _verified_source_path

    source = discover_cohort(root)[biopsy]["role_sources"]["targets"]
    genes = _retained_gene_axis(_verified_source_path(root, source))
    return genes, source


def freeze_fold_outcomes(prepared, metadata, base, training_biopsies, selection, directory):
    """Rank training RNA plus training ST detection, write panel before held-out I/O."""
    from .cohort import load_cohort_targets

    references, detection, sources = {}, {}, []
    for biopsy in training_biopsies:
        inputs, provenance = load_joint_inputs(
            Path(prepared) / "inputs" / biopsy, modalities=("R",)
        )
        if inputs.patient_id in references:
            raise ValueError("broader panel selection currently requires one biopsy per patient")
        genes, source = _spatial_axis(metadata["root"], biopsy)
        reference_genes = set(inputs.reference.gene_ids)
        genes = tuple(g for g in genes if g in reference_genes)
        targets, operator, record = load_cohort_targets(
            metadata["root"], biopsy, inputs, provenance, gene_ids=genes
        )
        groups = np.asarray(record["observation_replicates"])
        # These are bead-occupied H&E regions, before any outcome-dependent exposure filter.
        fraction = np.mean(
            [(targets.expression[groups == group] > 0).mean(0) for group in sorted(set(groups))],
            axis=0,
        )
        references[inputs.patient_id] = inputs.reference
        detection[inputs.patient_id] = (genes, fraction)
        sources.append(
            {
                "biopsy_id": biopsy,
                "patient_id": inputs.patient_id,
                "reference_sha256": _reference_hash(inputs.reference),
                "target_source": source,
                "n_regions_by_replicate": {
                    group: int((groups == group).sum()) for group in sorted(set(groups))
                },
                "observation_ids_sha256": hashlib.sha256(
                    json.dumps(operator.observation_ids).encode()
                ).hexdigest(),
            }
        )
    panel, record = select_training_outcomes(
        references,
        detection,
        tuple(sorted(references)),
        required_gene_ids=base.gene_ids,
        max_genes=selection["max_genes"],
        minimum_detection=selection["minimum_detection"],
        minimum_patients=selection["minimum_patients"],
    )
    record.update(
        sources=sources,
        frozen_before_validation_or_test_access=True,
        detection_unit="fraction_positive_bead_occupied_HE_regions_equal_assay_replicates",
        missing_heldout_panel_policy="explicit_unevaluable_no_reselection",
    )
    _write(Path(directory) / "outcome_panel.json", record)
    return replace(base, gene_ids=panel, supervised_gene_ids=panel, normalization_gene_ids=panel)


def panel_coverage(prepared, metadata, biopsies, genes, *, spatial):
    """Post-freeze availability check only; never changes the selected panel."""
    unavailable = {}
    for biopsy in biopsies:
        inputs, _ = load_joint_inputs(Path(prepared) / "inputs" / biopsy, modalities=("R",))
        missing_rna = sorted(set(genes) - set(inputs.reference.gene_ids))
        missing_spatial, source = [], None
        if spatial:
            measured, source = _spatial_axis(metadata["root"], biopsy)
            missing_spatial = sorted(set(genes) - set(measured))
        if missing_rna or missing_spatial:
            unavailable[biopsy] = {
                "patient_id": inputs.patient_id,
                "missing_RNA_gene_ids": missing_rna,
                "missing_spatial_gene_ids": missing_spatial,
                "target_source": source,
                "panel_changed": False,
            }
    return unavailable


def unevaluable_reports(metadata, biopsies, planned_fits, model, reason):
    """Retain missing patients and seeds in strict cohort/paired summaries."""
    return [
        {
            "biopsy_id": biopsy,
            "patient_id": metadata["biopsies"][biopsy]["patient_id"],
            "specimen_id": metadata["biopsies"][biopsy]["specimen_id"],
            "arm": planned["arm"],
            "seed": planned["seed"],
            "outer_fold": planned["outer_fold"],
            "component": "full",
            "evaluation_role": "internal_patient_held_out",
            "prediction_available": False,
            "coverage_failure": reason,
            "headline_status": "unevaluable_frozen_normalization_panel",
            "evaluation_observation_ids": [],
            "evaluation_gene_ids": list(model.gene_ids),
            "normalization_gene_ids": list(model.normalization_gene_ids),
            "target_gene_eligibility": [],
            "replicate_scores": [],
            _SCORE: None,
            "normalized_log_mse": None,
            "squared_bias": None,
            "centered_mse": None,
        }
        for biopsy in biopsies
        for planned in planned_fits
    ]


def _backbone_matrix(path):
    matrix = json.loads(Path(path).read_text())
    if matrix.get("schema") == "heir.joint_controls_matrix.v1":
        return _controls_matrix(path, matrix)
    if (
        matrix.get("schema") != "heir.joint_backbone_matrix.v1"
        or set(matrix.get("backbones", {})) != {"hoptimus1", "uni2h"}
        or matrix.get("seeds") != [0]
        or matrix.get("outer_folds") != [0, 1, 2, 3, 4]
        or matrix.get("principal_fits") != 55
        or matrix.get("mechanistic_fits") != 20
        or matrix.get("total_fits") != 75
        or matrix.get("shared_RP_source") != "hoptimus1"
        or matrix.get("further_seeds_in_this_revision") != []
        or matrix.get("panel_or_context_extensions_in_this_revision") is not False
        or matrix.get("decision_frozen_before_first_heldout_result") is not True
        or matrix.get("additional_seed_17_42_fits_authorized") is not False
        or matrix.get("expanded_or_broader_panels_scored") is not False
    ):
        raise ValueError("unsupported frozen seed-0 two-backbone study matrix")
    comparable = []
    encoder_ids = []
    for backbone in ("hoptimus1", "uni2h"):
        declaration = matrix["backbones"][backbone]
        expected_arms = (
            list(PRINCIPAL_ARMS)
            if backbone == "hoptimus1"
            else [a for a in PRINCIPAL_ARMS if a != "RP"]
        )
        experiment = Path(path).parent / declaration["experiment_config"]
        config, model = _configuration(experiment)
        if (
            declaration["principal_arms"] != expected_arms
            or declaration["control_arms"] != list(CONTROL_ARMS)
            or declaration["planned_fits"] != (40 if backbone == "hoptimus1" else 35)
            or _sha(experiment) != declaration["experiment_config_sha256"]
            or config["image_backbone"]["backbone_id"] != backbone
            or model.feature_dim != 1536
            or config.get("outcome_selection") is not None
        ):
            raise ValueError("backbone matrix changes the frozen arm or feature contract")
        encoder_ids.append(model.image_encoder_manifest_sha256)
        shared = _compatible_experiment_config(config, model)
        shared["preparation"] = {
            k: v
            for k, v in shared["preparation"].items()
            if k not in ("feature_cache_dir", "feature_encoder_manifest_sha256")
        }
        shared["model"].pop("image_encoder_manifest_sha256")
        comparable.append(shared)
    if comparable[0] != comparable[1] or len(set(encoder_ids)) != 2:
        raise ValueError("backbone alternatives must share all nonimage settings")
    return matrix


def _controls_matrix(path, matrix):
    if (
        set(matrix.get("backbones", {})) != {"hoptimus1", "uni2h"}
        or matrix.get("seeds") != [0]
        or matrix.get("outer_folds") != [0, 1, 2, 3, 4]
        or matrix.get("principal_fits") != 10
        or matrix.get("mechanistic_fits") != 20
        or matrix.get("total_fits") != 30
        or matrix.get("shared_RP_source") is not None
        or matrix.get("paired_backbone_arms") != ["HRP", *EXTENSION_CONTROL_ARMS]
        or matrix.get("correction_initialization") != "zero"
        or matrix.get("further_seeds_in_this_revision") != []
        or matrix.get("panel_or_context_extensions_in_this_revision") is not False
        or matrix.get("decision_frozen_before_first_heldout_result") is not True
        or matrix.get("additional_seed_17_42_fits_authorized") is not False
        or matrix.get("expanded_or_broader_panels_scored") is not False
    ):
        raise ValueError("unsupported matched 30-fit joint control matrix")
    comparable, encoders = [], []
    for name, declaration in matrix["backbones"].items():
        source = Path(path).parent / declaration["experiment_config"]
        config, model = _configuration(source)
        if (
            declaration["principal_arms"] != ["HRP"]
            or declaration["control_arms"] != list(EXTENSION_CONTROL_ARMS)
            or declaration["planned_fits"] != 15
            or _sha(source) != declaration["experiment_config_sha256"]
            or config.get("controls_revision") != "v010"
            or config["image_backbone"]["backbone_id"] != name
            or model.feature_dim != 1536
        ):
            raise ValueError("joint control matrix changes the bound model or fit contract")
        shared = _compatible_experiment_config(config, model)
        shared["preparation"] = {
            k: v
            for k, v in shared["preparation"].items()
            if k not in ("feature_cache_dir", "feature_encoder_manifest_sha256")
        }
        shared["model"].pop("image_encoder_manifest_sha256")
        comparable.append(shared)
        encoders.append(model.image_encoder_manifest_sha256)
        _parameter_comparison(model, config)
    if comparable[0] != comparable[1] or len(set(encoders)) != 2:
        raise ValueError("joint control backbones must share all nonimage settings")
    return matrix


def _compatible_experiment_config(config, model):
    # Narrative/backbone declarations and the explicitly revised pilot inventory
    # do not change the shared training experiment. All other fields must match.
    result = {
        k: v
        for k, v in config.items()
        if k not in ("image_backbone", "interpretation", "pilot_arms")
    }
    result["model"] = model.to_dict()
    return result


def _study_configuration(prepared, config_path):
    metadata, config, base = _prepared(prepared)
    protocol = json.loads(Path(config_path).read_text())
    if protocol.get("schema") != "heir.joint_patient_study.v1":
        raise ValueError("unsupported joint study protocol")
    source_config = Path(prepared) / "config.json"
    if "matrix_protocol" in protocol:
        matrix_path = Path(config_path).parent / protocol["matrix_protocol"]
        matrix = _backbone_matrix(matrix_path)
        if _sha(matrix_path) != protocol["matrix_protocol_sha256"]:
            raise ValueError("frozen backbone matrix identity differs")
        declaration = matrix["backbones"][protocol["backbone_id"]]
        source_config = Path(config_path).parent / declaration["experiment_config"]
        candidate, model = _configuration(source_config)
        if (
            _sha(source_config) != declaration["experiment_config_sha256"]
            or protocol["principal_arms"] != declaration["principal_arms"]
            or protocol["control_arms"] != declaration["control_arms"]
            or protocol["seeds"] != matrix["seeds"]
            or protocol["control_seeds"] != [0]
            or protocol["outer_folds"] != matrix["outer_folds"]
            or candidate.get("outcome_selection") is not None
            or _compatible_experiment_config(candidate, model)
            != _compatible_experiment_config(config, base)
            or candidate["image_backbone"]["backbone_id"] != protocol["backbone_id"]
            or Path(prepared).resolve() != Path(declaration["prepared_directory"]).resolve()
            or protocol.get("test_prediction_before_target_loading") is not True
            or protocol.get("no_train_validation_refit") is not True
            or protocol.get("no_further_seed_panel_or_context_runs_in_this_revision") is not True
        ):
            raise ValueError("backbone experiment override differs from bound prepared inputs")
        config, base = candidate, model
    if protocol["model_config_sha256"] != _sha(source_config):
        raise ValueError("joint study protocol differs from prepared model configuration")
    if config.get("controls_revision"):
        for biopsy in metadata["biopsies"]:
            inputs, _ = _inputs(prepared, biopsy, "HRP")
            if getattr(inputs.protein_features, "correspondence", None) is None:
                raise ValueError("new joint controls require explicit protein correspondence")
    return metadata, config, base, protocol, source_config


def study_joint(prepared, config_path, folds_path, seeds, output):
    metadata, config, base, protocol, source_config = _study_configuration(prepared, config_path)
    if _sha(folds_path) != metadata["folds_sha256"]:
        raise ValueError("joint study patient folds differ from prepared folds")
    schedule = fit_schedule(protocol, seeds)
    folds = validate_folds(
        folds_path,
        {e["patient_id"] for e in metadata["biopsies"].values()},
        protocol["outer_folds"],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, source in (
        ("protocol.json", config_path),
        ("folds.csv", folds_path),
        ("config.json", source_config),
    ):
        (output / name).write_bytes(Path(source).read_bytes())
    _write(output / "fit_plan.json", schedule)
    input_inventory = _input_inventory(prepared, metadata)
    if config.get("controls_revision"):
        _write(output / "parameter_comparison.json", _parameter_comparison(base, config))
    package = Path(__file__).resolve().parent
    _write(
        output / "identity.json",
        {
            "prepared_sha256": _sha(Path(prepared) / "prepared.json"),
            "input_metadata_sha256": input_inventory,
            "protocol_sha256": _sha(config_path),
            "experiment_config_sha256": _sha(source_config),
            "prepared_config_sha256": metadata["config_sha256"],
            "backbone_id": protocol.get("backbone_id"),
            "matrix_protocol_sha256": protocol.get("matrix_protocol_sha256"),
            "folds_sha256": _sha(folds_path),
            "source_file_sha256": {str(p): _sha(p) for p in sorted(package.glob("*.py"))},
            "prior_cohort_exposure": True,
            "no_train_validation_refit": True,
        },
    )
    all_reports, all_scores, all_errors, assignments, coverage, all_exports = [], [], [], [], [], []
    common_reports, common_rows, common_errors = [], [], []
    actual_fits, exclusions = 0, []
    for fold, roles in folds.items():
        directory = output / f"fold-{fold}"
        directory.mkdir()
        grouped = {
            role: [b for b, e in metadata["biopsies"].items() if roles[e["patient_id"]] == role]
            for role in ("train", "validation", "test")
        }
        planned_fits = [r for r in schedule if r["outer_fold"] == fold]
        fold_model = base
        unavailable_test = {}
        if config.get("outcome_selection"):
            if metadata.get("rna_reference_scope") != "full_measured_axis":
                raise ValueError("broader study needs separately prepared full RNA references")
            fold_model = freeze_fold_outcomes(
                prepared, metadata, base, grouped["train"], config["outcome_selection"], directory
            )
            # Panel has already been written. Held-out availability cannot alter it.
            unavailable_validation = panel_coverage(
                prepared, metadata, grouped["validation"], fold_model.gene_ids, spatial=True
            )
            if unavailable_validation:
                reason = {
                    "reason": "validation_panel_unavailable",
                    "details": unavailable_validation,
                }
                exclusions.append({"outer_fold": fold, **reason})
                _write(directory / "unevaluable.json", reason)
                skipped = unevaluable_reports(
                    metadata, grouped["test"], planned_fits, fold_model, reason
                )
                all_reports.extend(skipped)
                all_scores.extend(skipped)
                continue
            unavailable_test = panel_coverage(
                prepared, metadata, grouped["test"], fold_model.gene_ids, spatial=False
            )
            if unavailable_test:
                reason = {"reason": "test_RNA_panel_unavailable", "details": unavailable_test}
                exclusions.append({"outer_fold": fold, **reason})
                skipped = unevaluable_reports(
                    metadata, list(unavailable_test), planned_fits, fold_model, reason
                )
                all_reports.extend(skipped)
                all_scores.extend(skipped)
        test_biopsies = [b for b in grouped["test"] if b not in unavailable_test]
        targets = {
            b: _target(prepared, metadata, b, fold_model, directory / "targets")
            for b in grouped["train"] + grouped["validation"]
        }
        pooled, pool_metadata = typed_training_reference(
            [_inputs(prepared, b, "HR")[0] for b in grouped["train"]],
            gene_ids=fold_model.gene_ids,
            embedding_gene_ids=fold_model.state_gene_ids,
            max_states=config["preparation"]["max_states"],
        )
        exports = []
        for planned in planned_fits:
            exported, assigned, covered = _fit_and_export(
                prepared,
                config,
                fold_model,
                planned["arm"],
                planned["seed"],
                grouped["train"],
                grouped["validation"],
                test_biopsies,
                targets,
                pooled,
                pool_metadata,
                directory,
                fold,
                roles,
            )
            actual_fits += 1
            exports.extend(exported)
            assignments.extend(assigned)
            coverage.extend(covered)
        _write(
            directory / "test_predictions_complete.json",
            {
                "test_biopsies": test_biopsies,
                "unevaluable_test_RNA_panels": unavailable_test,
                "test_targets_loaded": False,
                "prediction_sha256": {Path(r["path"]).name: _sha(r["path"]) for r in exports},
            },
        )
        # No test target payload (including its measured-gene axis) is opened until
        # every fit/intervention has exported its ordinary target-free prediction.
        unavailable_spatial = (
            panel_coverage(prepared, metadata, test_biopsies, fold_model.gene_ids, spatial=True)
            if config.get("outcome_selection")
            else {}
        )
        if unavailable_spatial:
            reason = {"reason": "test_spatial_panel_unavailable", "details": unavailable_spatial}
            exclusions.append({"outer_fold": fold, **reason})
            for export in exports:
                if export["biopsy_id"] not in unavailable_spatial:
                    continue
                row = unevaluable_reports(
                    metadata, [export["biopsy_id"]], [export], fold_model, reason
                )[0]
                row.update(
                    component=export["component"],
                    prediction_available=True,
                    prediction_sha256=_sha(export["path"]),
                )
                _write(Path(export["path"]).with_suffix(".scores.json"), row)
                _write(
                    Path(export["path"]).with_suffix(".decomposition.json"),
                    {
                        "status": "unevaluable_frozen_normalization_panel",
                        "coverage_failure": reason,
                    },
                )
                all_reports.append(row)
                all_scores.append(row)
        test_targets = {
            b: _target(prepared, metadata, b, fold_model, directory / "targets")
            for b in test_biopsies
            if b not in unavailable_spatial
        }
        reports, scores, errors = _evaluate_exports(
            [e for e in exports if e["biopsy_id"] not in unavailable_spatial],
            test_targets,
            fold_model,
            _outcome_overlap(metadata),
        )
        all_reports.extend(reports)
        all_scores.extend(scores)
        all_errors.extend(errors)
        all_exports.extend(exports)
        if config.get("common_protein_coverage"):
            common = _evaluate_coverage_exports(
                prepared, exports, test_targets, fold_model, _outcome_overlap(metadata)
            )
            common_reports.extend(common[0])
            common_rows.extend(common[1])
            common_errors.extend(common[2])
    _check_input_inventory(prepared, metadata, input_inventory)
    return _complete_run(
        output,
        all_reports,
        all_scores,
        all_errors,
        assignments,
        coverage,
        all_exports,
        {
            "fit_count": actual_fits,
            "planned_fit_count": len(schedule),
            "outcome_selection": config.get("outcome_selection"),
            "explicit_panel_exclusions": exclusions,
            "evaluation_role": "internal_patient_held_out",
            "seeds": list(seeds),
            "protocol_sha256": _sha(config_path),
            "no_train_validation_refit": True,
            "backbone_id": protocol.get("backbone_id"),
            "matrix_protocol_sha256": protocol.get("matrix_protocol_sha256"),
        },
        coverage_results=(common_reports, common_rows, common_errors)
        if config.get("common_protein_coverage")
        else None,
    )


def diagnose_joint(prepared, study, output):
    """Collect saved diagnostics; interventions were frozen before test target loading."""
    metadata, _, _ = _prepared(prepared)
    study, output = Path(study), Path(output)
    identity = json.loads((study / "identity.json").read_text())
    if identity["prepared_sha256"] != _sha(Path(prepared) / "prepared.json"):
        raise ValueError("diagnostic inputs differ from the fitted study preparation")
    _check_input_inventory(prepared, metadata, identity["input_metadata_sha256"])
    exports = json.loads((study / "exports.json").read_text())
    for export in exports:
        path = Path(export["path"])
        scores = json.loads(path.with_suffix(".scores.json").read_text())
        if scores["prediction_sha256"] != _sha(path):
            raise ValueError("saved diagnostic prediction changed")
    output.mkdir(parents=True, exist_ok=False)
    names = (
        "patient_scores.tsv",
        "paired_contrasts.tsv",
        "error_decomposition.tsv",
        "modality_interventions.tsv",
        "assignment_diagnostics.tsv",
        "marker_coverage.tsv",
        "report.md",
        "summary.json",
    )
    names += tuple(
        name
        for name in (
            "common_coverage_patient_scores.tsv",
            "common_coverage_error_decomposition.tsv",
            "common_coverage_paired_contrasts.tsv",
            "parameter_comparison.json",
        )
        if (study / name).is_file()
    )
    for name in names:
        (output / name).write_bytes((study / name).read_bytes())
    receipt = {
        "schema": "heir.joint_diagnostics.v1",
        "fitting_performed": False,
        "target_payloads_reopened": False,
        "prepared_config_sha256": metadata["config_sha256"],
        "study": str(study.resolve()),
        "prediction_count": len(exports),
        "copied_artifact_sha256": {name: _sha(study / name) for name in names},
    }
    _write(output / "receipt.json", receipt)
    return receipt


def _read_table(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _matching_backbone_inputs(matrix, identities):
    """Require identical molecular inputs and crop geometry, with different encoders."""
    inputs = {}
    for backbone, declared in matrix["backbones"].items():
        prepared = Path(declared["prepared_directory"])
        metadata, _, model = _prepared(prepared)
        identity = identities[backbone]
        if _sha(prepared / "prepared.json") != identity["prepared_sha256"]:
            raise ValueError("backbone prepared-source identity changed")
        _check_input_inventory(prepared, metadata, identity["input_metadata_sha256"])
        records = {
            b: json.loads((prepared / "inputs" / b / "metadata.json").read_text())
            for b in metadata["biopsies"]
        }
        if matrix["schema"] == "heir.joint_controls_matrix.v1":
            for biopsy, record in records.items():
                entry = record["payloads"].get("protein_correspondence")
                if entry is None:
                    raise ValueError("control comparison lacks explicit protein correspondence")
                path = prepared / "inputs" / biopsy / entry["file"]
                if path.name != entry["file"] or _sha(path) != entry["sha256"]:
                    raise ValueError("protein correspondence payload integrity mismatch")
        inputs[backbone] = (records, model.image_encoder_manifest_sha256)
    left, right = inputs["hoptimus1"][0], inputs["uni2h"][0]
    if set(left) != set(right) or inputs["hoptimus1"][1] == inputs["uni2h"][1]:
        raise ValueError("backbone comparison requires the same biopsies and distinct encoders")
    for biopsy in left:
        a, b = left[biopsy], right[biopsy]
        for name in (
            "patient_id",
            "specimen_id",
            "cell_ids",
            "prediction_unit",
            "coordinate_units",
        ):
            if a[name] != b[name]:
                raise ValueError("backbone geometry or patient identity differs")
        payloads = ("geometry", "reference", "protein")
        if matrix["schema"] == "heir.joint_controls_matrix.v1":
            payloads += ("protein_correspondence",)
        for payload in payloads:
            if a["payloads"][payload]["sha256"] != b["payloads"][payload]["sha256"]:
                raise ValueError("backbone comparisons must share exact geometry, RNA and protein")
        for name in ("marker_ids", "replicate_labels"):
            if a["protein"][name] != b["protein"][name]:
                raise ValueError("backbone protein axes differ")
        if a["provenance"]["sources"] != b["provenance"]["sources"]:
            raise ValueError("backbone image and molecular source identities differ")
    return {
        "n_biopsies": len(left),
        "geometry_RNA_protein_payload_hashes_equal": True,
        "encoder_manifest_sha256": {name: value[1] for name, value in inputs.items()},
    }


def compare_backbones(hoptimus_study, uni2_study, config_path, output):
    """Combine a frozen backbone matrix, preserving its exact fit and export inventory."""
    matrix = _backbone_matrix(config_path)
    controls = matrix["schema"] == "heir.joint_controls_matrix.v1"
    paths = {"hoptimus1": Path(hoptimus_study), "uni2h": Path(uni2_study)}
    reports, identities, summaries = {}, {}, {}
    for backbone, directory in paths.items():
        declared = matrix["backbones"][backbone]
        protocol = json.loads((directory / "protocol.json").read_text())
        identity = json.loads((directory / "identity.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        experiment = json.loads((directory / "config.json").read_text())
        planned = fit_schedule(protocol, [0])
        prepared_metadata, _, _ = _prepared(declared["prepared_directory"])
        if _sha(directory / "folds.csv") != identity["folds_sha256"]:
            raise ValueError("copied backbone patient folds differ from their identity")
        folds = validate_folds(
            directory / "folds.csv",
            {row["patient_id"] for row in prepared_metadata["biopsies"].values()},
            protocol["outer_folds"],
        )
        if (
            protocol.get("matrix_protocol_sha256") != _sha(config_path)
            or protocol.get("backbone_id") != backbone
            or identity.get("matrix_protocol_sha256") != _sha(config_path)
            or identity["protocol_sha256"] != _sha(directory / "protocol.json")
            or identity["experiment_config_sha256"] != declared["experiment_config_sha256"]
            or _sha(directory / "config.json") != declared["experiment_config_sha256"]
            or len(planned) != declared["planned_fits"]
            or summary["fit_count"] != declared["planned_fits"]
            or len(list(directory.glob("fold-*/*.bundle"))) != declared["planned_fits"]
            or summary.get("explicit_panel_exclusions")
        ):
            raise ValueError("backbone study does not match the frozen fit matrix")
        records = []
        expected = set()
        for fit in planned:
            components = ["full"]
            if "R" in fit["arm"].split("_")[0] and fit["arm"] != "HRP_transfer_only":
                components.append("transfer")
            if fit["arm"] == "HRP":
                components.extend(_interventions(experiment))
            for biopsy, source in prepared_metadata["biopsies"].items():
                if folds[fit["outer_fold"]][source["patient_id"]] == "test":
                    expected.update(
                        (biopsy, fit["outer_fold"], fit["arm"], fit["seed"], c) for c in components
                    )
        exports = json.loads((directory / "exports.json").read_text())
        actual = [
            tuple(e[k] for k in ("biopsy_id", "outer_fold", "arm", "seed", "component"))
            for e in exports
        ]
        if len(set(actual)) != len(actual) or set(actual) != expected:
            raise ValueError("backbone export inventory omits or duplicates a declared patient fit")
        receipts = {
            fold: json.loads(
                (directory / f"fold-{fold}" / "test_predictions_complete.json").read_text()
            )
            for fold in folds
        }
        for fold, receipt in receipts.items():
            frozen = {
                Path(e["path"]).name: _sha(e["path"]) for e in exports if e["outer_fold"] == fold
            }
            if receipt["test_targets_loaded"] or frozen != receipt["prediction_sha256"]:
                raise ValueError("backbone export inventory differs from the pretarget receipt")
        for export in exports:
            path = Path(export["path"])
            report = json.loads(path.with_suffix(".scores.json").read_text())
            source = prepared_metadata["biopsies"][export["biopsy_id"]]
            if (
                report["prediction_sha256"] != _sha(path)
                or any(
                    report[k] != export[k]
                    for k in ("biopsy_id", "outer_fold", "arm", "seed", "component")
                )
                or any(report[k] != source[k] for k in ("patient_id", "specimen_id"))
            ):
                raise ValueError("backbone saved prediction integrity mismatch")
            records.append({**report, "backbone": backbone})
        reports[backbone], identities[backbone], summaries[backbone] = records, identity, summary
    if identities["hoptimus1"]["source_file_sha256"] != identities["uni2h"]["source_file_sha256"]:
        raise ValueError("backbone studies used different downstream source revisions")
    if identities["hoptimus1"]["folds_sha256"] != identities["uni2h"]["folds_sha256"]:
        raise ValueError("backbone studies used different patient folds")
    matching = _matching_backbone_inputs(matrix, identities)
    # Renaming arms here lets the existing summary validator enforce common
    # target rows, normalization and eligibility across both backbone collections.
    summarize_joint(
        [{**r, "arm": f"{backbone}:{r['arm']}"} for backbone, rows in reports.items() for r in rows]
    )
    shared_rp = [r for r in reports["hoptimus1"] if r["arm"] == "RP"]
    if not controls and (not shared_rp or any(r["arm"] == "RP" for r in reports["uni2h"])):
        raise ValueError("RP must be fitted exactly once in the shared comparator substudy")
    by_backbone = {
        name: summarize_joint(
            rows + ([{**r, "backbone": "shared_RP"} for r in shared_rp] if name == "uni2h" else [])
        )
        for name, rows in reports.items()
    }
    paired, backbone_rows = [], []
    for name, summary in by_backbone.items():
        paired.extend({"backbone": name, **row} for row in summary["paired_contrasts"])
    for arm in matrix["paired_backbone_arms"]:
        a = {r["patient_id"]: r for r in by_backbone["uni2h"]["arms"][arm]["patients"]}
        b = {r["patient_id"]: r for r in by_backbone["hoptimus1"]["arms"][arm]["patients"]}
        if set(a) != set(b):
            raise ValueError("paired backbone contrasts require identical patients")
        for patient in sorted(a):
            row = {"arm": arm, "patient_id": patient, "contrast": "UNI2h-minus-Hoptimus1"}
            for key in (_SCORE, "normalized_log_mse", "squared_bias", "centered_mse"):
                x, y = a[patient][key], b[patient][key]
                row["delta_" + key] = x - y if x is not None and y is not None else None
            backbone_rows.append(row)
    backbone_summary = {}
    for arm in matrix["paired_backbone_arms"]:
        rows = [r for r in backbone_rows if r["arm"] == arm]
        aggregate = {"n_patients": len(rows)}
        for key in (_SCORE, "normalized_log_mse", "squared_bias", "centered_mse"):
            values = [r["delta_" + key] for r in rows]
            aggregate["mean_paired_delta_" + key] = (
                float(np.mean(values)) if all(v is not None for v in values) else None
            )
        backbone_summary[arm] = aggregate
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    table_names = (
        "patient_scores",
        "error_decomposition",
        "modality_interventions",
        "assignment_diagnostics",
        "marker_coverage",
    )
    if controls:
        table_names += (
            "common_coverage_patient_scores",
            "common_coverage_error_decomposition",
            "common_coverage_paired_contrasts",
        )
    for name in table_names:
        rows = [
            {"backbone": "shared_RP" if r.get("arm") == "RP" else backbone, **r}
            for backbone, directory in paths.items()
            for r in _read_table(directory / f"{name}.tsv")
        ]
        _tsv(output / f"{name}.tsv", rows)
    _tsv(output / "paired_contrasts.tsv", paired)
    _tsv(output / "backbone_contrasts.tsv", backbone_rows)
    result = {
        "schema": "heir.joint_controls_matrix_result.v1"
        if controls
        else "heir.joint_backbone_matrix_result.v1",
        "matrix_sha256": _sha(config_path),
        "principal_fit_count": matrix["principal_fits"],
        "mechanistic_fit_count": matrix["mechanistic_fits"],
        "total_fit_count": matrix["total_fits"],
        "seeds": [0],
        "shared_RP_fit_count": 0 if controls else 5,
        "shared_RP_refitted_for_UNI2h": False,
        "backbones": by_backbone,
        "paired_backbone_summary": backbone_summary,
        "postfit_transfer": {b: summaries[b]["postfit_transfer"] for b in paths},
        "input_matching": matching,
        "scientific_qualification": False,
        "independent_confirmation": False,
        "no_further_seed_panel_or_context_runs_in_this_revision": True,
        "source_studies": {
            b: {
                "path": str(p.resolve()),
                "identity_sha256": _sha(p / "identity.json"),
                "summary_sha256": _sha(p / "summary.json"),
            }
            for b, p in paths.items()
        },
    }
    if controls:
        common = {}
        for backbone, directory in paths.items():
            rows = [
                json.loads(Path(e["path"]).with_suffix(".common-coverage.scores.json").read_text())
                for e in json.loads((directory / "exports.json").read_text())
            ]
            original_lookup = {
                tuple(r[k] for k in ("biopsy_id", "outer_fold", "arm", "seed", "component")): r
                for r in reports[backbone]
            }
            for report in rows:
                key = tuple(
                    report[k] for k in ("biopsy_id", "outer_fold", "arm", "seed", "component")
                )
                matching_report = original_lookup.get(key)
                if matching_report is None or any(
                    report[k] != matching_report[k]
                    for k in ("prediction_sha256", "patient_id", "specimen_id")
                ):
                    raise ValueError("common-coverage prediction integrity mismatch")
            common[backbone] = rows
        summarize_joint(
            [{**r, "arm": f"{b}:{r['arm']}"} for b, rows in common.items() for r in rows]
        )
        result[_COVERAGE] = {b: summarize_joint(rows) for b, rows in common.items()}
        for backbone, rows in common.items():
            transfer = [{**r, "component": "full"} for r in rows if r["component"] == "transfer"]
            result[_COVERAGE][backbone]["postfit_transfer"] = (
                summarize_joint(transfer)["arms"] if transfer else {}
            )
        result["parameter_comparison"] = {
            b: json.loads((p / "parameter_comparison.json").read_text()) for b, p in paths.items()
        }
        for backbone, directory in paths.items():
            config, model = _configuration(directory / "config.json")
            if result["parameter_comparison"][backbone] != _parameter_comparison(model, config):
                raise ValueError("saved parameter comparison differs from the declared models")
    _write(output / "summary.json", result)
    lines = [
        "Frozen-backbone joint comparison",
        "",
        f"{matrix['principal_fits']} principal fits and {matrix['mechanistic_fits']} "
        "separately identified mechanistic fits, all seed 0.",
        (
            "New zero-initialized attention, compatibility-MLP and coverage-only fits. "
            if controls
            else "RP is one shared five-fold comparator. "
        )
        + "H-optimus-1 and UNI2-h use identical "
        "regional geometry and molecular inputs, with their own declared preprocessing.",
        "",
        "Arm | UNI2-h minus H-optimus-1 spatial correlation | MSE difference",
        "--- | --- | ---",
    ]
    for arm, values in backbone_summary.items():
        lines.append(
            f"{arm} | {values['mean_paired_delta_' + _SCORE]} | "
            f"{values['mean_paired_delta_normalized_log_mse']}"
        )
    lines.extend(
        [
            "",
            "Backbone effects and within-backbone RNA/protein increments are separate.",
            "Transfer, centered error, level bias and fixed-model interventions remain "
            "separate diagnostics. Registered-pixel correspondence remains unqualified.",
            "No additional seeds, panel expansions, contextual crops or coarser targets "
            "are executed in this revision.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result
