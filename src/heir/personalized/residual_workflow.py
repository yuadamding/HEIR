"""Bounded UNI2-h residual study on unchanged regional inputs and targets.

The final family predicts normalized-log outcomes directly. Linear reference
transfer remains a separate export. Inner readouts use outer-training patients
only; each fold's complete test exports precede its first test target load.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from .joint_model import JointModelConfig
from .joint_workflow import (
    _arm_config,
    _check_input_inventory,
    _coverage_target,
    _input_inventory,
    _inputs,
    _prepared,
    _tsv,
    typed_training_reference,
)
from .observation import normalized_log_expression
from .probe_workflow import (
    ENDPOINTS,
    SCALE,
    _assert_matched_support,
    _export_probe,
    _sample_from_source,
    _summary,
    _verify_baseline_truth,
    _write,
    normalized_log_output_contract,
    score_probe,
)
from .schema import RNAReference
from .serialization import load_prediction, save_prediction
from .study import _sha, validate_folds

ARMS = ("H", "HR", "HP", "HRP", "HRP_coverage_only", "HRP_generic")
SUPPORTS = ("all_target_eligible", "common_original_protein_coverage")


def _protocol(path):
    record = json.loads(Path(path).read_text())
    if (
        record.get("schema") != "heir.residual_study.v1"
        or record.get("outer_folds") != list(range(5))
        or tuple(record.get("arms", ())) != ARMS
        or record.get("seed") != 0
        or record.get("alpha") != 10
        or record.get("inner_splits") != 3
        or record.get("expected_neural_fits") != 30
        or record.get("expected_inner_readouts") != 15
        or record.get("expected_reused_deployment_readouts") != 5
        or record.get("prediction_scale") != SCALE
        or record.get("transfer_scale") != "linear_relative_expression"
        or record.get("zero_residual_candidate") is not True
        or set(record.get("baseline_folds", {})) != set(map(str, range(5)))
        or record.get("training")
        != {
            "epochs": 100,
            "learning_rate": 0.001,
            "centered_weight": 0.25,
            "residual_weight": 0.001,
            "residual_bound": 1.0,
            "device": "cpu",
        }
    ):
        raise ValueError("unsupported fixed thirty-fit residual protocol")
    model = JointModelConfig.from_dict(record["model"])
    if (
        model.modalities != ("H", "R", "P")
        or model.feature_dim != 1536
        or model.hidden_dim != 32
        or len(model.gene_ids) != 9
        or model.gene_ids != model.supervised_gene_ids
        or model.gene_ids != model.normalization_gene_ids
        or model.assignment_network != "dot_product"
        or model.protein_signal != "measured"
        or model.assignment_mode != "learned"
        or model.transfer_only
    ):
        raise ValueError("residual study must preserve the declared nine-gene shallow UNI mapper")
    return record, model


def _source_contract(protocol, base):
    prepared = Path(protocol["prepared_directory"])
    source = Path(protocol["source_study_directory"])
    baseline = Path(protocol["baseline_directory"])
    for path, expected in (
        (source / "identity.json", protocol["source_identity_sha256"]),
        (source / "exports.json", protocol["source_exports_sha256"]),
        (prepared / "prepared.json", protocol["prepared_sha256"]),
        (source / "folds.csv", protocol["folds_sha256"]),
    ):
        if _sha(path) != expected:
            raise ValueError("residual source contract checksum differs")
    metadata, _, old_model = _prepared(prepared)
    identity = json.loads((source / "identity.json").read_text())
    source_model = JointModelConfig.from_dict(
        json.loads((source / "config.json").read_text())["model"]
    )
    if (
        identity["prepared_sha256"] != protocol["prepared_sha256"]
        or identity["folds_sha256"] != protocol["folds_sha256"]
        or identity["experiment_config_sha256"] != _sha(source / "config.json")
        or metadata["folds_sha256"] != protocol["folds_sha256"]
    ):
        raise ValueError("residual historical source and preparation identities differ")
    for name in (
        "gene_ids",
        "normalization_gene_ids",
        "state_gene_ids",
        "marker_ids",
        "feature_dim",
        "state_dim",
        "image_encoder_manifest_sha256",
    ):
        if not getattr(base, name) == getattr(old_model, name) == getattr(source_model, name):
            raise ValueError("residual model changes historical feature or molecular axes")
    _check_input_inventory(prepared, metadata, identity["input_metadata_sha256"])
    inventory = json.loads((source / "exports.json").read_text())
    prior_transfer = {}
    for entry in inventory:
        if entry["arm"] == "HRP" and entry["seed"] == 0 and entry["component"] == "transfer":
            key = (entry["outer_fold"], entry["biopsy_id"])
            if key in prior_transfer:
                raise ValueError("duplicate historical reference-transfer export")
            prior_transfer[key] = entry
    h_inputs = {biopsy: _inputs(prepared, biopsy, "H")[0] for biopsy in metadata["biopsies"]}
    folds = validate_folds(
        source / "folds.csv",
        {item.patient_id for item in h_inputs.values()},
        protocol["outer_folds"],
    )
    return prepared, source, baseline, metadata, h_inputs, folds, prior_transfer


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def save_residual_prediction(prediction, path, metadata):
    """Export direct outcomes without presenting them as linear expression."""
    arrays = {
        name: _array(getattr(prediction, name))
        for name in ("normalized_log_expression", "image_baseline", "residual")
    }
    shape = (len(prediction.cell_ids), len(prediction.gene_ids))
    if any(
        value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise ValueError("residual prediction requires finite region-by-gene direct outcomes")
    _validate_direct_identity(metadata, arrays["residual"])
    if not np.array_equal(
        arrays["image_baseline"] + arrays["residual"], arrays["normalized_log_expression"]
    ):
        raise ValueError("residual prediction does not equal baseline plus residual")
    record = {
        "schema": "heir.residual_prediction.v1",
        "prediction_scale": SCALE,
        "normalize_prediction_again": False,
        "patient_id": prediction.patient_id,
        "specimen_id": prediction.specimen_id,
        "prediction_unit": prediction.prediction_unit,
        "coordinate_units": prediction.coordinate_units,
        "test_targets_loaded": False,
        **metadata,
    }
    for key, value in prediction.diagnostics.items():
        value = _array(value)
        if value.dtype.kind != "O":
            arrays["diagnostic__" + key] = value
    with Path(path).open("xb") as stream:
        np.savez(
            stream,
            **arrays,
            region_ids=np.asarray(prediction.cell_ids),
            gene_ids=np.asarray(prediction.gene_ids),
            coordinates=_array(prediction.coordinates_um),
            metadata=np.asarray(json.dumps(record, allow_nan=False)),
        )


def load_residual_prediction(path, inputs, gene_ids):
    with np.load(path, allow_pickle=False) as archive:
        record = json.loads(archive["metadata"].item())
        if (
            record["schema"] != "heir.residual_prediction.v1"
            or record["prediction_scale"] != SCALE
            or record["normalize_prediction_again"] is not False
            or record["patient_id"] != inputs.patient_id
            or record["specimen_id"] != inputs.specimen_id
            or record["prediction_unit"] != inputs.prediction_unit
            or record["coordinate_units"] != inputs.coordinate_units
            or record["encoder_manifest_sha256"] != inputs.image_encoder_manifest_sha256
            or tuple(archive["region_ids"]) != inputs.cell_ids
            or tuple(archive["gene_ids"]) != tuple(gene_ids)
            or not np.array_equal(archive["coordinates"], inputs.coordinates)
        ):
            raise ValueError("residual prediction scale or named geometry differs")
        values = {
            key: archive[key].copy()
            for key in ("normalized_log_expression", "image_baseline", "residual")
        }
    if any(
        value.shape != (len(inputs.cell_ids), len(gene_ids))
        or not np.isfinite(value).all()
        or value.dtype != np.float64
        for value in values.values()
    ):
        raise ValueError("residual prediction arrays differ from their declared axes")
    if not np.array_equal(
        values["image_baseline"] + values["residual"], values["normalized_log_expression"]
    ):
        raise ValueError("saved residual decomposition differs")
    _validate_direct_identity(record, values["residual"])
    return values, record


def _validate_direct_identity(record, residual):
    from .schema import image_backbone_identity

    image_backbone_identity(record.get("encoder_manifest_sha256"), present=True)
    bound = record.get("residual_bound")
    if (
        isinstance(bound, bool)
        or not isinstance(bound, (int, float))
        or not np.isfinite(bound)
        or not 0 < bound <= 1
        or np.any(np.abs(residual) > bound + 1e-6)
    ):
        raise ValueError("direct residual bound must be declared, at most one and respected")


def _linear_log(prediction, inputs, genes, panel):
    if set(prediction.cell_ids) != set(inputs.cell_ids):
        raise ValueError("linear reference transfer region axis differs")
    rows = [prediction.cell_ids.index(key) for key in inputs.cell_ids]
    columns = [prediction.gene_ids.index(gene) for gene in genes]
    denominator = [prediction.gene_ids.index(gene) for gene in panel]
    values = prediction.expression[rows]
    normalized = normalized_log_expression(
        values, torch.ones_like(values, dtype=torch.bool), normalization_indices=denominator
    )
    return normalized[:, columns].detach().cpu().numpy()


def _shared_fit_identity(crossfit, baseline):
    digest = hashlib.sha256()
    for key, value in sorted(crossfit.predictions.items()):
        digest.update(json.dumps(key).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    for name in ("feature_mean", "feature_scale", "coefficients", "intercept"):
        digest.update(np.ascontiguousarray(getattr(baseline, name)).tobytes())
    return digest.hexdigest()


def _score_case(exports, sample, coverage_inputs, genes, panel, old_transfer, directory):
    """Score all components on identical truth/support only after the export barrier."""
    target = (sample.targets, sample.observation, sample.observation_groups, {})
    _, coverage = _coverage_target(target, coverage_inputs)
    retained = set(coverage["evaluation_observation_ids"])
    mask = np.asarray([key in retained for key in sample.targets.observation_ids])
    scored, gene_rows = [], []
    for entry in exports:
        path = Path(entry["path"])
        if _sha(path) != entry["sha256"]:
            raise ValueError("residual export changed after the prediction barrier")
        if entry["component"] == "final":
            values, _ = load_residual_prediction(path, sample.inputs, genes)
            predicted = values["normalized_log_expression"]
        elif entry["component"] == "baseline":
            from .probe_workflow import _load_probe_prediction

            predicted = _load_probe_prediction(path, sample.inputs, genes)
        else:
            prediction, _ = load_prediction(path)
            predicted = _linear_log(prediction, sample.inputs, genes, panel)
        output_contract = normalized_log_output_contract(predicted, genes, panel)
        for support, rows in zip(SUPPORTS, (None, mask)):
            report, errors = score_probe(predicted, sample, genes, panel, row_mask=rows)
            report.update(
                {key: value for key, value in entry.items() if key != "path"},
                evaluation_support=support,
                output_contract=output_contract,
            )
            if rows is not None:
                report["coverage_support"] = coverage
            _write(directory / f"{path.stem}.{support}.scores.json", report)
            scored.append(report)
            gene_rows.extend(
                {
                    **{
                        key: report[key]
                        for key in (
                            "outer_fold",
                            "biopsy_id",
                            "arm",
                            "component",
                            "evaluation_support",
                        )
                    },
                    **row,
                }
                for row in errors
            )
    old_path = Path(old_transfer["path"])
    for support, rows in zip(SUPPORTS, (None, mask)):
        suffix = ".scores.json" if rows is None else ".common-coverage.scores.json"
        old = json.loads(old_path.with_suffix(suffix).read_text())
        if old["prediction_sha256"] != _sha(old_path):
            raise ValueError("historical transfer prediction checksum differs")
        from .probe_workflow import VARIABILITY

        old[VARIABILITY] = _verify_baseline_truth(old_path, old, sample, genes, panel, rows)
        old.update(arm="historical_HRP", component="transfer", evaluation_support=support)
        scored.append(old)
    lookup = {(r["arm"], r["component"], r["evaluation_support"]): r for r in scored}
    contrasts = []
    comparisons = (
        [("HRP", "final", arm, "final") for arm in ARMS if arm != "HRP"]
        + [(arm, "final", "image_baseline", "baseline") for arm in ARMS]
        + [
            (arm, "transfer", "historical_HRP", "transfer")
            for arm in ARMS
            if "R" in arm.split("_")[0]
        ]
    )
    for support in SUPPORTS:
        for left_arm, left_component, right_arm, right_component in comparisons:
            left = lookup[(left_arm, left_component, support)]
            right = lookup[(right_arm, right_component, support)]
            _assert_matched_support(left, right)
            contrasts.append(
                {
                    "patient_id": sample.inputs.patient_id,
                    "specimen_id": sample.inputs.specimen_id,
                    "outer_fold": left["outer_fold"],
                    "biopsy_id": left["biopsy_id"],
                    "evaluation_support": support,
                    "contrast": f"{left_arm}:{left_component}-minus-{right_arm}:{right_component}",
                    **{
                        "delta_" + key: left[key] - right[key]
                        if left[key] is not None and right[key] is not None
                        else None
                        for key in ENDPOINTS
                    },
                }
            )
    return scored, gene_rows, contrasts


def run_residual_study(config_path, output):
    from .crossfit import crossfit_feature_probe, verify_deployment_probe
    from .residual_train import FittedResidualModel, fit_residual

    protocol, base = _protocol(config_path)
    prepared, source, baseline_root, metadata, h_inputs, folds, old_transfers = _source_contract(
        protocol, base
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(Path(config_path).read_bytes())
    source_files = {str(path): _sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    input_inventory = _input_inventory(prepared, metadata)
    _write(
        output / "identity.json",
        {
            "protocol_sha256": _sha(config_path),
            "source_file_sha256": source_files,
            "input_metadata_sha256": input_inventory,
            "prior_cohort_exposure": True,
            "test_targets_used_for_model_selection": False,
        },
    )
    genes, panel = base.gene_ids, base.normalization_gene_ids
    all_exports, all_scores, all_genes, all_contrasts, fits = [], [], [], [], []
    inner_count = reused_count = 0
    for fold, roles in folds.items():
        directory = output / f"fold-{fold}"
        directory.mkdir()
        historical = baseline_root / f"fold-{fold}"
        contract = protocol["baseline_folds"][str(fold)]
        if (
            _sha(historical / "probe.bundle/sha256.json") != contract["bundle_inventory_sha256"]
            or _sha(historical / "target_identity.json") != contract["target_identity_sha256"]
        ):
            raise ValueError("historical deployment baseline or target table identity changed")
        target_contract = json.loads((historical / "target_identity.json").read_text())
        if set(target_contract) != set(h_inputs):
            raise ValueError("historical target contract must cover exactly the prepared specimens")
        by_role = {
            role: [b for b, item in h_inputs.items() if roles[item.patient_id] == role]
            for role in ("train", "validation", "test")
        }
        target_receipt = {}

        def load_target(biopsy):
            path = source / f"fold-{fold}/targets/{biopsy}.npz"
            declaration = target_contract[biopsy]
            role = roles[h_inputs[biopsy].patient_id]
            if (
                Path(declaration["path"]).resolve() != path.resolve()
                or declaration["role"] != role
                or _sha(path) != declaration["sha256"]
            ):
                raise ValueError("reused target table identity or patient role differs")
            target_receipt[biopsy] = declaration
            return _sample_from_source(h_inputs[biopsy], path)

        samples = {b: load_target(b) for role in ("train", "validation") for b in by_role[role]}
        train = [samples[b] for b in by_role["train"]]
        validation_ids = sorted({h_inputs[b].patient_id for b in by_role["validation"]})
        baseline, baseline_audit = verify_deployment_probe(
            historical / "probe.bundle",
            train,
            genes,
            panel,
            alpha=protocol["alpha"],
            validation_patient_ids=validation_ids,
        )
        reused_count += 1
        _write(directory / "deployment_baseline_audit.json", baseline_audit)
        crossfit = crossfit_feature_probe(
            train,
            genes,
            panel,
            alpha=protocol["alpha"],
            n_splits=protocol["inner_splits"],
            seed=protocol["seed"],
            output=directory / "crossfit",
        )
        inner_count += len(crossfit.models)
        shared_identity = _shared_fit_identity(crossfit, baseline)
        pool_inputs = [_inputs(prepared, b, "HR")[0] for b in by_role["train"]]
        pooled, pool_metadata = typed_training_reference(
            pool_inputs,
            gene_ids=genes,
            embedding_gene_ids=base.state_gene_ids,
            max_states=32,
        )
        np.savez(
            directory / "pooled_reference.npz",
            **{
                field.name: np.asarray(getattr(pooled, field.name))
                for field in fields(RNAReference)
            },
            metadata_json=np.asarray(json.dumps(pool_metadata, sort_keys=True)),
        )
        _write(directory / "pooled_reference.json", pool_metadata)
        fold_exports = []
        for biopsy in by_role["test"]:
            path = directory / f"test-{biopsy}.baseline.npz"
            ids = {
                "outer_fold": fold,
                "biopsy_id": biopsy,
                "arm": "image_baseline",
                "component": "baseline",
                "seed": 0,
            }
            _export_probe(baseline, h_inputs[biopsy], path, ids)
            fold_exports.append({**ids, "path": str(path.resolve()), "sha256": _sha(path)})
        for arm in ARMS:
            print(f"joint residual: fold={fold} arm={arm}", flush=True)
            model = _arm_config(base, arm)
            current = {
                b: _inputs(prepared, b, arm, pooled=pooled)[0]
                for role in ("train", "validation")
                for b in by_role[role]
            }
            preparation = {
                "protocol_sha256": _sha(config_path),
                "outer_fold": fold,
                "arm": arm,
                "shared_baseline_sha256": shared_identity,
                "fold_roles": roles,
                "source_prepared_sha256": protocol["prepared_sha256"],
            }
            if arm == "HRP_generic":
                preparation["reference_pool"] = pool_metadata
            fitted = fit_residual(
                [replace(samples[b], inputs=current[b]) for b in by_role["train"]],
                [replace(samples[b], inputs=current[b]) for b in by_role["validation"]],
                model,
                training_baselines=crossfit,
                baseline=baseline,
                seed=0,
                reference_override=pooled if arm == "HRP_generic" else None,
                preparation=preparation,
                **protocol["training"],
            )
            bundle = directory / f"{arm}.bundle"
            fitted.save(bundle)
            restored = FittedResidualModel.load(bundle)
            check = current[by_role["train"][0]]
            if not np.array_equal(
                _array(fitted.predict(check).normalized_log_expression),
                _array(restored.predict(check).normalized_log_expression),
            ):
                raise ValueError("residual bundle reload changed normalized-log predictions")
            fits.append(
                {
                    "outer_fold": fold,
                    "arm": arm,
                    "selection": fitted.selection,
                    "bundle": str(bundle.resolve()),
                    "shared_baseline_sha256": shared_identity,
                }
            )
            for biopsy in by_role["test"]:
                inputs, _ = _inputs(prepared, biopsy, arm, pooled=pooled)
                prediction = restored.predict(inputs)
                ids = {
                    "outer_fold": fold,
                    "biopsy_id": biopsy,
                    "arm": arm,
                    "seed": 0,
                    **{
                        key: fitted.selection[key]
                        for key in (
                            "selected_epoch",
                            "selected_zero_residual",
                            "mapper_training_status",
                            "mapper_parameters_changed",
                        )
                    },
                }
                path = directory / f"test-{biopsy}.{arm}.residual.npz"
                save_residual_prediction(
                    prediction,
                    path,
                    {
                        **ids,
                        "component": "final",
                        "normalization_gene_ids": list(panel),
                        "encoder_manifest_sha256": model.image_encoder_manifest_sha256,
                        "residual_bound": protocol["training"]["residual_bound"],
                    },
                )
                fold_exports.append(
                    {**ids, "component": "final", "path": str(path.resolve()), "sha256": _sha(path)}
                )
                if prediction.linear_reference_transfer is not None:
                    path = directory / f"test-{biopsy}.{arm}.transfer.npz"
                    save_prediction(
                        prediction.linear_reference_transfer,
                        path,
                        {
                            **ids,
                            "component": "transfer",
                            "test_targets_loaded": False,
                            "prediction_scale": "linear_relative_expression",
                        },
                    )
                    fold_exports.append(
                        {
                            **ids,
                            "component": "transfer",
                            "path": str(path.resolve()),
                            "sha256": _sha(path),
                        }
                    )
            if _shared_fit_identity(crossfit, baseline) != shared_identity:
                raise ValueError("an arm mutated the shared crossfit or deployment baseline")
        _write(
            directory / "test_predictions_complete.json",
            {
                "test_targets_loaded": False,
                "test_biopsies": by_role["test"],
                "neural_fits": len(ARMS),
                "inner_readouts": len(crossfit.models),
                "reused_deployment_readouts": 1,
                "shared_baseline_sha256": shared_identity,
                "prediction_sha256": {Path(e["path"]).name: e["sha256"] for e in fold_exports},
            },
        )
        # The held-out expression values first enter this workflow below.
        for biopsy in by_role["test"]:
            sample = load_target(biopsy)
            coverage_inputs, _ = _inputs(prepared, biopsy, "HP")
            records, rows, contrasts = _score_case(
                [entry for entry in fold_exports if entry["biopsy_id"] == biopsy],
                sample,
                coverage_inputs,
                genes,
                panel,
                old_transfers[(fold, biopsy)],
                directory,
            )
            all_scores.extend(records)
            all_genes.extend(rows)
            all_contrasts.extend(contrasts)
        _write(directory / "target_identity.json", target_receipt)
        if any(_sha(record["path"]) != record["sha256"] for record in target_receipt.values()):
            raise ValueError("a reused target table changed during the residual fold")
        all_exports.extend(fold_exports)
    if (len(fits), inner_count, reused_count) != (30, 15, 5):
        raise ValueError("completed residual study differs from its fixed fit budget")
    _check_input_inventory(prepared, metadata, input_inventory)
    if any(_sha(path) != digest for path, digest in source_files.items()):
        raise ValueError("residual source code changed during execution")
    summary = {
        "schema": "heir.residual_study_result.v1",
        "neural_fits": len(fits),
        "inner_readouts": inner_count,
        "reused_deployment_readouts": reused_count,
        "additional_refits": 0,
        "selections": fits,
        "export_counts": {
            component: sum(e["component"] == component for e in all_exports)
            for component in ("baseline", "final", "transfer")
        },
        "prediction_scale": SCALE,
        "transfer_scale": "linear_relative_expression",
        "scores": {},
        "paired_contrasts": {},
        "prior_cohort_exposure": True,
        "scientific_qualification": False,
        "independent_confirmation": False,
        "interpretation": (
            "direct normalized-log prediction; transfer inspected separately; "
            "no validated molecular-state placement"
        ),
    }
    for support in SUPPORTS:
        summary["scores"][support] = {}
        for arm, component in sorted({(r["arm"], r["component"]) for r in all_scores}):
            selected = [
                r
                for r in all_scores
                if (r["arm"], r["component"], r["evaluation_support"]) == (arm, component, support)
            ]
            summary["scores"][support][f"{arm}:{component}"] = _summary(selected)
        summary["paired_contrasts"][support] = {}
        for contrast in sorted({r["contrast"] for r in all_contrasts}):
            selected = [
                {**r, **{key: r["delta_" + key] for key in ENDPOINTS}}
                for r in all_contrasts
                if r["contrast"] == contrast and r["evaluation_support"] == support
            ]
            summary["paired_contrasts"][support][contrast] = _summary(selected)
    _write(output / "exports.json", all_exports)
    _write(output / "summary.json", summary)
    _tsv(
        output / "patient_scores.tsv",
        [
            {
                key: row[key]
                for key in (
                    "patient_id",
                    "specimen_id",
                    "outer_fold",
                    "biopsy_id",
                    "arm",
                    "component",
                    "evaluation_support",
                    "headline_status",
                    *ENDPOINTS,
                )
            }
            for row in all_scores
        ],
    )
    _tsv(output / "gene_scores.tsv", all_genes)
    _tsv(output / "paired_contrasts.tsv", all_contrasts)
    _tsv(
        output / "output_contracts.tsv",
        [
            {
                **{key: row[key] for key in ("outer_fold", "biopsy_id", "arm", "component")},
                **row["output_contract"],
            }
            for row in all_scores
            if row["evaluation_support"] == SUPPORTS[0] and "output_contract" in row
        ],
    )
    return summary
