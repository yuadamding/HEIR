"""Target-free diagnostics of frozen features and saved joint-model representations."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .protein_features import load_joint_inputs
from .serialization import load_prediction
from .study import load_pooled_reference
from .train import FittedJointModel


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _distribution(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values):
        return {
            "n": 0,
            "mean": None,
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    quantiles = np.quantile(values, [0, 0.05, 0.5, 0.95, 1])
    return {
        "n": len(values),
        "mean": float(values.mean()),
        **dict(zip(("minimum", "p05", "median", "p95", "maximum"), map(float, quantiles))),
    }


def _variation(values, *, row_axis="prediction_region"):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("representation diagnostics require finite two-dimensional arrays")
    std = values.std(axis=0)
    return {
        "row_axis": row_axis,
        "n_rows": len(values),
        "feature_width": values.shape[1],
        "population_std_across_rows": _distribution(std),
        "mean_variance_across_rows": float(np.mean(std**2)) if len(std) else None,
        "constant_dimensions": int(np.count_nonzero(std == 0)),
        "row_l2_norm": _distribution(np.linalg.norm(values, axis=1)),
    }


def _tanh(values, preactivation):
    values, preactivation = np.asarray(values), np.asarray(preactivation)
    return {
        "threshold_absolute_output": 0.95,
        "fraction_absolute_output_at_least_threshold": float(np.mean(np.abs(values) >= 0.95)),
        "mean_local_derivative": float(np.mean(1 - values.astype(np.float64) ** 2)),
        "absolute_preactivation": _distribution(np.abs(preactivation)),
        "interpretation": (
            "descriptive activation saturation; not evidence of predictive information loss"
        ),
    }


def _similarity(values, types):
    values = np.asarray(values, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=1)
    valid = lengths > 0
    unit = np.divide(values, lengths[:, None], out=np.zeros_like(values), where=valid[:, None])
    rows, columns = np.triu_indices(len(values), 1)
    pair_valid = valid[rows] & valid[columns]
    cosine = np.sum(unit[rows] * unit[columns], axis=1)[pair_valid]
    same_type = (np.asarray(types)[rows] == np.asarray(types)[columns])[pair_valid]
    return {
        "all_distinct_state_pairs": _distribution(cosine),
        "within_broad_type_pairs": _distribution(cosine[same_type]),
        "between_broad_type_pairs": _distribution(cosine[~same_type]),
        "undefined_zero_norm_pairs": int((~pair_valid).sum()),
        "interpretation": "descriptor similarity; not biological state equivalence",
    }


def representation_diagnostics(fitted, inputs, *, expected_prediction=None):
    """Inspect one saved model and specimen without targets, optimization, or mutation.

    Variance is measured across all H&E prediction regions before assay
    observation/normalization. Different stage units are not commensurable.
    Optional expected predictions enforce exact reuse of historical outputs.
    """
    model, activations, hooks = fitted.model, {}, []
    transformed = fitted.transform(inputs)

    def capture(name):
        def hook(_module, _arguments, value):
            activations[name] = _numpy(value).copy()

        return hook

    for name, module in (
        ("image", model.image_encoder),
        ("fusion", model.fusion),
        ("RNA", model.state_encoder),
    ):
        if module is not None:
            hooks.append(module[0].register_forward_hook(capture(name)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction, diagnostics = model._forward(
                transformed, correction_enabled=fitted.selection["correction_enabled"]
            )
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    if expected_prediction is not None:
        for name in ("cell_ids", "gene_ids", "state_ids", "patient_id", "specimen_id"):
            if getattr(prediction, name) != getattr(expected_prediction, name):
                raise ValueError("representation replay differs from saved prediction identities")
        for name in ("expression", "transfer", "conditional_assignment", "log_correction"):
            if not np.array_equal(
                _numpy(getattr(prediction, name)), _numpy(getattr(expected_prediction, name))
            ):
                raise ValueError(f"representation replay changed saved {name}")
    values = {name: _numpy(value) for name, value in diagnostics.items()}
    stages, saturation, similarities, logits = {}, {}, {}, {}
    if inputs.image_features is not None:
        stages["raw_frozen_image_features"] = _variation(inputs.image_features)
        stages["training_standardized_image_features"] = _variation(transformed.image_features)
        stages["image_projection_h"] = _variation(values["image_embedding"])
        saturation["image_projection_h"] = _tanh(values["image_embedding"], activations["image"])
    stages["joint_fusion_z"] = _variation(values["fused_embedding"])
    saturation["joint_fusion_z"] = _tanh(values["fused_embedding"], activations["fusion"])
    if inputs.reference is not None:
        reference = inputs.reference
        order = [reference.embedding_gene_ids.index(g) for g in model.config.state_gene_ids]
        raw_descriptor = reference.embedding[:, order]
        standardized_descriptor = transformed.reference.embedding[:, order]
        stages["raw_RNA_descriptors"] = _variation(raw_descriptor, row_axis="reference_state")
        stages["training_standardized_RNA_descriptors"] = _variation(
            standardized_descriptor, row_axis="reference_state"
        )
        stages["encoded_RNA_descriptors"] = _variation(
            values["state_embedding"], row_axis="reference_state"
        )
        stages["pooled_type_descriptors"] = _variation(
            values["type_embedding"], row_axis="reference_broad_type"
        )
        saturation["encoded_RNA_descriptors"] = _tanh(values["state_embedding"], activations["RNA"])
        for name, descriptor in (
            ("raw", raw_descriptor),
            ("training_standardized", standardized_descriptor),
            ("encoded", values["state_embedding"]),
        ):
            similarities[name] = _similarity(descriptor, reference.state_types)
        similarities["pooled_types"] = _similarity(values["type_embedding"], values["type_ids"])
        for name, key in (
            ("broad_type_mixture_pi", "type_mixture"),
            ("within_type_mixture_omega", "within_type_mixture"),
            ("full_state_mixture_q", "state_mixture"),
        ):
            stages[name] = _variation(values[key])
        stages["linear_reference_transfer_qB"] = _variation(_numpy(prediction.transfer))
        if "type_logits" in values:
            stages["type_logits"] = _variation(values["type_logits"])
            stages["state_logits"] = _variation(values["state_logits"])
            logits["type_logit_std_across_types_per_region"] = _distribution(
                values["type_logits"].std(1)
            )
            logits["within_type_state_logit_std_per_region"] = {
                str(label): _distribution(
                    values["state_logits"][:, values["state_types"] == label].std(1)
                )
                for label in values["type_ids"]
            }
        else:
            logits["status"] = "constant_reference_control_has_no_active_compatibility_logits"
    stages["final_linear_expression"] = _variation(_numpy(prediction.expression))
    return {
        "schema": "heir.joint_representation_diagnostics.v1",
        "patient_id": inputs.patient_id,
        "specimen_id": inputs.specimen_id,
        "n_prediction_regions": len(inputs.cell_ids),
        "prediction_unit": inputs.prediction_unit,
        "coordinate_units": inputs.coordinate_units,
        "output_gene_ids": list(model.config.gene_ids),
        "descriptor_gene_ids": list(model.config.state_gene_ids),
        "selected_checkpoint": fitted.selection,
        "assignment_network": model.config.assignment_network,
        "protein_signal": model.config.protein_signal,
        "stages": stages,
        "tanh_saturation": saturation,
        "reference_descriptor_similarity": similarities,
        "assignment_logit_spread": logits,
        "saved_prediction_exact": expected_prediction is not None,
        "targets_used": False,
        "fitting_performed": False,
        "scope": "all_prediction_regions_before_assay_observation",
        "interpretation": (
            "Stage variances have different units; these diagnostics do not measure predictive "
            "information or prove that a bottleneck discards it."
        ),
    }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze_saved_representations(prepared, study, output, *, arms=None):
    """Publish new target-free JSON diagnostics; preserve every study/input artifact."""
    prepared, study, output = Path(prepared), Path(study), Path(output)
    identity = json.loads((study / "identity.json").read_text())
    if _sha(prepared / "prepared.json") != identity["prepared_sha256"]:
        raise ValueError("representation analysis prepared identity differs from saved study")
    exports = json.loads((study / "exports.json").read_text())
    available = {row["arm"] for row in exports if row["component"] == "full"}
    requested = available if arms is None else set(arms)
    if not requested or not requested <= available:
        raise ValueError("representation analysis requires existing saved full-prediction arms")
    records, models, payloads, seen = [], {}, {}, set()
    bound = {
        study / "identity.json": _sha(study / "identity.json"),
        study / "exports.json": _sha(study / "exports.json"),
    }
    for row in exports:
        if row["component"] != "full" or row["arm"] not in requested:
            continue
        key = (row["outer_fold"], row["arm"], row["seed"])
        prediction_key = (*key, row["biopsy_id"])
        if prediction_key in seen:
            raise ValueError("duplicate saved full-prediction identity")
        seen.add(prediction_key)
        bundle = study / f"fold-{row['outer_fold']}" / f"{row['arm']}-seed-{row['seed']}.bundle"
        if key not in models:
            fitted = FittedJointModel.load(bundle)
            pool = (
                load_pooled_reference(bundle) if fitted.preparation.get("reference_pool") else None
            )
            models[key] = fitted, pool
            for path in bundle.iterdir():
                if path.is_file():
                    bound[path] = _sha(path)
        fitted, pool = models[key]
        modalities = tuple(m for m in fitted.model.config.modalities if pool is None or m != "R")
        input_key = (row["biopsy_id"], modalities)
        if input_key not in payloads:
            directory = prepared / "inputs" / row["biopsy_id"]
            metadata = directory / "metadata.json"
            if _sha(metadata) != identity["input_metadata_sha256"][row["biopsy_id"]]:
                raise ValueError("representation analysis input metadata changed")
            payloads[input_key] = load_joint_inputs(directory, modalities=modalities)[0]
            bound[metadata] = _sha(metadata)
        inputs = payloads[input_key]
        if pool is not None:
            inputs = replace(inputs, reference=pool)
        path = Path(row["path"])
        expected, _ = load_prediction(path)
        bound[path] = _sha(path)
        record = representation_diagnostics(fitted, inputs, expected_prediction=expected)
        record.update(
            arm=row["arm"],
            outer_fold=row["outer_fold"],
            seed=row["seed"],
            biopsy_id=row["biopsy_id"],
            prediction_sha256=bound[path],
        )
        records.append(record)
    if any(_sha(path) != digest for path, digest in bound.items()):
        raise ValueError("saved representation input or model changed during analysis")
    summary = defaultdict(list)
    for record in records:
        summary[record["arm"]].append(record["patient_id"])
    result = {
        "schema": "heir.saved_joint_representations.v1",
        "prepared": str(prepared.resolve()),
        "study": str(study.resolve()),
        "arms": {
            name: {"n_predictions": len(patients), "patient_ids": sorted(set(patients))}
            for name, patients in sorted(summary.items())
        },
        "records": records,
        "historical_source_file_sha256": identity["source_file_sha256"],
        "analysis_source_sha256": _sha(__file__),
        "analysis_source_file_sha256": {
            name: _sha(Path(__file__).parent / name)
            for name in (
                "representations.py",
                "joint_model.py",
                "train.py",
                "schema.py",
                "protein_features.py",
            )
        },
        "bound_artifact_sha256": {str(path): digest for path, digest in bound.items()},
        "targets_used": False,
        "fitting_performed": False,
        "interpretation": (
            "Descriptive within-specimen representations; no new predictive performance estimate."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result
