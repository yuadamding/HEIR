"""Patient-only cross-fitted image baselines and audited deployment reuse.

Inner predictions never use the predicted patient's outcomes or image scaling.
The saved v0.11 global out-of-fold predictions are not inputs to this helper.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .probe_workflow import (
    FrozenFeatureProbe,
    _feature_identity,
    _ridge_design,
    _sample_from_source,
    _write,
    fit_fixed_feature_probe,
    patient_balanced_weights,
)
from .protein_features import load_joint_inputs
from .study import _sha


def _array_hash(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((values.dtype.str, values.shape)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _input_identity(inputs):
    return {
        "patient_id": inputs.patient_id,
        "specimen_id": inputs.specimen_id,
        "region_ids": list(inputs.cell_ids),
        "prediction_unit": inputs.prediction_unit,
        "coordinate_units": inputs.coordinate_units,
        "encoder_manifest_sha256": inputs.image_encoder_manifest_sha256,
        "coordinates_sha256": _array_hash(inputs.coordinates),
        "image_features_sha256": _array_hash(inputs.image_features),
    }


def _target_identity(sample):
    return {
        "patient_id": sample.targets.patient_id,
        "specimen_id": sample.targets.specimen_id,
        "observation_ids": list(sample.targets.observation_ids),
        "gene_ids": list(sample.targets.gene_ids),
        "observation_groups": list(sample.observation_groups),
        "operator_cell_ids": list(sample.observation.cell_ids),
        "operator_observation_ids": list(sample.observation.observation_ids),
        **{
            name + "_sha256": _array_hash(getattr(sample.targets, name))
            for name in ("expression", "observed")
        },
        **{
            name + "_sha256": _array_hash(getattr(sample.observation, name))
            for name in ("rows", "columns", "weights")
        },
    }


@dataclass(frozen=True)
class CrossFittedBaseline:
    predictions: dict
    region_ids: dict
    models: tuple[FrozenFeatureProbe, ...]
    folds: tuple[dict, ...]
    training_patient_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    normalization_gene_ids: tuple[str, ...]
    encoder_manifest_sha256: str
    source_records: tuple[dict, ...]

    def __post_init__(self):
        if len(self.models) != 3 or len(self.folds) != 3:
            raise ValueError("cross-fitted baselines require exactly three inner readouts")
        patients = set(self.training_patient_ids)
        heldout = []
        for index, (model, fold) in enumerate(zip(self.models, self.folds)):
            excluded = set(fold["heldout_patient_ids"])
            if (
                fold["inner_fold"] != index
                or not excluded
                or model.training_patient_ids != tuple(fold["training_patient_ids"])
                or set(model.training_patient_ids) != patients - excluded
                or model.validation_patient_ids
                or model.alpha != 10.0
                or model.gene_ids != self.gene_ids
                or model.normalization_gene_ids != self.normalization_gene_ids
                or model.encoder_manifest_sha256 != self.encoder_manifest_sha256
            ):
                raise ValueError(
                    "cross-fitted models do not match their patient-exclusion receipts"
                )
            heldout.extend(fold["heldout_patient_ids"])
        if len(heldout) != len(patients) or set(heldout) != patients:
            raise ValueError("every outer-training patient must be held out exactly once")
        keys = [(r["patient_id"], r["specimen_id"]) for r in self.source_records]
        if (
            len(set(keys)) != len(keys)
            or set(keys) != set(self.predictions)
            or set(keys) != set(self.region_ids)
        ):
            raise ValueError("cross-fitted predictions must have unique bound specimen axes")
        if {key[0] for key in keys} != patients:
            raise ValueError("cross-fitted predictions omit an outer-training patient")
        for key, record in zip(keys, self.source_records):
            values = self.predictions[key]
            if (
                key[0] not in self.folds[record["inner_fold"]]["heldout_patient_ids"]
                or tuple(record["input_identity"]["region_ids"]) != self.region_ids[key]
                or values.dtype != np.float64
                or values.shape != (len(self.region_ids[key]), len(self.gene_ids))
                or not np.isfinite(values).all()
                or _array_hash(values) != record["prediction_sha256"]
            ):
                raise ValueError(
                    "cross-fitted prediction values or contributing model identity differ"
                )
            values.setflags(write=False)

    def prediction_for(self, inputs):
        key = (inputs.patient_id, inputs.specimen_id)
        source = next(
            (r for r in self.source_records if (r["patient_id"], r["specimen_id"]) == key), None
        )
        if source is None or _input_identity(inputs) != source["input_identity"]:
            raise ValueError(
                "cross-fitted prediction requires the exact held-out image features and axes"
            )
        if tuple(inputs.cell_ids) != self.region_ids[key]:
            raise ValueError("cross-fitted prediction region order differs")
        values = self.predictions[key]
        if _array_hash(values) != source["prediction_sha256"]:
            raise ValueError("cross-fitted baseline values changed after construction")
        return values.copy()

    def to_metadata(self):
        return json.loads(
            json.dumps(
                {
                    "schema": "heir.patient_crossfit_baseline.v1",
                    "training_patient_ids": list(self.training_patient_ids),
                    "gene_ids": list(self.gene_ids),
                    "normalization_gene_ids": list(self.normalization_gene_ids),
                    "encoder_manifest_sha256": self.encoder_manifest_sha256,
                    "split_rule": "sorted_patient_ids_round_robin",
                    "seed": 0,
                    "randomness_used": False,
                    "fixed_alpha": self.models[0].alpha,
                    "inner_readouts": len(self.models),
                    "candidate_solutions": len(self.models),
                    "penalty_searches": 0,
                    "folds": list(self.folds),
                    "source_records": list(self.source_records),
                }
            )
        )


def crossfit_feature_probe(
    training, gene_ids, normalization_gene_ids, *, alpha=10.0, n_splits=3, seed=0, output=None
):
    """Fit three fixed-penalty readouts on outer-training patients only."""
    training = list(training)
    patients = tuple(sorted({sample.inputs.patient_id for sample in training}))
    if seed != 0 or n_splits != 3 or alpha != 10.0 or len(patients) < n_splits:
        raise ValueError(
            "cross-fitting requires fixed alpha10, three patient folds and declared seed0"
        )
    encoder, _, _, _ = _feature_identity(training)
    assignment = {patient: index % n_splits for index, patient in enumerate(patients)}
    directory = Path(output) if output is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    models, folds, records, predictions, axes = [], [], [], {}, {}
    for inner in range(n_splits):
        fit_samples = [
            sample for sample in training if assignment[sample.inputs.patient_id] != inner
        ]
        heldout = [sample for sample in training if assignment[sample.inputs.patient_id] == inner]
        fitted = fit_fixed_feature_probe(fit_samples, gene_ids, normalization_gene_ids, alpha=alpha)
        receipt = {
            "inner_fold": inner,
            "training_patient_ids": list(fitted.training_patient_ids),
            "heldout_patient_ids": sorted({sample.inputs.patient_id for sample in heldout}),
            "alpha": alpha,
        }
        if directory is not None:
            bundle = directory / f"inner-{inner}.bundle"
            fitted.save(bundle)
            receipt["model_inventory_sha256"] = _sha(bundle / "sha256.json")
        models.append(fitted)
        folds.append(receipt)
        for sample in heldout:
            key = (sample.inputs.patient_id, sample.inputs.specimen_id)
            prediction = fitted.predict(sample.inputs)
            prediction.setflags(write=False)
            predictions[key], axes[key] = prediction, tuple(sample.inputs.cell_ids)
            record = {
                "patient_id": key[0],
                "specimen_id": key[1],
                "inner_fold": inner,
                "input_identity": _input_identity(sample.inputs),
                "target_identity": _target_identity(sample),
                "prediction_sha256": _array_hash(prediction),
            }
            if directory is not None:
                path = directory / f"prediction-{len(records):03d}.npz"
                np.savez(
                    path,
                    normalized_log_expression=prediction,
                    region_ids=np.asarray(sample.inputs.cell_ids),
                )
                record["prediction_file"] = path.name
                record["prediction_file_sha256"] = _sha(path)
            records.append(record)
    result = CrossFittedBaseline(
        predictions,
        axes,
        tuple(models),
        tuple(folds),
        patients,
        tuple(gene_ids),
        tuple(normalization_gene_ids),
        encoder,
        tuple(records),
    )
    if directory is not None:
        _write(directory / "crossfit.json", result.to_metadata())
        _write(
            directory / "sha256.json",
            {
                path.relative_to(directory).as_posix(): _sha(path)
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            },
        )
    return result


def load_crossfit_baseline(directory):
    directory = Path(directory)
    inventory = json.loads((directory / "sha256.json").read_text())
    for name, digest in inventory.items():
        path = directory / name
        if path.resolve().is_relative_to(directory.resolve()) is False or _sha(path) != digest:
            raise ValueError("cross-fitted baseline inventory changed")
    metadata = json.loads((directory / "crossfit.json").read_text())
    if metadata["schema"] != "heir.patient_crossfit_baseline.v1":
        raise ValueError("unsupported cross-fitted baseline schema")
    models = tuple(FrozenFeatureProbe.load(directory / f"inner-{i}.bundle") for i in range(3))
    predictions, axes = {}, {}
    for record in metadata["source_records"]:
        key = (record["patient_id"], record["specimen_id"])
        with np.load(directory / record["prediction_file"], allow_pickle=False) as archive:
            predictions[key] = archive["normalized_log_expression"].copy()
            predictions[key].setflags(write=False)
            axes[key] = tuple(archive["region_ids"])
        if _array_hash(predictions[key]) != record["prediction_sha256"]:
            raise ValueError("cross-fitted prediction hash differs")
    return CrossFittedBaseline(
        predictions,
        axes,
        models,
        tuple(metadata["folds"]),
        tuple(metadata["training_patient_ids"]),
        tuple(metadata["gene_ids"]),
        tuple(metadata["normalization_gene_ids"]),
        metadata["encoder_manifest_sha256"],
        tuple(metadata["source_records"]),
    )


def verify_deployment_probe(
    bundle,
    training,
    gene_ids,
    normalization_gene_ids,
    *,
    alpha=10.0,
    validation_patient_ids=(),
    probe_study=None,
):
    """Verify a retained v0.11 readout against sources and fixed-alpha normal equations.

    No decomposition, linear solve, model fit or validation selection is performed.
    """
    bundle, training = Path(bundle), list(training)
    study = Path(probe_study) if probe_study is not None else bundle.parents[2]
    if bundle.parent.parent.parent.resolve() != study.resolve():
        raise ValueError("deployment readout is not inside the declared original probe study")
    model = FrozenFeatureProbe.load(bundle)
    identity = _feature_identity(training)
    if (
        model.alpha != alpha
        or alpha != 10.0
        or model.training_patient_ids != tuple(sorted({s.inputs.patient_id for s in training}))
        or model.validation_patient_ids != tuple(sorted(validation_patient_ids))
        or model.gene_ids != tuple(gene_ids)
        or model.normalization_gene_ids != tuple(normalization_gene_ids)
        or identity
        != (
            model.encoder_manifest_sha256,
            model.prediction_unit,
            model.coordinate_units,
            len(model.feature_mean),
        )
    ):
        raise ValueError(
            "retained deployment probe partition, penalty, feature or output contract differs"
        )
    backbone = bundle.parent.parent.name
    original = json.loads((study / "identity.json").read_text())
    source = original["sources"][backbone]
    prepared = Path(source["prepared_directory"])
    target_sources = json.loads((bundle.parent / "target_identity.json").read_text())
    current = {(s.inputs.patient_id, s.inputs.specimen_id): s for s in training}
    verified, records = set(), []
    for biopsy, target_source in target_sources.items():
        if target_source["role"] != "train":
            continue
        metadata_path = prepared / "inputs" / biopsy / "metadata.json"
        if _sha(metadata_path) != source["input_metadata_sha256"][biopsy]:
            raise ValueError("original deployment training input metadata changed")
        inputs, _ = load_joint_inputs(metadata_path.parent, modalities=("H",))
        key = (inputs.patient_id, inputs.specimen_id)
        if key not in current or _input_identity(inputs) != _input_identity(current[key].inputs):
            raise ValueError(
                "deployment probe training image values or axes differ from original sources"
            )
        if _sha(target_source["path"]) != target_source["sha256"]:
            raise ValueError("original deployment training target file changed")
        sample = _sample_from_source(inputs, target_source["path"])
        if _target_identity(sample) != _target_identity(current[key]):
            raise ValueError("deployment probe training targets or observation operators differ")
        verified.add(key)
        records.append(
            {
                "patient_id": key[0],
                "specimen_id": key[1],
                "input_metadata_sha256": _sha(metadata_path),
                "target_file_sha256": target_source["sha256"],
                "consumed_input_sha256": _json_hash(_input_identity(inputs)),
                "consumed_target_sha256": _json_hash(_target_identity(sample)),
            }
        )
    if verified != set(current):
        raise ValueError(
            "original deployment training sources do not cover the exact current partition"
        )
    mean, scale, x, y, observed, labels = _ridge_design(training, gene_ids, normalization_gene_ids)
    if not np.array_equal(mean, model.feature_mean) or not np.array_equal(
        scale, model.feature_scale
    ):
        raise ValueError(
            "deployment training-only feature scaler differs from the retained readout"
        )
    residuals = []
    for column, gene in enumerate(gene_ids):
        weights = patient_balanced_weights(labels, observed[:, column])
        truth = np.where(observed[:, column], y[:, column], 0.0)
        error = x @ model.coefficients[:, column] + model.intercept[column] - truth
        slope = x.T @ (weights * error) + alpha * model.coefficients[:, column]
        intercept = float(weights @ error)
        denominator = max(
            1.0,
            float(np.abs(x.T @ (weights * truth)).max()),
            float(np.abs(alpha * model.coefficients[:, column]).max()),
        )
        relative = float(np.abs(slope).max()) / denominator
        if relative > 1e-9 or abs(intercept) > 1e-9:
            raise ValueError(
                "retained deployment coefficients fail the fixed-alpha normal equations"
            )
        residuals.append(
            {
                "gene_id": gene,
                "scaled_slope_residual_max": relative,
                "intercept_residual": intercept,
            }
        )
    receipt = {
        "schema": "heir.deployment_probe_reuse.v1",
        "alpha": alpha,
        "training_patient_ids": list(model.training_patient_ids),
        "validation_patient_ids": list(model.validation_patient_ids),
        "original_probe_identity_sha256": _sha(study / "identity.json"),
        "original_bundle_inventory_sha256": _sha(bundle / "sha256.json"),
        "original_target_identity_sha256": _sha(bundle.parent / "target_identity.json"),
        "source_records": records,
        "normal_equation_residuals": residuals,
        "normal_equation_tolerance": 1e-9,
        "feature_scaler_exact": True,
        "new_readouts_fitted": 0,
        "decomposition_or_linear_solve_performed": False,
    }
    return model, receipt
