"""Cross-fitted residual supervision with a frozen float64 deployment baseline."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .joint_model import JointModelConfig
from .probe_workflow import (
    SCALE,
    SCORE,
    FrozenFeatureProbe,
    _region_rows,
    _summary,
    _truth,
    score_probe,
)
from .protein_features import ProteinFeatureScaler
from .residual_model import ResidualModel
from .study import _reference_hash, _save_pool, load_pooled_reference
from .train import FittedJointModel, _centered_expression_loss, _expression_loss


def _image_only(inputs):
    return replace(inputs, reference=None, protein_features=None)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class FittedResidualModel:
    model: ResidualModel
    baseline: FrozenFeatureProbe
    transformer: FittedJointModel
    history: list[dict]
    preparation: dict
    training_config: dict
    selection: dict
    reference_override: object = None

    @property
    def training_patient_ids(self):
        return self.transformer.training_patient_ids

    @property
    def validation_patient_ids(self):
        return self.transformer.validation_patient_ids

    def transform(self, inputs):
        if self.reference_override is not None:
            if inputs.reference is not None and _reference_hash(
                inputs.reference
            ) != _reference_hash(self.reference_override):
                raise ValueError(
                    "generic residual input reference differs from its bound training pool"
                )
            inputs = replace(inputs, reference=self.reference_override)
        return self.transformer.transform(inputs)

    def predict(self, inputs):
        baseline = self.baseline.predict(_image_only(inputs))
        self.model.eval()
        with torch.no_grad():
            return self.model(
                self.transform(inputs),
                baseline,
                residual_enabled=self.selection["residual_enabled"],
            )

    def predict_diagnostics(self, inputs):
        return {
            key: value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
            for key, value in self.predict(inputs).diagnostics.items()
        }

    def save(self, directory):
        if any(not torch.isfinite(value).all() for value in self.model.state_dict().values()):
            raise ValueError("cannot save nonfinite selected residual state")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.baseline.save(directory / "baseline.bundle")
        self.transformer.save(directory / "mapper.bundle")
        np.savez(
            directory / "residual_head.npz",
            **{
                key: value.detach().cpu().numpy()
                for key, value in self.model.residual_head.state_dict().items()
            },
        )
        record = {
            "schema": "heir.joint_residual.bundle.v1",
            "model_family": self.model.family,
            "model_config": self.model.config.to_dict(),
            "prediction_scale": SCALE,
            "normalize_prediction_again": False,
            "normalized_log_residual_bound": self.model.residual_bound,
            "legacy_expression_decoder_active": False,
            "training_patient_ids": self.training_patient_ids,
            "validation_patient_ids": self.validation_patient_ids,
            "training_config": self.training_config,
            "selection": self.selection,
            "history": self.history,
            "preparation": self.preparation,
            "reference_override": self.reference_override is not None,
            "scientific_qualification": False,
        }
        (directory / "bundle.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        hashes = {
            str(p.relative_to(directory)): _sha(p) for p in directory.rglob("*") if p.is_file()
        }
        (directory / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        if self.reference_override is not None:
            _save_pool(directory, self.reference_override, self.preparation["reference_pool"])

    @classmethod
    def load(cls, directory, *, device="cpu"):
        directory = Path(directory)
        inventory = json.loads((directory / "sha256.json").read_text())
        required = {
            "bundle.json",
            "residual_head.npz",
            "baseline.bundle/probe.json",
            "baseline.bundle/readout.npz",
            "baseline.bundle/sha256.json",
            "mapper.bundle/bundle.json",
            "mapper.bundle/weights.npz",
            "mapper.bundle/feature_transform.npz",
            "mapper.bundle/sha256.json",
        }
        if set(inventory) not in (required, required | {"pooled_reference.npz"}):
            raise ValueError("residual bundle integrity inventory differs")
        for name, digest in inventory.items():
            if _sha(directory / name) != digest:
                raise ValueError("residual bundle integrity mismatch")
        record = json.loads((directory / "bundle.json").read_text())
        if (
            record.get("schema") != "heir.joint_residual.bundle.v1"
            or record.get("model_family") != ResidualModel.family
            or record.get("prediction_scale") != SCALE
            or record.get("normalize_prediction_again") is not False
            or record.get("legacy_expression_decoder_active") is not False
        ):
            raise ValueError("residual bundle scale or family differs")
        baseline = FrozenFeatureProbe.load(directory / "baseline.bundle")
        transformer = FittedJointModel.load(directory / "mapper.bundle", device=device)
        model = ResidualModel(
            JointModelConfig.from_dict(record["model_config"]),
            residual_bound=record["normalized_log_residual_bound"],
        )
        if model.config != transformer.model.config:
            raise ValueError("residual mapper configuration differs from its transform bundle")
        model.mapper.load_state_dict(transformer.model.state_dict())
        transformer.model = model.mapper
        with np.load(directory / "residual_head.npz", allow_pickle=False) as archive:
            if any(not np.isfinite(archive[key]).all() for key in archive):
                raise ValueError("nonfinite residual weights")
            model.residual_head.load_state_dict(
                {key: torch.from_numpy(archive[key].copy()) for key in archive}
            )
        selection = record["selection"]
        epoch = selection.get("selected_epoch")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or selection.get("selected_zero_residual") != (epoch == 0)
            or selection.get("residual_enabled") != (epoch > 0)
            or not isinstance(selection.get("residual_enabled"), bool)
            or not isinstance(selection.get("selected_zero_residual"), bool)
        ):
            raise ValueError("residual bundle must retain its selected zero-residual candidate")
        if (
            tuple(record["training_patient_ids"]) != transformer.training_patient_ids
            or tuple(record["validation_patient_ids"]) != transformer.validation_patient_ids
        ):
            raise ValueError("residual bundle patient roles differ from fitted transforms")
        _validate_baseline(baseline, model.config, transformer.training_patient_ids)
        if not np.array_equal(
            transformer.feature_mean, baseline.feature_mean
        ) or not np.array_equal(transformer.feature_scale, baseline.feature_scale):
            raise ValueError("residual image transform differs from the frozen deployment baseline")
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("nonfinite selected residual state")
        reference = load_pooled_reference(directory) if record["reference_override"] else None
        if record["reference_override"] != ("pooled_reference.npz" in inventory):
            raise ValueError("residual pooled-reference inventory differs")
        model.to(device).eval()
        return cls(
            model,
            baseline,
            transformer,
            record["history"],
            record["preparation"],
            record["training_config"],
            selection,
            reference,
        )


def _validate_baseline(baseline, config, training_ids):
    if (
        tuple(baseline.gene_ids) != config.gene_ids
        or tuple(baseline.normalization_gene_ids) != config.normalization_gene_ids
        or baseline.encoder_manifest_sha256 != config.image_encoder_manifest_sha256
        or set(baseline.training_patient_ids) != set(training_ids)
        or baseline.feature_mean.shape != (config.feature_dim,)
    ):
        raise ValueError(
            "deployment baseline must match residual axes, encoder and training patients"
        )


def _validate_crossfit(crossfit, samples, config, baseline):
    from .crossfit import CrossFittedBaseline, _target_identity

    if not isinstance(crossfit, CrossFittedBaseline):
        raise ValueError("residual training requires a verified CrossFittedBaseline object")
    patients = {s.inputs.patient_id for s in samples}
    if set(crossfit.training_patient_ids) != patients or len(crossfit.folds) != len(
        crossfit.models
    ):
        raise ValueError("cross-fitted training baseline identities differ")
    heldout_models = {}
    for fold, model in zip(crossfit.folds, crossfit.models):
        heldout, training = set(fold["heldout_patient_ids"]), set(fold["training_patient_ids"])
        if (
            not heldout
            or training != patients - heldout
            or not heldout <= patients
            or training & heldout
            or heldout & set(heldout_models)
            or model.validation_patient_ids
            or model.alpha != baseline.alpha
            or fold["alpha"] != baseline.alpha
        ):
            raise ValueError(
                "cross-fit receipts must exclude each held-out patient's outcomes and scaler"
            )
        _validate_baseline(model, config, training)
        heldout_models.update({p: model for p in heldout})
    if set(heldout_models) != patients:
        raise ValueError(
            "cross-fit held-out partitions must cover every training patient exactly once"
        )
    source_records = {(r["patient_id"], r["specimen_id"]): r for r in crossfit.source_records}
    if len(source_records) != len(samples):
        raise ValueError("cross-fit source specimen inventory differs")
    for sample in samples:
        key = sample.inputs.patient_id, sample.inputs.specimen_id
        if key not in source_records or source_records[key]["target_identity"] != _target_identity(
            sample
        ):
            raise ValueError("cross-fit training target identity differs")
        supplied = crossfit.prediction_for(sample.inputs)
        expected = heldout_models[sample.inputs.patient_id].predict(_image_only(sample.inputs))
        if supplied.dtype != np.float64 or not np.array_equal(supplied, expected):
            raise ValueError("cross-fit values differ from the excluding model's exact predictions")
    metadata = crossfit.to_metadata()
    if (
        tuple(metadata["gene_ids"]) != config.gene_ids
        or tuple(metadata["normalization_gene_ids"]) != config.normalization_gene_ids
    ):
        raise ValueError("cross-fit baseline gene axes differ")
    return metadata


def _validation(fitted, samples):
    reports = []
    for sample in samples:
        prediction = fitted.predict(sample.inputs).normalized_log_expression.cpu().numpy()
        report, _ = score_probe(
            prediction,
            sample,
            fitted.model.config.gene_ids,
            fitted.model.config.normalization_gene_ids,
        )
        reports.append(report)
    result = _summary(reports)
    result["eligible"] = result[SCORE] is not None
    return result


def fit_residual(
    samples,
    validation_samples,
    model_config,
    *,
    training_baselines,
    baseline,
    epochs=100,
    learning_rate=1e-3,
    centered_weight=0.25,
    residual_weight=0.001,
    residual_bound=1.0,
    seed=0,
    device="cpu",
    preparation=None,
    reference_override=None,
):
    """Train against outer-training cross-fits; select using deployment predictions.

    Epoch zero competes with every trained epoch and wins exact ties. The frozen
    ridge arrays stay float64 throughout; no legacy expression normalizer sees
    the final output. There is no transfer or state-placement auxiliary loss.
    """
    samples, validation_samples = list(samples), list(validation_samples)
    if not samples or not validation_samples:
        raise ValueError("residual fitting requires separate training and validation specimens")
    for name, value, minimum in (("epochs", epochs, 1), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (
        ("learning_rate", learning_rate),
        ("centered_weight", centered_weight),
        ("residual_weight", residual_weight),
    ):
        if not np.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
            raise ValueError(f"invalid {name}")
    training_ids = tuple(sorted({s.inputs.patient_id for s in samples}))
    validation_ids = tuple(sorted({s.inputs.patient_id for s in validation_samples}))
    if set(training_ids) & set(validation_ids):
        raise ValueError("residual training and validation patients must be disjoint")
    for role in (samples, validation_samples):
        keys = [(s.inputs.patient_id, s.inputs.specimen_id) for s in role]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate residual specimen")
    units = {
        (s.inputs.prediction_unit, s.inputs.coordinate_units) for s in samples + validation_samples
    }
    if len(units) != 1:
        raise ValueError("residual samples must share geometry units")
    unit, coordinates = units.pop()
    _validate_baseline(baseline, model_config, training_ids)
    if (unit, coordinates) != (baseline.prediction_unit, baseline.coordinate_units):
        raise ValueError("deployment baseline geometry differs")
    crossfit_record = _validate_crossfit(training_baselines, samples, model_config, baseline)
    preparation = dict(preparation or {})
    preparation["crossfit_baseline"] = crossfit_record
    if reference_override is not None:
        pool = preparation.get("reference_pool", {})
        if (
            "R" not in model_config.modalities
            or pool.get("reference_sha256") != _reference_hash(reference_override)
            or set(pool.get("training_patient_ids", ())) != set(training_ids)
        ):
            raise ValueError("generic residual reference must bind the outer-training pool")
    torch.manual_seed(seed)
    model = ResidualModel(model_config, residual_bound=residual_bound).to(device)
    state_arrays = []
    if "R" in model_config.modalities:
        for sample in samples:
            reference = reference_override or sample.inputs.reference
            if reference is None:
                raise ValueError("RNA residual requires a reference")
            lookup = {g: j for j, g in enumerate(reference.embedding_gene_ids)}
            if not set(model_config.state_gene_ids) <= set(lookup):
                raise ValueError("residual RNA descriptor axis differs")
            state_arrays.append(
                reference.embedding[:, [lookup[g] for g in model_config.state_gene_ids]]
            )
    states = np.concatenate(state_arrays) if state_arrays else np.empty((0, 0))
    state_mean = states.mean(0) if len(states) else np.empty(0)
    state_scale = np.maximum(states.std(0), 1e-6) if len(states) else np.empty(0)
    scaler = (
        ProteinFeatureScaler.fit([s.inputs for s in samples])
        if "P" in model_config.modalities
        else None
    )
    settings = dict(
        epochs=epochs,
        learning_rate=learning_rate,
        centered_weight=centered_weight,
        residual_weight=residual_weight,
        residual_bound=residual_bound,
        seed=seed,
        device=device,
        prediction_unit=unit,
        coordinate_units=coordinates,
        prediction_scale=SCALE,
        transfer_weight=0.0,
        residual_regularization_rows="all_prediction_regions",
        image_transform="frozen_deployment_ridge_training_transform",
        state_transform="outer_training_reference_states_population_mean_sd",
    )
    transformer = FittedJointModel(
        model.mapper,
        baseline.feature_mean.copy(),
        baseline.feature_scale.copy(),
        state_mean,
        state_scale,
        scaler,
        training_ids,
        validation_ids,
        [],
        {},
        settings,
        {"correction_enabled": False},
    )
    fitted = FittedResidualModel(
        model,
        baseline,
        transformer,
        [],
        preparation,
        settings,
        {"residual_enabled": False},
        reference_override,
    )
    prepared = []
    for sample in samples:
        inputs = fitted.transform(sample.inputs)
        # The cross-fit implementation checks the raw H/region identity before any transform.
        image_prediction = training_baselines.prediction_for(sample.inputs)
        mapping = _region_rows(sample)
        truth, observed = _truth(sample, model_config.gene_ids, model_config.normalization_gene_ids)
        prepared.append((sample, inputs, image_prediction, mapping, truth, observed))
    for sample in validation_samples:
        fitted.transform(sample.inputs)
        _region_rows(sample)
    patient_specimens = {p: sum(s.inputs.patient_id == p for s in samples) for p in training_ids}
    best_key, best_state = None, None
    initial_mapper = {
        k: p.detach().clone() for k, p in model.mapper.named_parameters() if p.requires_grad
    }

    def consider(epoch):
        nonlocal best_key, best_state
        fitted.selection = {"residual_enabled": epoch > 0}
        report = _validation(fitted, validation_samples)
        row = {
            "epoch": epoch,
            "stage": "zero_residual" if epoch == 0 else "joint_residual",
            "validation": report,
        }
        fitted.history.append(row)
        if report["eligible"]:
            key = (report[SCORE], -report["centered_mse"], -report["normalized_log_mse"])
            if best_key is None or key > best_key:
                best_key = key
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                row["selected_at_epoch"] = True
                mapper_changed = any(
                    not torch.equal(p, initial_mapper[k])
                    for k, p in model.mapper.named_parameters()
                    if k in initial_mapper
                )
                selected = dict(
                    selected_epoch=epoch,
                    selected_zero_residual=epoch == 0,
                    residual_enabled=epoch > 0,
                    validation=report,
                    mapper_training_status="not_applicable_no_RNA"
                    if "R" not in model_config.modalities
                    else "untrained_selected_epoch_zero"
                    if epoch == 0
                    else "trained_through_residual"
                    if mapper_changed
                    else "unchanged_after_selected_training",
                    mapper_parameters_changed=mapper_changed,
                    policy=(
                        "validation_spatial_correlation_then_centered_error_then_total_mse; "
                        "earliest_exact_tie"
                    ),
                )
                return selected
        return None

    selected = consider(0)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        totals = np.zeros(3)
        for sample, inputs, image_prediction, mapping, truth, observed in prepared:
            output = model(inputs, image_prediction)
            p = output.normalized_log_expression[mapping]
            y = torch.as_tensor(truth, dtype=p.dtype, device=p.device)
            mask = torch.as_tensor(observed, dtype=torch.bool, device=p.device)
            mse = _expression_loss((p - y).square(), mask, sample.observation_groups)
            centered = _centered_expression_loss(p, y, mask, sample.observation_groups)
            penalty = output.residual.square().mean()
            weight = 1 / (len(training_ids) * patient_specimens[sample.inputs.patient_id])
            loss = weight * (mse + centered_weight * centered + residual_weight * penalty)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite normalized-log residual objective")
            loss.backward()
            totals += weight * np.asarray(
                [float(mse.detach()), float(centered.detach()), float(penalty.detach())]
            )
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("nonfinite residual gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite residual parameter update")
        candidate = consider(epoch)
        if candidate is not None:
            selected = candidate
        fitted.history[-1]["training"] = dict(
            zip(("final_mse", "centered_mse", "residual_l2"), totals.tolist())
        )
    if selected is None:
        raise ValueError("no validation-eligible residual checkpoint, including zero residual")
    model.load_state_dict(best_state)
    fitted.selection = selected
    model.eval()
    return fitted
