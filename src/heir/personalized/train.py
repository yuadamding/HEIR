"""Whole-specimen optimization and portable inference bundles for cached features."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .evaluate import aligned_observations, normalized_observations
from .model import ModelConfig, PersonalizedModel
from .observation import ObservationOperator
from .schema import PredictionInputs, SpatialTargets


@dataclass(frozen=True)
class TrainingSample:
    inputs: PredictionInputs
    targets: SpatialTargets
    observation: ObservationOperator
    observation_groups: Optional[tuple[str, ...]] = None

    def __post_init__(self):
        if (self.inputs.patient_id, self.inputs.specimen_id) != (
            self.targets.patient_id,
            self.targets.specimen_id,
        ):
            raise ValueError("training sample input/target identity mismatch")
        if set(self.inputs.cell_ids) != set(self.observation.cell_ids):
            raise ValueError("training geometry does not describe the input cells")
        if self.observation_groups is not None:
            if isinstance(self.observation_groups, str):
                raise ValueError("observation_groups must be a sequence of group identifiers")
            groups = tuple(self.observation_groups)
            if len(groups) != len(self.observation.observation_ids) or any(
                not isinstance(group, str) or not group.strip() for group in groups
            ):
                raise ValueError(
                    "observation_groups must name a nonempty group for every observation row"
                )
            object.__setattr__(self, "observation_groups", groups)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    protein_weight: float = 0.1
    abundance_weight: float = 0.01
    residual_weight: float = 0.001
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self):
        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("epochs must be positive")
        for name in ("learning_rate", "protein_weight", "abundance_weight", "residual_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("initial training supports cpu or cuda devices")


@dataclass
class FittedModel:
    model: PersonalizedModel
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    training_patient_ids: tuple[str, ...]
    history: list[dict]
    preparation: dict
    optimizer_state: Optional[dict] = None
    training_config: Optional[dict] = None

    def transform(self, inputs: PredictionInputs) -> PredictionInputs:
        settings = self.training_config or {}
        if inputs.prediction_unit != settings.get(
            "prediction_unit", "nucleus"
        ) or inputs.coordinate_units != settings.get("coordinate_units", "um"):
            raise ValueError("prediction_unit or coordinate_units differs from fitted model")
        if inputs.image_features.shape[1:] != self.feature_mean.shape:
            raise ValueError("image feature dimension differs from fitted transform")
        return replace(
            inputs, image_features=(inputs.image_features - self.feature_mean) / self.feature_scale
        )

    def predict(self, inputs: PredictionInputs):
        self.model.eval()
        with torch.no_grad():
            return self.model(self.transform(inputs))

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(
            directory / "weights.npz",
            **{key: value.detach().cpu().numpy() for key, value in self.model.state_dict().items()},
        )
        np.savez(
            directory / "feature_transform.npz", mean=self.feature_mean, scale=self.feature_scale
        )
        metadata = {
            "schema": "heir.personalized.bundle.v2",
            "model_config": self.model.config.to_dict(),
            "training_patient_ids": list(self.training_patient_ids),
            "history": self.history,
            "training_config": self.training_config,
            "preparation": self.preparation,
            "fixed_mapper_initialized": self.model.fixed_mapper_initialized,
            "expression_scale": "relative",
            "scientific_qualification": False,
        }
        (directory / "bundle.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # Inference never deserializes this optimizer file. It is retained for future resumption.
        if self.optimizer_state is not None:
            torch.save(self.optimizer_state, directory / "optimizer.pt")
        hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir()
            if p.is_file()
        }
        (directory / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")

    @classmethod
    def load(cls, directory: str | Path, *, device: str = "cpu") -> FittedModel:
        directory = Path(directory)
        hashes = json.loads((directory / "sha256.json").read_text())
        required = {"bundle.json", "weights.npz", "feature_transform.npz"}
        if not required <= set(hashes):
            raise ValueError("incomplete inference bundle integrity inventory")
        for name, expected in hashes.items():
            if (
                Path(name).name != name
                or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected
            ):
                raise ValueError("inference bundle integrity mismatch")
        metadata = json.loads((directory / "bundle.json").read_text())
        if metadata["schema"] != "heir.personalized.bundle.v2":
            raise ValueError(
                "unsupported inference bundle schema; retrain legacy bundles with fixed "
                "normalization and conditional reference transfer"
            )
        model = PersonalizedModel(ModelConfig.from_dict(metadata["model_config"]))
        with np.load(directory / "weights.npz", allow_pickle=False) as weights:
            model.load_state_dict({key: torch.from_numpy(weights[key].copy()) for key in weights})
        model.fixed_mapper_initialized = metadata["fixed_mapper_initialized"]
        if model.fixed_mapper_initialized:
            model.freeze_mapper()
        with np.load(directory / "feature_transform.npz", allow_pickle=False) as data:
            mean, scale = data["mean"].copy(), data["scale"].copy()
        if mean.shape != (model.config.feature_dim,) or scale.shape != mean.shape:
            raise ValueError("invalid saved feature transform shape")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("invalid saved feature transform values")
        model.eval()
        model.to(device)
        return cls(
            model,
            mean,
            scale,
            tuple(metadata["training_patient_ids"]),
            metadata["history"],
            metadata["preparation"],
            training_config=metadata.get("training_config"),
        )


def fit(
    samples: Sequence[TrainingSample],
    model_config: ModelConfig,
    training_config: Optional[TrainingConfig] = None,
    *,
    baseline: Optional[FittedModel] = None,
    preparation: Optional[dict] = None,
) -> FittedModel:
    """Fit using full specimens, so the abundance penalty is never tile-local.

    Caller supplies training patients only. Context variants require the same fitted
    protein baseline and reuse its feature transform and frozen base assignment.
    """
    if not samples or not model_config.supervised_gene_ids:
        raise ValueError("training needs specimens and an explicit supervised gene panel")
    input_units = {(s.inputs.prediction_unit, s.inputs.coordinate_units) for s in samples}
    if len(input_units) != 1:
        raise ValueError("training samples must share prediction_unit and coordinate_units")
    prediction_unit, coordinate_units = next(iter(input_units))
    settings = training_config or TrainingConfig()
    specimen_ids = [(s.inputs.patient_id, s.inputs.specimen_id) for s in samples]
    if len(set(specimen_ids)) != len(samples):
        raise ValueError("duplicate training specimen")
    observed_genes = set()
    for sample in samples:
        observed_genes.update(
            gene
            for j, gene in enumerate(sample.targets.gene_ids)
            if sample.targets.observed[:, j].any()
        )
    unavailable = set(model_config.supervised_gene_ids) - observed_genes
    if unavailable:
        raise ValueError(
            f"supervised genes have no observed training targets: {sorted(unavailable)}"
        )
    torch.manual_seed(settings.seed)
    model = PersonalizedModel(model_config).to(settings.device)
    patient_ids = tuple(sorted({s.inputs.patient_id for s in samples}))
    context = model_config.variant.startswith(("generic_", "lr_"))
    if model_config.variant in {"generic_fixed", "lr_fixed"} and model_config.transfer_only:
        raise ValueError("fixed transfer-only is an inference diagnostic; fit the decoder first")
    if context:
        if baseline is None or set(baseline.training_patient_ids) != set(patient_ids):
            raise ValueError(
                "context training requires a protein baseline fitted on the same patients"
            )
        if baseline.preparation != (preparation or {}):
            raise ValueError("context baseline preparation settings differ")
        baseline_settings = baseline.training_config or {}
        if (
            baseline_settings.get("prediction_unit", "nucleus") != prediction_unit
            or baseline_settings.get("coordinate_units", "um") != coordinate_units
        ):
            raise ValueError("context baseline prediction_unit or coordinate_units differs")
        if baseline.model.config.normalization_gene_ids != model_config.normalization_gene_ids:
            raise ValueError("context baseline normalization panel differs")
        model.initialize_fixed_mapper(baseline.model)
        mean, scale = baseline.feature_mean.copy(), baseline.feature_scale.copy()
    else:
        if baseline is not None:
            raise ValueError("baseline initialization is only used for context configurations")
        features = np.concatenate([sample.inputs.image_features for sample in samples])
        mean, scale = features.mean(0), features.std(0)
        scale = np.maximum(scale, 1e-6)
    fitted = FittedModel(model, mean, scale, patient_ids, [], preparation or {})
    fitted.training_config = asdict(settings)
    fitted.training_config.update(
        prediction_unit=prediction_unit,
        coordinate_units=coordinate_units,
    )
    if baseline is not None:
        digest = hashlib.sha256()
        for key, value in sorted(baseline.model.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        fitted.training_config["baseline_model_sha256"] = digest.hexdigest()
    prepared = [replace(s, inputs=fitted.transform(s.inputs)) for s in samples]
    _validate_training_normalization(model, prepared)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=settings.learning_rate
    )
    for epoch in range(settings.epochs):
        model.train()
        losses = []
        # One optimizer update averages whole-specimen losses, equalized by patient.
        optimizer.zero_grad()
        patient_specimens = {
            p: sum(s.inputs.patient_id == p for s in prepared) for p in patient_ids
        }
        for sample in prepared:
            prediction = model(sample.inputs)
            predicted, truth, mask = normalized_observations(
                prediction,
                sample.targets,
                sample.observation,
                model_config.supervised_gene_ids,
                normalization_gene_ids=model_config.normalization_gene_ids,
            )
            if not mask.any():
                raise ValueError("training specimen has no observed supervised values")
            error = (predicted - truth).square()
            loss = _expression_loss(error, mask, sample.observation_groups)
            loss = loss + settings.protein_weight * model.protein_loss(sample.inputs, prediction)
            if model_config.variant != "he":
                abundance = _abundance_penalty(
                    prediction.conditional_assignment,
                    sample.inputs.reference.proportions,
                )
                loss = loss + settings.abundance_weight * abundance
            loss = loss + settings.residual_weight * prediction.log_correction.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            weight = 1 / (len(patient_ids) * patient_specimens[sample.inputs.patient_id])
            (loss * weight).backward()
            losses.append(float(loss.detach()) * weight)
        optimizer.step()
        fitted.history.append({"epoch": epoch + 1, "loss": sum(losses)})
    fitted.optimizer_state = optimizer.state_dict()
    model.eval()
    return fitted


def _expression_loss(
    error: torch.Tensor,
    observed: torch.Tensor,
    observation_groups: Optional[tuple[str, ...]] = None,
) -> torch.Tensor:
    """Equal observed-gene MSE, optionally averaged equally across assay replicates.

    Group labels follow observation-operator row order. They affect expression
    supervision only; whole-specimen protein and abundance terms stay separate.
    Without labels, preserve the original specimen-level gene weighting exactly.
    """

    def measured_gene_mean(values, mask):
        counts = mask.sum(0)
        if not (counts > 0).any():
            raise ValueError("training observation group has no observed supervised values")
        gene_losses = (values * mask).sum(0) / counts.clamp_min(1)
        return gene_losses[counts > 0].mean()

    if observation_groups is None:
        return measured_gene_mean(error, observed)
    losses = []
    for group in dict.fromkeys(observation_groups):
        rows = torch.tensor([value == group for value in observation_groups], device=error.device)
        losses.append(measured_gene_mean(error[rows], observed[rows]))
    return torch.stack(losses).mean()


def _validate_training_normalization(model: PersonalizedModel, samples: Sequence[TrainingSample]):
    """Reject effective one-gene ratios even when the configured panel has more IDs.

    Sparse individual regions are valid. Each supervised denominator gene needs
    at least one observed training region where both measured and initially
    predicted companion-gene mass are positive, so its ratio can carry a signal.
    """
    panel = model.config.normalization_gene_ids
    required = set(model.config.supervised_gene_ids) & set(panel)
    supported = set()
    with torch.no_grad():
        for sample in samples:
            prediction = model(sample.inputs)
            # Also validate complete, positive denominator coverage before any update.
            normalized_observations(
                prediction,
                sample.targets,
                sample.observation,
                model.config.supervised_gene_ids,
                normalization_gene_ids=panel,
            )
            predicted, truth, mask = aligned_observations(
                prediction,
                sample.targets,
                sample.observation,
                panel,
            )
            for gene in required:
                index = panel.index(gene)
                companions = [j for j in range(len(panel)) if j != index]
                usable = (
                    mask[:, index]
                    & (truth[:, companions].sum(1) > 0)
                    & (predicted[:, companions].sum(1) > 0)
                )
                if usable.any():
                    supported.add(gene)
    missing = required - supported
    if missing:
        raise ValueError(
            "degenerate normalization for supervised genes without positive measured and "
            f"predicted companion-gene mass in a common training region: {sorted(missing)}"
        )


def _abundance_penalty(conditional_assignment: torch.Tensor, proportions: np.ndarray):
    """Specimen composition of conditional states, independent of support mass."""
    q = conditional_assignment.mean(0)
    prior = torch.as_tensor(proportions, dtype=q.dtype, device=q.device)
    return (q * (q.clamp_min(1e-12).log() - prior.clamp_min(1e-12).log())).sum()


def read_fold_roles(path: str | Path, outer_fold: int) -> dict[str, str]:
    """Patient roles are unique; cores and sections never get independent splits."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    roles = {}
    for row in rows:
        if int(row["outer_fold"]) != outer_fold:
            continue
        patient, role = row["patient_id"], row["role"]
        if not patient.strip() or role not in {"train", "validation", "test"}:
            raise ValueError("invalid patient split row")
        if patient in roles:
            raise ValueError("each patient must have exactly one role per outer fold")
        roles[patient] = role
    if set(roles.values()) != {"train", "validation", "test"}:
        raise ValueError("fold requires train, validation, and test patients")
    return roles


def make_patient_folds(patient_ids: Sequence[str], *, n_folds=5, validation_size=2, seed=0):
    """Deterministic patient grouping; metadata balance must be reviewed before freezing."""
    patients = sorted(set(patient_ids))
    if len(patients) != len(patient_ids):
        raise ValueError("fold construction expects unique biological patient IDs")
    if n_folds < 2 or len(patients) < n_folds or validation_size < 1:
        raise ValueError("invalid fold count or validation size")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(patients)[rng.permutation(len(patients))]
    rows = []
    for outer_fold, test in enumerate(np.array_split(shuffled, n_folds)):
        remaining = shuffled[~np.isin(shuffled, test)]
        if len(remaining) <= validation_size:
            raise ValueError("not enough patients for training after validation selection")
        validation = set(remaining[:validation_size])
        for patient in patients:
            role = "test" if patient in test else "validation" if patient in validation else "train"
            rows.append({"outer_fold": outer_fold, "patient_id": patient, "role": role})
    return rows
