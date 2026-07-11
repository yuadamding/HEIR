"""Strict, dependency-light experiment configuration for HEIR."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar

import yaml


@dataclass(frozen=True)
class OptimizationConfig:
    epochs: int = 100
    learning_rate: float = 1.0e-4
    adapter_learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    warmup_fraction: float = 0.05
    gradient_clip_norm: float = 1.0
    bag_size: int = 2048
    reference_batch_size: int = 2048
    maximum_sample_cells: int = 16384
    early_stopping_patience: int = 15
    mixed_precision: bool = True

    def validate(self) -> None:
        if (
            self.epochs <= 0
            or self.bag_size <= 0
            or self.reference_batch_size <= 0
            or self.maximum_sample_cells <= 0
        ):
            raise ValueError("epochs and batch sizes must be positive")
        if self.learning_rate <= 0 or self.adapter_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("weight decay must be non-negative and clipping positive")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")


@dataclass(frozen=True)
class LossWeightConfig:
    cell_type: float = 1.0
    molecular_posterior: float = 1.0
    marker: float = 0.2
    uot: float = 1.0
    program: float = 1.0
    pseudobulk: float = 0.5
    composition: float = 0.5
    cycle: float = 0.25
    residual: float = 0.1
    domain: float = 0.1
    latent_kl: float = 0.1
    graph: float = 0.05
    calibration: float = 0.05
    hierarchy: float = 0.1
    scgpt: float = 0.5

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError("loss weight %s must be non-negative" % name)


@dataclass(frozen=True)
class RefinementConfig:
    enabled: bool = True
    maximum_rounds: int = 4
    min_probability: float = 0.90
    max_normalized_entropy: float = 0.20
    teacher_ema: float = 0.99
    # Keep the measured molecular prior fixed in the primary refinement path.
    # Lower values remain an explicit prior-update sensitivity analysis.
    prior_old_weight: float = 1.0
    minimum_segmentation_confidence: float = 0.50
    require_view_agreement: bool = True
    maximum_prior_total_variation: float = 0.10
    max_anchors_per_class: int = 10000
    stable_rounds_required: int = 1
    objective_stability_tolerance: float = 0.01
    # Anchor lifecycle requires two agreeing rounds before an anchor becomes
    # trusted, so a one-round broad phase cannot affect parent supervision.
    broad_refinement_rounds: int = 2

    def validate(self) -> None:
        if self.maximum_rounds < 0 or self.maximum_rounds > 5:
            raise ValueError("maximum_rounds must be between 0 and 5")
        for name in ("min_probability", "max_normalized_entropy"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError("%s must be in [0, 1]" % name)
        if not 0.0 <= self.teacher_ema < 1.0:
            raise ValueError("teacher_ema must be in [0, 1)")
        if not 0.0 <= self.prior_old_weight <= 1.0:
            raise ValueError("prior_old_weight must be in [0, 1]")
        if self.minimum_segmentation_confidence < 0 or self.minimum_segmentation_confidence > 1:
            raise ValueError("minimum_segmentation_confidence must be in [0, 1]")
        if self.maximum_prior_total_variation < 0:
            raise ValueError("maximum_prior_total_variation must be non-negative")
        if self.max_anchors_per_class <= 0 or self.stable_rounds_required <= 0:
            raise ValueError("anchor cap and stable_rounds_required must be positive")
        if self.objective_stability_tolerance < 0:
            raise ValueError("objective_stability_tolerance must be non-negative")
        if self.broad_refinement_rounds < 0 or self.broad_refinement_rounds > self.maximum_rounds:
            raise ValueError("broad_refinement_rounds must lie within maximum_rounds")
        if self.broad_refinement_rounds == 1:
            raise ValueError(
                "broad_refinement_rounds must be 0 for fine-only refinement or at least 2 "
                "for two-round parent-anchor trust"
            )
        if (
            self.broad_refinement_rounds > 0
            and self.maximum_rounds - self.broad_refinement_rounds < 2
        ):
            raise ValueError(
                "broad refinement must leave at least two subsequent fine rounds for "
                "two-round fine-anchor trust"
            )

    @property
    def prior_new_weight(self) -> float:
        return 1.0 - self.prior_old_weight


@dataclass(frozen=True)
class UncertaintyConfig:
    unknown_probability_threshold: float = 0.60
    ood_reference_quantile: float = 0.95
    latent_samples: int = 20
    ensemble_seeds: Tuple[int, ...] = (17, 41, 89)

    def validate(self) -> None:
        if not 0.0 < self.unknown_probability_threshold <= 1.0:
            raise ValueError("unknown_probability_threshold must be in (0, 1]")
        if not 0.0 < self.ood_reference_quantile < 1.0:
            raise ValueError("ood_reference_quantile must be in (0, 1)")
        if self.latent_samples <= 0:
            raise ValueError("latent_samples must be positive")
        if not self.ensemble_seeds or any(seed < 0 for seed in self.ensemble_seeds):
            raise ValueError("ensemble_seeds must contain non-negative values")


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration shared by train, refine, predict and evaluate."""

    name: str
    manifest: str
    output_dir: str
    mode: str = "personalized"
    seed: int = 17
    device: str = "auto"
    model: Dict[str, Any] = field(default_factory=dict)
    rna: Dict[str, Any] = field(default_factory=dict)
    graph: Dict[str, Any] = field(default_factory=dict)
    targets: Dict[str, Any] = field(default_factory=dict)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    losses: LossWeightConfig = field(default_factory=LossWeightConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    random_seeds: Tuple[int, ...] = (17, 41, 89)
    spatial_validation_only: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment name cannot be empty")
        if self.mode not in {"personalized", "atlas", "distilled", "pretraining"}:
            raise ValueError("mode must be personalized, atlas, distilled, or pretraining")
        if self.seed < 0 or any(seed < 0 for seed in self.random_seeds):
            raise ValueError("seeds must be non-negative")
        if not self.random_seeds:
            raise ValueError("random_seeds cannot be empty")
        if self.mode == "personalized" and not self.spatial_validation_only:
            raise ValueError(
                "personalized HEIR requires target spatial expression to be validation-only"
            )
        self.optimization.validate()
        self.losses.validate()
        self.refinement.validate()
        self.uncertainty.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _construct_dataclass(cls: Type[T], values: Mapping[str, Any], context: str) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError("unknown %s keys: %s" % (context, ", ".join(sorted(unknown))))
    data: MutableMapping[str, Any] = dict(values)
    if cls in {UncertaintyConfig, ExperimentConfig}:
        key = "ensemble_seeds" if cls is UncertaintyConfig else "random_seeds"
        if key in data:
            data[key] = tuple(int(value) for value in data[key])
    return cls(**data)  # type: ignore[arg-type]


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def config_from_mapping(
    values: Mapping[str, Any], base_dir: Optional[Path] = None
) -> ExperimentConfig:
    """Parse a mapping and reject misspelled configuration keys."""

    data: Dict[str, Any] = dict(values)
    nested = {
        "optimization": OptimizationConfig,
        "losses": LossWeightConfig,
        "refinement": RefinementConfig,
        "uncertainty": UncertaintyConfig,
    }
    for key, cls in nested.items():
        if key in data:
            raw = data[key]
            if not isinstance(raw, Mapping):
                raise TypeError("%s must be a mapping" % key)
            data[key] = _construct_dataclass(cls, raw, key)
    config = _construct_dataclass(ExperimentConfig, data, "experiment")
    if base_dir is not None:
        config = ExperimentConfig(
            **{
                **config.to_dict(),
                "manifest": _resolve_path(config.manifest, base_dir),
                "output_dir": _resolve_path(config.output_dir, base_dir),
                "optimization": config.optimization,
                "losses": config.losses,
                "refinement": config.refinement,
                "uncertainty": config.uncertainty,
            }
        )
    config.validate()
    return config


def load_config(path: str) -> ExperimentConfig:
    """Load a YAML experiment file with paths relative to that file."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, Mapping):
        raise TypeError("configuration root must be a mapping")
    return config_from_mapping(values, source.parent)
