"""Hierarchical, sample-prototype-informed HEIR model."""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.autograd import Function

from .graph import GraphContextConfig, GraphContextEncoder
from .rna import RNAVAE, RNADecoder, RNAVAEConfig

_HEIR_CHECKPOINT_SCHEMA = "heir.model.v3"
_LEGACY_TIED_CHECKPOINT_SCHEMA = "heir.model.v2"


def _positive_dims(values: Sequence[int], name: str) -> Tuple[int, ...]:
    dims = tuple(int(value) for value in values)
    if not dims or any(value <= 0 for value in dims):
        raise ValueError("%s must contain positive widths" % name)
    return dims


def _hidden_stack(input_dim: int, widths: Sequence[int], dropout: float) -> nn.Sequential:
    layers = []
    previous = input_dim
    for width in widths:
        layers.extend(
            [
                nn.Linear(previous, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        previous = width
    return nn.Sequential(*layers)


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx: Any, value: Tensor, scale: float) -> Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tuple[Tensor, None]:
        return -ctx.scale * gradient, None


@dataclass(frozen=True)
class HEIRConfig:
    """Checkpoint-safe architecture and routing configuration."""

    morphology_dim: int
    num_cell_types: int
    expression_dim: int
    latent_dim: int = 32
    graph_hidden_dim: int = 128
    graph_output_dim: int = 128
    graph_layers: int = 2
    trunk_hidden_dims: Tuple[int, ...] = (256, 128)
    decoder_hidden_dims: Tuple[int, ...] = (128, 256)
    dropout: float = 0.1
    normalize_messages: bool = True
    graph_residual: bool = True
    fine_to_parent: Optional[Tuple[int, ...]] = None
    num_parent_types: int = 0
    prototype_temperature: float = 0.5
    prototype_match_level: str = "fine"
    hard_type_routing: bool = True
    unknown_logit_bias: float = 0.0
    abstain_threshold: float = 0.6
    logvar_min: float = -12.0
    logvar_max: float = 8.0
    nonnegative_expression: bool = False
    num_domains: int = 0
    domain_gradient_scale: float = 1.0
    prototype_type_cost_weight: float = 1.0
    prototype_abundance_logit_weight: float = 0.0
    prototype_variance_floor: float = 1.0e-4
    covariance_aware_uot: bool = True
    legacy_independent_prototype_query: bool = False
    legacy_unrestricted_residual: bool = False
    residual_rank: int = 0
    residual_max_norm: float = 0.5
    residual_type_strategy: str = "detached_max"
    residual_type_concentration_threshold: float = 0.6
    scgpt_embedding_dim: int = 0

    def __post_init__(self) -> None:
        dimensions = (
            self.morphology_dim,
            self.expression_dim,
            self.latent_dim,
            self.graph_hidden_dim,
            self.graph_output_dim,
            self.graph_layers,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("feature dimensions and graph_layers must be positive")
        if self.num_cell_types < 2:
            raise ValueError("num_cell_types must be at least two")
        if self.num_domains < 0 or self.num_domains == 1:
            raise ValueError("num_domains must be zero or at least two")
        if self.domain_gradient_scale < 0:
            raise ValueError("domain_gradient_scale must be non-negative")
        if self.prototype_type_cost_weight < 0:
            raise ValueError("prototype_type_cost_weight must be non-negative")
        if self.prototype_abundance_logit_weight < 0:
            raise ValueError("prototype_abundance_logit_weight must be non-negative")
        if self.prototype_variance_floor <= 0:
            raise ValueError("prototype_variance_floor must be positive")
        if self.residual_rank < 0:
            raise ValueError("residual_rank must be non-negative")
        resolved_residual_rank = self.residual_rank or min(4, self.latent_dim)
        if resolved_residual_rank > self.latent_dim:
            raise ValueError("residual_rank cannot exceed latent_dim")
        if not math.isfinite(self.residual_max_norm) or self.residual_max_norm <= 0:
            raise ValueError("residual_max_norm must be finite and positive")
        object.__setattr__(self, "residual_rank", resolved_residual_rank)
        if self.residual_type_strategy not in {"detached_max", "legacy_weighted_basis"}:
            raise ValueError("residual_type_strategy must be detached_max or legacy_weighted_basis")
        if not 0.0 <= self.residual_type_concentration_threshold < 1.0:
            raise ValueError("residual_type_concentration_threshold must be in [0, 1)")
        if self.legacy_independent_prototype_query:
            # This flag only exists to reproduce the original, unrestricted
            # v1 model. Keep it from creating a hybrid architecture.
            object.__setattr__(self, "legacy_unrestricted_residual", True)
        if self.scgpt_embedding_dim < 0:
            raise ValueError("scgpt_embedding_dim must be non-negative")
        object.__setattr__(
            self,
            "trunk_hidden_dims",
            _positive_dims(self.trunk_hidden_dims, "trunk_hidden_dims"),
        )
        object.__setattr__(
            self,
            "decoder_hidden_dims",
            _positive_dims(self.decoder_hidden_dims, "decoder_hidden_dims"),
        )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")
        if self.prototype_match_level not in {"fine", "parent"}:
            raise ValueError("prototype_match_level must be fine or parent")
        if not 0.0 < self.abstain_threshold <= 1.0:
            raise ValueError("abstain_threshold must be in (0, 1]")
        if self.logvar_min >= self.logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")

        if self.fine_to_parent is None:
            if self.num_parent_types != 0:
                raise ValueError("num_parent_types requires fine_to_parent")
            if self.prototype_match_level == "parent":
                raise ValueError("parent prototype matching requires fine_to_parent")
        else:
            mapping = tuple(int(value) for value in self.fine_to_parent)
            if len(mapping) != self.num_cell_types or any(value < 0 for value in mapping):
                raise ValueError("fine_to_parent must contain one nonnegative parent per fine type")
            inferred = max(mapping) + 1
            if self.num_parent_types not in (0, inferred):
                raise ValueError("num_parent_types does not agree with fine_to_parent")
            object.__setattr__(self, "fine_to_parent", mapping)
            object.__setattr__(self, "num_parent_types", inferred)

    def to_dict(self) -> Dict[str, Any]:
        """Return standard-type checkpoint metadata."""

        result = asdict(self)
        result["trunk_hidden_dims"] = list(self.trunk_hidden_dims)
        result["decoder_hidden_dims"] = list(self.decoder_hidden_dims)
        if self.fine_to_parent is not None:
            result["fine_to_parent"] = list(self.fine_to_parent)
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "HEIRConfig":
        """Reconstruct a config from metadata."""

        data = dict(values)
        for name in ("trunk_hidden_dims", "decoder_hidden_dims", "fine_to_parent"):
            if data.get(name) is not None:
                data[name] = tuple(data[name])
        return cls(**data)


@dataclass
class HEIROutput:
    """All per-cell predictions and prototype-routing diagnostics."""

    type_logits: Tensor
    type_probabilities: Tensor
    fine_type_logits: Tensor
    fine_type_probabilities: Tensor
    parent_type_logits: Optional[Tensor]
    parent_type_probabilities: Optional[Tensor]
    hierarchy_parent_probabilities: Optional[Tensor]
    prototype_query: Tensor
    prototype_cost: Tensor
    prototype_logits: Tensor
    prototype_probabilities: Tensor
    conditional_prototype_probabilities: Tensor
    prototype_types: Tensor
    prototype_weights: Tensor
    prototype_variances: Tensor
    prototype_mask: Tensor
    prototype_latent: Tensor
    residual_coefficients: Optional[Tensor]
    residual_coefficient_logvar: Optional[Tensor]
    residual_basis: Optional[Tensor]
    residual_gate: Optional[Tensor]
    residual_mu: Tensor
    residual_logvar: Tensor
    residual: Tensor
    latent_mu: Tensor
    latent: Tensor
    expression: Tensor
    type_entropy: Tensor
    prototype_entropy: Tensor
    residual_uncertainty: Tensor
    unknown_probability: Tensor
    abstain_score: Tensor
    abstain: Tensor
    cell_embedding: Tensor
    scgpt_embedding: Optional[Tensor]
    domain_logits: Optional[Tensor]

    @property
    def decoded_expression(self) -> Tensor:
        """Alias for decoder output."""

        return self.expression

    @property
    def unknown_signal(self) -> Tensor:
        """Alias for explicit unassigned prototype probability."""

        return self.unknown_probability

    def as_dict(self) -> Dict[str, Optional[Tensor]]:
        """Return a generic mapping for training and export code."""

        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class HEIRModel(nn.Module):
    """Infer a type-compatible molecular prototype plus morphology residual."""

    def __init__(self, config: HEIRConfig) -> None:
        super().__init__()
        self.config = config
        graph_config = GraphContextConfig(
            input_dim=config.morphology_dim,
            hidden_dim=config.graph_hidden_dim,
            output_dim=config.graph_output_dim,
            num_layers=config.graph_layers,
            dropout=config.dropout,
            normalize_messages=config.normalize_messages,
            residual=config.graph_residual,
        )
        self.graph_encoder = GraphContextEncoder(graph_config)
        self.trunk = _hidden_stack(
            config.morphology_dim + config.graph_output_dim,
            config.trunk_hidden_dims,
            config.dropout,
        )
        hidden_dim = config.trunk_hidden_dims[-1]
        self.fine_type_head = nn.Linear(hidden_dim, config.num_cell_types)
        self.parent_type_head: Optional[nn.Linear]
        if config.fine_to_parent is None:
            self.parent_type_head = None
            mapping = torch.empty(0, dtype=torch.long)
        else:
            self.parent_type_head = nn.Linear(hidden_dim, config.num_parent_types)
            mapping = torch.tensor(config.fine_to_parent, dtype=torch.long)
        self.register_buffer("fine_to_parent_index", mapping, persistent=True)
        self.register_buffer(
            "residual_type_max_norms",
            torch.full((config.num_cell_types,), config.residual_max_norm),
            persistent=False,
        )

        self.prototype_query_head = nn.Linear(hidden_dim, config.latent_dim)
        self.residual_mu_head: Optional[nn.Linear]
        self.residual_coefficient_head: Optional[nn.Linear]
        self.residual_gate_head: Optional[nn.Linear]
        self.residual_type_basis: Optional[nn.Parameter]
        if config.legacy_unrestricted_residual:
            self.residual_mu_head = nn.Linear(hidden_dim, config.latent_dim)
            self.residual_logvar_head = nn.Linear(hidden_dim, config.latent_dim)
            self.residual_coefficient_head = None
            self.residual_gate_head = None
            self.register_parameter("residual_type_basis", None)
        else:
            self.residual_mu_head = None
            self.residual_coefficient_head = nn.Linear(hidden_dim, config.residual_rank)
            self.residual_logvar_head = nn.Linear(hidden_dim, config.residual_rank)
            self.residual_gate_head = nn.Linear(hidden_dim, 1)
            basis = torch.empty(
                config.num_cell_types,
                config.latent_dim,
                config.residual_rank,
            )
            for type_basis in basis:
                nn.init.orthogonal_(type_basis)
            self.residual_type_basis = nn.Parameter(basis)
            # The low-rank coefficient mean is the actual residual head. Its
            # exact zero initialization makes a fresh model inherit the routed
            # RNA prototype before any image-supported correction is learned.
            nn.init.zeros_(self.residual_coefficient_head.weight)
            nn.init.zeros_(self.residual_coefficient_head.bias)
            # Start stochastic refinement close to the routed RNA prototype as
            # well.  A random variance head made training samples depart from
            # the prototype even though the deterministic coefficient mean was
            # exactly zero.
            nn.init.zeros_(self.residual_logvar_head.weight)
            nn.init.constant_(self.residual_logvar_head.bias, -6.0)
            nn.init.zeros_(self.residual_gate_head.weight)
            nn.init.constant_(self.residual_gate_head.bias, -2.0)
        self.unknown_head = nn.Linear(hidden_dim, 1)
        self.scgpt_head = (
            nn.Linear(hidden_dim, config.scgpt_embedding_dim)
            if config.scgpt_embedding_dim > 0
            else None
        )
        self.domain_head = (
            nn.Linear(hidden_dim, config.num_domains) if config.num_domains >= 2 else None
        )

        decoder_config = RNAVAEConfig(
            input_dim=config.expression_dim,
            latent_dim=config.latent_dim,
            hidden_dims=tuple(reversed(config.decoder_hidden_dims)),
            decoder_hidden_dims=config.decoder_hidden_dims,
            dropout=config.dropout,
            logvar_min=config.logvar_min,
            logvar_max=config.logvar_max,
            nonnegative_output=config.nonnegative_expression,
        )
        self.expression_decoder = RNADecoder(decoder_config)

    def _hierarchical_types(
        self,
        embedding: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        fine_logits = self.fine_type_head(embedding)
        fine_probabilities = torch.softmax(fine_logits, dim=-1)
        if self.parent_type_head is None:
            return fine_logits, fine_probabilities, None, None, None
        parent_logits = self.parent_type_head(embedding)
        parent_probabilities = torch.softmax(parent_logits, dim=-1)
        parent_log_prior = parent_probabilities.clamp_min(1e-12).log()
        hierarchical_logits = fine_logits + parent_log_prior.index_select(
            1,
            self.fine_to_parent_index,
        )
        probabilities = torch.softmax(hierarchical_logits, dim=-1)
        aggregate = probabilities.new_zeros((probabilities.shape[0], self.config.num_parent_types))
        aggregate = aggregate.index_add(1, self.fine_to_parent_index, probabilities)
        return hierarchical_logits, probabilities, fine_logits, parent_logits, aggregate

    def _prepare_prototypes(
        self,
        num_cells: int,
        reference: Tensor,
        prototype_means: Optional[Tensor],
        prototype_variances: Optional[Tensor],
        prototype_types: Optional[Tensor],
        prototype_weights: Optional[Tensor],
        prototype_mask: Optional[Tensor],
        sample_index: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if prototype_means is None:
            empty_means = reference.new_empty((num_cells, 0, self.config.latent_dim))
            empty_variances = reference.new_empty((num_cells, 0, self.config.latent_dim))
            empty_types = torch.empty((num_cells, 0), dtype=torch.long, device=reference.device)
            empty_weights = reference.new_empty((num_cells, 0))
            empty_mask = torch.empty((num_cells, 0), dtype=torch.bool, device=reference.device)
            return empty_means, empty_variances, empty_types, empty_weights, empty_mask
        if prototype_means.device != reference.device or not torch.is_floating_point(
            prototype_means
        ):
            raise ValueError("prototype_means must be floating point on the morphology device")
        if prototype_means.shape[-1] != self.config.latent_dim:
            raise ValueError("prototype_means has the wrong latent dimension")
        if not torch.isfinite(prototype_means).all():
            raise ValueError("prototype_means must be finite")

        expected_meta_shape: Tuple[int, ...]
        if prototype_means.ndim == 2:
            samples = 1
            prototypes = prototype_means.shape[0]
            means = prototype_means.unsqueeze(0).expand(num_cells, -1, -1)
            expected_meta_shape = (prototypes,)
        elif prototype_means.ndim == 3:
            samples, prototypes = prototype_means.shape[:2]
            if sample_index is None:
                if samples != 1:
                    raise ValueError("sample_index is required for multiple prototype banks")
                cell_samples = torch.zeros(num_cells, dtype=torch.long, device=reference.device)
            else:
                if sample_index.shape != (num_cells,) or sample_index.dtype != torch.long:
                    raise ValueError("sample_index must be long with one value per cell")
                if sample_index.device != reference.device:
                    raise ValueError("sample_index and morphology must share a device")
                if sample_index.numel() and (
                    bool((sample_index < 0).any()) or int(sample_index.max()) >= samples
                ):
                    raise ValueError("sample_index references an unavailable prototype bank")
                cell_samples = sample_index
            means = prototype_means.index_select(0, cell_samples)
            expected_meta_shape = (samples, prototypes)
        else:
            raise ValueError("prototype_means must have shape (P, L) or (S, P, L)")

        if prototype_variances is None:
            variances = torch.ones_like(means)
        else:
            if prototype_variances.device != reference.device or not torch.is_floating_point(
                prototype_variances
            ):
                raise ValueError(
                    "prototype_variances must be floating point on the morphology device"
                )
            if prototype_variances.shape != prototype_means.shape:
                raise ValueError("prototype_variances must align to prototype_means")
            if prototype_means.ndim == 2:
                variances = prototype_variances.unsqueeze(0).expand(num_cells, -1, -1)
            else:
                variances = prototype_variances.index_select(0, cell_samples)
            if not torch.isfinite(variances).all() or bool((variances <= 0).any()):
                raise ValueError("prototype_variances must be finite and positive")
            variances = variances.to(reference.dtype)

        def metadata(value: Optional[Tensor], name: str, default: Tensor) -> Tensor:
            if value is None:
                value = default
            if value.device != reference.device:
                raise ValueError("%s and morphology must share a device" % name)
            if prototype_means.ndim == 2:
                if value.shape != expected_meta_shape:
                    raise ValueError("%s has the wrong shape" % name)
                return value.unsqueeze(0).expand(num_cells, -1)
            if value.shape != expected_meta_shape:
                raise ValueError("%s has the wrong shape" % name)
            return value.index_select(0, cell_samples)

        default_types = torch.full(
            expected_meta_shape,
            -1,
            dtype=torch.long,
            device=reference.device,
        )
        default_weights = reference.new_ones(expected_meta_shape)
        default_mask = torch.ones(expected_meta_shape, dtype=torch.bool, device=reference.device)
        types = metadata(prototype_types, "prototype_types", default_types)
        weights = metadata(prototype_weights, "prototype_weights", default_weights)
        mask = metadata(prototype_mask, "prototype_mask", default_mask)
        if types.dtype != torch.long:
            raise TypeError("prototype_types must have dtype torch.long")
        if types.numel() and (
            bool((types < -1).any()) or bool((types >= self.config.num_cell_types).any())
        ):
            raise ValueError("prototype_types must be -1 or a valid fine type")
        if not torch.is_floating_point(weights):
            raise TypeError("prototype_weights must be floating point")
        if not torch.isfinite(weights).all() or bool((weights < 0).any()):
            raise ValueError("prototype_weights must be finite and nonnegative")
        if mask.dtype != torch.bool:
            raise TypeError("prototype_mask must have dtype torch.bool")
        return means, variances, types, weights.to(reference.dtype), mask & (weights > 0)

    def _prototype_compatibility(
        self,
        type_probabilities: Tensor,
        hierarchy_parent_probabilities: Optional[Tensor],
        prototype_types: Tensor,
        base_mask: Tensor,
        cell_type_constraints: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        typed = prototype_types >= 0
        gathered_types = prototype_types.clamp_min(0)
        if self.config.prototype_match_level == "parent":
            assert hierarchy_parent_probabilities is not None
            prototype_level = self.fine_to_parent_index.index_select(
                0,
                gathered_types.reshape(-1),
            ).reshape_as(gathered_types)
            probabilities = hierarchy_parent_probabilities.gather(1, prototype_level)
        else:
            prototype_level = gathered_types
            probabilities = type_probabilities.gather(1, prototype_level)
        compatibility = torch.where(typed, probabilities, torch.ones_like(probabilities))
        valid = base_mask

        if cell_type_constraints is not None:
            if cell_type_constraints.device != type_probabilities.device:
                raise ValueError("cell_type_constraints and morphology must share a device")
            if cell_type_constraints.ndim == 1:
                if cell_type_constraints.shape[0] != type_probabilities.shape[0]:
                    raise ValueError("cell_type_constraints must contain one value per cell")
                if cell_type_constraints.dtype != torch.long:
                    raise TypeError("hard cell_type_constraints must be long")
                constrained = cell_type_constraints >= 0
                if bool((cell_type_constraints >= self.config.num_cell_types).any()):
                    raise ValueError("cell_type_constraints contains an invalid type")
                required = cell_type_constraints.clamp_min(0)
                if self.config.prototype_match_level == "parent":
                    required = self.fine_to_parent_index.index_select(0, required)
                allowed = prototype_level == required.unsqueeze(1)
                allowed = allowed | ~typed | ~constrained.unsqueeze(1)
                valid = valid & allowed
                compatibility = torch.where(allowed, torch.ones_like(compatibility), compatibility)
            elif cell_type_constraints.ndim == 2:
                if cell_type_constraints.shape != type_probabilities.shape:
                    raise ValueError("soft constraints must have shape (cells, fine types)")
                allowed_fine = cell_type_constraints.to(dtype=type_probabilities.dtype)
                if bool((allowed_fine < 0).any()) or not torch.isfinite(allowed_fine).all():
                    raise ValueError("soft constraints must be finite and nonnegative")
                allowed = allowed_fine.gather(1, gathered_types)
                allowed = torch.where(typed, allowed, torch.ones_like(allowed))
                compatibility = compatibility * allowed
                valid = valid & (allowed > 0)
            else:
                raise ValueError("cell_type_constraints must have one or two dimensions")
        elif self.config.hard_type_routing:
            routed = type_probabilities.argmax(dim=-1)
            if self.config.prototype_match_level == "parent":
                routed = self.fine_to_parent_index.index_select(0, routed)
            allowed = (prototype_level == routed.unsqueeze(1)) | ~typed
            valid = valid & allowed
            compatibility = torch.where(allowed, torch.ones_like(compatibility), compatibility)
        return compatibility, valid

    @staticmethod
    def _masked_prototype_softmax(
        known_logits: Tensor,
        known_mask: Tensor,
        unknown_logits: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        masked = known_logits.masked_fill(~known_mask, -torch.inf)
        if masked.shape[1]:
            maximum = torch.maximum(masked.max(dim=1).values, unknown_logits)
        else:
            maximum = unknown_logits
        known_exp = torch.exp(masked - maximum.unsqueeze(1))
        known_exp = torch.where(known_mask, known_exp, torch.zeros_like(known_exp))
        unknown_exp = torch.exp(unknown_logits - maximum)
        denominator = known_exp.sum(dim=1) + unknown_exp
        unconditional = known_exp / denominator.unsqueeze(1)
        unknown = unknown_exp / denominator
        if not masked.shape[1]:
            return unconditional, unknown, known_exp

        # Compute the conditional known-state simplex independently of the
        # unknown logit. Dividing unconditional probabilities by ``1-u`` is
        # mathematically equivalent but becomes unstable when u approaches one.
        has_known = known_mask.any(dim=1)
        known_maximum = torch.where(
            has_known,
            masked.max(dim=1).values,
            torch.zeros_like(unknown_logits),
        )
        conditional_exp = torch.exp(masked - known_maximum.unsqueeze(1))
        conditional_exp = torch.where(
            known_mask,
            conditional_exp,
            torch.zeros_like(conditional_exp),
        )
        conditional = conditional_exp / conditional_exp.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(conditional_exp.dtype).eps
        )
        return unconditional, unknown, conditional

    @staticmethod
    def _bounded_low_rank_residual(
        basis: Tensor,
        coefficients: Tensor,
        gate: Tensor,
    ) -> Tensor:
        """Map coefficient vectors into a smooth, gated latent unit ball."""

        if basis.ndim != 3 or gate.ndim != 1 or basis.shape[0] != gate.shape[0]:
            raise ValueError("residual basis/gate must contain one entry per cell")
        # Frozen RNA geometry remains float32 under CUDA autocast, whereas the
        # learned coefficient/gate heads emit float16. Sampling occurs outside
        # the forward autocast context, so einsum will not reconcile those
        # dtypes for us. Promote the complete residual calculation explicitly;
        # this also keeps the bounded norm calculation in the safer precision.
        work_dtype = torch.promote_types(basis.dtype, coefficients.dtype)
        work_dtype = torch.promote_types(work_dtype, gate.dtype)
        basis = basis.to(dtype=work_dtype)
        coefficients = coefficients.to(dtype=work_dtype)
        gate = gate.to(dtype=work_dtype)
        if coefficients.ndim == 2:
            if coefficients.shape != (basis.shape[0], basis.shape[2]):
                raise ValueError("residual coefficients do not align to the basis")
            projected = torch.einsum("nlr,nr->nl", basis, coefficients)
            scale = gate.unsqueeze(-1)
        elif coefficients.ndim == 3:
            if coefficients.shape[1:] != (basis.shape[0], basis.shape[2]):
                raise ValueError("sampled residual coefficients do not align to the basis")
            projected = torch.einsum("nlr,dnr->dnl", basis, coefficients)
            scale = gate.reshape(1, -1, 1)
        else:
            raise ValueError("residual coefficients must have two or three dimensions")
        # x / sqrt(1 + ||x||^2) is smooth at zero and has norm strictly below
        # one, so multiplying by the bounded gate gives a hard per-cell bound.
        denominator = torch.sqrt(1.0 + projected.square().sum(dim=-1, keepdim=True))
        return scale * projected / denominator

    def sample_residuals(self, output: HEIROutput, draws: int) -> Tensor:
        """Sample residuals under the checkpoint's explicit residual contract."""

        if draws <= 0:
            raise ValueError("draws must be positive")
        if output.residual_coefficients is None:
            noise = torch.randn(
                (draws, *output.residual_mu.shape),
                device=output.residual_mu.device,
                dtype=output.residual_mu.dtype,
            )
            return output.residual_mu.unsqueeze(0) + noise * torch.exp(
                0.5 * output.residual_logvar
            ).unsqueeze(0)
        if (
            output.residual_coefficient_logvar is None
            or output.residual_basis is None
            or output.residual_gate is None
        ):
            raise ValueError("restricted residual output is incomplete")
        noise = torch.randn(
            (draws, *output.residual_coefficients.shape),
            device=output.residual_coefficients.device,
            dtype=output.residual_coefficients.dtype,
        )
        coefficients = output.residual_coefficients.unsqueeze(0) + noise * torch.exp(
            0.5 * output.residual_coefficient_logvar
        ).unsqueeze(0)
        return self._bounded_low_rank_residual(
            output.residual_basis,
            coefficients,
            output.residual_gate,
        )

    def forward(
        self,
        morphology: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        *,
        prototype_means: Optional[Tensor] = None,
        prototype_variances: Optional[Tensor] = None,
        prototype_types: Optional[Tensor] = None,
        prototype_weights: Optional[Tensor] = None,
        prototype_mask: Optional[Tensor] = None,
        sample_index: Optional[Tensor] = None,
        cell_type_constraints: Optional[Tensor] = None,
        sample_latent: Optional[bool] = None,
    ) -> HEIROutput:
        """Predict cell types and molecular states for a bag of nuclei."""

        if morphology.ndim != 2 or morphology.shape[1] != self.config.morphology_dim:
            raise ValueError("morphology has the wrong shape")
        if not torch.is_floating_point(morphology):
            raise TypeError("morphology must be floating point")
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=morphology.device)
        context = self.graph_encoder(morphology, edge_index, edge_weight)
        embedding = self.trunk(torch.cat((morphology, context), dim=-1))
        type_logits, type_probabilities, raw_fine_logits, parent_logits, hierarchy_parent = (
            self._hierarchical_types(embedding)
        )
        fine_logits = type_logits if raw_fine_logits is None else raw_fine_logits
        fine_probabilities = torch.softmax(fine_logits, dim=-1)
        parent_probabilities = (
            None if parent_logits is None else torch.softmax(parent_logits, dim=-1)
        )

        means, variances, types, weights, mask = self._prepare_prototypes(
            morphology.shape[0],
            morphology,
            prototype_means,
            prototype_variances,
            prototype_types,
            prototype_weights,
            prototype_mask,
            sample_index,
        )
        routing_query = self.prototype_query_head(embedding)
        residual_coefficients: Optional[Tensor]
        residual_coefficient_logvar: Optional[Tensor]
        residual_basis: Optional[Tensor]
        residual_gate: Optional[Tensor]
        if self.config.legacy_unrestricted_residual:
            assert self.residual_mu_head is not None
            image_latent_mu = self.residual_mu_head(embedding)
            residual_logvar = self.residual_logvar_head(embedding).clamp(
                min=self.config.logvar_min,
                max=self.config.logvar_max,
            )
            if not self.config.legacy_independent_prototype_query:
                routing_query = image_latent_mu
            residual_coefficients = None
            residual_coefficient_logvar = None
            residual_basis = None
            residual_gate = None
            residual_mu = image_latent_mu
            residual_variance = residual_logvar.exp()
        else:
            assert self.residual_coefficient_head is not None
            assert self.residual_gate_head is not None
            assert self.residual_type_basis is not None
            residual_coefficients = self.residual_coefficient_head(embedding)
            residual_coefficient_logvar = self.residual_logvar_head(embedding).clamp(
                min=self.config.logvar_min,
                max=self.config.logvar_max,
            )
            if self.config.residual_type_strategy == "legacy_weighted_basis":
                residual_basis = torch.einsum(
                    "nc,clr->nlr",
                    type_probabilities,
                    self.residual_type_basis,
                )
                cell_residual_max_norm = type_probabilities @ self.residual_type_max_norms.to(
                    dtype=type_probabilities.dtype
                )
                concentration_gate = torch.ones_like(cell_residual_max_norm)
            else:
                # Coefficients retain their RNA-program interpretation by using
                # one frozen, orthonormal type basis per cell.  Type selection
                # and the concentration decision are detached: the residual
                # loss cannot sharpen the classifier merely to unlock a larger
                # molecular correction.
                type_concentration, selected_type = type_probabilities.detach().max(dim=1)
                residual_basis = self.residual_type_basis.index_select(0, selected_type)
                cell_residual_max_norm = self.residual_type_max_norms.to(
                    dtype=type_probabilities.dtype
                ).index_select(0, selected_type)
                concentration_gate = (
                    type_concentration >= self.config.residual_type_concentration_threshold
                ).to(dtype=type_probabilities.dtype)
            residual_gate = (
                cell_residual_max_norm
                * torch.sigmoid(self.residual_gate_head(embedding).squeeze(-1))
                * concentration_gate
            )
            residual_mu = self._bounded_low_rank_residual(
                residual_basis,
                residual_coefficients,
                residual_gate,
            )
            # A diagonal moment approximation is retained for the trainer's
            # Gaussian transport cost. Normalize its trace by the same unit-ball
            # envelope, so uncertainty cannot evade the residual magnitude bound.
            projected_variance = torch.einsum(
                "nlr,nr->nl",
                residual_basis.square(),
                residual_coefficient_logvar.exp(),
            )
            residual_variance = residual_gate.square().unsqueeze(-1) * (
                projected_variance / (1.0 + projected_variance.sum(dim=-1, keepdim=True))
            )
            residual_logvar = (
                residual_variance.clamp_min(torch.finfo(residual_variance.dtype).tiny)
                .log()
                .clamp(min=self.config.logvar_min, max=self.config.logvar_max)
            )
            residual_variance = residual_logvar.exp()

        routing_cost = (routing_query.unsqueeze(1) - means).square().mean(dim=-1)
        if self.config.covariance_aware_uot and means.shape[1]:
            total_variance = residual_variance.unsqueeze(1) + variances.clamp_min(
                self.config.prototype_variance_floor
            )
            routing_molecular_cost = 0.5 * (
                (routing_query.unsqueeze(1) - means).square() / total_variance
                + total_variance.log()
            ).mean(dim=-1)
        else:
            routing_molecular_cost = routing_cost
        compatibility, compatible_mask = self._prototype_compatibility(
            type_probabilities,
            hierarchy_parent,
            types,
            mask,
            cell_type_constraints,
        )
        known_logits = -routing_molecular_cost / self.config.prototype_temperature
        known_logits = known_logits + (
            self.config.prototype_abundance_logit_weight * weights.clamp_min(1e-12).log()
        )
        known_logits = known_logits + compatibility.clamp_min(1e-12).log()
        unknown_logits = self.unknown_head(embedding).squeeze(-1)
        unknown_logits = unknown_logits + self.config.unknown_logit_bias
        (
            prototype_probabilities,
            unknown_probability,
            conditional_prototype_probabilities,
        ) = self._masked_prototype_softmax(
            known_logits,
            compatible_mask,
            unknown_logits,
        )
        mixture_probabilities = (
            prototype_probabilities
            if self.config.legacy_independent_prototype_query
            else conditional_prototype_probabilities
        )
        prototype_latent = (mixture_probabilities.unsqueeze(-1) * means).sum(dim=1)
        if self.config.legacy_unrestricted_residual:
            residual_mu = (
                image_latent_mu
                if self.config.legacy_independent_prototype_query
                else image_latent_mu - prototype_latent
            )
        should_sample = self.training if sample_latent is None else sample_latent
        if should_sample:
            if residual_coefficients is None:
                residual = residual_mu + torch.randn_like(residual_mu) * torch.exp(
                    0.5 * residual_logvar
                )
            else:
                assert residual_coefficient_logvar is not None
                assert residual_basis is not None
                assert residual_gate is not None
                sampled_coefficients = residual_coefficients + torch.randn_like(
                    residual_coefficients
                ) * torch.exp(0.5 * residual_coefficient_logvar)
                residual = self._bounded_low_rank_residual(
                    residual_basis,
                    sampled_coefficients,
                    residual_gate,
                )
        else:
            residual = residual_mu
        latent_mu = prototype_latent + residual_mu
        latent = prototype_latent + residual
        expression = self.expression_decoder(latent)

        # UOT now acts on the same posterior mean decoded into expression. The
        # diagonal Gaussian cost incorporates both image and RNA-state uncertainty.
        if not self.config.legacy_unrestricted_residual:
            transport_query = latent_mu
            if self.config.covariance_aware_uot and means.shape[1]:
                transport_variance = residual_variance.unsqueeze(1) + variances.clamp_min(
                    self.config.prototype_variance_floor
                )
                transport_base_cost = 0.5 * (
                    (latent_mu.unsqueeze(1) - means).square() / transport_variance
                    + transport_variance.log()
                ).mean(dim=-1)
            else:
                transport_base_cost = (latent_mu.unsqueeze(1) - means).square().mean(dim=-1)
        elif self.config.covariance_aware_uot and means.shape[1]:
            transport_base_cost = routing_molecular_cost
            transport_query = latent_mu
        else:
            transport_base_cost = routing_cost
            transport_query = routing_query
        # HEIRTrainer applies the paired Bernoulli real/dustbin gate at the UOT
        # boundary so both mutually exclusive outcomes stay coherent.
        transport_cost = transport_base_cost - (
            self.config.prototype_type_cost_weight * compatibility.clamp_min(1e-12).log()
        )

        eps = torch.finfo(type_probabilities.dtype).tiny
        type_entropy = -(type_probabilities * type_probabilities.clamp_min(eps).log()).sum(
            dim=-1
        ) / math.log(self.config.num_cell_types)
        all_prototypes = torch.cat(
            (prototype_probabilities, unknown_probability.unsqueeze(1)),
            dim=1,
        )
        if all_prototypes.shape[1] > 1:
            prototype_entropy = -(all_prototypes * all_prototypes.clamp_min(eps).log()).sum(
                dim=-1
            ) / math.log(all_prototypes.shape[1])
        else:
            prototype_entropy = unknown_probability.new_zeros(unknown_probability.shape)
        mean_residual_variance = residual_variance.mean(dim=-1)
        residual_uncertainty = -torch.expm1(-mean_residual_variance)
        certainty = (
            (1.0 - unknown_probability)
            * (1.0 - type_entropy)
            * (1.0 - prototype_entropy)
            * (1.0 - residual_uncertainty)
        )
        abstain_score = (1.0 - certainty).clamp(0.0, 1.0)
        domain_logits = None
        scgpt_embedding = None if self.scgpt_head is None else self.scgpt_head(embedding)
        if self.domain_head is not None:
            reversed_embedding = _GradientReverse.apply(
                embedding,
                self.config.domain_gradient_scale,
            )
            domain_logits = self.domain_head(reversed_embedding)

        return HEIROutput(
            type_logits=type_logits,
            type_probabilities=type_probabilities,
            fine_type_logits=fine_logits,
            fine_type_probabilities=fine_probabilities,
            parent_type_logits=parent_logits,
            parent_type_probabilities=parent_probabilities,
            hierarchy_parent_probabilities=hierarchy_parent,
            prototype_query=transport_query,
            prototype_cost=transport_cost,
            prototype_logits=known_logits,
            prototype_probabilities=prototype_probabilities,
            conditional_prototype_probabilities=conditional_prototype_probabilities,
            prototype_types=types,
            prototype_weights=weights,
            prototype_variances=variances,
            prototype_mask=compatible_mask,
            prototype_latent=prototype_latent,
            residual_coefficients=residual_coefficients,
            residual_coefficient_logvar=residual_coefficient_logvar,
            residual_basis=residual_basis,
            residual_gate=residual_gate,
            residual_mu=residual_mu,
            residual_logvar=residual_logvar,
            residual=residual,
            latent_mu=latent_mu,
            latent=latent,
            expression=expression,
            type_entropy=type_entropy,
            prototype_entropy=prototype_entropy,
            residual_uncertainty=residual_uncertainty,
            unknown_probability=unknown_probability,
            abstain_score=abstain_score,
            abstain=abstain_score >= self.config.abstain_threshold,
            cell_embedding=embedding,
            scgpt_embedding=scgpt_embedding,
            domain_logits=domain_logits,
        )

    def load_rna_decoder(self, rna_vae: RNAVAE, freeze: bool = True) -> None:
        """Transfer a topology-compatible RNA decoder and optionally freeze it."""

        if rna_vae.config.latent_dim != self.config.latent_dim:
            raise ValueError("RNA VAE latent_dim does not match HEIR")
        if rna_vae.config.input_dim != self.config.expression_dim:
            raise ValueError("RNA VAE input_dim does not match HEIR")
        try:
            self.expression_decoder.load_state_dict(rna_vae.decoder.state_dict(), strict=True)
        except RuntimeError as error:
            raise ValueError("RNA VAE decoder topology does not match HEIR") from error
        self.freeze_expression_decoder(freeze)

    def freeze_expression_decoder(self, freeze: bool = True) -> None:
        """Freeze or unfreeze the expression decoder."""

        for parameter in self.expression_decoder.parameters():
            parameter.requires_grad_(not freeze)

    @torch.no_grad()
    def configure_residual_geometry(
        self,
        type_bases: Tensor,
        type_max_norms: Tensor,
        *,
        freeze_basis: bool = True,
    ) -> None:
        """Install RNA-derived within-type bases and calibrated latent bounds.

        ``type_bases`` must contain one orthonormal latent-by-rank basis per
        modeled type. ``type_max_norms`` expresses the allowed displacement in
        the same frozen RNA latent geometry. The configuration scalar remains
        only the backward-compatible fallback before geometry is installed.
        """

        if self.config.legacy_unrestricted_residual or self.residual_type_basis is None:
            raise ValueError("RNA residual geometry requires the restricted residual model")
        expected_basis = (
            self.config.num_cell_types,
            self.config.latent_dim,
            self.config.residual_rank,
        )
        if tuple(type_bases.shape) != expected_basis:
            raise ValueError("type_bases has the wrong shape")
        if tuple(type_max_norms.shape) != (self.config.num_cell_types,):
            raise ValueError("type_max_norms must contain one value per cell type")
        if not torch.isfinite(type_bases).all() or not torch.isfinite(type_max_norms).all():
            raise ValueError("residual geometry must be finite")
        if bool((type_max_norms <= 0).any()):
            raise ValueError("type_max_norms must be positive")
        gram = type_bases.transpose(1, 2) @ type_bases
        identity = torch.eye(
            self.config.residual_rank,
            device=type_bases.device,
            dtype=type_bases.dtype,
        ).expand_as(gram)
        if not torch.allclose(gram, identity, atol=1.0e-4, rtol=1.0e-4):
            raise ValueError("type_bases must be orthonormal")
        self.residual_type_basis.copy_(
            type_bases.to(
                device=self.residual_type_basis.device,
                dtype=self.residual_type_basis.dtype,
            )
        )
        self.residual_type_max_norms.copy_(
            type_max_norms.to(
                device=self.residual_type_max_norms.device,
                dtype=self.residual_type_max_norms.dtype,
            )
        )
        self.residual_type_basis.requires_grad_(not freeze_basis)

    def checkpoint(self) -> Dict[str, Any]:
        """Create a self-describing checkpoint."""

        return {
            "schema": _HEIR_CHECKPOINT_SCHEMA,
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "residual_geometry": {
                "type_max_norms": self.residual_type_max_norms.detach().cpu(),
                "type_strategy": self.config.residual_type_strategy,
                "type_concentration_threshold": (self.config.residual_type_concentration_threshold),
                "basis_trainable": bool(
                    self.residual_type_basis is not None and self.residual_type_basis.requires_grad
                ),
            },
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        strict: bool = True,
        *,
        allow_legacy_mixed_residual_basis: bool = False,
    ) -> "HEIRModel":
        """Reconstruct a model from :meth:`checkpoint` output.

        Early v3 restricted-residual checkpoints did not record their use of
        a probability-weighted mixture of non-aligned type bases.  Loading one
        requires an explicit opt-in so it cannot silently acquire the new
        detached-type semantics.
        """

        if "config" not in checkpoint or "state_dict" not in checkpoint:
            raise KeyError("checkpoint must contain config and state_dict")
        schema = checkpoint.get("schema")
        if schema not in {None, _LEGACY_TIED_CHECKPOINT_SCHEMA, _HEIR_CHECKPOINT_SCHEMA}:
            raise ValueError("unsupported HEIR model checkpoint schema")
        config_values = dict(checkpoint["config"])
        if schema is None:
            # v1 checkpoints predate the tied absolute-latent formulation.
            # Preserve their abundance-weighted local routing, independent
            # query, unconditioned known mixture, and Euclidean UOT exactly.
            config_values.setdefault("prototype_abundance_logit_weight", 1.0)
            config_values.setdefault("covariance_aware_uot", False)
            config_values.setdefault("legacy_independent_prototype_query", True)
            config_values["legacy_unrestricted_residual"] = True
        elif schema == _LEGACY_TIED_CHECKPOINT_SCHEMA:
            # v2 implemented ``prototype + (image - prototype)``. Recreate that
            # algebraic cancellation intentionally instead of silently loading
            # its full-rank head into the restricted v3 residual.
            config_values["legacy_unrestricted_residual"] = True
        legacy_unrestricted = bool(
            config_values.get("legacy_unrestricted_residual", False)
            or config_values.get("legacy_independent_prototype_query", False)
        )
        if "residual_type_strategy" not in config_values and not legacy_unrestricted:
            if not allow_legacy_mixed_residual_basis:
                raise ValueError(
                    "legacy mixed residual basis requires allow_legacy_mixed_residual_basis=True"
                )
            config_values["residual_type_strategy"] = "legacy_weighted_basis"
            config_values["residual_type_concentration_threshold"] = 0.0
        model = cls(HEIRConfig.from_dict(config_values))
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        geometry = checkpoint.get("residual_geometry")
        if not model.config.legacy_unrestricted_residual and not isinstance(geometry, Mapping):
            raise ValueError("restricted checkpoint residual geometry is missing")
        if isinstance(geometry, Mapping) and not model.config.legacy_unrestricted_residual:
            strategy = geometry.get("type_strategy")
            threshold = geometry.get("type_concentration_threshold")
            if model.config.residual_type_strategy == "legacy_weighted_basis":
                if strategy not in {None, "legacy_weighted_basis"}:
                    raise ValueError("checkpoint residual type strategy differs from config")
            elif strategy != model.config.residual_type_strategy:
                raise ValueError("checkpoint residual type strategy differs from config")
            if threshold is None:
                if model.config.residual_type_strategy != "legacy_weighted_basis":
                    raise ValueError("checkpoint residual concentration threshold is missing")
            elif not math.isclose(
                float(threshold),
                model.config.residual_type_concentration_threshold,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("checkpoint residual concentration threshold differs from config")
            norms = torch.as_tensor(geometry["type_max_norms"])
            if norms.shape != model.residual_type_max_norms.shape:
                raise ValueError("checkpoint residual geometry has the wrong type count")
            model.residual_type_max_norms.copy_(norms.to(model.residual_type_max_norms))
            assert model.residual_type_basis is not None
            model.residual_type_basis.requires_grad_(bool(geometry.get("basis_trainable", True)))
        return model


HEIR = HEIRModel


__all__ = ["HEIRConfig", "HEIROutput", "HEIRModel", "HEIR"]
