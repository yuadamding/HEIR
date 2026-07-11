"""Distributional weak-supervision losses, including unbalanced transport."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor


def _require_float(name: str, value: Tensor, ndim: Optional[int] = None) -> None:
    if not torch.is_floating_point(value):
        raise TypeError("%s must be floating point" % name)
    if ndim is not None and value.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))


def _require_nonnegative(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all() or bool((value < 0).any()):
        raise ValueError("%s must contain finite nonnegative values" % name)


def _weights(value: Optional[Tensor], size: int, reference: Tensor, name: str) -> Tensor:
    if value is None:
        return reference.new_ones(size)
    if value.shape != (size,) or value.device != reference.device:
        raise ValueError("%s must have shape (%d,) on the input device" % (name, size))
    if value.dtype == torch.bool:
        return value.to(dtype=reference.dtype)
    _require_float(name, value)
    _require_nonnegative(name, value)
    return value.to(dtype=reference.dtype)


def aggregate_composition(
    type_probabilities: Tensor,
    sample_index: Optional[Tensor] = None,
    cell_weights: Optional[Tensor] = None,
    num_samples: Optional[int] = None,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Aggregate per-cell posteriors into sample compositions and validity flags."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    _require_float("type_probabilities", type_probabilities, 2)
    _require_nonnegative("type_probabilities", type_probabilities)
    row_mass = type_probabilities.sum(dim=-1, keepdim=True)
    if bool((row_mass <= eps).any()):
        raise ValueError("every type-probability row must have positive mass")
    probabilities = type_probabilities / row_mass.clamp_min(eps)
    weights = _weights(cell_weights, probabilities.shape[0], probabilities, "cell_weights")
    if sample_index is None:
        mass = weights.sum()
        if not bool(mass > eps):
            return probabilities.new_zeros((1, probabilities.shape[1])), torch.zeros(
                1,
                dtype=torch.bool,
                device=probabilities.device,
            )
        composition = (probabilities * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
        return composition / mass.clamp_min(eps), torch.ones(
            1,
            dtype=torch.bool,
            device=probabilities.device,
        )
    if sample_index.shape != (probabilities.shape[0],) or sample_index.dtype != torch.long:
        raise ValueError("sample_index must be long with one entry per cell")
    if sample_index.device != probabilities.device:
        raise ValueError("sample_index and probabilities must share a device")
    if num_samples is None:
        num_samples = int(sample_index.max()) + 1 if sample_index.numel() else 0
    if num_samples < 0 or (
        sample_index.numel()
        and (bool((sample_index < 0).any()) or int(sample_index.max()) >= num_samples)
    ):
        raise ValueError("sample_index contains an invalid sample")
    numerator = probabilities.new_zeros((num_samples, probabilities.shape[1]))
    numerator = numerator.index_add(0, sample_index, probabilities * weights.unsqueeze(-1))
    denominator = probabilities.new_zeros(num_samples)
    denominator = denominator.index_add(0, sample_index, weights)
    return numerator / denominator.clamp_min(eps).unsqueeze(-1), denominator > eps


def _target_rows(target: Tensor, samples: int, types: int, reference: Tensor) -> Tensor:
    if target.device != reference.device:
        raise ValueError("target and predictions must share a device")
    _require_float("target", target)
    if target.ndim == 1:
        if samples != 1 or target.shape[0] != types:
            raise ValueError("one-dimensional target requires one sample")
        return target.unsqueeze(0).to(reference.dtype)
    if target.shape != (samples, types):
        raise ValueError("target has the wrong sample/type shape")
    return target.to(reference.dtype)


def _valid_mean(values: Tensor, valid: Tensor, weights: Optional[Tensor], eps: float) -> Tensor:
    effective = values.new_ones(values.shape) if weights is None else weights.to(values.dtype)
    if effective.shape != values.shape:
        raise ValueError("sample weights have the wrong shape")
    _require_nonnegative("sample weights", effective)
    effective = effective * valid.to(values.dtype)
    mass = effective.sum()
    return (
        (values * effective).sum() / mass.clamp_min(eps) if bool(mass > eps) else values.sum() * 0.0
    )


def jensen_shannon_composition_loss(
    type_probabilities: Tensor,
    target_composition: Tensor,
    sample_index: Optional[Tensor] = None,
    cell_weights: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Jensen-Shannon divergence between inferred and snRNA compositions."""

    samples = target_composition.shape[0] if target_composition.ndim == 2 else 1
    predicted, valid = aggregate_composition(
        type_probabilities,
        sample_index,
        cell_weights,
        samples,
        eps,
    )
    target = _target_rows(
        target_composition,
        predicted.shape[0],
        predicted.shape[1],
        predicted,
    )
    _require_nonnegative("target_composition", target)
    target_mass = target.sum(dim=-1, keepdim=True)
    valid = valid & (target_mass.squeeze(-1) > eps)
    target = target / target_mass.clamp_min(eps)
    midpoint = 0.5 * (predicted + target)
    left = predicted * (predicted.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    right = target * (target.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    per_sample = 0.5 * (left.sum(dim=-1) + right.sum(dim=-1))
    reduction_weights = None
    if sample_weights is not None:
        reduction_weights = _weights(
            sample_weights,
            predicted.shape[0],
            per_sample,
            "sample_weights",
        )
    return _valid_mean(per_sample, valid, reduction_weights, eps)


def soft_composition_bounds_loss(
    type_probabilities: Tensor,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
    sample_index: Optional[Tensor] = None,
    cell_weights: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Squared soft penalty outside biologically plausible composition bounds."""

    samples = lower_bounds.shape[0] if lower_bounds.ndim == 2 else 1
    predicted, valid = aggregate_composition(
        type_probabilities,
        sample_index,
        cell_weights,
        samples,
        eps,
    )
    lower = _target_rows(lower_bounds, samples, predicted.shape[1], predicted)
    upper = _target_rows(upper_bounds, samples, predicted.shape[1], predicted)
    _require_nonnegative("lower_bounds", lower)
    _require_nonnegative("upper_bounds", upper)
    if bool((lower > upper).any()) or bool((upper > 1).any()):
        raise ValueError("composition bounds must satisfy 0 <= lower <= upper <= 1")
    per_sample = (
        torch.relu(lower - predicted).square() + torch.relu(predicted - upper).square()
    ).mean(dim=-1)
    reduction_weights = None
    if sample_weights is not None:
        reduction_weights = _weights(sample_weights, samples, per_sample, "sample_weights")
    return _valid_mean(per_sample, valid, reduction_weights, eps)


def dirichlet_composition_prior_loss(
    type_probabilities: Tensor,
    concentration: Tensor,
    sample_index: Optional[Tensor] = None,
    cell_weights: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Negative log density under a sample-specific soft Dirichlet prior."""

    samples = concentration.shape[0] if concentration.ndim == 2 else 1
    predicted, valid = aggregate_composition(
        type_probabilities,
        sample_index,
        cell_weights,
        samples,
        eps,
    )
    alpha = _target_rows(concentration, samples, predicted.shape[1], predicted)
    if not torch.isfinite(alpha).all() or bool((alpha <= 0).any()):
        raise ValueError("Dirichlet concentration must be finite and positive")
    log_normalizer = torch.lgamma(alpha.sum(dim=-1)) - torch.lgamma(alpha).sum(dim=-1)
    log_density = log_normalizer + ((alpha - 1.0) * predicted.clamp_min(eps).log()).sum(dim=-1)
    per_sample = -log_density
    reduction_weights = None
    if sample_weights is not None:
        reduction_weights = _weights(sample_weights, samples, per_sample, "sample_weights")
    return _valid_mean(per_sample, valid, reduction_weights, eps)


@dataclass
class UnbalancedSinkhornResult:
    """Transport plan, objective, and marginal diagnostics."""

    loss: Tensor
    plan: Tensor
    source_marginal: Tensor
    target_marginal: Tensor
    desired_source: Tensor
    desired_target: Tensor
    transport_cost: Tensor
    entropy: Tensor
    source_marginal_error: Tensor
    target_marginal_error: Tensor
    unassigned_mass: Tensor

    def diagnostics(self) -> Dict[str, Tensor]:
        """Return scalar tensors suitable for structured logging."""

        return {
            "uot/transport_cost": self.transport_cost.mean(),
            "uot/entropy": self.entropy.mean(),
            "uot/source_marginal_error": self.source_marginal_error.mean(),
            "uot/target_marginal_error": self.target_marginal_error.mean(),
            "uot/unassigned_mass": self.unassigned_mass.mean(),
        }


def _as_batched(value: Tensor, dimensions: int, name: str) -> Tuple[Tensor, bool]:
    if value.ndim == dimensions - 1:
        return value.unsqueeze(0), True
    if value.ndim != dimensions:
        raise ValueError("%s has the wrong number of dimensions" % name)
    return value, False


def _generalized_kl(marginal: Tensor, desired: Tensor, mask: Tensor, eps: float) -> Tensor:
    safe_marginal = marginal.clamp_min(eps)
    safe_desired = desired.clamp_min(eps)
    terms = marginal * (safe_marginal.log() - safe_desired.log()) - marginal + desired
    return (terms * mask.to(terms.dtype)).sum(dim=-1)


def unbalanced_sinkhorn(
    cost: Tensor,
    source_mass: Optional[Tensor] = None,
    target_mass: Optional[Tensor] = None,
    source_mask: Optional[Tensor] = None,
    target_mask: Optional[Tensor] = None,
    pair_mask: Optional[Tensor] = None,
    *,
    epsilon: float = 0.1,
    marginal_relaxation: float = 1.0,
    iterations: int = 80,
    unknown_mass: Union[float, Tensor] = 0.05,
    unknown_cost: Union[float, Tensor] = 1.0,
    add_unknown: bool = True,
    eps: float = 1e-8,
) -> UnbalancedSinkhornResult:
    """Compute differentiable entropic unbalanced OT in the log domain.

    ``cost`` is ``(sources, prototypes)`` or batched.  When ``add_unknown`` is
    true, a dustbin prototype is appended; transport into its final column is
    reported as unassigned mass.  KL-relaxed marginals permit reference capture
    bias and unmatched molecular states without producing invalid plans.
    """

    if epsilon <= 0 or marginal_relaxation <= 0 or eps <= 0:
        raise ValueError("epsilon, marginal_relaxation, and eps must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not isinstance(unknown_cost, Tensor) and unknown_cost < 0:
        raise ValueError("unknown_cost must be nonnegative")
    _require_float("cost", cost)
    if not torch.isfinite(cost).all() or bool((cost < 0).any()):
        raise ValueError("cost must be finite and nonnegative")
    batched_cost, squeezed = _as_batched(cost, 3, "cost")
    batch, sources, targets = batched_cost.shape
    work_dtype = (
        torch.float32
        if batched_cost.dtype in (torch.float16, torch.bfloat16)
        else batched_cost.dtype
    )
    work_cost = batched_cost.to(dtype=work_dtype)

    def mass_tensor(value: Optional[Tensor], size: int, name: str) -> Tensor:
        if value is None:
            return work_cost.new_ones((batch, size))
        batched, was_squeezed = _as_batched(value, 2, name)
        if batched.shape != (batch, size):
            if batch == 1 and was_squeezed and batched.shape == (1, size):
                pass
            else:
                raise ValueError("%s has the wrong shape" % name)
        if batched.device != work_cost.device:
            raise ValueError("%s and cost must share a device" % name)
        _require_float(name, batched)
        _require_nonnegative(name, batched)
        return batched.to(dtype=work_dtype)

    def mask_tensor(value: Optional[Tensor], size: int, name: str) -> Tensor:
        if value is None:
            return torch.ones((batch, size), dtype=torch.bool, device=work_cost.device)
        batched, _ = _as_batched(value, 2, name)
        if batched.shape != (batch, size) or batched.dtype != torch.bool:
            raise ValueError("%s must be boolean with the matching shape" % name)
        if batched.device != work_cost.device:
            raise ValueError("%s and cost must share a device" % name)
        return batched

    source = mass_tensor(source_mass, sources, "source_mass")
    target = mass_tensor(target_mass, targets, "target_mass")
    source_valid = mask_tensor(source_mask, sources, "source_mask") & (source > 0)
    target_valid = mask_tensor(target_mask, targets, "target_mask") & (target > 0)
    if pair_mask is None:
        pair_valid = torch.ones(
            (batch, sources, targets),
            dtype=torch.bool,
            device=work_cost.device,
        )
    else:
        pair_valid, _ = _as_batched(pair_mask, 3, "pair_mask")
        if pair_valid.shape != (batch, sources, targets) or pair_valid.dtype != torch.bool:
            raise ValueError("pair_mask must be boolean with the same shape as cost")
        if pair_valid.device != work_cost.device:
            raise ValueError("pair_mask and cost must share a device")
    source = source * source_valid
    target = target * target_valid
    source_total = source.sum(dim=-1, keepdim=True)
    if bool((source_total <= eps).any()):
        raise ValueError("every batch must contain positive source mass")
    source = source / source_total.clamp_min(eps)

    if isinstance(unknown_mass, Tensor):
        unknown = unknown_mass.to(device=work_cost.device, dtype=work_dtype)
        if unknown.ndim == 0:
            unknown = unknown.expand(batch)
        if unknown.shape != (batch,):
            raise ValueError("unknown_mass tensor must be scalar or have one value per batch")
    else:
        unknown = work_cost.new_full((batch,), float(unknown_mass))
    if not torch.isfinite(unknown).all() or bool((unknown < 0).any()) or bool((unknown >= 1).any()):
        raise ValueError("unknown_mass must be in [0, 1)")

    target_total = target.sum(dim=-1, keepdim=True)
    if add_unknown:
        no_real_target = target_total.squeeze(-1) <= eps
        real_fraction = torch.where(no_real_target, torch.zeros_like(unknown), 1.0 - unknown)
        target = target / target_total.clamp_min(eps) * real_fraction.unsqueeze(-1)
        effective_unknown = torch.where(no_real_target, torch.ones_like(unknown), unknown)
        target = torch.cat((target, effective_unknown.unsqueeze(-1)), dim=-1)
        target_valid = torch.cat(
            (target_valid, (effective_unknown > 0).unsqueeze(-1)),
            dim=-1,
        )
        if isinstance(unknown_cost, Tensor):
            dustbin_cost = unknown_cost.to(device=work_cost.device, dtype=work_dtype)
            if dustbin_cost.ndim == 0:
                dustbin_cost = dustbin_cost.expand(batch, sources)
            elif dustbin_cost.ndim == 1:
                if dustbin_cost.shape != (sources,):
                    raise ValueError("unknown_cost vector must have one value per source")
                dustbin_cost = dustbin_cost.unsqueeze(0).expand(batch, -1)
            elif dustbin_cost.shape != (batch, sources):
                raise ValueError(
                    "unknown_cost tensor must be scalar, (sources,), or (batch, sources)"
                )
            if not torch.isfinite(dustbin_cost).all() or bool((dustbin_cost < 0).any()):
                raise ValueError("unknown_cost must be finite and nonnegative")
            unknown_column = dustbin_cost.unsqueeze(-1)
        else:
            unknown_column = work_cost.new_full((batch, sources, 1), float(unknown_cost))
        work_cost = torch.cat((work_cost, unknown_column), dim=-1)
        pair_valid = torch.cat(
            (
                pair_valid,
                torch.ones((batch, sources, 1), dtype=torch.bool, device=pair_valid.device),
            ),
            dim=-1,
        )
    else:
        if bool((target_total <= eps).any()):
            raise ValueError("every batch must contain positive target mass")
        target = target / target_total.clamp_min(eps)
        effective_unknown = target.new_zeros(batch)

    log_kernel = -work_cost / epsilon
    valid_pairs = source_valid.unsqueeze(-1) & target_valid.unsqueeze(-2) & pair_valid
    scalable_source = source_valid & valid_pairs.any(dim=-1)
    scalable_target = target_valid & valid_pairs.any(dim=-2)
    log_kernel = log_kernel.masked_fill(~valid_pairs, -torch.inf)
    log_source = source.clamp_min(eps).log()
    log_target = target.clamp_min(eps).log()
    log_u = work_cost.new_zeros((batch, sources))
    log_v = work_cost.new_zeros((batch, target.shape[1]))
    exponent = marginal_relaxation / (marginal_relaxation + epsilon)
    for _ in range(iterations):
        # logsumexp over an all -inf row/column has undefined gradients even
        # when a later torch.where masks its output. Insert a finite dummy
        # reduction only for zero-mass sources/targets; those updates are then
        # discarded exactly by the scalable masks.
        source_update_kernel = torch.where(
            scalable_source.unsqueeze(-1), log_kernel, torch.zeros_like(log_kernel)
        )
        kernel_v = torch.logsumexp(source_update_kernel + log_v.unsqueeze(-2), dim=-1)
        updated_u = exponent * (log_source - kernel_v)
        log_u = torch.where(scalable_source, updated_u, torch.zeros_like(updated_u))
        target_update_kernel = torch.where(
            scalable_target.unsqueeze(-2), log_kernel, torch.zeros_like(log_kernel)
        )
        kernel_u = torch.logsumexp(target_update_kernel + log_u.unsqueeze(-1), dim=-2)
        updated_v = exponent * (log_target - kernel_u)
        log_v = torch.where(scalable_target, updated_v, torch.zeros_like(updated_v))

    log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    plan = torch.exp(log_plan)
    plan = torch.where(valid_pairs, plan, torch.zeros_like(plan))
    source_marginal = plan.sum(dim=-1)
    target_marginal = plan.sum(dim=-2)
    transport_cost = (plan * work_cost).sum(dim=(-2, -1))
    entropy = (plan * (plan.clamp_min(eps).log() - 1.0)).sum(dim=(-2, -1))
    source_kl = _generalized_kl(source_marginal, source, source_valid, eps)
    target_kl = _generalized_kl(target_marginal, target, target_valid, eps)
    # Entropic regularization is part of the optimized objective, not merely
    # the fixed-point solver. ``entropy`` is sum T(log T - 1), so adding it is
    # equivalent to subtracting the usual positive Shannon entropy.
    per_batch_loss = (
        transport_cost + epsilon * entropy + marginal_relaxation * (source_kl + target_kl)
    )
    source_error = (source_marginal - source).abs().sum(dim=-1)
    target_error = (target_marginal - target).abs().sum(dim=-1)
    unassigned = target_marginal[:, -1] if add_unknown else effective_unknown

    if squeezed:
        plan = plan.squeeze(0)
        source_marginal = source_marginal.squeeze(0)
        target_marginal = target_marginal.squeeze(0)
        desired_source = source.squeeze(0)
        desired_target = target.squeeze(0)
        transport_cost = transport_cost.squeeze(0)
        entropy = entropy.squeeze(0)
        source_error = source_error.squeeze(0)
        target_error = target_error.squeeze(0)
        unassigned = unassigned.squeeze(0)
    else:
        desired_source = source
        desired_target = target
    return UnbalancedSinkhornResult(
        loss=per_batch_loss.mean(),
        plan=plan,
        source_marginal=source_marginal,
        target_marginal=target_marginal,
        desired_source=desired_source,
        desired_target=desired_target,
        transport_cost=transport_cost,
        entropy=entropy,
        source_marginal_error=source_error,
        target_marginal_error=target_error,
        unassigned_mass=unassigned,
    )


def unbalanced_sinkhorn_loss(
    cost: Tensor,
    source_mass: Optional[Tensor] = None,
    target_mass: Optional[Tensor] = None,
    source_mask: Optional[Tensor] = None,
    target_mask: Optional[Tensor] = None,
    pair_mask: Optional[Tensor] = None,
    return_diagnostics: bool = False,
    **kwargs: Any,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    """Scalar convenience wrapper around :func:`unbalanced_sinkhorn`."""

    result = unbalanced_sinkhorn(
        cost,
        source_mass,
        target_mass,
        source_mask,
        target_mask,
        pair_mask,
        **kwargs,
    )
    if return_diagnostics:
        return result.loss, result.diagnostics()
    return result.loss


composition_js_loss = jensen_shannon_composition_loss
composition_bounds_loss = soft_composition_bounds_loss
dirichlet_prior_loss = dirichlet_composition_prior_loss
prototype_uot = unbalanced_sinkhorn


__all__ = [
    "aggregate_composition",
    "jensen_shannon_composition_loss",
    "composition_js_loss",
    "soft_composition_bounds_loss",
    "composition_bounds_loss",
    "dirichlet_composition_prior_loss",
    "dirichlet_prior_loss",
    "UnbalancedSinkhornResult",
    "unbalanced_sinkhorn",
    "unbalanced_sinkhorn_loss",
    "prototype_uot",
]
