"""Biological, graph, residual, and anchor constraints for HEIR."""

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from ..expression import EXPRESSION_MAX
from ..models.graph import validate_edge_index, validate_edge_weight


def _float_tensor(name: str, value: Tensor, ndim: Optional[int] = None) -> None:
    if not torch.is_floating_point(value):
        raise TypeError("%s must be floating point" % name)
    if ndim is not None and value.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))


def _finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError("%s must be finite" % name)


def _vector_weights(
    value: Optional[Tensor],
    size: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    if value is None:
        return reference.new_ones(size)
    if value.shape != (size,) or value.device != reference.device:
        raise ValueError("%s has the wrong shape or device" % name)
    if value.dtype == torch.bool:
        return value.to(reference.dtype)
    _float_tensor(name, value)
    if not torch.isfinite(value).all() or bool((value < 0).any()):
        raise ValueError("%s must be finite and nonnegative" % name)
    return value.to(reference.dtype)


def _weighted_mean(values: Tensor, weights: Tensor, eps: float) -> Tuple[Tensor, Tensor]:
    mass = weights.sum()
    if not bool(mass > eps):
        return values.new_zeros(values.shape[1]), mass
    return (values * weights.unsqueeze(-1)).sum(dim=0) / mass.clamp_min(eps), mass


def _aggregate_expression(
    expression: Tensor,
    samples: int,
    sample_index: Optional[Tensor],
    cell_weights: Optional[Tensor],
    eps: float,
    log1p_expression: bool = False,
) -> Tuple[Tensor, Tensor]:
    values = expression
    if log1p_expression:
        if bool((expression < 0).any()):
            raise ValueError("log1p expression must be non-negative")
        values = torch.expm1(expression.clamp_max(EXPRESSION_MAX))
    weights = _vector_weights(cell_weights, expression.shape[0], expression, "cell_weights")
    if sample_index is None:
        if samples != 1:
            raise ValueError("sample_index is required for multiple pseudobulks")
        mean, mass = _weighted_mean(values, weights, eps)
        if log1p_expression:
            mean = torch.log1p(mean)
        return mean.unsqueeze(0), (mass > eps).unsqueeze(0)
    if sample_index.shape != (expression.shape[0],) or sample_index.dtype != torch.long:
        raise ValueError("sample_index must be long with one entry per cell")
    if sample_index.device != expression.device:
        raise ValueError("sample_index and expression must share a device")
    if sample_index.numel() and (
        bool((sample_index < 0).any()) or int(sample_index.max()) >= samples
    ):
        raise ValueError("sample_index contains an invalid sample")
    numerator = expression.new_zeros((samples, expression.shape[1]))
    numerator = numerator.index_add(0, sample_index, values * weights.unsqueeze(-1))
    denominator = expression.new_zeros(samples)
    denominator = denominator.index_add(0, sample_index, weights)
    aggregated = numerator / denominator.clamp_min(eps).unsqueeze(-1)
    if log1p_expression:
        aggregated = torch.log1p(aggregated)
    return aggregated, denominator > eps


def _masked_regression(
    predicted: Tensor,
    target: Tensor,
    feature_weights: Tensor,
    metric: str,
    eps: float,
) -> Tuple[Tensor, Tensor]:
    finite = torch.isfinite(target)
    safe_target = torch.where(finite, target, torch.zeros_like(target))
    weights = feature_weights * finite.to(feature_weights.dtype)
    valid = weights.sum(dim=-1) > eps
    if metric == "mse":
        values = ((predicted - safe_target).square() * weights).sum(dim=-1)
        values = values / weights.sum(dim=-1).clamp_min(eps)
    elif metric == "smooth_l1":
        errors = F.smooth_l1_loss(predicted, safe_target, reduction="none")
        values = (errors * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(eps)
    elif metric == "cosine":
        left = predicted * weights.sqrt()
        right = safe_target * weights.sqrt()
        denominator = left.norm(dim=-1) * right.norm(dim=-1)
        valid = valid & (denominator > eps)
        values = 1.0 - (left * right).sum(dim=-1) / denominator.clamp_min(eps)
    else:
        raise ValueError("metric must be mse, smooth_l1, or cosine")
    return values, valid


def _reduce(values: Tensor, valid: Tensor, weights: Optional[Tensor], eps: float) -> Tensor:
    effective = values.new_ones(values.shape) if weights is None else weights.to(values.dtype)
    if effective.shape != values.shape:
        raise ValueError("reduction weights have the wrong shape")
    effective = effective * valid.to(values.dtype)
    mass = effective.sum()
    return (
        (values * effective).sum() / mass.clamp_min(eps) if bool(mass > eps) else values.sum() * 0.0
    )


def pseudobulk_loss(
    predicted_expression: Tensor,
    target_pseudobulk: Tensor,
    sample_index: Optional[Tensor] = None,
    cell_weights: Optional[Tensor] = None,
    gene_weights: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
    metric: str = "mse",
    eps: float = 1e-8,
    log1p_expression: bool = False,
) -> Tensor:
    """Match aggregated decoded expression to sample snRNA pseudobulk."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    _float_tensor("predicted_expression", predicted_expression, 2)
    _finite("predicted_expression", predicted_expression)
    if target_pseudobulk.device != predicted_expression.device:
        raise ValueError("target_pseudobulk and expression must share a device")
    _float_tensor("target_pseudobulk", target_pseudobulk)
    if target_pseudobulk.ndim == 1:
        target = target_pseudobulk.unsqueeze(0)
    elif target_pseudobulk.ndim == 2:
        target = target_pseudobulk
    else:
        raise ValueError("target_pseudobulk must have one or two dimensions")
    if target.shape[1] != predicted_expression.shape[1]:
        raise ValueError("target_pseudobulk has the wrong gene dimension")
    aggregated, valid = _aggregate_expression(
        predicted_expression,
        target.shape[0],
        sample_index,
        cell_weights,
        eps,
        log1p_expression,
    )
    if gene_weights is None:
        features = predicted_expression.new_ones(target.shape)
    else:
        genes = _vector_weights(
            gene_weights,
            target.shape[1],
            predicted_expression,
            "gene_weights",
        )
        features = genes.unsqueeze(0).expand_as(target)
    values, finite_rows = _masked_regression(
        aggregated,
        target.to(predicted_expression.dtype),
        features,
        metric,
        eps,
    )
    reduction_weights = None
    if sample_weights is not None:
        reduction_weights = _vector_weights(
            sample_weights,
            target.shape[0],
            values,
            "sample_weights",
        )
    return _reduce(values, valid & finite_rows, reduction_weights, eps)


def program_score_loss(
    predicted_expression: Tensor,
    program_matrix: Tensor,
    target_program_scores: Tensor,
    sample_index: Optional[Tensor] = None,
    cell_weights: Optional[Tensor] = None,
    program_weights: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
    metric: str = "mse",
    eps: float = 1e-8,
) -> Tensor:
    """Match sample-level scores for signed or unsigned gene programs."""

    _float_tensor("program_matrix", program_matrix, 2)
    if program_matrix.device != predicted_expression.device:
        raise ValueError("program_matrix and expression must share a device")
    if program_matrix.shape[0] != predicted_expression.shape[1]:
        raise ValueError("program_matrix must have shape (genes, programs)")
    _finite("program_matrix", program_matrix)
    scores = predicted_expression.matmul(program_matrix.to(predicted_expression.dtype))
    return pseudobulk_loss(
        scores,
        target_program_scores,
        sample_index,
        cell_weights,
        program_weights,
        sample_weights,
        metric,
        eps,
    )


def type_conditioned_program_score_loss(
    predicted_expression: Tensor,
    type_probabilities: Tensor,
    program_matrix: Tensor,
    target_type_program_scores: Tensor,
    cell_weights: Optional[Tensor] = None,
    program_weights: Optional[Tensor] = None,
    type_weights: Optional[Tensor] = None,
    metric: str = "mse",
    eps: float = 1e-8,
) -> Tensor:
    """Match program activity within cell types, not only in sample means."""

    _float_tensor("predicted_expression", predicted_expression, 2)
    _float_tensor("type_probabilities", type_probabilities, 2)
    _float_tensor("program_matrix", program_matrix, 2)
    if type_probabilities.shape[0] != predicted_expression.shape[0]:
        raise ValueError("type probabilities and expression must share cells")
    if program_matrix.shape[0] != predicted_expression.shape[1]:
        raise ValueError("program_matrix must have shape (genes, programs)")
    expected = (type_probabilities.shape[1], program_matrix.shape[1])
    if target_type_program_scores.shape != expected:
        raise ValueError("target type-program scores must have shape (types, programs)")
    if any(
        value.device != predicted_expression.device
        for value in (type_probabilities, program_matrix, target_type_program_scores)
    ):
        raise ValueError("type-conditioned program tensors must share a device")
    scores = predicted_expression.matmul(program_matrix.to(predicted_expression.dtype))
    weights = _vector_weights(
        cell_weights, predicted_expression.shape[0], predicted_expression, "cell_weights"
    )
    probabilities = type_probabilities / type_probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
    assignment = probabilities * weights.unsqueeze(-1)
    mass = assignment.sum(dim=0)
    predicted = assignment.transpose(0, 1).matmul(scores)
    predicted = predicted / mass.clamp_min(eps).unsqueeze(-1)
    if program_weights is None:
        features = predicted.new_ones(expected)
    else:
        features = (
            _vector_weights(program_weights, expected[1], predicted, "program_weights")
            .unsqueeze(0)
            .expand(expected)
        )
    values, finite = _masked_regression(
        predicted,
        target_type_program_scores.to(predicted.dtype),
        features,
        metric,
        eps,
    )
    reduction = None
    if type_weights is not None:
        reduction = _vector_weights(type_weights, expected[0], values, "type_weights")
    return _reduce(values, finite & (mass > eps), reduction, eps)


def marker_ranking_loss(
    predicted_expression: Tensor,
    type_probabilities: Tensor,
    marker_mask: Tensor,
    cell_weights: Optional[Tensor] = None,
    margin: float = 0.1,
    eps: float = 1e-8,
) -> Tensor:
    """Rank the likely type's marker score above competing marker sets."""

    _float_tensor("predicted_expression", predicted_expression, 2)
    _float_tensor("type_probabilities", type_probabilities, 2)
    if margin < 0 or eps <= 0:
        raise ValueError("marker margin must be nonnegative and eps positive")
    expected = (type_probabilities.shape[1], predicted_expression.shape[1])
    if marker_mask.shape != expected or marker_mask.device != predicted_expression.device:
        raise ValueError("marker_mask must have shape (types, genes)")
    if marker_mask.dtype == torch.bool:
        markers = marker_mask.to(predicted_expression.dtype)
    else:
        _float_tensor("marker_mask", marker_mask, 2)
        if bool((marker_mask < 0).any()) or not torch.isfinite(marker_mask).all():
            raise ValueError("marker_mask must be finite and nonnegative")
        markers = marker_mask.to(predicted_expression.dtype)
    supported = markers.sum(dim=-1) > 0
    if not bool(supported.any()):
        return predicted_expression.sum() * 0.0
    markers = markers / markers.sum(dim=-1, keepdim=True).clamp_min(eps)
    scores = predicted_expression.matmul(markers.transpose(0, 1))
    classes = scores.shape[1]
    diagonal = torch.eye(classes, dtype=torch.bool, device=scores.device).unsqueeze(0)
    competitors = scores.unsqueeze(1).expand(-1, classes, -1).masked_fill(diagonal, float("-inf"))
    strongest_competitor = competitors.max(dim=-1).values
    hinge = torch.relu(margin - scores + strongest_competitor)
    probabilities = type_probabilities / type_probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
    values = (probabilities * hinge * supported.to(probabilities.dtype)).sum(dim=-1)
    weights = _vector_weights(
        cell_weights, predicted_expression.shape[0], predicted_expression, "cell_weights"
    )
    return _reduce(values, weights > 0, weights, eps)


def marker_centroid_loss(
    predicted_expression: Tensor,
    type_probabilities: Tensor,
    target_centroids: Tensor,
    marker_mask: Optional[Tensor] = None,
    cell_weights: Optional[Tensor] = None,
    type_weights: Optional[Tensor] = None,
    metric: str = "mse",
    min_type_mass: float = 1e-6,
    eps: float = 1e-8,
) -> Tensor:
    """Match soft type-specific marker centroids to the RNA reference."""

    _float_tensor("predicted_expression", predicted_expression, 2)
    _float_tensor("type_probabilities", type_probabilities, 2)
    _finite("predicted_expression", predicted_expression)
    if predicted_expression.shape[0] != type_probabilities.shape[0]:
        raise ValueError("expression and type probabilities must share cells")
    expected = (type_probabilities.shape[1], predicted_expression.shape[1])
    if target_centroids.shape != expected or target_centroids.device != predicted_expression.device:
        raise ValueError("target_centroids has the wrong shape or device")
    if min_type_mass < 0 or eps <= 0:
        raise ValueError("min_type_mass must be nonnegative and eps positive")
    if bool((type_probabilities < 0).any()) or not torch.isfinite(type_probabilities).all():
        raise ValueError("type probabilities must be finite and nonnegative")
    probabilities = type_probabilities / type_probabilities.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    weights = _vector_weights(
        cell_weights,
        predicted_expression.shape[0],
        predicted_expression,
        "cell_weights",
    )
    assignment = probabilities * weights.unsqueeze(-1)
    mass = assignment.sum(dim=0)
    centroids = assignment.transpose(0, 1).matmul(predicted_expression)
    centroids = centroids / mass.clamp_min(eps).unsqueeze(-1)
    if marker_mask is None:
        feature_weights = predicted_expression.new_ones(expected)
    else:
        if marker_mask.shape != expected or marker_mask.device != predicted_expression.device:
            raise ValueError("marker_mask has the wrong shape or device")
        if marker_mask.dtype == torch.bool:
            feature_weights = marker_mask.to(predicted_expression.dtype)
        else:
            _float_tensor("marker_mask", marker_mask)
            if bool((marker_mask < 0).any()) or not torch.isfinite(marker_mask).all():
                raise ValueError("marker_mask must be nonnegative and finite")
            feature_weights = marker_mask.to(predicted_expression.dtype)
    values, valid_features = _masked_regression(
        centroids,
        target_centroids.to(predicted_expression.dtype),
        feature_weights,
        metric,
        eps,
    )
    reduction_weights = None
    if type_weights is not None:
        reduction_weights = _vector_weights(
            type_weights,
            expected[0],
            values,
            "type_weights",
        )
    return _reduce(values, (mass > min_type_mass) & valid_features, reduction_weights, eps)


def scgpt_representation_loss(
    predicted_embedding: Tensor,
    type_probabilities: Tensor,
    type_prototypes: Tensor,
    type_variances: Optional[Tensor] = None,
    cell_weights: Optional[Tensor] = None,
    contrastive_temperature: float = 0.1,
    covariance_weight: float = 0.25,
    eps: float = 1e-8,
) -> Tensor:
    """Align the image student to a frozen scGPT teacher distribution.

    The cosine and soft-label contrastive terms implement the blueprint's
    prototype alignment.  When diagonal teacher variances are available, a
    type-weighted moment term prevents the image branch from collapsing every
    cell onto its type centroid.
    """

    _float_tensor("predicted_embedding", predicted_embedding, 2)
    _float_tensor("type_probabilities", type_probabilities, 2)
    _float_tensor("type_prototypes", type_prototypes, 2)
    _finite("predicted_embedding", predicted_embedding)
    _finite("type_probabilities", type_probabilities)
    _finite("type_prototypes", type_prototypes)
    cells, embedding_dim = predicted_embedding.shape
    types = type_probabilities.shape[1]
    if type_probabilities.shape[0] != cells:
        raise ValueError("scGPT embeddings and type probabilities must share cells")
    if type_prototypes.shape != (types, embedding_dim):
        raise ValueError("scGPT prototypes must have shape (types, embedding_dim)")
    if predicted_embedding.device != type_prototypes.device:
        raise ValueError("scGPT predictions and prototypes must share a device")
    if contrastive_temperature <= 0 or covariance_weight < 0 or eps <= 0:
        raise ValueError("invalid scGPT loss temperature, covariance weight, or epsilon")
    if bool((type_probabilities < 0).any()):
        raise ValueError("type probabilities must be nonnegative")
    probabilities = type_probabilities / type_probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
    weights = _vector_weights(cell_weights, cells, predicted_embedding, "cell_weights")
    valid = weights > 0
    target = probabilities.matmul(type_prototypes.to(predicted_embedding.dtype))
    cosine = 1.0 - F.cosine_similarity(predicted_embedding, target, dim=-1, eps=eps)
    normalized_prediction = F.normalize(predicted_embedding, dim=-1, eps=eps)
    normalized_prototypes = F.normalize(type_prototypes, dim=-1, eps=eps)
    logits = normalized_prediction.matmul(normalized_prototypes.transpose(0, 1))
    contrastive = -(probabilities * F.log_softmax(logits / contrastive_temperature, dim=-1)).sum(
        dim=-1
    )
    per_cell = cosine + contrastive
    result = _reduce(per_cell, valid, weights, eps)

    if type_variances is not None and covariance_weight:
        if type_variances.shape != type_prototypes.shape:
            raise ValueError("scGPT variances must match type prototypes")
        if type_variances.device != predicted_embedding.device:
            raise ValueError("scGPT variances and predictions must share a device")
        _float_tensor("type_variances", type_variances, 2)
        _finite("type_variances", type_variances)
        if bool((type_variances < 0).any()):
            raise ValueError("scGPT variances must be nonnegative")
        assignment = probabilities * weights.unsqueeze(-1)
        mass = assignment.sum(dim=0)
        predicted_mean = assignment.transpose(0, 1).matmul(predicted_embedding)
        predicted_mean = predicted_mean / mass.clamp_min(eps).unsqueeze(-1)
        centered = predicted_embedding.unsqueeze(1) - predicted_mean.unsqueeze(0)
        predicted_variance = (assignment.unsqueeze(-1) * centered.square()).sum(
            dim=0
        ) / mass.clamp_min(eps).unsqueeze(-1)
        moment_error = F.smooth_l1_loss(
            predicted_variance,
            type_variances.to(predicted_embedding.dtype),
            reduction="none",
        ).mean(dim=-1)
        supported = mass > eps
        moment = _reduce(moment_error, supported, None, eps)
        result = result + covariance_weight * moment
    return result


def residual_mahalanobis_loss(
    residual: Tensor,
    precision: Optional[Tensor] = None,
    assignment_probabilities: Optional[Tensor] = None,
    residual_logvar: Optional[Tensor] = None,
    cell_weights: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Penalize morphology residuals under reference prototype precision.

    Precision can be shared diagonal ``(L,)``, shared full ``(L, L)``,
    prototype diagonal ``(P, L)``, or prototype full ``(P, L, L)``.  The latter
    two require soft prototype assignments.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    _float_tensor("residual", residual, 2)
    _finite("residual", residual)
    cells, latent_dim = residual.shape
    weights = _vector_weights(cell_weights, cells, residual, "cell_weights")
    valid = torch.ones(cells, dtype=torch.bool, device=residual.device)

    if precision is None:
        if residual_logvar is None or residual_logvar.shape != residual.shape:
            raise ValueError("supply precision or residual_logvar with residual shape")
        if residual_logvar.device != residual.device:
            raise ValueError("residual_logvar and residual must share a device")
        _float_tensor("residual_logvar", residual_logvar)
        _finite("residual_logvar", residual_logvar)
        values = (residual.square() * torch.exp(-residual_logvar)).mean(dim=-1)
    else:
        if precision.device != residual.device:
            raise ValueError("precision and residual must share a device")
        _float_tensor("precision", precision)
        _finite("precision", precision)
        precision = precision.to(residual.dtype)
        if assignment_probabilities is None:
            if precision.shape == (latent_dim,):
                if bool((precision < 0).any()):
                    raise ValueError("diagonal precision must be nonnegative")
                values = (residual.square() * precision).sum(dim=-1) / latent_dim
            elif precision.shape == (latent_dim, latent_dim):
                values = torch.einsum("ni,ij,nj->n", residual, precision, residual)
                values = values.clamp_min(0.0) / latent_dim
            else:
                raise ValueError("shared precision has the wrong shape")
        else:
            _float_tensor("assignment_probabilities", assignment_probabilities, 2)
            if assignment_probabilities.shape[0] != cells:
                raise ValueError("assignments must contain one row per cell")
            if assignment_probabilities.device != residual.device:
                raise ValueError("assignments and residual must share a device")
            if bool((assignment_probabilities < 0).any()):
                raise ValueError("assignments must be nonnegative")
            prototypes = assignment_probabilities.shape[1]
            assignment_mass = assignment_probabilities.sum(dim=-1, keepdim=True)
            valid = assignment_mass.squeeze(-1) > eps
            assignment = assignment_probabilities / assignment_mass.clamp_min(eps)
            if precision.shape == (prototypes, latent_dim):
                if bool((precision < 0).any()):
                    raise ValueError("diagonal precision must be nonnegative")
                per_prototype = (
                    torch.einsum(
                        "nl,pl->np",
                        residual.square(),
                        precision,
                    )
                    / latent_dim
                )
            elif precision.shape == (prototypes, latent_dim, latent_dim):
                per_prototype = (
                    torch.einsum(
                        "ni,pij,nj->np",
                        residual,
                        precision,
                        residual,
                    ).clamp_min(0.0)
                    / latent_dim
                )
            else:
                raise ValueError("prototype precision has the wrong shape")
            values = (assignment * per_prototype).sum(dim=-1)
    return _reduce(values, valid, weights, eps)


def cycle_consistency_loss(
    latent: Tensor,
    reconstructed_latent: Tensor,
    latent_logvar: Optional[Tensor] = None,
    cell_weights: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Require decoded expression re-encoded by the RNA VAE to recover its latent."""

    if latent.shape != reconstructed_latent.shape or latent.ndim != 2:
        raise ValueError("cycle latents must be two-dimensional with identical shapes")
    if latent.device != reconstructed_latent.device:
        raise ValueError("cycle latents must share a device")
    _float_tensor("latent", latent)
    _float_tensor("reconstructed_latent", reconstructed_latent)
    _finite("latent", latent)
    _finite("reconstructed_latent", reconstructed_latent)
    errors = (latent - reconstructed_latent).square()
    if latent_logvar is not None:
        if latent_logvar.shape != latent.shape or latent_logvar.device != latent.device:
            raise ValueError("latent_logvar must match latent")
        _finite("latent_logvar", latent_logvar)
        errors = errors * torch.exp(-latent_logvar)
    values = errors.mean(dim=-1)
    weights = _vector_weights(cell_weights, latent.shape[0], latent, "cell_weights")
    return _reduce(values, torch.ones_like(values, dtype=torch.bool), weights, eps)


def boundary_graph_loss(
    values: Tensor,
    edge_index: Tensor,
    type_probabilities: Optional[Tensor] = None,
    edge_weight: Optional[Tensor] = None,
    boundary_margin: float = 0.0,
    power: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Smooth within likely compartments without blurring predicted boundaries."""

    if values.ndim not in (1, 2) or not torch.is_floating_point(values):
        raise ValueError("values must be a floating tensor of nodes or node features")
    if power < 1 or boundary_margin < 0 or eps <= 0:
        raise ValueError("power must be >=1; boundary_margin nonnegative; eps positive")
    _finite("values", values)
    matrix = values.unsqueeze(-1) if values.ndim == 1 else values
    validate_edge_index(edge_index, matrix.shape[0], matrix.device)
    validate_edge_weight(edge_weight, edge_index)
    if edge_index.shape[1] == 0:
        return values.sum() * 0.0
    weights = _vector_weights(edge_weight, edge_index.shape[1], matrix, "edge_weight")
    source, target = edge_index
    differences = matrix.index_select(0, source) - matrix.index_select(0, target)
    magnitude = differences.abs().pow(power).mean(dim=-1)
    if type_probabilities is None:
        similarity = magnitude.new_ones(magnitude.shape)
    else:
        if type_probabilities.ndim != 2 or type_probabilities.shape[0] != matrix.shape[0]:
            raise ValueError("type_probabilities must have one row per node")
        if type_probabilities.device != matrix.device:
            raise ValueError("type probabilities and graph values must share a device")
        probabilities = type_probabilities / type_probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)
        similarity = (
            (probabilities.index_select(0, source) * probabilities.index_select(0, target))
            .sum(dim=-1)
            .clamp(0.0, 1.0)
        )
    smooth = similarity * magnitude
    if boundary_margin:
        boundary = (1.0 - similarity) * torch.relu(boundary_margin - magnitude).square()
        smooth = smooth + boundary
    mass = weights.sum()
    return (
        (smooth * weights).sum() / mass.clamp_min(eps) if bool(mass > eps) else values.sum() * 0.0
    )


def anchor_classification_loss(
    type_logits: Tensor,
    labels: Tensor,
    anchor_weights: Optional[Tensor] = None,
    class_weights: Optional[Tensor] = None,
    unknown_probability: Optional[Tensor] = None,
    unknown_label: int = -1,
    ignore_index: int = -100,
    eps: float = 1e-8,
) -> Tensor:
    """Sparse cell-type anchors, including explicit unknown/abstain anchors."""

    _float_tensor("type_logits", type_logits, 2)
    if labels.shape != (type_logits.shape[0],) or labels.dtype != torch.long:
        raise ValueError("labels must be long with one entry per cell")
    if labels.device != type_logits.device:
        raise ValueError("labels and logits must share a device")
    weights = _vector_weights(
        anchor_weights,
        type_logits.shape[0],
        type_logits,
        "anchor_weights",
    )
    classes = type_logits.shape[1]
    class_weight = _vector_weights(class_weights, classes, type_logits, "class_weights")
    known = (labels >= 0) & (labels != ignore_index)
    if bool((labels[known] >= classes).any()):
        raise ValueError("anchor label exceeds the number of types")
    total = type_logits.sum() * 0.0
    denominator = weights.new_zeros(())
    if bool(known.any()):
        selected = labels[known]
        losses = (
            -torch.log_softmax(type_logits[known], dim=-1)
            .gather(
                1,
                selected.unsqueeze(1),
            )
            .squeeze(1)
        )
        effective = weights[known] * class_weight[selected]
        total = total + (losses * effective).sum()
        denominator = denominator + effective.sum()
    unknown = labels == unknown_label
    if bool(unknown.any()):
        if unknown_probability is None or unknown_probability.shape != labels.shape:
            raise ValueError("unknown anchors require one unknown probability per cell")
        losses = -unknown_probability[unknown].clamp_min(eps).log()
        effective = weights[unknown]
        total = total + (losses * effective).sum()
        denominator = denominator + effective.sum()
    return total / denominator.clamp_min(eps) if bool(denominator > eps) else total


def hierarchy_consistency_loss(
    fine_probabilities: Tensor,
    parent_probabilities: Tensor,
    fine_to_parent: Sequence[int],
    eps: float = 1e-8,
) -> Tensor:
    """Match the parent head to probabilities aggregated from fine types."""

    _float_tensor("fine_probabilities", fine_probabilities, 2)
    _float_tensor("parent_probabilities", parent_probabilities, 2)
    if fine_probabilities.shape[0] != parent_probabilities.shape[0]:
        raise ValueError("fine and parent probabilities must share cells")
    mapping = torch.as_tensor(fine_to_parent, dtype=torch.long, device=fine_probabilities.device)
    if mapping.shape != (fine_probabilities.shape[1],):
        raise ValueError("fine_to_parent must contain one entry per fine type")
    if mapping.numel() and (
        bool((mapping < 0).any()) or int(mapping.max()) >= parent_probabilities.shape[1]
    ):
        raise ValueError("fine_to_parent contains an invalid parent")
    fine = fine_probabilities / fine_probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
    parent = parent_probabilities / parent_probabilities.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    aggregate = fine.new_zeros(parent.shape).index_add(1, mapping, fine)
    midpoint = 0.5 * (aggregate + parent)
    left = aggregate * (aggregate.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    right = parent * (parent.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    values = 0.5 * (left.sum(dim=-1) + right.sum(dim=-1))
    return values.mean() if values.numel() else fine.sum() * 0.0


def residual_gaussian_kl_loss(
    residual_mu: Tensor,
    residual_logvar: Tensor,
    cell_weights: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """KL regularizer for the residual posterior."""

    if residual_mu.shape != residual_logvar.shape or residual_mu.ndim != 2:
        raise ValueError("residual posterior tensors must have identical 2-D shapes")
    values = -0.5 * (1.0 + residual_logvar - residual_mu.square() - residual_logvar.exp()).mean(
        dim=-1
    )
    weights = _vector_weights(cell_weights, values.shape[0], values, "cell_weights")
    return _reduce(values, torch.ones_like(values, dtype=torch.bool), weights, eps)


def unknown_calibration_loss(
    unknown_probability: Tensor,
    unknown_targets: Tensor,
    cell_weights: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Binary cross-entropy for known-versus-unknown calibration targets."""

    if unknown_probability.shape != unknown_targets.shape or unknown_probability.ndim != 1:
        raise ValueError("unknown probabilities and targets must be matching vectors")
    if unknown_probability.device != unknown_targets.device:
        raise ValueError("unknown probabilities and targets must share a device")
    targets = unknown_targets.to(dtype=unknown_probability.dtype)
    if bool((targets < 0).any()) or bool((targets > 1).any()):
        raise ValueError("unknown_targets must be in [0, 1]")
    probabilities = unknown_probability.clamp(min=eps, max=1.0 - eps)
    values = -(targets * probabilities.log() + (1.0 - targets) * (1.0 - probabilities).log())
    weights = _vector_weights(cell_weights, values.shape[0], values, "cell_weights")
    return _reduce(values, torch.ones_like(values, dtype=torch.bool), weights, eps)


program_loss = program_score_loss
marker_loss = marker_centroid_loss
mahalanobis_residual_loss = residual_mahalanobis_loss
cycle_loss = cycle_consistency_loss
graph_boundary_loss = boundary_graph_loss
anchor_loss = anchor_classification_loss
hierarchy_loss = hierarchy_consistency_loss


__all__ = [
    "pseudobulk_loss",
    "program_score_loss",
    "program_loss",
    "marker_centroid_loss",
    "marker_loss",
    "residual_mahalanobis_loss",
    "mahalanobis_residual_loss",
    "cycle_consistency_loss",
    "cycle_loss",
    "boundary_graph_loss",
    "graph_boundary_loss",
    "anchor_classification_loss",
    "anchor_loss",
    "hierarchy_consistency_loss",
    "hierarchy_loss",
    "residual_gaussian_kl_loss",
    "unknown_calibration_loss",
]
