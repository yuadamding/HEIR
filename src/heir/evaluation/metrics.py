"""Core evaluation metrics with NaN-safe per-gene reporting."""

from typing import Dict, Optional

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)


def _probabilities(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape (items, classes>=2)")
    if np.any(probabilities < 0) or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite and non-negative")
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("every probability row needs positive mass")
    return probabilities / totals


def cell_type_metrics(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    ignore_index: int = -1,
) -> Dict[str, object]:
    """Return balanced metrics without treating cells as biological replicates."""

    truth = np.asarray(true_labels, dtype=np.int64)
    predicted_probabilities = _probabilities(probabilities)
    if truth.shape != (predicted_probabilities.shape[0],):
        raise ValueError("labels and probabilities are misaligned")
    valid = truth != ignore_index
    truth = truth[valid]
    predicted_probabilities = predicted_probabilities[valid]
    if truth.size == 0:
        raise ValueError("no evaluable labels remain")
    predicted = predicted_probabilities.argmax(axis=1)
    classes = np.arange(predicted_probabilities.shape[1])
    result: Dict[str, object] = {
        "n": int(truth.size),
        "macro_f1": float(
            f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "per_class_f1": f1_score(
            truth,
            predicted,
            labels=classes,
            average=None,
            zero_division=0,
        ).tolist(),
    }
    one_hot = np.eye(len(classes), dtype=np.float64)[truth]
    try:
        result["macro_auroc"] = float(
            roc_auc_score(one_hot, predicted_probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        result["macro_auroc"] = float("nan")
    try:
        result["macro_auprc"] = float(
            average_precision_score(one_hot, predicted_probabilities, average="macro")
        )
    except ValueError:
        result["macro_auprc"] = float("nan")
    result["brier"] = float(np.square(predicted_probabilities - one_hot).sum(axis=1).mean())
    result["ece"] = float(_ece_numpy(predicted_probabilities, truth))
    return result


def _ece_numpy(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == 0:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if np.any(mask):
            result += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(result)


def composition_metrics(predicted: np.ndarray, observed: np.ndarray) -> Dict[str, object]:
    """Compare locations-by-type compositions with compositional safeguards."""

    prediction = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(observed, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("composition matrices must have identical 2-D shapes")
    if np.any(prediction < 0) or np.any(truth < 0):
        raise ValueError("compositions cannot be negative")
    prediction = prediction / np.maximum(prediction.sum(axis=1, keepdims=True), 1.0e-12)
    truth = truth / np.maximum(truth.sum(axis=1, keepdims=True), 1.0e-12)
    pearson = [
        _safe_correlation(prediction[:, index], truth[:, index], "pearson")
        for index in range(prediction.shape[1])
    ]
    spearman = [
        _safe_correlation(prediction[:, index], truth[:, index], "spearman")
        for index in range(prediction.shape[1])
    ]
    js = np.asarray(
        [jensenshannon(left, right, base=2.0) ** 2 for left, right in zip(prediction, truth)]
    )
    eps = 1.0e-6
    prediction_clr = np.log(prediction + eps) - np.log(prediction + eps).mean(axis=1, keepdims=True)
    truth_clr = np.log(truth + eps) - np.log(truth + eps).mean(axis=1, keepdims=True)
    aitchison = np.sqrt(np.square(prediction_clr - truth_clr).sum(axis=1))
    return {
        "per_type_pearson": pearson,
        "per_type_spearman": spearman,
        "median_type_pearson": float(np.nanmedian(pearson)),
        "median_type_spearman": float(np.nanmedian(spearman)),
        "mean_js_divergence": float(js.mean()),
        "mean_aitchison_distance": float(aitchison.mean()),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray, kind: str) -> float:
    if left.size < 2 or np.isclose(np.std(left), 0) or np.isclose(np.std(right), 0):
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(left, right).statistic)
    return float(spearmanr(left, right).statistic)


def expression_metrics(predicted: np.ndarray, observed: np.ndarray) -> Dict[str, object]:
    """Report every prespecified gene, including genes with undefined correlation."""

    prediction = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(observed, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("expression matrices must have identical locations-by-genes shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("expression matrices must be finite")
    pearson = [
        _safe_correlation(prediction[:, index], truth[:, index], "pearson")
        for index in range(prediction.shape[1])
    ]
    spearman = [
        _safe_correlation(prediction[:, index], truth[:, index], "spearman")
        for index in range(prediction.shape[1])
    ]
    mse = np.square(prediction - truth).mean(axis=0)
    location_cosine = np.sum(prediction * truth, axis=1) / np.maximum(
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1),
        1.0e-12,
    )
    return {
        "per_gene_pearson": pearson,
        "per_gene_spearman": spearman,
        "per_gene_mse": mse.tolist(),
        "median_gene_pearson": float(np.nanmedian(pearson)),
        "median_gene_spearman": float(np.nanmedian(spearman)),
        "median_gene_mse": float(np.median(mse)),
        "mean_location_cosine": float(location_cosine.mean()),
        "fraction_genes_defined": float(np.isfinite(pearson).mean()),
    }


def within_type_residuals(
    expression: np.ndarray,
    labels: np.ndarray,
    sample_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Subtract each sample/type mean for the decisive residual endpoint."""

    values = np.asarray(expression, dtype=np.float64)
    types = np.asarray(labels)
    if values.ndim != 2 or types.shape != (values.shape[0],):
        raise ValueError("expression and labels are misaligned")
    samples = (
        np.zeros(values.shape[0], dtype=np.int64) if sample_ids is None else np.asarray(sample_ids)
    )
    if samples.shape != types.shape:
        raise ValueError("sample_ids must align to rows")
    residual = np.empty_like(values)
    keys = np.asarray(
        ["%s\x1f%s" % (sample, cell_type) for sample, cell_type in zip(samples, types)]
    )
    for key in np.unique(keys):
        mask = keys == key
        residual[mask] = values[mask] - values[mask].mean(axis=0, keepdims=True)
    return residual.astype(np.float32)


def risk_coverage_curve(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    uncertainty: np.ndarray,
) -> Dict[str, list]:
    """Selective error as increasingly uncertain cells are abstained."""

    truth = np.asarray(true_labels)
    predicted = np.asarray(predicted_labels)
    scores = np.asarray(uncertainty, dtype=np.float64)
    if truth.shape != predicted.shape or truth.shape != scores.shape or truth.ndim != 1:
        raise ValueError("labels and uncertainty must be aligned vectors")
    order = np.argsort(scores)
    correct = (truth[order] == predicted[order]).astype(np.float64)
    cumulative_accuracy = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    coverage = np.arange(1, len(correct) + 1) / len(correct)
    return {"coverage": coverage.tolist(), "risk": (1.0 - cumulative_accuracy).tolist()}
