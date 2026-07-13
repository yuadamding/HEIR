"""Oracle ladder that localizes molecular/spatial prediction bottlenecks."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

ORACLE_LADDER_SCHEMA = "heir.oracle_ladder.v5"
ORACLE_TRUTH_MASK_SCHEMA = "heir.oracle_truth_gene_mask.v2"
DEFAULT_TRUTH_VARIANCE_THRESHOLD = 1.0e-12


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("oracle arrays cannot use object dtype")
    if array.dtype.kind in {"U", "S"}:
        return _canonical_sha256(
            {
                "dtype": "string",
                "shape": list(array.shape),
                "values": [str(item) for item in array.reshape(-1).tolist()],
            }
        )
    if np.issubdtype(array.dtype, np.floating):
        normalized = np.ascontiguousarray(array, dtype="<f8")
        dtype = "float64-le"
    elif np.issubdtype(array.dtype, np.signedinteger):
        normalized = np.ascontiguousarray(array, dtype="<i8")
        dtype = "int64-le"
    elif np.issubdtype(array.dtype, np.unsignedinteger):
        normalized = np.ascontiguousarray(array, dtype="<u8")
        dtype = "uint64-le"
    elif np.issubdtype(array.dtype, np.bool_):
        normalized = np.ascontiguousarray(array, dtype=np.uint8)
        dtype = "bool-u8"
    else:
        raise ValueError("unsupported oracle array dtype")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": dtype, "shape": list(normalized.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _validate_sha256(value: str, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("%s must be a lowercase SHA-256 digest" % name)
    return digest


def _identities(
    values: Sequence[object],
    name: str,
    length: int,
    *,
    unique: bool,
) -> Tuple[str, ...]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape != (length,):
        raise ValueError("%s must contain one ordered identity per row" % name)
    result = tuple(str(value) for value in raw.tolist())
    if any(not value for value in result) or (unique and len(set(result)) != len(result)):
        qualifier = "non-empty and unique" if unique else "non-empty"
        raise ValueError("%s must be %s" % (name, qualifier))
    return result


def _matrix(value: np.ndarray, name: str, rows: int, columns: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (rows, columns):
        raise ValueError("%s must have shape (%d, %d)" % (name, rows, columns))
    if not np.isfinite(result).all():
        raise ValueError("%s must be finite" % name)
    return result


def _nearest_within_type(
    latent: np.ndarray,
    selected_types: np.ndarray,
    prototype_means: np.ndarray,
    prototype_types: np.ndarray,
) -> np.ndarray:
    result = np.empty(len(latent), dtype=np.int64)
    for row, selected_type in enumerate(selected_types):
        candidates = np.flatnonzero(prototype_types == selected_type)
        if not len(candidates):
            raise ValueError("oracle ladder type has no molecular prototype")
        distance = np.square(prototype_means[candidates] - latent[row]).sum(axis=1)
        result[row] = candidates[int(np.argmin(distance))]
    return result


def _correlation(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    rank: bool,
    variance_threshold: float,
) -> float:
    if float(np.var(prediction)) <= variance_threshold:
        return 0.0
    statistic = (
        spearmanr(prediction, truth).statistic if rank else pearsonr(prediction, truth).statistic
    )
    return float(statistic) if np.isfinite(statistic) else 0.0


def _score(
    prediction: np.ndarray,
    truth: np.ndarray,
    gene_mask: np.ndarray,
    *,
    variance_threshold: float,
) -> Mapping[str, object]:
    selected_prediction = prediction[:, gene_mask]
    selected_truth = truth[:, gene_mask]
    pearson = []
    spearman = []
    constant = []
    for column in range(selected_prediction.shape[1]):
        left = selected_prediction[:, column]
        right = selected_truth[:, column]
        is_constant = float(np.var(left)) <= variance_threshold
        constant.append(is_constant)
        pearson.append(
            _correlation(
                left,
                right,
                rank=False,
                variance_threshold=variance_threshold,
            )
        )
        spearman.append(
            _correlation(
                left,
                right,
                rank=True,
                variance_threshold=variance_threshold,
            )
        )
    mse = np.square(selected_prediction - selected_truth).mean(axis=0)
    location_cosine = np.sum(selected_prediction * selected_truth, axis=1) / np.maximum(
        np.linalg.norm(selected_prediction, axis=1) * np.linalg.norm(selected_truth, axis=1),
        1.0e-12,
    )
    return {
        "per_gene_pearson": pearson,
        "per_gene_spearman": spearman,
        "per_gene_mse": mse.tolist(),
        "median_gene_pearson": float(np.median(pearson)),
        "median_gene_spearman": float(np.median(spearman)),
        "median_gene_mse": float(np.median(mse)),
        "mean_location_cosine": float(location_cosine.mean()),
        "fraction_genes_defined": 1.0,
        "constant_prediction_count": int(sum(constant)),
        "constant_prediction_policy": "correlation_scored_zero",
        "cell_gene_mse": float(np.mean(np.square(selected_prediction - selected_truth))),
        "cell_gene_mae": float(np.mean(np.abs(selected_prediction - selected_truth))),
    }


def _spot_aggregation(
    expression: np.ndarray,
    spot_ids_by_cell: Sequence[str],
    cell_rna_mass: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Aggregate log1p cell expression with frozen RNA-mass weights."""

    spot_order = tuple(dict.fromkeys(str(value) for value in spot_ids_by_cell))
    if len(spot_order) < 2:
        raise ValueError("oracle spatial scoring requires at least two occupied spots")
    lookup = {spot_id: index for index, spot_id in enumerate(spot_order)}
    spot_index = np.asarray([lookup[str(value)] for value in spot_ids_by_cell], dtype=np.int64)
    mass = np.asarray(cell_rna_mass, dtype=np.float64)
    spot_mass = np.bincount(spot_index, weights=mass, minlength=len(spot_order))
    if bool((spot_mass <= 0).any()):
        raise ValueError("every oracle spot requires positive frozen RNA mass")
    linear = np.expm1(np.asarray(expression, dtype=np.float64))
    if not np.isfinite(linear).all() or bool((linear < 0).any()):
        raise ValueError("oracle expression cannot be converted to finite linear abundance")
    sums = np.zeros((len(spot_order), expression.shape[1]), dtype=np.float64)
    np.add.at(sums, spot_index, linear * mass[:, None])
    return np.log1p(sums / spot_mass[:, None]), spot_mass, spot_order


def _pseudobulk(
    spot_expression: np.ndarray,
    spot_mass: np.ndarray,
) -> np.ndarray:
    linear = np.expm1(np.asarray(spot_expression, dtype=np.float64))
    weights = np.asarray(spot_mass, dtype=np.float64)
    return np.log1p(np.average(linear, axis=0, weights=weights))


def _pseudobulk_score(
    prediction: np.ndarray,
    truth: np.ndarray,
    gene_mask: np.ndarray,
    *,
    variance_threshold: float,
) -> Mapping[str, object]:
    selected_prediction = np.asarray(prediction, dtype=np.float64)[gene_mask]
    selected_truth = np.asarray(truth, dtype=np.float64)[gene_mask]
    difference = selected_prediction - selected_truth
    denominator = max(
        float(np.linalg.norm(selected_prediction) * np.linalg.norm(selected_truth)),
        1.0e-12,
    )
    return {
        "pearson_across_genes": _correlation(
            selected_prediction,
            selected_truth,
            rank=False,
            variance_threshold=variance_threshold,
        ),
        "spearman_across_genes": _correlation(
            selected_prediction,
            selected_truth,
            rank=True,
            variance_threshold=variance_threshold,
        ),
        "cosine_across_genes": float(selected_prediction.dot(selected_truth) / denominator),
        "gene_mse": float(np.mean(np.square(difference))),
        "gene_mae": float(np.mean(np.abs(difference))),
        "per_gene_absolute_error": np.abs(difference).tolist(),
        "per_gene_squared_error": np.square(difference).tolist(),
    }


def evaluate_oracle_ladder(
    *,
    truth_expression: np.ndarray,
    truth_latent: np.ndarray,
    true_types: np.ndarray,
    decoder_expression: np.ndarray,
    type_mean_expression: np.ndarray,
    prototype_means: np.ndarray,
    prototype_expression: np.ndarray,
    prototype_types: np.ndarray,
    predicted_type_probabilities: np.ndarray,
    oracle_type_conditioned_heir_expression: np.ndarray,
    residual_disabled_heir_expression: np.ndarray,
    full_heir_expression: np.ndarray,
    cell_ids: Sequence[object],
    gene_names: Sequence[object],
    spot_ids: Sequence[object],
    cell_rna_mass: np.ndarray,
    input_artifact_sha256: str,
    decoder_checkpoint_sha256: str,
    heir_checkpoint_sha256: str,
) -> Mapping[str, object]:
    """Score oracle endpoints at cell, spot, and pseudobulk resolution.

    ``spot_ids`` is aligned to cell rows and may repeat when multiple nuclei map
    to the same spatial location. The input-artifact, decoder-checkpoint, and
    HEIR-checkpoint digests are required assertions: the standalone script
    computes all three from files, while this array-level helper records and
    validates the supplied provenance. The oracle-type-conditioned and
    residual-disabled expressions are explicit same-checkpoint forward-pass
    fixtures; the evaluator never reconstructs either from decoded prototype
    profiles.
    """

    input_digest = _validate_sha256(input_artifact_sha256, "input artifact sha256")
    checkpoint_digest = _validate_sha256(
        decoder_checkpoint_sha256,
        "decoder checkpoint sha256",
    )
    heir_checkpoint_digest = _validate_sha256(
        heir_checkpoint_sha256,
        "HEIR checkpoint sha256",
    )
    truth = np.asarray(truth_expression, dtype=np.float64)
    latent = np.asarray(truth_latent, dtype=np.float64)
    labels = np.asarray(true_types)
    prototype_latent = np.asarray(prototype_means, dtype=np.float64)
    prototype_labels = np.asarray(prototype_types)
    if (
        truth.ndim != 2
        or latent.ndim != 2
        or not truth.shape[0]
        or not truth.shape[1]
        or len(truth) != len(latent)
    ):
        raise ValueError("truth expression/latent arrays must be aligned non-empty matrices")
    cells, genes = truth.shape
    cell_identity = _identities(cell_ids, "cell_ids", cells, unique=True)
    gene_identity = _identities(gene_names, "gene_names", genes, unique=True)
    spot_identity = _identities(spot_ids, "spot_ids", cells, unique=False)
    rna_mass = np.asarray(cell_rna_mass, dtype=np.float64)
    if rna_mass.shape != (cells,) or not np.isfinite(rna_mass).all() or np.any(rna_mass <= 0):
        raise ValueError("cell_rna_mass must contain one positive finite value per cell")
    prototypes = len(prototype_latent)
    if labels.shape != (cells,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("true_types must be one integer per cell")
    if prototype_latent.ndim != 2 or not prototypes or prototype_latent.shape[1] != latent.shape[1]:
        raise ValueError("prototype_means must share the truth latent width")
    if prototype_labels.shape != (prototypes,) or not np.issubdtype(
        prototype_labels.dtype,
        np.integer,
    ):
        raise ValueError("prototype_types must contain one integer per prototype")
    if (
        not np.isfinite(truth).all()
        or not np.isfinite(latent).all()
        or not np.isfinite(prototype_latent).all()
    ):
        raise ValueError("oracle truth/prototype arrays must be finite")
    if bool((labels < 0).any()) or bool((prototype_labels < 0).any()):
        raise ValueError("oracle type indices must be non-negative")
    types = int(max(labels.max(initial=0), prototype_labels.max(initial=0))) + 1

    decoder = _matrix(decoder_expression, "decoder_expression", cells, genes)
    type_means = _matrix(type_mean_expression, "type_mean_expression", types, genes)
    prototype_profiles = _matrix(
        prototype_expression,
        "prototype_expression",
        prototypes,
        genes,
    )
    type_probabilities = _matrix(
        predicted_type_probabilities,
        "predicted_type_probabilities",
        cells,
        types,
    )
    oracle_type_conditioned = _matrix(
        oracle_type_conditioned_heir_expression,
        "oracle_type_conditioned_heir_expression",
        cells,
        genes,
    )
    residual_disabled = _matrix(
        residual_disabled_heir_expression,
        "residual_disabled_heir_expression",
        cells,
        genes,
    )
    full = _matrix(full_heir_expression, "full_heir_expression", cells, genes)
    for name, expression in (
        ("truth_expression", truth),
        ("decoder_expression", decoder),
        ("type_mean_expression", type_means),
        ("prototype_expression", prototype_profiles),
        ("oracle_type_conditioned_heir_expression", oracle_type_conditioned),
        ("residual_disabled_heir_expression", residual_disabled),
        ("full_heir_expression", full),
    ):
        if bool((expression < 0).any()):
            raise ValueError("%s must be non-negative log expression" % name)
    if bool((type_probabilities < 0).any()):
        raise ValueError("oracle ladder probabilities must be non-negative")
    if bool((type_probabilities.sum(axis=1) <= 0).any()):
        raise ValueError("oracle ladder probability rows need positive mass")

    oracle_prototype_index = _nearest_within_type(
        latent,
        labels,
        prototype_latent,
        prototype_labels,
    )
    predicted_types = type_probabilities.argmax(axis=1)
    predicted_type_oracle_state_index = _nearest_within_type(
        latent,
        predicted_types,
        prototype_latent,
        prototype_labels,
    )
    predictions: Dict[str, np.ndarray] = {
        "rna_decoder_ceiling": decoder,
        "oracle_type_mean": type_means[labels],
        "oracle_type_oracle_prototype": prototype_profiles[oracle_prototype_index],
        "predicted_type_oracle_state": prototype_profiles[predicted_type_oracle_state_index],
        "oracle_type_predicted_state": oracle_type_conditioned,
        "full_heir_residual_disabled": residual_disabled,
        "full_heir": full,
    }
    variance_threshold = DEFAULT_TRUTH_VARIANCE_THRESHOLD
    spot_truth, spot_mass, unique_spot_ids = _spot_aggregation(
        truth,
        spot_identity,
        rna_mass,
    )
    pseudobulk_truth = _pseudobulk(spot_truth, spot_mass)
    gene_mask = np.isfinite(spot_truth).all(axis=0) & (
        np.var(spot_truth, axis=0) > variance_threshold
    )
    if not bool(gene_mask.any()):
        raise ValueError("oracle ladder truth has no finite variable genes")

    arrays = {
        "truth_expression": truth,
        "truth_latent": latent,
        "true_types": labels.astype(np.int64, copy=False),
        "decoder_expression": decoder,
        "type_mean_expression": type_means,
        "prototype_means": prototype_latent,
        "prototype_expression": prototype_profiles,
        "prototype_types": prototype_labels.astype(np.int64, copy=False),
        "predicted_type_probabilities": type_probabilities,
        "oracle_type_conditioned_heir_expression": oracle_type_conditioned,
        "residual_disabled_heir_expression": residual_disabled,
        "full_heir_expression": full,
        "cell_rna_mass": rna_mass,
    }
    array_digests = {name: _array_sha256(value) for name, value in arrays.items()}
    identity_digests = {
        "ordered_cell_ids_sha256": _array_sha256(np.asarray(cell_identity)),
        "ordered_gene_names_sha256": _array_sha256(np.asarray(gene_identity)),
        "ordered_spot_ids_sha256": _array_sha256(np.asarray(spot_identity)),
        "ordered_unique_spot_ids_sha256": _array_sha256(np.asarray(unique_spot_ids)),
    }
    truth_mask_digest = _canonical_sha256(
        {
            "schema": ORACLE_TRUTH_MASK_SCHEMA,
            "policy": "finite_and_spot_truth_variance_above_threshold",
            "variance_threshold": variance_threshold,
            "gene_names": list(gene_identity),
            "mask": gene_mask.astype(int).tolist(),
            "truth_expression_sha256": array_digests["truth_expression"],
            "aggregated_truth_expression_sha256": _array_sha256(spot_truth),
            "spot_mass_sha256": _array_sha256(spot_mass),
            **identity_digests,
        }
    )
    bundle_digest = _canonical_sha256(
        {
            "schema": ORACLE_LADDER_SCHEMA,
            "input_artifact_sha256": input_digest,
            "decoder_checkpoint_sha256": checkpoint_digest,
            "heir_checkpoint_sha256": heir_checkpoint_digest,
            "array_sha256": array_digests,
            **identity_digests,
        }
    )
    endpoint_reports: Dict[str, Mapping[str, object]] = {}
    for name, prediction in predictions.items():
        spot_prediction, endpoint_spot_mass, endpoint_spot_ids = _spot_aggregation(
            prediction,
            spot_identity,
            rna_mass,
        )
        if endpoint_spot_ids != unique_spot_ids or not np.array_equal(
            endpoint_spot_mass,
            spot_mass,
        ):
            raise RuntimeError("oracle endpoints produced inconsistent spatial aggregation")
        pseudobulk_prediction = _pseudobulk(spot_prediction, endpoint_spot_mass)
        endpoint = {
            "metrics": _score(
                prediction,
                truth,
                gene_mask,
                variance_threshold=variance_threshold,
            ),
            "spot_metrics": _score(
                spot_prediction,
                spot_truth,
                gene_mask,
                variance_threshold=variance_threshold,
            ),
            "pseudobulk_metrics": _pseudobulk_score(
                pseudobulk_prediction,
                pseudobulk_truth,
                gene_mask,
                variance_threshold=variance_threshold,
            ),
            "truth_gene_mask_sha256": truth_mask_digest,
            "prediction_sha256": _array_sha256(prediction),
            "spot_prediction_sha256": _array_sha256(spot_prediction),
            "pseudobulk_prediction_sha256": _array_sha256(pseudobulk_prediction),
            "oracle_input_bundle_sha256": bundle_digest,
        }
        if name == "rna_decoder_ceiling":
            endpoint["decoder_checkpoint_sha256"] = checkpoint_digest
            endpoint["decoder_binding"] = "precomputed_output_declared_for_checkpoint_hash"
        if name in {
            "oracle_type_predicted_state",
            "full_heir_residual_disabled",
            "full_heir",
        }:
            endpoint["heir_checkpoint_sha256"] = heir_checkpoint_digest
            endpoint["heir_checkpoint_binding"] = "precomputed_output_declared_for_checkpoint_hash"
        if name == "full_heir_residual_disabled":
            endpoint["control_semantics"] = {
                "learned_morphology_residual": "disabled_during_same_checkpoint_forward",
                "construction": ("precomputed_exact_model_output_with_residual_branch_forced_off"),
                "not_reconstructed_by_evaluator": True,
            }
        if name == "oracle_type_predicted_state":
            endpoint["control_semantics"] = {
                "broad_type": "oracle_true_type_forced_during_same_checkpoint_forward",
                "within_type_state": "model_predicted",
                "construction": (
                    "precomputed_exact_same_checkpoint_oracle_type_conditioned_forward"
                ),
                "not_reconstructed_by_evaluator": True,
            }
        endpoint_reports[name] = endpoint

    return {
        "schema": ORACLE_LADDER_SCHEMA,
        "cells": cells,
        "spots": len(unique_spot_ids),
        "genes_total": genes,
        "genes_evaluated": int(gene_mask.sum()),
        "truth_gene_mask_sha256": truth_mask_digest,
        "truth_gene_mask": {
            "schema": ORACLE_TRUTH_MASK_SCHEMA,
            "sha256": truth_mask_digest,
            "policy": "finite_and_spot_truth_variance_above_threshold",
            "variance_threshold": variance_threshold,
            "selected_gene_names": [
                name for name, selected in zip(gene_identity, gene_mask) if selected
            ],
        },
        "claim_scope": {
            "purpose": "diagnostic_oracle_ladder",
            "eligible_for_primary_performance_claims": False,
            "reason": (
                "precomputed decoder, oracle-type-conditioned HEIR, residual-disabled HEIR, "
                "and full HEIR outputs are provenance-bound but not regenerated by this evaluator"
            ),
        },
        "spatial_aggregation": {
            "policy": "linear_space_rna_mass_weighted_mean_then_log1p",
            "ordered_spot_ids": list(unique_spot_ids),
            "ordered_spot_ids_sha256": identity_digests["ordered_unique_spot_ids_sha256"],
            "cell_spot_assignment_sha256": identity_digests["ordered_spot_ids_sha256"],
            "cell_rna_mass_sha256": array_digests["cell_rna_mass"],
            "spot_mass_sha256": _array_sha256(spot_mass),
            "aggregated_truth_expression_sha256": _array_sha256(spot_truth),
            "truth_pseudobulk_sha256": _array_sha256(pseudobulk_truth),
        },
        "provenance": {
            "input_artifact_sha256": input_digest,
            "decoder_checkpoint_sha256": checkpoint_digest,
            "heir_checkpoint_sha256": heir_checkpoint_digest,
            "array_sha256": array_digests,
            **identity_digests,
            "oracle_input_bundle_sha256": bundle_digest,
            "decoder_ceiling_binding": (
                "asserted checkpoint hash bound to the precomputed decoder output; "
                "this array-level evaluator does not inspect a checkpoint file or regenerate "
                "decoder inference"
            ),
            "heir_prediction_binding": (
                "asserted HEIR checkpoint hash bound to the oracle-type-conditioned, "
                "residual-disabled, and full precomputed predictions; this array-level "
                "evaluator does not inspect the checkpoint or regenerate any forward pass"
            ),
        },
        "endpoints": endpoint_reports,
    }


__all__ = ["ORACLE_LADDER_SCHEMA", "evaluate_oracle_ladder"]
