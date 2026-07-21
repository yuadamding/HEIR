#!/usr/bin/env python3
"""Validation-only audits of the frozen NatCommun v2 predictions.

This script never fits, updates, rescales, or selects a predictor.  It first
validates every target-free prediction artifact and its frozen identity.  Only
after that global preflight succeeds does it open the separated score targets
to perform the prespecified composition/state, image-correction, and molecular-
distribution audits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "4"

# isort: off
import numpy as np  # noqa: E402
# isort: on


ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path("/mnt/seagate/HEIR_runs/natcommun_generative_development")
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/natcommun_frozen_validation_diagnostics/report.json")
RUNNER_PATH = ROOT / "scripts/benchmark_natcommun_generative_development.py"
CORE_PATH = ROOT / "src/heir/evaluation/generative_fusion.py"
VALIDATION_PROTOCOL_PATH = ROOT / "configs/natcommun_frozen_validation_protocol.json"
EXPECTED = {
    "runner_sha256": "cf27504e25dfd8cd7e8bfe2894efc8b4a8f79306b47bc492d0e61406d20668ce",
    "core_sha256": "55a63f1360e8cc76267e4b00ba8e2167f36259789e9bfdf2aa929c8cadd83b17",
    "protocol_sha256": "2cb92b22b6870488a06e64b213e37ffbbdfe3044f1da8fc7442f506915e78197",
    "fit_predict_manifest_sha256": (
        "cb7ebdf9e22090a046937204993a7b2aa3ac1ba2d4c434883e43ed45d1e826ca"
    ),
    "prepared_manifest_sha256": (
        "d1b1353abb9ee80c3132fa08a9ebaea3aeeee607aaa11f2c90e7121be02addde"
    ),
    "baseline_report_sha256": "bf3144cf22405752488509dbb1a65b573967fe1b14110881020787187828cf29",
}
ARMS = ("M0", "M1", "M2", "M3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(np.asarray(array))
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("heir_frozen_natcommun_v2", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen NatCommun runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or first.size < 2:
        return math.nan
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.sqrt(np.sum(first**2) * np.sum(second**2))
    return math.nan if denominator <= 0 else float(np.sum(first * second) / denominator)


def _section_alignment(
    correction: np.ndarray,
    residual: np.ndarray,
    sections: np.ndarray,
) -> Mapping[str, object]:
    records = []
    section_ids = np.asarray(sections).astype(str)
    for section in sorted(set(section_ids.tolist())):
        keep = section_ids == section
        delta = np.asarray(correction[keep], dtype=np.float64).reshape(-1)
        error = np.asarray(residual[keep], dtype=np.float64).reshape(-1)
        delta_norm2 = float(np.dot(delta, delta))
        error_norm2 = float(np.dot(error, error))
        dot = float(np.dot(delta, error))
        denominator = math.sqrt(max(delta_norm2 * error_norm2, 0.0))
        records.append(
            {
                "section": section,
                "spots": int(np.count_nonzero(keep)),
                "dot_per_element": dot / max(len(delta), 1),
                "cosine": dot / denominator if denominator > 0 else math.nan,
                "correlation": _correlation(delta, error),
                "optimal_correction_scale": dot / delta_norm2 if delta_norm2 > 0 else math.nan,
                "correction_to_residual_norm": (
                    math.sqrt(delta_norm2 / error_norm2) if error_norm2 > 0 else math.nan
                ),
                "mse_change_after_correction": (
                    float(np.mean((error - delta) ** 2) - np.mean(error**2))
                ),
                "positive_inner_product": dot > 0,
            }
        )
    fields = (
        "dot_per_element",
        "cosine",
        "correlation",
        "optimal_correction_scale",
        "correction_to_residual_norm",
        "mse_change_after_correction",
    )
    return {
        "sections": records,
        "section_balanced": {
            field: _mean([float(record[field]) for record in records]) for field in fields
        },
        "positive_inner_product_section_fraction": float(
            np.mean([bool(record["positive_inner_product"]) for record in records])
        ),
    }


def _section_weighted_count_alignment(
    correction: np.ndarray,
    residual: np.ndarray,
    predictive_variance: np.ndarray,
    latent_variance: np.ndarray,
    sections: np.ndarray,
) -> Mapping[str, object]:
    """Count-scale correction audit weighted by frozen M1 uncertainty."""

    records = []
    section_ids = np.asarray(sections).astype(str)
    for section in sorted(set(section_ids.tolist())):
        keep = section_ids == section
        delta = np.asarray(correction[keep], dtype=np.float64).reshape(-1)
        error = np.asarray(residual[keep], dtype=np.float64).reshape(-1)
        variance = np.maximum(
            np.asarray(predictive_variance[keep], dtype=np.float64).reshape(-1), 1.0e-8
        )
        latent = np.maximum(np.asarray(latent_variance[keep], dtype=np.float64).reshape(-1), 1.0e-8)
        weighted_dot = float(np.mean(delta * error / variance))
        delta_energy = float(np.sum(delta**2 / variance))
        residual_energy = float(np.sum(error**2 / variance))
        denominator = math.sqrt(max(delta_energy * residual_energy, 0.0))
        records.append(
            {
                "section": section,
                "spots": int(np.count_nonzero(keep)),
                "raw_inner_product_per_element": float(np.mean(delta * error)),
                "predictive_variance_weighted_inner_product": weighted_dot,
                "predictive_variance_weighted_cosine": (
                    float(np.sum(delta * error / variance)) / denominator
                    if denominator > 0
                    else math.nan
                ),
                "centered_count_correlation": _correlation(delta, error),
                "optimal_correction_scale": (
                    float(np.sum(delta * error / variance)) / delta_energy
                    if delta_energy > 0
                    else math.nan
                ),
                "correction_RMS_to_predictive_SD": float(
                    np.sqrt(np.mean(delta**2) / np.mean(variance))
                ),
                "correction_RMS_to_latent_SD": float(np.sqrt(np.mean(delta**2) / np.mean(latent))),
                "positive_weighted_inner_product": weighted_dot > 0,
            }
        )
    fields = (
        "raw_inner_product_per_element",
        "predictive_variance_weighted_inner_product",
        "predictive_variance_weighted_cosine",
        "centered_count_correlation",
        "optimal_correction_scale",
        "correction_RMS_to_predictive_SD",
        "correction_RMS_to_latent_SD",
    )
    return {
        "sections": records,
        "section_balanced": {
            field: _mean([float(record[field]) for record in records]) for field in fields
        },
        "positive_weighted_inner_product_section_fraction": float(
            np.mean([bool(record["positive_weighted_inner_product"]) for record in records])
        ),
    }


def _weighted_section_effect(
    values: np.ndarray,
    sections: np.ndarray,
    weights: np.ndarray | None = None,
    minimum_effective_rows: float = 3.0,
    minimum_total_weight: float = 0.0,
) -> Mapping[str, object]:
    effects: list[float] = []
    section_records = []
    section_ids = np.asarray(sections).astype(str)
    value_array = np.asarray(values, dtype=np.float64)
    weight_array = (
        np.ones(len(value_array), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    for section in sorted(set(section_ids.tolist())):
        keep = (section_ids == section) & np.isfinite(value_array) & np.isfinite(weight_array)
        local_weight = np.maximum(weight_array[keep], 0.0)
        total_weight = float(local_weight.sum())
        if not len(local_weight) or total_weight <= 0 or total_weight < minimum_total_weight:
            continue
        effective = float(total_weight**2 / np.sum(local_weight**2))
        if effective < minimum_effective_rows:
            continue
        effect = float(np.average(value_array[keep], weights=local_weight))
        effects.append(effect)
        section_records.append(
            {
                "section": section,
                "rows": int(np.count_nonzero(keep)),
                "total_weight": total_weight,
                "effective_rows": effective,
                "mean_effect": effect,
            }
        )
    return {
        "evaluable": bool(effects),
        "mean_effect": float(np.mean(effects)) if effects else None,
        "sections": section_records,
    }


def _reliable_variance_ratio(
    prediction_rate: np.ndarray,
    half_a: np.ndarray,
    half_b: np.ndarray,
    library_a: np.ndarray,
    library_b: np.ndarray,
    sections: np.ndarray,
    reliable_gene: np.ndarray,
    *,
    scale: float,
) -> Mapping[str, object]:
    predicted = np.log1p(np.asarray(prediction_rate, dtype=np.float64) * scale)
    first = np.log1p(
        np.asarray(half_a, dtype=np.float64)
        * (scale / np.maximum(np.asarray(library_a, dtype=np.float64), 1.0)[:, None])
    )
    second = np.log1p(
        np.asarray(half_b, dtype=np.float64)
        * (scale / np.maximum(np.asarray(library_b, dtype=np.float64), 1.0)[:, None])
    )
    ratios = []
    section_ids = np.asarray(sections).astype(str)
    training_reliable = np.asarray(reliable_gene, dtype=bool)
    for section in sorted(set(section_ids.tolist())):
        keep = section_ids == section
        if int(np.count_nonzero(keep)) < 3:
            continue
        covariance = np.mean(
            (first[keep] - first[keep].mean(axis=0)) * (second[keep] - second[keep].mean(axis=0)),
            axis=0,
        )
        variance = np.var(predicted[keep], axis=0, ddof=1)
        valid = training_reliable & np.isfinite(covariance) & (covariance > 0)
        ratios.extend((variance[valid] / covariance[valid]).tolist())
    array = np.asarray(ratios, dtype=np.float64)
    return {
        "median": float(np.median(array)) if len(array) else None,
        "q10": float(np.quantile(array, 0.10)) if len(array) else None,
        "q90": float(np.quantile(array, 0.90)) if len(array) else None,
        "strata": int(len(array)),
    }


def _dynamic_ranges(
    observed: np.ndarray,
    predicted: np.ndarray,
    sections: np.ndarray,
) -> np.ndarray:
    ratios = []
    section_ids = np.asarray(sections).astype(str)
    for section in sorted(set(section_ids.tolist())):
        keep = section_ids == section
        truth_range = np.quantile(observed[keep], 0.95, axis=0) - np.quantile(
            observed[keep], 0.05, axis=0
        )
        prediction_range = np.quantile(predicted[keep], 0.95, axis=0) - np.quantile(
            predicted[keep], 0.05, axis=0
        )
        ratios.append(
            np.divide(
                prediction_range,
                truth_range,
                out=np.full_like(prediction_range, np.nan),
                where=truth_range > 1.0e-8,
            )
        )
    return np.nanmean(np.vstack(ratios), axis=0)


def _covariance_audit(
    observed: np.ndarray,
    predicted: np.ndarray,
    half_a: np.ndarray,
    half_b: np.ndarray,
    sections: np.ndarray,
) -> Mapping[str, object]:
    relative_error, off_diagonal_correlation, noise_aware_error = [], [], []
    section_ids = np.asarray(sections).astype(str)
    for section in sorted(set(section_ids.tolist())):
        keep = section_ids == section
        if int(np.count_nonzero(keep)) < 3:
            continue
        truth = np.cov(observed[keep], rowvar=False)
        estimate = np.cov(predicted[keep], rowvar=False)
        cross = (
            (half_a[keep] - half_a[keep].mean(axis=0)).T
            @ (half_b[keep] - half_b[keep].mean(axis=0))
        ) / max(int(np.count_nonzero(keep)) - 1, 1)
        cross = 0.5 * (cross + cross.T)
        relative_error.append(
            float(np.linalg.norm(estimate - truth) / max(np.linalg.norm(truth), 1.0e-8))
        )
        noise_aware_error.append(
            float(np.linalg.norm(estimate - cross) / max(np.linalg.norm(cross), 1.0e-8))
        )
        off = ~np.eye(truth.shape[0], dtype=bool)
        off_diagonal_correlation.append(_correlation(estimate[off], truth[off]))
    return {
        "full_target_relative_Frobenius_error": _mean(relative_error),
        "full_target_off_diagonal_correlation": _mean(off_diagonal_correlation),
        "split_cross_covariance_relative_Frobenius_error": _mean(noise_aware_error),
        "sections_evaluable": len(relative_error),
    }


def _comparison_summary(runner: Any, effects: Mapping[str, float], *, seed: int) -> object:
    ordered = np.asarray([effects[key] for key in sorted(effects)], dtype=np.float64)
    if len(ordered) < 2 or not np.isfinite(ordered).all():
        return {"evaluable": False, "donor_effects": effects}
    return {
        "evaluable": True,
        "donor_effects": effects,
        "mean_effect": float(np.mean(ordered)),
        "median_effect": float(np.median(ordered)),
        "positive_donor_fraction": float(np.mean(ordered > 0)),
        "paired_bootstrap_interval": runner._donor_bootstrap_interval(ordered, seed=seed),
        "exact_one_sided_sign_flip": runner._sign_flip(runner._import_core(), ordered),
    }


def _validate_score_target_identity(
    runner: Any,
    secret: Mapping[str, np.ndarray],
    public: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    donor: str,
) -> None:
    """Fail closed on target identity immediately after the target opens."""

    expected_rows = len(np.asarray(public["query_spot_ids"]))
    expected_genes = len(np.asarray(public["gene_ids"]))
    scalar_identities = {
        "schema": runner.PREPARED_SCHEMA,
        "heldout_donor": donor,
    }
    for field, expected in scalar_identities.items():
        if field not in secret or runner._scalar_text(secret[field]) != expected:
            raise ValueError(f"score target {field} identity differs for {donor}")

    exact_rows = {
        "heldout_spot_ids": "query_spot_ids",
        "heldout_section_ids": "query_section_ids",
        "heldout_indication_ids": "query_indication_ids",
    }
    for secret_field, public_field in exact_rows.items():
        if not np.array_equal(
            np.asarray(secret[secret_field]).astype(str),
            np.asarray(public[public_field]).astype(str),
        ):
            raise ValueError(f"score target {secret_field} differs for {donor}")
    if not np.array_equal(
        np.asarray(predictions["query_spot_ids"]).astype(str),
        np.asarray(secret["heldout_spot_ids"]).astype(str),
    ):
        raise ValueError(f"prediction rows differ from the score target for {donor}")
    if not np.array_equal(
        np.asarray(predictions["gene_ids"]).astype(str),
        np.asarray(public["gene_ids"]).astype(str),
    ):
        raise ValueError(f"prediction genes differ from the public fold for {donor}")

    count_fields = ("heldout_st_counts", "heldout_st_half_a", "heldout_st_half_b")
    for field in count_fields:
        value = np.asarray(secret[field])
        if value.shape != (expected_rows, expected_genes):
            raise ValueError(f"score target {field} shape differs for {donor}")
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"score target {field} values are invalid for {donor}")
    row_fields = (
        "heldout_st_library",
        "heldout_st_library_half_a",
        "heldout_st_library_half_b",
        "primary_score_eligible",
    )
    for field in row_fields:
        value = np.asarray(secret[field])
        if value.shape != (expected_rows,):
            raise ValueError(f"score target {field} shape differs for {donor}")
        if field != "primary_score_eligible" and (
            not np.isfinite(value).all() or np.any(value < 0)
        ):
            raise ValueError(f"score target {field} values are invalid for {donor}")
    zero_depth = np.asarray(secret.get("zero_depth_excluded_count"))
    if zero_depth.shape != () or not np.isfinite(zero_depth):
        raise ValueError(f"score target zero-depth receipt is invalid for {donor}")

    full = np.asarray(secret["heldout_st_counts"])
    half_a = np.asarray(secret["heldout_st_half_a"])
    half_b = np.asarray(secret["heldout_st_half_b"])
    library = np.asarray(secret["heldout_st_library"])
    library_a = np.asarray(secret["heldout_st_library_half_a"])
    library_b = np.asarray(secret["heldout_st_library_half_b"])
    eligible = np.asarray(secret["primary_score_eligible"], dtype=bool)
    if not np.array_equal(full, half_a + half_b):
        raise ValueError(f"score target count halves do not reconstruct for {donor}")
    if not np.array_equal(library, library_a + library_b):
        raise ValueError(f"score target library halves do not reconstruct for {donor}")
    if not np.array_equal(eligible, library > 0):
        raise ValueError(f"score target eligibility differs from the frozen policy for {donor}")
    if int(zero_depth) != int(np.count_nonzero(~eligible)):
        raise ValueError(f"score target zero-depth count differs for {donor}")
    if np.any(full.sum(axis=1) > library):
        raise ValueError(f"score target panel counts exceed full exposure for {donor}")
    if np.any(half_a.sum(axis=1) > library_a) or np.any(half_b.sum(axis=1) > library_b):
        raise ValueError(f"score target panel half counts exceed half exposure for {donor}")


def _verify_identities(baseline: Path) -> Mapping[str, str]:
    observed = {
        "runner_sha256": _sha256(RUNNER_PATH),
        "core_sha256": _sha256(CORE_PATH),
        "prepared_manifest_sha256": _sha256(baseline / "prepared_manifest.json"),
        "fit_predict_manifest_sha256": _sha256(baseline / "fit_predict_manifest.json"),
        "baseline_report_sha256": _sha256(baseline / "report.json"),
    }
    mismatched = {
        key: {"expected": EXPECTED[key], "observed": value}
        for key, value in observed.items()
        if value != EXPECTED[key]
    }
    if mismatched:
        raise ValueError(f"frozen baseline identity mismatch: {mismatched}")
    return observed


def run(args: argparse.Namespace) -> Mapping[str, object]:
    if args.threads < 1 or args.threads > 4:
        raise ValueError("threads must remain between 1 and 4")
    identities = _verify_identities(args.baseline)
    validation_protocol = json.loads(VALIDATION_PROTOCOL_PATH.read_text(encoding="utf-8"))
    validator_sha256 = _sha256(Path(__file__).resolve())
    frozen_protocol_artifacts = validation_protocol.get("frozen_artifacts", {})
    required_protocol_bindings = {
        "development_runner_sha256": EXPECTED["runner_sha256"],
        "generative_core_sha256": EXPECTED["core_sha256"],
        "development_protocol_sha256": EXPECTED["protocol_sha256"],
        "prepared_manifest_sha256": EXPECTED["prepared_manifest_sha256"],
        "fit_predict_manifest_sha256": EXPECTED["fit_predict_manifest_sha256"],
        "development_report_sha256": EXPECTED["baseline_report_sha256"],
    }
    if (
        validation_protocol.get("schema") != "heir.natcommun_frozen_validation_protocol.v1"
        or frozen_protocol_artifacts.get("frozen_prediction_validator_sha256")
        != validator_sha256
        or any(
            frozen_protocol_artifacts.get(field) != expected
            for field, expected in required_protocol_bindings.items()
        )
    ):
        raise ValueError("validation protocol does not bind this validator")
    identities = {
        **identities,
        "frozen_prediction_validator_sha256": validator_sha256,
        "validation_protocol_sha256": _sha256(VALIDATION_PROTOCOL_PATH),
    }
    runner = _load_runner()
    core = runner._import_core()
    prepared = json.loads((args.baseline / "prepared_manifest.json").read_text(encoding="utf-8"))
    predictions_manifest = json.loads(
        (args.baseline / "fit_predict_manifest.json").read_text(encoding="utf-8")
    )
    baseline_report = json.loads((args.baseline / "report.json").read_text(encoding="utf-8"))
    if (
        prepared.get("protocol_sha256") != EXPECTED["protocol_sha256"]
        or baseline_report.get("schema") != "heir.natcommun_generative_development_report.v2"
        or baseline_report.get("uni2_h_run") is not False
    ):
        raise ValueError("baseline protocol/report identity is not the frozen H-optimus-1 v2 run")

    # This is the same target-free global preflight used by the primary scorer.
    preflight_args = argparse.Namespace(
        output=args.baseline,
        epochs=80,
        latent_dim=20,
        batch_size=256,
        device="cuda:0",
    )
    runner._validate_prediction_manifest_binding(
        preflight_args, core, prepared, predictions_manifest
    )

    # Freeze every target-free subgroup definition before opening any score
    # target. The scoring pass recomputes and verifies these hashes per donor.
    subgroup_mask_receipts: dict[str, Mapping[str, object]] = {}
    for donor_value in prepared["donors"]:
        donor = str(donor_value)
        fold = prepared["folds"][donor]
        public = runner._verify_semantic_file(
            Path(str(fold["public_path"])), str(fold["public_semantic_sha256"])
        )
        receipt = predictions_manifest["folds"][donor]
        predictions = runner._verify_semantic_file(
            Path(str(receipt["prediction_path"])),
            str(receipt["prediction_semantic_sha256"]),
        )
        runner._validate_prediction_artifact(predictions, public, donor=donor, epochs=80)
        eligible = np.asarray(predictions["gate3_supported_score_eligible"], dtype=bool)
        coverage = np.asarray(predictions["query_reference_coverage_mass"], dtype=np.float32)
        composition = np.asarray(predictions["query_H_composition"], dtype=np.float32)
        supported = np.asarray(predictions["gate3_supported_type_mask"], dtype=bool)
        supported_composition = composition * supported[None]
        supported_mass = supported_composition.sum(axis=1)
        supported_composition = np.divide(
            supported_composition,
            supported_mass[:, None],
            out=np.zeros_like(supported_composition),
            where=supported_mass[:, None] > 0,
        )
        subgroup_mask_receipts[donor] = {
            "rows": len(eligible),
            "gate3_eligible_sha256": _array_digest(eligible),
            "coverage_high_sha256": _array_digest(eligible & (coverage >= 0.90)),
            "coverage_low_sha256": _array_digest(eligible & (coverage < 0.90)),
            "supported_composition_weights_sha256": _array_digest(supported_composition),
            "observed_target_used": False,
        }

    fold_payload: dict[str, object] = {}
    state_donor_effects: dict[str, float] = {}
    state_by_indication: dict[str, dict[str, float]] = {}
    state_by_type: dict[str, dict[str, float]] = {}
    state_by_coverage: dict[str, dict[str, float]] = {"high": {}, "low": {}}
    correction_by_donor: dict[str, Mapping[str, object]] = {}
    shuffled_by_donor: dict[str, Mapping[str, object]] = {}
    program_by_donor: dict[str, dict[str, Mapping[str, object]]] = {}
    distribution_by_donor: dict[str, dict[str, Mapping[str, object]]] = {}
    dynamic_by_arm_gene: dict[str, dict[str, list[float]]] = {arm: {} for arm in ARMS}

    for donor in prepared["donors"]:
        donor = str(donor)
        fold = prepared["folds"][donor]
        public = runner._verify_semantic_file(
            Path(str(fold["public_path"])), str(fold["public_semantic_sha256"])
        )
        prediction_receipt = predictions_manifest["folds"][donor]
        predictions = runner._verify_semantic_file(
            Path(str(prediction_receipt["prediction_path"])),
            str(prediction_receipt["prediction_semantic_sha256"]),
        )
        # Immediate target-free revalidation before this donor's target opens.
        runner._validate_prediction_artifact(predictions, public, donor=donor, epochs=80)
        eligible_pre = np.asarray(predictions["gate3_supported_score_eligible"], dtype=bool)
        coverage_pre = np.asarray(predictions["query_reference_coverage_mass"], dtype=np.float32)
        composition_pre = np.asarray(predictions["query_H_composition"], dtype=np.float32)
        supported_pre = np.asarray(predictions["gate3_supported_type_mask"], dtype=bool)
        supported_composition_pre = composition_pre * supported_pre[None]
        supported_mass_pre = supported_composition_pre.sum(axis=1)
        supported_composition_pre = np.divide(
            supported_composition_pre,
            supported_mass_pre[:, None],
            out=np.zeros_like(supported_composition_pre),
            where=supported_mass_pre[:, None] > 0,
        )
        observed_mask_receipt = {
            "rows": len(eligible_pre),
            "gate3_eligible_sha256": _array_digest(eligible_pre),
            "coverage_high_sha256": _array_digest(eligible_pre & (coverage_pre >= 0.90)),
            "coverage_low_sha256": _array_digest(eligible_pre & (coverage_pre < 0.90)),
            "supported_composition_weights_sha256": _array_digest(supported_composition_pre),
            "observed_target_used": False,
        }
        if observed_mask_receipt != subgroup_mask_receipts[donor]:
            raise ValueError(f"target-free subgroup receipt changed for {donor}")
        secret = runner._verify_semantic_file(
            Path(str(fold["score_target_path"])), str(fold["score_target_semantic_sha256"])
        )
        _validate_score_target_identity(
            runner,
            secret,
            public,
            predictions,
            donor=donor,
        )
        keep = np.asarray(secret["primary_score_eligible"], dtype=bool)
        counts = np.asarray(secret["heldout_st_counts"], dtype=np.float64)[keep]
        library = np.asarray(secret["heldout_st_library"], dtype=np.float64)[keep]
        sections = np.asarray(secret["heldout_section_ids"]).astype(str)[keep]
        query_indications = np.unique(np.asarray(public["query_indication_ids"]).astype(str))
        if len(query_indications) != 1:
            raise ValueError(f"query indication is not unique for {donor}")
        indication = str(query_indications[0])
        theta = np.asarray(predictions["training_only_dispersion"], dtype=np.float64)
        scale = float(predictions["diagnostic_normalization_scale"])
        observed = np.log1p(counts * (scale / library[:, None]))
        rates = {
            arm: np.asarray(predictions[f"rate_{arm}"], dtype=np.float64)[keep] for arm in ARMS
        }
        normalized = {arm: np.log1p(rates[arm] * scale) for arm in ARMS}

        correction = normalized["M3"] - normalized["M1"]
        shuffled_correction = (
            np.log1p(np.asarray(predictions["rate_M4"], dtype=np.float64)[keep] * scale)
            - normalized["M1"]
        )
        residual = observed - normalized["M1"]
        log_correction_result = dict(_section_alignment(correction, residual, sections))
        log_shuffled_result = dict(_section_alignment(shuffled_correction, residual, sections))
        m1_variance = np.asarray(predictions["posterior_rate_variance_M1"], dtype=np.float64)[keep]
        mean_m1_counts = rates["M1"] * library[:, None]
        mean_m3_counts = rates["M3"] * library[:, None]
        mean_m4_counts = (
            np.asarray(predictions["rate_M4"], dtype=np.float64)[keep] * library[:, None]
        )
        latent_count_variance = m1_variance * library[:, None] ** 2
        predictive_count_variance = (
            mean_m1_counts
            + mean_m1_counts**2 / theta[None]
            + latent_count_variance * (1.0 + 1.0 / theta[None])
        )
        count_residual = counts - mean_m1_counts
        correction_result = dict(
            _section_weighted_count_alignment(
                mean_m3_counts - mean_m1_counts,
                count_residual,
                predictive_count_variance,
                latent_count_variance,
                sections,
            )
        )
        shuffled_result = dict(
            _section_weighted_count_alignment(
                mean_m4_counts - mean_m1_counts,
                count_residual,
                predictive_count_variance,
                latent_count_variance,
                sections,
            )
        )
        correction_result["log_normalized_secondary"] = log_correction_result
        shuffled_result["log_normalized_secondary"] = log_shuffled_result
        correction_by_donor[donor] = correction_result
        shuffled_by_donor[donor] = shuffled_result

        membership = np.asarray(predictions["diagnostic_program_membership"], dtype=bool)
        active = np.asarray(predictions["diagnostic_program_active"], dtype=bool)
        names = np.asarray(predictions["diagnostic_program_names"]).astype(str)
        donor_programs = {}
        for index in np.flatnonzero(active):
            donor_programs[names[index]] = _section_alignment(
                correction[:, membership[index]].mean(axis=1, keepdims=True),
                residual[:, membership[index]].mean(axis=1, keepdims=True),
                sections,
            )
        program_by_donor[donor] = donor_programs

        eligible = np.asarray(predictions["gate3_supported_score_eligible"], dtype=bool)[keep]
        conditional_rates = {
            arm: np.asarray(predictions[f"rate_{arm}_supported"], dtype=np.float64)[keep]
            for arm in ("M2", "M3")
        }
        mean_m2 = conditional_rates["M2"] * library[:, None]
        mean_m3 = conditional_rates["M3"] * library[:, None]
        row_effect = runner._nb_deviance_rows(
            core, counts, mean_m2, theta
        ) - runner._nb_deviance_rows(core, counts, mean_m3, theta)
        overall_state = _weighted_section_effect(row_effect[eligible], sections[eligible])
        if overall_state["evaluable"]:
            state_donor_effects[donor] = float(overall_state["mean_effect"])
            state_by_indication.setdefault(indication, {})[donor] = float(
                overall_state["mean_effect"]
            )

        composition = np.asarray(supported_composition_pre, dtype=np.float64)[keep]
        type_names = np.asarray(predictions["reference_model_type_names"]).astype(str)
        supported_types = np.asarray(predictions["gate3_supported_type_mask"], dtype=bool)
        for index, type_name in enumerate(type_names):
            if not supported_types[index]:
                continue
            result = _weighted_section_effect(
                row_effect[eligible],
                sections[eligible],
                weights=composition[eligible, index],
                minimum_total_weight=3.0,
            )
            if result["evaluable"]:
                state_by_type.setdefault(type_name, {})[donor] = float(result["mean_effect"])

        coverage = np.asarray(predictions["query_reference_coverage_mass"], dtype=np.float64)[keep]
        for label, mask in (
            ("high", eligible & (coverage >= 0.90)),
            ("low", eligible & (coverage < 0.90)),
        ):
            result = _weighted_section_effect(row_effect[mask], sections[mask])
            if result["evaluable"]:
                state_by_coverage[label][donor] = float(result["mean_effect"])

        reliable = np.asarray(predictions["diagnostic_training_reliable_gene"], dtype=bool)
        half_a = np.asarray(secret["heldout_st_half_a"], dtype=np.float64)[keep][eligible]
        half_b = np.asarray(secret["heldout_st_half_b"], dtype=np.float64)[keep][eligible]
        library_a = np.asarray(secret["heldout_st_library_half_a"], dtype=np.float64)[keep][
            eligible
        ]
        library_b = np.asarray(secret["heldout_st_library_half_b"], dtype=np.float64)[keep][
            eligible
        ]
        supported_variance_rates = {
            "M0_same_rows": rates["M0"][eligible],
            "M2_supported": conditional_rates["M2"][eligible],
            "M3_supported": conditional_rates["M3"][eligible],
        }
        supported_variance = {
            arm: _reliable_variance_ratio(
                arm_rate,
                half_a,
                half_b,
                library_a,
                library_b,
                sections[eligible],
                reliable,
                scale=scale,
            )
            for arm, arm_rate in supported_variance_rates.items()
        }

        donor_distribution = {}
        full_half_a_library = np.asarray(secret["heldout_st_library_half_a"], dtype=np.float64)[
            keep
        ]
        full_half_b_library = np.asarray(secret["heldout_st_library_half_b"], dtype=np.float64)[
            keep
        ]
        full_half_a = np.log1p(
            np.asarray(secret["heldout_st_half_a"], dtype=np.float64)[keep]
            * (scale / np.maximum(full_half_a_library, 1.0)[:, None])
        )
        full_half_b = np.log1p(
            np.asarray(secret["heldout_st_half_b"], dtype=np.float64)[keep]
            * (scale / np.maximum(full_half_b_library, 1.0)[:, None])
        )
        for arm in ARMS:
            quality = runner._quality_metrics(
                core,
                counts,
                library,
                rates[arm],
                np.asarray(predictions[f"posterior_rate_variance_{arm}"], dtype=np.float64)[keep],
                theta,
                sections,
                secret,
                predictions,
            )
            covariance = _covariance_audit(
                observed, normalized[arm], full_half_a, full_half_b, sections
            )
            dynamic = _dynamic_ranges(observed, normalized[arm], sections)
            for gene, value in zip(np.asarray(predictions["gene_ids"]).astype(str), dynamic):
                dynamic_by_arm_gene[arm].setdefault(gene, []).append(float(value))
            donor_distribution[arm] = {
                **quality,
                "gene_covariance": covariance,
                "calibration_interpretable": bool(
                    np.mean(np.var(normalized[arm], axis=0)) > 1.0e-8
                ),
                "dynamic_range_ratio_median_gene": float(np.nanmedian(dynamic)),
                "dynamic_range_ratio_q10_gene": float(np.nanquantile(dynamic, 0.10)),
                "dynamic_range_ratio_q90_gene": float(np.nanquantile(dynamic, 0.90)),
                "genes_with_at_least_0.8_dynamic_range_fraction": float(
                    np.nanmean(dynamic >= 0.80)
                ),
            }
        distribution_by_donor[donor] = donor_distribution
        fold_payload[donor] = {
            "indication": indication,
            "scored_spots": int(np.count_nonzero(keep)),
            "scored_sections": sorted(set(sections.tolist())),
            "gate3_eligible_spots": int(np.count_nonzero(eligible)),
            "gate3_eligible_fraction": float(np.mean(eligible)),
            "state_effect": overall_state,
            "supported_reliability_adjusted_variance": supported_variance,
            "image_correction": correction_result,
            "shuffled_image_correction": shuffled_result,
            "program_correction": donor_programs,
            "molecular_distribution": donor_distribution,
        }

    state_overall = _comparison_summary(runner, state_donor_effects, seed=1729 + 101)
    indication_summary = {
        indication: _comparison_summary(runner, effects, seed=1729 + 200 + index)
        for index, (indication, effects) in enumerate(sorted(state_by_indication.items()))
    }
    type_summary = {
        type_name: _comparison_summary(runner, effects, seed=1729 + 300 + index)
        for index, (type_name, effects) in enumerate(sorted(state_by_type.items()))
    }
    coverage_summary = {
        label: _comparison_summary(runner, effects, seed=1729 + 400 + index)
        for index, (label, effects) in enumerate(sorted(state_by_coverage.items()))
    }

    def add_holm(summaries: Mapping[str, object]) -> None:
        names, pvalues = [], []
        for name, summary in summaries.items():
            if not isinstance(summary, Mapping) or summary.get("evaluable") is not True:
                continue
            test = summary.get("exact_one_sided_sign_flip")
            if isinstance(test, Mapping) and test.get("p_value") is not None:
                names.append(name)
                pvalues.append(float(test["p_value"]))
        if not names:
            return
        adjusted = core.holm_adjust(pvalues)
        for name, value in zip(names, adjusted):
            summaries[name]["holm_adjusted_p_value"] = float(value)  # type: ignore[index]

    add_holm(type_summary)
    add_holm(coverage_summary)

    correction_indications: dict[str, list[str]] = {}
    for donor, fold in fold_payload.items():
        correction_indications.setdefault(str(fold["indication"]), []).append(donor)

    def aggregate_alignment(
        source: Mapping[str, Mapping[str, object]], donors: Sequence[str]
    ) -> Mapping[str, object]:
        fields = (
            "raw_inner_product_per_element",
            "predictive_variance_weighted_inner_product",
            "predictive_variance_weighted_cosine",
            "centered_count_correlation",
            "optimal_correction_scale",
            "correction_RMS_to_predictive_SD",
            "correction_RMS_to_latent_SD",
        )
        return {
            field: _mean([float(source[donor]["section_balanced"][field]) for donor in donors])
            for field in fields
        } | {
            "positive_weighted_inner_product_donor_fraction": float(
                np.mean(
                    [
                        float(
                            source[donor]["section_balanced"][
                                "predictive_variance_weighted_inner_product"
                            ]
                        )
                        > 0
                        for donor in donors
                    ]
                )
            ),
            "donors": list(donors),
        }

    all_donors = sorted(correction_by_donor)
    correction_aggregate = aggregate_alignment(correction_by_donor, all_donors)
    shuffled_aggregate = aggregate_alignment(shuffled_by_donor, all_donors)
    pairing_effects = {
        donor: float(
            correction_by_donor[donor]["section_balanced"][
                "predictive_variance_weighted_inner_product"
            ]
            - shuffled_by_donor[donor]["section_balanced"][
                "predictive_variance_weighted_inner_product"
            ]
        )
        for donor in all_donors
    }
    pairing_comparison = _comparison_summary(runner, pairing_effects, seed=1729 + 501)
    correction_by_indication = {
        indication: aggregate_alignment(correction_by_donor, sorted(donors))
        for indication, donors in sorted(correction_indications.items())
    }

    def aggregate_log_alignment(
        source: Mapping[str, Mapping[str, object]], donors: Sequence[str]
    ) -> Mapping[str, object]:
        fields = (
            "dot_per_element",
            "cosine",
            "correlation",
            "optimal_correction_scale",
            "correction_to_residual_norm",
            "mse_change_after_correction",
        )
        return {
            field: _mean([float(source[donor]["section_balanced"][field]) for donor in donors])
            for field in fields
        } | {
            "positive_inner_product_donor_fraction": float(
                np.mean(
                    [
                        float(source[donor]["section_balanced"]["dot_per_element"]) > 0
                        for donor in donors
                    ]
                )
            ),
            "donors": list(donors),
        }

    program_names = sorted({name for values in program_by_donor.values() for name in values})
    correction_by_program = {
        name: {
            **aggregate_log_alignment(
                {
                    donor: program_by_donor[donor][name]
                    for donor in all_donors
                    if name in program_by_donor[donor]
                },
                [donor for donor in all_donors if name in program_by_donor[donor]],
            ),
            "donor_test_on_dot_per_element": _comparison_summary(
                runner,
                {
                    donor: float(
                        program_by_donor[donor][name]["section_balanced"]["dot_per_element"]
                    )
                    for donor in all_donors
                    if name in program_by_donor[donor]
                },
                seed=1729 + 600 + index,
            ),
        }
        for index, name in enumerate(program_names)
    }
    program_pvalues, program_pvalue_names = [], []
    for name, summary in correction_by_program.items():
        test_summary = summary["donor_test_on_dot_per_element"]
        if test_summary.get("evaluable") is True:
            program_pvalue_names.append(name)
            program_pvalues.append(float(test_summary["exact_one_sided_sign_flip"]["p_value"]))
    if program_pvalues:
        for name, value in zip(program_pvalue_names, core.holm_adjust(program_pvalues)):
            correction_by_program[name]["holm_adjusted_p_value"] = float(value)

    distribution_aggregate = {}
    for arm in ARMS:
        metric_names = (
            "latent_MSE_20D",
            "gene_correlation",
            "program_correlation",
            "reliability_adjusted_variance_median",
            "program_covariance_relative_error",
            "rare_program_state_recall",
            "dynamic_range_ratio_median_gene",
            "dynamic_range_ratio_q10_gene",
            "dynamic_range_ratio_q90_gene",
            "genes_with_at_least_0.8_dynamic_range_fraction",
        )
        distribution_aggregate[arm] = {
            metric: _mean(
                [
                    float(distribution_by_donor[donor][arm][metric])
                    for donor in all_donors
                    if distribution_by_donor[donor][arm].get(metric) is not None
                ]
            )
            for metric in metric_names
        }
        distribution_aggregate[arm]["predictive_interval_coverage"] = {
            level: _mean(
                [
                    float(distribution_by_donor[donor][arm]["predictive_interval_coverage"][level])
                    for donor in all_donors
                ]
            )
            for level in ("50", "80", "95")
        }
        distribution_aggregate[arm]["gene_covariance"] = {
            field: _mean(
                [
                    float(distribution_by_donor[donor][arm]["gene_covariance"][field])
                    for donor in all_donors
                    if distribution_by_donor[donor][arm]["gene_covariance"].get(field) is not None
                ]
            )
            for field in (
                "full_target_relative_Frobenius_error",
                "full_target_off_diagonal_correlation",
                "split_cross_covariance_relative_Frobenius_error",
            )
        }
        calibration_donors = [
            donor
            for donor in all_donors
            if bool(distribution_by_donor[donor][arm]["calibration_interpretable"])
            and distribution_by_donor[donor][arm].get("calibration_slope") is not None
            and np.isfinite(distribution_by_donor[donor][arm]["calibration_slope"])
        ]
        distribution_aggregate[arm]["calibration_slope"] = _mean(
            [
                float(distribution_by_donor[donor][arm]["calibration_slope"])
                for donor in calibration_donors
            ]
        )
        distribution_aggregate[arm]["calibration_interpretable_donor_count"] = len(
            calibration_donors
        )
        distribution_aggregate[arm]["calibration_interpretable_donor_fraction"] = float(
            len(calibration_donors) / len(all_donors)
        )
        distribution_aggregate[arm]["calibration_interpretable_in_all_donors"] = (
            len(calibration_donors) == len(all_donors)
        )

    per_gene_dynamic_range = {
        gene: {arm: _mean(dynamic_by_arm_gene[arm].get(gene, [])) for arm in ARMS}
        for gene in sorted(dynamic_by_arm_gene["M0"])
    }
    reliable_ratio = float(
        distribution_aggregate["M3"]["reliability_adjusted_variance_median"]
        / distribution_aggregate["M0"]["reliability_adjusted_variance_median"]
    )
    supported_variance_aggregate = {
        arm: _mean(
            [
                float(fold_payload[donor]["supported_reliability_adjusted_variance"][arm]["median"])
                for donor in all_donors
                if fold_payload[donor]["supported_reliability_adjusted_variance"][arm]["median"]
                is not None
            ]
        )
        for arm in ("M0_same_rows", "M2_supported", "M3_supported")
    }
    conditional_reliable_ratio = float(
        supported_variance_aggregate["M3_supported"] / supported_variance_aggregate["M0_same_rows"]
    )
    indication_reversal = any(
        summary.get("evaluable") is True and float(summary["mean_effect"]) <= 0
        for summary in indication_summary.values()
    )
    adequate_support_coverage = all(
        fold_payload[donor]["state_effect"].get("evaluable") is True
        and {
            str(record["section"])
            for record in fold_payload[donor]["state_effect"].get("sections", ())
        }
        == set(fold_payload[donor]["scored_sections"])
        for donor in all_donors
    )
    state_test = state_overall.get("exact_one_sided_sign_flip", {})
    state_interval = state_overall.get("paired_bootstrap_interval", (None, None))
    state_inferential_support = bool(
        isinstance(state_test, Mapping)
        and state_test.get("p_value") is not None
        and float(state_test["p_value"]) <= 0.05
        and len(state_interval) == 2
        and state_interval[0] is not None
        and float(state_interval[0]) > 0
    )
    state_supported = bool(
        state_overall.get("evaluable") is True
        and float(state_overall["mean_effect"]) > 0
        and float(state_overall["positive_donor_fraction"]) >= 0.70
        and not indication_reversal
        and adequate_support_coverage
        and state_inferential_support
        and conditional_reliable_ratio >= 0.80
    )
    correction_dot = float(correction_aggregate["predictive_variance_weighted_inner_product"])
    correction_scale = float(correction_aggregate["optimal_correction_scale"])
    if correction_dot <= 0:
        correction_interpretation = (
            "nonpositive_alignment_H_and_E_correction_does_not_target_reference_residual"
        )
    elif 0 < correction_scale < 1:
        correction_interpretation = "positive_alignment_but_correction_is_over_applied_on_average"
    else:
        correction_interpretation = (
            "positive_alignment_without_evidence_of_average_over_application"
        )

    report = {
        "schema": "heir.natcommun_frozen_prediction_validation.v1",
        "analysis_scope": "outcome_exposed_validation_diagnostics_only_non_confirmatory",
        "frozen_model_changed": False,
        "fit_or_parameter_update_performed": False,
        "iterative_refinement_run": False,
        "image_encoder": baseline_report["image_encoder"],
        "uni2_h_run": False,
        "reference_assay": baseline_report["reference_assay"],
        "artifact_identities": {
            **identities,
            "protocol_sha256": EXPECTED["protocol_sha256"],
            "prepared_manifest_sha256": _sha256(args.baseline / "prepared_manifest.json"),
        },
        "target_boundary": {
            "global_prediction_preflight_before_any_target": True,
            "all_subgroup_masks_hashed_before_any_target": True,
            "subgroup_mask_receipts": subgroup_mask_receipts,
            "immediate_prediction_revalidation_before_each_target": True,
            "immediate_target_identity_revalidation_after_open": True,
            "target_gene_axis_bound_by_public_fold_and_prediction_gene_identity": True,
            "prediction_rows_or_values_changed_after_target_open": False,
        },
        "composition_state_validation": {
            "comparison": "M3_supported_vs_M2_supported_identical_supported_composition",
            "overall": state_overall,
            "by_indication": indication_summary,
            "by_H_and_E_composition_proxy_type": type_summary,
            "type_minimum_composition_mass_per_section": 3.0,
            "type_scope": (
                "soft_H_and_E_predicted_composition_weights_not_observed_spot_type_truth_"
                "not_cell_level_evidence"
            ),
            "by_reference_coverage": coverage_summary,
            "coverage_split": "frozen_query_reference_coverage_mass_at_0.90",
            "global_M3_to_M0_reliability_adjusted_variance_ratio": reliable_ratio,
            "same_supported_rows_reliability_adjusted_variance": supported_variance_aggregate,
            "same_supported_rows_M3_to_M0_reliability_ratio": conditional_reliable_ratio,
            "minimum_required_reliability_ratio": 0.80,
            "adequate_support_coverage": adequate_support_coverage,
            "adequate_support_definition": (
                "every_scored_donor_section_has_at_least_three_identically_supported_spots"
            ),
            "inferential_support": state_inferential_support,
            "inferential_support_definition": (
                "one_sided_exact_sign_flip_p_at_most_0.05_and_paired_bootstrap_lower_bound_"
                "above_zero"
            ),
            "general_continuous_state_claim_supported": state_supported,
            "decision_rule": (
                "positive_mean_and_at_least_70_percent_positive_donors_and_no_indication_"
                "reversal_and_every_section_meets_frozen_support_and_inferential_support_and_"
                "same_supported_rows_M3_to_M0_reliability_ratio_at_least_0.80"
            ),
        },
        "image_correction_alignment": {
            "primary_space": "raw_counts_weighted_by_frozen_M1_predictive_variance",
            "program_space": "log1p_10000_normalized_expression",
            "correction": "M3_minus_M1",
            "reference_residual": "heldout_ST_minus_M1",
            "matched_H_and_E": correction_aggregate,
            "shuffled_H_and_E": shuffled_aggregate,
            "correct_minus_shuffled_weighted_alignment": pairing_comparison,
            "by_indication": correction_by_indication,
            "by_program": correction_by_program,
            "interpretation": correction_interpretation,
            "correction_magnitude_vs_reference_uncertainty_by_donor": {
                donor: {
                    "predictive_SD": correction_by_donor[donor]["section_balanced"][
                        "correction_RMS_to_predictive_SD"
                    ],
                    "latent_SD": correction_by_donor[donor]["section_balanced"][
                        "correction_RMS_to_latent_SD"
                    ],
                }
                for donor in all_donors
            },
        },
        "molecular_distribution_audit": {
            "arms": distribution_aggregate,
            "per_gene_P95_minus_P05_dynamic_range_ratio": per_gene_dynamic_range,
            "dynamic_range_denominator": "heldout_ST_log1p_10000_P95_minus_P05_within_section",
            "variance_pattern": (
                "M1_less_than_M2_less_than_M3_less_than_M0"
                if all(
                    float(distribution_aggregate[left]["reliability_adjusted_variance_median"])
                    < float(distribution_aggregate[right]["reliability_adjusted_variance_median"])
                    for left, right in (("M1", "M2"), ("M2", "M3"), ("M3", "M0"))
                )
                else "ordering_differs_from_M1_less_than_M2_less_than_M3_less_than_M0"
            ),
            "state_variance_preserved": reliable_ratio >= 0.80,
        },
        "folds": fold_payload,
        "scientific_decision": {
            "central_gate1_result_unchanged": True,
            "H_and_E_reference_synergy_confirmed": False,
            "continuous_state_recovery_supported": state_supported,
            "cell_level_claim_authorized": False,
            "independent_regional_confirmation_obtained": False,
            "measurement_floor_validated": False,
            "iterative_refinement_authorized": False,
        },
        "limitations": [
            "NatCommun outcomes influenced the frozen panel and architecture",
            "per-type strata use H&E-predicted composition rather than observed spot type truth",
            "regional Visium cannot validate individual-cell expression or state",
            "the registered suspension reference is not independently verified as snRNA",
            (
                "the diagnostics do not replace the unrun fold-local panel or fixed-support "
                "bank refits"
            ),
            "no independent measurement replicate is available for an ST floor",
        ],
    }
    _atomic_json(args.output, _safe(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "continuous_state_supported": report["composition_state_validation"][
                    "general_continuous_state_claim_supported"
                ],
                "correction_interpretation": report["image_correction_alignment"]["interpretation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
